# AGENTS.md — llm-orch

Guidance for coding agents working in this repository.

## What this project is

`llm-orch` is a single-host LLM orchestrator written in Rust. It acts as a smart
OpenAI-compatible proxy in front of multiple `llama.cpp` (`llama-server`)
instances:

`Client → llm-orch (Auth → Scheduler → Load Balancer) → llama-server instances`

Key behaviors:

- Spawns `llama-server` backends on demand; unloads them after `idle_ttl`.
- GPU-aware scheduling: picks GPUs by free VRAM from configured Vulkan/CUDA
  device pools; supports multi-GPU tensor splits.
- Load-based autoscaling of parallel instances per model (EMA metrics).
- Hot-reload of `config.yaml` and the apikeys file at runtime (validated
  atomically; a rejected config leaves the old one running).
- File-based API key auth, FIFO request queueing, model aliases with system
  prompt injection.

Design invariants (from `docs/001-plan.md` — do not break these):

- **Default-deny auth**: no configured apikeys ⇒ deny all access.
- **No model preload**: GPUs stay idle until the first request; not configurable.
- **Atomic config reload**: never partially apply an invalid config.
- **Graceful shutdown**: drain in-flight requests and streams before killing backends.
- **TLS is out of scope**: a reverse proxy (nginx) terminates TLS.

## Build, test, lint

```bash
cargo build                 # debug build
cargo build --release       # release binary at target/release/llm-orch
cargo test                  # integration tests (use the stub backend; no GPU needed)
cargo clippy                # lint — keep it warning-free
cargo fmt                   # formatting
./target/release/llm-orch --check-config config.yaml   # validate a config, exit non-zero on error
```

Logging when running locally: `RUST_LOG=info` for verbosity,
`LLM_ORCH_LOG_JSON=1` for structured JSON logs.

Nix (optional): `nix develop` enters a devShell with cargo/clippy/rustfmt/
rust-analyzer/cargo-expand; `nix build` builds the package; the flake also
exports `nixosModules.llm-orch` (module source in `nix/nixos.nix`).

## Codebase layout

- `src/lib.rs` — all orchestrator logic lives in the library; the binary is thin.
- `src/main.rs` — CLI entry point (clap), startup wiring.
- `src/config.rs` — YAML config model + validation (models, devices, aliases).
- `src/scheduler.rs` — instance manager: spawn/evict/route, VRAM-aware GPU
  selection, autoscaling. Largest file; core of the project.
- `src/handlers.rs` — HTTP handlers for the OpenAI-compatible endpoints.
- `src/server.rs` — axum router/server setup and route registration.
- `src/backend.rs` — backend process abstraction (spawn, readiness, forwarding).
- `src/instance.rs` — per-instance state.
- `src/gpu.rs` — AMD GPU sysfs monitoring; `src/nvidia.rs` — NVIDIA metrics via
  `nvidia-smi` (must be on PATH for live CUDA metrics).
- `src/keepalive.rs` — GPU keep-alive manager (AMD-only; NVIDIA should use
  persistence mode instead).
- `src/apikeys.rs` — API key file parsing/validation (hot-reloadable).
- `src/watcher.rs` + `src/reload.rs` — file watching and atomic hot-reload.
- `src/debug_log.rs` — per-model JSONL debug logging.
- `src/http_client.rs` — shared `reqwest::Client` tuned for backend forwarding.
- `src/port_alloc.rs` — backend port allocation.
- `src/types.rs` — OpenAI-compatible request/response types.
- `src/bin/llm-orch-stub-backend.rs` — stub `llama.cpp` server used by tests
  (mimics the endpoints llm-orch uses; no GPU/model required).
- `tests/integration.rs` — end-to-end tests booting a real orchestrator against
  the stub backend.
- `docs/` — design docs (`001-plan.md` is the authoritative design spec).
- `scripts/` — helper shell scripts (`get-status.sh`, `query-llm.sh`).
- `config.example.yaml`, `apikeys.example.txt` — copy to `config.yaml` /
  `apikeys.txt` to run locally.

## Conventions

- Rust edition 2024; async with tokio; HTTP via axum 0.8 + tower-http.
- Errors: `thiserror`-derived error types; no `unwrap()` in production paths.
- Observability: `tracing` (not `println!`); per-request correlation IDs use
  `uuid` v4.
- YAML: **use `serde_yaml_ng`** — `serde_yaml` is deprecated/unmaintained.
- Forwarding philosophy: passthrough endpoints handle request bodies as raw
  `serde_json::Value` rather than typed structs, so new upstream API fields
  keep working (see the rerank/responses handlers for the pattern).
- Backend-facing config snippets are split shell-style with `shlex`.
- Keep logic in the library (`src/lib.rs` modules) so integration tests and the
  thin binary can compose it; put test-support endpoints in the stub backend,
  not in test-only `cfg` hacks in the library.
- Unix-only process control (`libc`, `PR_SET_PDEATHSIG`) is gated behind
  `cfg(unix)`; the project targets Linux.

## Working with config/apikeys files at runtime

- Both files are hot-reloaded via filesystem watch. When tooling or tests
  rewrite them, **replace atomically** (write temp file, then `mv`) so the
  watcher fires reliably.
- The apikeys file must be readable only by the service user (`0400`/`0600`).
- Relative paths in the config resolve against the working/state directory.
