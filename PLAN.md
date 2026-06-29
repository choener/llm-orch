# LLM Orchestration Layer — Plan

## Goal

A Rust systemd service (`llm-orch`) that manages `llama-server` instances across
multiple GPUs, providing an OpenAI-compatible HTTP API with auto-scaling, session
affinity, transparent instance migration, and API key enforcement.

## Architecture

```
Client (pi, curl, etc.)
    │
    ▼
  nginx (TLS, reverse proxy)
    │
    ▼
llm-orch (Rust, axum/tokio)
    │  ├── Config watcher (TOML, hot-reload)
    │  ├── API key watcher (hot-reload, immediate sever)
    │  ├── GPU monitor (user script, extensible)
    │  ├── Pressure engine (scale up/down decisions)
    │  ├── Session tracker (affinity + migration)
    │  └── Instance manager (spawn/keep-alive/kill)
    │
    ├── llama-server :8081 --gpu 0 (qwen3-6b.gguf)
    └── llama-server :8082 --gpu 1 (qwen3-6b.gguf)
```

### Config (TOML, hot-reload)

```toml
[server]
listen = "127.0.0.1:8080"

[scale-down]
timeout = "5m"       # global default idle timeout

[scripts]
timeout = "10s"      # global timeout for all external commands (gpu query, keep-alive)

[pressure]
vram_usage_weight = 1.0
overcommit_penalty = 2.0
instance_count_weight = 0.1
active_requests_weight = 0.05
stickiness_grace_period = 120  # seconds

[gpus.0]
name = "AMD Radeon 1"
query-command = "bash /etc/llm-orch/gpu0-info.sh"
query-interval = "5s"

[gpus.1]
name = "AMD Radeon 2"
query-command = "bash /etc/llm-orch/gpu1-info.sh"
query-interval = "5s"

[models.qwen3-6b]
path = "/path/to/qwen3-6b.gguf"
allowed-gpus = [0, 1]   # references GPU IDs defined in [gpus]
scale-down-timeout = "10m"   # per-model override
keep-alive-command = "bash /etc/llm-orch/keepalive.sh"
keep-alive-interval = "30s"
pre-load-command = "bash /etc/llm-orch/pre-load.sh"   # run before model loading starts
post-load-command = "bash /etc/llm-orch/post-load.sh" # run after model is ready
log-traffic = true
log-traffic-path = "/var/log/llm-orch/qwen3-6b/traffic.jsonl"
log-traffic-max-size = "1GB"
log-traffic-max-age = "7d"
llama-server-bin = "llama-server"
llama-server-args = ["--ctx-size", "8192", "--threads", "8"]
```

### API Keys (separate TOML, hot-reload)

```toml
# One key per line, or array
keys = ["sk-abc123", "sk-def456"]
```

### Pressure Model

Decisions about which model instances to spin up or down are driven by a scoring
system rather than simple thresholds:

**Scale-up triggers** (per model):
- Any running instance has `active_requests > concurrency_threshold` (default: 2)
- Any running instance has `avg_response_time > latency_threshold` (configurable)
- New request arrives and no instance is running → immediate scale-up

**GPU selection** (when spinning up):
- Score each allowed GPU:
  ```
  score = (vram_used / vram_total) * 1.0
        + overcommit_penalty      # if vram_used > vram_total: +2.0
        + instance_count * 0.1    # prefer GPUs with fewer instances
        + active_requests_on_gpu * 0.05
  ```
- Pick GPU with lowest score

**Scale-down triggers** (per instance):
- Idle for `scale-down-timeout` (per-model, or global default)
- No in-flight requests
- Stickiness grace period elapsed (configurable, default: 2 minutes after last scale-up)

**Context-aware migration cost** (for routing decisions):
- When an instance is being drained, active sessions with large context windows
  are prioritized to stay on the current instance longer
- Sessions are migrated only when the instance is being shut down; the next
  request naturally replays the full history to the new instance

## Tasks

### 1. Project Skeleton
- [ ] Initialize Cargo project (`llm-orch`, binary crate)
- [ ] Add dependencies: `tokio`, `axum`, `serde`/`toml`, `reqwest`, `notify`, `tracing`, `uuid`, `hyper-util`
- [ ] Define TOML config schema (`ServerConfig`, `ModelConfig`, `ScaleDownConfig`)
- [ ] Define API keys config schema
- [ ] CLI argument parsing (config path, keys path, log level)
- [ ] Basic `main()` that loads config and exits

### 2. GPU Discovery & Monitoring
- [ ] GPU definitions parsed from `[gpus]` section of TOML config
  - Each GPU: `id`, `name`, `query-command`, `query-interval`
- [ ] GPU query module: executes per-GPU script, parses JSON output:
  ```json
  {"vram_total_mb": 16384, "vram_used_mb": 8192}
  ```
- [ ] Periodic GPU state polling at per-GPU configured interval
- [ ] All external commands (GPU query, keep-alive) share a global timeout (`[scripts].timeout`, default 10s)
- [ ] GPU state stored in shared state (Arc<RwLock> or similar)
- [ ] Handle script failures gracefully (log warning, retain last known state, don't crash)

### 3. llama-server Instance Lifecycle
- [ ] Instance struct: pid, local port, model name, GPU, state (starting/loading/running/stopping/stopped)
- [ ] Pre-load action: execute `pre-load-command` (per-model, optional) before spawning `llama-server`
  - Subject to global `[scripts].timeout`; on failure, block instance from starting and log error
- [ ] Spawn `llama-server` process with configured args + `--host` + `--port` + GPU placement
- [ ] Wait for instance to be ready (poll `/v1/models` until 200, with timeout)
- [ ] Post-load action: execute `post-load-command` (per-model, optional) after instance is ready
  - Subject to global `[scripts].timeout`; on failure, log warning but mark instance as ready (GPU tuning is best-effort)
- [ ] Kill instance gracefully (SIGTERM → wait → SIGKILL after timeout)
- [ ] Keep-alive/heartbeat: execute user-defined shell command at configurable interval
  - Command is per-model in config, optional (no command = no keep-alive)
  - If command exits non-zero, log warning but don't auto-kill (user-defined semantics)
- [ ] Detect crashed instances (process exit monitoring via `tokio::process::Child`)
- [ ] Instance registry: track all running instances per model

### 4. OpenAI-Compatible HTTP Server
- [ ] `POST /v1/chat/completions` — streaming (SSE) and non-streaming
- [ ] `GET /v1/models` — list available models
- [ ] `GET /info` — orchestrator info endpoint (metadata only, behind API key auth)
  - Per model: running instances (GPU, port, uptime), active sessions, total requests, avg response time, cached vs new tokens
  - Per GPU: VRAM usage, instances running
  - Recent requests: timestamp, model, session_id, input/output tokens, cached tokens, duration, serving GPU
- [ ] Request validation: model name must exist in config
- [ ] API key extraction from `Authorization: Bearer <key>` header
- [ ] API key validation against loaded key set
- [ ] OpenAI-compatible request/response types (`ChatCompletionRequest`, `ChatCompletionChunk`, etc.)
- [ ] Error responses in OpenAI format (`{ "error": { "message": ..., "type": ... } }`)

### 5. Session Affinity & Routing
- [ ] Session ID extraction: from `X-Session-Id` header, or generate/assign one
- [ ] Session registry: maps session_id → instance_id (with last-accessed timestamp)
- [ ] Routing logic:
  - If session has an affinity and that instance is healthy → route there
  - If affinity instance is gone/unhealthy → pick new instance, update affinity
  - If no affinity → pick instance via pressure model
- [ ] Forward request to chosen `llama-server` instance via `reqwest`
- [ ] Pipe SSE streaming response from instance back to client (no buffering)
- [ ] Handle streaming errors: instance dies mid-stream → close stream with error (no retry)

### 6. API Key Enforcement & Hot-Reload
- [ ] Load API keys from separate config file at startup
- [ ] Watch API keys file for changes (`notify` crate)
- [ ] On reload: compute diff (added/removed keys)
- [ ] Track active connections by API key (insert into shared set on auth success)
- [ ] On key removal: signal all active streaming connections using that key to abort
- [ ] Reject new requests with removed keys immediately
- [ ] New keys take effect immediately without restart

### 7. Config Hot-Reload
- [ ] Watch main TOML config file for changes
- [ ] On reload: parse new config, validate
- [ ] Apply changes:
  - New models → available for future requests (no auto-spin)
  - Removed models → stop accepting new requests, drain existing
  - Changed `allowed-gpus` → affects next scale-up decision
  - Changed timeouts → applied immediately
- [ ] Active LLM connections stay alive during reload
- [ ] Failed reload → log error, keep old config

### 8. Pressure Engine & Auto-Scaling
- [ ] Pressure scorer module with configurable weights (TOML):
  - `vram_usage_weight` — VRAM used / total ratio (default: 1.0)
  - `overcommit_penalty` — additive penalty when VRAM used > total (default: 2.0)
  - `instance_count_weight` — per-instance additive cost (default: 0.1)
  - `active_requests_weight` — per-active-request additive cost (default: 0.05)
  - `stickiness_grace_period` — seconds after scale-up before scale-down allowed (default: 120)
- [ ] Compute GPU scores from VRAM usage, instance count, active requests using weights
- [ ] Compute instance load (active requests, avg response time)
- [ ] Scale-up logic:
  - Triggered by new request (no instance running) or load threshold exceeded
  - Select GPU via pressure scorer
  - Spawn instance, stream "loading ..." progress to client while waiting
  - Add to registry when ready
- [ ] Scale-down logic:
  - Periodic check (e.g., every 30s) for idle instances
  - Check idle timeout (per-model or global)
  - Check stickiness grace period
  - Drain: stop accepting new sessions, wait for in-flight requests to finish
  - Kill instance after drain
- [ ] Concurrency limits: max instances per model, max instances per GPU

### 9. Streaming Implementation
- [ ] Async SSE streaming from `llama-server` → `axum` `Stream` response
- [ ] Proper `Content-Type: text/event-stream` headers
- [ ] Pipe chunks through without buffering (use `tokio_stream` or similar)
- [ ] Handle partial reads, connection resets
- [ ] Include `session_id` in response headers (for clients that need it)
- [ ] Support `stream_options: { include_usage: true }` for token usage in final chunk
- [ ] Loading progress stream: when scaling up from zero, emit SSE events:
  - Initial: `loading <model-name> on GPU <N> ...
` (append `.` each second)
  - Final: `loading <model-name> on GPU <N> ..... <X>s done
`
  - Only emitted before the first chunk of the actual response

### 10. Error Handling & Resilience
- [ ] Timeout handling: per-request timeout, per-instance startup timeout
- [ ] Graceful shutdown: on SIGTERM, stop accepting requests, drain all instances
- [ ] Instance crash recovery: auto-respawn if configured
- [ ] Comprehensive error types with proper HTTP status codes

### 11. Observability & Operations
- [ ] Structured logging (`tracing` → `tracing-subscriber` with JSON or pretty)
- [ ] Metrics: requests per model, response times, VRAM usage, scale events
- [ ] Per-model traffic logging (`log-traffic = true`):
  - JSONL format, one entry per completed request
  - Fields: timestamp, model, session_id, request (messages), response (full text), input/output tokens, cached tokens, duration, serving GPU/instance
  - Async write via channel/buffer (must not block request path)
  - Log rotation: `log-traffic-max-size` (default 1GB), `log-traffic-max-age` (default 7d)
  - Path: `log-traffic-path` (required when `log-traffic = true`)
- [ ] Health endpoint: `GET /health` for systemd `ExecStartPost` or `Type=notify`
- [ ] Systemd socket activation support (optional, nice-to-have)
- [ ] Log rotation configuration guidance

### 12. Testing
- [ ] Unit tests: config parsing, pressure scoring, session routing logic
- [ ] Integration tests: mock `llama-server` (or use a tiny model)
- [ ] Test scenarios:
  - Scale up from 0 to 1 instance
  - Scale up from 1 to 2 instances under load
  - Scale down after idle timeout
  - Session affinity maintained across requests
  - Transparent migration when instance dies
  - API key revocation severs active connections
  - Config hot-reload adds/removes models
  - Streaming response piped correctly

## Known Risks & Open Questions

1. **GPU monitoring via per-GPU scripts.** Each GPU has its own query script
   returning JSON. The pressure model only works as well as these scripts.
   The global script timeout (default 10s) prevents hung scripts from blocking.
   Future: native Vulkan or ROCm queries to replace scripts entirely.
   Open question: what VRAM metrics are reliably available on AMD + Vulkan?

2. **Model file locking.** Multiple `llama-server` instances can read the same
   `.gguf` file simultaneously (it's read-only). However, if the model file
   changes on disk, running instances won't see it. Document this limitation.

3. **Context window migration.** When a session is migrated to a new instance,
   the full context is reprocessed. For very long conversations (128k+ tokens),
   this adds significant latency. Consider: context summarization, or allowing
   the client to truncate history.

4. **Race conditions on scale events.** A scale-up and scale-down for the same
   model could race. Mitigation: use a mutex or atomic state machine per model.

5. **Keep-alive command is user-defined.** The orchestrator runs it at the
   configured interval but has no way to validate its output or semantics.
   If the command hangs, it blocks the keep-alive task. Need a timeout on
   command execution.

6. **API key storage security.** The keys file should have restricted permissions
   (0600). The orchestrator should warn if permissions are too open.

7. **Multiple clients, same session.** If two clients use the same session ID,
   they share affinity. This is intentional (allows load balancer behind
   llm-orch to route by session). Document this behavior.

8. **GPU process isolation.** `llama-server` uses `--main-gpu` and Vulkan device
   selection for GPU placement. Ensure our spawned processes pin correctly to
   avoid cross-GPU memory transfers. Exact flags depend on llama.cpp version.

9. **Pre/post-load command semantics.** Pre-load failure blocks the instance
   (hard gate). Post-load failure is soft (warn, continue). This asymmetry is
   intentional: pre-load typically prepares the GPU (must succeed), post-load
   typically tunes it (best-effort).

10. **Traffic logging disk impact.** With `log-traffic = true` and high throughput,
    JSONL files can grow fast. Rotation defaults (1GB, 7d) should be sufficient
    for most workloads but worth monitoring. Async writes mean logs may be lost
    on crash (acceptable tradeoff).

11. **KV-cache migration complexity.** Deferred to last because it's the most
    intricate feature. Depends on llama.cpp slot API stability, slot ID tracking,
    and binary format compatibility. The system works fine without it (slower
    post-migration PP).

### 13. KV-Cache Migration (Deferred — Lowest Priority)
- [ ] When scaling down an instance with active sessions:
  - Pause: put incoming requests for affected sessions on hold
  - Save: call `/slots/<id>/save` on source instance for each active session (KV cache → SSD)
  - Transfer: make saved KV cache available to target instance (HTTP download or shared path)
  - Restore: call `/slots/<id>/restore` on target instance
  - Resume: update session affinity, release held requests
- [ ] Fallback: if save/restore fails at any point, release sessions to target without cache
  (full context replay, just slower)
- [ ] Skip KV transfer for sessions below a context-size threshold (configurable, default: 4096 tokens)
  — the overhead of save/restore exceeds the PP savings for short contexts
- [ ] Version compatibility check: refuse KV transfer between instances running different llama.cpp versions

## Non-Goals (For Now)

- Model quantization or conversion
- Multi-node orchestration (single server only)
- gRPC or non-HTTP protocols
- Model fine-tuning or LoRA support
- Web UI / dashboard (metrics are for logging/scraping)
