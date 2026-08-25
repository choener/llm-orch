# CPU-only models

Original question: How to define models that run on the CPU only? What about
llama-cpp built for Vulkan? How to setup?

## Answer: yes, one Vulkan build handles both

Vulkan is an *additive* compile-time backend — the CPU path (SIMD, optional
BLAS) is always present in the same binary. A single `llama-server` built
with Vulkan serves GPU models *and* CPU-only models; the difference is purely
launch-time:

- GPU model:  `vulkan_devices: [...]` → llm-orch pins `GGML_VK_VISIBLE_DEVICES`,
  cmd offloads layers (`--n-gpu-layers N`).
- CPU-only:   omit both `vulkan_devices` and `cuda_devices` → no GPU pinning,
  cmd must pass `--n-gpu-layers 0`.

Caveats:

- llama.cpp's default is `--n-gpu-layers 99` ("offload as much as fits"), so a
  CPU-only model that does **not** explicitly pass `--n-gpu-layers 0` will
  silently load onto Vulkan device 0. This is the trap in the current example
  config (see TODO 2).
- The binary still links the Vulkan loader, so libvulkan must be present at
  runtime — harmless on a GPU host (our case); a driver-less box may log a
  warning but works with `-ngl 0`.
- CPU perf is determined by the build flags (AVX2/AVX-512, OpenBLAS),
  independent of the GPU backend. No separate CPU build needed.

## Current state (already works, no code needed)

- Config: omitting both device lists = CPU-only; validation enforces
  `gpus: 1` (src/config.rs, "CPU-only models must use gpus: 1").
- Spawn: empty device list → no `GGML_VK_VISIBLE_DEVICES` /
  `CUDA_VISIBLE_DEVICES` (`LlamaCppBackend::gpu_env(&[])` returns nothing).
- VRAM/keep-alive: CPU models reserve no VRAM (`vram: 0`) and acquire no
  keep-alive refs — already correct.

## Non-llama.cpp backends on CPU (vision models, audio.cpp)

- **Vision models (`--mmproj`)** are plain llama.cpp — same binary, same
  flags. The `-ngl 0` injection covers them; the vision projector just runs
  on CPU. No manual ngl needed.
- **audio.cpp** has a different interface: no `-ngl` at all — device
  selection is an all-or-nothing `"backend"` field in `server.json`
  (`"cuda" | "cpu" | "vulkan" | "metal"`, overridable via `--backend`).
  CPU is always available (no build flag). So a CPU audio model is
  `"backend": "cpu"` (or `--backend cpu` in the alias) — manually, yes, but
  via that field, not ngl.
- **Consequence for TODO 1:** the spawn path hardcodes `LlamaCppBackend` for
  *every* model (src/scheduler.rs:1261), so a blanket `--n-gpu-layers 0`
  injection would append a flag `audiocpp_server` does not understand and
  break its spawn. The injection must be gated to llama.cpp programs.

## Gaps

1. **llm-orch never forces `-ngl 0`.** `LlamaCppBackend::gpu_args(&[])`
   returns `Vec::new()`; CPU-ness depends entirely on the user's cmd.
2. **`config.example.yaml` is wrong for its own CPU example.** "Example 3"
   (deepseek-coder-7b, "runs on CPU") uses the `{llama}` alias, which bakes in
   `--n-gpu-layers 99`, and declares `vram: 8000` — a Vulkan build will
   offload to GPU 0 while llm-orch thinks it's CPU-only.
3. The `ram` field is declared but unused (reserved for the deferred eviction
   policy) — yet CPU-only models are exactly the ones that pressure system RAM.
4. No docs/README section on CPU-only models.

## TODO

- [x] **1. Force CPU offload — two stages (core fix).** [done]
      - **Validation** (src/config.rs, at load + hot-reload): reject a
        CPU-only model (both device pools empty) whose *resolved* cmd (alias
        expansion + shlex split, same substitution as `Scheduler::resolve_cmd`
        — factor the resolution out so validation can reuse it) runs a
        `llama-*` program with an offload flag of any value ≠ 0
        (`--n-gpu-layers` / `-ngl` / `--n-offload`, both `--flag value` and
        `--flag=value` forms; `-1`/"auto" counts as a conflict).  Error:
        "model 'X' is CPU-only (no device pool) but its command requests GPU
        offload (--n-gpu-layers 99) — add vulkan_devices/cuda_devices or drop
        the flag (use a CPU alias)".  Fits the existing atomic-reload style:
        a rejected config leaves the old one running.
      - **Spawn** (src/backend.rs): append `--n-gpu-layers 0` — unconditional
        for (CPU-only ∧ `llama-*` program basename), because validation has
        already ruled out a conflicting flag; *skip* the append when the cmd
        already sets the flag explicitly (validation then guarantees the
        value is 0, so no duplicate is needed and we never bet on
        last-flag-wins for duplicate llama.cpp args).
      - Gates, restated: CPU-only is the primary gate — GPU models never
        reach `gpu_args(&[])` (unplaceable ones are refused earlier in
        `spawn_instance`).  The `llama-*` basename gate excludes audio.cpp,
        which selects its device via `server.json` `backend` and must never
        receive llama.cpp flags (the spawn path hardcodes `LlamaCppBackend`
        for all models, src/scheduler.rs:1261).
      - Tests: config-validation unit tests (ngl>0 on CPU-only rejected;
        explicit `-ngl 0` accepted; GPU model with ngl untouched; audiocpp
        cmd with device pool untouched) + spawn-injection test in backend.rs.
- [x] **2. Fix `config.example.yaml`.** [done alongside item 1] Give the CPU example a dedicated
      `llama-cpu` alias (same as `llama` but `--n-gpu-layers 0`), point
      deepseek-coder-7b at it, and correct its hints to `vram: 0` /
      `ram: <model size in MB>`. Keep the `# vulkan_devices: [0]  # uncomment
      for GPU offload` comment pattern.
- [ ] **3. Integration test (tests/integration.rs) + stub support.**
      - Stub backend (src/bin/llm-orch-stub-backend.rs) prerequisites —
        it is a `clap::Parser`, so an injected `--n-gpu-layers 0` currently
        makes it *exit immediately* (unknown flag), and it exposes no way
        to observe the child's argv/env:
        - Add `--n-gpu-layers` (and `--n-offload`) to the accepted-and-ignored
          compatibility flags.
        - Add a test-support endpoint (per AGENTS.md convention), e.g.
          `GET /__test/launch` returning `{ argv: std::env::args(),
          env: { GGML_VK_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES } }`.
      - Test setup: launch the stub through a `llama-server` symlink in a
        temp dir so the program-name gate fires (the stub's own binary name
        does not).
      - Assert: a CPU-only model (no device lists) spawns with **no**
        `GGML_VK_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` and with
        `--n-gpu-layers 0` in the child argv; a GPU model is unaffected.
      - Regression unit test (scheduler.rs): make-room for a CPU-only target
        at its instance cap never evicts *other* models' idle instances
        (plan returns None) — `plan_make_room` only commits plans that make
        the spawn feasible, so this is safe today but subtle.
- [ ] **4. Docs.** Add a "CPU-only models" section (README or
      docs/002-gpu-selection.md): definition (empty device lists, `gpus: 1`),
      the `-ngl 0` requirement (and that llm-orch now injects it), CPU tuning
      guidance (`--threads`/`--threads-batch` in `cmd` to avoid core
      oversubscription when `max_instances > 1` or several CPU models run
      concurrently), and how non-llama backends do it (vision: same as
      llama.cpp; audio.cpp: `"backend": "cpu"` in server.json).  Note that
      the `ram` estimate must cover weights + KV cache (big `--ctx-size` on
      CPU is where RAM goes).
- [ ] **5. RAM gating (agreed).** The `ram` hint is currently
      dead. Consider gating CPU-only spawns on free system RAM
      (`/proc/meminfo` MemAvailable vs. sum of declared `ram` for
      loading/loaded CPU instances) before the deferred eviction policy —
      otherwise two big CPU models can OOM the box. If deferred, say so in
      the docs from item 4.
- [ ] **6. Make spawn timeout configurable (agreed).** It's hardcoded to
      120 s (`src/scheduler.rs`); large CPU-only loads (no offload, big KV
      cache in RAM) are the slowest spawns and may want more. New config key
      (e.g. `server.spawn_timeout_secs`) with the current 120 s as default.

## Decisions

- **Inject, don't document.** Making the "CPU-only" declaration
  self-enforcing beats a cmd convention — it means llm-orch rewrites
  llama.cpp args, which the project had otherwise avoided, but the
  contradiction (CPU-only declaration + offload flag) becomes impossible to
  ship, and it's caught at config-validation time, not mid-spawn.
- **Reject at validation, don't override at runtime.** A CPU-only model
  whose cmd asks for offload is a config error; failing at load with a
  precise message (atomic reload preserved) is more honest than silently
  rewriting the user's flag value at spawn.
- **`llama-*` program-name gate** excludes audio.cpp (its own `backend`
  field), and vision models (`--mmproj` on llama.cpp) benefit from the
  injection for free.
