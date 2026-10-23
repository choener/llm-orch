# 008 — Native radiance engine

[radiance](https://codeberg.org/StillDeadcode/radiance) is a native AMD
inference engine (C++/HIP, Apache-2.0) serving `.rad` model containers.
It speaks the same OpenAI-compatible API as llama-server (plus Responses,
Anthropic Messages, images/audio/video), so everything downstream of
llm-orch's scheduler keeps working unchanged. This document supersedes the
"next step" of `004-native-vllm-radiance.md`: radiance replaces the
vLLM-radiance Python stack with a single self-contained binary — no
container, no Python, ROCm userspace linked in.

## Backend integration

radiance is detected by program basename (`radiance*`, via
`backend::is_radiance_program`) and mapped to `RadianceBackend`
(`src/backend.rs`). The differences that define the backend:

- **Device pinning is env-only.** radiance has *no* device-selector CLI
  flag; it is pinned purely through `HIP_VISIBLE_DEVICES` with the host
  indices of the selected `rocm_devices` (reusing the existing ROCm pool,
  VRAM accounting and sysfs mapping). Injecting llama.cpp's
  `--device ROCmN` would kill it (`E unknown option`) — that is why the
  default llama fallback in `backend_for_prog` could not stay.
- **Tensor parallelism is operator-declared.** radiance tensor-splits
  across *all* visible cards under `--tp N`, which interacts with
  `--tp-wire` / `--p2p`. llm-orch never injects `--tp`; `gpus:` must match
  the declared `--tp` in `cmd` (validation does not cross-check them —
  `--tp` is free-form like every other radiance flag).
- **HIP-only.** A `radiance` model with a `vulkan_devices` or
  `cuda_devices` pool is a config error (the pin would silently do
  nothing). The engine's CPU fallback (AVX host kernels) keeps an empty
  pool valid.
- **Readiness is the default `GET /health`** — radiance keeps `/health`,
  `/v1/health` and `/ping` unauthenticated even when `--api-key` is set.

Everything else is business as usual: passthrough forwarding, internal
load EMAs for autoscaling (no backend metrics scraping), idle eviction
(SIGTERM drain like any process model), `/v1/models` via `upstream_model`
if radiance's served name differs from the llm-orch model name.

## Operational notes

- **Spawn budget.** Loading a `.rad` reads the whole container into
  VRAM/pinned RAM (~29 GiB for Qwen3.8-27B-FP8; fast with a warm page
  cache, minutes on a cold one). Raise `spawn_timeout_secs` (e.g. 600) per
  model; the 120 s default is tight.
- **The engine claims its cards eagerly** (`--gpu-headroom-mib` aside),
  so a loaded radiance instance owns its GPUs — declare `vram:` so the
  scheduler does not try to place a second model there (it is attributed
  per occupied GPU, so declare the *per-card* footprint), and keep
  `max_instances: 1` for a full-pool TP deployment.
- **Locked memory.** radiance's host backend (libavx) builds io_uring
  rings and needs `LimitMEMLOCK=infinity` on the llm-orch service (set on
  the cheyenne host config).
- Serve radiance on `127.0.0.1` without `--api-key`; llm-orch's own key
  is the public-side auth.

## Example config

```yaml
- name: qwen3.8-27b
  context_length: 262144
  max_instances: 1
  max_concurrent: 8
  vram: 16000            # per card (FP8 weights + KV pool ≈ 15 GiB/card)
  idle_ttl: 900
  queue_depth: 16
  gpus: 2                # must match --tp below
  rocm_devices: [0, 1]
  spawn_timeout_secs: 600
  cmd: |
    radiance
      --model /data/models/qwen3.8-27b-fp8.rad
      --tp 2 --tp-wire wht6
      --max-model-len {context_length}
      --max-num-seqs {max_concurrent}
      --kv-cache-dtype fp8
      --num-speculative-tokens 7
      --gpu-headroom-mib 96
      --host 127.0.0.1 --port {port}
```
