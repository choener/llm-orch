# Bug: GPU placement leaks onto devices outside the instance's placement

## Symptom

`laguna-xs-2.1` with `vulkan_devices: [0, 1]` and `gpus: 1` — a **single** `llama-server`
instance spans **both** GPUs. The same model on the ROCm namespace is correctly restricted,
because llm-orch injects `--device ROCm0` there and nowhere else:

```
llama-server ... --n-gpu-layers all ... --model .../Laguna-XS-2.1-...gguf              # Vulkan: spans 2 GPUs
llama-server ... --n-gpu-layers all ... --model .../Laguna-XS-2.1-...gguf --device ROCm0  # ROCm: 1 GPU
```

## Root cause

llm-orch pins devices with the **per-backend visibility env var only**
(`GGML_VK_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`,
`src/backend.rs:280-295`). Those variables filter **only their own backend's** device list.
In a multi-backend `llama-server` build (Vulkan + ROCm, or Vulkan + CUDA), the other backend
still enumerates the remaining GPUs, so llama.cpp sees more than one accel device, selects
row-split, and `--n-gpu-layers all` offloads across all of them.

The device-name selector that would prevent this — `--device <Backend><rank>` — is emitted
**only for ROCm** (`LlamaCppBackend::gpu_args`, `src/backend.rs:266-277`). Vulkan and CUDA
placements get no `--device` at all.

## Secondary defect

`gpu_args` appends `--device` unconditionally. A ROCm model whose `cmd` already declares its
own `--device` receives a second one, and llama.cpp keeps the last occurrence — llm-orch
silently overrides the operator. The draft-device path has the correct guard
(`DRAFT_DEVICE_FLAGS`, `src/backend.rs:87` and `:227`); the main device path does not.

## Consequence for scheduling

VRAM accounting is keyed to the placement's PCI slots (`src/scheduler.rs:3602-3614`). Memory
consumed outside the placement is invisible to the scheduler, so it can pack another model
onto the GPU the leaked instance is already using.

---

# TODO — Fix: hard `--device` pinning for all llama.cpp GPU kinds

## Goal

Every GPU-pinned llama.cpp instance gets an explicit `--device <Prefix><rank>[,…]` selector
(Vulkan0, CUDA0, ROCm0, …) in addition to the per-backend visibility env var, so a
multi-backend llama-server build can never spread an instance beyond its placement.

## Design decisions (verified against llama.cpp source @ master)

- **Device names are post-filter ranks.** llama.cpp names devices `<Prefix><i>` where `i`
  is the index into the *filtered* list; `GGML_VK_VISIBLE_DEVICES=1` makes that GPU
  `Vulkan0` (env-list order = rank order; same for `CUDA_VISIBLE_DEVICES` /
  `HIP_VISIBLE_DEVICES`). So `--device Vulkan0,…` with ranks `0..indices.len()` is exact —
  this is the non-fragile naming scheme; the env var stays set alongside (shrinks
  enumeration, keeps rank↔PCI-slot mapping valid).
- **Without `--device`, llama.cpp uses *all* enumerated GPUs** (all backends: ROCm0/1 +
  Vulkan0/1 on the reporter's box) and row-splits across them — the leak. `--device` is
  the only hard stop.
- **User-declared device flags are a config error** (`--device`/`-dev`, and the draft
  flags `--spec-draft-device`/`--device-draft`/`-devd`): llm-orch owns placement; mixing
  operator and scheduler placement breaks VRAM accounting. Operators wanting a CPU draft
  use `--spec-draft-ngl 0` (already respected). Validation scans the *resolved* cmd
  (alias-expanded), llama- programs only — same pattern as `cpu_cmd_offload_decl`.
- **Spawn-time append stays unconditional** (defense in depth): llama.cpp is
  last-value-wins, and llm-orch appends after user args, so an unvalidated cmd can never
  escape its placement.
- **CPU-only models need no new knob**: omitting all device pools already means CPU-only
  (`gpus: 1` required, offload flags rejected, `--n-gpu-layers 0` enforced). Missing piece
  is documentation/an example.
- `--list-devices` is **not** wired in (per discussion): free-MiB is backend-blind across
  Vulkan/ROCm and the operator-maintained `devices.{vulkan,cuda,rocm}` index→PCI-slot
  maps remain the source of truth.

## Tasks

### 1. Hard `--device` pin for Vulkan and CUDA (`src/backend.rs`)
- [x] `LlamaCppBackend::gpu_args`: emit `--device <Prefix><rank>[,…]` for all three kinds
      (drop the `kind != Rocm` early-return; keep post-filter ranks `0..indices.len()`,
      keep empty-indices → no args)
- [x] Update `DeviceKind` / `LlamaCppBackend` doc comments (pinning is now env var +
      `--device` for every kind)
- [x] Update unit tests: replace `gpu_args_for_non_rocm_backends_are_empty` with
      per-kind assertions (`Vulkan0,Vulkan1` for `[0,2]`, `CUDA0` for `[1]`, etc.)

### 2. Config validation: reject operator device flags (`src/config.rs`)
- [x] Add `cmd_device_flag_decl()` helper mirroring `cpu_cmd_offload_decl`: resolve
      aliases, skip non-`llama-` programs, detect `--device`/`-dev` and
      `--spec-draft-device`/`--device-draft`/`-devd` in both `--flag value` and
      `--flag=value` forms; return a human-readable description of the offender
- [x] Wire into model validation for **every** llama.cpp model (GPU and CPU-only):
      error message points at llm-orch's placement ownership (`vulkan_devices`/
      `cuda_devices`/`rocm_devices`; CPU draft via `--spec-draft-ngl 0`)
- [x] Unit tests: each flag, both forms, alias-resolved cmds, non-llama program exempt,
      `--spec-draft-ngl 0` still accepted

### 3. Scheduler comment (`src/scheduler.rs`)
- [x] Update the "Device pinning follows the model's namespace" comment (~:3160) to
      state that all kinds now get env var + `--device` rank pin

### 4. Integration tests (`tests/integration.rs`, stub `--argv-log`)
- [x] Vulkan-model test: spawned argv contains `--device Vulkan0` (pool `[[0]]` form) and
      env pin — extend or mirror the existing draft-device test section
- [x] ROCm-model test: still `--device ROCm0` (no regression) — covered by the CUDA
      namespace test (`--device CUDA0`), same code path
- [x] Multi-GPU pool test: `vulkan_devices: [0,1]`, `gpus: 2` → `--device Vulkan0,Vulkan1`
      in deterministic selection order
- [x] Post-filter rank test: `vulkan_devices: [[1]]` → `--device Vulkan0`
      (needed a `real_drm_slots()` helper — generic PCI slots fail devices.vulkan
      validation)
- [x] Dropped the `explicit` model from the draft-device test: user-declared
      `--spec-draft-device` is now a config error by design

### 5. Docs & example config
- [x] `config.example.yaml`: fix Example 2d (ROCm-only `--device` claim), document the
      rank naming (`GGML_VK_VISIBLE_DEVICES=2,4` → `Vulkan0,Vulkan1`), state that
      user-declared `--device`/draft-device flags are rejected
- [x] `config.example.yaml`: add a CPU-only embedding-model example (no device pool,
      `gpus: 1`, `--embeddings`, note that llm-orch enforces `--n-gpu-layers 0`)

### 6. Verify
- [x] `cargo test` green (221 unit + 25 integration, incl. 4 new device-pin tests
      exercising the host's real 2× AMD DRM cards)
- [x] `cargo clippy --all-targets` warning-free, `cargo fmt`

## Notes

- Behavior change is backward-incompatible for configs that declare their own
  `--device`/draft-device flags: hot-reload rejects such a config (old one keeps
  running), startup fails fast. This is deliberate (fail-fast precedent:
  `cpu_cmd_offload_decl`).
- Non-llama.cpp backends (sd.cpp, audiocpp) untouched — they have no `--device`.
- Draft-device *injection* logic (`inject_spec_draft_device`, incl. `--spec-draft-ngl 0`
  and ngram handling) is unchanged; its user-flag guards become unreachable for
  validated configs but stay as defense in depth.
