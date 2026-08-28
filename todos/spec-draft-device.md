# spec-draft-device control

## Goal

llama.cpp speculative decoding: when a model's `cmd` declares a draft model
(`--spec-draft-model`/`-md`/`--model-draft`, `--spec-draft-hf`/`-hfd`/`--hf-repo-draft`),
the draft must be pinned to the **same device(s) the instance is placed on**
via `--spec-draft-device <VulkanN|CUDAN|ROCmN,…>` (post-filter visible names,
like the existing ROCm `--device ROCmN` handling). Previously llm-orch never
injected the flag, so the draft silently ran on CPU.

Also: explicit per-model placement lists
`vulkan_devices: [[0],[1]]` (single-GPU, either one) and
`vulkan_devices: [[0,1]]` (dual-GPU, using both) in addition to the legacy
flat pool form `[0,1]` + `gpus`.

## Implementation

- [x] `src/backend.rs`: `LlamaCppBackend::inject_spec_draft_device(program, args, kind, indices)`
  - llama-* programs only; no-op when `indices` empty (CPU-only → draft stays CPU)
  - only when a draft-model flag is declared
  - never override an explicit draft-device flag (`--spec-draft-device`/`-devd`/`--device-draft`)
  - skip when draft offload explicitly 0 (`--spec-draft-ngl 0`/`-ngld 0`/`--gpu-layers-draft 0`/`--n-gpu-layers-draft 0`)
  - append `--spec-draft-device Vulkan0,…` using post-filter ranks
  - unit tests (5 new)
- [x] `src/config.rs`: `DevicePool` enum (Pool / Placements), custom Deserialize
  (flat `[0,1]` → Pool; nested `[[0],[1]]` → Placements; `[]` → empty Pool)
- [x] `src/config.rs`: validation — nested form: `gpus` must stay 1, inner lists
  non-empty + no duplicate indices, all indices defined in the device map
  (config tests: parse forms, accept nested, reject gpus/empty/dup/undefined)
- [x] `src/scheduler.rs`: fingerprint hashes canonical (sorted, deduped) index
  set — placement structure and order don't change the spawned process
- [x] `src/scheduler.rs`: selection picks best feasible alternative for
  Placements (not occupied / blocked / enough VRAM; least-loaded, most free
  VRAM, lowest index sum); legacy Pool path unchanged
- [x] `src/scheduler.rs`: `explicit_placement_snapshot` requires an exact
  configured alternative; `spawn_feasible` works via selection
- [x] `src/scheduler.rs`: inject draft-device flag at spawn
  (`try_spawn_with_options`)
- [x] tests: 5 scheduler tests (dual placement, occupied alternative,
  heterogeneous, snapshot rejection, fingerprint) + 1 integration test
  (spawns a `llama-stub` symlink, asserts `--spec-draft-device CUDA0` in the
  real spawned argv; no-injection and explicit-`none` cases)
- [x] docs: config.example.yaml (both forms + Example 5 spec-draft), README
  note; stub backend gained accepted `--spec-draft-model`/`--spec-draft-device`
  flags + `--argv-log` test support

## Verified facts (llama.cpp)

- `--spec-draft-device <dev1,dev2,..>` takes device names as printed by
  `--list-devices` (`Vulkan0`, `CUDA0`, `ROCm0`); `none` = don't offload;
  absent flag = draft on CPU.
- Visibility env vars renumber devices, so names must use post-filter ranks
  (position in the instance's device list) — same rule as the existing
  `--device ROCmN` code.
