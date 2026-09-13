# 004 — Native vLLM-radiance on NixOS (no containers)

Issue #2 follow-through: run the [vllm-radiance](https://github.com/magiccodingman/vllm-radiance)
stack (vLLM 0.28.0 + libr4d RDNA4 kernels, pinned for gfx1201: RX 9070 XT,
Radeon AI PRO R9700, Strix Halo) directly on NixOS — no docker/podman, no ROCm
source build. Verified 2026-09-13: `vllm serve` boots, radiance hooks arm,
completions + chat work on an RX 9070 (gfx1201).

Companion piece: `nix/rocm.nix` provides the ROCm toolchain (hipcc +
headers) as prebuilt binaries — `rocmPackages_10_0` (TheRock 10.0) or
`rocmPackages_7_14` (AMD's pip SDK wheels) — either can compile libr4d and
the mxfp4 extension; the prototype used a TheRock tree directly.

## Components & sources

| piece | source | how |
|---|---|---|
| torch 2.12.0+rocm7.14.1, torchvision 0.27.0, triton 3.8 | AMD wheel index `https://repo.amd.com/rocm/whl-multi-arch/` | prebuilt wheels: `pip install --index-url ... "torch[device-gfx1201]==2.12.0+rocm7.14.1"` — pulls the ROCm pip SDK (`rocm_sdk_*`) with it |
| aiter 0.1.20 (`fc2e5d57`) | ROCm/aiter | `GPU_ARCHS=gfx1201 PREBUILD_KERNELS=0 AITER_USE_SYSTEM_TRITON=1 pip wheel --no-build-isolation` |
| vllm 0.28.0 (`2cf0a691`) | vllm-project/vllm | `python use_existing_torch.py; VLLM_TARGET_DEVICE=rocm VLLM_VERSION_OVERRIDE=0.28.0 pip wheel --no-build-isolation --no-deps` |
| libr4d v0.5.0 (`e8de4bc1`) | codeberg StillDeadcode/libr4d | `./build.sh` (plain hipcc, 17 TUs) + `r4d_radiance_extras.patch` from the radiance repo |
| radiance_mxfp4_fp8.so | radiance repo | `hipcc -shared --offload-arch=gfx1201 $(python -m pybind11 --includes)` |
| ~41 `patch_*.py` + `radiance_*.py` + configs | radiance repo | run in the Dockerfile's exact order against site-packages |

## NixOS gotchas (all hit, all solved)

1. **pip-vendored manylinux binaries are dead on NixOS.** `pip install
   cmake ninja` puts unrunnable `/lib64`-interp binaries into `venv/bin`.
   Symlink `venv/bin/{cmake,ninja}` to nix store binaries (cmake 3.x —
   vllm rejects cmake 4; `which -a` order matters).
2. **Never put a glibc dir on `LD_LIBRARY_PATH`** for nix-built python —
   instant segfault. libstdc++/numa/elf dirs are fine and required
   (`libnuma` for rocSHMEM, `libstdc++.so.6` for the ROCm SDK preload).
3. **An ambient `PYTHONPATH` (e.g. from a devshell) silently poisons the
   venv** — a 3.14 setuptools leaked into 3.12 builds (cp314 wheel tags).
   `unset PYTHONPATH` before anything.
4. **clang-HIP cannot consume gcc-15 libstdc++ headers**
   (`__glibcxx_assert_fail` / `operator new` in `__host__ __device__`).
   The upstream Dockerfile pins gcc-13 for the same reason. Inject gcc-13
   c++ headers via `-isystem` in a hipcc/clang++ wrapper, and put TheRock's
   `lib/llvm/lib/clang/23/include/cuda_wrappers` FIRST so `<new>`/`<cmath>`
   interception isn't bypassed by later `-isystem`s.
5. **cmake ≥ 3.30 refuses hipcc** as `CMAKE_HIP_COMPILER` — point it at a
   wrapped `clang++`. Host (CXX) side can stay gcc-15.
6. **TheRock's `hipcc_cmake_linker_helper` has a `#!/bin/bash` shebang**
   (absent on NixOS) and derives `HIP_PATH` from its own location — copy it
   next to your wrapper `bin/` under a layout `<root>/bin`, and make the
   root look like a ROCm root (`lib`/`include`/`llvm` symlinks).
7. **amdsmi**: install TheRock's `share/amd_smi` python package as a
   symlink into site-packages (PyPI `amdsmi` can't locate its `.so` on
   nix). `radiance_amdsmi.pth` then enumerates both GPUs correctly at
   interpreter start.
8. Linking consumer HIP binaries needs
   `-Wl,--dynamic-linker=<nix ld>`, `-L <gcc>/lib/gcc/...` (bare `ld.lld`
   wants `-lgcc`) and RUNPATH entries for the ROCm tree + libstdc++/libgcc_s.

## Runtime env (serve)

`HIP_PLATFORM=amd`, `VLLM_TARGET_DEVICE=rocm`,
`PYTORCH_ROCM_ARCH=RADIANCE_GFX_ARCH=gfx1201`,
`LD_LIBRARY_PATH=<gcc-lib>:<numa>:<libelf>` (torch's `rocm-bootstrap`
locates the rest of ROCm inside site-packages). Optional radiance switches
per the image: `RADIANCE_USE_R4D=1 RADIANCE_USE_R4D_GDN=1
RADIANCE_VERIFY_HEAD=1`.

## Prototype location & next step

The working prototype lives in `tmp/vr/` (gitignored): `env.sh`,
`setup-venv.sh`, `build.sh` (torch/aiter/vllm wheels), `build2.sh`
(libr4d/mxfp4/patches), `serve.sh`. Verified: Qwen3-0.6B completions and
chat on gfx1201.

Next: turn this into a proper `pkgs.vllm-radiance` derivation (python
venv build + TheRock toolchain from `nix/rocm.nix`), benchmark against the
published container image, and wire it into the llm-orch config examples.
ROCm 7.14 *source* overlay (`rocmPackages_7_14`) remains optional — the
binary SDK route proved sufficient.
