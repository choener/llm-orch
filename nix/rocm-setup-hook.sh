# Setup hook for TheRock binary ROCm trees (rocmPackages_10_0.rocm-core).
# Adds the tree's bin/lib dirs and the usual ROCm env vars to consumers.
addEnvHooks "$hostOffset" ''
  if [ -n "''${hostRoot:-}" ]; then
    export HIP_PATH="''${HIP_PATH:-$hostRoot}"
    export CUDA_PATH="''${CUDA_PATH:-$hostRoot}"
    export ROcm_PATH="''${ROcM_PATH:-$hostRoot}"
    export LLVM_HOME="''${LLVM_HOME:-$hostRoot/llvm}"
    addToSearchPath PATH "$hostRoot/bin"
    addToSearchPath PATH "$hostRoot/llvm/bin"
    addToSearchPath LD_LIBRARY_PATH "$hostRoot/lib"
    addToSearchPath LD_LIBRARY_PATH "$hostRoot/llvm/lib"
    addToSearchPath LD_LIBRARY_PATH "$hostRoot/lib/rocm_sysdeps/lib"
    # crt objects for the bare AMD linker driver (ld.lld is invoked without
    # cc-wrapper conventions by hipcc):
    addToSearchPath NIX_LDFLAGS "-B$hostRoot/__LIBC_LIB__/lib"
  fi
''
