# ROCm overlay for llm-orch consumers.
#
# rocmPackages_10_0: "TheRock" 10.0 — AMD's prebuilt binary ROCm distribution
# (stable.repo.amd.com; the same artifacts the Arch AUR rocm-gfxXXXXx-bin
# packages consume). Zero compilation: we fetch AMD's per-ISA tarball, rewrite
# the RUNPATH of every ELF, and wrap the driver binaries with the host
# (glibc/libstdc++) include and crt flags they need on NixOS. Everything
# (hipcc, rocBLAS, hipBLASLt, hipDNN, MIOpen, RCCL, rocFFT, ...) comes from a
# single ~2.4 GB download.
#
# Usage:
#   nixpkgs.overlays = [ inputs.llm-orch.overlays.rocm ];
#   pkgs.rocmPackages_10_0.hipcc                       # default ISA: gfx120x (RDNA4)
#   (pkgs.rocmPackages_10_0.passthru.forIsa "gfx110x").hipcc
#   pkgs.rocmPackages_10_0.rocm-core                   # the full raw tree
#
# rocmPackages_7_14 (source build, adapted from nixpkgs rocm-modules) is
# tracked in .fj/todo-2-recent-rocm-builds.md — not yet implemented.
final: prev:

let
  inherit (final) lib stdenv runCommand fetchurl;

  # TheRock binary distributions, per ISA. All gfx1201 targets (RX 9070 XT,
  # Radeon AI PRO R9700, Strix Halo) are covered by gfx120x.
  theRockVersions = {
    "10_0" = {
      version = "10.0.0";
      isas = {
        gfx120x = {
          url = "https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx120X-all-10.0.0.tar.gz";
          hash = "sha256-65nbQ0oXOP2DsMO5MxRs23ZBjzX89GR3Q/vf73boxx8=";
        };
        gfx110x = {
          url = "https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx110X-all-10.0.0.tar.gz";
          hash = "sha256-H9rEz9of9/LTGolzfUoh+c6Hy4V9JvpGDjXx5FN6Mwo=";
        };
      };
    };
  };

  theRock = theRockVersions."10_0";

  # e.g. x86_64-unknown-linux-gnu (what gcc's include dirs are named after)
  triple = stdenv.hostPlatform.config;

  ccUnwrapped = stdenv.cc.cc or stdenv.cc;
  gccVersion = ccUnwrapped.version or "";
  libc = stdenv.cc.libc or stdenv.cc.cc.libc;

  # Host include/crt flags AMD's bare clang++ needs on NixOS (it does not
  # know cc-wrapper conventions). Validated against glibc-2.42 / gcc-15.
  hostIncludes = lib.concatMapStringsSep " " (p: "-isystem ${p}") ([
    "${ccUnwrapped}/include/c++/${gccVersion}"
    "${ccUnwrapped}/include/c++/${gccVersion}/${triple}"
    "${ccUnwrapped}/lib/gcc/${triple}/${gccVersion}/include"
    "${ccUnwrapped}/lib/gcc/${triple}/${gccVersion}/include-fixed"
    "${stdenv.cc.libc_dev}/include"
  ]);

  # Link-time flags: crt objects, libstdc++, and — crucially — the Nix-store
  # program interpreter, since bare ld.lld would otherwise bake /lib64/ld.so
  # (absent on NixOS) into every consumer binary.
  hostLibs = lib.concatStringsSep " " ([
    "-B${libc}/lib"
    "-L${libc}/lib"
    "-L${stdenv.cc.libc_dev}/lib"
    "-L${ccUnwrapped}/lib/gcc/${triple}/${gccVersion}"
    "-L${ccUnwrapped.lib or ccUnwrapped}/lib"
    "-Wl,--dynamic-linker=${loader}"
  ]);

  loader = stdenv.cc.bintools.dynamicLinker
    or "${libc}/lib/ld-linux-x86-64.so.2";

  # One self-contained ROCm tree for one ISA (raw upstream layout).
  mkRocmCore =
    { version, isa, url, hash }:
    stdenv.mkDerivation {
      pname = "rocm-core-bin-${isa}";
      inherit version isa;

      src = fetchurl { inherit url hash; };

      # The tarball is a flat tree (bin/ lib/ llvm/ ...); unpack it straight
      # into $out ourselves.
      dontUnpack = true;

      # 9+ GB of prebuilt bits: no stripping, no fixup.
      dontStrip = true;
      dontFixup = true;

      installPhase = ''
        runHook preInstall

        mkdir -p $out
        tar xzf $src -C $out

        # Rewrite interpreter + RUNPATH of every ELF so the tree is
        # self-contained (upstream binaries assume a stock /lib64 ld.so and
        # system libs; NixOS refuses to run such binaries at all).
        RPATH="$out/lib:$out/llvm/lib:$out/lib/rocm_sysdeps/lib:$out/lib/rdc/grpc/lib/rocm_sysdeps/lib:${libc}/lib:${ccUnwrapped.lib or ccUnwrapped}/lib"
        find $out -type f \( -perm -u+x -o -name '*.so*' \) |
        while read -r f; do
          head -c4 "$f" | grep -q ELF || continue
          patchelf --set-rpath "$RPATH" --set-interpreter ${loader} "$f" 2>/dev/null || true
        done
        # any ELF left without a Nix-store interpreter is broken — fail loudly
        # Every ELF that carries (or should carry) a program interpreter —
        # i.e. all executables — must point it into the Nix store. Shared
        # objects without PT_INTERP are fine.
        bad=0
        find $out -type f \( -perm -u+x -o -name '*.so*' \) > /tmp/elfs.txt
        while read -r f; do
          head -c4 "$f" | grep -q ELF || continue
          interp=$(patchelf --print-interpreter "$f" 2>/dev/null) || continue
          [ -z "$interp" ] && continue
          case "$interp" in
            /nix/store/*) ;;
            *) echo "ERROR: non-Nix interpreter ($interp): $f"; bad=1 ;;
          esac
        done < /tmp/elfs.txt
        [ "$bad" = 0 ] || exit 1

        runHook postInstall
      '';

      setupHook = ./rocm-setup-hook.sh;

      passthru = {
        inherit isa;
        theRockVersion = version;
      };

      meta = with lib; {
        description = "AMD ROCm ${version} (TheRock) prebuilt binary distribution for ${isa}";
        homepage = "https://rocm.docs.amd.com";
        license = licenses.unfreeRedistributable;
        platforms = platforms.linux;
        sourceProvenance = with sourceTypes; [ binaryNativeCode ];
      };
    };

  # Driver wrappers: set ROCm env, inject host-target include/lib flags
  # exactly once per driver chain (hipcc resolves clang++ through PATH), and
  # fix the interpreter of the wrapped ELF so it runs under NixOS.
  mkWrappedBin = tree:
    let
      driverNames = [ "clang" "clang++" "amdclang" "amdclang++" "hipcc" "hipconfig" "hipify-clang" "hipify-perl" "rocminfo" ];
      treeRpath = lib.concatStringsSep ":" ([
        "${tree}/lib"
        "${tree}/llvm/lib"
        "${tree}/lib/rocm_sysdeps/lib"
        "${tree}/lib/rdc/grpc/lib/rocm_sysdeps/lib"
      ] ++ [
        # runtime libs of the host toolchain: hipcc links libstdc++/libgcc_s
        # but a bare ld.lld link leaves them off the RUNPATH
        "${ccUnwrapped.lib or ccUnwrapped}/lib"
        "${(stdenv.cc.libgcc or stdenv.cc.cc.libgcc or ccUnwrapped).lib or (ccUnwrapped.lib or ccUnwrapped)}/lib"
      ]);
      linkFor = name:
        let
          inLlvm = builtins.elem name [ "clang" "clang++" "amdclang" "amdclang++" ];
          real = "${tree}${if inLlvm then "/llvm/bin/" else "/bin/"}${name}";
          libs = "${hostLibs} -Wl,-rpath,${treeRpath}";
        in
        ''
          if [ -e ${real} ]; then
            sed -e 's|@REAL@|${real}|g' \
                -e 's|@TREE@|${tree}|g' \
                -e 's|@INCLUDES@|${hostIncludes}|g' \
                -e 's|@LIBS@|${libs}|g' \
                ${./driver-wrapper.sh.in} > $out/bin/${name}
            chmod +x $out/bin/${name}
          fi
        '';
    in
    runCommand "rocm-drivers-${tree.isa}" { } ''
      mkdir -p $out/bin
      ${lib.concatStringsSep "\n" (map linkFor driverNames)}
    '';

  # Compile-time smoke test: builds (does not run) a trivial HIP kernel with
  # the wrapped hipcc. The result runs on any gfx1201 machine: `nix run
  # .#hip-smoke` prints the device name.
  mkHipSmoke = treeSet:
    final.runCommand "hip-smoke-${treeSet.rocm-core.isa}"
      {
        nativeBuildInputs = [ treeSet.hipcc ];
      }
      ''
        cat > smoke.hip <<'EOF'
        #include <hip/hip_runtime.h>
        #include <cstdio>
        __global__ void add(const float*b,float*a){a[0]+=b[0];}
        int main(){
          float *a,*b; hipMalloc(&a,4); hipMalloc(&b,4);
          float h=1,g=2;
          hipMemcpy(a,&h,4,hipMemcpyHostToDevice); hipMemcpy(b,&g,4,hipMemcpyHostToDevice);
          add<<<1,1>>>(b,a);
          auto e=hipDeviceSynchronize(); if(e!=hipSuccess){printf("ERR %s\n",hipGetErrorString(e));return 1;}
          hipMemcpy(&h,a,4,hipMemcpyDeviceToHost);
          hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
          printf("kernel result 1+2=%.0f | GPU: %s (%s, %zu MB)\n",h,p.name,p.gcnArchName,p.totalGlobalMem>>20);
          return 0;}
        EOF
        hipcc smoke.hip -o $out --offload-arch=${hipTargetOf treeSet.rocm-core.isa}
      '';

  hipTargetOf = isa: {
    gfx120x = "gfx1201";
    gfx110x = "gfx1101";
  }.${isa} or "gfx1201";

  forIsa = isa:
    assert lib.assertMsg (lib.hasAttr isa theRock.isas)
      "rocmPackages_10_0: unknown ISA '${isa}' (known: ${lib.concatStringsSep ", " (builtins.attrNames theRock.isas)})";
    let
      rocm-core = mkRocmCore ({ isa = isa; } // (theRock.isas)."${isa}" // { version = theRock.version; });
    in
    let
      hipcc = mkWrappedBin rocm-core;
    in
    {
      inherit rocm-core hipcc;
      hip-smoke = mkHipSmoke {
        inherit rocm-core hipcc;
      };
      passthru = { inherit isa; };
    };

in
{
  rocmPackages_10_0 =
    let
      default = forIsa "gfx120x";
    in
    lib.recurseIntoAttrs ({
      inherit (default) rocm-core hipcc hip-smoke;
      passthru = {
        forIsa = isa: forIsa isa;
        isas = builtins.attrNames theRock.isas;
      };
    });
}
