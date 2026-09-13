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
# rocmPackages_7_14: ROCm 7.14.1 assembled from AMD's official prebuilt pip
# SDK wheels (repo.amd.com/whl-multi-arch: rocm_sdk_core + rocm_sdk_devel +
# per-arch rocm_sdk_device_gfxNNNN), merged into one classic ROCm tree and
# NixOS-ified the same way. Also zero compilation.
#   pkgs.rocmPackages_7_14.hipcc            # default gfx: gfx1201
#   (pkgs.rocmPackages_7_14.passthru.forGfx "gfx1250")
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

  # ROCm 7.14.1 via AMD's pip SDK wheels (the exact userspace the AMD
  # torch[device-gfx1201] wheels run on). core = runtime libs + hipcc,
  # devel = llvm/clang/headers/cmake, device = per-ISA kernel packages
  # (rocBLAS/hipBLASLt Tensile, RCCL, MIOpen, ... as .kpack + plain files).
  sdkVersions = {
    "7_14" = {
      version = "7.14.1";
      base = "https://repo.amd.com/rocm/whl-multi-arch";
      core = {
        url = "https://repo.amd.com/rocm/whl-multi-arch/rocm_sdk_core-7.14.1-py3-none-linux_x86_64.whl";
        hash = "sha256-2h+clgpVpkwxNABLoTRtJCQJDMPiM2zP7JTp9iFpz4U=";
      };
      devel = {
        url = "https://repo.amd.com/rocm/whl-multi-arch/rocm_sdk_devel-7.14.1-py3-none-linux_x86_64.whl";
        hash = "sha256-hYCIrJSdRb5mIVV+XXt/AgI14SBoxitr4r/QtCNL490=";
      };
      # per-ISA device wheels published on the same index
      isas = [ "gfx1200" "gfx1201" "gfx1250" ];
      deviceUrls = {
        gfx1201 = {
          url = "https://repo.amd.com/rocm/whl-multi-arch/rocm_sdk_device_gfx1201-7.14.1-py3-none-linux_x86_64.whl";
          hash = "sha256-61VNSdDNMElE3iGZ7EKN9fmdgL2Va3y3L8H5qJeDf6U=";
        };
      };
    };
  };

  # e.g. x86_64-unknown-linux-gnu (what gcc's include dirs are named after)
  triple = stdenv.hostPlatform.config;

  ccUnwrapped = stdenv.cc.cc or stdenv.cc;
  gccVersion = ccUnwrapped.version or "";
  libc = stdenv.cc.libc or stdenv.cc.cc.libc;

  # Host include/crt flags AMD's bare clang++ needs on NixOS (it does not
  # know cc-wrapper conventions).
  #
  # C++ headers come from gcc-13, NOT stdenv's gcc: clang-HIP chokes on
  # gcc-15's libstdc++ (__glibcxx_assert_fail referenced from
  # __host__ __device__ code); AMD's own containers pin gcc-13 for the same
  # reason. libstdc++ forward-compat (13 headers on 15 runtime) is safe.
  # The wrapper additionally front-loads the tree's clang cuda_wrappers dir
  # (see mkWrappedBin) so <new>/<cmath> interception isn't bypassed.
  gcc13 = final.gcc13.cc or final.gcc13 or ccUnwrapped; # unwrapped: has include/c++ headers
  gcc13Version = gcc13.version or gccVersion;

  hostIncludes = lib.concatMapStringsSep " " (p: "-isystem ${p}") ([
    "${gcc13}/include/c++/${gcc13Version}"
    "${gcc13}/include/c++/${gcc13Version}/${triple}"
    "${gcc13}/lib/gcc/${triple}/${gcc13Version}/include"
    "${gcc13}/lib/gcc/${triple}/${gcc13Version}/include-fixed"
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

  # Rewrite interpreter + RUNPATH of every ELF in $out so a fetched ROCm
  # tree is self-contained (upstream binaries assume a stock /lib64 ld.so
  # and system libs; NixOS refuses to run such binaries at all).
  elfSweep = ''
    RPATH="$out/lib:$out/llvm/lib:$out/lib/rocm_sysdeps/lib:$out/lib/rdc/grpc/lib/rocm_sysdeps/lib:${libc}/lib:${ccUnwrapped.lib or ccUnwrapped}/lib"
    find $out -type f \( -perm -u+x -o -name '*.so*' \) |
    while read -r f; do
      head -c4 "$f" | grep -q ELF || continue
      patchelf --set-rpath "$RPATH" --set-interpreter ${loader} "$f" 2>/dev/null || true
    done
    # Every ELF that carries (or should carry) a program interpreter —
    # i.e. all executables — must point it into the Nix store. Shared
    # objects without PT_INTERP are fine. Fail loudly on leftovers.
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
  '';

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

        ${elfSweep}

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
                -e "s|@INCLUDES@|$CUDA_INJ ${hostIncludes}|g" \
                -e 's|@LIBS@|${libs}|g' \
                ${./driver-wrapper.sh.in} > $out/bin/${name}
            chmod +x $out/bin/${name}
          fi
        '';
    in
    runCommand "rocm-drivers-${tree.isa}" { } ''
      mkdir -p $out/bin
      # Front-load clang's cuda_wrappers dir so <new>/<cmath> interception
      # is not bypassed by the later libstdc++ -isystem entries (HIP mode).
      CUDA_WRAP=$(ls -d ${tree}/llvm/lib/clang/*/include/cuda_wrappers 2>/dev/null | head -1)
      CUDA_INJ=""
      [ -n "$CUDA_WRAP" ] && CUDA_INJ="-isystem $CUDA_WRAP"
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

  hipTargetOf =
    isa:
    let map = { gfx120x = "gfx1201"; gfx110x = "gfx1101"; }; in
    if builtins.hasAttr isa map then map.${isa} else isa;

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

  # ---- ROCm 7.14.x from AMD's pip SDK wheels -----------------------------
  # Merges _rocm_sdk_core (runtime) + _rocm_sdk_libraries (ISA kernels,
  # incl. .kpack archives) + _rocm_sdk_devel (llvm/clang/headers, from a
  # nested tar) into one classic /opt/rocm-style tree, then applies the
  # same NixOS ELF sweep.
  mkSdkCore =
    { version, gfx, core, devel, device }:
    stdenv.mkDerivation {
      pname = "rocm-sdk-${gfx}";
      inherit version gfx;
      # isa attribute: driver wrappers and the smoke test key off it
      passthru.isa = gfx;

      srcs = [ core devel device ];
      dontUnpack = true;
      nativeBuildInputs = [ final.unzip final.patchelf ];

      dontStrip = true;
      dontFixup = true;

      installPhase = ''
        runHook preInstall

        mkdir -p $out work
        # unpack all three wheels into one dir, mirroring how pip installs
        # them as siblings in site-packages (their cross-package relative
        # symlinks, e.g. devel's bin/hipcc -> ../../_rocm_sdk_core/bin/hipcc,
        # only resolve in that sibling layout)
        unzip -q ${core} -d work
        unzip -q ${device} -d work
        unzip -q ${devel} -d work
        tar xf work/rocm_sdk_devel/_devel.tar -C work

        # flatten into one classic ROCm tree: runtime first, ISA kernels
        # next, devel last so its fuller hipcc/llvm/headers win overlaps
        cp -a work/_rocm_sdk_core/. $out/
        cp -a work/_rocm_sdk_libraries/. $out/
        cp -a work/_rocm_sdk_devel/. $out/

        # flattening breaks the cross-package relative links: materialise
        # them by resolving against the intact sibling layout in work/
        find $out -type l | while read -r l; do
          [ -e "$l" ] && continue
          t=$(readlink "$l")
          case "$t" in /*) continue ;; esac
          rel=''${l#$out/}
          for pkg in _rocm_sdk_core _rocm_sdk_libraries _rocm_sdk_devel; do
            abs=$(realpath -m "work/$pkg/$(dirname "$rel")/$t" 2>/dev/null)
            case "$abs" in "$PWD/work"/*) [ -e "$abs" ] || continue ;; *) continue ;; esac
            rm -f "$l"
            cp -fL "$abs" "$l"
            break
          done
        done
        rm -rf work

        ${elfSweep}

        runHook postInstall
      '';

      setupHook = ./rocm-setup-hook.sh;

      meta = with lib; {
        description = "AMD ROCm ${version} prebuilt SDK wheels for ${gfx}";
        homepage = "https://rocm.docs.amd.com";
        license = licenses.unfreeRedistributable;
        platforms = platforms.linux;
        sourceProvenance = with sourceTypes; [ binaryNativeCode ];
      };
    };

  forSdk =
    v: gfx:
    let
      spec = sdkVersions."${v}";
      dev = spec.deviceUrls.${gfx}
        or (throw "rocmPackages_${v}: no cached hash for device wheel '${gfx}' (known: ${lib.concatStringsSep ", " (builtins.attrNames spec.deviceUrls)}); add url+hash from ${spec.base}");
      rocm-core = mkSdkCore {
        inherit (spec) version;
        inherit gfx;
        core = fetchurl spec.core;
        devel = fetchurl spec.devel;
        device = fetchurl dev;
      };
      hipcc = mkWrappedBin rocm-core;
    in
    {
      inherit rocm-core hipcc;
      hip-smoke = mkHipSmoke { inherit rocm-core hipcc; };
      passthru = { inherit gfx; };
    };

in
{
  rocmPackages_7_14 =
    let
      default = forSdk "7_14" "gfx1201";
    in
    lib.recurseIntoAttrs ({
      inherit (default) rocm-core hipcc hip-smoke;
      passthru = {
        forGfx = gfx: forSdk "7_14" gfx;
        gfxes = builtins.attrNames sdkVersions."7_14".deviceUrls;
      };
    });

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
