# NixOS module for llm-orch — single-host LLM orchestrator.
#
# The flake wires `services.llm-orch.package` to the flake's own build;
# when using this module standalone, set it yourself.
{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.llm-orch;
  defaultUserGroup = "llm-orch";
in
{
  options.services.llm-orch = {
    enable = lib.mkEnableOption "llm-orch — single-host LLM/prompt orchestrator for llama.cpp and stable-diffusion.cpp backends";

    enableContainerSupport = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Enable podman container support.  When enabled, llm-orch can manage
        container-based backends (vLLM, TGI, etc.) via `podman run --rm` in
        addition to native subprocess backends.

        This option configures:
        - podman and NixOS's `newuidmap`/`newgidmap` wrappers on the service
          PATH
        - Rootless Podman compatibility settings: `NoNewPrivileges=false`,
          `PrivateUsers=false`, `RestrictSUIDSGID=false`, `AF_NETLINK`, and
          cgroup delegation. The unit-level capability bound is intentionally
          unset so Podman can set up nested namespaces and unpack normal image
          metadata.
        - Subordinate UID/GID mappings and a persistent systemd user session
          (linger) for the configured llm-orch service user
        - `HOME` set to the service state directory so podman's storage
          (`~/.local/share/containers`) lives under `/var/lib/llm-orch`
        - Podman runtime state (`runroot` and the engine `tmp_dir`) on
          tmpfs under `/run/llm-orch`, so a stale boot ID cannot survive a
          reboot; image layers stay in the state directory
        - `podman system migrate` before every service start, so persisted
          rootless storage uses the configured subordinate-ID mappings
        - Optional one-time image loading from tar archives (see
          {option}`services.llm-orch.containerImages`)

        The configured service user must be able to pull images (`podman
        login` if needed).
      '';
    };

    containerImages = lib.mkOption {
      type = lib.types.listOf (
        lib.types.submodule {
          options = {
            image = lib.mkOption {
              type = lib.types.str;
              example = "vllm-radiance:0.11.0";
              description = ''
                Container image name as it will appear in `podman images`
                after loading (e.g. `vllm-radiance:0.11.0`).
              '';
            };
            path = lib.mkOption {
              type = lib.types.str;
              example = "/data/vllm/container/vllm-radiance.tar";
              description = ''
                Absolute path to the tar archive (created with `podman
                save`) that contains the image.  Must be readable by the
                service user.
              '';
            };
          };
        }
      );
      default = [ ];
      example = lib.literalExpression ''
        [
          { image = "vllm-radiance:0.11.0"; path = "/data/vllm/container/vllm-radiance.tar"; }
        ]
      '';
      description = ''
        Pre-built container images to load before starting the service.
        Each tar archive is loaded with `podman load -i` only if the
        named image is not already present in podman storage.  Images
        are loaded on every start (the check is fast — `podman image
        exists` queries the local database — and loading is skipped
        when the image is found, making subsequent starts cheap).

        Tar archives must be created with `podman save IMAGE -o
        OUTPUT.tar` on the host that built the container. If loading a valid
        archive reports insufficient UIDs/GIDs while unpacking a path such as
        `/etc/gshadow`, that is a rootless user-namespace mapping problem,
        not an archive-format problem. Enabling container support configures
        the mapping and migrates this service's Podman storage before loading.
      '';
    };

    package = lib.mkOption {
      type = lib.types.package;
      description = ''
        The llm-orch package to run. Defaults to the flake's own build when
        the module is imported via `nixosModules.llm-orch`.
      '';
    };

    configFile = lib.mkOption {
      type = lib.types.path;
      example = "/var/lib/llm-orch/config.yaml";
      description = ''
        Path to the llm-orch configuration file. It must be readable by the
        service user (see {option}`services.llm-orch.user`); world-readable
        is fine.

        Relative paths inside the config are resolved against the state
        directory, `/var/lib/llm-orch`. The config deliberately does not say
        where API keys come from — see {option}`services.llm-orch.apikeysFile`
        — so this file can be handed to a wider group of editors without
        letting them redirect authentication.

        The file is hot-reloaded at runtime; replace it atomically (write to a
        temp file, then `mv`) so the watcher fires reliably.
      '';
    };

    apikeysFile = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = "/var/lib/llm-orch/apikeys.txt";
      example = lib.literalExpression ''"/run/secrets/llm-orch-apikeys"'';
      description = ''
        Path to the API keys file, passed to llm-orch as `--apikeyfile`.
        It lives outside the config on purpose: whoever may edit
        {option}`configFile` must not be able to point authentication at a
        file of their own.

        The file must be owned by the service user and readable by no one
        else (mode 0400 or 0600); one `label: key` per line. Its *contents*
        are hot-reloaded, its path is fixed for the lifetime of the service.

        Set to `null` to run without any keys — llm-orch is default-deny, so
        every request is rejected.
      '';
    };

    requireHashedApikeys = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Pass `--require-hashed-apikeys`: refuse plaintext apikeys and accept
        only `sha256:` digest entries. Turn this on once every entry in
        {option}`apikeysFile` has been converted with
        `llm-orch hash-apikeys --write`; llm-orch itself never writes the file.
      '';
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = defaultUserGroup;
      description = ''
        User account under which llm-orch runs.

        If left at the default, a dedicated system user is created
        automatically (with membership in the `video` and `render` groups
        for GPU device access). Set this to an existing user if your model
        files are owned by that user and not otherwise readable.
      '';
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = defaultUserGroup;
      description = ''
        Group under which llm-orch runs. Created automatically if left at
        the default.
      '';
    };

    llamaPackage = lib.mkOption {
      type = lib.types.nullOr lib.types.package;
      default = pkgs.llama-cpp;
      defaultText = lib.literalExpression "pkgs.llama-cpp";
      description = ''
        llama.cpp package providing `llama-server`, added to the service's
        PATH so model `cmd` definitions can invoke it by name. Set to `null`
        to opt out (e.g. if your config uses absolute store paths).
      '';
    };

    stableDiffusionCppPackage = lib.mkOption {
      type = lib.types.nullOr lib.types.package;
      default = null;
      defaultText = lib.literalExpression "pkgs.stable-diffusion-cpp";
      example = lib.literalExpression "pkgs.stable-diffusion-cpp";
      description = ''
        stable-diffusion.cpp package providing `sd-server`, added to the
        service's PATH so diffusion model `cmd` definitions can invoke it
        by name. Set to `null` (the default) if you don't use diffusion
        models.
      '';
    };

    nvidiaPackage = lib.mkOption {
      type = lib.types.nullOr lib.types.package;
      default =
        if lib.any (d: lib.hasPrefix "nvidia" d) config.services.xserver.videoDrivers then
          config.hardware.nvidia.package.bin
        else
          null;
      defaultText = lib.literalExpression ''
        if NVIDIA is in services.xserver.videoDrivers:
          config.hardware.nvidia.package.bin
        else null
      '';
      description = ''
        NVIDIA driver userspace package providing `nvidia-smi`, added to
        the service's PATH on CUDA hosts. Without it, GPU metrics degrade
        to static `vram_mb` totals from the config (a warning is logged
        once). The default uses the exact driver version loaded by the
        kernel. Set to `null` to opt out.
      '';
    };

    extraPackages = lib.mkOption {
      type = lib.types.listOf lib.types.package;
      default = [ ];
      example = lib.literalExpression "[ pkgs.python3 ]";
      description = ''
        Additional packages placed on the service's PATH, available to model
        `cmd` commands and keep-alive hooks.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    # Keep all attributes for the dynamic service-user key in one definition:
    # Nix does not merge separate dynamic attribute paths for the same user.
    users.users.${cfg.user} =
      lib.optionalAttrs (cfg.user == defaultUserGroup) {
        isSystemUser = true;
        group = cfg.group;
        extraGroups = [
          "video"
          "render"
        ];
        description = "llm-orch service user";
      }
      // lib.optionalAttrs cfg.enableContainerSupport {
        # Rootless podman needs subordinate UID/GID mappings for user
        # namespace creation. These must follow `cfg.user`, including when
        # the service runs as an existing account. Allocate 65536 subordinate
        # IDs starting at a high offset to avoid collisions with real users.
        subUidRanges = [
          {
            startUid = 200000;
            count = 65536;
          }
        ];
        subGidRanges = [
          {
            startGid = 200000;
            count = 65536;
          }
        ];
        # Keep the user's systemd manager available for Podman's systemd
        # cgroup manager even though llm-orch itself is a system service.
        linger = true;
      };

    users.groups.${cfg.group} = lib.mkIf (cfg.group == defaultUserGroup) { };

    systemd.services.llm-orch = {
      description = "llm-orch — single-host LLM orchestrator";
      wantedBy = [ "multi-user.target" ];
      wants = lib.optional cfg.enableContainerSupport "linger-users.service";
      after = [ "network.target" ] ++ lib.optional cfg.enableContainerSupport "linger-users.service";

      # Crash-loop protection: llm-orch exits immediately on e.g. a broken
      # config, so without a limit systemd would restart it forever in a
      # tight loop. 3 failed starts within 5 minutes mark the unit failed
      # (`start-limit-hit`); recover with `systemctl reset-failed llm-orch`
      # followed by `systemctl start llm-orch`. mkDefault so host configs
      # can override without mkForce.
      unitConfig = {
        StartLimitBurst = lib.mkDefault 3;
        StartLimitIntervalSec = lib.mkDefault 300;
      };

      # bash provides `sh` and coreutils `cat` — both are required by the
      # keep-alive mechanism (`sh -c <cmd>`, default cmd reads
      # gpu_busy_percent via cat). Without them on PATH the keep-alive
      # spawn fails with ENOENT: NixOS replaces the unit PATH entirely once
      # `path` is non-empty, and llama-cpp's bin dir has no shell.
      path = [
        pkgs.runtimeShellPackage
        pkgs.coreutils
      ]
      # Rootless Podman must find NixOS's setuid newuidmap/newgidmap
      # wrappers, which live at /run/wrappers/bin.
      ++ lib.optional cfg.enableContainerSupport "/run/wrappers"
      ++ lib.optional cfg.enableContainerSupport pkgs.podman
      ++ lib.optional (cfg.llamaPackage != null) cfg.llamaPackage
      # stable-diffusion.cpp (sd-server) for diffusion model backends.
      ++ lib.optional (cfg.stableDiffusionCppPackage != null) cfg.stableDiffusionCppPackage
      # nvidia-smi (live VRAM metrics on CUDA hosts): the driver bin
      # output is only on PATH when NVIDIA was detected on the host.
      ++ lib.optional (cfg.nvidiaPackage != null) cfg.nvidiaPackage
      ++ cfg.extraPackages;

      serviceConfig = {
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${lib.getExe cfg.package} --config ${cfg.configFile}${
          lib.optionalString (cfg.apikeysFile != null) " --apikeyfile ${cfg.apikeysFile}"
        }${lib.optionalString cfg.requireHashedApikeys " --require-hashed-apikeys"}";
        Restart = "on-failure";
        RestartSec = 5;

        # Relative paths in the config (e.g. a relative stats file) resolve
        # against this directory. Owned by the service user.
        StateDirectory = "llm-orch";
        WorkingDirectory = "/var/lib/llm-orch";

        # ── Hardening ─────────────────────────────────────────────────
        # Native backends get the hardened service profile.  Container mode
        # instead relies on rootless Podman as the isolation boundary and
        # drops the systemd sandbox: a container engine cannot run under
        # filesystem/namespace/device sandboxing, and punching holes one
        # knob at a time only produces a sequence of runtime failures.
        # llm-orch still runs as the unprivileged configured user in both
        # modes — containers are isolated by Podman's user namespace, not
        # by host root.
      } // (if cfg.enableContainerSupport then {
        # Rootless-Podman compatibility profile: keep only delegation,
        # writable state, and a predictable umask.  Everything a container
        # engine needs (nested user namespaces, SUID helpers, /dev/net/tun,
        # writing runroot under XDG_RUNTIME_DIR, mounting overlayfs) is
        # incompatible with the sandbox knobs below, so they stay at
        # systemd defaults (all permissive).
        Delegate = true; # let Podman manage container cgroups below this service
        UMask = "0077";
        # Volatile runtime tree for Podman.  Podman records the host boot
        # ID in `$tmp_dir/alive` and refuses to start when it differs from
        # the running boot ID ("an unhandled reboot has occurred") until
        # that state is deleted.  On tmpfs it is discarded every boot, so
        # stale boot IDs can no longer accumulate in the persistent state
        # directory.  Created by systemd, owned by the service user, and
        # removed with the unit.
        RuntimeDirectory = "llm-orch";
        # Optional (`-` prefix): written by preStart, carrying the
        # runtime-resolved XDG_RUNTIME_DIR for the main process.
        EnvironmentFile = [ "-/var/lib/llm-orch/runtime.env" ];
      } else {
        # Hardened profile for native-only backends.
        NoNewPrivileges = true;
        CapabilityBoundingSet = "";
        ProtectSystem = "strict";
        ProtectHome = "read-only";
        PrivateTmp = true;
        # Device path wildcards are NOT supported by DeviceAllow (only
        # char-*/block-* group names support globs), and PrivateDevices
        # would hide the nodes from the mount namespace anyway — with
        # DevicePolicy=closed implied, a path rule that matches nothing
        # silently denies all GPU access (CUDA/Vulkan init then falls
        # back to CPU).  Cgroup-level filtering via DevicePolicy=closed
        # plus device groups is the recommended pattern for GPU
        # services.  Groups that don't resolve at unit start (e.g.
        # char-nvidia* on a Vulkan-only host) are skipped harmlessly.
        PrivateDevices = false;
        DevicePolicy = "closed";
        DeviceAllow = [
          "char-drm rw" # Vulkan/ROCm: /dev/dri/* (AMD/Intel; NVIDIA with nvidia-drm)
          "char-kfd rw" # ROCm/HIP: /dev/kfd
          "char-nvidia* rw" # CUDA: /dev/nvidia*, /dev/nvidia-uvm*, /dev/nvidia-caps/*
        ];
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictAddressFamilies = [
          "AF_UNIX" # journald, local IPC
          "AF_INET" # HTTP listener + loopback backends
          "AF_INET6"
        ];
        RestrictRealtime = true;
        LockPersonality = true;
        # Vulkan/CUDA JIT-compile shaders/kernels at runtime.
        MemoryDenyWriteExecute = false;
        UMask = "0077";
      });
    }
    // lib.optionalAttrs cfg.enableContainerSupport {
      # Rootless podman needs writable HOME, a real XDG_RUNTIME_DIR, and a
      # storage.conf + policy.json — a system user has none of these
      # by default and /tmp is read-only under ProtectSystem=strict.
      # XDG_RUNTIME_DIR must be the user's actual runtime dir: podman
      # detects the systemd user session via $XDG_RUNTIME_DIR/bus and
      # silently falls back to --cgroup-manager=cgroupfs otherwise.
      # The service user's uid is auto-assigned (null at eval time) and
      # systemd's %U specifier resolves to root in system units, so the
      # path is resolved at runtime: preStart (which runs as the service
      # user) writes it into an environment file the unit loads.
      environment = {
        HOME = "/var/lib/llm-orch";
        # General scratch (which can be large, e.g. image tars) stays on
        # disk; only Podman's engine tmp_dir — the one that holds the boot
        # ID — is redirected to tmpfs via containers.conf below.
        TMPDIR = "/var/lib/llm-orch/tmp";
        CONTAINERS_STORAGE_CONF = "/var/lib/llm-orch/storage.conf";
      };

      preStart = ''
            # Resolve the service user's runtime dir at start time (the uid
            # is auto-assigned, so it cannot be hard-coded at eval time).
            # Linger keeps /run/user/<uid> present; export it for the podman
            # commands below and persist it for the main process, which reads
            # it via EnvironmentFile.
            export XDG_RUNTIME_DIR="/run/user/$(id -u)"
            printf 'XDG_RUNTIME_DIR=%s\n' "$XDG_RUNTIME_DIR" > /var/lib/llm-orch/runtime.env

            mkdir -p /var/lib/llm-orch/tmp \
                     /var/lib/llm-orch/storage/graphroot \
                     /run/llm-orch/tmp \
                     /run/llm-orch/storage/runroot \
                     /var/lib/llm-orch/.config/containers

            # Remove runroot/tmp_dir left by older revisions that kept them
            # under the persistent state directory; a stale boot ID there
            # would otherwise be picked up again.  Image layers (graphroot)
            # are untouched.
            rm -rf /var/lib/llm-orch/storage/runroot \
                   /var/lib/llm-orch/runtime/libpod

            # runroot and the engine tmp_dir hold only per-boot runtime
            # state (lock files, the boot-ID `alive` file) and belong on
            # tmpfs.  Podman compares the recorded boot ID against the
            # running one on every start and aborts if an unclean reboot
            # left them behind; tmpfs discards them at boot.
            cat > /var/lib/llm-orch/storage.conf <<'STORAGE_EOF'
        [storage]
        driver = "overlay"
        graphroot = "/var/lib/llm-orch/storage/graphroot"
        runroot = "/run/llm-orch/storage/runroot"
        STORAGE_EOF

            cat > /var/lib/llm-orch/.config/containers/containers.conf <<'CONTAINERS_EOF'
        [engine]
        tmp_dir = "/run/llm-orch/tmp"
        CONTAINERS_EOF

            cat > /var/lib/llm-orch/.config/containers/policy.json <<'POLICY_EOF'
        {"default":[{"type":"insecureAcceptAnything"}]}
        POLICY_EOF

            # Reconcile any persisted rootless storage with the configured
            # subordinate UID/GID mappings before inspecting or loading images.
            podman system migrate
      ''
      + lib.optionalString (cfg.containerImages != [ ]) (
        lib.concatMapStringsSep "\n" (img: ''
          if podman image exists ${lib.escapeShellArg img.image}; then
            echo "llm-orch: container image ${img.image} already loaded"
          elif [ -f ${lib.escapeShellArg img.path} ]; then
            echo "llm-orch: loading container image from ${img.path} ..."
            podman load -i ${lib.escapeShellArg img.path}
          else
            echo "llm-orch: WARNING - tar file ${img.path} not found, skipping image ${img.image}" >&2
          fi
        '') cfg.containerImages
      );
    };
  };
}
