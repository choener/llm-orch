# NixOS module for llm-orch — single-host LLM orchestrator.
#
# The flake wires `services.llm-orch.package` to the flake's own build;
# when using this module standalone, set it yourself.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.llm-orch;
  defaultUserGroup = "llm-orch";
in
{
  options.services.llm-orch = {
    enable = lib.mkEnableOption "llm-orch — single-host LLM orchestrator for llama.cpp backends";

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

        The apikeys file referenced by `apikeys_file` inside the config must
        be owned by the service user and readable by no one else (mode 0400
        or 0600). Relative paths inside the config are resolved against the
        state directory, `/var/lib/llm-orch`.

        Both files are hot-reloaded at runtime; replace them atomically
        (write to a temp file, then `mv`) so the watcher fires reliably.
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
    users.users.${cfg.user} = lib.mkIf (cfg.user == defaultUserGroup) {
      isSystemUser = true;
      group = cfg.group;
      extraGroups = [ "video" "render" ];
      description = "llm-orch service user";
    };

    users.groups.${cfg.group} = lib.mkIf (cfg.group == defaultUserGroup) { };

    systemd.services.llm-orch = {
      description = "llm-orch — single-host LLM orchestrator";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      path = lib.optional (cfg.llamaPackage != null) cfg.llamaPackage ++ cfg.extraPackages;

      serviceConfig = {
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${lib.getExe cfg.package} --config ${cfg.configFile}";
        Restart = "on-failure";
        RestartSec = 5;

        # Relative paths in the config (e.g. apikeys_file: "apikeys.txt")
        # resolve against this directory. Owned by the service user.
        StateDirectory = "llm-orch";
        WorkingDirectory = "/var/lib/llm-orch";

        # ── Hardening ─────────────────────────────────────────────────
        # Constrained so that spawning llama.cpp backends and accessing
        # GPUs (Vulkan via /dev/dri, CUDA via /dev/nvidia*) keeps working.
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = "read-only";
        PrivateTmp = true;
        PrivateDevices = true;
        DeviceAllow = [
          "/dev/dri rw"      # Vulkan / AMD / Intel GPUs
          "/dev/nvidia* rw"  # CUDA devices (nvidia0, nvidiactl, nvidia-uvm, ...)
        ];
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictSUIDSGID = true;
        RestrictAddressFamilies = [
          "AF_UNIX"   # journald, local IPC
          "AF_INET"   # HTTP listener + loopback backends
          "AF_INET6"
        ];
        RestrictRealtime = true;
        LockPersonality = true;
        # Vulkan/CUDA JIT-compile shaders/kernels at runtime.
        MemoryDenyWriteExecute = false;
        CapabilityBoundingSet = "";
        UMask = "0077";
      };
    };
  };
}
