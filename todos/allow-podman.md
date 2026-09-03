# TODO

## Goal
Replace incompatible bespoke hardening with a coherent, documented rootless-Podman compatibility profile when `enableContainerSupport` is enabled.

## Decision
Container support remains **rootless**: llm-orch runs as `services.llm-orch.user`, not as root, and uses the service user's isolated Podman storage. Podman itself provides the container isolation boundary.

After a series of deployment failures caused by punching holes in the systemd sandbox one knob at a time (`chage` SUID unpack, `/dev/net/tun` for pasta, `/run/user/<uid>/libpod` read-only, systemd cgroup manager detection), the module now **drops the systemd sandbox entirely in container mode** rather than relaxing settings incrementally. A container engine cannot run under filesystem/namespace/device sandboxing; the isolation boundary is Podman's user namespace, and llm-orch still runs as the unprivileged configured user. Native-only mode retains the full hardened profile.

## Container-mode unit profile (relaxed)
When `enableContainerSupport = true`, the unit sets only:
- `Delegate = true` — Podman manages container cgroups below the service cgroup.
- `UMask = "0077"`.
- `EnvironmentFile = [ "-/var/lib/llm-orch/runtime.env" ]` — carries the runtime-resolved `XDG_RUNTIME_DIR` to the main process.
- Everything else (`NoNewPrivileges`, `CapabilityBoundingSet`, `ProtectSystem`, `ProtectHome`, `PrivateTmp`, `PrivateUsers`, `PrivateDevices`, `DevicePolicy`/`DeviceAllow`, `ProtectKernel*`, `ProtectControlGroups`, `RestrictSUIDSGID`, `RestrictAddressFamilies`, `RestrictRealtime`, `LockPersonality`, `MemoryDenyWriteExecute`) is left at systemd defaults (permissive).

Supporting pieces (unchanged, still required):
- Subordinate UID/GID ranges and `linger = true` for the service user.
- `/run/wrappers` + `podman` on the unit `path` (setuid `newuidmap`/`newgidmap`).
- `preStart` resolves `XDG_RUNTIME_DIR=/run/user/$(id -u)` at runtime (the service uid is auto-assigned and `%U` resolves to root in system units), exports it for `podman system migrate`/`podman load`, and persists it to `runtime.env` for the main process.
- `storage.conf` + `policy.json` under `/var/lib/llm-orch`; storage migration before every start.

## Native-mode unit profile (hardened, unchanged)
`NoNewPrivileges=true`, `CapabilityBoundingSet=""`, `ProtectSystem=strict`, `ProtectHome=read-only`, `PrivateTmp=true`, `PrivateDevices=false` + `DevicePolicy=closed` with GPU `DeviceAllow` groups (`char-drm`, `char-kfd`, `char-nvidia*`), `ProtectKernelTunables/Modules/ControlGroups=true`, `RestrictSUIDSGID=true`, `RestrictAddressFamilies=AF_UNIX/AF_INET/AF_INET6`, `RestrictRealtime=true`, `LockPersonality=true`, `MemoryDenyWriteExecute=false`, `UMask=0077`. No `Delegate`, no `EnvironmentFile`.

## Structural note (root cause of repeated eval failures)
`//` in Nix is a **shallow** attrset merge. The module builds `serviceConfig` by merging the base set with the mode-specific set via `} // (if cfg.enableContainerSupport then { ... } else { ... })` **inside** the `serviceConfig` value. The outer `// lib.optionalAttrs cfg.enableContainerSupport { environment = ...; preStart = ...; }` only touches `environment`/`preStart` and does not clobber `serviceConfig`. Earlier attempts that put mode-specific `serviceConfig` keys in the outer `optionalAttrs` block silently replaced the entire base `serviceConfig` (wiping `User`, `ExecStart`, hardening) — this was the cause of every `attribute 'llm-orch' missing` / wrong-rendered-unit failure.

## Tasks
- [x] Rootless Podman compatibility profile (no unit-level capability bound, SUID/SGID metadata, cgroup delegation).
- [x] Document the security boundary and trade-off (module option + README).
- [x] `/dev/net/tun` access for pasta (superseded: sandbox dropped, no `DevicePolicy=closed`).
- [x] Real `XDG_RUNTIME_DIR` via runtime resolution + `EnvironmentFile` (supersedes bogus `/var/lib/llm-orch/runtime` and the broken `%U` attempt).
- [x] Drop the systemd sandbox wholesale in container mode instead of per-knob relaxation.
- [x] Verify both modes evaluate with correct mode-specific settings.
- [ ] Deployment verification on cheyenne (see below).
- [ ] Sync README + module option description with the relaxed container-mode profile.

## Deployment verification
After rebuild, inspect the rendered unit:

```sh
systemctl cat llm-orch.service
systemctl show llm-orch.service -p User -p Delegate -p Environment
```

For container mode: `Delegate=yes`, non-root `User`, no hardening lines (`CapabilityBoundingSet`, `NoNewPrivileges`, `ProtectSystem`, `DeviceAllow` all absent), `EnvironmentFile=-/var/lib/llm-orch/runtime.env`, and after start `Environment` contains `XDG_RUNTIME_DIR=/run/user/<uid>` with the real service uid.

Also re-run `nix flake check --no-build --no-write-lock-file` and `cargo test` if the module expression changed since the last run; a shallow-`//` mistake surfaces as `attribute 'llm-orch' missing` or a wrongly rendered unit.

## Notes
- A separate concurrent change is adjusting crash-timeout handling; it is independent of this Podman/systemd work.
