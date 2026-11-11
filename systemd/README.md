# systemd service for plain Linux hosts

This directory runs llm-orch as a hardened systemd service on a plain Linux
host: an Ubuntu or Debian droplet, a plain cloud VM, or a bare-metal box.
It is the counterpart to the NixOS module (`nix/nixos.nix`) and installs the
same hardened profile that module uses for native (non-container) backends.

## Files

- `llm-orch.service` — the unit file. It runs llm-orch as a dedicated system
  user and drops every privilege the daemon does not need (details below).
- `install.sh` — the installer. It creates the service user, installs the
  binary, config, and apikeys file, validates the config, and starts the
  service. Rerunning it is safe and is the upgrade path.

## Quick start

```bash
cargo build --release
cp config.example.yaml config.yaml   # then edit models, devices, listen
./target/release/llm-orch keygen --label me   # paste the sha256: line into apikeys.txt
sudo systemd/install.sh
```

This droplet only proxies a vLLM server already running on
`127.0.0.1:8000`, so use the ready-made remote-only config:

```bash
cp systemd/digitalocean.config.yaml config.yaml
# edit the model name/context length, validate:
./target/release/llm-orch --check-config config.yaml
```

The installer picks up `target/release/llm-orch`, `config.yaml`, and
`apikeys.txt` from the working directory. Pass explicit paths when they live
elsewhere:

```bash
sudo systemd/install.sh --binary target/release/llm-orch \
                        --config /srv/llm-orch/config.yaml \
                        --apikeys /srv/llm-orch/apikeys.txt
```

Verify:

```bash
systemctl status llm-orch
journalctl -u llm-orch -f
curl http://127.0.0.1:8080/health
```

## What the installer does

1. Creates the `llm-orch` system user and group on first run (no login shell,
   home `/var/lib/llm-orch`, not created).
2. Installs the binary to `/usr/local/bin/llm-orch` (0755).
3. Installs the config to `/etc/llm-orch/config.yaml` (0644).
4. Installs the apikeys file to `/etc/llm-orch/apikeys.txt`, owned by the
   service user, mode 0400. It never overwrites an installed apikeys file
   unless a source file is present, so reruns do not clobber live keys.
5. Installs the unit to `/etc/systemd/system/llm-orch.service` and reloads
   systemd.
6. Validates the deployment with `llm-orch --check-config` (including the
   apikeys file when present) and refuses to start or restart the service if
   the check fails.
7. Enables the service and starts it (or restarts it, if it was running).
   `--no-start` skips this step.

The state directory `/var/lib/llm-orch` is created by systemd, owned by the
service user, when the service first starts. Relative paths in the config
(the stats file, the debug log) resolve against it.

## Installed layout

| Path | Owner:Mode | Purpose |
| --- | --- | --- |
| `/usr/local/bin/llm-orch` | root:root 0755 | the daemon |
| `/etc/llm-orch/config.yaml` | root:root 0644 | configuration, hot-reloaded |
| `/etc/llm-orch/apikeys.txt` | llm-orch:llm-orch 0400 | API keys, hot-reloaded |
| `/etc/systemd/system/llm-orch.service` | root:root 0644 | the unit |
| `/var/lib/llm-orch/` | llm-orch:llm-orch 0750 | state dir (relative config paths resolve here) |

## Model files

Models live under `/data/vllm` on this host, outside the sandbox's writable
area. That is fine: backends only read model files, and `ProtectSystem=strict`
leaves every readable path readable. Make sure the service user can read them
once, if they are owned by root or an admin user:

```bash
sudo chgrp -R llm-orch /data/vllm
sudo chmod -R g+rX /data/vllm
sudo chmod g+s /data/vllm        # new files inherit the group
```

## Upgrades

Rebuild and rerun the installer:

```bash
cargo build --release
sudo systemd/install.sh
```

The validation step runs before the restart, so a broken config never replaces
a running service. A bare binary swap also works:

```bash
sudo install -m 0755 target/release/llm-orch /usr/local/bin/llm-orch
sudo systemctl restart llm-orch
```

## Hot reload and key rotation

llm-orch watches the config and the apikeys file and reloads both without a
restart. Replace files atomically (write a temporary file in the same
directory, then `mv` it over the target) so the watcher fires reliably. A
config that fails validation is rejected and the old one keeps running.

To mint a key and add it to a live service:

```bash
./target/release/llm-orch keygen --label alice
# print the token, then append the printed sha256: line:
sudo sh -c 'echo "alice: sha256:…" >> /etc/llm-orch/apikeys.txt.new'
sudo chown llm-orch:llm-orch /etc/llm-orch/apikeys.txt.new
sudo chmod 0400 /etc/llm-orch/apikeys.txt.new
sudo mv /etc/llm-orch/apikeys.txt.new /etc/llm-orch/apikeys.txt
```

The apikeys path is fixed by `--apikeyfile` at startup and is never a config
field, so editing `config.yaml` cannot redirect authentication. Convert any
remaining plaintext entries with `llm-orch hash-apikeys --write`.

## Privileges the service drops

The unit runs as a dedicated system user and removes everything the daemon
does not need:

- No capabilities (`CapabilityBoundingSet=`) and no privilege escalation
  (`NoNewPrivileges=`). The default listener `127.0.0.1:8080` is a high port,
  so nothing is lost by dropping `CAP_NET_BIND_SERVICE`.
- Read-only filesystem except the state directory: `ProtectSystem=strict`
  makes the whole hierarchy read-only and the `StateDirectory` whitelist
  opens `/var/lib/llm-orch` for writing. Model files in any readable location
  (for example `/data/vllm`) keep working, since backends only read them.
  `ProtectHome=yes` hides `/home` entirely; nothing on this host lives there.
- Private `/tmp` (`PrivateTmp=`), so backends and the daemon do not share
  temporary files with other services.
- Kernel interfaces are walled off: `ProtectKernelTunables`,
  `ProtectKernelModules`, `ProtectControlGroups`, `RestrictSUIDSGID`,
  `RestrictRealtime`, `LockPersonality`.
- Network sockets are limited to `AF_UNIX`, `AF_INET`, and `AF_INET6`
  (`RestrictAddressFamilies=`): the HTTP listener, the loopback backends,
  and journald.
- Device access is whitelisted with `DevicePolicy=closed` plus
  `char-drm`, `char-kfd`, and `char-nvidia*` for Vulkan, ROCm, and CUDA.
  `PrivateDevices` stays off on purpose: it hides the GPU nodes and CUDA or
  Vulkan then silently falls back to CPU. Rules whose group matches nothing
  on the host (for example `char-nvidia*` on an AMD box) are skipped.
- `MemoryDenyWriteExecute` stays off because NVIDIA and Vulkan JIT-compile
  kernels and shaders into writable, executable memory at runtime.

The restrictions apply to the spawned `llama-server` backends as well, since
they inherit the unit's sandbox.

## Troubleshooting

- **`start-limit-hit`**: a broken config makes llm-orch exit immediately, and
  systemd stops restarting it after three failed starts within five minutes.
  Fix the config, then run
  `sudo systemctl reset-failed llm-orch && sudo systemctl start llm-orch`.
  Validate a deployment before starting it with
  `llm-orch --check-config /etc/llm-orch/config.yaml --apikeyfile /etc/llm-orch/apikeys.txt`.
- **Requests denied after install**: the apikeys file is missing or unreadable.
  The daemon then runs fail-closed and denies every request. Check
  `journalctl -u llm-orch` for the load warning and the file's owner and mode.
- **Requests denied via public address**: the default listener is
  `127.0.0.1:8080`. Point a reverse proxy (nginx, or a cloud load balancer)
  at it; TLS is terminated there. Bind `0.0.0.0` in `config.yaml` only if
  the droplet is otherwise firewalled.
- **Verbose or JSON logs**: `sudo systemctl edit llm-orch` and add
  `Environment=RUST_LOG=debug` (or `Environment=LLM_ORCH_LOG_JSON=1`) under
  `[Service]`.
- **NVIDIA GPUs idle between requests**: the built-in GPU keep-alive is
  AMD-only. Enable driver persistence mode on the host with
  `sudo nvidia-smi -pm 1` so the GPU stays initialized between requests.
