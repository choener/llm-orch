# llm-orch

`llm-orch` is a high-performance, single-host LLM orchestrator designed to manage multiple `llama.cpp` (server) instances. It acts as a smart proxy that handles request routing, dynamic resource allocation, and automated scaling to maximize GPU utilization while maintaining low latency.

## 🚀 Key Features

- **Dynamic Instance Management**: Spawns `llama.cpp` backends on demand. Models are loaded when requested and can be unloaded after a period of inactivity (`idle_ttl`).
- **GPU VRAM-Aware Scheduling**: Intelligently selects GPUs based on available VRAM and specified Vulkan, ROCm, or CUDA device pools to prevent oversubscription.
- **Load-Based Autoscaling**: Automatically scales the number of parallel instances for a model based on exponentially-weighted moving average (EMA) load metrics.
- **Hot-Reloading**: Update your `config.yaml` or `apikeys.txt` at runtime; changes are applied immediately without restarting the orchestrator.
- **Request Queueing**: Implements a FIFO queue for requests when all available instances are at capacity, preventing server crashes under heavy load.
- **API Key Authentication**: Simple, file-based API key management with hot-reload support.
- **Multi-GPU Support**: Supports models that span multiple GPUs by coordinating tensor splits across specified devices.
- **Model Aliasing**: Create aliases for models with optional system prompt injections for different use cases.
- **Cluster Peering**: Front further llm-orch instances as one endpoint — peer models under `<peer>/<model>` names, one apikey per peer, unified stats and admin views.

## 🏗 Architecture

`llm-orch` sits between your client (e.g., Open WebUI, a custom app) and your LLM backends:

`Client` $\rightarrow$ `llm-orch (Auth $\rightarrow$ Scheduler $\rightarrow$ Load Balancer)` $\rightarrow$ `llama-server instances`

1. **Auth**: Validates the API key.
2. **Scheduler**: Determines if a ready instance exists or if a new one needs to be spawned.
3. **Load Balancer**: Routes the request to the least-loaded instance.
4. **Backend**: Forwards the request to the corresponding `llama-server` process.

## 📖 Quick Start

### 1. Build
Ensure you have the Rust toolchain installed.
```bash
cargo build --release
```

### 2. Configure
Copy the example configuration files to your working directory:
```bash
cp config.example.yaml config.yaml
cp apikeys.example.txt apikeys.txt
```

Edit `config.yaml` to point to your model files and define your GPU layout. Make sure your `devices.vulkan` / `devices.rocm` mappings match the corresponding output of `llama-server --list-devices`.

### 3. Run
Start the orchestrator:
```bash
./target/release/llm-orch --config config.yaml
```

### 4. Use
The orchestrator provides an OpenAI-compatible API. You can query it using any LLM client:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-alice-6a8f3b2d1c" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## ⚙️ Configuration Deep Dive

### Models
In `config.yaml`, each model definition controls its lifecycle:
- `max_instances`: Limits the total number of parallel processes for this model.
- `max_concurrent`: How many parallel requests a single instance can handle (slots).
- `vram`: Declared VRAM usage in MB used by the scheduler to pick a GPU.
- `idle_ttl`: Seconds to keep the model loaded after the last request.
- `vulkan_devices`, `cuda_devices`, or `rocm_devices`: The pool of backend GPU indices this model is allowed to use. These namespaces are mutually exclusive per model. Two forms are accepted: a flat pool (`[0, 1]`, each instance spans `gpus` devices chosen from it) or explicit placement alternatives (`[[0],[1]]` = single-GPU, either one; `[[0,1]]` = dual-GPU, using both; `gpus` must stay 1).
- **Speculative decoding**: when a model's `cmd` enables draft-model speculation (a draft-model flag like `--spec-draft-model`/`-md`, or `--spec-type draft-simple|draft-eagle3|draft-mtp`, including internal MTP heads), llm-orch pins the draft context to the same device(s) as the instance by appending `--spec-draft-device VulkanN,…` (CUDA/ROCm prefixes for those namespaces) at spawn — llama.cpp would otherwise run it on the process-default backend (not the model's pinned device), which can OOM against the loaded model. Self-speculative `ngram-*` types are never touched. An explicit `--spec-draft-device`/`--device-draft`/`-devd` or `--spec-draft-ngl 0` in `cmd` is always respected.

### Remote models (http/s endpoints)
A model whose `cmd` is an `http://` or `https://` URL is automatically detected as a **remote** model — a pure passthrough to an already-running backend (llama-server, vLLM, …) on another host. No process is spawned and no port or GPU is allocated: the first request health-probes `GET {cmd}/health` (10 s budget) and registers the endpoint as a single passthrough instance.

```yaml
key_files:                       # {name} → file contents, for remote_auth
  deploy_key: /etc/llm-orch/keys/deploy.key
models:
  - name: remote-qwen
    context_length: 32768
    cmd: "https://gpu-host.example.com:8080"   # no {port} — none is allocated
    remote_auth: "Bearer {deploy_key}"          # sent as the Authorization header
    max_concurrent: 8
```

Lifecycle differences vs. local models:

- `autoscale` is a config error (there is nothing to spawn or despawn); device pools, `vram`, and `gpus > 1` are rejected the same way.
- The instance is never idle-evicted and nothing is killed on config reload or shutdown — there is no local process. A down endpoint simply fails its probe; the model is **not** blocked, and the next request re-probes, so the model recovers without operator intervention.
- `remote_auth` placeholders resolve against `key_files` **at request time**: rotating the key file (or hot-reloading the config) applies immediately, no instance replacement needed. Files are watched; contents are re-read on file change, config reload, and startup.
- EMA load/latency metrics work as usual (in-flight + queued request counts).

### Cluster peering (gateway mode)

Two hosts, one endpoint: a master llm-orch can front further llm-orch
instances (peers) and re-export their models. The peers are ordinary
orchestrators — each sees a single client holding one apikey, so user keys
and per-user accounting stay entirely on the master. Full design:
[`docs/006-cluster-peering.md`](docs/006-cluster-peering.md).

```yaml
node_name: cheyenne
peers:
  - name: cortana                    # becomes part of model names
    url: "http://cortana.lan:8080"
    remote_auth: "Bearer {peer_key_cortana}"
```

- **Naming**: peer models are pinned as `cortana/qwen-flash`; a bare
  `qwen-flash` works when the name is unique (local models win bare-name
  collisions). Local models are also addressable as `cheyenne/<model>`.
  Client-side provider prefixes (e.g. `llm/cortana/qwen-flash`) are
  stripped automatically — llm-orch never needs to know them.
- **Discovery**: peer catalogs are fetched lazily (request routing,
  `/v1/models`, before config hot-reloads) behind a TTL — no background
  poller. A vanished model 404s once, then the catalog refreshes.
- **Forwarding**: bodies pass through opaque except the top-level `model`
  field (rewritten to the leaf-side name and back). The master never
  counts forwarded requests in its own stats — `/v1/stats` merges the
  leaves' numbers on demand (node-qualified entries plus bare-name sums
  across duplicated models); `/v1/info` and `/admin/status` merge the same
  way. `/admin/load|unload|unblock` and `/v1/requests/{id}` are proxied to
  the owning leaf.
- **No shim-side autoscaling or admission control**: the per-peer
  `max_concurrent` is a circuit breaker; queueing and autoscaling happen
  on the peers.

### Autoscaling

Per-model load-based autoscaling adjusts the number of parallel instances of a
model to sustained demand. It complements the default lifecycle (spawn on first
request, unload after idle) and is configured per model:

```yaml
server:
  scale_up_value: m1        # EMA window for scale-up (m1 | m5 | m15)
  scale_down_value: m1      # EMA window for scale-down (m1 | m5 | m15)

models:
  - name: "qwen3-32b"
    max_instances: 2        # hard cap on parallel instances
    max_concurrent: 4       # slots per instance (match --parallel in cmd)
    queue_depth: 20         # how many requests may park while waiting
    autoscale:
      enabled: true
      scale_up_at: 7        # spawn when the active-EMA exceeds 7
      scale_down_at: 4      # despawn when the active-EMA drops below 4
      cooldown_secs: 120
```

The `autoscale` block is optional. Omit it (or set `enabled: false`) and the
model follows the plain lifecycle: spawned on demand, surplus idle instances
evicted past the global drain timeout, and no proactive scale-up.

**What the EMA measures.** The scheduler tracks Unix-style load averages per
model - exponentially weighted moving averages of *active requests* =
`in_flight + queued` (`load_m1` / `load_m5` / `load_m15`), updated on every
slot acquire, queue entry, queue assignment, and release (event-driven, no
polling). A request parked in the queue is therefore visible to the
autoscaler the moment it parks. A background task re-evaluates every model
every 30 seconds:

- **Scale up (n -> n+1):** when `EMA(scale_up_value) > scale_up_at +
  (n-1) * max_concurrent` (and `n < max_instances` and a GPU with enough
  free VRAM exists), a new instance is spawned. Since `in_flight` is capped
  at `n * max_concurrent`, this rule can only fire with a non-empty queue:
  at full saturation it means "spawn when the queue EMA exceeds
  `scale_up_at - max_concurrent`". The same gate applies on the request
  path: a request that finds no free slot triggers a spawn when the gate
  passes, and otherwise parks in the queue. Cold start (zero instances)
  always spawns immediately.
- **Scale down (n -> n-1):** when `EMA(scale_down_value) < scale_down_at +
  (n-2) * max_concurrent`, the least-loaded *idle* instance is despawned.
  With `scale_down_at = max_concurrent` (the safe setting) this means the
  remaining instances stay strictly below 100% utilization. In-flight
  requests are never interrupted; busy instances are skipped.
- **Unload (n -> 0):** autoscaling never removes the last instance - that
  happens after `server.drain_idle_timeout_secs` without requests (default
  3600 s), the same path used when autoscaling is disabled.

The hysteresis - a non-empty band `scale_down_at < scale_up_at`, validated
at config load - plus the per-model `cooldown_secs` are what keep the
instance count from flapping on bursts.

**Validation.** `scale_up_at` must be `> max_concurrent` (otherwise a full
instance with an empty queue would trigger scale-up), `scale_down_at` must
be `>= max_concurrent` (the remaining instances must not be left saturated),
and `scale_down_at` must be `< scale_up_at`. Violations are rejected at
config load. Two combinations that make autoscaling *inert* only produce
warnings (visible in `--check-config` and at startup): `autoscale.enabled`
with `max_instances: 1` (scale-up is impossible), and with `queue_depth: 0`
(requests never park, so the scale-up rule can never fire).

**Parameters to watch.**

| Parameter | Default | Meaning |
| --- | --- | --- |
| `autoscale.enabled` | - | Master switch for load-based scaling. |
| `autoscale.scale_up_at` | - (required) | Spawn when the active-EMA exceeds `scale_up_at + (n-1) * max_concurrent` - i.e. when the queue EMA exceeds `scale_up_at - max_concurrent`. Must be `> max_concurrent`. |
| `autoscale.scale_down_at` | - (required) | Despawn when the active-EMA drops below `scale_down_at + (n-2) * max_concurrent`. Must be `>= max_concurrent` and `< scale_up_at`. |
| `autoscale.cooldown_secs` | `120` | Minimum seconds between any two scale actions for the model. |
| `server.scale_up_value` | `m1` | EMA window for the scale-up comparison (`m1` · `m5` · `m15`). |
| `server.scale_down_value` | `m1` | EMA window for the scale-down comparison (`m1` · `m5` · `m15`). |
| `max_instances` | `1` | Hard cap on scale-up (VRAM permitting). |
| `max_concurrent` | `4` | Slots per instance - the per-instance cap on requests committed to the backend. |
| `queue_depth` | `10` | How many requests may park with heartbeats while waiting; 429 when full. |
| `server.queue_wait_timeout_secs` | `3600` | Maximum accepted scheduler wait in seconds; `0` means no scheduler deadline. |
| `server.queue_heartbeat_secs` | `10` | Pre-assignment SSE comment cadence; must be greater than zero. |
| `server.migration_timeout_secs` | `0` | Migration transition deadline; `0` means no transition deadline. |
| `server.eviction_policy` | `lru` | Victim ordering policy; only `lru` is currently implemented. |
| `server.drain_idle_timeout_secs` | `3600` | The only n -> 0 mechanism. |
| `vram` * `gpus` | - | Per-instance VRAM reservation; scale-up is also bounded by free VRAM in the device pool. |

The one that bites in practice is `max_concurrent`: because every threshold
is relative to it, the *useful* parallelism per instance belongs there, not
the theoretical maximum. Worked example - a 27B-class model that fits x2 on
your GPUs:

- `max_concurrent: 4`, one instance, defaults: the scale-up threshold is
  `0.7 * 4 * 1 = 2.8`. Two steady parallel requests give a load of about 2.0
  - below the threshold **forever**, so a second instance is never spawned.
- Set `max_concurrent: 2` (two is all one instance of this model can usefully
  handle): the threshold becomes `0.7 * 2 = 1.4 < 2.0`, and the second
  instance spawns once two parallel requests are sustained.
- Alternatively, lower `scale_up_at` to *strictly* below
  `load / (max_concurrent * instances)`. The comparison is a strict `>`, so
  to scale up at exactly 2.0 concurrent on 4.0 of capacity you need
  `scale_up_at < 0.5` (e.g. `0.49`) - not `0.5`.
- Load must be *sustained*: `load_m5` has a ~5-minute time constant, so a
  burst of a few minutes only moves it partway (after 5 minutes it has
  reached about 63% of a step change). Short spikes queue instead of scaling.
- While the gate refuses to spawn, excess requests queue up to `queue_depth`
  and then start getting 429s - watch `queue_depth_used` to see if that is
  what is happening to you.

**Observing autoscaling.** `GET /v1/info` and `GET /admin/status` (see
`scripts/get-status.sh`) report the decision inputs per model:

```bash
scripts/get-status.sh | jq '.models[] | {name, instance_count, max_instances,
  load_m5, load_m15, queue_depth_used, blocked}'
```

Compare `load_m5` against `scale_up_at * max_concurrent * instance_count` and
`load_m15` against `scale_down_at * max_concurrent * (instance_count - 1)` -
that is exactly the decision the autoscaler makes. With `RUST_LOG=info` you
also get the decision log lines: `autoscale: scaling up (load_m5=..., threshold=...)`,
`autoscale: scaled down (load_m15=..., threshold=...)`, and `drain idle timeout
expired, ...` for the n -> 0 unload. Actions are sparse by design: evaluated
every 30 s and at most one per `cooldown_secs`.

### Long queue waits and streaming heartbeats

Accepted requests may wait for a slot, model load, or GPU migration for up to
`server.queue_wait_timeout_secs` (3600 seconds by default). Set it to `0` to
remove llm-orch's scheduler deadline; a reverse proxy or client can still
impose its own total or idle timeout. `queue_depth` remains the per-model cap
for new waiters, and a full queue returns `429` with `Retry-After`.

For `stream: true`, llm-orch returns the SSE response before assignment and
sends standards-compliant SSE comment heartbeats every
`server.queue_heartbeat_secs` seconds. Comments do not become assistant text
and are ignored by ordinary OpenAI-compatible parsers. After assignment, the
normal backend SSE events and terminal marker are forwarded unchanged. Clients
that opt into `X-LLM-Orch-Wait-Events: 1` may also receive versioned
`llm-orch.status` events while waiting.

A non-streaming JSON response cannot carry interim heartbeat bytes while
remaining valid JSON. Such requests stay open and return one JSON response
when assigned, or a gateway timeout when their scheduler deadline expires.
Use streaming mode or poll the authenticated `GET /v1/requests/{request_id}`
endpoint when a client or proxy needs visible progress during a long wait.
Request statuses are kept in a bounded in-memory registry (up to 4096 entries;
terminal records are retained for 15 minutes) and contain lifecycle metadata
only—never prompts or API keys. Valid config reloads preserve accepted queue
route snapshots; a migration rolls back before handoff if its placement becomes
stale, while a completed victim handoff is not undone.

### Error attribution

Every error message llm-orch generates itself — HTTP error bodies and the
`error` events it injects into streams — is prefixed with `llm-orch: `:

```
llm-orch: backend sent no data for 300s (idle timeout)
llm-orch: no capacity: my-model
```

Responses forwarded verbatim from a backend keep their original body, so the
prefix tells you which hop in the chain (client, proxy, llm-orch, backend)
produced a failure.

### ROCm (AMD) devices
ROCm models use a separate llama.cpp backend namespace from Vulkan. This is
important when `llama-server` was built with both backends: an AMD GPU may be
listed once as `Vulkan0` and once as `ROCm0`, so selecting only
`HIP_VISIBLE_DEVICES` is not enough to force the backend. llm-orch sets both
`HIP_VISIBLE_DEVICES` and llama.cpp's explicit `--device ROCmN` selector.

Configure ROCm devices as an index-to-PCI mapping, just like Vulkan:

```yaml
devices:
  rocm:
    0: "0000:03:00.0"   # find with `llama-server --list-devices`
    1: "0000:04:00.0"
  # Optional limits shared by Vulkan and ROCm device indices.
  vram_limit_mb:
    0: 24576

models:
  - name: "qwen3-32b-rocm"
    context_length: 32768
    vram: 20000          # reserved per occupied GPU
    gpus: 1
    rocm_devices: [0, 1]
    cmd: |
      llama-server
        --host 127.0.0.1
        --port {port}
        --model /models/qwen3-32b-q4_k_m.gguf
        --ctx-size {context_length}
```

For a multi-GPU instance, `rocm_devices: [0, 1]` with `gpus: 2` results in
`HIP_VISIBLE_DEVICES=0,1` and `--device ROCm0,ROCm1`. If the selected pool is
`[2, 4]`, llm-orch uses `HIP_VISIBLE_DEVICES=2,4` and still selects
`ROCm0,ROCm1`, because HIP renumbers the restricted visible devices.

ROCm mappings must point to AMD DRM devices. The same physical AMD GPU may be
listed in both `devices.vulkan` and `devices.rocm`; the model chooses exactly
one backend, while scheduler VRAM accounting is shared by PCI slot. Thus the
GPU can host both workloads when the declared VRAM reservations fit, but it
will not be oversubscribed according to those reservations. A ROCm-enabled
llama.cpp build and the ROCm runtime must be available to the service.

### CUDA (NVIDIA) devices
Device placement works for NVIDIA GPUs with full parity to the Vulkan path:

```yaml
devices:
  cuda:
    0:
      pci: "0000:65:00.0"   # see nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader
      vram_mb: 24576         # optional: capacity cap + fallback when nvidia-smi is absent
    1:
      pci: "0000:c1:00.0"
      vram_mb: 24576

models:
  - name: "qwen3-32b"
    # ...
    vram: 20000             # reserved per occupied GPU
    gpus: 2                 # span two GPUs (CUDA_VISIBLE_DEVICES=<a>,<b>)
    cuda_devices: [0, 1]    # mutually exclusive with Vulkan/ROCm pools
```

- Instances are pinned via `CUDA_VISIBLE_DEVICES` (selection order = emission order, like Vulkan's `GGML_VK_VISIBLE_DEVICES`).
- VRAM capacity comes from periodic `nvidia-smi` queries; `vram_mb` caps that value and serves as the fallback total when `nvidia-smi` is unavailable. A CUDA device with neither metrics nor `vram_mb` is unusable for placement.
- `nvidia-smi` must be on llm-orch's PATH for live metrics. Under the NixOS module: `services.llm-orch.extraPackages = [ config.hardware.nvidia.package ];`.
- You need a CUDA-enabled llama.cpp build (e.g. `pkgs.llama-cpp.override { cudaSupport = true; }` as the module's `llamaPackage`, or an absolute store path in your `cmd`).
- GPU keep-alive is AMD-only and applies to both Vulkan and ROCm instances; NVIDIA GPUs don't need it — enable persistence mode instead (`nvidia-smi -pm 1`).
- GPUs that neither metrics reader can see (e.g. an Intel iGPU on Vulkan — no AMD sysfs, no `nvidia-smi`) become placeable when `devices.vram_limit_mb` declares their size: the limit is then the total capacity and usage is tracked purely from declared `vram` reservations. Without a limit such a device stays unusable for placement.

### Aliases
Aliases allow you to expose the same model under different names with different personas:
```yaml
aliases:
  - name: "coder"
    target: "deepseek-coder-7b"
    system_prompt: "You are an expert software engineer."
```
When a client requests the `coder` model, `llm-orch` routes it to `deepseek-coder-7b` and injects the system prompt.

### Audio models (audio.cpp)
`llm-orch` can front [audio.cpp](https://github.com/0xShug0/audio.cpp) (`audiocpp_server`) for TTS and speech-to-text, exposing the OpenAI-compatible endpoints `POST /v1/audio/speech`, `POST /v1/audio/transcriptions` (JSON and multipart), and `GET /v1/audio/voices`. Audio models are regular models: spawned on demand (one `audiocpp_server` process per model), evicted after `idle_ttl`, queued under load, and authenticated like everything else.

Setup:
1. **Author one `server.json` per audio model** (see `config.example-audiocpp.json`). The model `id` **must equal** the llm-orch model name — audiocpp validates the request's `model` field against it. Recommend `lazy_load: false` so llm-orch's `/health`-based readiness means the model is actually loaded. `host`/`port` in the file are placeholders; llm-orch overrides them at spawn time.
2. **Download model weights with audio.cpp's own tooling** (`tools/model_manager_v2.py` or its WebUI) — llm-orch does not manage audio model files.
3. **Add the `audiocpp` cmd alias and the model** (see `config.example.yaml`):
```yaml
cmd_aliases:
  audiocpp: |
    audiocpp_server
      --config /etc/llm-orch/audio/{name}.json
      --host 127.0.0.1
      --port {port}
      --backend vulkan
models:
  - name: "qwen3-tts"          # == id in /etc/llm-orch/audio/qwen3-tts.json
    max_concurrent: 1          # audiocpp serializes requests per model
    vram: 4000
    idle_ttl: 300
    cmd: "{audiocpp}"
```
`context_length` may be omitted for audio models. Responses are forwarded with the backend's Content-Type intact: WAV bytes, base64 JSON, SSE streams (TTS deltas, transcript deltas), or raw PCM all work; `stream=true` multipart transcriptions stream SSE. The bidirectional `/v1/audio/transcriptions/live` ingest endpoint is not proxied yet.

## ❄️ NixOS Module

The flake exports `nixosModules.llm-orch` (also aliased as `nixosModules.default`), which runs llm-orch as a hardened systemd service:

```nix
# flake.nix (consumer)
{
  inputs.llm-orch.url = "github:you/llm-orch";

  outputs = { self, nixpkgs, llm-orch, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        llm-orch.nixosModules.llm-orch
        {
          services.llm-orch = {
            enable = true;
            configFile = "/var/lib/llm-orch/config.yaml";
            # user = "choener";        # optional: run as an existing user
          };
        }
      ];
    };
  };
}
```

The module automatically uses the flake's own build of llm-orch; override with `services.llm-orch.package` if needed.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `enable` | `false` | Enable the llm-orch service. |
| `package` | this flake's build | The llm-orch package to run. |
| `configFile` | — (required) | Path to `config.yaml`; hot-reloaded at runtime. |
| `user` / `group` | `"llm-orch"` | Account to run under. The default is created as a system user with `video`/`render` group membership for GPU access. Set to your own user if your model files are owned by it. |
| `llamaPackage` | `pkgs.llama-cpp` | Provides `llama-server` on the service PATH. Set to `null` to opt out. |
| `extraPackages` | `[]` | Additional packages on the service PATH (available to model `cmd`s and keep-alive hooks). |
| `enableContainerSupport` | `false` | Enable rootless Podman backends; supplies Podman, the NixOS mapping wrappers, required service hardening, subordinate IDs, and a lingered user session. |
| `containerImages` | `[]` | Images to load from local `podman save` tar archives before starting llm-orch. |

### Container backends (rootless Podman)

Enable container support when a model command begins with `podman run`; llm-orch
then manages the container's lifecycle. The module runs Podman rootlessly as the
configured service user, gives that user subordinate UID/GID ranges, exposes
NixOS's setuid `newuidmap`/`newgidmap` wrappers on the service PATH, and enables
that user's lingering systemd session. To map the subordinate range into
Podman's child namespace, the service explicitly runs with `PrivateUsers=false`
and `NoNewPrivileges=false`. Container mode deliberately leaves
`CapabilityBoundingSet` unset and sets `RestrictSUIDSGID=false`: rootless Podman
needs its normal nested-user-namespace capabilities and must preserve ordinary
image-layer SUID/SGID metadata (for example `/usr/bin/chage`) while unpacking.
The service itself remains the configured non-root user; rootless Podman is the
container isolation boundary. The service waits for `linger-users.service`,
delegates its cgroup subtree to Podman, and runs `podman system migrate` before
it examines or loads images. The migration keeps existing rootless storage
compatible when mappings are added or changed.

```nix
services.llm-orch = {
  enable = true;
  enableContainerSupport = true;
  containerImages = [
    {
      image = "vllm-radiance:0.11.0";
      path = "/data/vllm/container/vllm-radiance.tar";
    }
  ];
};
```

Create an archive with `podman save IMAGE -o archive.tar`; `containerImages`
loads it only when the named image is absent from the service user's storage. A
load failure referring to insufficient UIDs/GIDs while unpacking a container
path such as `/etc/gshadow` signals a rootless user-namespace mapping problem,
not an unsupported image format. This module's container-support setup handles
those mappings and performs the required storage migration before the load.

### File ownership and permissions

The service reads its files directly (hot-reload requires this), so ownership matters:

- **Config file**: owned by the service user, may be world-readable (e.g. `0644 llm-orch:llm-orch`).
- **Apikeys file** (path set via `apikeys_file` in the config): owned by the service user, readable by **no one else** (`0400` or `0600`).
- Relative paths in the config (like `apikeys_file: "apikeys.txt"`) resolve against the state directory `/var/lib/llm-orch`, which the module creates owned by the service user.
- Replace both files **atomically** (write temp file, then `mv`) so the hot-reload watcher fires reliably.

### Hardening notes

By default the unit runs with `ProtectSystem=strict`, `ProtectHome=read-only`, `PrivateTmp`, `NoNewPrivileges`, an empty capability bound, and cgroup-level device filtering (`DevicePolicy=closed` with `DeviceAllow` for the `char-drm` (Vulkan/ROCm), `char-kfd` (ROCm/HIP), and `char-nvidia*` (CUDA) device groups; unresolvable groups are skipped on hosts without the matching hardware). Container mode switches to a rootless-Podman compatibility profile: `NoNewPrivileges=false`, `PrivateUsers=false`, `RestrictSUIDSGID=false`, cgroup delegation, and no unit-level `CapabilityBoundingSet`. These relaxations are necessary for rootless namespace setup and normal image-layer extraction; Podman still runs as the configured non-root service user and supplies the container boundary. `MemoryDenyWriteExecute=false` is set because GPU runtimes JIT-compile at runtime. Model files under `/home` remain readable; tighten permissions yourself if they must stay private.

## 🧩 ROCm Toolchain Overlays

This flake exposes `overlays.rocm` providing modern ROCm for consumers on
NixOS — entirely from AMD's prebuilt binaries, zero compilation:

| attr | source | ISAs |
|---|---|---|
| `pkgs.rocmPackages_10_0` | TheRock 10.0 dist (stable.repo.amd.com) | `gfx120x`, `gfx110x` (`passthru.forIsa`) |
| `pkgs.rocmPackages_7_14` | AMD's official pip SDK wheels (rocm_sdk_core + devel + device, 7.14.1) | `gfx1201` today (`passthru.forGfx`; other gfxes need url+hash added) |

Both expose `rocm-core` (the raw tree), `hipcc` (NixOS-wrapped driver: host
headers/crt flags, gcc-13 libstdc++ for clang-HIP), and `hip-smoke`.
Verify on a machine with a matching GPU:

```bash
nix run .#hip-smoke        # TheRock 10.0
nix run .#hip-smoke-7_14   # ROCm 7.14.1
# -> kernel result 1+2=3 | GPU: AMD Radeon RX 9070 (gfx1201, 16304 MB)
```

See `docs/004-native-vllm-radiance.md` for the full story: building and
running vLLM-radiance (vLLM + gfx1201 kernels) natively on NixOS without
containers.

## 🛠 Development

- **Tests**: Run `cargo test` to execute integration tests.
- **Logging**: Set `LLM_ORCH_LOG_JSON=1` for structured JSON logs, or use `RUST_LOG=info` to control verbosity.
- **Check Config**: Validate your configuration without starting the server:
  ```bash
  ./target/release/llm-orch --check-config config.yaml
  ```
