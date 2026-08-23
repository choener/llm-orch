# llm-orch

`llm-orch` is a high-performance, single-host LLM orchestrator designed to manage multiple `llama.cpp` (server) instances. It acts as a smart proxy that handles request routing, dynamic resource allocation, and automated scaling to maximize GPU utilization while maintaining low latency.

## 🚀 Key Features

- **Dynamic Instance Management**: Spawns `llama.cpp` backends on demand. Models are loaded when requested and can be unloaded after a period of inactivity (`idle_ttl`).
- **GPU VRAM-Aware Scheduling**: Intelligently selects GPUs based on available VRAM and specified Vulkan device pools to prevent oversubscription.
- **Load-Based Autoscaling**: Automatically scales the number of parallel instances for a model based on exponentially-weighted moving average (EMA) load metrics.
- **Hot-Reloading**: Update your `config.yaml` or `apikeys.txt` at runtime; changes are applied immediately without restarting the orchestrator.
- **Request Queueing**: Implements a FIFO queue for requests when all available instances are at capacity, preventing server crashes under heavy load.
- **API Key Authentication**: Simple, file-based API key management with hot-reload support.
- **Multi-GPU Support**: Supports models that span multiple GPUs by coordinating tensor splits across specified devices.
- **Model Aliasing**: Create aliases for models with optional system prompt injections for different use cases.

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

Edit `config.yaml` to point to your model files and define your GPU layout. Make sure your `devices.vulkan` mapping matches the output of `llama-server --list-devices`.

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
- `vulkan_devices`: The pool of GPU indices this model is allowed to use.

### Autoscaling

Per-model load-based autoscaling adjusts the number of parallel instances of a
model to sustained demand. It complements the default lifecycle (spawn on first
request, unload after idle) and is configured per model:

```yaml
models:
  - name: "qwen3-32b"
    max_instances: 2        # hard cap on parallel instances
    max_concurrent: 4       # slots per instance - the capacity denominator
    autoscale:
      enabled: true
      scale_up_at: 0.7
      scale_down_at: 0.4
      cooldown_secs: 120
```

The `autoscale` block is optional. Omit it (or set `enabled: false`) and the
model follows the plain lifecycle: spawned on demand, surplus idle instances
evicted past the global drain timeout, and no proactive scale-up.

**How it works.** The scheduler tracks Unix-style load averages per model -
exponentially weighted moving averages of *concurrent in-flight requests*
(`load_m1` / `load_m5` / `load_m15`), updated on every slot acquire/release
(event-driven, no polling). A background task re-evaluates every model every
30 seconds:

- **Scale up (n -> n+1):** when `load_m5 > scale_up_at * max_concurrent * n`
  (and `n < max_instances` and a GPU with enough free VRAM exists), a new
  instance is spawned. The same gate applies on the request path: a request
  that finds no free slot only triggers a spawn when the gate passes, and
  otherwise queues. Cold start (zero instances) always spawns immediately.
- **Scale down (n -> n-1):** when `load_m15 < scale_down_at *
  max_concurrent * (n - 1)`, the least-loaded *idle* instance is despawned.
  In-flight requests are never interrupted; busy instances are skipped.
- **Unload (n -> 0):** autoscaling never removes the last instance - that
  happens after `server.drain_idle_timeout_secs` without requests (default
  3600 s), the same path used when autoscaling is disabled.

The hysteresis - scale-up on the fast 5-minute EMA and the higher threshold,
scale-down on the slow 15-minute EMA and the lower threshold - plus the
per-model `cooldown_secs` are what keep the instance count from flapping on
bursts.

**Parameters to watch.**

| Parameter | Default | Meaning |
| --- | --- | --- |
| `autoscale.enabled` | - | Master switch for load-based scaling. |
| `autoscale.scale_up_at` | `0.7` | Spawn when `load_m5` exceeds this fraction of `max_concurrent * instances`. |
| `autoscale.scale_down_at` | `0.4` | Despawn when `load_m15` drops below this fraction of `max_concurrent * (instances - 1)`. |
| `autoscale.cooldown_secs` | `120` | Minimum seconds between any two scale actions for the model. |
| `max_instances` | `1` | Hard cap on scale-up (VRAM permitting). |
| `max_concurrent` | `4` | Slots per instance - **the denominator of every threshold**. |
| `queue_depth` | `10` | Where excess requests go while the gate refuses to spawn; 429 when full. |
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
    cuda_devices: [0, 1]    # mutually exclusive with vulkan_devices
```

- Instances are pinned via `CUDA_VISIBLE_DEVICES` (selection order = emission order, like Vulkan's `GGML_VK_VISIBLE_DEVICES`).
- VRAM capacity comes from periodic `nvidia-smi` queries; `vram_mb` caps that value and serves as the fallback total when `nvidia-smi` is unavailable. A CUDA device with neither metrics nor `vram_mb` is unusable for placement.
- `nvidia-smi` must be on llm-orch's PATH for live metrics. Under the NixOS module: `services.llm-orch.extraPackages = [ config.hardware.nvidia.package ];`.
- You need a CUDA-enabled llama.cpp build (e.g. `pkgs.llama-cpp.override { cudaSupport = true; }` as the module's `llamaPackage`, or an absolute store path in your `cmd`).
- GPU keep-alive is AMD-only; NVIDIA GPUs don't need it — enable persistence mode instead (`nvidia-smi -pm 1`).

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

### File ownership and permissions

The service reads its files directly (hot-reload requires this), so ownership matters:

- **Config file**: owned by the service user, may be world-readable (e.g. `0644 llm-orch:llm-orch`).
- **Apikeys file** (path set via `apikeys_file` in the config): owned by the service user, readable by **no one else** (`0400` or `0600`).
- Relative paths in the config (like `apikeys_file: "apikeys.txt"`) resolve against the state directory `/var/lib/llm-orch`, which the module creates owned by the service user.
- Replace both files **atomically** (write temp file, then `mv`) so the hot-reload watcher fires reliably.

### Hardening notes

The unit runs with `ProtectSystem=strict`, `ProtectHome=read-only`, `PrivateTmp`, `NoNewPrivileges`, no capabilities, and cgroup-level device filtering (`DevicePolicy=closed` with `DeviceAllow` for the `char-drm` (Vulkan) and `char-nvidia*` (CUDA) device groups; unresolvable groups are skipped on hosts without the matching hardware). `MemoryDenyWriteExecute=false` is set because GPU runtimes JIT-compile at runtime. Model files under `/home` remain readable; tighten permissions yourself if they must stay private.

## 🛠 Development

- **Tests**: Run `cargo test` to execute integration tests.
- **Logging**: Set `LLM_ORCH_LOG_JSON=1` for structured JSON logs, or use `RUST_LOG=info` to control verbosity.
- **Check Config**: Validate your configuration without starting the server:
  ```bash
  ./target/release/llm-orch --check-config config.yaml
  ```
