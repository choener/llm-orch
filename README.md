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

### Aliases
Aliases allow you to expose the same model under different names with different personas:
```yaml
aliases:
  - name: "coder"
    target: "deepseek-coder-7b"
    system_prompt: "You are an expert software engineer."
```
When a client requests the `coder` model, `llm-orch` routes it to `deepseek-coder-7b` and injects the system prompt.

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

The unit runs with `ProtectSystem=strict`, `ProtectHome=read-only`, `PrivateTmp`, `NoNewPrivileges`, no capabilities, and `PrivateDevices=true` with `DeviceAllow` for `/dev/dri` (Vulkan) and `/dev/nvidia*` (CUDA). `MemoryDenyWriteExecute=false` is set because GPU runtimes JIT-compile at runtime. Model files under `/home` remain readable; tighten permissions yourself if they must stay private.

## 🛠 Development

- **Tests**: Run `cargo test` to execute integration tests.
- **Logging**: Set `LLM_ORCH_LOG_JSON=1` for structured JSON logs, or use `RUST_LOG=info` to control verbosity.
- **Check Config**: Validate your configuration without starting the server:
  ```bash
  ./target/release/llm-orch --check-config config.yaml
  ```
