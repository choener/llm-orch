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

## 🛠 Development

- **Tests**: Run `cargo test` to execute integration tests.
- **Logging**: Set `LLM_ORCH_LOG_JSON=1` for structured JSON logs, or use `RUST_LOG=info` to control verbosity.
- **Check Config**: Validate your configuration without starting the server:
  ```bash
  ./target/release/llm-orch --check-config config.yaml
  ```
