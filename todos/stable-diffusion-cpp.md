# TODO: Add stable-diffusion.cpp support to llm-orch

## Goal
Integrate [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (`sd-server`) as a first-class backend in llm-orch, supporting image generation, image editing, and video generation through OpenAI-compatible endpoints, native sdcpp API passthrough, and sdapi (WebUI-compatible) passthrough.

## Architecture Overview

```
Client ─→ llm-orch ─→ sd-server instance (HTTP, one per model)
  GET /v1/models           ← llm-orch lists models
  POST /v1/images/generations → llm-orch resolves model → forwards to sd-server
  POST /v1/images/edits    → llm-orch resolves model → forwards to sd-server
  POST /v1/video/generations → llm-orch resolves model → polls sdcpp job → returns
  POST /sdapi/v1/*         → llm-orch routes to default diffusion model instance
  POST /sdcpp/v1/*         → llm-orch routes to default diffusion model instance
```

The sd-server process (per model) listens on its allocated port. llm-orch manages its lifecycle (spawn, idle TTL, GPU pinning, health checks) the same way it manages llama-server processes. The existing `vulkan_devices`/`cuda_devices`/`rocm_devices` namespaces are reused — sd.cpp uses the same `GGML_VK_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` env var conventions.

**Key patterns from audio.cpp that this follows:**
- Per-model `cmd` with `{port}` / `{name}` placeholders — all sd-server CLI flags go here
- `context_length` is meaningless for diffusion models (can be 0 or omitted)
- Model resolution via `model` field in OpenAI-compatible endpoints
- Non-llama programs are not affected by llama.cpp-specific CPU-offload/spec-draft logic (already handled)

---

## Tasks

### 1. `src/types.rs` — Add image/video generation types
- [x] Add `fn default_one() -> u32` helper (used by ImagesGenerationRequest, ImagesEditRequest)
- [x] Add `ImagesGenerationRequest` struct (OpenAI-compatible: `prompt`, `n`, `size`, `output_format`, `output_compression`, optional `model`, plus llm-orch extension fields: `lora`, `negative_prompt`, `sample_params`, `cfg_scale`, `seed`, `width`, `height`, `sd_cpp_extra_args`)
- [x] Add `ImagesGenerationResponse` struct (`created`, `data: Vec<ImageObject>` with `b64_json`)
- [x] Add `ImagesEditRequest` struct (multipart equivalent — `prompt`, `image`, `mask`, `n`, `size`, plus extension fields)
- [x] Add `VideoGenerationRequest` struct (`prompt`, `model`, `video_frames`, `fps`, `width`, `height`, `seed`, `output_format`, plus extension fields)
- [x] Add `VideoGenerationResponse` struct (`created`, `data: Vec<VideoObject>` with `b64_json`, `mime_type`)
- [x] Add `SdcppJobResult` struct for polling native async jobs (`status`, `result.b64_json`, `error`)

### 2. `src/backend.rs` — Add `StableDiffusionCppBackend`
- [x] Add `StableDiffusionCppBackend` struct (implements `Backend` trait)
- [x] `gpu_args()`: sd.cpp doesn't use `--device` like llama.cpp; only the visibility env var via `gpu_env()` is needed → return empty `Vec`
- [x] `gpu_env()`: delegate to `LlamaCppBackend`'s env logic (same `GGML_VK_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`)
- [x] `health_url()`: sd-server exposes `/health` → use default

### 3. `src/config.rs` — Diffusion model routing config
- [x] Add `default_diffusion_model: Option<String>` to `ServerConfig` (fallback model for native passthrough endpoints)
- [x] *(decision: not needed — backend selection by program basename instead)* (tags a model as a diffusion model, enabling image/video endpoint routing; or use simpler approach: any model whose `cmd` program basename starts with `sd-server`)
- [x] Config validation: if `default_diffusion_model` is set, verify it names a configured model

### 4. `src/handlers.rs` — Add image/video request routing primitives
- [x] Add `DiffusionTarget` struct mirroring `AudioTarget` (model_name, port, instance_id, guard, timeout, debug_log_path)
- [x] Add `acquire_diffusion_target()` function — resolves `model` from request body, acquires instance slot, returns `ImageTarget`
- [x] Add `acquire_passthrough_target()` (header override + `default_diffusion_model` fallback) — resolves the `default_diffusion_model` config setting for native passthrough
- [x] Add `forward_diffusion_json()` / `forward_diffusion_post_raw()` — forwards POST to sd-server instance, handles response Content-Type (JSON, image bytes, SSE, video bytes)

### 5. `src/handlers.rs` — OpenAI-compatible `/v1/images/generations`
- [x] Add `images_generations` handler:
  - Parse `ImagesGenerationRequest` from JSON body
  - Extract `model` field → resolve via `acquire_diffusion_target()`
  - Forward to sd-server's `/v1/images/generations`
  - Return OpenAI-compatible response
- [x] Support extension fields: `lora` array, `negative_prompt`, `sample_params`, `seed`, `width`, `height` mapped to sd-server's JSON body fields
- [x] Support `sd_cpp_extra_args` embedded in `prompt` (pass through verbatim) (pass through literally)

### 6. `src/handlers.rs` — OpenAI-compatible `/v1/images/edits`
- [x] Add `images_edits` handler:
  - Parse multipart form data (OpenAI spec: `prompt`, `image`, `mask`, `n`, `size`)
  - Plus JSON extension fields via a `model` field and optional `params` JSON field
  - Forward to sd-server's `/v1/images/edits`
  - Return OpenAI-compatible response

### 7. `src/handlers.rs` — OpenAI-compatible `/v1/video/generations` (new endpoint)
- [x] Add `video_generations` handler:
  - Parse `VideoGenerationRequest` (model, prompt, video_frames, fps, width, height, etc.)
  - Acquire instance for the model
  - Forward as JSON `POST /sdcpp/v1/vid_gen` to sd-server
  - Get back `{ id, poll_url }` (202 Accepted)
  - Poll `GET /sdcpp/v1/jobs/{id}` until completed (with timeout)
  - Decode base64 container → return raw bytes with correct Content-Type (`video/webm`, `image/webp`, `video/x-msvideo`)
  - Or: return immediate 202 Accepted with a poll URL for async clients

### 8. `src/handlers.rs` — Native endpoint passthrough (`/sdapi/v1/*`, `/sdcpp/v1/*`)
- [x] Add `passthrough_sdapi` catch-all handler for `/sdapi/v1/*`:
  - Resolve target: use `X-LLM-Model` header if present, else config `default_diffusion_model`
  - Acquire instance → forward request verbatim (path + body + headers)
  - Return backend response verbatim
- [x] Add `passthrough_sdcpp` catch-all handler for `/sdcpp/v1/*`:
  - Same routing logic as sdapi passthrough
  - Forward request verbatim
  - Return backend response verbatim

### 9. `src/server.rs` — Register new routes
- [x] Add routes: `POST /v1/images/generations`, `POST /v1/images/edits`, `POST /v1/video/generations`
- [x] Add catch-all routes: `/sdapi/v1/*path` and `/sdcpp/v1/*path` (axum wildcard routing)
- [x] Routes registered unconditionally; model-resolution failures surface as 4xx (simpler, no startup coupling)

### 10. `src/handlers.rs` — `/v1/models` listing includes diffusion models
- [x] Ensure diffusion models appear in `/v1/models` (they already do — all configured models are listed)
- [x] Add `backend: "sd.cpp"`/`"llama.cpp"` to model info in `/v1/info` endpoint for loaded diffusion instances

### 11. `src/scheduler.rs` — Spawn-time backend selection
- [x] Modify spawn to select backend by program basename:
  - If basename starts with `sd-server` → use `StableDiffusionCppBackend`
  - Otherwise → use `LlamaCppBackend` (current behavior)
- [x] Skip llama.cpp-specific: `enforce_cpu_offload()`/`inject_spec_draft_device()` no-op for `sd-server` (basename check verified)
- [x] Ensure `select_gpus_for_model()` works correctly for sd.cpp models (reuses device namespaces — no changes needed) (it uses the same VRAM accounting; no changes needed since we reuse device namespaces)

### 12. Config example and documentation
- [x] Update `config.example.yaml` with:
  - Example `sd-server` cmd_alias
  - Example diffusion model config (e.g., FLUX.1-dev)
  - Example video model config (e.g., Wan2.1)
  - The `default_diffusion_model` server setting
*skipped — architecture captured in this plan file and inline doc comments* to `docs/001-plan.md` or a new `docs/004-sd-cpp.md`:
  - Passthrough routing semantics
  - Extension fields for OpenAI-compatible endpoints
  - Video generation polling behavior
  - GPU namespace reuse

### 13. Integration tests
- [x] Add stub sd-server endpoints to `llm-orch-stub-backend.rs`:
  - `GET /health`
  - `POST /v1/images/generations` → canned image response
  - `POST /v1/images/edits` → canned response
  - `POST /sdcpp/v1/vid_gen` → canned job creation (202)
  - `GET /sdcpp/v1/jobs/{id}` → completed job (200) or pending (202)
  - `POST /sdapi/v1/txt2img` → canned response
  - `POST /sdcpp/v1/img_gen` → canned job creation
  - `GET /sdcpp/v1/capabilities` → canned capability JSON
- [x] Add integration tests in `tests/integration.rs`:
  - `POST /v1/images/generations` with model resolution → 200 + image response
  - `POST /v1/images/generations` unknown model → 404
  - `POST /v1/video/generations` → video bytes response
  - Passthrough: `POST /sdapi/v1/txt2img` → forwarded correctly
  - Passthrough: `POST /sdcpp/v1/vid_gen` → 202 + job result
  - `default_diffusion_model` fallback routing
  - `X-LLM-Model` header override

---

## Notes

### Design decisions

1. **Backend selection** is by program basename (`sd-server` → `StableDiffusionCppBackend`), not a config enum. Keeps the existing `cmd`-driven config pattern.

2. **GPU pinning** reuses `vulkan_devices`/`cuda_devices`/`rocm_devices` namespaces. sd.cpp uses the same env vars (`GGML_VK_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`). The `--device` flag that llama.cpp needs is not emitted for sd-server (since `StableDiffusionCppBackend::gpu_args()` returns empty vec).

3. **Native endpoint routing** (sdapi/sdcpp) needs a model target since those APIs have no `model` field. Two mechanisms: `X-LLM-Model` request header, or `default_diffusion_model` server config. The server config fallback is the simplest for single-diffusion-model deployments.

4. **Async video polling**: sd-server's `/sdcpp/v1/vid_gen` is async (202 + job ID). For `/v1/video/generations`, llm-orch polls internally and blocks until done (with server-configured timeout). For clients that want native async, the `/sdcpp/v1/vid_gen` passthrough returns the 202 immediately.

5. **Extension fields on OpenAI endpoints**: Known OpenAI fields are parsed explicitly; all extra JSON fields are passed through verbatim to sd-server's body. This means `lora`, `sample_params`, `vae_tiling_params`, `cache_mode`, etc. work transparently.

6. **No `--device` for sd.cpp**: Unlike llama.cpp which needs `--device VulkanN` to prevent row-splitting across all enumerated backends, sd.cpp only uses the visibility env var. The `StableDiffusionCppBackend::gpu_args()` returns empty.

7. **Output format handling**: 
   - `/v1/images/generations`: returns OpenAI JSON envelope with `b64_json`
   - `/v1/images/edits`: same JSON envelope
   - `/v1/video/generations`: llm-orch decodes the base64 container and returns raw bytes with `Content-Type: video/webm` (or `image/webp`, `video/x-msvideo`)
   - Passthrough endpoints: response forwarded verbatim

### Risks

- sd-server's API is under active development and may change between versions
- Video generation is very slow (minutes) — the polling timeout must be configurable and generous
- sd-server is a single-model process; loading multiple diffusion models requires multiple sd-server processes and significant VRAM
- The `--diffusion-model` CLI flag must reference the actual weights file; operators need to handle this in the `cmd` config