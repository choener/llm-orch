// ── HTTP Handlers ─────────────────────────────────────────────────────────────
//
// All OpenAI-compatible and admin endpoints live here.
//
// # Solving the axum `Send` bound issue
//
// axum 0.8 requires every handler's future to be `Send`, because it runs
// handlers on a multi-threaded tokio runtime.  `tokio::sync::RwLockReadGuard`
// is `!Send` — it must be unlocked on the same task that acquired it.
//
// **The fix**: extract all data behind locks, *clone* it, and *drop the guard*
// before any `.await` point.  The pattern looks like this:
//
//     // BAD — guard held across `.await` → future is `!Send`
//     let cfg = state.config.read().await;
//     let name = cfg.server.listen.clone();  // guard still alive
//     some_async_fn(&name).await;            // ← `!Send` compile error
//
//     // GOOD — guard dropped before `.await`
//     let name = {
//         let cfg = state.config.read().await;
//         cfg.server.listen.clone()
//     };  // guard dropped here
//     some_async_fn(&name).await;  // future is `Send`
//
// Every handler in this module follows this pattern.  If you add a new handler,
// check that no `tokio::sync::RwLockReadGuard` or `RwLockWriteGuard` crosses
// an `.await` boundary.

use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures_util::stream::Stream;
use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};
use tracing::{debug, info, warn};

use crate::config::AliasConfig;
use crate::instance::InstanceHandle;
use crate::server::{ApiKey, ApiError, AppState};
use crate::types::*;
use tracing::Instrument;

// ── /v1/models ───────────────────────────────────────────────────────────────

/// `GET /v1/models` — list available models and aliases.
///
/// # Send safety
/// The `config` read guard is dropped before we build the response — no
/// `.await` while the guard is alive.
pub async fn list_models(
    State(state): State<AppState>,
    _key: ApiKey,
) -> Result<Json<ModelsResponse>, ApiError> {
    let span = tracing::info_span!("list_models", id = %uuid::Uuid::new_v4().to_string());
    async {
    // ── 1. Read config, clone what we need, drop the guard. ──────────────
    let (models, aliases) = {
        let cfg = state.config.read().await;
        (
            cfg.models.clone(),
            cfg.aliases.clone(),
        )
    }; // cfg read guard dropped here — future is `Send`

    // ── 2. Build response (pure computation, no locks held). ────────────
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let mut data: Vec<ModelEntry> = models
        .iter()
        .map(|m| ModelEntry {
            id: m.name.clone(),
            object: "model".into(),
            created: now,
            owned_by: "llm-orch".into(),
            context_length: Some(m.context_length),
        })
        .collect();

    // Aliases also appear as models.
    for a in &aliases {
        let ctx = models
            .iter()
            .find(|m| m.name == a.target)
            .map(|m| m.context_length);
        data.push(ModelEntry {
            id: a.name.clone(),
            object: "model".into(),
            created: now,
            owned_by: "llm-orch".into(),
            context_length: ctx,
        });
    }

    Ok(Json(ModelsResponse {
        object: "list".into(),
        data,
    }))
    }.instrument(span).await
}

// ── /v1/info ─────────────────────────────────────────────────────────────────

/// `GET /v1/info` — runtime information about loaded models and instances.
///
/// # Send safety
/// Two locks are acquired (`config` and `instances`), but both are dropped
/// before the response is returned — never across `.await`.
pub async fn info_endpoint(
    State(state): State<AppState>,
    _key: ApiKey,
) -> Result<Json<InfoResponse>, ApiError> {
    let span = tracing::info_span!("info", id = %uuid::Uuid::new_v4().to_string());
    async {
    // ── 1. Read config, clone what we need, drop the guard. ──────────────
    let (models, aliases) = {
        let cfg = state.config.read().await;
        (cfg.models.clone(), cfg.aliases.clone())
    }; // cfg read guard dropped

    // ── 2. Read instance state, clone what we need, drop the guard. ─────
    let instance_counts = {
        let instances = state.manager.instance_counts();
        instances
    }; // instances read guard dropped

    // ── 3. Read GPU snapshot. ────────────────────────────────────────────
    let gpu_metrics = {
        let gpu = state.gpu.read().await;
        gpu.clone()
    }; // gpu read guard dropped

    // ── 4. Build response. ──────────────────────────────────────────────
    let model_infos: Vec<ModelInfo> = models
        .iter()
        .map(|m| ModelInfo {
            name: m.name.clone(),
            context_length: m.context_length,
            instance_count: instance_counts.get(&m.name).copied().unwrap_or(0),
            max_instances: m.max_instances,
            queue_depth_used: 0, // TODO: expose from manager
            queue_depth_max: m.queue_depth,
            blocked: state.manager.is_blocked(&m.name),
        })
        .collect();

    let alias_infos: Vec<AliasInfo> = aliases
        .iter()
        .map(|a| AliasInfo {
            name: a.name.clone(),
            target: a.target.clone(),
            has_system_prompt: a.system_prompt.is_some(),
        })
        .collect();

    let gpu_statuses: Vec<GpuStatus> = gpu_metrics
        .iter()
        .map(|g| {
            let vram_util_pct = if g.vram_total_bytes > 0 {
                (g.vram_used_bytes as f64 / g.vram_total_bytes as f64) * 100.0
            } else {
                0.0
            };
            GpuStatus {
                index: g.index,
                pci_slot: g.pci_slot.clone(),
                vram_vendor: g.vram_vendor.clone(),
                vram_used_bytes: g.vram_used_bytes,
                vram_total_bytes: g.vram_total_bytes,
                vram_util_pct,
                temperature_c: g.temperature_c,
                power_w: g.power_w,
                gpu_busy_pct: g.gpu_busy_pct,
                sclk_mhz: g.sclk_mhz,
                mclk_mhz: g.mclk_mhz,
            }
        })
        .collect();

    Ok(Json(InfoResponse {
        models: model_infos,
        aliases: alias_infos,
        gpus: gpu_statuses,
    }))
    }.instrument(span).await
}

// ── /v1/chat/completions ─────────────────────────────────────────────────────

/// `POST /v1/chat/completions` — OpenAI-compatible chat completions.
///
/// This is the main endpoint.  It handles:
/// - Alias resolution (system prompt injection, prompt template)
/// - Model resolution
/// - Instance management (spawn or reuse)
/// - Request forwarding to the backend
/// - SSE streaming or aggregated response
///
/// # Send safety
/// This handler is the most complex case.  The pattern is:
///
/// 1. Read config → clone alias/model data → drop config guard.
/// 2. Call `get_or_spawn()` → get `InstanceHandle` → read port/URL from it
///    → drop the inner lock.
/// 3. Spawn a background task for streaming (the `InstanceHandle` is moved
///    into the stream wrapper so its `Drop` releases the in-flight slot
///    when the stream ends).
/// 4. Return the SSE response.
///
/// **No lock guard crosses an `.await` boundary.**
pub async fn chat_completions(
    State(state): State<AppState>,
    _key: ApiKey,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!(
        "chat_completion",
        id = %request_id,
        model = %request.model,
    );
    async {
    // ── 1. Resolve alias → underlying model. ────────────────────────────
    //    Read config, extract what we need, drop the guard.
    let (model_name, alias_system_prompt, alias_prompt_template) = {
        let cfg = state.config.read().await;
        resolve_alias(&cfg.aliases, &request.model)
    }; // cfg read guard dropped — future is `Send`

    debug!(
        request_model = %request.model,
        resolved_model = %model_name,
        has_system_prompt = alias_system_prompt.is_some(),
        "resolved model"
    );

    // ── 2. Apply alias prompt injection. ────────────────────────────────
    //    Pure computation — no locks involved.
    apply_alias_prompts(&mut request.messages, alias_system_prompt, alias_prompt_template);

    // ── 3. Verify the model exists in the config. ───────────────────────
    {
        let cfg = state.config.read().await;
        let exists = cfg.models.iter().any(|m| m.name == model_name);
        if !exists {
            return Err(ApiError::ModelNotFound(model_name));
        }
    } // cfg read guard dropped

    // ── 4. Acquire an instance (or queue). ──────────────────────────────
    let handle = state
        .manager
        .get_or_spawn(&model_name)
        .await
        .ok_or_else(|| {
            if state.manager.is_blocked(&model_name) {
                ApiError::ModelBlocked(model_name.clone())
            } else {
                ApiError::NoCapacity(format!(
                    "model '{}' is at capacity and the queue is full",
                    model_name
                ))
            }
        })?;

    // ── 5. Read the port from the instance, then drop the inner lock. ───
    let port = {
        let inst = handle.inner().lock().unwrap();
        inst.port
    }; // inner lock dropped — `handle` itself is still alive (Arc)

    let backend_url = format!("http://127.0.0.1:{}/v1/chat/completions", port);

    // ── 6. Forward the request. ─────────────────────────────────────────
    if request.stream {
        // Streaming mode: forward SSE stream from backend to client.
        // The `InstanceHandle` is moved into the stream wrapper so the
        // in-flight slot is released when the stream ends (Drop).
        let instance_id = {
            let inst = handle.inner().lock().unwrap();
            inst.id.clone()
        };
        info!("stream start model={} inst={}", model_name, instance_id);
        let stream = build_sse_stream(
            state.client.clone(),
            backend_url,
            serde_json::to_value(&request).unwrap_or_default(),
            handle,
        )
        .await
        .map_err(|e| ApiError::Internal(format!("backend request failed: {}", e)))?;

        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // Non-streaming mode: aggregate the backend response.
        let t0 = std::time::Instant::now();
        let response_body = forward_request_aggregate(
            &state.client,
            &backend_url,
            &serde_json::to_value(&request).unwrap_or_default(),
        )
        .await?;
        let elapsed = t0.elapsed();

        // Capture instance ID before dropping the handle.
        let instance_id = {
            let inst = handle.inner().lock().unwrap();
            inst.id.clone()
        };
        drop(handle);

        log_completion(&response_body, &model_name, &instance_id, elapsed);

        Ok(Json(response_body).into_response())
    }
    }.instrument(span).await
}

// ── /v1/completions ──────────────────────────────────────────────────────────

/// `POST /v1/completions` — legacy (non-chat) completions endpoint.
///
/// Same routing and alias logic as chat completions, but with a simpler
/// request body (prompt string instead of messages).
///
/// # Send safety
/// Same pattern as `chat_completions` — locks are dropped before `.await`.
pub async fn completions(
    State(state): State<AppState>,
    _key: ApiKey,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!(
        "completion",
        id = %request_id,
        model = %request.model,
    );
    async {
    // ── 1. Resolve alias. ───────────────────────────────────────────────
    let (model_name, _alias_system_prompt, _alias_prompt_template) = {
        let cfg = state.config.read().await;
        resolve_alias(&cfg.aliases, &request.model)
    };

    // ── 2. Verify model exists. ─────────────────────────────────────────
    {
        let cfg = state.config.read().await;
        let exists = cfg.models.iter().any(|m| m.name == model_name);
        if !exists {
            return Err(ApiError::ModelNotFound(model_name));
        }
    }

    // ── 3. Acquire instance. ────────────────────────────────────────────
    let handle = state
        .manager
        .get_or_spawn(&model_name)
        .await
        .ok_or_else(|| {
            if state.manager.is_blocked(&model_name) {
                ApiError::ModelBlocked(model_name.clone())
            } else {
                ApiError::NoCapacity(format!(
                    "model '{}' is at capacity and the queue is full",
                    model_name
                ))
            }
        })?;

    // ── 4. Read port, drop inner lock. ──────────────────────────────────
    let port = {
        let inst = handle.inner().lock().unwrap();
        inst.port
    };

    let backend_url = format!("http://127.0.0.1:{}/v1/completions", port);

    // ── 5. Forward request. ─────────────────────────────────────────────
    if request.stream {
        let instance_id = {
            let inst = handle.inner().lock().unwrap();
            inst.id.clone()
        };
        info!("stream start model={} inst={}", model_name, instance_id);
        let stream = build_sse_stream(
            state.client.clone(),
            backend_url,
            serde_json::to_value(&request).unwrap_or_default(),
            handle,
        )
        .await
        .map_err(|e| ApiError::Internal(format!("backend request failed: {}", e)))?;

        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        let t0 = std::time::Instant::now();
        let response_body = forward_request_aggregate(
            &state.client,
            &backend_url,
            &serde_json::to_value(&request).unwrap_or_default(),
        )
        .await?;
        let elapsed = t0.elapsed();

        let instance_id = {
            let inst = handle.inner().lock().unwrap();
            inst.id.clone()
        };
        drop(handle);

        log_completion(&response_body, &model_name, &instance_id, elapsed);

        Ok(Json(response_body).into_response())
    }
    }.instrument(span).await
}

// ── Admin endpoints ──────────────────────────────────────────────────────────

/// `GET /admin/status` — detailed runtime status for all models.
///
/// # Send safety
/// Config read guard is cloned and dropped before building the response.
pub async fn admin_status(
    State(state): State<AppState>,
    _key: ApiKey,
) -> Result<Json<InfoResponse>, ApiError> {
    // Reuse /v1/info logic.
    info_endpoint(State(state), _key).await
}

/// `POST /admin/load` — force-load a model (pre-spawn an instance).
///
/// # Send safety
/// Config read guard is dropped before the async `get_or_spawn` call.
pub async fn admin_load(
    State(state): State<AppState>,
    _key: ApiKey,
    Json(body): Json<AdminModelAction>,
) -> Result<Json<AdminResponse>, ApiError> {
    let span = tracing::info_span!("admin_load", id = %uuid::Uuid::new_v4().to_string(), model = %body.model);
    async {
    // Verify the model exists.
    {
        let cfg = state.config.read().await;
        if !cfg.models.iter().any(|m| m.name == body.model) {
            return Err(ApiError::ModelNotFound(body.model));
        }
    }

    // Spawn an instance (or get an existing one).
    let _handle = state
        .manager
        .get_or_spawn(&body.model)
        .await
        .ok_or_else(|| {
            if state.manager.is_blocked(&body.model) {
                ApiError::ModelBlocked(body.model.clone())
            } else {
                ApiError::NoCapacity(format!(
                    "cannot load model '{}': instance cap reached and queue is full",
                    body.model
                ))
            }
        })?;

    // Drop the handle immediately — we just wanted to ensure an instance exists.
    // The in-flight counter is decremented on drop.
    drop(_handle);

    Ok(Json(AdminResponse {
        status: "ok".into(),
        message: format!("model '{}' loaded (instance spawned or already running)", body.model),
    }))
    }.instrument(span).await
}

/// `POST /admin/unload` — force-unload all instances of a model.
///
/// # Send safety
/// Config read guard is dropped before the async unload call.
pub async fn admin_unload(
    State(state): State<AppState>,
    _key: ApiKey,
    Json(body): Json<AdminModelAction>,
) -> Result<Json<AdminResponse>, ApiError> {
    let span = tracing::info_span!("admin_unload", id = %uuid::Uuid::new_v4().to_string(), model = %body.model);
    async {
    // Verify the model exists.
    {
        let cfg = state.config.read().await;
        if !cfg.models.iter().any(|m| m.name == body.model) {
            return Err(ApiError::ModelNotFound(body.model));
        }
    }

    state.manager.unload_model(&body.model).await;

    Ok(Json(AdminResponse {
        status: "ok".into(),
        message: format!("all instances of model '{}' unloaded", body.model),
    }))
    }.instrument(span).await
}

/// `POST /admin/unblock` — clear the crash-block on a model.
///
/// # Send safety
/// No locks cross `.await` — `unblock_model` is synchronous on `std::sync::RwLock`.
pub async fn admin_unblock(
    State(state): State<AppState>,
    _key: ApiKey,
    Json(body): Json<AdminModelAction>,
) -> Result<Json<AdminResponse>, ApiError> {
    let span = tracing::info_span!("admin_unblock", id = %uuid::Uuid::new_v4().to_string(), model = %body.model);
    async {
    // Verify the model exists.
    {
        let cfg = state.config.read().await;
        if !cfg.models.iter().any(|m| m.name == body.model) {
            return Err(ApiError::ModelNotFound(body.model));
        }
    }

    if !state.manager.is_blocked(&body.model) {
        return Ok(Json(AdminResponse {
            status: "ok".into(),
            message: format!("model '{}' is not blocked", body.model),
        }));
    }

    state.manager.unblock_model(&body.model);

    info!(model = %body.model, "model unblocked via admin");
    Ok(Json(AdminResponse {
        status: "ok".into(),
        message: format!("model '{}' unblocked", body.model),
    }))
    }.instrument(span).await
}

// ── Helpers (lock-free) ──────────────────────────────────────────────────────

/// Resolve an alias name to its underlying model.
///
/// Returns `(model_name, optional_system_prompt, optional_prompt_template)`.
/// If the name is not an alias, returns the name unchanged with `None` prompts.
///
/// This function is pure — it doesn't touch any locks.  The caller is
/// responsible for passing in the alias list (already cloned from config).
fn resolve_alias(
    aliases: &[AliasConfig],
    requested_model: &str,
) -> (String, Option<String>, Option<String>) {
    // Check if the requested model is an alias.
    if let Some(alias) = aliases.iter().find(|a| a.name == requested_model) {
        return (
            alias.target.clone(),
            alias.system_prompt.clone(),
            alias.prompt_template.clone(),
        );
    }
    // Not an alias — use as-is.
    (requested_model.to_owned(), None, None)
}

/// Apply alias system prompt and/or prompt template injection to messages.
///
/// If `system_prompt` is `Some(...)`, it is prepended as a system message
/// (unless the first message is already a system message with content).
///
/// If `prompt_template` is `Some(...)`, it is applied to the last user message.
///
/// This function is pure — no locks, no `.await`.
fn apply_alias_prompts(
    messages: &mut Vec<ChatMessage>,
    system_prompt: Option<String>,
    prompt_template: Option<String>,
) {
    // ── System prompt injection ────────────────────────────────────────
    if let Some(sp) = system_prompt {
        // Only inject if the first message isn't already a system message.
        let needs_injection = match messages.first() {
            Some(m) if m.role == "system" => false,
            _ => true,
        };
        if needs_injection {
            messages.insert(
                0,
                ChatMessage {
                    role: "system".into(),
                    content: Some(MessageContent::Text(sp)),
                    name: None,
                },
            );
        }
    }

    // ── Prompt template injection ──────────────────────────────────────
    //    A simple template substitution: `{prompt}` in the template is
    //    replaced with the last user message's content.
    if let Some(tmpl) = prompt_template {
        if let Some(last_user) = messages.iter_mut().rev().find(|m| m.role == "user") {
            if let Some(MessageContent::Text(ref text)) = last_user.content {
                let rendered = tmpl.replace("{prompt}", text);
                last_user.content = Some(MessageContent::Text(rendered));
            }
        }
    }
}

// ── Completion logging ───────────────────────────────────────────────────────

/// Extract timing and usage from the backend response JSON and emit an
/// `info!` log line suitable for request auditing.
fn log_completion(
    resp: &serde_json::Value,
    model_name: &str,
    instance_id: &str,
    elapsed: std::time::Duration,
) {
    let prompt_tokens = resp
        .pointer("/usage/prompt_tokens")
        .and_then(|v| v.as_u64());
    let gen_tokens = resp
        .pointer("/usage/completion_tokens")
        .and_then(|v| v.as_u64());
    let cached_tokens = resp
        .pointer("/usage/prompt_tokens_details/cached_tokens")
        .and_then(|v| v.as_u64());

    let prompt_per_second = resp
        .pointer("/timings/prompt_per_second")
        .and_then(|v| v.as_f64());
    let predicted_per_second = resp
        .pointer("/timings/predicted_per_second")
        .and_then(|v| v.as_f64());

    let req_id = resp
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("-");

    let server_ms = elapsed.as_millis();

    // Build a compact log line.
    let prompt_cached = match (prompt_tokens, cached_tokens) {
        (Some(p), Some(c)) if c > 0 => format!("prompt={}(cached:{})", p, c),
        (Some(p), _) => format!("prompt={}", p),
        _ => String::new(),
    };
    let gen_part = gen_tokens
        .map(|g| format!(" generated={}", g))
        .unwrap_or_default();
    let speed = match (prompt_per_second, predicted_per_second) {
        (Some(pp), Some(tg)) => format!(" pp={:.0}t/s tg={:.0}t/s", pp, tg),
        _ => String::new(),
    };

    info!(
        "id={} model={} inst={}{}{}{} server_ms={}",
        req_id, model_name, instance_id, prompt_cached, gen_part, speed, server_ms
    );
}

// ── Backend forwarding ───────────────────────────────────────────────────────

/// Forward a request to a backend and aggregate the full response.
///
/// Used for non-streaming mode.  Returns the raw JSON body from the backend.
///
/// # Send safety
/// This function takes `&Client` and `&Value` by reference — it owns no
/// lock guards, so its returned future is `Send`.
async fn forward_request_aggregate(
    client: &reqwest::Client,
    backend_url: &str,
    body: &serde_json::Value,
) -> Result<serde_json::Value, ApiError> {
    let resp = client
        .post(backend_url)
        .json(body)
        .send()
        .await
        .map_err(|e| ApiError::Internal(format!("backend request failed: {}", e)))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp
            .text()
            .await
            .unwrap_or_else(|_| "unknown error".into());
        return Err(ApiError::Internal(format!(
            "backend returned {}: {}",
            status, text
        )));
    }

    resp.json::<serde_json::Value>()
        .await
        .map_err(|e| ApiError::Internal(format!("failed to parse backend response: {}", e)))
}

/// Build an SSE stream that forwards chunks from the backend to the client.
///
/// Returns a `Stream<Item = Result<Event, Infallible>>` suitable for axum's
/// `Sse` response type.
///
/// The `InstanceHandle` is moved into the returned stream so the in-flight
/// slot is released when the stream ends (via `Drop`).
///
/// # Send safety
/// The backend response is initiated *before* returning the stream.  The
/// response headers are captured, and the stream body is forwarded chunk
/// by chunk.  No lock guards are held across `.await` boundaries inside
/// the stream's `poll_next`.
async fn build_sse_stream(
    client: reqwest::Client,
    backend_url: String,
    body: serde_json::Value,
    handle: InstanceHandle,
) -> Result<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>, reqwest::Error> {
    let resp = client
        .post(&backend_url)
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        // Non-success before streaming — return an error event stream.
        let error_text = resp.text().await.unwrap_or_else(|_| "unknown error".into());
        warn!(backend_status = %status, error = %error_text, "backend returned non-success");
        // Return a stream that emits a single error event and then ends.
        let stream = futures_util::stream::once(async move {
            let error_sse = serde_json::json!({
                "error": {
                    "message": format!("backend error: {}", error_text),
                    "type": "backend_error",
                    "code": status.as_u16(),
                }
            });
            Ok(Event::default()
                .data(error_sse.to_string())
                .event("error"))
        });
        // Keep `handle` alive until the stream is consumed (just one event).
        let stream = StreamWithHandle::new(Box::pin(stream), handle);
        return Ok(Box::pin(stream));
    }

    // Success — forward the byte stream as SSE events.
    let byte_stream = resp.bytes_stream();
    let sse_stream = SseForwarder::new(byte_stream, handle);

    Ok(Box::pin(sse_stream))
}

// ── Stream wrappers ──────────────────────────────────────────────────────────

/// A wrapper around a byte stream that converts backend SSE chunks into
/// axum `Event` objects.
///
/// Holds an `InstanceHandle` so the in-flight slot is released when the
/// stream is dropped (end of response or client disconnect).
struct SseForwarder<S> {
    inner: S,
    /// Accumulated partial line from previous chunks.
    buffer: Vec<u8>,
    /// Kept alive until the stream ends.
    _handle: InstanceHandle,
}

impl<S> SseForwarder<S>
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    fn new(stream: S, handle: InstanceHandle) -> Self {
        Self {
            inner: stream,
            buffer: Vec::new(),
            _handle: handle,
        }
    }
}

impl<S> Stream for SseForwarder<S>
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // Try to extract a complete event from the buffer.
            if let Some(event) = extract_sse_event(&mut self.buffer) {
                return Poll::Ready(Some(Ok(event)));
            }

            // Need more data — poll the inner stream.
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buffer.extend_from_slice(&chunk);
                    // Loop back to try extracting an event.
                }
                Poll::Ready(Some(Err(e))) => {
                    warn!(error = %e, "backend stream error");
                    // Emit an error event and end.
                    let error_event = Event::default()
                        .data(format!("{{\"error\":\"{}\"}}", e))
                        .event("error");
                    return Poll::Ready(Some(Ok(error_event)));
                }
                Poll::Ready(None) => {
                    // Stream ended — flush any remaining buffer.
                    if !self.buffer.is_empty() {
                        let remaining = String::from_utf8_lossy(&self.buffer).to_string();
                        self.buffer.clear();
                        if !remaining.trim().is_empty() {
                            return Poll::Ready(Some(Ok(Event::default().data(remaining))));
                        }
                    }
                    // Send [DONE] marker.
                    return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Extract a complete SSE event from a byte buffer.
///
/// SSE events are terminated by a double newline (`\n\n` or `\r\n\r\n`).
/// Returns the extracted event and removes it from the buffer.
fn extract_sse_event(buffer: &mut Vec<u8>) -> Option<Event> {
    // Look for a double newline.
    let double_nl = find_double_newline(buffer)?;
    let event_bytes: Vec<u8> = buffer.drain(..double_nl).collect();
    // Also drain the double newline itself.
    let sep_len = detect_separator_len(buffer);
    buffer.drain(..sep_len);

    let event_str = String::from_utf8_lossy(&event_bytes);

    // Parse the SSE fields.
    let mut data_lines: Vec<String> = Vec::new();
    let mut event_type: Option<String> = None;

    for line in event_str.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("event:") {
            event_type = Some(rest.trim().to_string());
        }
        // Ignore other fields (id:, retry:, comments).
    }

    if data_lines.is_empty() {
        return None;
    }

    let data = data_lines.join("\n");
    let mut event = Event::default().data(data);
    if let Some(et) = event_type {
        event = event.event(et);
    }
    Some(event)
}

/// Find the position of a double newline (`\n\n` or `\r\n\r\n`) in the buffer.
fn find_double_newline(buf: &[u8]) -> Option<usize> {
    for i in 0..buf.len().saturating_sub(1) {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some(i);
        }
        if i + 3 < buf.len()
            && buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some(i);
        }
    }
    None
}

/// Detect the length of the separator at the start of the buffer.
fn detect_separator_len(buf: &[u8]) -> usize {
    if buf.starts_with(b"\r\n\r\n") {
        4
    } else if buf.starts_with(b"\n\n") {
        2
    } else {
        0
    }
}

/// A wrapper that keeps an `InstanceHandle` alive alongside an arbitrary stream.
///
/// Used when we need to return an error stream that still holds the handle
/// (so the in-flight slot is released after the single error event is consumed).
struct StreamWithHandle<S> {
    inner: S,
    _handle: InstanceHandle,
}

impl<S> StreamWithHandle<S> {
    fn new(inner: S, handle: InstanceHandle) -> Self {
        Self {
            inner,
            _handle: handle,
        }
    }
}

impl<S> Stream for StreamWithHandle<S>
where
    S: Stream<Item = Result<Event, Infallible>> + Unpin,
{
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}
