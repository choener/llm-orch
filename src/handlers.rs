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
    Json,
    body::Bytes,
    extract::{Query, State},
    http::{HeaderMap, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures_util::stream::Stream;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tracing::{debug, info, warn};

use crate::config::{AliasConfig, AliasPolicy, MakeRoomMode};
use crate::debug_log::{DebugLogEntry, DebugStreamContext, ts_now};
use crate::instance::SlotGuard;
use crate::scheduler::{AcquireError, InstanceManager};
use crate::server::{ApiError, ApiKey, AppState};
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
            (cfg.models.clone(), cfg.aliases.clone())
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
            let ctx = a
                .targets
                .first()
                .and_then(|t| models.iter().find(|m| &m.name == t))
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
    }
    .instrument(span)
    .await
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
        // ── 0. Force-refresh metrics so returned values are never stale. ─
        state.manager.force_refresh();

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

        // ── 4. Read metrics snapshot. ──────────────────────────────────────
        let metrics_snapshot = state.manager.model_metrics_snapshot();

        // ── 4b. Read recent completion records. ────────────────────────────
        let recent_completions = state.manager.recent_completions_snapshot();
        for (name, recs) in &recent_completions {
            debug!(model = %name, count = recs.len(), "recent completions");
        }

        // ── 5. Build response. ──────────────────────────────────────────────
        let model_infos: Vec<ModelInfo> = models
            .iter()
            .map(|m| {
                let mets = metrics_snapshot.get(&m.name);
                ModelInfo {
                    name: m.name.clone(),
                    context_length: m.context_length,
                    instance_count: instance_counts.get(&m.name).copied().unwrap_or(0),
                    max_instances: m.max_instances,
                    queue_depth_used: state.manager.queue_depth(&m.name),
                    queue_depth_max: m.queue_depth,
                    blocked: state.manager.is_blocked(&m.name),
                    load_m1: mets.map(|x| x.load_m1).unwrap_or(0.0),
                    load_m5: mets.map(|x| x.load_m5).unwrap_or(0.0),
                    load_m15: mets.map(|x| x.load_m15).unwrap_or(0.0),
                    req_rate_m1: mets.map(|x| x.req_rate_m1).unwrap_or(0.0),
                    req_rate_m5: mets.map(|x| x.req_rate_m5).unwrap_or(0.0),
                    req_rate_m15: mets.map(|x| x.req_rate_m15).unwrap_or(0.0),
                    completions_total: mets.map(|x| x.completions_total).unwrap_or(0),
                    recent_completions: recent_completions
                        .get(&m.name)
                        .cloned()
                        .unwrap_or_default(),
                }
            })
            .collect();

        let alias_infos: Vec<AliasInfo> = aliases
            .iter()
            .map(|a| AliasInfo {
                name: a.name.clone(),
                targets: a.targets.clone(),
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
    }
    .instrument(span)
    .await
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
    key: ApiKey,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!(
        "chat_completion",
        id = %request_id,
        model = %request.model,
    );
    async {
        // ── 1. Resolve alias → candidate models. ────────────────────────────
        //    Read config, extract what we need, drop the guard.
        //    Phase 1: only the first candidate is routed; Phase 4 switches
        //    all call sites to `acquire_for_candidates`.
        let (candidates, ..) = {
            let cfg = state.config.read().await;
            resolve_alias(&cfg.aliases, &request.model)
        }; // cfg read guard dropped — future is `Send`
        let model_name = candidates.into_iter().next().unwrap_or_default();

        debug!(
            request_model = %request.model,
            resolved_model = %model_name,
            "resolved model"
        );

        // ── 3. Verify the model exists in the config; extract debug_log and
        //    the backend timeouts. ─
        let request_model = request.model.clone();
        let (debug_log_path, idle_timeout, total_timeout): (Option<std::path::PathBuf>, _, _) = {
            let cfg = state.config.read().await;
            let idle = std::time::Duration::from_secs(cfg.server.backend_idle_timeout_secs);
            let total = std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs);
            match cfg.models.iter().find(|m| m.name == model_name) {
                Some(m) => (m.debug_log.clone(), idle, total),
                None => return Err(ApiError::ModelNotFound(model_name)),
            }
        }; // cfg read guard dropped

        // ── 4. Acquire an instance slot (or queue). ─────────────────────────
        let guard = state
            .manager
            .get_or_spawn(&model_name)
            .await
            .map_err(|e| acquire_error(&model_name, e))?;

        // Capture instance ID now that we hold the slot guard.
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        }; // inner lock dropped — `guard` itself is still alive

        // ── 5. Read the port from the instance. ────────────────────────────
        let port = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.port
        }; // inner lock dropped — `guard` itself is still alive

        let backend_url = format!("http://127.0.0.1:{}/v1/chat/completions", port);

        // ── 6. Forward the request. ─────────────────────────────────────────
        // Forward the *resolved* model name to the backend, not the alias —
        // strict backends validate the model field.
        request.model = model_name.clone();
        let request_body = serde_json::to_value(&request).unwrap_or_default();

        // Debug log: request (exact body being forwarded).
        if let Some(ref log_path) = debug_log_path {
            state.debug_loggers.write_line(
                log_path,
                &DebugLogEntry {
                    ts: ts_now(),
                    request_id: request_id.clone(),
                    model: model_name.clone(),
                    alias: Some(request_model.clone()),
                    instance_id: Some(instance_id.clone()),
                    dir: "request".into(),
                    stream: Some(request.stream),
                    body: Some(request_body.clone()),
                    usage: None,
                    duration_ms: None,
                    error: None,
                },
            );
        }

        if request.stream {
            // Streaming mode: forward SSE stream from backend to client.
            // The `SlotGuard` is moved into the stream wrapper so the
            // in-flight slot is released when the stream ends (Drop).
            info!("stream start model={} inst={}", model_name, instance_id);
            let debug_ctx = debug_log_path.map(|path| DebugStreamContext {
                loggers: state.debug_loggers.clone(),
                path,
                request_id: request_id.clone(),
                model_name: model_name.clone(),
                alias: Some(request_model.clone()),
                instance_id: instance_id.clone(),
                t0: std::time::Instant::now(),
            });
            let stream = build_sse_stream(
                state.client.clone(),
                backend_url,
                request_body,
                guard,
                model_name.clone(),
                instance_id.clone(),
                debug_ctx,
                state.manager.clone(),
                key.label.clone(),
                request_id.clone(),
                idle_timeout,
            )
            .await
            .map_err(|e| ApiError::Internal(format!("backend request failed: {}", e)))?;

            Ok(Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response())
        } else {
            // Non-streaming mode: aggregate the backend response.
            let t0 = std::time::Instant::now();
            let response_body = forward_request_aggregate(
                &state.client,
                &backend_url,
                &request_body,
                total_timeout,
            )
            .await?;
            let elapsed = t0.elapsed();

            // Debug log: response.
            if let Some(ref log_path) = debug_log_path {
                let usage = response_body.pointer("/usage").cloned();
                state.debug_loggers.write_line(
                    log_path,
                    &DebugLogEntry {
                        ts: ts_now(),
                        request_id: request_id.clone(),
                        model: model_name.clone(),
                        alias: Some(request_model),
                        instance_id: Some(instance_id.clone()),
                        dir: "response".into(),
                        stream: Some(false),
                        body: Some(response_body.clone()),
                        usage,
                        duration_ms: Some(elapsed.as_millis() as u64),
                        error: None,
                    },
                );
            }

            log_completion(&response_body, &model_name, &instance_id, elapsed);

            // Record in per-model ring buffer for /admin/status.
            {
                let prompt_tokens = response_body
                    .pointer("/usage/prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let gen_tokens = response_body
                    .pointer("/usage/completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let cached_tokens = response_body
                    .pointer("/usage/prompt_tokens_details/cached_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                state.manager.record_completion(
                    &model_name,
                    crate::types::CompletionRecord {
                        ts: crate::debug_log::ts_now(),
                        request_id: request_id.clone(),
                        instance_id: instance_id.clone(),
                        api_user: key.label.clone(),
                        prompt_tokens,
                        generated_tokens: gen_tokens,
                        cached_tokens,
                        duration_ms: elapsed.as_millis() as u64,
                    },
                );
            }

            Ok(Json(response_body).into_response())
        }
    }
    .instrument(span)
    .await
}

// ── /v1/responses ────────────────────────────────────────────────────────────

/// `POST /v1/responses` — OpenAI Responses API (pass-through).
///
/// llama.cpp (≥ b8126) implements `/v1/responses` natively — it converts the
/// request to chat completions internally — so this handler is a thin
/// pass-through over the same machinery as `chat_completions`:
/// - Alias resolution (system prompt → `instructions`, prompt template →
///   last user input)
/// - Model resolution; the *resolved* model name is forwarded, not the alias
/// - Instance management (spawn or reuse)
/// - SSE streaming or aggregated response
///
/// The body is handled as a raw `serde_json::Value` so current and future
/// Responses API fields pass through to the backend untouched.
///
/// # Send safety
/// Same pattern as `chat_completions` — no lock guard crosses an `.await`.
pub async fn responses(
    State(state): State<AppState>,
    key: ApiKey,
    Json(mut request): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let requested_model = request
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::BadRequest("missing or invalid 'model' field".into()))?
        .to_string();
    let stream_requested = request
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let span = tracing::info_span!(
        "responses",
        id = %request_id,
        model = %requested_model,
    );
    async {
        // ── 1. Resolve alias → candidate models. ────────────────────────────
        let (candidates, ..) = {
            let cfg = state.config.read().await;
            resolve_alias(&cfg.aliases, &requested_model)
        }; // cfg read guard dropped — future is `Send`
        let model_name = candidates.into_iter().next().unwrap_or_default();

        debug!(
            request_model = %requested_model,
            resolved_model = %model_name,
            "resolved model"
        );

        // ── 3. Verify the model exists; extract debug_log and timeouts. ─────
        let (debug_log_path, idle_timeout, total_timeout): (Option<std::path::PathBuf>, _, _) = {
            let cfg = state.config.read().await;
            let idle = std::time::Duration::from_secs(cfg.server.backend_idle_timeout_secs);
            let total = std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs);
            match cfg.models.iter().find(|m| m.name == model_name) {
                Some(m) => (m.debug_log.clone(), idle, total),
                None => return Err(ApiError::ModelNotFound(model_name)),
            }
        }; // cfg read guard dropped

        // ── 4. Acquire an instance slot (or queue). ─────────────────────────
        let guard = state
            .manager
            .get_or_spawn(&model_name)
            .await
            .map_err(|e| acquire_error(&model_name, e))?;

        // Capture instance ID now that we hold the slot guard.
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        }; // inner lock dropped — `guard` itself is still alive

        // ── 5. Read the port from the instance. ─────────────────────────────
        let port = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.port
        }; // inner lock dropped

        let backend_url = format!("http://127.0.0.1:{}/v1/responses", port);

        // ── 6. Forward the *resolved* model name to the backend, not the
        //    alias — strict backends validate the model field. ───────────────
        request["model"] = serde_json::Value::String(model_name.clone());

        // Debug log: request (exact body being forwarded).
        if let Some(ref log_path) = debug_log_path {
            state.debug_loggers.write_line(
                log_path,
                &DebugLogEntry {
                    ts: ts_now(),
                    request_id: request_id.clone(),
                    model: model_name.clone(),
                    alias: Some(requested_model.clone()),
                    instance_id: Some(instance_id.clone()),
                    dir: "request".into(),
                    stream: Some(stream_requested),
                    body: Some(request.clone()),
                    usage: None,
                    duration_ms: None,
                    error: None,
                },
            );
        }

        if stream_requested {
            // Streaming mode: forward the backend's Responses SSE events.
            // The `SlotGuard` is moved into the stream wrapper so the
            // in-flight slot is released when the stream ends (Drop).
            info!("stream start model={} inst={}", model_name, instance_id);
            let debug_ctx = debug_log_path.map(|path| DebugStreamContext {
                loggers: state.debug_loggers.clone(),
                path,
                request_id: request_id.clone(),
                model_name: model_name.clone(),
                alias: Some(requested_model.clone()),
                instance_id: instance_id.clone(),
                t0: std::time::Instant::now(),
            });
            let stream = build_sse_stream(
                state.client.clone(),
                backend_url,
                request,
                guard,
                model_name.clone(),
                instance_id.clone(),
                debug_ctx,
                state.manager.clone(),
                key.label.clone(),
                request_id.clone(),
                idle_timeout,
            )
            .await
            .map_err(|e| ApiError::Internal(format!("backend request failed: {}", e)))?;

            Ok(Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response())
        } else {
            // Non-streaming mode: aggregate the backend response.
            let t0 = std::time::Instant::now();
            let response_body =
                forward_request_aggregate(&state.client, &backend_url, &request, total_timeout)
                    .await?;
            let elapsed = t0.elapsed();

            // Debug log: response.
            if let Some(ref log_path) = debug_log_path {
                let usage = response_body.pointer("/usage").cloned();
                state.debug_loggers.write_line(
                    log_path,
                    &DebugLogEntry {
                        ts: ts_now(),
                        request_id: request_id.clone(),
                        model: model_name.clone(),
                        alias: Some(requested_model),
                        instance_id: Some(instance_id.clone()),
                        dir: "response".into(),
                        stream: Some(false),
                        body: Some(response_body.clone()),
                        usage,
                        duration_ms: Some(elapsed.as_millis() as u64),
                        error: None,
                    },
                );
            }

            // Responses API usage shape: input_tokens / output_tokens.
            let prompt_tokens = response_body
                .pointer("/usage/input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let gen_tokens = response_body
                .pointer("/usage/output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cached_tokens = response_body
                .pointer("/usage/input_tokens_details/cached_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            info!(
                "id={} model={} inst={} prompt={} generated={} server_ms={}",
                response_body
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-"),
                model_name,
                instance_id,
                prompt_tokens,
                gen_tokens,
                elapsed.as_millis()
            );

            // Record in per-model ring buffer for /admin/status.
            state.manager.record_completion(
                &model_name,
                crate::types::CompletionRecord {
                    ts: crate::debug_log::ts_now(),
                    request_id: request_id.clone(),
                    instance_id: instance_id.clone(),
                    api_user: key.label.clone(),
                    prompt_tokens,
                    generated_tokens: gen_tokens,
                    cached_tokens,
                    duration_ms: elapsed.as_millis() as u64,
                },
            );

            Ok(Json(response_body).into_response())
        }
    }
    .instrument(span)
    .await
}

// ── /v1/responses/input_tokens ────────────────────────────────────────────────

/// `POST /v1/responses/input_tokens` — Responses API token counting
/// (pass-through, non-streaming only).
///
/// Same routing/alias pattern as `responses`, but aggregated only — the
/// endpoint returns a small JSON object (`{object, input_tokens}`).
pub async fn responses_input_tokens(
    State(state): State<AppState>,
    key: ApiKey,
    Json(mut request): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let requested_model = request
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::BadRequest("missing or invalid 'model' field".into()))?
        .to_string();
    let span = tracing::info_span!(
        "responses_input_tokens",
        id = %request_id,
        model = %requested_model,
    );
    async {
        // ── 1. Resolve alias. ───────────────────────────────────────────────
        let (candidates, ..) = {
            let cfg = state.config.read().await;
            resolve_alias(&cfg.aliases, &requested_model)
        };
        let model_name = candidates.into_iter().next().unwrap_or_default();

        // ── 2. Verify model exists; extract debug_log and the timeout. ──────
        let (debug_log_path, total_timeout): (Option<std::path::PathBuf>, _) = {
            let cfg = state.config.read().await;
            let total = std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs);
            match cfg.models.iter().find(|m| m.name == model_name) {
                Some(m) => (m.debug_log.clone(), total),
                None => return Err(ApiError::ModelNotFound(model_name)),
            }
        };

        // ── 3. Acquire instance slot. ───────────────────────────────────────
        let guard = state
            .manager
            .get_or_spawn(&model_name)
            .await
            .map_err(|e| acquire_error(&model_name, e))?;

        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };

        // ── 4. Read port. ───────────────────────────────────────────────────
        let port = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.port
        };

        let backend_url = format!("http://127.0.0.1:{}/v1/responses/input_tokens", port);

        // Forward the *resolved* model name to the backend, not the alias.
        request["model"] = serde_json::Value::String(model_name.clone());

        // Debug log: request (exact body being forwarded).
        if let Some(ref log_path) = debug_log_path {
            state.debug_loggers.write_line(
                log_path,
                &DebugLogEntry {
                    ts: ts_now(),
                    request_id: request_id.clone(),
                    model: model_name.clone(),
                    alias: Some(requested_model.clone()),
                    instance_id: Some(instance_id.clone()),
                    dir: "request".into(),
                    stream: Some(false),
                    body: Some(request.clone()),
                    usage: None,
                    duration_ms: None,
                    error: None,
                },
            );
        }

        // ── 5. Forward request, aggregate, release. ─────────────────────────
        let t0 = std::time::Instant::now();
        let response_body =
            forward_request_aggregate(&state.client, &backend_url, &request, total_timeout).await?;
        let elapsed = t0.elapsed();

        // Debug log: response.
        if let Some(ref log_path) = debug_log_path {
            state.debug_loggers.write_line(
                log_path,
                &DebugLogEntry {
                    ts: ts_now(),
                    request_id: request_id.clone(),
                    model: model_name.clone(),
                    alias: Some(requested_model),
                    instance_id: Some(instance_id.clone()),
                    dir: "response".into(),
                    stream: Some(false),
                    body: Some(response_body.clone()),
                    usage: None,
                    duration_ms: Some(elapsed.as_millis() as u64),
                    error: None,
                },
            );
        }

        // Record in per-model ring buffer for /admin/status.
        // Token counting carries only input tokens — no generation.
        let input_tokens = response_body
            .pointer("/input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        state.manager.record_completion(
            &model_name,
            crate::types::CompletionRecord {
                ts: crate::debug_log::ts_now(),
                request_id: request_id.clone(),
                instance_id: instance_id.clone(),
                api_user: key.label.clone(),
                prompt_tokens: input_tokens,
                generated_tokens: 0,
                cached_tokens: 0,
                duration_ms: elapsed.as_millis() as u64,
            },
        );

        info!(
            "id={} model={} inst={} input_tokens={} server_ms={}",
            request_id,
            model_name,
            instance_id,
            input_tokens,
            elapsed.as_millis()
        );

        Ok(Json(response_body))
    }
    .instrument(span)
    .await
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
    key: ApiKey,
    Json(mut request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!(
        "completion",
        id = %request_id,
        model = %request.model,
    );
    async {
        // ── 1. Resolve alias. ───────────────────────────────────────────────
        let (candidates, ..) = {
            let cfg = state.config.read().await;
            resolve_alias(&cfg.aliases, &request.model)
        };
        let model_name = candidates.into_iter().next().unwrap_or_default();

        // ── 2. Verify model exists; read the backend timeouts. ────────────
        let (idle_timeout, total_timeout) = {
            let cfg = state.config.read().await;
            let exists = cfg.models.iter().any(|m| m.name == model_name);
            if !exists {
                return Err(ApiError::ModelNotFound(model_name));
            }
            (
                std::time::Duration::from_secs(cfg.server.backend_idle_timeout_secs),
                std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs),
            )
        };

        // ── 3. Acquire instance slot. ───────────────────────────────────────
        let guard = state
            .manager
            .get_or_spawn(&model_name)
            .await
            .map_err(|e| acquire_error(&model_name, e))?;

        // Capture instance ID now that we hold the slot guard.
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };

        // ── 4. Read port. ──────────────────────────────────────────────────
        let port = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.port
        };

        let backend_url = format!("http://127.0.0.1:{}/v1/completions", port);

        // ── 5. Forward request. ─────────────────────────────────────────────
        // Forward the *resolved* model name to the backend, not the alias.
        request.model = model_name.clone();
        if request.stream {
            info!("stream start model={} inst={}", model_name, instance_id);
            let stream = build_sse_stream(
                state.client.clone(),
                backend_url,
                serde_json::to_value(&request).unwrap_or_default(),
                guard,
                model_name.clone(),
                instance_id.clone(),
                None,
                state.manager.clone(),
                key.label.clone(),
                request_id.clone(),
                idle_timeout,
            )
            .await
            .map_err(|e| ApiError::Internal(format!("backend request failed: {}", e)))?;

            Ok(Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response())
        } else {
            let t0 = std::time::Instant::now();
            let response_body = forward_request_aggregate(
                &state.client,
                &backend_url,
                &serde_json::to_value(&request).unwrap_or_default(),
                total_timeout,
            )
            .await?;
            let elapsed = t0.elapsed();

            log_completion(&response_body, &model_name, &instance_id, elapsed);

            Ok(Json(response_body).into_response())
        }
    }
    .instrument(span)
    .await
}

// ── /v1/embeddings ──────────────────────────────────────────────────────────

/// `POST /v1/embeddings` — OpenAI-compatible embeddings endpoint.
///
/// Same alias resolution and instance management as chat completions,
/// but always non-streaming and forwarded to the backend's `/v1/embeddings`.
pub async fn embeddings(
    State(state): State<AppState>,
    _key: ApiKey,
    Json(mut request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!(
        "embedding",
        id = %request_id,
        model = %request.model,
    );
    async {
        // ── 1. Resolve alias. ───────────────────────────────────────────────
        let (candidates, ..) = {
            let cfg = state.config.read().await;
            resolve_alias(&cfg.aliases, &request.model)
        };
        let model_name = candidates.into_iter().next().unwrap_or_default();

        // ── 2. Verify model exists; read the backend timeout. ─────────────
        let total_timeout = {
            let cfg = state.config.read().await;
            let exists = cfg.models.iter().any(|m| m.name == model_name);
            if !exists {
                return Err(ApiError::ModelNotFound(model_name));
            }
            std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs)
        };

        // Forward the *resolved* model name to the backend, not the alias.
        request.model = model_name.clone();

        // ── 3. Acquire instance slot. ───────────────────────────────────────
        let guard = state
            .manager
            .get_or_spawn(&model_name)
            .await
            .map_err(|e| acquire_error(&model_name, e))?;

        // ── 4. Read port. ──────────────────────────────────────────────────
        let port = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.port
        };

        let backend_url = format!("http://127.0.0.1:{}/v1/embeddings", port);

        // ── 5. Forward request, aggregate, release. ────────────────────────
        let t0 = std::time::Instant::now();
        let response_body = forward_request_aggregate(
            &state.client,
            &backend_url,
            &serde_json::to_value(&request).unwrap_or_default(),
            total_timeout,
        )
        .await
        .map_err(|e| {
            warn!("embedding backend request failed: {}", e);
            e
        })?;
        let elapsed = t0.elapsed();

        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };
        drop(guard);

        let resp: EmbeddingResponse = serde_json::from_value(response_body).map_err(|e| {
            ApiError::Internal(format!("failed to parse backend embedding response: {}", e))
        })?;

        info!(
            "id={} model={} inst={} embeddings={} server_ms={}",
            request_id,
            model_name,
            instance_id,
            resp.data.len(),
            elapsed.as_millis()
        );

        Ok(Json(resp))
    }
    .instrument(span)
    .await
}

// ── /v1/rerank ──────────────────────────────────────────────────────────────

/// `POST /v1/rerank` — Jina-compatible reranking endpoint (llama.cpp).
///
/// Same alias resolution and instance management as embeddings, always
/// non-streaming, forwarded to the backend's `/v1/rerank`.  Includes
/// per-request debug logging (like chat completions) and completion
/// tracking with `generated_tokens: 0` — reranking has no generation.
///
/// It is the user's responsibility to point this at a model launched as a
/// reranker (llama.cpp `--reranking`); the orchestrator does not validate
/// the model type.
pub async fn rerank(
    State(state): State<AppState>,
    key: ApiKey,
    Json(mut request): Json<RerankRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!(
        "rerank",
        id = %request_id,
        model = %request.model,
    );
    async {
        // ── 1. Resolve alias. ───────────────────────────────────────────────
        let (candidates, ..) = {
            let cfg = state.config.read().await;
            resolve_alias(&cfg.aliases, &request.model)
        };
        let model_name = candidates.into_iter().next().unwrap_or_default();

        let request_model = request.model.clone();

        // ── 2. Verify model exists; extract debug_log and the backend timeout. ─
        let (debug_log_path, total_timeout): (Option<std::path::PathBuf>, _) = {
            let cfg = state.config.read().await;
            let total = std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs);
            match cfg.models.iter().find(|m| m.name == model_name) {
                Some(m) => (m.debug_log.clone(), total),
                None => return Err(ApiError::ModelNotFound(model_name)),
            }
        };

        // Forward the *resolved* model name to the backend, not the alias.
        request.model = model_name.clone();

        // ── 3. Acquire instance slot. ───────────────────────────────────────
        let guard = state
            .manager
            .get_or_spawn(&model_name)
            .await
            .map_err(|e| acquire_error(&model_name, e))?;

        // Capture instance ID now that we hold the slot guard.
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };

        // ── 4. Read port. ──────────────────────────────────────────────────
        let port = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.port
        };

        let backend_url = format!("http://127.0.0.1:{}/v1/rerank", port);

        // ── 5. Forward request, aggregate, release. ────────────────────────
        let request_body = serde_json::to_value(&request).unwrap_or_default();

        // Debug log: request (exact body being forwarded).
        if let Some(ref log_path) = debug_log_path {
            state.debug_loggers.write_line(
                log_path,
                &DebugLogEntry {
                    ts: ts_now(),
                    request_id: request_id.clone(),
                    model: model_name.clone(),
                    alias: Some(request_model.clone()),
                    instance_id: Some(instance_id.clone()),
                    dir: "request".into(),
                    stream: Some(false),
                    body: Some(request_body.clone()),
                    usage: None,
                    duration_ms: None,
                    error: None,
                },
            );
        }

        let t0 = std::time::Instant::now();
        let response_body =
            forward_request_aggregate(&state.client, &backend_url, &request_body, total_timeout)
                .await
                .map_err(|e| {
                    warn!("rerank backend request failed: {}", e);
                    e
                })?;
        let elapsed = t0.elapsed();

        // Debug log: response.
        if let Some(ref log_path) = debug_log_path {
            let usage = response_body.pointer("/usage").cloned();
            state.debug_loggers.write_line(
                log_path,
                &DebugLogEntry {
                    ts: ts_now(),
                    request_id: request_id.clone(),
                    model: model_name.clone(),
                    alias: Some(request_model.clone()),
                    instance_id: Some(instance_id.clone()),
                    dir: "response".into(),
                    stream: Some(false),
                    body: Some(response_body.clone()),
                    usage,
                    duration_ms: Some(elapsed.as_millis() as u64),
                    error: None,
                },
            );
        }

        // Record in per-model ring buffer for /admin/status.
        // Rerank responses carry only prompt/total tokens — no generation.
        {
            let prompt_tokens = response_body
                .pointer("/usage/prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            state.manager.record_completion(
                &model_name,
                crate::types::CompletionRecord {
                    ts: ts_now(),
                    request_id: request_id.clone(),
                    instance_id: instance_id.clone(),
                    api_user: key.label.clone(),
                    prompt_tokens,
                    generated_tokens: 0,
                    cached_tokens: 0,
                    duration_ms: elapsed.as_millis() as u64,
                },
            );
        }

        let n_results = response_body
            .pointer("/results")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        info!(
            "id={} model={} inst={} results={} server_ms={}",
            request_id,
            model_name,
            instance_id,
            n_results,
            elapsed.as_millis()
        );

        Ok(Json(response_body))
    }
    .instrument(span)
    .await
}

// ── /v1/audio/* ──────────────────────────────────────────────────────────────
//
// OpenAI-compatible audio endpoints (audio.cpp / `audiocpp_server`).
//
// Routing is purely by the request's `model` field — the same alias
// resolution and per-model instance lifecycle as chat completions.  Audio
// models are regular llm-orch models whose `cmd` spawns one single-model
// `audiocpp_server` process; the llm-orch model name must equal the
// server.json `id` (audiocpp validates the request's `model` field).
//
// Unlike the chat endpoints, responses are content-type-agnostic: TTS
// returns `audio/wav` bytes, base64 JSON, SSE, or raw PCM depending on the
// request — the forwarding layer inspects the *response* Content-Type and
// either streams SSE events or passes the aggregated body through with the
// backend's status and Content-Type preserved.

/// Everything needed to forward one audio request to a concrete instance.
struct AudioTarget {
    /// Resolved (alias-free) model name — forwarded to the backend.
    model_name: String,
    /// Model name as the client sent it (for logging / debug log alias).
    request_model: String,
    port: u16,
    instance_id: String,
    guard: SlotGuard,
    idle_timeout: std::time::Duration,
    total_timeout: std::time::Duration,
    debug_log_path: Option<std::path::PathBuf>,
}

/// Resolve `request_model` (alias → config model), verify it exists, and
/// acquire an instance slot (spawning on demand).  Lock-free across awaits.
async fn acquire_audio_target(
    state: &AppState,
    request_model: &str,
) -> Result<AudioTarget, ApiError> {
    let (candidates, ..) = {
        let cfg = state.config.read().await;
        resolve_alias(&cfg.aliases, request_model)
    };
    let model_name = candidates.into_iter().next().unwrap_or_default();

    let (debug_log_path, idle_timeout, total_timeout): (Option<std::path::PathBuf>, _, _) = {
        let cfg = state.config.read().await;
        let idle = std::time::Duration::from_secs(cfg.server.backend_idle_timeout_secs);
        let total = std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs);
        match cfg.models.iter().find(|m| m.name == model_name) {
            Some(m) => (m.debug_log.clone(), idle, total),
            None => return Err(ApiError::ModelNotFound(model_name)),
        }
    };

    let guard = state
        .manager
        .get_or_spawn(&model_name)
        .await
        .map_err(|e| acquire_error(&model_name, e))?;

    let (port, instance_id) = {
        let inst = guard.handle().inner().lock().unwrap();
        (inst.port, inst.id.clone())
    };

    Ok(AudioTarget {
        model_name,
        request_model: request_model.to_owned(),
        port,
        instance_id,
        guard,
        idle_timeout,
        total_timeout,
        debug_log_path,
    })
}

/// Forward a raw POST (body bytes + Content-Type) to an audio backend and
/// build the client response from the backend's response Content-Type:
///
/// - `text/event-stream` → SSE passthrough via [`SseForwarder`] (idle
///   timeout per chunk; the `SlotGuard` is released when the stream ends).
/// - anything else → aggregate the body (bounded by the total timeout) and
///   return it with the backend's status and Content-Type.
async fn forward_audio_post(
    state: &AppState,
    target: AudioTarget,
    path: &str,
    body: Vec<u8>,
    content_type: &str,
    request_id: &str,
    api_user: &str,
) -> Result<Response, ApiError> {
    let backend_url = format!("http://127.0.0.1:{}{}", target.port, path);

    // Debug log: request.  JSON bodies are logged verbatim; anything else
    // (multipart with embedded audio) is logged as metadata only.
    if let Some(ref log_path) = target.debug_log_path {
        let debug_body = if content_type.starts_with("application/json") {
            serde_json::from_slice(&body).ok()
        } else {
            Some(serde_json::json!({
                "content_type": content_type,
                "bytes": body.len(),
            }))
        };
        state.debug_loggers.write_line(
            log_path,
            &DebugLogEntry {
                ts: ts_now(),
                request_id: request_id.to_owned(),
                model: target.model_name.clone(),
                alias: Some(target.request_model.clone()),
                instance_id: Some(target.instance_id.clone()),
                dir: "request".into(),
                stream: None,
                body: debug_body,
                usage: None,
                duration_ms: None,
                error: None,
            },
        );
    }

    let send = state
        .client
        .post(&backend_url)
        .header(header::CONTENT_TYPE, content_type)
        .body(body)
        .send();
    // The idle timeout also bounds the wait for the first response
    // headers — a hung backend must not hold the slot while silent.
    let resp = if target.idle_timeout.is_zero() {
        send.await
    } else {
        match tokio::time::timeout(target.idle_timeout, send).await {
            Ok(r) => r,
            Err(_) => {
                warn!(url = %backend_url, "audio backend sent no response headers within idle timeout");
                return Err(ApiError::BackendTimeout(format!(
                    "backend sent no response headers within {}s",
                    target.idle_timeout.as_secs()
                )));
            }
        }
    }
    .map_err(|e| ApiError::Internal(format!("backend request failed: {}", e)))?;

    let status = resp.status();
    let resp_content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if !status.is_success() {
        let text = resp.text().await.unwrap_or_else(|_| "unknown error".into());
        warn!(backend_status = %status, error = %text, "audio backend returned non-success");
        if let Some(ref log_path) = target.debug_log_path {
            state.debug_loggers.write_line(
                log_path,
                &DebugLogEntry {
                    ts: ts_now(),
                    request_id: request_id.to_owned(),
                    model: target.model_name.clone(),
                    alias: Some(target.request_model.clone()),
                    instance_id: Some(target.instance_id.clone()),
                    dir: "response".into(),
                    stream: None,
                    body: None,
                    usage: None,
                    duration_ms: None,
                    error: Some(format!("backend {}: {}", status.as_u16(), text)),
                },
            );
        }
        return Err(ApiError::Internal(format!(
            "backend returned {}: {}",
            status, text
        )));
    }

    if resp_content_type.starts_with("text/event-stream") {
        info!(
            "audio stream start model={} inst={}",
            target.model_name, target.instance_id
        );
        let stream = SseForwarder::new(
            resp.bytes_stream(),
            target.guard,
            target.model_name,
            target.instance_id,
            None, // debug context: SSE chunks may carry base64 audio — skip
            state.manager.clone(),
            api_user.to_owned(),
            request_id.to_owned(),
            target.idle_timeout,
        );
        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    // Aggregate the full body (wav, JSON, raw PCM) with the total timeout.
    let read = resp.bytes();
    let bytes = if target.total_timeout.is_zero() {
        read.await
    } else {
        match tokio::time::timeout(target.total_timeout, read).await {
            Ok(r) => r,
            Err(_) => {
                warn!(url = %backend_url, "audio backend aggregate timed out");
                return Err(ApiError::BackendTimeout(format!(
                    "backend did not complete within {}s",
                    target.total_timeout.as_secs()
                )));
            }
        }
    }
    .map_err(|e| ApiError::Internal(format!("failed to read backend response: {}", e)))?;

    if let Some(ref log_path) = target.debug_log_path {
        state.debug_loggers.write_line(
            log_path,
            &DebugLogEntry {
                ts: ts_now(),
                request_id: request_id.to_owned(),
                model: target.model_name.clone(),
                alias: Some(target.request_model.clone()),
                instance_id: Some(target.instance_id.clone()),
                dir: "response".into(),
                stream: Some(false),
                body: Some(serde_json::json!({
                    "content_type": resp_content_type,
                    "bytes": bytes.len(),
                })),
                usage: None,
                duration_ms: None,
                error: None,
            },
        );
    }

    let mut response = (status, bytes).into_response();
    if let Ok(ct) = header::HeaderValue::from_str(&resp_content_type) {
        response.headers_mut().insert(header::CONTENT_TYPE, ct);
    }
    Ok(response)
}

/// `POST /v1/audio/speech` — OpenAI-compatible text-to-speech.
///
/// The JSON body is forwarded as a raw `serde_json::Value` (new upstream
/// fields keep working); only `model` is inspected — and rewritten to the
/// *resolved* model name, since audiocpp validates it against its
/// configured server.json `id`.
pub async fn audio_speech(
    State(state): State<AppState>,
    key: ApiKey,
    Json(mut body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let request_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'model' field".into()))?
        .to_owned();
    let span = tracing::info_span!("audio_speech", id = %request_id, model = %request_model);
    async {
        let target = acquire_audio_target(&state, &request_model).await?;
        // Forward the resolved model name (strict backends validate it).
        body["model"] = serde_json::Value::String(target.model_name.clone());
        let bytes = serde_json::to_vec(&body)
            .map_err(|e| ApiError::BadRequest(format!("failed to encode request: {}", e)))?;
        forward_audio_post(
            &state,
            target,
            "/v1/audio/speech",
            bytes,
            "application/json",
            &request_id,
            &key.label,
        )
        .await
    }
    .instrument(span)
    .await
}

/// `POST /v1/audio/transcriptions` — OpenAI/Whisper-compatible ASR.
///
/// Two request shapes, dispatched on Content-Type:
/// - `application/json` — like `/v1/audio/speech` (server-side audio path).
/// - `multipart/form-data` — the raw body is buffered (bounded by the
///   256 MB body limit), the `model` form field is extracted for routing
///   and rewritten to the resolved name, and the original multipart body
///   is forwarded unchanged otherwise — audiocpp parses the form itself.
pub async fn audio_transcriptions(
    State(state): State<AppState>,
    key: ApiKey,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();

    if content_type.starts_with("multipart/form-data") {
        let boundary = multipart_boundary(&content_type)
            .ok_or_else(|| ApiError::BadRequest("multipart request without boundary".into()))?;
        let request_model =
            extract_multipart_field(&body, &boundary, "model").ok_or_else(|| {
                ApiError::BadRequest("multipart request missing 'model' field".into())
            })?;
        let span =
            tracing::info_span!("audio_transcription", id = %request_id, model = %request_model);
        return async {
            let target = acquire_audio_target(&state, &request_model).await?;
            let body = if target.model_name == request_model {
                body.to_vec()
            } else {
                // Alias used — rewrite the model field so the strict
                // backend accepts it.
                rewrite_multipart_field(&body, &boundary, "model", &target.model_name)
            };
            forward_audio_post(
                &state,
                target,
                "/v1/audio/transcriptions",
                body,
                &content_type,
                &request_id,
                &key.label,
            )
            .await
        }
        .instrument(span)
        .await;
    }

    let mut json: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::BadRequest(format!("invalid JSON body: {}", e)))?;
    let request_model = json
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'model' field".into()))?
        .to_owned();
    let span = tracing::info_span!("audio_transcription", id = %request_id, model = %request_model);
    async {
        let target = acquire_audio_target(&state, &request_model).await?;
        json["model"] = serde_json::Value::String(target.model_name.clone());
        let bytes = serde_json::to_vec(&json)
            .map_err(|e| ApiError::BadRequest(format!("failed to encode request: {}", e)))?;
        forward_audio_post(
            &state,
            target,
            "/v1/audio/transcriptions",
            bytes,
            "application/json",
            &request_id,
            &key.label,
        )
        .await
    }
    .instrument(span)
    .await
}

/// `GET /v1/audio/voices?model=<id>` — list voice ids for a TTS model.
pub async fn audio_voices(
    State(state): State<AppState>,
    _key: ApiKey,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let request_model = q
        .get("model")
        .ok_or_else(|| ApiError::BadRequest("missing 'model' query parameter".into()))?
        .clone();
    let span = tracing::info_span!("audio_voices", id = %request_id, model = %request_model);
    async {
        let target = acquire_audio_target(&state, &request_model).await?;
        let backend_url = format!(
            "http://127.0.0.1:{}/v1/audio/voices?model={}",
            target.port,
            urlencoding_simple(&target.model_name),
        );
        let send = state.client.get(&backend_url).send();
        let resp = if target.total_timeout.is_zero() {
            send.await
        } else {
            match tokio::time::timeout(target.total_timeout, send).await {
                Ok(r) => r,
                Err(_) => {
                    return Err(ApiError::BackendTimeout(format!(
                        "backend did not complete within {}s",
                        target.total_timeout.as_secs()
                    )));
                }
            }
        }
        .map_err(|e| ApiError::Internal(format!("backend request failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_else(|_| "unknown error".into());
            return Err(ApiError::Internal(format!(
                "backend returned {}: {}",
                status, text
            )));
        }
        let body = resp
            .json::<serde_json::Value>()
            .await
            .map_err(|e| ApiError::Internal(format!("failed to parse backend response: {}", e)))?;
        Ok(Json(body).into_response())
    }
    .instrument(span)
    .await
}

// ── Multipart helpers ────────────────────────────────────────────────────────

/// Extract the `boundary` parameter from a `multipart/form-data`
/// Content-Type header.
fn multipart_boundary(content_type: &str) -> Option<String> {
    for part in content_type.split(';') {
        let part = part.trim();
        if let Some(b) = part.strip_prefix("boundary=") {
            let b = b.trim_matches('"');
            if !b.is_empty() {
                return Some(b.to_owned());
            }
        }
    }
    None
}

/// Find a named text field in a multipart body.
///
/// Minimal scanner, not a full parser: locates the part whose
/// Content-Disposition carries `name="<field>"` and returns its content
/// (a short text value — audio payloads live in the `file` part, which we
/// never touch).  Returns `None` if the field or its framing is absent.
fn extract_multipart_field(body: &[u8], boundary: &str, field: &str) -> Option<String> {
    let text = String::from_utf8_lossy(body);
    let needle = format!("name=\"{}\"", field);
    let delimiter = format!("--{}", boundary);
    for part in text.split(&delimiter).skip(1) {
        let part = part
            .strip_prefix('\r')
            .and_then(|p| p.strip_prefix('\n'))
            .unwrap_or(part);
        if part.starts_with("--") {
            continue; // closing delimiter
        }
        let Some(header_end) = part.find("\r\n\r\n") else {
            continue;
        };
        let headers = &part[..header_end];
        if !headers.contains(&needle) {
            continue;
        }
        let content = &part[header_end + 4..];
        let content = content.strip_suffix("\r\n").unwrap_or(content);
        return Some(content.to_owned());
    }
    None
}

/// Rewrite the value of a named text field inside a multipart body,
/// keeping every other part byte-identical.
fn rewrite_multipart_field(body: &[u8], boundary: &str, field: &str, value: &str) -> Vec<u8> {
    let needle = format!("name=\"{}\"", field).into_bytes();
    let delimiter = format!("--{}", boundary).into_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(body.len() + value.len());
    let mut rest: &[u8] = body;
    let mut first = true;
    loop {
        let pos = find_subslice(rest, &delimiter);
        let (segment, next) = match pos {
            Some(p) => (&rest[..p], Some(&rest[p + delimiter.len()..])),
            None => (rest, None),
        };
        if !first {
            out.extend_from_slice(&delimiter);
        }
        first = false;
        // `segment` is one part (headers + content) when not the preamble.
        if let Some(header_end) = find_subslice(segment, b"\r\n\r\n") {
            let headers = &segment[..header_end];
            if find_subslice(headers, &needle).is_some() {
                out.extend_from_slice(&segment[..header_end + 4]);
                out.extend_from_slice(value.as_bytes());
                out.extend_from_slice(b"\r\n");
                match next {
                    Some(n) => {
                        rest = n;
                        continue;
                    }
                    None => break,
                }
            }
        }
        out.extend_from_slice(segment);
        match next {
            Some(n) => rest = n,
            None => break,
        }
    }
    out
}

/// Naive byte-substring search (multiparts are small enough for this).
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Minimal percent-encoding for a query parameter value.
fn urlencoding_simple(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

// ── Admin endpoints ──────────────────────────────────────────────────────────

/// `GET /admin/status` — detailed runtime status for all models.
pub async fn admin_status(
    State(state): State<AppState>,
    _key: ApiKey,
) -> Result<Json<InfoResponse>, ApiError> {
    info_endpoint(State(state), _key).await
}

/// `POST /admin/load` — force-load a model (pre-spawn an instance).
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

        // Ensure an instance exists (spawning if necessary).  Unlike the
        // request path this never parks on the request queue — an admin
        // operation must not hang behind user traffic.
        state
            .manager
            .ensure_instance(&body.model)
            .await
            .map_err(|e| acquire_error(&body.model, e))?;

        Ok(Json(AdminResponse {
            status: "ok".into(),
            message: format!(
                "model '{}' loaded (instance spawned or already running)",
                body.model
            ),
        }))
    }
    .instrument(span)
    .await
}

/// `POST /admin/unload` — force-unload all instances of a model.
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

        let (removed, draining) = state.manager.unload_model(&body.model).await;

        Ok(Json(AdminResponse {
            status: "ok".into(),
            message: format!(
                "model '{}': {} instance(s) unloaded, {} draining (busy — removed when idle)",
                body.model, removed, draining
            ),
        }))
    }
    .instrument(span)
    .await
}

/// `POST /admin/unblock` — clear the crash-block on a model.
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
    }
    .instrument(span)
    .await
}

// ── Helpers (lock-free) ──────────────────────────────────────────────────────

/// Map an instance-acquisition failure to the HTTP error for this model.
fn acquire_error(model_name: &str, e: AcquireError) -> ApiError {
    match e {
        AcquireError::Blocked => ApiError::ModelBlocked(model_name.to_owned()),
        AcquireError::NoCapacity => ApiError::NoCapacity(format!(
            "model '{}' is at capacity and the queue is full",
            model_name
        )),
        AcquireError::Unavailable => ApiError::ModelUnavailable(format!(
            "model '{}' is unavailable (spawn failed or instances retiring — see server logs)",
            model_name
        )),
    }
}

/// Resolve a requested model name to its routing information.
///
/// Returns `(candidates, policy, make_room, drain_timeout_secs)`:
///
/// - an alias resolves to its ordered target list and eviction settings;
/// - any other name resolves to a one-element candidate list with
///   make-room disabled — direct (non-alias) requests never evict
///   (`docs/003-smart-handling.md`, decision 9).
fn resolve_alias(
    aliases: &[AliasConfig],
    requested_model: &str,
) -> (Vec<String>, AliasPolicy, MakeRoomMode, u64) {
    if let Some(alias) = aliases.iter().find(|a| a.name == requested_model) {
        return (
            alias.targets.clone(),
            alias.policy,
            alias.make_room,
            alias.drain_timeout,
        );
    }
    (
        vec![requested_model.to_owned()],
        AliasPolicy::PreferLoaded,
        MakeRoomMode::None,
        0,
    )
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

    let req_id = resp.get("id").and_then(|v| v.as_str()).unwrap_or("-");

    let server_ms = elapsed.as_millis();

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
/// `total_timeout` bounds the whole exchange (request + full body): a
/// hung-but-alive backend must not tie up the in-flight slot and the
/// client request forever.  `Duration::ZERO` disables the timeout.
async fn forward_request_aggregate(
    client: &reqwest::Client,
    backend_url: &str,
    body: &serde_json::Value,
    total_timeout: std::time::Duration,
) -> Result<serde_json::Value, ApiError> {
    let work = async {
        let resp = client
            .post(backend_url)
            .json(body)
            .send()
            .await
            .map_err(|e| ApiError::Internal(format!("backend request failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_else(|_| "unknown error".into());
            warn!(backend_status = %status, error = %text, "backend returned non-success");
            return Err(ApiError::Internal(format!(
                "backend returned {}: {}",
                status, text
            )));
        }

        resp.json::<serde_json::Value>()
            .await
            .map_err(|e| ApiError::Internal(format!("failed to parse backend response: {}", e)))
    };

    if total_timeout.is_zero() {
        return work.await;
    }
    match tokio::time::timeout(total_timeout, work).await {
        Ok(result) => result,
        Err(_) => {
            warn!(
                url = %backend_url,
                timeout_secs = total_timeout.as_secs(),
                "backend aggregate request timed out"
            );
            Err(ApiError::BackendTimeout(format!(
                "backend did not complete within {}s",
                total_timeout.as_secs()
            )))
        }
    }
}

/// Build an SSE stream that forwards chunks from the backend to the client.
///
/// Returns a `Stream<Item = Result<Event, Infallible>>` suitable for axum's
/// `Sse` response type.
///
/// The `SlotGuard` is moved into the returned stream so the in-flight
/// slot is released when the stream ends (via `Drop`).
async fn build_sse_stream(
    client: reqwest::Client,
    backend_url: String,
    body: serde_json::Value,
    guard: SlotGuard,
    model_name: String,
    instance_id: String,
    debug: Option<DebugStreamContext>,
    manager: Arc<InstanceManager>,
    api_user: String,
    request_id: String,
    idle_timeout: std::time::Duration,
) -> Result<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>, reqwest::Error> {
    // The idle timeout also bounds the wait for the first response
    // headers — a hung backend must not hold the slot while silent.
    let send = client.post(&backend_url).json(&body).send();
    let resp = if idle_timeout.is_zero() {
        send.await?
    } else {
        match tokio::time::timeout(idle_timeout, send).await {
            Ok(r) => r?,
            Err(_) => {
                warn!(
                    url = %backend_url,
                    timeout_secs = idle_timeout.as_secs(),
                    "backend sent no response headers within idle timeout"
                );
                if let Some(ctx) = debug {
                    ctx.loggers.write_line(
                        &ctx.path,
                        &DebugLogEntry {
                            ts: ts_now(),
                            request_id: ctx.request_id,
                            model: ctx.model_name,
                            alias: ctx.alias,
                            instance_id: Some(ctx.instance_id),
                            dir: "response".into(),
                            stream: Some(true),
                            body: None,
                            usage: None,
                            duration_ms: Some(ctx.t0.elapsed().as_millis() as u64),
                            error: Some(format!(
                                "backend idle timeout ({}s) waiting for response headers",
                                idle_timeout.as_secs()
                            )),
                        },
                    );
                }
                let stream = futures_util::stream::once(async move {
                    let error_sse = serde_json::json!({
                        "error": {
                            "message": format!("backend sent no response for {}s (idle timeout)", idle_timeout.as_secs()),
                            "type": "backend_idle_timeout",
                        }
                    });
                    Ok(Event::default().data(error_sse.to_string()).event("error"))
                });
                let stream = StreamWithGuard::new(Box::pin(stream), guard);
                return Ok(Box::pin(stream));
            }
        }
    };

    let status = resp.status();
    if !status.is_success() {
        // Non-success before streaming — return an error event stream.
        let error_text = resp.text().await.unwrap_or_else(|_| "unknown error".into());
        warn!(backend_status = %status, error = %error_text, "backend returned non-success");

        // Debug log: error response.
        if let Some(ctx) = debug {
            ctx.loggers.write_line(
                &ctx.path,
                &DebugLogEntry {
                    ts: ts_now(),
                    request_id: ctx.request_id,
                    model: ctx.model_name,
                    alias: ctx.alias,
                    instance_id: Some(ctx.instance_id),
                    dir: "response".into(),
                    stream: Some(true),
                    body: None,
                    usage: None,
                    duration_ms: Some(ctx.t0.elapsed().as_millis() as u64),
                    error: Some(format!("backend {}: {}", status.as_u16(), error_text)),
                },
            );
        }

        // Return a stream that emits a single error event and then ends.
        let stream = futures_util::stream::once(async move {
            let error_sse = serde_json::json!({
                "error": {
                    "message": format!("backend error: {}", error_text),
                    "type": "backend_error",
                    "code": status.as_u16(),
                }
            });
            Ok(Event::default().data(error_sse.to_string()).event("error"))
        });
        // Keep `guard` alive until the stream is consumed (just one event).
        let stream = StreamWithGuard::new(Box::pin(stream), guard);
        return Ok(Box::pin(stream));
    }

    // Success — forward the byte stream as SSE events.
    let byte_stream = resp.bytes_stream();
    let sse_stream = SseForwarder::new(
        byte_stream,
        guard,
        model_name,
        instance_id,
        debug,
        manager,
        api_user,
        request_id,
        idle_timeout,
    );

    Ok(Box::pin(sse_stream))
}

// ── Stream wrappers ──────────────────────────────────────────────────────────

/// A wrapper around a byte stream that converts backend SSE chunks into
/// axum `Event` objects.
///
/// Holds a `SlotGuard` so the in-flight slot is released when the
/// stream is dropped (end of response or client disconnect).
///
/// Stream-end semantics: the backend's own `data: [DONE]` is the only
/// end-of-stream signal forwarded (`saw_done`).  A stream that ends
/// without it is *truncated* — the client gets an explicit error event,
/// never a synthetic `[DONE]` that would make partial output look
/// complete, and no completion is recorded.
struct SseForwarder<S> {
    inner: Option<S>,
    /// Accumulated partial line from previous chunks.
    buffer: Vec<u8>,
    /// Debug log context + accumulated SSE data chunks (logged on Drop).
    debug: Option<DebugStreamContext>,
    /// SSE data chunks.  With a debug log: the full generation (logged on
    /// Drop).  Without: only the most recent chunk is kept (the usage
    /// chunk, if any, is always last) — no unbounded buffering.
    chunks: Vec<String>,
    /// The backend sent its own `data: [DONE]` terminator.
    saw_done: bool,
    /// Maximum gap between backend chunks before the backend is declared
    /// hung.  `Duration::ZERO` disables the idle timeout.
    idle_timeout: std::time::Duration,
    /// Deadline for the next backend chunk (`None` = no idle timeout).
    /// Reset every time a chunk arrives.
    idle_sleep: Option<Pin<Box<tokio::time::Sleep>>>,
    /// Used to record completion stats on stream end.
    manager: Arc<InstanceManager>,
    model_name: String,
    api_user: String,
    request_id: String,
    t0: std::time::Instant,
    /// Kept alive until the stream ends.
    _guard: SlotGuard,
    instance_id: String,
}

impl<S> SseForwarder<S>
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    fn new(
        stream: S,
        guard: SlotGuard,
        model_name: String,
        instance_id: String,
        debug: Option<DebugStreamContext>,
        manager: Arc<InstanceManager>,
        api_user: String,
        request_id: String,
        idle_timeout: std::time::Duration,
    ) -> Self {
        debug!(
            "SseForwarder::new: model={} inst={}",
            model_name, instance_id
        );
        let idle_sleep = if idle_timeout.is_zero() {
            None
        } else {
            Some(Box::pin(tokio::time::sleep(idle_timeout)))
        };
        Self {
            inner: Some(stream),
            buffer: Vec::new(),
            debug,
            chunks: Vec::new(),
            saw_done: false,
            idle_timeout,
            idle_sleep,
            manager,
            model_name,
            api_user,
            request_id,
            t0: std::time::Instant::now(),
            _guard: guard,
            instance_id,
        }
    }
}

impl<S> Stream for SseForwarder<S>
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Fast path: stream already exhausted.
        if self.inner.is_none() {
            return Poll::Ready(None);
        }
        loop {
            // Try to extract a complete event from the buffer.
            if let Some((event, data)) = extract_sse_event(&mut self.buffer) {
                if data == "[DONE]" {
                    // The backend's own terminator — forwarded like any
                    // other event and remembered so the stream end is
                    // recognized as a clean completion.
                    self.saw_done = true;
                } else if !data.is_empty() {
                    if self.debug.is_some() {
                        // Full accumulation for the debug log body.
                        self.chunks.push(data);
                    } else {
                        // Without a debug log only the final (usage) chunk
                        // matters — don't buffer the whole generation.
                        self.chunks.clear();
                        self.chunks.push(data);
                    }
                }
                return Poll::Ready(Some(Ok(event)));
            }

            // Need more data — poll the inner stream.
            let inner = match self.inner.as_mut() {
                Some(s) => s,
                None => unreachable!(),
            };
            match Pin::new(inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buffer.extend_from_slice(&chunk);
                    // Backend is alive — reset the idle deadline.
                    let idle_timeout = self.idle_timeout;
                    if let Some(sleep) = self.idle_sleep.as_mut() {
                        sleep
                            .as_mut()
                            .reset(tokio::time::Instant::now() + idle_timeout);
                    }
                    // Loop back to try extracting an event.
                }
                Poll::Ready(Some(Err(e))) => {
                    warn!(error = %e, "backend stream error");
                    self.inner = None;
                    let error_sse = serde_json::json!({
                        "error": {
                            "message": format!("backend stream error: {}", e),
                            "type": "backend_stream_error",
                        }
                    });
                    let error_event = Event::default().data(error_sse.to_string()).event("error");
                    return Poll::Ready(Some(Ok(error_event)));
                }
                Poll::Ready(None) => {
                    debug!(
                        "SseForwarder::poll_next: stream ended model={} inst={}",
                        self.model_name, self.instance_id
                    );
                    self.inner = None;
                    if !self.buffer.is_empty() {
                        // Leftover bytes = an SSE event without its
                        // terminating blank line — incomplete JSON that
                        // must not be forwarded as if valid.
                        debug!(
                            leftover = self.buffer.len(),
                            "discarding unterminated SSE event at stream end"
                        );
                        self.buffer.clear();
                    }
                    if self.saw_done {
                        // Clean completion — the backend's own [DONE] was
                        // already forwarded; nothing more to emit.
                        return Poll::Ready(None);
                    }
                    // The backend stream ended without [DONE]: the
                    // response is truncated.  Signal it explicitly —
                    // clients must not mistake partial output for a
                    // complete answer.
                    warn!(
                        model = %self.model_name,
                        inst = %self.instance_id,
                        "backend stream ended without [DONE] — truncated response"
                    );
                    let error_sse = serde_json::json!({
                        "error": {
                            "message": "backend stream ended without completing (truncated response)",
                            "type": "stream_truncated",
                        }
                    });
                    return Poll::Ready(Some(Ok(Event::default()
                        .data(error_sse.to_string())
                        .event("error"))));
                }
                Poll::Pending => {
                    // No backend data right now — enforce the idle timeout
                    // so a hung-but-alive backend can't hold the slot
                    // (and the client connection) forever.
                    let idle_timeout = self.idle_timeout;
                    let timed_out = match self.idle_sleep.as_mut() {
                        Some(sleep) => std::future::Future::poll(sleep.as_mut(), cx).is_ready(),
                        None => false,
                    };
                    if timed_out {
                        warn!(
                            model = %self.model_name,
                            inst = %self.instance_id,
                            timeout_secs = idle_timeout.as_secs(),
                            "backend stream idle timeout"
                        );
                        self.inner = None;
                        let error_sse = serde_json::json!({
                            "error": {
                                "message": format!("backend sent no data for {}s (idle timeout)", idle_timeout.as_secs()),
                                "type": "backend_idle_timeout",
                            }
                        });
                        return Poll::Ready(Some(Ok(Event::default()
                            .data(error_sse.to_string())
                            .event("error"))));
                    }
                    return Poll::Pending;
                }
            }
        }
    }
}

impl<S> Drop for SseForwarder<S> {
    fn drop(&mut self) {
        debug!(
            "SseForwarder::drop: model={} inst={}",
            self.model_name, self.instance_id
        );
        let duration_ms = self.t0.elapsed().as_millis() as u64;

        // Only clean completions (backend sent [DONE]) are recorded —
        // truncated streams and client disconnects must not leave phantom
        // 0-token records in the ring buffer.
        if self.saw_done {
            // Parse the last usage-bearing chunk for token counts.
            let mut prompt_tokens = 0u64;
            let mut gen_tokens = 0u64;
            let mut cached_tokens = 0u64;
            for chunk in self.chunks.iter().rev() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(chunk) {
                    if let Some(usage) = v.get("usage") {
                        prompt_tokens = usage
                            .get("prompt_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        gen_tokens = usage
                            .get("completion_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        cached_tokens = usage
                            .pointer("/prompt_tokens_details/cached_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        break;
                    }
                }
            }
            self.manager.record_completion(
                &self.model_name,
                crate::types::CompletionRecord {
                    ts: crate::debug_log::ts_now(),
                    request_id: self.request_id.clone(),
                    instance_id: self.instance_id.clone(),
                    api_user: self.api_user.clone(),
                    prompt_tokens,
                    generated_tokens: gen_tokens,
                    cached_tokens,
                    duration_ms,
                },
            );
        } else {
            debug!(
                model = %self.model_name,
                inst = %self.instance_id,
                "stream ended without [DONE] (truncated or client disconnect) — no completion recorded"
            );
        }

        // Debug log: streaming response (logged on drop so we capture partial
        // output even on client disconnect).
        if let Some(ctx) = self.debug.take() {
            let body = serde_json::json!({"chunks": self.chunks});
            let error = if self.saw_done {
                None
            } else {
                Some("stream ended before [DONE] (truncated or client disconnect)".to_string())
            };
            ctx.loggers.write_line(
                &ctx.path,
                &DebugLogEntry {
                    ts: ts_now(),
                    request_id: ctx.request_id,
                    model: ctx.model_name,
                    alias: ctx.alias,
                    instance_id: Some(ctx.instance_id),
                    dir: "response".into(),
                    stream: Some(true),
                    body: Some(body),
                    usage: None,
                    duration_ms: Some(duration_ms),
                    error,
                },
            );
        }
    }
}

/// Extract a complete SSE event from a byte buffer.
///
/// SSE events are terminated by a double newline (`\n\n` or `\r\n\r\n`).
/// Returns the extracted event with its raw data string and removes both
/// from the buffer.
fn extract_sse_event(buffer: &mut Vec<u8>) -> Option<(Event, String)> {
    let double_nl = find_double_newline(buffer)?;
    let event_bytes: Vec<u8> = buffer.drain(..double_nl).collect();
    let sep_len = detect_separator_len(buffer);
    buffer.drain(..sep_len);

    let event_str = String::from_utf8_lossy(&event_bytes);

    let mut data_lines: Vec<String> = Vec::new();
    let mut event_type: Option<String> = None;

    for line in event_str.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("event:") {
            event_type = Some(rest.trim().to_string());
        }
    }

    if data_lines.is_empty() {
        return None;
    }

    let data = data_lines.join("\n");
    let mut event = Event::default().data(data.clone());
    if let Some(et) = event_type {
        event = event.event(et);
    }
    Some((event, data))
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

/// A wrapper that keeps a `SlotGuard` alive alongside an arbitrary stream.
///
/// Used when we need to return an error stream that still holds the guard
/// (so the in-flight slot is released after the single error event is consumed).
struct StreamWithGuard<S> {
    inner: S,
    _guard: SlotGuard,
}

impl<S> StreamWithGuard<S> {
    fn new(inner: S, guard: SlotGuard) -> Self {
        Self {
            inner,
            _guard: guard,
        }
    }
}

impl<S> Stream for StreamWithGuard<S>
where
    S: Stream<Item = Result<Event, Infallible>> + Unpin,
{
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{Instance, InstanceHandle, InstanceState};
    use futures_util::StreamExt;

    const TEST_YAML: &str = r#"
server: {}
apikeys_file: apikeys.txt
models:
  - name: m
    context_length: 4096
    cmd: "sleep 3600"
    idle_ttl: 60
"#;

    fn test_manager() -> Arc<InstanceManager> {
        let config: crate::config::Config = serde_yaml_ng::from_str(TEST_YAML).unwrap();
        let gpu_snapshot = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, gpu_snapshot, None);
        Arc::new(mgr)
    }

    fn test_guard() -> SlotGuard {
        let mut inst = Instance::new("m", vec![], 9999, None);
        inst.state = InstanceState::Ready;
        let handle = InstanceHandle::new(inst);
        handle.try_acquire(4).expect("slot available")
    }

    // ── Multipart helpers (audio endpoints) ──────────────────────────────

    const MULTIPART: &str = "--bX\r\n\
Content-Disposition: form-data; name=\"model\"\r\n\r\n\
qwen3-asr\r\n\
--bX\r\n\
Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
Content-Type: audio/wav\r\n\r\n\
RIFF\x24\x00\x00\x00WAVE-binary-junk\r\n\
--bX\r\n\
Content-Disposition: form-data; name=\"stream\"\r\n\r\n\
true\r\n\
--bX--\r\n";

    #[test]
    fn multipart_boundary_parsed_from_content_type() {
        assert_eq!(
            multipart_boundary("multipart/form-data; boundary=bX"),
            Some("bX".into())
        );
        assert_eq!(
            multipart_boundary("multipart/form-data; charset=utf-8; boundary=\"abc-123\""),
            Some("abc-123".into())
        );
        assert_eq!(multipart_boundary("application/json"), None);
        assert_eq!(multipart_boundary("multipart/form-data; boundary="), None);
    }

    #[test]
    fn extract_multipart_field_finds_text_fields() {
        assert_eq!(
            extract_multipart_field(MULTIPART.as_bytes(), "bX", "model"),
            Some("qwen3-asr".into())
        );
        assert_eq!(
            extract_multipart_field(MULTIPART.as_bytes(), "bX", "stream"),
            Some("true".into())
        );
        assert_eq!(
            extract_multipart_field(MULTIPART.as_bytes(), "bX", "nope"),
            None
        );
    }

    #[test]
    fn rewrite_multipart_field_replaces_value_keeps_rest() {
        let out = rewrite_multipart_field(MULTIPART.as_bytes(), "bX", "model", "resolved-id");
        let out = String::from_utf8(out).unwrap();
        assert!(
            out.contains("name=\"model\"\r\n\r\nresolved-id\r\n"),
            "{out}"
        );
        assert!(!out.contains("qwen3-asr"), "old value must be gone: {out}");
        // Other parts are byte-identical.
        assert!(out.contains("RIFF\x24\x00\x00\x00WAVE-binary-junk"));
        assert!(out.contains("name=\"stream\"\r\n\r\ntrue\r\n"));
        assert!(out.ends_with("--bX--\r\n"));
    }

    #[test]
    fn rewrite_multipart_field_noop_when_field_absent() {
        let body = "--b\r\nContent-Disposition: form-data; name=\"x\"\r\n\r\n1\r\n--b--\r\n";
        let out = rewrite_multipart_field(body.as_bytes(), "b", "model", "v");
        assert_eq!(out, body.as_bytes());
    }

    fn forwarder_from(
        chunks: &[&str],
        mgr: Arc<InstanceManager>,
    ) -> SseForwarder<impl Stream<Item = Result<bytes::Bytes, reqwest::Error>>> {
        let byte_stream = futures_util::stream::iter(
            chunks.iter().map(|c| Ok(bytes::Bytes::from(c.to_string()))),
        );
        SseForwarder::new(
            byte_stream,
            test_guard(),
            "m".into(),
            "m@cpu#0".into(),
            None,
            mgr,
            "user".into(),
            "req-1".into(),
            std::time::Duration::ZERO, // idle timeout disabled
        )
    }

    /// Event content via its Debug impl (buffer bytes are shown).
    fn event_text(ev: &Event) -> String {
        format!("{:?}", ev)
    }

    fn completions_recorded(mgr: &InstanceManager) -> usize {
        mgr.recent_completions_snapshot()
            .get("m")
            .map(|v| v.len())
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn clean_completion_forwards_done_exactly_once() {
        let mgr = test_manager();
        let mut stream = forwarder_from(
            &[
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                "data: {\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
                "data: [DONE]\n\n",
            ],
            mgr.clone(),
        );

        let mut events = Vec::new();
        while let Some(ev) = stream.next().await {
            events.push(event_text(&ev.unwrap()));
        }
        assert_eq!(events.len(), 3, "chunk + usage + [DONE], nothing synthetic");
        assert!(events[2].contains("[DONE]"));
        assert!(
            !events.iter().any(|e| e.contains("stream_truncated")),
            "clean completion must not emit a truncation error"
        );
        assert!(stream.saw_done);

        drop(stream);
        assert_eq!(
            completions_recorded(&mgr),
            1,
            "clean completion must be recorded"
        );
        let recs = mgr.recent_completions_snapshot();
        let rec = &recs["m"][0];
        assert_eq!((rec.prompt_tokens, rec.generated_tokens), (3, 2));
    }

    #[tokio::test]
    async fn truncated_stream_signals_error_not_fake_done() {
        // Backend connection ends mid-generation without [DONE]: the
        // client must get an explicit truncation error, never a synthetic
        // [DONE] that makes partial output look complete.
        let mgr = test_manager();
        let mut stream = forwarder_from(
            &["data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"],
            mgr.clone(),
        );

        let mut events = Vec::new();
        while let Some(ev) = stream.next().await {
            events.push(event_text(&ev.unwrap()));
        }
        assert_eq!(events.len(), 2, "content chunk + truncation error");
        assert!(events[1].contains("stream_truncated"));
        assert!(!events[1].contains("[DONE]"));
        assert!(!stream.saw_done);

        drop(stream);
        assert_eq!(
            completions_recorded(&mgr),
            0,
            "truncated stream must not record a phantom completion"
        );
    }

    #[tokio::test]
    async fn client_disconnect_records_no_completion() {
        // Dropping the stream mid-generation (client gone) must not
        // record a phantom completion either.
        let mgr = test_manager();
        let mut stream = forwarder_from(
            &[
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                "data: [DONE]\n\n",
            ],
            mgr.clone(),
        );

        // Consume one event, then drop before [DONE] is seen.
        let first = stream.next().await;
        assert!(first.is_some());
        assert!(!stream.saw_done);
        drop(stream);

        assert_eq!(completions_recorded(&mgr), 0);
    }

    #[tokio::test]
    async fn unterminated_tail_event_is_not_forwarded() {
        // A partial SSE event at stream end (no terminating blank line)
        // is incomplete JSON — discard it and signal truncation instead
        // of forwarding it as if valid.
        let mgr = test_manager();
        let mut stream = forwarder_from(
            &["data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]"],
            mgr.clone(),
        );

        let mut events = Vec::new();
        while let Some(ev) = stream.next().await {
            events.push(event_text(&ev.unwrap()));
        }
        assert_eq!(events.len(), 1, "only the truncation error");
        assert!(events[0].contains("stream_truncated"));
    }

    #[tokio::test(start_paused = true)]
    async fn idle_timeout_signals_error_and_ends_stream() {
        // First chunk arrives, then the backend goes silent forever:
        // the idle timeout must end the stream with an explicit error
        // instead of hanging the client (and the slot) forever.
        let mgr = test_manager();
        let byte_stream = futures_util::stream::iter(vec![Ok(bytes::Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n".to_string(),
        ))])
        .chain(futures_util::stream::pending::<
            Result<bytes::Bytes, reqwest::Error>,
        >());

        let mut stream = SseForwarder::new(
            byte_stream,
            test_guard(),
            "m".into(),
            "m@cpu#0".into(),
            None,
            mgr.clone(),
            "user".into(),
            "req-1".into(),
            std::time::Duration::from_millis(50),
        );

        let first = stream.next().await.unwrap().unwrap();
        assert!(event_text(&first).contains("hi"));

        // Paused time auto-advances past the idle deadline while we await.
        let second = stream.next().await.unwrap().unwrap();
        assert!(event_text(&second).contains("backend_idle_timeout"));
        assert!(
            stream.next().await.is_none(),
            "stream must end after the idle error"
        );
        assert!(!stream.saw_done);

        drop(stream);
        assert_eq!(
            completions_recorded(&mgr),
            0,
            "idle-timed-out stream must not record a completion"
        );
    }

    #[tokio::test]
    async fn aggregate_request_times_out_on_hung_backend() {
        // Backend that accepts connections but never responds: the total
        // timeout must fail the request instead of hanging forever.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((sock, _)) = listener.accept().await {
                held.push(sock); // hold the socket open, never answer
            }
        });

        let client = reqwest::Client::new();
        let url = format!("http://{}/v1/chat/completions", addr);
        let err = forward_request_aggregate(
            &client,
            &url,
            &serde_json::json!({"model": "m"}),
            std::time::Duration::from_millis(200),
        )
        .await
        .expect_err("hung backend must time out");
        assert!(
            matches!(err, ApiError::BackendTimeout(_)),
            "expected BackendTimeout, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn chunks_are_not_buffered_without_debug_log() {
        // Without a debug log, only the most recent chunk (usage) is
        // retained — no unbounded per-request buffering.
        let mgr = test_manager();
        let mut stream = forwarder_from(
            &[
                "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\n\n",
                "data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":3}}\n\n",
                "data: [DONE]\n\n",
            ],
            mgr.clone(),
        );
        while stream.next().await.is_some() {}
        assert!(
            stream.chunks.len() <= 1,
            "without a debug log only the usage chunk is kept, got {}",
            stream.chunks.len()
        );
        drop(stream);
        let recs = mgr.recent_completions_snapshot();
        assert_eq!(recs["m"][0].generated_tokens, 3);
    }
}
