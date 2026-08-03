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
use std::sync::Arc;
use std::task::{Context, Poll};
use tracing::{debug, info, warn};

use crate::config::AliasConfig;
use crate::debug_log::{DebugLogEntry, DebugStreamContext, ts_now};
use crate::instance::SlotGuard;
use crate::scheduler::{AcquireError, InstanceManager};
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

    // ── 4. Forward the request. ─────────────────────────────────────────
    // Forward the *resolved* model name to the backend, not the alias —
    // strict backends validate the model field.
    request.model = model_name.clone();
    let request_body = serde_json::to_value(&request).unwrap_or_default();

    if request.stream {
        // Streaming mode: acquire a slot (spawning/queueing as needed,
        // emitting loading-indicator dots while waiting) and forward the
        // backend SSE stream to the client.  The `SlotGuard` is moved into
        // the stream wrapper so the in-flight slot is released when the
        // stream ends (Drop).
        let stream = build_response_stream(
            state.client.clone(),
            state.manager.clone(),
            model_name.clone(),
            request_model.clone(),
            request_body,
            key.label.clone(),
            request_id.clone(),
            state.debug_loggers.clone(),
            debug_log_path,
            idle_timeout,
            "/v1/chat/completions",
        );

        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // Non-streaming mode: acquire a slot first, then aggregate the
        // response (no loading dots — the response hasn't started yet).
        let guard = state
            .manager
            .get_or_spawn(&model_name)
            .await
            .map_err(|e| acquire_error(&model_name, e))?;
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };
        let port = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.port
        };
        let backend_url = format!("http://127.0.0.1:{}/v1/chat/completions", port);

        // Debug log: request (exact body being forwarded).
        if let Some(ref log_path) = debug_log_path {
            state.debug_loggers.write_line(log_path, &DebugLogEntry {
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
            });
        }

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
            state.debug_loggers.write_line(log_path, &DebugLogEntry {
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
            });
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
    let (model_name, _alias_system_prompt, _alias_prompt_template) = {
        let cfg = state.config.read().await;
        resolve_alias(&cfg.aliases, &request.model)
    };

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

    // ── 3. Forward request. ─────────────────────────────────────────────
    // Forward the *resolved* model name to the backend, not the alias.
    request.model = model_name.clone();
    let request_body = serde_json::to_value(&request).unwrap_or_default();

    if request.stream {
        // Streaming mode: acquire a slot (emitting loading-indicator dots
        // while waiting) and forward the backend SSE stream.
        let stream = build_response_stream(
            state.client.clone(),
            state.manager.clone(),
            model_name.clone(),
            model_name.clone(),
            request_body,
            key.label.clone(),
            request_id.clone(),
            state.debug_loggers.clone(),
            None,
            idle_timeout,
            "/v1/completions",
        );

        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        let guard = state
            .manager
            .get_or_spawn(&model_name)
            .await
            .map_err(|e| acquire_error(&model_name, e))?;
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };
        let port = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.port
        };
        let backend_url = format!("http://127.0.0.1:{}/v1/completions", port);
        let t0 = std::time::Instant::now();
        let response_body = forward_request_aggregate(
            &state.client,
            &backend_url,
            &request_body,
            total_timeout,
        )
        .await?;
        let elapsed = t0.elapsed();

        log_completion(&response_body, &model_name, &instance_id, elapsed);

        Ok(Json(response_body).into_response())
    }
    }.instrument(span).await
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
    let (model_name, _alias_system_prompt, _alias_prompt_template) = {
        let cfg = state.config.read().await;
        resolve_alias(&cfg.aliases, &request.model)
    };

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

    let resp: EmbeddingResponse = serde_json::from_value(response_body)
        .map_err(|e| ApiError::Internal(format!("failed to parse backend embedding response: {}", e)))?;

    info!(
        "id={} model={} inst={} embeddings={} server_ms={}",
        request_id,
        model_name,
        instance_id,
        resp.data.len(),
        elapsed.as_millis()
    );

    Ok(Json(resp))
    }.instrument(span).await
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
    let (model_name, _alias_system_prompt, _alias_prompt_template) = {
        let cfg = state.config.read().await;
        resolve_alias(&cfg.aliases, &request.model)
    };

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
        state.debug_loggers.write_line(log_path, &DebugLogEntry {
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
        });
    }

    let t0 = std::time::Instant::now();
    let response_body = forward_request_aggregate(
        &state.client,
        &backend_url,
        &request_body,
        total_timeout,
    )
    .await
    .map_err(|e| {
        warn!("rerank backend request failed: {}", e);
        e
    })?;
    let elapsed = t0.elapsed();

    // Debug log: response.
    if let Some(ref log_path) = debug_log_path {
        let usage = response_body.pointer("/usage").cloned();
        state.debug_loggers.write_line(log_path, &DebugLogEntry {
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
        });
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
    }.instrument(span).await
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
        message: format!("model '{}' loaded (instance spawned or already running)", body.model),
    }))
    }.instrument(span).await
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
    }.instrument(span).await
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
    }.instrument(span).await
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

/// Resolve an alias name to its underlying model.
///
/// Returns `(model_name, optional_system_prompt, optional_prompt_template)`.
/// If the name is not an alias, returns the name unchanged with `None` prompts.
fn resolve_alias(
    aliases: &[AliasConfig],
    requested_model: &str,
) -> (String, Option<String>, Option<String>) {
    if let Some(alias) = aliases.iter().find(|a| a.name == requested_model) {
        return (
            alias.target.clone(),
            alias.system_prompt.clone(),
            alias.prompt_template.clone(),
        );
    }
    (requested_model.to_owned(), None, None)
}

/// Apply alias system prompt and/or prompt template injection to messages.
fn apply_alias_prompts(
    messages: &mut Vec<ChatMessage>,
    system_prompt: Option<String>,
    prompt_template: Option<String>,
) {
    if let Some(sp) = system_prompt {
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
                    tool_calls: None,
                    tool_call_id: None,
                    extra: serde_json::Map::new(),
                },
            );
        }
    }

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
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".into());
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

/// Build an SSE stream for a streaming request, emitting optional
/// loading-indicator dots while the backend instance is acquired.
///
/// This *reuses* the canonical `InstanceManager::get_or_spawn` routing and
/// queueing — so `queue_depth`/429, autoscale gating, and the fast-path
/// acquire all still apply to streaming requests.  Loading dots are a pure
/// presentation layer on top of that same acquisition future: when
/// acquisition takes longer than the configured interval, a keep-alive `.`
/// comment is emitted (clients ignore SSE comments), preventing client-side
/// timeouts on slow model loads without ever touching the model's
/// conversation context.
///
/// Because dots are gated on the *actual* wait (not on a separate
/// loading-state probe), they also cover queue waits behind a saturated
/// model, not just cold loads.  The debug "request" log is written here,
/// once the real instance is known, so the dots path records the actual
/// backend instance id.
///
/// Returns a `Stream<Item = Result<Event, Infallible>>` suitable for axum's
/// `Sse` response type.  The `SlotGuard` is moved into the stream so the
/// in-flight slot is released when the stream ends (via `Drop`).
#[allow(clippy::too_many_arguments)]
fn build_response_stream(
    client: reqwest::Client,
    manager: Arc<InstanceManager>,
    model_name: String,
    request_model: String,
    body: serde_json::Value,
    api_user: String,
    request_id: String,
    debug_loggers: Arc<crate::debug_log::DebugLoggers>,
    debug_log_path: Option<std::path::PathBuf>,
    idle_timeout: std::time::Duration,
    backend_endpoint: &'static str,
) -> Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        // Effective loading-dots interval for this model (None = disabled).
        let dots = manager.loading_dots_interval(&model_name);
        let acquire = manager.get_or_spawn(&model_name);

        // ── Phase 1: acquire a slot, emitting dots while waiting. ──
        // The acquisition future is pinned and polled across ticks (never
        // dropped/restarted), so queueing state is preserved and waiters
        // aren't re-queued at the back on every dot.  The first tick is
        // scheduled one interval out, so a fast (fast-path) acquire emits
        // no spurious dots.
        let (guard, emitted_dots) = match dots {
            Some(interval) => {
                let mut ticker = tokio::time::interval_at(
                    tokio::time::Instant::now() + interval,
                    interval,
                );
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                let mut emitted = false;
                // Pin the acquisition future so it can be polled across
                // ticks without being dropped (dropping would lose queue
                // position / re-queue the waiter at the back).
                tokio::pin!(acquire);
                let result = loop {
                    tokio::select! {
                        r = &mut acquire => break (r, emitted),
                        _ = ticker.tick() => {
                            emitted = true;
                            // Comment events are ignored by SSE clients.
                            if tx
                                .send(Ok(Event::default().comment(".")))
                                .await
                                .is_err()
                            {
                                return; // client disconnected
                            }
                        }
                    }
                };
                match result {
                    (Ok(guard), emitted) => (guard, emitted),
                    (Err(e), _) => {
                        let _ = tx
                            .send(Ok(Event::default()
                                .data(acquire_error_sse(&model_name, e))
                                .event("error")))
                            .await;
                        return;
                    }
                }
            }
            None => match acquire.await {
                Ok(guard) => (guard, false),
                Err(e) => {
                    let _ = tx
                        .send(Ok(Event::default()
                            .data(acquire_error_sse(&model_name, e))
                            .event("error")))
                        .await;
                    return;
                }
            },
        };

        // Summarise a long load so the client can show progress state.
        if emitted_dots {
            let _ = tx
                .send(Ok(Event::default()
                    .comment(format!("< {} loaded >", model_name))))
                .await;
        }

        // ── Phase 2: instance is ready — forward the request. ──
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };
        let port = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.port
        };
        let backend_url = format!("http://127.0.0.1:{}{}", port, backend_endpoint);

        // Debug log: request (exact body being forwarded).
        if let Some(ref path) = debug_log_path {
            debug_loggers.write_line(path, &DebugLogEntry {
                ts: ts_now(),
                request_id: request_id.clone(),
                model: model_name.clone(),
                alias: Some(request_model.clone()),
                instance_id: Some(instance_id.clone()),
                dir: "request".into(),
                stream: Some(true),
                body: Some(body.clone()),
                usage: None,
                duration_ms: None,
                error: None,
            });
        }

        info!("stream start model={} inst={}", model_name, instance_id);
        let debug_ctx = debug_log_path.map(|path| DebugStreamContext {
            loggers: debug_loggers.clone(),
            path,
            request_id: request_id.clone(),
            model_name: model_name.clone(),
            alias: Some(request_model.clone()),
            instance_id: instance_id.clone(),
            t0: std::time::Instant::now(),
        });

        match build_sse_stream(
            client,
            backend_url,
            body,
            guard,
            model_name,
            instance_id,
            debug_ctx,
            manager,
            api_user,
            request_id,
            idle_timeout,
        )
        .await
        {
            Ok(mut stream) => {
                use futures_util::StreamExt;
                while let Some(ev) = stream.next().await {
                    if tx.send(ev).await.is_err() {
                        break; // client disconnected
                    }
                }
            }
            Err(e) => {
                let _ = tx
                    .send(Ok(Event::default()
                        .data(
                            serde_json::json!({ "error": {
                                "message": format!("backend request failed: {}", e),
                                "type": "backend_error",
                            }})
                            .to_string(),
                        )
                        .event("error")))
                    .await;
            }
        }
    });

    use futures_util::StreamExt;
    tokio_stream::wrappers::ReceiverStream::new(rx).boxed()
}

/// Map an `AcquireError` to an SSE error body for the streaming path, so a
/// streaming request that couldn't acquire a slot (blocked / 429 / spawn
/// failed) still surfaces the failure to the client as an in-band event.
fn acquire_error_sse(model_name: &str, e: AcquireError) -> String {
    let (message, etype) = match e {
        AcquireError::Blocked => (
            format!("model '{}' is blocked", model_name),
            "model_blocked",
        ),
        AcquireError::NoCapacity => (
            format!("model '{}' is at capacity and the queue is full", model_name),
            "no_capacity",
        ),
        AcquireError::Unavailable => (
            format!(
                "model '{}' is unavailable (spawn failed or instances retiring — see server logs)",
                model_name
            ),
            "model_unavailable",
        ),
    };
    serde_json::json!({ "error": { "message": message, "type": etype } }).to_string()
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
                    ctx.loggers.write_line(&ctx.path, &DebugLogEntry {
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
                    });
                }
                let stream = futures_util::stream::once(async move {
                    let error_sse = serde_json::json!({
                        "error": {
                            "message": format!("backend sent no response for {}s (idle timeout)", idle_timeout.as_secs()),
                            "type": "backend_idle_timeout",
                        }
                    });
                    Ok(Event::default()
                        .data(error_sse.to_string())
                        .event("error"))
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
            ctx.loggers.write_line(&ctx.path, &DebugLogEntry {
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
            });
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
            Ok(Event::default()
                .data(error_sse.to_string())
                .event("error"))
        });
        // Keep `guard` alive until the stream is consumed (just one event).
        let stream = StreamWithGuard::new(Box::pin(stream), guard);
        return Ok(Box::pin(stream));
    }

    // Success — forward the byte stream as SSE events.
    let byte_stream = resp.bytes_stream();
    let sse_stream = SseForwarder::new(byte_stream, guard, model_name, instance_id, debug, manager, api_user, request_id, idle_timeout);

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
    fn new(stream: S, guard: SlotGuard, model_name: String, instance_id: String, debug: Option<DebugStreamContext>, manager: Arc<InstanceManager>, api_user: String, request_id: String, idle_timeout: std::time::Duration) -> Self {
        debug!("SseForwarder::new: model={} inst={}", model_name, instance_id);
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
                    let error_event = Event::default()
                        .data(error_sse.to_string())
                        .event("error");
                    return Poll::Ready(Some(Ok(error_event)));
                }
                Poll::Ready(None) => {
                    debug!("SseForwarder::poll_next: stream ended model={} inst={}", self.model_name, self.instance_id);
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
        debug!("SseForwarder::drop: model={} inst={}", self.model_name, self.instance_id);
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
                        prompt_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                        gen_tokens = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
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
            ctx.loggers.write_line(&ctx.path, &DebugLogEntry {
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
            });
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
        let (mgr, _release_rx, _crash_rx) =
            InstanceManager::new(&config, gpu_snapshot, None);
        Arc::new(mgr)
    }

    fn test_guard() -> SlotGuard {
        let mut inst = Instance::new("m", vec![], 9999, None);
        inst.state = InstanceState::Ready;
        let handle = InstanceHandle::new(inst);
        handle.try_acquire(4).expect("slot available")
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
        .chain(futures_util::stream::pending::<Result<bytes::Bytes, reqwest::Error>>());

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
        assert!(stream.next().await.is_none(), "stream must end after the idle error");
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
