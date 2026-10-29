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
    extract::{Path, Query, State},
    http::{HeaderMap, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures_util::stream::{self, Stream, StreamExt};
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tracing::{debug, info, warn};

use crate::config::{AliasPolicy, MakeRoomMode};
use crate::debug_log::{DebugLogEntry, DebugStreamContext, ts_now};
use crate::instance::{Liveness, SlotGuard};
use crate::scheduler::{AcquireError, Admission, InstanceManager, PendingAssignment};
use crate::server::{ApiError, ApiKey, AppState, err_msg};
use crate::types::*;
use tracing::Instrument;

// ── /v1/models ───────────────────────────────────────────────────────────────

/// `GET /v1/models` — list available models and aliases.
///
/// # Send safety
/// The `config` read guard is dropped before we build the response — no
/// `.await` while the guard is alive.
/// `/v1/models` capability advertisement for one model, mirroring
/// llama.cpp's `architecture` object.
///
/// A model declared `capabilities: [decisions]` is advertised as a native
/// decision model, so a client can find the System One lane *before*
/// anything is loaded — llm-orch synthesizes `/v1/models` from config, and
/// config is the only truth about an unloaded model.
fn architecture_for(m: &crate::config::ModelConfig) -> crate::types::Architecture {
    if m.is_decision_model() {
        crate::types::Architecture::text_and_decisions()
    } else {
        crate::types::Architecture::text()
    }
}

pub async fn list_models(
    State(state): State<AppState>,
    _key: ApiKey,
) -> Result<Response, ApiError> {
    let span = tracing::info_span!("list_models", id = %uuid::Uuid::new_v4().to_string());
    async {
        // ── 0. Cluster peering: refresh stale peer catalogs so the
        //    listing reflects the peers' current model sets (docs/006 —
        //    a stale `/v1/models` listing is a refresh trigger).
        state.peers.refresh_and_materialize().await;

        // ── 1. Read config, clone what we need, drop the guard. ──────────────
        let (models, aliases, node_name) = {
            let cfg = state.config.read().await;
            (
                cfg.models.clone(),
                cfg.aliases.clone(),
                cfg.node_name.clone(),
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
                architecture: architecture_for(m),
            })
            .collect();

        // Cluster peering: local models are also listed under their
        // node-qualified form (`<node>/<model>`).  Peer models are
        // already listed by their materialized names (`<peer>/<model>`
        // plus bare aliases) — the client-side provider prefix (e.g.
        // pi's `llm/`) never appears.
        if let Some(ref node) = node_name {
            let qualified: Vec<ModelEntry> = models
                .iter()
                .filter(|m| m.peer.is_none())
                .map(|m| ModelEntry {
                    id: format!("{node}/{}", m.name),
                    object: "model".into(),
                    created: now,
                    owned_by: "llm-orch".into(),
                    context_length: Some(m.context_length),
                    architecture: architecture_for(m),
                })
                .collect();
            data.extend(qualified);
        }

        // Aliases also appear as models.
        for a in &aliases {
            let target = a
                .targets
                .first()
                .and_then(|t| models.iter().find(|m| &m.name == t));
            let ctx = target.map(|m| m.context_length);
            data.push(ModelEntry {
                id: a.name.clone(),
                object: "model".into(),
                created: now,
                owned_by: "llm-orch".into(),
                context_length: ctx,
                // An alias can only reach its first target's kind of model,
                // so clients see the same capabilities through either name.
                architecture: target.map(architecture_for).unwrap_or_default(),
            });
        }

        // Cluster peering: announce this instance's node name so a
        // master's catalog fetch can filter our node-qualified self-listing
        // (otherwise it would re-export `cortana/cortana/qwen-flash` —
        // nested qualification, docs/006).
        let mut response = Json(ModelsResponse {
            object: "list".into(),
            data,
        })
        .into_response();
        if let Some(node) = node_name.as_deref()
            && let Ok(value) = header::HeaderValue::from_str(node)
        {
            response.headers_mut().insert("x-llm-orch-node", value);
        }
        Ok(response)
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
        let instance_counts = state.manager.instance_counts();

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
            // Cluster peering: skip materialized peer models here — the
            // owning leaf's own /v1/info is merged below (merge_peer_info)
            // and is authoritative for them.
            .filter(|m| m.peer.is_none())
            .map(|m| {
                let mets = metrics_snapshot.get(&m.name);
                let backend = model_backend_kind(m);
                ModelInfo {
                    name: m.name.clone(),
                    context_length: m.context_length,
                    instance_count: instance_counts.get(&m.name).copied().unwrap_or(0),
                    max_instances: m.max_instances,
                    queue_depth_used: state.manager.queue_depth(&m.name),
                    queue_depth_max: m.queue_depth,
                    queue: state.manager.queue_status(&m.name),
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
                    backend,
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
                    node: None,
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

        let merged = merge_peer_info(
            &state,
            InfoResponse {
                models: model_infos,
                aliases: alias_infos,
                gpus: gpu_statuses,
                metrics: state.manager.scheduler_metrics_snapshot(),
                peers: crate::peers::peer_status_snapshot(&state.peers),
            },
        )
        .await;
        Ok(Json(merged))
    }
    .instrument(span)
    .await
}

/// Merge the cluster peers' `/v1/info` into an `InfoResponse` (docs/006,
/// decision 9): peer models and aliases are prefixed with the peer's
/// node name, GPUs are labeled with the peer name, and scheduler metrics
/// stay local (they describe *this* scheduler).  The master's own entries
/// keep their bare names; local models additionally appear under their
/// node-qualified form when `node_name` is set.
async fn merge_peer_info(state: &AppState, mut response: InfoResponse) -> InfoResponse {
    let node_name = state.peers.node_name();
    // Self-qualification is a gateway feature (see stats_endpoint).
    let is_gateway = !state.peers.peer_names().is_empty();
    if is_gateway && let Some(node) = &node_name {
        let qualified: Vec<ModelInfo> = response
            .models
            .iter()
            .filter(|m| !m.name.contains('/'))
            .map(|m| {
                let mut q = m.clone();
                q.name = format!("{node}/{}", m.name);
                q
            })
            .collect();
        response.models.extend(qualified);
        let qualified_aliases: Vec<AliasInfo> = response
            .aliases
            .iter()
            .filter(|a| !a.name.contains('/'))
            .map(|a| {
                let mut q = a.clone();
                q.name = format!("{node}/{}", a.name);
                q
            })
            .collect();
        response.aliases.extend(qualified_aliases);
    }

    let peer_names = state.peers.peer_names();
    if peer_names.is_empty() {
        return response;
    }
    let fetched =
        futures_util::future::join_all(peer_names.iter().map(|p| state.peers.peer_info(p))).await;
    for (peer, info) in peer_names.iter().zip(fetched) {
        let Some(info) = info else {
            continue;
        };
        for mut model in info.models {
            model.name = format!("{peer}/{}", model.name);
            response.models.push(model);
        }
        for mut alias in info.aliases {
            alias.name = format!("{peer}/{}", alias.name);
            alias.targets = alias
                .targets
                .into_iter()
                .map(|t| format!("{peer}/{t}"))
                .collect();
            response.aliases.push(alias);
        }
        for mut gpu in info.gpus {
            gpu.node = Some(peer.clone());
            response.gpus.push(gpu);
        }
    }
    response
}

// ── /v1/stats ─────────────────────────────────────────────────────────────────

/// `GET /v1/stats` — token / model statistics, auth-gated.
///
/// Designed for external polling (e.g. Home Assistant's HTTP
/// integration): always returns 200 — `enabled: false` when stats are
/// not configured — so pollers never have to handle error states.
/// The response assembly lives in `stats::build_response` (shared with
/// the MQTT publisher).
pub async fn stats_endpoint(
    State(state): State<AppState>,
    _key: ApiKey,
) -> Json<crate::stats::StatsResponse> {
    Json(crate::stats::build_response(&state.stats, &state.gpu, &state.peers).await)
}

/// `GET /v1/requests/{request_id}` — authenticated request lifecycle status.
///
/// Cluster peering: forwarded requests are authoritative on the leaf —
/// when the request was routed to a peer, the leaf's status is proxied
/// (docs/006, decision 10).
pub async fn request_status(
    State(state): State<AppState>,
    _key: ApiKey,
    Path(request_id): Path<String>,
) -> Result<Json<RequestStatus>, ApiError> {
    if let Some(route) = state.manager.peer_request_route(&request_id)
        && let Some((peer_cfg, auth)) = state.peers.resolved_peer(&route.peer)
    {
        let url = format!(
            "{}/v1/requests/{}",
            peer_cfg.url.trim_end_matches('/'),
            route.leaf_id
        );
        let mut builder = state
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(10));
        if let Some(auth) = auth {
            builder = builder.header(header::AUTHORIZATION, auth);
        }
        match builder.send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(mut status) = resp.json::<RequestStatus>().await {
                    // Present the master's view: the client asked about
                    // its own request ID and master-side model names.
                    status.request_id = request_id.clone();
                    let upstream = state.manager.upstream_model(&route.model_name);
                    if status.logical_model == upstream.as_deref().unwrap_or("") {
                        status.logical_model = route.model_name.clone();
                    }
                    if let Some(resolved) = &status.resolved_model
                        && Some(resolved.as_str()) == upstream.as_deref()
                    {
                        status.resolved_model = Some(route.model_name.clone());
                    }
                    return Ok(Json(status));
                }
            }
            Ok(resp) => {
                // The leaf no longer knows the request (expired there or
                // restarted) — fall through to the local record.
                debug!(
                    peer = %route.peer,
                    status = %resp.status(),
                    "peer request status unavailable — falling back to local record"
                );
            }
            Err(e) => {
                warn!(peer = %route.peer, error = %e, "peer request status fetch failed");
            }
        }
    }
    state
        .manager
        .request_status(&request_id)
        .map(Json)
        .ok_or(ApiError::RequestNotFound(request_id))
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
    headers: HeaderMap,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!(
        "chat_completion",
        id = %request_id,
        model = %request.model,
    );
    let wait_events = wait_events_requested(&headers);
    async {
        // ── 1. Resolve alias → candidate models; extract the backend
        //    timeouts.  Read config, extract what we need, drop the guard.
        let request_model = request.model.clone();
        let (idle_timeout, total_timeout, heartbeat_secs) = {
            let cfg = state.config.read().await;
            (
                std::time::Duration::from_secs(cfg.server.backend_idle_timeout_secs),
                std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs),
                cfg.server.queue_heartbeat_secs,
            )
        }; // cfg read guard dropped — future is `Send`
        let (candidates, policy, make_room, drain_timeout) =
            resolve_candidates(&state, &request.model).await?;
        state.manager.register_request(&request_id, &request_model);

        debug!(
            request_model = %request_model,
            candidates = ?candidates,
            policy = ?policy,
            make_room = ?make_room,
            "resolved candidates"
        );

        // ── 2. Register admission before any slow spawn or migration.
        let admission = state
            .manager
            .admit(
                &candidates,
                policy,
                make_room,
                drain_timeout,
                Some(&request_id),
            )
            .await
            .map_err(|e| {
                acquire_error_for_request(&state.manager, &request_id, &request_model, e)
            })?;
        let (model_name, guard) = match admission {
            Admission::Assigned(model_name, guard) => {
                state.manager.update_request_state(
                    &request_id,
                    RequestState::Assigned,
                    Some(&model_name),
                    None,
                );
                (model_name, guard)
            }
            Admission::Waiting(pending) if request.stream => {
                state.manager.update_request_state(
                    &request_id,
                    RequestState::WaitingForSlot,
                    None,
                    None,
                );
                let context = WaitingSseContext {
                    state: state.clone(),
                    request_body: serde_json::to_value(&request).unwrap_or_default(),
                    requested_model: request_model.clone(),
                    backend_path: "/v1/chat/completions".into(),
                    idle_timeout,
                    heartbeat_secs,
                    api_user: key.label.clone(),
                    request_id: request_id.clone(),
                    alias: Some(request_model.clone()),
                    record_debug: true,
                    status_events: wait_events,
                };
                let stream = waiting_sse_stream(pending, context);
                return Ok(sse_response(stream, &request_id));
            }
            Admission::Waiting(pending) => {
                let assigned = pending
                    .wait()
                    .await
                    .map_err(|e| acquire_error(&request_model, e))?;
                state.manager.update_request_state(
                    &request_id,
                    RequestState::Assigned,
                    Some(&assigned.0),
                    None,
                );
                assigned
            }
        };

        // ── 3. Per-model debug log for the *chosen* model. ──────────────────
        let debug_log_path: Option<std::path::PathBuf> = {
            let cfg = state.config.read().await;
            cfg.models
                .iter()
                .find(|m| m.name == model_name)
                .and_then(|m| m.debug_log.clone())
        }; // cfg read guard dropped

        // ── 4. Capture instance ID, backend base, auth. ────────────────────
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        }; // inner lock dropped — `guard` itself is still alive

        let base = guard.handle().backend_base();
        let auth = state.manager.resolve_auth(guard.handle());

        let backend_url = format!("{base}/v1/chat/completions");

        // ── 5. Forward the request. ─────────────────────────────────────────
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
                auth.clone(),
                guard,
                model_name.clone(),
                instance_id.clone(),
                debug_ctx,
                state.manager.clone(),
                state.stats.clone(),
                key.label.clone(),
                request_id.clone(),
                idle_timeout,
                Some(Arc::clone(&state.peers)),
            )
            .await
            .map_err(|e| ApiError::BackendError(format!("backend request failed: {}", e)))?;

            Ok(sse_response(stream, &request_id))
        } else {
            // Non-streaming mode: aggregate the backend response.
            let t0 = std::time::Instant::now();
            let response_body = forward_request_aggregate(
                &state.client,
                &backend_url,
                &request_body,
                &auth,
                total_timeout,
                peer_forward(&state, &model_name, &request_id),
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

            // Statistics (opt-in).
            record_stats(
                &state,
                &model_name,
                response_body
                    .pointer("/usage/prompt_tokens")
                    .and_then(|v| v.as_u64()),
                response_body
                    .pointer("/usage/prompt_tokens_details/cached_tokens")
                    .and_then(|v| v.as_u64()),
                response_body
                    .pointer("/usage/completion_tokens")
                    .and_then(|v| v.as_u64()),
                elapsed.as_millis() as u64,
                None,
            );

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
            state.manager.update_request_state(
                &request_id,
                RequestState::Completed,
                Some(&model_name),
                None,
            );

            Ok(response_with_request_id(
                Json(response_body).into_response(),
                &request_id,
            ))
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
    headers: HeaderMap,
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
    let wait_events = wait_events_requested(&headers);
    async {
        // ── 1. Resolve alias → candidate models; extract the backend
        //    timeouts. ───────────────────────────────────────────────────────
        let (idle_timeout, total_timeout, heartbeat_secs) = {
            let cfg = state.config.read().await;
            (
                std::time::Duration::from_secs(cfg.server.backend_idle_timeout_secs),
                std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs),
                cfg.server.queue_heartbeat_secs,
            )
        }; // cfg read guard dropped — future is `Send`
        let (candidates, policy, make_room, drain_timeout) =
            resolve_candidates(&state, &requested_model).await?;
        state
            .manager
            .register_request(&request_id, &requested_model);

        debug!(
            request_model = %requested_model,
            candidates = ?candidates,
            policy = ?policy,
            make_room = ?make_room,
            "resolved candidates"
        );

        // ── 2. Register admission before any slow spawn or migration.
        let admission = state
            .manager
            .admit(
                &candidates,
                policy,
                make_room,
                drain_timeout,
                Some(&request_id),
            )
            .await
            .map_err(|e| {
                acquire_error_for_request(&state.manager, &request_id, &requested_model, e)
            })?;
        let (model_name, guard) = match admission {
            Admission::Assigned(model_name, guard) => {
                state.manager.update_request_state(
                    &request_id,
                    RequestState::Assigned,
                    Some(&model_name),
                    None,
                );
                (model_name, guard)
            }
            Admission::Waiting(pending) if stream_requested => {
                state.manager.update_request_state(
                    &request_id,
                    RequestState::WaitingForSlot,
                    None,
                    None,
                );
                let context = WaitingSseContext {
                    state: state.clone(),
                    request_body: request.clone(),
                    requested_model: requested_model.clone(),
                    backend_path: "/v1/responses".into(),
                    idle_timeout,
                    heartbeat_secs,
                    api_user: key.label.clone(),
                    request_id: request_id.clone(),
                    alias: Some(requested_model.clone()),
                    record_debug: true,
                    status_events: wait_events,
                };
                let stream = waiting_sse_stream(pending, context);
                return Ok(sse_response(stream, &request_id));
            }
            Admission::Waiting(pending) => {
                let assigned = pending
                    .wait()
                    .await
                    .map_err(|e| acquire_error(&requested_model, e))?;
                state.manager.update_request_state(
                    &request_id,
                    RequestState::Assigned,
                    Some(&assigned.0),
                    None,
                );
                assigned
            }
        };

        // ── 3. Per-model debug log for the *chosen* model. ──────────────────
        let debug_log_path: Option<std::path::PathBuf> = {
            let cfg = state.config.read().await;
            cfg.models
                .iter()
                .find(|m| m.name == model_name)
                .and_then(|m| m.debug_log.clone())
        }; // cfg read guard dropped

        // ── 4. Capture instance ID and port. ────────────────────────────────
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        }; // inner lock dropped — `guard` itself is still alive

        let base = guard.handle().backend_base();
        let auth = state.manager.resolve_auth(guard.handle());

        let backend_url = format!("{base}/v1/responses");

        // ── 5. Forward the *resolved* model name to the backend, not the
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
                auth.clone(),
                guard,
                model_name.clone(),
                instance_id.clone(),
                debug_ctx,
                state.manager.clone(),
                state.stats.clone(),
                key.label.clone(),
                request_id.clone(),
                idle_timeout,
                Some(Arc::clone(&state.peers)),
            )
            .await
            .map_err(|e| ApiError::BackendError(format!("backend request failed: {}", e)))?;

            Ok(sse_response(stream, &request_id))
        } else {
            // Non-streaming mode: aggregate the backend response.
            let t0 = std::time::Instant::now();
            let response_body = forward_request_aggregate(
                &state.client,
                &backend_url,
                &request,
                &auth,
                total_timeout,
                peer_forward(&state, &model_name, &request_id),
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

            // Statistics (opt-in).
            record_stats(
                &state,
                &model_name,
                response_body
                    .pointer("/usage/input_tokens")
                    .and_then(|v| v.as_u64()),
                response_body
                    .pointer("/usage/input_tokens_details/cached_tokens")
                    .and_then(|v| v.as_u64()),
                response_body
                    .pointer("/usage/output_tokens")
                    .and_then(|v| v.as_u64()),
                elapsed.as_millis() as u64,
                None,
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
            state.manager.update_request_state(
                &request_id,
                RequestState::Completed,
                Some(&model_name),
                None,
            );

            Ok(response_with_request_id(
                Json(response_body).into_response(),
                &request_id,
            ))
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
        // ── 1. Resolve alias → candidates; extract the backend timeout. ─────
        let (total_timeout,) = {
            let cfg = state.config.read().await;
            (std::time::Duration::from_secs(
                cfg.server.backend_total_timeout_secs,
            ),)
        };
        let (candidates, policy, make_room, drain_timeout) =
            resolve_candidates(&state, &requested_model).await?;

        // ── 2. Acquire an instance slot on one of the candidates (or queue).
        let (model_name, guard) = state
            .manager
            .acquire_for_candidates(&candidates, policy, make_room, drain_timeout)
            .await
            .map_err(|e| acquire_error(&requested_model, e))?;

        // ── 3. Per-model debug log for the *chosen* model. ──────────────────
        let debug_log_path: Option<std::path::PathBuf> = {
            let cfg = state.config.read().await;
            cfg.models
                .iter()
                .find(|m| m.name == model_name)
                .and_then(|m| m.debug_log.clone())
        };

        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };

        // ── 4. Read backend base + auth. ────────────────────────────────────
        let base = guard.handle().backend_base();
        let auth = state.manager.resolve_auth(guard.handle());

        let backend_url = format!("{base}/v1/responses/input_tokens");

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
        let response_body = forward_request_aggregate(
            &state.client,
            &backend_url,
            &request,
            &auth,
            total_timeout,
            peer_forward(&state, &model_name, &request_id),
        )
        .await?;
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
    headers: HeaderMap,
    Json(mut request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let span = tracing::info_span!(
        "completion",
        id = %request_id,
        model = %request.model,
    );
    let wait_events = wait_events_requested(&headers);
    async {
        // ── 1. Resolve alias → candidates; read the backend timeouts. ──────
        let (idle_timeout, total_timeout, heartbeat_secs) = {
            let cfg = state.config.read().await;
            (
                std::time::Duration::from_secs(cfg.server.backend_idle_timeout_secs),
                std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs),
                cfg.server.queue_heartbeat_secs,
            )
        };
        let (candidates, policy, make_room, drain_timeout) =
            resolve_candidates(&state, &request.model).await?;
        state.manager.register_request(&request_id, &request.model);

        // ── 2. Register admission before any slow spawn or migration.
        let admission = state
            .manager
            .admit(
                &candidates,
                policy,
                make_room,
                drain_timeout,
                Some(&request_id),
            )
            .await
            .map_err(|e| {
                acquire_error_for_request(&state.manager, &request_id, &request.model, e)
            })?;
        let (model_name, guard) = match admission {
            Admission::Assigned(model_name, guard) => {
                state.manager.update_request_state(
                    &request_id,
                    RequestState::Assigned,
                    Some(&model_name),
                    None,
                );
                (model_name, guard)
            }
            Admission::Waiting(pending) if request.stream => {
                state.manager.update_request_state(
                    &request_id,
                    RequestState::WaitingForSlot,
                    None,
                    None,
                );
                let context = WaitingSseContext {
                    state: state.clone(),
                    request_body: serde_json::to_value(&request).unwrap_or_default(),
                    requested_model: request.model.clone(),
                    backend_path: "/v1/completions".into(),
                    idle_timeout,
                    heartbeat_secs,
                    api_user: key.label.clone(),
                    request_id: request_id.clone(),
                    alias: Some(request.model.clone()),
                    record_debug: false,
                    status_events: wait_events,
                };
                let stream = waiting_sse_stream(pending, context);
                return Ok(sse_response(stream, &request_id));
            }
            Admission::Waiting(pending) => {
                let assigned = pending
                    .wait()
                    .await
                    .map_err(|e| acquire_error(&request.model, e))?;
                state.manager.update_request_state(
                    &request_id,
                    RequestState::Assigned,
                    Some(&assigned.0),
                    None,
                );
                assigned
            }
        };

        // Capture instance ID now that we hold the slot guard.
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };

        // ── 4. Read backend base + auth. ────────────────────────────────────
        let base = guard.handle().backend_base();
        let auth = state.manager.resolve_auth(guard.handle());

        let backend_url = format!("{base}/v1/completions");

        // ── 5. Forward request. ─────────────────────────────────────────────
        // Forward the *resolved* model name to the backend, not the alias.
        request.model = model_name.clone();
        if request.stream {
            info!("stream start model={} inst={}", model_name, instance_id);
            let stream = build_sse_stream(
                state.client.clone(),
                backend_url,
                serde_json::to_value(&request).unwrap_or_default(),
                auth.clone(),
                guard,
                model_name.clone(),
                instance_id.clone(),
                None,
                state.manager.clone(),
                state.stats.clone(),
                key.label.clone(),
                request_id.clone(),
                idle_timeout,
                Some(Arc::clone(&state.peers)),
            )
            .await
            .map_err(|e| ApiError::BackendError(format!("backend request failed: {}", e)))?;

            Ok(sse_response(stream, &request_id))
        } else {
            let t0 = std::time::Instant::now();
            let response_body = forward_request_aggregate(
                &state.client,
                &backend_url,
                &serde_json::to_value(&request).unwrap_or_default(),
                &auth,
                total_timeout,
                peer_forward(&state, &model_name, &request_id),
            )
            .await?;
            let elapsed = t0.elapsed();

            log_completion(&response_body, &model_name, &instance_id, elapsed);

            // Statistics (opt-in).
            record_stats(
                &state,
                &model_name,
                response_body
                    .pointer("/usage/prompt_tokens")
                    .and_then(|v| v.as_u64()),
                response_body
                    .pointer("/usage/prompt_tokens_details/cached_tokens")
                    .and_then(|v| v.as_u64()),
                response_body
                    .pointer("/usage/completion_tokens")
                    .and_then(|v| v.as_u64()),
                elapsed.as_millis() as u64,
                None,
            );

            state.manager.update_request_state(
                &request_id,
                RequestState::Completed,
                Some(&model_name),
                None,
            );

            Ok(response_with_request_id(
                Json(response_body).into_response(),
                &request_id,
            ))
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
        // ── 1. Resolve alias → candidates; read the backend timeout. ───────
        let (total_timeout,) = {
            let cfg = state.config.read().await;
            (std::time::Duration::from_secs(
                cfg.server.backend_total_timeout_secs,
            ),)
        };
        let (candidates, policy, make_room, drain_timeout) =
            resolve_candidates(&state, &request.model).await?;

        // ── 2. Acquire an instance slot on one of the candidates (or queue).
        let (model_name, guard) = state
            .manager
            .acquire_for_candidates(&candidates, policy, make_room, drain_timeout)
            .await
            .map_err(|e| acquire_error(&request.model, e))?;

        // Forward the *resolved* model name to the backend, not the alias.
        request.model = model_name.clone();

        // ── 4. Read backend base + auth. ────────────────────────────────────
        let base = guard.handle().backend_base();
        let auth = state.manager.resolve_auth(guard.handle());

        let backend_url = format!("{base}/v1/embeddings");

        // ── 5. Forward request, aggregate, release. ────────────────────────
        let t0 = std::time::Instant::now();
        let response_body = forward_request_aggregate(
            &state.client,
            &backend_url,
            &serde_json::to_value(&request).unwrap_or_default(),
            &auth,
            total_timeout,
            peer_forward(&state, &model_name, &request_id),
        )
        .await
        .map_err(|e| {
            warn!("embedding backend request failed: {}", e);
            e
        })?;
        let elapsed = t0.elapsed();
        // Captured before `response_body` is consumed by the parse below.
        let stats_prompt = response_body
            .pointer("/usage/prompt_tokens")
            .and_then(|v| v.as_u64());

        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };
        drop(guard);

        let resp: EmbeddingResponse = serde_json::from_value(response_body).map_err(|e| {
            ApiError::BackendError(format!("failed to parse backend embedding response: {}", e))
        })?;

        info!(
            "id={} model={} inst={} embeddings={} server_ms={}",
            request_id,
            model_name,
            instance_id,
            resp.data.len(),
            elapsed.as_millis()
        );

        // Statistics (opt-in): embeddings have no generation tokens.
        record_stats(
            &state,
            &model_name,
            stats_prompt,
            None,
            None,
            elapsed.as_millis() as u64,
            None,
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
        // ── 1. Resolve alias → candidates; extract the backend timeout. ─────
        let (total_timeout,) = {
            let cfg = state.config.read().await;
            (std::time::Duration::from_secs(
                cfg.server.backend_total_timeout_secs,
            ),)
        };
        let (candidates, policy, make_room, drain_timeout) =
            resolve_candidates(&state, &request.model).await?;

        let request_model = request.model.clone();

        // ── 2. Acquire an instance slot on one of the candidates (or queue).
        let (model_name, guard) = state
            .manager
            .acquire_for_candidates(&candidates, policy, make_room, drain_timeout)
            .await
            .map_err(|e| acquire_error(&request_model, e))?;

        // ── 3. Per-model debug log for the *chosen* model. ──────────────────
        let debug_log_path: Option<std::path::PathBuf> = {
            let cfg = state.config.read().await;
            cfg.models
                .iter()
                .find(|m| m.name == model_name)
                .and_then(|m| m.debug_log.clone())
        };

        // Forward the *resolved* model name to the backend, not the alias.
        request.model = model_name.clone();

        // Capture instance ID now that we hold the slot guard.
        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };

        // ── 4. Read backend base + auth. ────────────────────────────────────
        let base = guard.handle().backend_base();
        let auth = state.manager.resolve_auth(guard.handle());

        let backend_url = format!("{base}/v1/rerank");

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
        let response_body = forward_request_aggregate(
            &state.client,
            &backend_url,
            &request_body,
            &auth,
            total_timeout,
            peer_forward(&state, &model_name, &request_id),
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

        // Statistics (opt-in): reranking has no generation tokens.
        record_stats(
            &state,
            &model_name,
            response_body
                .pointer("/usage/prompt_tokens")
                .and_then(|v| v.as_u64()),
            None,
            None,
            elapsed.as_millis() as u64,
            None,
        );

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

// ── /v1/systemone ───────────────────────────────────────────────────────────

/// `POST /v1/systemone` — TypeSafe-compatible System One decision endpoint.
///
/// Decision models (llama.cpp's `decisions` models, halogen's `decider`/`d3`
/// models) answer typed questions about a `state` in one forward pass and
/// return a probability per option: no sampling, no generation.  Same alias
/// resolution and instance lifecycle as the other endpoints; always
/// non-streaming (streaming is not part of the upstream contract).
///
/// The body is opaque passthrough: only the top-level `model` field is read
/// (and rewritten to the resolved name), `state` / `questions` /
/// `files`/`images` are forwarded untouched, so the backend stays the
/// authority on question validation — it answers 400 on a malformed request
/// and 501 when the loaded model is not a decision model.  Those 4xx answers
/// are passed through with their status; 5xx still surfaces as 502.
/// Streaming is not part of the upstream contract, so it is not supported
/// here either.
///
/// Auth note: neither llama.cpp nor halogen requires a key of its own for
/// this endpoint, so llm-orch's API-key gate is the only thing in front of a
/// decision model here — the key label is recorded with the request.
pub async fn systemone(
    State(state): State<AppState>,
    key: ApiKey,
    Json(mut request): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    // `model` is the only field llm-orch interprets.
    let request_model = request
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::BadRequest("missing or empty \"model\" field".into()))?
        .to_owned();
    let span = tracing::info_span!(
        "systemone",
        id = %request_id,
        model = %request_model,
    );
    async {
        // ── 1. Resolve alias → candidates; read the backend timeout. ────────
        let total_timeout = {
            let cfg = state.config.read().await;
            std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs)
        };
        let (candidates, policy, make_room, drain_timeout) =
            resolve_candidates(&state, &request_model).await?;

        // ── 2. Acquire an instance slot on one of the candidates (or queue).
        let (model_name, guard) = state
            .manager
            .acquire_for_candidates(&candidates, policy, make_room, drain_timeout)
            .await
            .map_err(|e| acquire_error(&request_model, e))?;

        // ── 3. Per-model debug log for the *chosen* model. ──────────────────
        let debug_log_path: Option<std::path::PathBuf> = {
            let cfg = state.config.read().await;
            cfg.models
                .iter()
                .find(|m| m.name == model_name)
                .and_then(|m| m.debug_log.clone())
        };

        // Forward the *resolved* model name to the backend, not the alias.
        // (Peer forwarding rewrites it once more to the leaf-side name, and
        // restores the master-side name in the response, as elsewhere.)
        request["model"] = serde_json::Value::String(model_name.clone());

        let instance_id = {
            let inst = guard.handle().inner().lock().unwrap();
            inst.id.clone()
        };

        // ── 4. Read backend base + auth. ────────────────────────────────────
        let base = guard.handle().backend_base();
        let auth = state.manager.resolve_auth(guard.handle());

        let backend_url = format!("{base}/v1/systemone");

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
                    body: Some(request.clone()),
                    usage: None,
                    duration_ms: None,
                    error: None,
                },
            );
        }

        // ── 5. Forward request, aggregate, release. ─────────────────────────
        let t0 = std::time::Instant::now();
        let aggregated = forward_request_aggregate_keep_4xx(
            &state.client,
            &backend_url,
            &request,
            &auth,
            total_timeout,
            peer_forward(&state, &model_name, &request_id),
        )
        .await
        .map_err(|e| {
            warn!("systemone backend request failed: {}", e);
            e
        })?;
        let elapsed = t0.elapsed();
        let status = aggregated.status;
        let response_body = aggregated.value;
        let rejected = !status.is_success();

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
                    error: rejected.then(|| response_body.to_string()),
                },
            );
        }

        // A rejected decision (bad question, model that cannot decide) is
        // not a served request: it is passed back verbatim and never counted
        // into the token or completion statistics.
        if rejected {
            return Ok((status, Json(response_body)).into_response());
        }

        // Statistics (opt-in): a decision answers without generation, and
        // reports its token use as `usage.input_tokens`.
        record_stats(
            &state,
            &model_name,
            response_body
                .pointer("/usage/input_tokens")
                .and_then(|v| v.as_u64()),
            None,
            None,
            elapsed.as_millis() as u64,
            None,
        );

        // Record in per-model ring buffer for /admin/status.
        {
            let input_tokens = response_body
                .pointer("/usage/input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            state.manager.record_completion(
                &model_name,
                crate::types::CompletionRecord {
                    ts: ts_now(),
                    request_id: request_id.clone(),
                    instance_id: instance_id.clone(),
                    api_user: key.label.clone(),
                    prompt_tokens: input_tokens,
                    generated_tokens: 0,
                    cached_tokens: 0,
                    duration_ms: elapsed.as_millis() as u64,
                },
            );
        }

        let n_answers = response_body
            .pointer("/answers")
            .and_then(|v| v.as_object())
            .map(|m| m.len())
            .unwrap_or(0);

        info!(
            "id={} model={} inst={} answers={} server_ms={}",
            request_id,
            model_name,
            instance_id,
            n_answers,
            elapsed.as_millis()
        );

        Ok((status, Json(response_body)).into_response())
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
    /// Backend base (`http://127.0.0.1:{port}` or the remote base URL).
    base: String,
    /// Resolved `remote_auth` (remote models), if configured.
    auth: Option<String>,
    instance_id: String,
    guard: SlotGuard,
    idle_timeout: std::time::Duration,
    total_timeout: std::time::Duration,
    debug_log_path: Option<std::path::PathBuf>,
}

enum AudioAdmission {
    Assigned(AudioTarget),
    Waiting {
        pending: PendingAssignment,
        idle_timeout: std::time::Duration,
        total_timeout: std::time::Duration,
        heartbeat_secs: u64,
    },
}

/// Resolve `request_model` (alias → config model), verify it exists, and
/// acquire an instance slot (spawning on demand).  Lock-free across awaits.
async fn acquire_audio_target(
    state: &AppState,
    request_model: &str,
    request_id: &str,
) -> Result<AudioAdmission, ApiError> {
    let (idle_timeout, total_timeout, heartbeat_secs) = {
        let cfg = state.config.read().await;
        (
            std::time::Duration::from_secs(cfg.server.backend_idle_timeout_secs),
            std::time::Duration::from_secs(cfg.server.backend_total_timeout_secs),
            cfg.server.queue_heartbeat_secs,
        )
    };
    let (candidates, policy, make_room, drain_timeout) =
        resolve_candidates(state, request_model).await?;

    let admission = state
        .manager
        .admit(
            &candidates,
            policy,
            make_room,
            drain_timeout,
            Some(request_id),
        )
        .await
        .map_err(|e| acquire_error_for_request(&state.manager, request_id, request_model, e))?;
    match admission {
        Admission::Waiting(pending) => Ok(AudioAdmission::Waiting {
            pending,
            idle_timeout,
            total_timeout,
            heartbeat_secs,
        }),
        Admission::Assigned(model_name, guard) => Ok(AudioAdmission::Assigned(
            build_audio_target(
                state,
                request_model,
                model_name,
                guard,
                idle_timeout,
                total_timeout,
            )
            .await?,
        )),
    }
}

async fn build_audio_target(
    state: &AppState,
    request_model: &str,
    model_name: String,
    guard: SlotGuard,
    idle_timeout: std::time::Duration,
    total_timeout: std::time::Duration,
) -> Result<AudioTarget, ApiError> {
    let debug_log_path: Option<std::path::PathBuf> = {
        let cfg = state.config.read().await;
        cfg.models
            .iter()
            .find(|m| m.name == model_name)
            .and_then(|m| m.debug_log.clone())
    };
    let base = guard.handle().backend_base();
    let auth = state.manager.resolve_auth(guard.handle());
    let instance_id = {
        let inst = guard.handle().inner().lock().unwrap();
        inst.id.clone()
    };
    Ok(AudioTarget {
        model_name,
        request_model: request_model.to_owned(),
        base,
        auth,
        instance_id,
        guard,
        idle_timeout,
        total_timeout,
        debug_log_path,
    })
}

async fn finish_audio_admission(
    state: &AppState,
    request_model: &str,
    admission: AudioAdmission,
) -> Result<AudioTarget, ApiError> {
    match admission {
        AudioAdmission::Assigned(target) => Ok(target),
        AudioAdmission::Waiting {
            pending,
            idle_timeout,
            total_timeout,
            ..
        } => {
            let (model_name, guard) = pending
                .wait()
                .await
                .map_err(|e| acquire_error(request_model, e))?;
            build_audio_target(
                state,
                request_model,
                model_name,
                guard,
                idle_timeout,
                total_timeout,
            )
            .await
        }
    }
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
    let backend_url = format!("{}{}", target.base, path);

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

    // Cluster peering: rewrite the body's model to the leaf-side name
    // (JSON bodies only; multipart model fields were rewritten upstream).
    let body = rewrite_raw_model(state, &target.model_name, body, content_type);
    let send = with_auth(
        state
            .client
            .post(&backend_url)
            .header(header::CONTENT_TYPE, content_type)
            .body(body),
        &target.auth,
    )
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
    .map_err(|e| ApiError::BackendError(format!("backend request failed: {}", e)))?;

    let status = resp.status();
    note_peer_from_response(state, &target.model_name, request_id, &resp, status);
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
        return Err(ApiError::BackendError(format!(
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
            None, // peer model rewrite: audio chunks carry no model field
            state.manager.clone(),
            state.stats.clone(),
            api_user.to_owned(),
            request_id.to_owned(),
            target.idle_timeout,
        );
        return Ok(sse_response(Box::pin(stream), request_id));
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
    .map_err(|e| ApiError::BackendError(format!("failed to read backend response: {}", e)))?;

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
    headers: HeaderMap,
    Json(mut body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let request_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'model' field".into()))?
        .to_owned();
    let span = tracing::info_span!("audio_speech", id = %request_id, model = %request_model);
    let wait_events = wait_events_requested(&headers);
    async {
        let admission = acquire_audio_target(&state, &request_model, &request_id).await?;
        let wants_stream = body
            .get("stream")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let target = match admission {
            AudioAdmission::Waiting {
                pending,
                idle_timeout,
                heartbeat_secs,
                ..
            } if wants_stream => {
                let context = WaitingSseContext {
                    state: state.clone(),
                    request_body: body.clone(),
                    requested_model: request_model.clone(),
                    backend_path: "/v1/audio/speech".into(),
                    idle_timeout,
                    heartbeat_secs,
                    api_user: key.label.clone(),
                    request_id: request_id.clone(),
                    alias: Some(request_model.clone()),
                    record_debug: true,
                    status_events: wait_events,
                };
                return Ok(sse_response(
                    waiting_sse_stream(pending, context),
                    &request_id,
                ));
            }
            admission => finish_audio_admission(&state, &request_model, admission).await?,
        };
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
    let wait_events = wait_events_requested(&headers);

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
            let admission = acquire_audio_target(&state, &request_model, &request_id).await?;
            let target = finish_audio_admission(&state, &request_model, admission).await?;
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
        let admission = acquire_audio_target(&state, &request_model, &request_id).await?;
        let wants_stream = json
            .get("stream")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let target = match admission {
            AudioAdmission::Waiting {
                pending,
                idle_timeout,
                heartbeat_secs,
                ..
            } if wants_stream => {
                let context = WaitingSseContext {
                    state: state.clone(),
                    request_body: json.clone(),
                    requested_model: request_model.clone(),
                    backend_path: "/v1/audio/transcriptions".into(),
                    idle_timeout,
                    heartbeat_secs,
                    api_user: key.label.clone(),
                    request_id: request_id.clone(),
                    alias: Some(request_model.clone()),
                    record_debug: true,
                    status_events: wait_events,
                };
                return Ok(sse_response(
                    waiting_sse_stream(pending, context),
                    &request_id,
                ));
            }
            admission => finish_audio_admission(&state, &request_model, admission).await?,
        };
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
        let admission = acquire_audio_target(&state, &request_model, &request_id).await?;
        let target = finish_audio_admission(&state, &request_model, admission).await?;
        let backend_url = format!(
            "{}/v1/audio/voices?model={}",
            target.base,
            urlencoding_simple(&target.model_name),
        );
        let send = with_auth(state.client.get(&backend_url), &target.auth).send();
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
        .map_err(|e| ApiError::BackendError(format!("backend request failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_else(|_| "unknown error".into());
            return Err(ApiError::BackendError(format!(
                "backend returned {}: {}",
                status, text
            )));
        }
        let body = resp.json::<serde_json::Value>().await.map_err(|e| {
            ApiError::BackendError(format!("failed to parse backend response: {}", e))
        })?;
        Ok(Json(body).into_response())
    }
    .instrument(span)
    .await
}

// ── /v1/images/*, /v1/video/* — stable-diffusion.cpp backends ──────────────
//
// OpenAI-compatible image generation/editing endpoints and a custom video
// endpoint, all forwarding to sd-server instances.  Native sd-api and
// sdcpp endpoints are also proxied via passthrough.

/// Everything needed to forward one image/video request to a concrete instance.
struct DiffusionTarget {
    model_name: String,
    request_model: String,
    /// Backend base (`http://127.0.0.1:{port}` or the remote base URL).
    base: String,
    /// Resolved `remote_auth` (remote models), if configured.
    auth: Option<String>,
    instance_id: String,
    /// Keeps the instance slot held for the duration of the request —
    /// never read directly, but dropping it would free the instance mid-
    /// request.  Deliberately kept as a value field.
    #[allow(dead_code)]
    guard: SlotGuard,
    total_timeout: std::time::Duration,
    debug_log_path: Option<std::path::PathBuf>,
}

/// Resolve a diffusion model name (or alias) and acquire an instance slot.
///
/// Works exactly like `acquire_audio_target`: aliases are resolved, the
/// model is looked up in config, and an instance is spawned on demand.
async fn acquire_diffusion_target(
    state: &AppState,
    request_model: &str,
    request_id: &str,
) -> Result<DiffusionTarget, ApiError> {
    let (total_timeout,) = {
        let cfg = state.config.read().await;
        (std::time::Duration::from_secs(
            cfg.server.backend_total_timeout_secs,
        ),)
    };
    let (candidates, policy, make_room, drain_timeout) =
        resolve_candidates(state, request_model).await?;

    let admission = state
        .manager
        .admit(
            &candidates,
            policy,
            make_room,
            drain_timeout,
            Some(request_id),
        )
        .await
        .map_err(|e| acquire_error_for_request(&state.manager, request_id, request_model, e))?;

    let (model_name, guard) = match admission {
        Admission::Assigned(model_name, guard) => (model_name, guard),
        Admission::Waiting(pending) => {
            let (model_name, guard) = pending
                .wait()
                .await
                .map_err(|e| acquire_error(request_model, e))?;
            (model_name, guard)
        }
    };

    build_diffusion_target(state, request_model, model_name, guard, total_timeout).await
}

async fn build_diffusion_target(
    state: &AppState,
    request_model: &str,
    model_name: String,
    guard: SlotGuard,
    total_timeout: std::time::Duration,
) -> Result<DiffusionTarget, ApiError> {
    let debug_log_path: Option<std::path::PathBuf> = {
        let cfg = state.config.read().await;
        cfg.models
            .iter()
            .find(|m| m.name == model_name)
            .and_then(|m| m.debug_log.clone())
    };
    let base = guard.handle().backend_base();
    let auth = state.manager.resolve_auth(guard.handle());
    let instance_id = {
        let inst = guard.handle().inner().lock().unwrap();
        inst.id.clone()
    };
    Ok(DiffusionTarget {
        model_name,
        request_model: request_model.to_owned(),
        base,
        auth,
        instance_id,
        guard,
        total_timeout,
        debug_log_path,
    })
}

/// Aggregated reply from one sd-server backend call.
struct DiffusionResponse {
    status: axum::http::StatusCode,
    content_type: String,
    bytes: Bytes,
}

/// Send one diffusion request and aggregate the whole backend exchange.
///
/// `total_timeout` bounds response headers *and* body together: sd-server
/// only starts answering once generation finished, and a body that stalls
/// mid-transfer would otherwise hold the instance slot (and its VRAM) forever.
/// `total_timeout == 0` disables the bound.
///
/// Every failure is logged with the backend URL — the tower-http trace layer
/// only reports the status code, so an unlogged backend fault leaves nothing
/// but "500" in the journal — and maps to [`ApiError::BackendError`] (502),
/// keeping "the backend broke" distinct from "llm-orch is broken" (500).
async fn send_diffusion_request(
    state: &AppState,
    target: &DiffusionTarget,
    backend_url: &str,
    request_id: &str,
    default_content_type: &str,
    req: reqwest::RequestBuilder,
) -> Result<DiffusionResponse, ApiError> {
    let exchange = async {
        let resp = req
            .send()
            .await
            .map_err(|e| ApiError::BackendError(format!("backend request failed: {}", e)))?;
        let status = resp.status();
        note_peer_from_response(state, &target.model_name, request_id, &resp, status);
        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or(default_content_type)
            .to_string();
        // A torn body is an error, not an empty success: swallowing it turns a
        // dead backend into a silent 200 with no image (or a 500 that says
        // nothing).
        let bytes = resp.bytes().await.map_err(|e| {
            ApiError::BackendError(format!("failed to read backend response: {}", e))
        })?;
        Ok(DiffusionResponse {
            status,
            content_type,
            bytes,
        })
    };

    let result: Result<DiffusionResponse, ApiError> = if target.total_timeout.is_zero() {
        exchange.await
    } else {
        tokio::time::timeout(target.total_timeout, exchange)
            .await
            .map_err(|_| {
                ApiError::BackendTimeout(format!(
                    "backend did not respond within {}s",
                    target.total_timeout.as_secs()
                ))
            })?
    };

    match &result {
        Ok(ok) if ok.status.is_success() => {}
        Ok(ok) => warn!(
            url = %backend_url,
            inst = %target.instance_id,
            backend_status = %ok.status,
            error = %String::from_utf8_lossy(&ok.bytes),
            "diffusion backend returned non-success"
        ),
        Err(e) => {
            // Include whether the backend process is still there: "the port
            // stopped answering" and "the process exited" produce the same
            // reqwest error, and the remedies are completely different.
            warn!(
                url = %backend_url,
                inst = %target.instance_id,
                backend_process = %target.guard.handle().liveness(),
                error = %e,
                "diffusion backend request failed"
            );
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
                        error: Some(e.to_string()),
                    },
                );
            }
        }
    }

    result
}

/// Forward a JSON POST to an sd-server backend endpoint.
///
/// Sends the request body to `http://127.0.0.1:{port}{path}`, aggregates
/// the response (bounded by total_timeout), and returns it with the
/// backend's Content-Type preserved.
async fn forward_diffusion_json(
    state: &AppState,
    target: DiffusionTarget,
    path: &str,
    body: Vec<u8>,
    request_id: &str,
    _api_user: &str,
) -> Result<Response, ApiError> {
    let backend_url = format!("{}{}", target.base, path);

    // Debug log: request body as JSON
    if let Some(ref log_path) = target.debug_log_path {
        let debug_body: Option<serde_json::Value> = serde_json::from_slice(&body).ok();
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

    // Cluster peering: rewrite the body's model to the leaf-side name.
    let body = rewrite_raw_model(state, &target.model_name, body, "application/json");
    let resp = send_diffusion_request(
        state,
        &target,
        &backend_url,
        request_id,
        "application/json",
        with_auth(
            state
                .client
                .post(&backend_url)
                .header(header::CONTENT_TYPE, "application/json")
                .body(body),
            &target.auth,
        ),
    )
    .await?;
    let status = resp.status;
    let content_type = resp.content_type;
    let bytes = resp.bytes;

    // Debug log: response
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
                body: Some(serde_json::json!({
                    "status": status.as_u16(),
                    "bytes": bytes.len(),
                    "content_type": &content_type,
                })),
                usage: None,
                duration_ms: None,
                error: if status.is_success() {
                    None
                } else {
                    Some(format!(
                        "backend {}: {}",
                        status.as_u16(),
                        String::from_utf8_lossy(&bytes)
                    ))
                },
            },
        );
    }

    if !status.is_success() {
        return Err(ApiError::BackendError(format!(
            "backend returned {}: {}",
            status,
            String::from_utf8_lossy(&bytes)
        )));
    }

    let mut response = (status, bytes).into_response();
    if let Ok(ct) = header::HeaderValue::from_str(&content_type) {
        response.headers_mut().insert(header::CONTENT_TYPE, ct);
    }
    Ok(response)
}

/// `POST /v1/images/generations` — OpenAI-compatible text-to-image.
///
/// Accepts an `ImagesGenerationRequest` JSON body.  The `model` field
/// selects which diffusion backend to use.  Extra fields (lora,
/// sample_params, negative_prompt, etc.) are passed through verbatim to
/// sd-server's `/v1/images/generations`.
pub async fn images_generations(
    State(state): State<AppState>,
    key: ApiKey,
    Json(body): Json<ImagesGenerationRequest>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let request_model = body
        .model
        .as_deref()
        .ok_or_else(|| ApiError::BadRequest("missing 'model' field".into()))?
        .to_owned();
    let target = acquire_diffusion_target(&state, &request_model, &request_id).await?;

    // Build the body to forward to sd-server.
    // Start from the known fields, then merge in the extra fields.
    let mut forward_body = serde_json::json!({
        "prompt": body.prompt,
        "n": body.n,
        "output_format": body.output_format,
        "output_compression": body.output_compression,
    });
    if let Some(size) = &body.size {
        forward_body["size"] = serde_json::Value::String(size.clone());
    }
    // Pass through extra (extension) fields verbatim.
    for (k, v) in &body.extra {
        forward_body[k] = v.clone();
    }
    // Rewrite `model` to the resolved name for logging/debugging.
    forward_body["model"] = serde_json::Value::String(target.model_name.clone());

    let bytes = serde_json::to_vec(&forward_body)
        .map_err(|e| ApiError::BadRequest(format!("serialization error: {}", e)))?;

    forward_diffusion_json(
        &state,
        target,
        "/v1/images/generations",
        bytes,
        &request_id,
        &key.label,
    )
    .await
}

/// `POST /v1/images/edits` — OpenAI-compatible image editing.
///
/// sd-server expects multipart/form-data.  The `model` field is read from
/// the multipart body (or, as a fallback, from a JSON `params` field).
/// This handler forwards the multipart body to sd-server's `/v1/images/edits`
/// after (optionally) rewriting the `model` field to the resolved name.
pub async fn images_edits(
    State(state): State<AppState>,
    key: ApiKey,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();

    // Extract the multipart boundary and the model field.
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let boundary = multipart_boundary(content_type)
        .ok_or_else(|| ApiError::BadRequest("missing multipart boundary".into()))?;
    let request_model = extract_multipart_field(&body, &boundary, "model")
        .or_else(|| {
            // Fallback: try the inner `params` JSON.
            extract_multipart_field(&body, &boundary, "params").and_then(|p| {
                serde_json::from_str::<serde_json::Value>(&p)
                    .ok()
                    .and_then(|v| v.get("model").and_then(|m| m.as_str().map(String::from)))
            })
        })
        .ok_or_else(|| ApiError::BadRequest("missing 'model' field in multipart body".into()))?;

    let target = acquire_diffusion_target(&state, &request_model, &request_id).await?;

    // Rewrite the model field in the multipart body to the resolved name.
    let forwarded_body = rewrite_multipart_field(&body, &boundary, "model", &target.model_name);

    forward_diffusion_post_raw(
        &state,
        target,
        "/v1/images/edits",
        forwarded_body,
        content_type,
        &request_id,
        &key.label,
    )
    .await
}

/// Forward a raw POST (arbitrary Content-Type) to an sd-server backend.
///
/// Like `forward_diffusion_json` but preserves the caller's Content-Type
/// (needed for multipart uploads).
async fn forward_diffusion_post_raw(
    state: &AppState,
    target: DiffusionTarget,
    path: &str,
    body: Vec<u8>,
    content_type: &str,
    _request_id: &str,
    _api_user: &str,
) -> Result<Response, ApiError> {
    let backend_url = format!("{}{}", target.base, path);

    let body = rewrite_raw_model(state, &target.model_name, body, content_type);
    let resp = send_diffusion_request(
        state,
        &target,
        &backend_url,
        _request_id,
        "application/octet-stream",
        with_auth(
            state
                .client
                .post(&backend_url)
                .header(header::CONTENT_TYPE, content_type)
                .body(body),
            &target.auth,
        ),
    )
    .await?;
    let status = resp.status;
    let resp_ct = resp.content_type;
    let bytes = resp.bytes;

    if !status.is_success() {
        return Err(ApiError::BackendError(format!(
            "backend returned {}: {}",
            status,
            String::from_utf8_lossy(&bytes)
        )));
    }

    let mut response = (status, bytes).into_response();
    if let Ok(ct) = header::HeaderValue::from_str(&resp_ct) {
        response.headers_mut().insert(header::CONTENT_TYPE, ct);
    }
    Ok(response)
}

/// `POST /v1/video/generations` — llm-orch custom video generation.
///
/// Accepts a `VideoGenerationRequest` JSON body.  Forwards to sd-server's
/// `/sdcpp/v1/vid_gen` (async), then polls `/sdcpp/v1/jobs/{id}` until
/// completion.  On success, decodes the base64 container and returns raw
/// bytes with the correct Content-Type (`video/webm`, `image/webp`, or
/// `video/x-msvideo`).
///
/// The polling interval starts at 500 ms and doubles up to 5 s.
pub async fn video_generations(
    State(state): State<AppState>,
    _key: ApiKey,
    Json(body): Json<VideoGenerationRequest>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let request_model = body
        .model
        .as_deref()
        .ok_or_else(|| ApiError::BadRequest("missing 'model' field".into()))?
        .to_owned();
    let target = acquire_diffusion_target(&state, &request_model, &request_id).await?;

    // Build the body for sd-server's /sdcpp/v1/vid_gen.
    let mut forward_body = serde_json::json!({
        "prompt": body.prompt,
        "output_format": body.output_format,
    });
    if let Some(v) = body.video_frames {
        forward_body["video_frames"] = serde_json::json!(v);
    }
    if let Some(v) = body.fps {
        forward_body["fps"] = serde_json::json!(v);
    }
    if let Some(v) = body.width {
        forward_body["width"] = serde_json::json!(v);
    }
    if let Some(v) = body.height {
        forward_body["height"] = serde_json::json!(v);
    }
    if let Some(v) = body.seed {
        forward_body["seed"] = serde_json::json!(v);
    }
    // Pass through extra fields.
    for (k, v) in &body.extra {
        forward_body[k] = v.clone();
    }

    let bytes = serde_json::to_vec(&forward_body)
        .map_err(|e| ApiError::BadRequest(format!("serialization error: {}", e)))?;

    let backend_url = format!("{}/sdcpp/v1/vid_gen", target.base);

    // Submit the video job.
    let submit_resp = with_auth(
        state
            .client
            .post(&backend_url)
            .header(header::CONTENT_TYPE, "application/json")
            .body(bytes),
        &target.auth,
    )
    .send()
    .await
    .map_err(|e| {
        warn!(url = %backend_url, error = %e, "diffusion backend request failed");
        ApiError::BackendError(format!("backend request failed: {}", e))
    })?;

    let submit_status = submit_resp.status();
    let submit_body: serde_json::Value = submit_resp.json().await.map_err(|e| {
        warn!(url = %backend_url, error = %e, "failed to parse video job submit response");
        ApiError::BackendError(format!("failed to parse submit response: {}", e))
    })?;

    if !submit_status.is_success() && submit_status != 202 {
        warn!(url = %backend_url, backend_status = %submit_status, error = %submit_body, "video job submission failed");
        return Err(ApiError::BackendError(format!(
            "video job submission failed ({}): {}",
            submit_status, submit_body
        )));
    }

    // Extract the job ID.
    let job_id = submit_body["id"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| ApiError::BackendError("no job id in submit response".into()))?;

    // Poll until completed or timeout.
    let poll_base = format!("{}/sdcpp/v1/jobs/{}", target.base, job_id);
    let deadline = tokio::time::Instant::now() + target.total_timeout;
    let mut delay = std::time::Duration::from_millis(500);
    let mut poll_transport_failures: u32 = 0;

    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(ApiError::BackendTimeout(format!(
                "video generation did not complete within {}s",
                target.total_timeout.as_secs()
            )));
        }
        tokio::time::sleep(delay).await;
        delay = std::cmp::min(delay * 2, std::time::Duration::from_secs(5));

        let poll_resp = match with_auth(state.client.get(&poll_base), &target.auth)
            .send()
            .await
        {
            Ok(resp) => {
                if poll_transport_failures > 0 {
                    info!(
                        url = %poll_base,
                        inst = %target.instance_id,
                        failed_polls = poll_transport_failures,
                        "job poll is reachable again"
                    );
                    poll_transport_failures = 0;
                }
                resp
            }
            // A transport error means "no answer right now", not "the job
            // failed": sd-server can stop answering while it generates, and one
            // reset connection must not throw away minutes of GPU work.  Keep
            // polling — but stop at once when the process is gone, because the
            // job died with it and no retry can succeed.
            Err(e) => {
                poll_transport_failures += 1;
                if let Liveness::Gone(reason) = target.guard.handle().liveness() {
                    warn!(
                        url = %poll_base,
                        inst = %target.instance_id,
                        job = %job_id,
                        backend_process = %reason,
                        error = %e,
                        "video job poll failed and the backend process is gone"
                    );
                    return Err(ApiError::BackendError(format!(
                        "poll failed: {}; the backend {reason}, so job {job_id} can no longer complete",
                        e
                    )));
                }
                if poll_transport_failures == 1 {
                    warn!(
                        url = %poll_base,
                        inst = %target.instance_id,
                        job = %job_id,
                        error = %e,
                        "diffusion job poll failed; backend process is alive, continuing to poll"
                    );
                } else {
                    debug!(
                        url = %poll_base,
                        job = %job_id,
                        failed_polls = poll_transport_failures,
                        error = %e,
                        "diffusion job poll still unreachable"
                    );
                }
                continue;
            }
        };
        let poll_status = poll_resp.status();
        let poll_body: serde_json::Value = poll_resp.json().await.map_err(|e| {
            warn!(url = %poll_base, error = %e, "failed to parse poll response");
            ApiError::BackendError(format!("failed to parse poll response: {}", e))
        })?;

        if poll_status == 404 {
            warn!(url = %poll_base, "video job not found");
            return Err(ApiError::BackendError("video job not found".into()));
        }

        let job_status = poll_body["status"].as_str().unwrap_or("unknown");
        match job_status {
            "completed" => {
                // Extract the result payload.
                let result = poll_body["result"].as_object().ok_or_else(|| {
                    ApiError::BackendError("missing result in completed job".into())
                })?;
                let b64 = result["b64_json"]
                    .as_str()
                    .ok_or_else(|| ApiError::BackendError("missing b64_json in result".into()))?;
                let mime_type = result["mime_type"]
                    .as_str()
                    .unwrap_or("video/webm")
                    .to_owned();

                use base64::Engine;
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| ApiError::BackendError(format!("base64 decode error: {}", e)))?;

                let mut response = (decoded).into_response();
                if let Ok(ct) = header::HeaderValue::from_str(&mime_type) {
                    response.headers_mut().insert(header::CONTENT_TYPE, ct);
                }
                return Ok(response);
            }
            "failed" => {
                let err = poll_body["error"]
                    .as_object()
                    .map(|e| format!("{}", serde_json::json!(e)))
                    .unwrap_or_else(|| "unknown error".into());
                warn!(url = %poll_base, inst = %target.instance_id, error = %err, "video job failed");
                return Err(ApiError::BackendError(format!(
                    "video generation failed: {}",
                    err
                )));
            }
            "cancelled" => {
                warn!(url = %poll_base, inst = %target.instance_id, "video job cancelled");
                return Err(ApiError::BackendError("video generation cancelled".into()));
            }
            _ => {
                // queued or generating → continue polling
            }
        }
    }
}

/// Resolve a target for passthrough requests (sdapi / sdcpp native API).
///
/// Passthrough endpoints have no `model` field in their body.  The target
/// model is determined by (in order):
/// 1. The `X-LLM-Model` request header (for operator control).
/// 2. The config's `default_diffusion_model` server setting.
///
/// Returns an error if neither is available or the target model is unknown.
async fn acquire_passthrough_target(
    state: &AppState,
    headers: &HeaderMap,
    request_id: &str,
) -> Result<DiffusionTarget, ApiError> {
    let request_model = {
        // Check for explicit header override.
        if let Some(h) = headers
            .get("x-llm-model")
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty())
        {
            h.to_owned()
        } else {
            // Fall back to the configured default.
            let cfg = state.config.read().await;
            cfg.server.default_diffusion_model.clone().ok_or_else(|| {
                ApiError::BadRequest(
                    "no model specified for passthrough — set X-LLM-Model header or configure server.default_diffusion_model"
                        .into(),
                )
            })?
        }
    };

    acquire_diffusion_target(state, &request_model, request_id).await
}

/// `POST /sdapi/v1/*path` — passthrough to sd-server's WebUI-compatible API.
///
/// Forwards the request verbatim (same path, body, Content-Type) to the
/// target sd-server instance.  The model is resolved via
/// [`acquire_passthrough_target`].
pub async fn passthrough_sdapi(
    State(state): State<AppState>,
    key: ApiKey,
    headers: HeaderMap,
    Path(path): Path<String>,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let target = acquire_passthrough_target(&state, &headers, &request_id).await?;
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json");
    let path = format!("/sdapi/v1/{}", path);
    forward_diffusion_post_raw(
        &state,
        target,
        &path,
        body.to_vec(),
        content_type,
        &request_id,
        &key.label,
    )
    .await
}

/// `POST /sdcpp/v1/*path` — passthrough to sd-server's native sdcpp API.
///
/// Forwards the request verbatim.  The model is resolved via
/// [`acquire_passthrough_target`].
pub async fn passthrough_sdcpp(
    State(state): State<AppState>,
    key: ApiKey,
    headers: HeaderMap,
    Path(path): Path<String>,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let target = acquire_passthrough_target(&state, &headers, &request_id).await?;
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json");
    let path = format!("/sdcpp/v1/{}", path);
    forward_diffusion_post_raw(
        &state,
        target,
        &path,
        body.to_vec(),
        content_type,
        &request_id,
        &key.label,
    )
    .await
}

/// `GET /sdcpp/v1/*path` — passthrough for sdcpp GET endpoints
/// (e.g. `/sdcpp/v1/jobs/{id}`, `/sdcpp/v1/capabilities`).
pub async fn passthrough_sdcpp_get(
    State(state): State<AppState>,
    _key: ApiKey,
    headers: HeaderMap,
    Path(path): Path<String>,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let target = acquire_passthrough_target(&state, &headers, &request_id).await?;
    let backend_url = format!("{}/sdcpp/v1/{}", target.base, path);

    let resp = send_diffusion_request(
        &state,
        &target,
        &backend_url,
        &request_id,
        "application/json",
        with_auth(state.client.get(&backend_url), &target.auth),
    )
    .await?;
    let status = resp.status;
    let content_type = resp.content_type;
    let bytes = resp.bytes;

    if !status.is_success() {
        return Err(ApiError::BackendError(format!(
            "backend returned {}: {}",
            status,
            String::from_utf8_lossy(&bytes)
        )));
    }

    let mut response = (status, bytes).into_response();
    if let Ok(ct) = header::HeaderValue::from_str(&content_type) {
        response.headers_mut().insert(header::CONTENT_TYPE, ct);
    }
    Ok(response)
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
        // Cluster peering: forward to the owning leaf (docs/006, decision
        // 7) — the leaf spawns the real instance.
        if let Some(response) = forward_admin_action(&state, "/admin/load", &body).await? {
            return Ok(response);
        }

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
        // Cluster peering: forward to the owning leaf (docs/006, decision 7).
        if let Some(response) = forward_admin_action(&state, "/admin/unload", &body).await? {
            return Ok(response);
        }

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
        // Cluster peering: forward to the owning leaf (docs/006, decision 7).
        if let Some(response) = forward_admin_action(&state, "/admin/unblock", &body).await? {
            return Ok(response);
        }

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

        state.manager.unblock_model(&body.model, "admin unblock");
        Ok(Json(AdminResponse {
            status: "ok".into(),
            message: format!("model '{}' unblocked", body.model),
        }))
    }
    .instrument(span)
    .await
}

// ── Helpers (lock-free) ──────────────────────────────────────────────────────

/// Forward an admin model action to the peer that owns the model
/// (cluster peering, docs/006 decision 7).  Returns `Ok(None)` when the
/// model is local (the caller proceeds with the local code path) and
/// `Ok(Some(response))` with the leaf's answer when it is a peer model.
/// The leaf-side model name is rewritten into the body.
async fn forward_admin_action(
    state: &AppState,
    path: &str,
    body: &AdminModelAction,
) -> Result<Option<Json<AdminResponse>>, ApiError> {
    let Some(peer) = state.manager.peer_of_model(&body.model) else {
        return Ok(None);
    };
    let Some((peer_cfg, auth)) = state.peers.resolved_peer(&peer) else {
        return Err(ApiError::Internal(format!("peer '{peer}' not configured")));
    };
    let upstream = state.manager.backend_model_name(&body.model);
    let url = format!("{}{path}", peer_cfg.url.trim_end_matches('/'));
    let payload = serde_json::json!({ "model": upstream });
    let mut builder = state
        .client
        .post(&url)
        .json(&payload)
        .timeout(std::time::Duration::from_secs(30));
    if let Some(auth) = auth {
        builder = builder.header(header::AUTHORIZATION, auth);
    }
    let resp = builder
        .send()
        .await
        .map_err(|e| ApiError::BackendError(format!("peer request failed: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_else(|_| "unknown error".into());
        return Err(ApiError::BackendError(format!(
            "peer returned {status}: {text}"
        )));
    }
    let answer: AdminResponse = resp
        .json()
        .await
        .map_err(|e| ApiError::BackendError(format!("invalid peer response: {e}")))?;
    info!(peer = %peer, path = %path, model = %body.model, "admin action forwarded to peer");
    Ok(Some(Json(answer)))
}

/// Map an instance-acquisition failure to the HTTP error for this model.
fn acquire_error_for_request(
    manager: &InstanceManager,
    request_id: &str,
    model_name: &str,
    error: AcquireError,
) -> ApiError {
    // Make every rejected request self-explanatory in the journal: the
    // tower_http on_failure line carries only status code and latency, so
    // this is where the 503/429/504 diagnosis lives.
    warn!(
        request_id = %request_id,
        model = %model_name,
        error = ?error,
        blocked = manager.is_blocked(model_name),
        instances = manager.instance_counts().get(model_name).copied().unwrap_or(0),
        shutting_down = manager.is_shutting_down(),
        "request rejected"
    );
    manager.update_request_state(
        request_id,
        if matches!(error, AcquireError::QueueTimeout) {
            RequestState::Expired
        } else {
            RequestState::Failed
        },
        None,
        Some("request admission failed"),
    );
    acquire_error(model_name, error)
}

fn acquire_error(model_name: &str, e: AcquireError) -> ApiError {
    match e {
        AcquireError::Blocked => ApiError::ModelBlocked(model_name.to_owned()),
        AcquireError::NoCapacity => ApiError::NoCapacity(format!(
            "model '{}' is at capacity and the queue is full",
            model_name
        )),
        AcquireError::QueueTimeout => ApiError::QueueTimeout(format!(
            "model '{}' did not receive capacity before the queue wait deadline",
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
    cfg: &crate::config::Config,
    requested_model: &str,
) -> (Vec<String>, AliasPolicy, MakeRoomMode, u64) {
    // Cluster peering (docs/006): strip client-side path prefixes from
    // the requested name — our own node qualification and opaque client
    // provider prefixes (e.g. pi's `llm/`), while keeping `<peer>/...`
    // pins intact.
    let name =
        crate::peers::strip_client_prefixes(requested_model, cfg.node_name.as_deref(), &cfg.peers);
    if let Some(alias) = cfg.aliases.iter().find(|a| a.name == name) {
        return (
            alias.targets.clone(),
            alias.policy,
            alias.make_room,
            alias.drain_timeout,
        );
    }
    (vec![name], AliasPolicy::PreferLoaded, MakeRoomMode::None, 0)
}

/// Resolve a requested model into admission candidates, with the lazy
/// peer-catalog refresh from `docs/006-cluster-peering.md`: an unknown
/// name may be a new leaf model, so stale catalogs are refreshed (once,
/// singleflight) and the effective config re-materialized before the
/// request is rejected.
async fn resolve_candidates(
    state: &AppState,
    requested_model: &str,
) -> Result<(Vec<String>, AliasPolicy, MakeRoomMode, u64), ApiError> {
    let (candidates, policy, make_room, drain_timeout) = {
        let cfg = state.config.read().await;
        resolve_alias(&cfg, requested_model)
    }; // cfg read guard dropped — future is `Send`
    if !candidates.iter().any(|n| state.manager.has_model(n)) {
        state.peers.refresh_and_materialize().await;
        // Re-*resolve* (not just re-check): materialization may have
        // created a new bare-name alias for the requested model.
        let (candidates, policy, make_room, drain_timeout) = {
            let cfg = state.config.read().await;
            resolve_alias(&cfg, requested_model)
        };
        if !candidates.iter().any(|n| state.manager.has_model(n)) {
            return Err(ApiError::ModelNotFound(requested_model.to_string()));
        }
        return Ok((candidates, policy, make_room, drain_timeout));
    }
    Ok((candidates, policy, make_room, drain_timeout))
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

// ── Statistics ───────────────────────────────────────────────────────────────

/// Record one served request in the stats store.  Requests whose
/// backend reported no usage are still counted (requests/duration
/// only); the store warns about the missing usage.  No-op when stats
/// are not configured; the store is best-effort and never fails the
/// request.
fn record_stats(
    state: &AppState,
    model_name: &str,
    prompt_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    duration_ms: u64,
    ttft_ms: Option<u64>,
) {
    // Cluster peering: the master never counts forwarded requests into
    // its own per-model metrics — the leaf's numbers are authoritative
    // for its models (docs/006, decision 9; the union aggregates them
    // at query time).
    if state.manager.is_peer_model(model_name) {
        return;
    }
    // Every served request counts, even when the backend reported no
    // usage (some OpenAI-compatible providers omit it): the token
    // fields stay absent and the request/duration counters still
    // advance (the store warns about the missing usage).
    state.stats.record(
        model_name,
        prompt_tokens,
        cached_tokens,
        completion_tokens,
        duration_ms,
        ttft_ms,
    );
}

// ── Backend forwarding ───────────────────────────────────────────────────────

/// Attach the resolved `Authorization` header (remote models with
/// `remote_auth`) to a backend request builder.
fn with_auth(builder: reqwest::RequestBuilder, auth: &Option<String>) -> reqwest::RequestBuilder {
    match auth {
        Some(value) => builder.header(header::AUTHORIZATION, value),
        None => builder,
    }
}

// ── Cluster peering: forwarding boundary ─────────────────────────────────

/// Peer-forwarding context for a request routed to a cluster peer model
/// (see `docs/006-cluster-peering.md`).  `None` (from [`peer_forward`])
/// for local models — their forwarding path is byte-identical to today's.
pub(crate) struct PeerForward<'a> {
    pub state: &'a AppState,
    /// Master-side model name (the name the client used / the effective
    /// config's entry name).
    pub model_name: &'a str,
    pub request_id: &'a str,
}

/// Build the peer-forwarding context for a model, or `None` when the
/// model is local.
fn peer_forward<'a>(
    state: &'a AppState,
    model_name: &'a str,
    request_id: &'a str,
) -> Option<PeerForward<'a>> {
    state
        .manager
        .is_peer_model(model_name)
        .then_some(PeerForward {
            state,
            model_name,
            request_id,
        })
}

/// Rewrite a request body's top-level `model` field to the leaf-side
/// name (the only documented exception to opaque-body passthrough).
fn rewrite_request_model(
    body: &serde_json::Value,
    manager: &crate::scheduler::InstanceManager,
    model_name: &str,
) -> serde_json::Value {
    match manager.upstream_model(model_name) {
        Some(upstream) => {
            let mut rewritten = body.clone();
            rewritten["model"] = serde_json::Value::String(upstream);
            rewritten
        }
        None => body.clone(),
    }
}

/// Handle a leaf-side response at the forwarding boundary: record the
/// leaf's request ID (for `/v1/requests/{id}` proxying) from its
/// `x-llm-orch-request-id` header.
fn note_leaf_request(peer: &PeerForward<'_>, resp: &reqwest::Response) {
    if let Some(leaf_id) = resp
        .headers()
        .get("x-llm-orch-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        && let Some(peer_name) = peer.state.manager.peer_of_model(peer.model_name)
    {
        peer.state
            .manager
            .note_peer_request(peer.request_id, &peer_name, peer.model_name, leaf_id);
    }
}

/// A non-success leaf response whose status suggests the leaf's model
/// set drifted (404): schedule a background catalog refresh so
/// subsequent requests recover.  The current request still fails.
fn maybe_schedule_peer_refresh(peer: &PeerForward<'_>, status: reqwest::StatusCode) {
    if status == reqwest::StatusCode::NOT_FOUND
        && let Some(peer_name) = peer.state.manager.peer_of_model(peer.model_name)
    {
        peer.state.peers.schedule_refresh(&peer_name);
    }
}

/// Rewrite a raw (already-serialized) request body's top-level `model`
/// field to the leaf-side name — cluster peering, JSON bodies only.
/// Non-JSON bodies (multipart audio) are returned unchanged: their
/// model fields were rewritten before serialization.
fn rewrite_raw_model(
    state: &AppState,
    model_name: &str,
    body: Vec<u8>,
    content_type: &str,
) -> Vec<u8> {
    if !state.manager.is_peer_model(model_name) || !content_type.starts_with("application/json") {
        return body;
    }
    let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return body;
    };
    let upstream = state.manager.backend_model_name(model_name);
    value["model"] = serde_json::Value::String(upstream);
    serde_json::to_vec(&value).unwrap_or(body)
}

/// Record the leaf-side request ID (from its `x-llm-orch-request-id` response
/// header) and schedule a catalog refresh on 404 — the forwarding
/// boundary for the raw-body (audio / diffusion) helpers.
fn note_peer_from_response(
    state: &AppState,
    model_name: &str,
    request_id: &str,
    resp: &reqwest::Response,
    status: reqwest::StatusCode,
) {
    if !state.manager.is_peer_model(model_name) {
        return;
    }
    if let Some(leaf_id) = resp
        .headers()
        .get("x-llm-orch-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        && let Some(peer_name) = state.manager.peer_of_model(model_name)
    {
        state
            .manager
            .note_peer_request(request_id, &peer_name, model_name, leaf_id);
    }
    if status == reqwest::StatusCode::NOT_FOUND
        && let Some(peer_name) = state.manager.peer_of_model(model_name)
    {
        state.peers.schedule_refresh(&peer_name);
    }
}

/// Aggregated backend response: status plus parsed JSON body.
struct AggregatedResponse {
    status: reqwest::StatusCode,
    value: serde_json::Value,
}

/// Forward a request to a backend and aggregate the full response.
///
/// `total_timeout` bounds the whole exchange (request + full body): a
/// hung-but-alive backend must not tie up the in-flight slot and the
/// client request forever.  `Duration::ZERO` disables the timeout.
///
/// `peer` carries the cluster-peering forwarding context for peer
/// models: the body's `model` is rewritten to the leaf-side name, the
/// response's `model` back to the master-side name, and the leaf's
/// request ID recorded.
async fn forward_request_aggregate(
    client: &reqwest::Client,
    backend_url: &str,
    body: &serde_json::Value,
    auth: &Option<String>,
    total_timeout: std::time::Duration,
    peer: Option<PeerForward<'_>>,
) -> Result<serde_json::Value, ApiError> {
    Ok(
        aggregate(client, backend_url, body, auth, total_timeout, peer, false)
            .await?
            .value,
    )
}

/// Like [`forward_request_aggregate`], but a backend error that is the
/// backend *answering the request* — a 4xx, or 501 Not Implemented — is
/// returned to the caller verbatim instead of being flattened into a 502.
///
/// Endpoints whose request shape is validated by the backend rather than
/// by llm-orch (currently `/v1/systemone`) need this: a caller must be
/// able to tell "your question was malformed" (400) or "this model cannot
/// answer decisions" (501 — llama.cpp's answer for a model that is not a
/// decision model) apart from "the orchestrator or backend is broken"
/// (502, which every other 5xx still maps to).
async fn forward_request_aggregate_keep_4xx(
    client: &reqwest::Client,
    backend_url: &str,
    body: &serde_json::Value,
    auth: &Option<String>,
    total_timeout: std::time::Duration,
    peer: Option<PeerForward<'_>>,
) -> Result<AggregatedResponse, ApiError> {
    aggregate(client, backend_url, body, auth, total_timeout, peer, true).await
}

/// Shared body of the two aggregate forwarders.
///
/// `keep_client_errors` — pass 4xx and 501 responses through as-is (JSON
/// body required); every other 5xx becomes [`ApiError::BackendError`].
async fn aggregate(
    client: &reqwest::Client,
    backend_url: &str,
    body: &serde_json::Value,
    auth: &Option<String>,
    total_timeout: std::time::Duration,
    peer: Option<PeerForward<'_>>,
    keep_client_errors: bool,
) -> Result<AggregatedResponse, ApiError> {
    let work = async {
        let rewritten;
        let body = match &peer {
            Some(p) => {
                rewritten = rewrite_request_model(body, &p.state.manager, p.model_name);
                &rewritten
            }
            None => body,
        };
        let resp = with_auth(client.post(backend_url).json(body), auth)
            .send()
            .await
            .map_err(|e| ApiError::BackendError(format!("backend request failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            if let Some(p) = &peer {
                maybe_schedule_peer_refresh(p, status);
            }
            let text = resp.text().await.unwrap_or_else(|_| "unknown error".into());
            warn!(backend_status = %status, error = %text, "backend returned non-success");
            // A 4xx — or llama.cpp's 501 "this model is not a decision
            // model" — that carries a JSON body is a meaningful answer
            // from the backend, not an orchestrator fault: hand it back
            // as-is.
            let answerable =
                status.is_client_error() || status == reqwest::StatusCode::NOT_IMPLEMENTED;
            if keep_client_errors
                && answerable
                && let Ok(value) = serde_json::from_str::<serde_json::Value>(&text)
            {
                return Ok(AggregatedResponse { status, value });
            }
            return Err(ApiError::BackendError(format!(
                "backend returned {}: {}",
                status, text
            )));
        }

        if let Some(p) = &peer {
            note_leaf_request(p, &resp);
        }

        let mut value: serde_json::Value = resp.json().await.map_err(|e| {
            ApiError::BackendError(format!("failed to parse backend response: {}", e))
        })?;

        // Restore the master-side model name in the response body.
        if let Some(p) = &peer
            && value.get("model").is_some_and(serde_json::Value::is_string)
        {
            value["model"] = serde_json::Value::String(p.model_name.to_string());
        }
        Ok(AggregatedResponse { status, value })
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
#[allow(clippy::too_many_arguments)]
async fn build_sse_stream(
    client: reqwest::Client,
    backend_url: String,
    body: serde_json::Value,
    auth: Option<String>,
    guard: SlotGuard,
    model_name: String,
    instance_id: String,
    debug: Option<DebugStreamContext>,
    manager: Arc<InstanceManager>,
    stats: Arc<crate::stats::StatsStore>,
    api_user: String,
    request_id: String,
    idle_timeout: std::time::Duration,
    peers: Option<Arc<crate::peers::PeerRegistry>>,
) -> Result<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>, reqwest::Error> {
    // Cluster peering: rewrite the body's model to the leaf-side name
    // and remember the (upstream, master) pair for the per-chunk rewrite.
    let upstream = manager.upstream_model(&model_name);
    let model_rewrite = upstream.as_ref().map(|up| (up.clone(), model_name.clone()));
    let body = match &upstream {
        Some(up) => {
            let mut rewritten = body;
            rewritten["model"] = serde_json::Value::String(up.clone());
            rewritten
        }
        None => body,
    };
    // The idle timeout also bounds the wait for the first response
    // headers — a hung backend must not hold the slot while silent.
    let send = with_auth(client.post(&backend_url).json(&body), &auth).send();
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
                            "message": err_msg(format!(
                                "backend sent no response for {}s (idle timeout)",
                                idle_timeout.as_secs()
                            )),
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
        // Cluster peering: a 404 from a leaf means its model set
        // drifted — schedule a catalog refresh so subsequent requests
        // recover (this one still fails).
        if status == reqwest::StatusCode::NOT_FOUND
            && let Some(peers) = &peers
            && let Some(peer_name) = manager.peer_of_model(&model_name)
        {
            peers.schedule_refresh(&peer_name);
        }
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
                    "message": err_msg(format!("backend error: {}", error_text)),
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
    // Cluster peering: record the leaf's request ID for
    // /v1/requests/{id} proxying.
    if let Some(leaf_id) = resp
        .headers()
        .get("x-llm-orch-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        && let Some(peer_name) = manager.peer_of_model(&model_name)
    {
        manager.note_peer_request(&request_id, &peer_name, &model_name, leaf_id);
    }
    let byte_stream = resp.bytes_stream();
    let sse_stream = SseForwarder::new(
        byte_stream,
        guard,
        model_name,
        instance_id,
        debug,
        model_rewrite,
        manager,
        stats,
        api_user,
        request_id,
        idle_timeout,
    );

    Ok(Box::pin(sse_stream))
}

type EventStream = Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;
type AssignmentFuture =
    Pin<Box<dyn Future<Output = Result<(String, SlotGuard), AcquireError>> + Send>>;

/// Data needed to finish constructing an SSE backend stream after a queued
/// request receives a slot.
struct WaitingSseContext {
    state: AppState,
    request_body: serde_json::Value,
    requested_model: String,
    backend_path: String,
    idle_timeout: std::time::Duration,
    heartbeat_secs: u64,
    api_user: String,
    request_id: String,
    alias: Option<String>,
    record_debug: bool,
    status_events: bool,
}

/// State for the pre-assignment SSE response.  The assignment future remains
/// alive between comment heartbeats; once it resolves the stream transitions
/// through a connecting phase that keeps emitting heartbeats while the backend
/// HTTP connection is established — this prevents client timeouts during the
/// gap between the last queue heartbeat and the first backend SSE chunk.
enum WaitingSseState {
    Waiting {
        assignment: AssignmentFuture,
        context: Box<WaitingSseContext>,
        heartbeat: Pin<Box<tokio::time::Sleep>>,
    },
    Connecting {
        connect: Pin<Box<dyn Future<Output = EventStream> + Send>>,
        request_id: String,
        requested_model: String,
        heartbeat: Pin<Box<tokio::time::Sleep>>,
        heartbeat_secs: u64,
    },
    Forward(EventStream),
}

/// Build an SSE stream immediately after queue admission.  Before assignment
/// it emits only standard SSE comment frames; after assignment it splices the
/// normal OpenAI/backend stream without synthetic data chunks.
fn waiting_sse_stream(pending: PendingAssignment, context: WaitingSseContext) -> EventStream {
    let heartbeat_secs = context.heartbeat_secs.max(1);
    let assignment: AssignmentFuture = Box::pin(pending.wait());
    // Start with a zero-duration sleep so the first keep-alive comment
    // is emitted immediately — the client already received HTTP headers
    // and should not see an unnecessary silent gap before learning it is
    // queued.
    let heartbeat = Box::pin(tokio::time::sleep(std::time::Duration::ZERO));
    Box::pin(stream::unfold(
        WaitingSseState::Waiting {
            assignment,
            context: Box::new(context),
            heartbeat,
        },
        move |state| async move {
            match state {
                WaitingSseState::Waiting {
                    mut assignment,
                    context,
                    mut heartbeat,
                } => {
                    tokio::select! {
                        result = assignment.as_mut() => {
                            match result {
                                Ok((model_name, guard)) => {
                                    // Capture these before context is moved into the
                                    // connect future — the Connecting heartbeat still
                                    // needs them.
                                    let request_id = context.request_id.clone();
                                    let requested_model = context.requested_model.clone();
                                    let connect = Box::pin(build_assigned_sse_stream(
                                        *context, model_name, guard,
                                    ));
                                    let heartbeat: Pin<Box<tokio::time::Sleep>> = Box::pin(
                                        tokio::time::sleep(std::time::Duration::from_secs(
                                            heartbeat_secs,
                                        )),
                                    );
                                    Some((
                                        Ok(Event::default().comment(format!(
                                            "llm-orch slot assigned; request={}; model={}",
                                            request_id, requested_model,
                                        ))),
                                        WaitingSseState::Connecting {
                                            connect,
                                            request_id,
                                            requested_model,
                                            heartbeat,
                                            heartbeat_secs,
                                        },
                                    ))
                                }
                                Err(error) => {
                                    let state = if matches!(error, AcquireError::QueueTimeout) {
                                        RequestState::Expired
                                    } else {
                                        RequestState::Failed
                                    };
                                    context.state.manager.update_request_state(
                                        &context.request_id,
                                        state,
                                        None,
                                        Some("queued request could not be assigned"),
                                    );
                                    let error_stream =
                                        Box::pin(stream::once(async move {
                                            Ok(queue_wait_error_event(error))
                                        })) as EventStream;
                                    let mut next = error_stream;
                                    next.next()
                                        .await
                                        .map(|item| (item, WaitingSseState::Forward(next)))
                                }
                            }
                        }
                        _ = heartbeat.as_mut() => {
                            let event = if context.status_events {
                                Event::default().event("llm-orch.status").data(
                                    serde_json::json!({
                                        "version": 1,
                                        "state": "queued",
                                        "request_id": context.request_id,
                                        "model": context.requested_model,
                                    })
                                    .to_string(),
                                )
                            } else {
                                let comment = format!(
                                    "llm-orch waiting; request={}; model={}",
                                    context.request_id, context.requested_model
                                );
                                Event::default().comment(comment)
                            };
                            heartbeat.as_mut().reset(
                                tokio::time::Instant::now()
                                    + std::time::Duration::from_secs(heartbeat_secs),
                            );
                            Some((
                                Ok(event),
                                WaitingSseState::Waiting {
                                    assignment,
                                    context,
                                    heartbeat,
                                },
                            ))
                        }
                    }
                }
                WaitingSseState::Connecting {
                    mut connect,
                    request_id,
                    requested_model,
                    mut heartbeat,
                    heartbeat_secs,
                } => {
                    tokio::select! {
                        next_stream = connect.as_mut() => {
                            let mut next = next_stream;
                            next.next()
                                .await
                                .map(|item| (item, WaitingSseState::Forward(next)))
                        }
                        _ = heartbeat.as_mut() => {
                            let comment = format!(
                                "llm-orch connecting; request={}; model={}",
                                request_id, requested_model
                            );
                            heartbeat.as_mut().reset(
                                tokio::time::Instant::now()
                                    + std::time::Duration::from_secs(heartbeat_secs),
                            );
                            Some((
                                Ok(Event::default().comment(comment)),
                                WaitingSseState::Connecting {
                                    connect,
                                    request_id,
                                    requested_model,
                                    heartbeat,
                                    heartbeat_secs,
                                },
                            ))
                        }
                    }
                }
                WaitingSseState::Forward(mut stream) => stream
                    .next()
                    .await
                    .map(|item| (item, WaitingSseState::Forward(stream))),
            }
        },
    ))
}

fn queue_wait_error_event(error: AcquireError) -> Event {
    Event::default().event("error").data(
        serde_json::json!({
            "error": {
                "message": err_msg(format!("request could not be assigned: {error:?}")),
                "type": "queue_error"
            }
        })
        .to_string(),
    )
}

fn wait_events_requested(headers: &HeaderMap) -> bool {
    headers
        .get("x-llm-orch-wait-events")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.trim() == "1")
}

/// Construct an SSE response with transport headers that keep intermediaries
/// from caching or buffering the pre-assignment heartbeat stream.
fn response_with_request_id(mut response: Response, request_id: &str) -> Response {
    if let Ok(value) = header::HeaderValue::from_str(request_id) {
        response.headers_mut().insert(
            header::HeaderName::from_static("x-llm-orch-request-id"),
            value,
        );
    }
    response
}

fn sse_response(stream: EventStream, request_id: &str) -> Response {
    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response();
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    response.headers_mut().insert(
        header::HeaderName::from_static("x-accel-buffering"),
        header::HeaderValue::from_static("no"),
    );
    if let Ok(value) = header::HeaderValue::from_str(request_id) {
        response.headers_mut().insert(
            header::HeaderName::from_static("x-llm-orch-request-id"),
            value,
        );
    }
    response
}

async fn build_assigned_sse_stream(
    context: WaitingSseContext,
    model_name: String,
    guard: SlotGuard,
) -> EventStream {
    let WaitingSseContext {
        state,
        mut request_body,
        requested_model: _,
        backend_path,
        idle_timeout,
        api_user,
        request_id,
        alias,
        record_debug,
        status_events: _,
        heartbeat_secs: _,
    } = context;
    request_body["model"] = serde_json::Value::String(model_name.clone());
    state
        .manager
        .update_request_state(&request_id, RequestState::Running, Some(&model_name), None);

    let debug_log_path = if record_debug {
        let cfg = state.config.read().await;
        cfg.models
            .iter()
            .find(|model| model.name == model_name)
            .and_then(|model| model.debug_log.clone())
    } else {
        None
    };
    let base = guard.handle().backend_base();
    let auth = state.manager.resolve_auth(guard.handle());
    let instance_id = {
        let instance = guard.handle().inner().lock().unwrap();
        instance.id.clone()
    };
    if let Some(ref path) = debug_log_path {
        state.debug_loggers.write_line(
            path,
            &DebugLogEntry {
                ts: ts_now(),
                request_id: request_id.clone(),
                model: model_name.clone(),
                alias: alias.clone(),
                instance_id: Some(instance_id.clone()),
                dir: "request".into(),
                stream: Some(true),
                body: Some(request_body.clone()),
                usage: None,
                duration_ms: None,
                error: None,
            },
        );
    }
    let backend_url = format!("{base}{backend_path}");
    let debug = debug_log_path.map(|path| DebugStreamContext {
        loggers: state.debug_loggers.clone(),
        path,
        request_id: request_id.clone(),
        model_name: model_name.clone(),
        alias,
        instance_id: instance_id.clone(),
        t0: std::time::Instant::now(),
    });
    match build_sse_stream(
        state.client.clone(),
        backend_url,
        request_body,
        auth,
        guard,
        model_name,
        instance_id,
        debug,
        state.manager,
        state.stats.clone(),
        api_user,
        request_id,
        idle_timeout,
        Some(Arc::clone(&state.peers)),
    )
    .await
    {
        Ok(stream) => stream,
        Err(error) => Box::pin(stream::once(async move {
            Ok(Event::default().event("error").data(
                serde_json::json!({
                    "error": {
                        "message": err_msg(format!("backend request failed: {error}")),
                        "type": "backend_error"
                    }
                })
                .to_string(),
            ))
        })),
    }
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
    /// Used to record token statistics on stream end (no-op when
    /// stats are not configured).
    stats: Arc<crate::stats::StatsStore>,
    model_name: String,
    api_user: String,
    request_id: String,
    t0: std::time::Instant,
    /// When the first non-empty SSE data chunk arrived (TTFT).
    first_chunk_at: Option<std::time::Instant>,
    /// Kept alive until the stream ends.
    _guard: SlotGuard,
    instance_id: String,
    /// Cluster peering: rewrite each data chunk's top-level `model` from
    /// the leaf-side name back to the master-side name.  `None` for local
    /// models — chunks pass through untouched.
    model_rewrite: Option<(String, String)>,
}

impl<S> SseForwarder<S>
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: S,
        guard: SlotGuard,
        model_name: String,
        instance_id: String,
        debug: Option<DebugStreamContext>,
        model_rewrite: Option<(String, String)>,
        manager: Arc<InstanceManager>,
        stats: Arc<crate::stats::StatsStore>,
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
            model_rewrite,
            manager,
            stats,
            model_name,
            api_user,
            request_id,
            t0: std::time::Instant::now(),
            first_chunk_at: None,
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
            let model_rewrite = self.model_rewrite.clone();
            if let Some((event, data)) = extract_sse_event(&mut self.buffer, model_rewrite.as_ref())
            {
                if data == "[DONE]" {
                    // The backend's own terminator — forwarded like any
                    // other event and remembered so the stream end is
                    // recognized as a clean completion.
                    self.saw_done = true;
                } else if !data.is_empty() {
                    if self.first_chunk_at.is_none() {
                        // TTFT: first non-empty data chunk (a proxy for
                        // the first generated token).
                        self.first_chunk_at = Some(std::time::Instant::now());
                    }
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
                            "message": err_msg(format!("backend stream error: {}", e)),
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
                            "message": err_msg(
                                "backend stream ended without completing (truncated response)",
                            ),
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
                                "message": err_msg(format!(
                                    "backend sent no data for {}s (idle timeout)",
                                    idle_timeout.as_secs()
                                )),
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
            let mut prompt_opt: Option<u64> = None;
            let mut gen_opt: Option<u64> = None;
            let mut cached_opt: Option<u64> = None;
            for chunk in self.chunks.iter().rev() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(chunk)
                    && let Some(usage) = v.get("usage")
                {
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
                    // Statistics: accept both naming schemes (chat /
                    // completions and the Responses API).
                    prompt_opt = usage
                        .get("prompt_tokens")
                        .and_then(|v| v.as_u64())
                        .or_else(|| usage.get("input_tokens").and_then(|v| v.as_u64()));
                    gen_opt = usage
                        .get("completion_tokens")
                        .and_then(|v| v.as_u64())
                        .or_else(|| usage.get("output_tokens").and_then(|v| v.as_u64()));
                    cached_opt = usage
                        .pointer("/prompt_tokens_details/cached_tokens")
                        .and_then(|v| v.as_u64())
                        .or_else(|| {
                            usage
                                .pointer("/input_tokens_details/cached_tokens")
                                .and_then(|v| v.as_u64())
                        });
                    break;
                }
            }
            // Statistics (opt-in) — clean completions only, same rule
            // as the ring buffer above.  Recorded even when the stream
            // carried no usage chunk: the token fields stay absent and
            // only the request/duration counters advance.  Peer
            // models are skipped — the leaf's numbers are
            // authoritative (docs/006, decision 9; mirrors
            // `record_stats` for the non-streaming path).
            let ttft_ms = self
                .first_chunk_at
                .map(|t| t.saturating_duration_since(self.t0).as_millis() as u64);
            if !self.manager.is_peer_model(&self.model_name) {
                self.stats.record(
                    &self.model_name,
                    prompt_opt,
                    cached_opt,
                    gen_opt,
                    duration_ms,
                    ttft_ms,
                );
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
        self.manager.update_request_state(
            &self.request_id,
            if self.saw_done {
                RequestState::Completed
            } else {
                RequestState::Cancelled
            },
            Some(&self.model_name),
            None,
        );

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
/// Extract one complete SSE event from `buffer`.  When `model_rewrite`
/// is set (cluster peering), the data's top-level `model` field is
/// rewritten from the leaf-side name back to the master-side name —
/// chunks without a matching string `model` pass through byte-identical.
fn extract_sse_event(
    buffer: &mut Vec<u8>,
    model_rewrite: Option<&(String, String)>,
) -> Option<(Event, String)> {
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

    let mut data = data_lines.join("\n");
    if let Some((upstream, master)) = model_rewrite
        && data.starts_with('{')
        && data.contains("\"model\"")
        && let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&data)
        && value.get("model").and_then(|m| m.as_str()) == Some(upstream.as_str())
    {
        value["model"] = serde_json::Value::String(master.clone());
        data = value.to_string();
    }
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

/// Determine the backend kind string for a model based on its command
/// program basename.  Used by the `/v1/info` endpoint.
fn model_backend_kind(cfg: &crate::config::ModelConfig) -> String {
    let parts = shlex::split(&cfg.cmd).unwrap_or_default();
    let prog = parts.first().map(|s| s.as_str()).unwrap_or("");
    let basename = prog.rsplit('/').next().unwrap_or(prog);
    if basename.starts_with("sd-server") {
        "sd.cpp".to_string()
    } else {
        "llama.cpp".to_string()
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
            None,
            mgr,
            Arc::new(crate::stats::StatsStore::default()),
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
        assert!(
            events[1].contains("llm-orch: backend stream ended without completing"),
            "truncation error must be attributed to llm-orch: {}",
            events[1]
        );
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
            None,
            mgr.clone(),
            Arc::new(crate::stats::StatsStore::default()),
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
            event_text(&second).contains("llm-orch: backend sent no data"),
            "idle-timeout error must be attributed to llm-orch: {}",
            event_text(&second)
        );
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
            &None,
            std::time::Duration::from_millis(200),
            None,
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
