// ── HTTP server ──────────────────────────────────────────────────────────────
//
// Sets up the axum router with tracing, compression, API key auth, and the
// OpenAI-compatible endpoints.

use crate::apikeys::ApikeysStore;
use crate::config::Config;
use crate::debug_log::DebugLoggers;
use crate::gpu::GpuMetrics;
use crate::handlers;
use crate::peers::PeerRegistry;
use crate::scheduler::InstanceManager;

use axum::{
    Router,
    extract::{DefaultBodyLimit, FromRequestParts},
    http::{StatusCode, header, request::Parts},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use reqwest::Client;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::compression::CompressionLayer;
use tower_http::trace::TraceLayer;
use tracing::{debug, info, warn};

// ── Application state ───────────────────────────────────────────────────────

/// Shared application state, cloned per-request by axum.
#[derive(Clone)]
pub struct AppState {
    /// Hot-reloaded config.
    pub config: Arc<RwLock<Config>>,
    /// Hot-reloaded API keys.
    pub apikeys: Arc<RwLock<ApikeysStore>>,
    /// Instance manager (model lifecycle).
    pub manager: Arc<InstanceManager>,
    /// Shared HTTP client for forwarding requests to backends.
    pub client: Client,
    /// Latest GPU metrics snapshot (updated periodically).
    pub gpu: Arc<RwLock<Vec<GpuMetrics>>>,
    /// Per-model debug log writers (JSONL).
    pub debug_loggers: Arc<DebugLoggers>,
    /// Token / model statistics store (disabled unless configured).
    pub stats: Arc<crate::stats::StatsStore>,
    /// Cluster peer registry (empty / inert unless `peers:` is
    /// configured).  See `docs/006-cluster-peering.md`.
    pub peers: Arc<PeerRegistry>,
}

// ── Router ───────────────────────────────────────────────────────────────────

/// Build the full axum router with all middleware and routes.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Health check — no auth required.
        .route("/health", get(health))
        // OpenAI-compatible endpoints.
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/info", get(handlers::info_endpoint))
        .route("/v1/stats", get(handlers::stats_endpoint))
        .route("/v1/requests/{request_id}", get(handlers::request_status))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/responses", post(handlers::responses))
        .route(
            "/v1/responses/input_tokens",
            post(handlers::responses_input_tokens),
        )
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/rerank", post(handlers::rerank))
        // TypeSafe-compatible decision-model endpoint (llama.cpp `decisions`
        // models, halogen `decider`/`d3`).  Auth-gated like every /v1 route.
        .route("/v1/systemone", post(handlers::systemone))
        // OpenAI-compatible audio endpoints (audio.cpp backends).
        .route("/v1/audio/speech", post(handlers::audio_speech))
        .route(
            "/v1/audio/transcriptions",
            post(handlers::audio_transcriptions),
        )
        .route("/v1/audio/voices", get(handlers::audio_voices))
        // OpenAI-compatible image/video endpoints (stable-diffusion.cpp
        // backends).  These are registered unconditionally — passthrough
        // routing (sdapi/sdcpp) only activates for operative deployment
        // configurations, and model-resolution failures surface as 4xx.
        .route("/v1/images/generations", post(handlers::images_generations))
        .route("/v1/images/edits", post(handlers::images_edits))
        .route("/v1/video/generations", post(handlers::video_generations))
        // Native sd-server API passthrough (WebUI-compatible + sdcpp native).
        // Path parameters capture the remainder.  GET methods forward
        // to sd-server too (capabilities, job polling).
        .route("/sdapi/v1/{*path}", post(handlers::passthrough_sdapi))
        .route("/sdcpp/v1/{*path}", post(handlers::passthrough_sdcpp))
        .route("/sdcpp/v1/{*path}", get(handlers::passthrough_sdcpp_get))
        // Admin endpoints.
        .route("/admin/status", get(handlers::admin_status))
        .route("/admin/load", post(handlers::admin_load))
        .route("/admin/unload", post(handlers::admin_unload))
        .route("/admin/unblock", post(handlers::admin_unblock))
        // Large-context chat requests legitimately exceed axum's default
        // 2 MB body limit — raise it explicitly instead of 413-ing them.
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Start the HTTP server with graceful shutdown support.
///
/// The server stops accepting new connections when `shutdown` is triggered
/// (oneshot sender dropped).  In-flight requests are allowed to drain before
/// the function returns.
///
/// Returns `Err` on an invalid `server.listen` address, bind failure, or a
/// server error — the caller (main) treats an early exit as fatal instead
/// of idling forever without a listener.
pub async fn serve(
    state: AppState,
    shutdown: tokio::sync::oneshot::Receiver<()>,
) -> Result<(), String> {
    let addr: SocketAddr = state
        .config
        .read()
        .await
        .server
        .listen
        .parse()
        .map_err(|e| format!("invalid server.listen address: {}", e))?;

    let router = build_router(state);

    info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("failed to bind {}: {}", addr, e))?;
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            shutdown.await.ok();
        })
        .await
        .map_err(|e| format!("http server error: {}", e))
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// `GET /health` — always returns 200 OK.
async fn health() -> StatusCode {
    StatusCode::OK
}

// ── Error type ───────────────────────────────────────────────────────────────

/// Prefix prepended to every error message llm-orch generates itself.
///
/// Requests typically traverse a chain of services (reverse proxy, SDK,
/// frontend).  Without attribution it is impossible to tell whether an error
/// came from llm-orch or from one of the backends behind it, so *every*
/// client-visible message produced here is prefixed with `llm-orch: `.
/// Messages forwarded verbatim from a backend are left untouched.
pub const ERROR_PREFIX: &str = "llm-orch: ";

/// Format an llm-orch-generated error message with [`ERROR_PREFIX`].
pub fn err_msg(msg: impl std::fmt::Display) -> String {
    format!("{ERROR_PREFIX}{msg}")
}

/// Unified error type for HTTP responses.
///
/// The rendered message is prefixed with [`ERROR_PREFIX`]; the variants'
/// `Display` text below stays free of it.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("unauthorized")]
    Unauthorized,

    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("model blocked: {0}")]
    ModelBlocked(String),

    #[error("no capacity: {0}")]
    NoCapacity(String),

    #[error("queue wait timeout: {0}")]
    QueueTimeout(String),

    #[error("model unavailable: {0}")]
    ModelUnavailable(String),

    #[error("request not found: {0}")]
    RequestNotFound(String),

    #[error("backend timeout: {0}")]
    BackendTimeout(String),

    /// The backend was unreachable, answered with a non-success status, or
    /// broke mid-response.  Rendered as 502 so a broken upstream is never
    /// confused with an llm-orch bug (500).
    #[error("backend error: {0}")]
    BackendError(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self {
            ApiError::Unauthorized => StatusCode::UNAUTHORIZED,
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::ModelNotFound(_) => StatusCode::NOT_FOUND,
            ApiError::ModelBlocked(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::ModelUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::RequestNotFound(_) => StatusCode::NOT_FOUND,
            ApiError::NoCapacity(_) => StatusCode::TOO_MANY_REQUESTS,
            ApiError::QueueTimeout(_) => StatusCode::GATEWAY_TIMEOUT,
            ApiError::BackendTimeout(_) => StatusCode::GATEWAY_TIMEOUT,
            ApiError::BackendError(_) => StatusCode::BAD_GATEWAY,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        // The tower-http TraceLayer only reports the *status code*, so without
        // this the reason for a failed request never reaches the log — a 500 in
        // the journal carries no explanation at all.
        if status.is_server_error() {
            warn!(status = %status, error = %self, "request failed");
        } else if status.is_client_error() {
            debug!(status = %status, error = %self, "request rejected");
        }
        let mut resp = (status, err_msg(&self)).into_response();
        // 429s ask the client to retry later — say when (plan §5).
        if matches!(self, ApiError::NoCapacity(_)) {
            resp.headers_mut()
                .insert(header::RETRY_AFTER, header::HeaderValue::from_static("5"));
        }
        resp
    }
}

// ── Auth extractor ───────────────────────────────────────────────────────────

/// Extract and validate the `Authorization: Bearer <key>` header.
///
/// **Fail-closed**: if no API keys are configured, all requests are denied.
pub struct ApiKey {
    /// The label associated with this key (for logging).
    pub label: String,
}

impl FromRequestParts<AppState> for ApiKey {
    type Rejection = ApiError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let apikeys = state.apikeys.read().await;

        // Fail-closed: if no keys are configured, deny all access.
        if apikeys.is_empty() {
            return Err(ApiError::Unauthorized);
        }

        // Extract the Authorization header.
        let auth = parts
            .headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        // Expect "Bearer <key>"
        let key = auth
            .strip_prefix("Bearer ")
            .or_else(|| auth.strip_prefix("bearer "))
            .ok_or(ApiError::Unauthorized)?;

        // Look up the key.
        match apikeys.authenticate(key) {
            Some(label) => Ok(ApiKey {
                label: label.to_owned(),
            }),
            None => Err(ApiError::Unauthorized),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every error llm-orch returns must say so — a client behind a proxy
    /// chain cannot otherwise tell our errors from a backend's.
    #[tokio::test]
    async fn error_bodies_are_attributed_to_llm_orch() {
        let errors = [
            ApiError::Unauthorized,
            ApiError::BadRequest("missing model".into()),
            ApiError::ModelNotFound("gpt-x".into()),
            ApiError::ModelBlocked("gpt-x".into()),
            ApiError::NoCapacity("gpt-x".into()),
            ApiError::QueueTimeout("gpt-x".into()),
            ApiError::ModelUnavailable("gpt-x".into()),
            ApiError::RequestNotFound("req-1".into()),
            ApiError::BackendTimeout("gpt-x".into()),
            ApiError::BackendError("backend returned 500".into()),
            ApiError::Internal("boom".into()),
        ];
        for err in errors {
            let response = err.into_response();
            let body = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .expect("error body must be readable");
            let body = String::from_utf8_lossy(&body).into_owned();
            assert!(
                body.starts_with(ERROR_PREFIX),
                "body must start with {ERROR_PREFIX:?}, got: {body}"
            );
        }
    }

    /// A broken backend must be distinguishable from an llm-orch bug: the
    /// upstream's fault is a 502 (bad gateway), our own is a 500.
    #[test]
    fn backend_faults_are_bad_gateway_not_internal() {
        assert_eq!(
            ApiError::BackendError("backend returned 500".into())
                .into_response()
                .status(),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            ApiError::BackendTimeout("slow".into())
                .into_response()
                .status(),
            StatusCode::GATEWAY_TIMEOUT
        );
        assert_eq!(
            ApiError::Internal("bug".into()).into_response().status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    /// Every failed response must be logged with its reason: the tower-http
    /// trace layer only logs the status code, and a bare "500" in the journal
    /// is undiagnosable.
    #[test]
    fn server_error_responses_are_logged_with_their_reason() {
        let captured = crate::test_log::Capture::install();
        let err = ApiError::BackendError("backend returned 500: out of memory".into());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

        let lines = captured.lines();
        assert!(
            lines.iter().any(|l| l.contains("request failed")
                && l.contains("backend returned 500: out of memory")),
            "expected a warn log carrying the error reason, got: {lines:?}"
        );
    }
}
