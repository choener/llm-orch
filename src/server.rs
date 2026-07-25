// ── HTTP server ──────────────────────────────────────────────────────────────
//
// Sets up the axum router with tracing, compression, API key auth, and the
// OpenAI-compatible endpoints.

use crate::apikeys::ApikeysStore;
use crate::config::Config;
use crate::debug_log::DebugLoggers;
use crate::gpu::GpuMetrics;
use crate::handlers;
use crate::scheduler::InstanceManager;

use axum::{
    extract::{DefaultBodyLimit, FromRequestParts},
    http::{header, request::Parts, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use reqwest::Client;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::compression::CompressionLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

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
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/rerank", post(handlers::rerank))
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

/// Unified error type for HTTP responses.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("unauthorized")]
    Unauthorized,

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("model blocked: {0}")]
    ModelBlocked(String),

    #[error("no capacity: {0}")]
    NoCapacity(String),

    #[error("model unavailable: {0}")]
    ModelUnavailable(String),

    #[error("backend timeout: {0}")]
    BackendTimeout(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, self.to_string()),
            ApiError::ModelNotFound(_) => (StatusCode::NOT_FOUND, self.to_string()),
            ApiError::ModelBlocked(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            ApiError::ModelUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            ApiError::NoCapacity(_) => (StatusCode::TOO_MANY_REQUESTS, self.to_string()),
            ApiError::BackendTimeout(_) => (StatusCode::GATEWAY_TIMEOUT, self.to_string()),
            ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };
        let mut resp = (status, message).into_response();
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
