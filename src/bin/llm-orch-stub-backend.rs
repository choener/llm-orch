// ── Stub llama.cpp backend ───────────────────────────────────────────────────
//
// A minimal HTTP server that mimics the llama.cpp endpoints llm-orch uses,
// so integration tests can exercise the full spawn → forward → stream path
// without a real GPU or model.
//
// Endpoints:
//   GET  /health                  → 200 OK (readiness probe)
//   POST /v1/chat/completions     → canned completion, SSE-streamed or aggregated
//   POST /v1/embeddings           → fixed 3-dimensional embedding
//
// Flags control the canned response: number of chunks, per-chunk delay
// (slow streaming for drain/timeout tests).

use axum::{
    Json, Router,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use clap::Parser;
use futures_util::StreamExt;
use std::convert::Infallible;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "llm-orch-stub-backend")]
struct Cli {
    /// Port to listen on (llm-orch substitutes {port} in the model cmd).
    #[arg(long)]
    port: u16,

    /// Number of content chunks per streaming response.
    #[arg(long, default_value_t = 3)]
    chunks: usize,

    /// Delay between streamed chunks (simulates slow generation).
    #[arg(long, default_value_t = 0)]
    chunk_delay_ms: u64,
}

#[derive(Clone)]
struct StubConfig {
    chunks: usize,
    chunk_delay: Duration,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let cfg = StubConfig {
        chunks: cli.chunks,
        chunk_delay: Duration::from_millis(cli.chunk_delay_ms),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(cfg);

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", cli.port))
        .await
        .expect("bind failed");
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    "ok"
}

async fn chat_completions(
    axum::extract::State(cfg): axum::extract::State<StubConfig>,
    Json(body): Json<serde_json::Value>,
) -> axum::response::Response {
    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("stub")
        .to_string();

    if stream {
        // Canned SSE stream: N content chunks, a usage chunk, [DONE].
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        for i in 0..cfg.chunks {
            let chunk = serde_json::json!({
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": format!("tok-{i} ")},
                    "finish_reason": null
                }]
            });
            events.push(Ok(Event::default().data(chunk.to_string())));
        }
        let usage = serde_json::json!({
            "id": "chatcmpl-stub",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": cfg.chunks, "total_tokens": 5 + cfg.chunks}
        });
        events.push(Ok(Event::default().data(usage.to_string())));
        events.push(Ok(Event::default().data("[DONE]")));

        let delay = cfg.chunk_delay;
        let stream = futures_util::stream::iter(events).then(move |ev| async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            ev
        });
        Sse::new(stream).into_response()
    } else {
        let mut content = String::new();
        for i in 0..cfg.chunks {
            content.push_str(&format!("tok-{i} "));
        }
        Json(serde_json::json!({
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": cfg.chunks, "total_tokens": 5 + cfg.chunks}
        }))
        .into_response()
    }
}

async fn embeddings(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("stub")
        .to_string();
    Json(serde_json::json!({
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
        "model": model,
        "usage": {"prompt_tokens": 3, "total_tokens": 3}
    }))
}
