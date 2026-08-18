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
//   POST /v1/responses            → canned Responses API reply, SSE or aggregated
//   POST /v1/responses/input_tokens → fixed token count
//   POST /v1/audio/speech         → tiny WAV, base64 JSON, or speech SSE
//   POST /v1/audio/transcriptions → canned transcript (JSON + multipart), SSE optional
//   GET  /v1/audio/voices         → static voice list
//
// Flags control the canned response: number of chunks, per-chunk delay
// (slow streaming for drain/timeout tests).  audiocpp_server-style flags
// (--config, --host, --backend, …) are accepted and ignored so the stub
// can be spawned with an audio-style cmd line.

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

    // ── audiocpp_server compatibility flags (accepted, ignored) ─────────
    #[arg(long)]
    config: Option<String>,
    #[arg(long)]
    host: Option<String>,
    #[arg(long)]
    backend: Option<String>,
    #[arg(long)]
    device: Option<usize>,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long)]
    busy_timeout_ms: Option<u64>,
    #[arg(long)]
    model_spec_override: Option<String>,
    #[arg(long)]
    voice_dir: Option<String>,
    #[arg(long)]
    cors_origins: Option<String>,
    #[arg(long)]
    log_file: Option<String>,
    #[arg(long)]
    ui: bool,
    #[arg(long)]
    log: bool,
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
        .route("/v1/responses", post(responses))
        .route("/v1/responses/input_tokens", post(responses_input_tokens))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/audio/voices", get(audio_voices))
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

/// Canned Response object shared by the aggregated reply and the final
/// `response.completed` SSE event.
fn response_object(model: &str, chunks: usize) -> serde_json::Value {
    let mut text = String::new();
    for i in 0..chunks {
        text.push_str(&format!("tok-{i} "));
    }
    serde_json::json!({
        "id": "resp-stub",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [{
            "type": "message",
            "id": "msg-stub",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]
        }],
        "usage": {"input_tokens": 5, "output_tokens": chunks, "total_tokens": 5 + chunks}
    })
}

async fn responses(
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
        // Mimic the Responses SSE lifecycle: response.created → N ×
        // response.output_text.delta → response.completed (no [DONE] — the
        // real API terminates the stream after the completed event).
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        let created = serde_json::json!({
            "type": "response.created",
            "response": {"id": "resp-stub", "object": "response", "created_at": 1,
                         "status": "in_progress", "model": model, "output": []}
        });
        events.push(Ok(Event::default()
            .event("response.created")
            .data(created.to_string())));
        for i in 0..cfg.chunks {
            let delta = serde_json::json!({
                "type": "response.output_text.delta",
                "item_id": "msg-stub",
                "output_index": 0,
                "content_index": 0,
                "delta": format!("tok-{i} ")
            });
            events.push(Ok(Event::default()
                .event("response.output_text.delta")
                .data(delta.to_string())));
        }
        let completed = serde_json::json!({
            "type": "response.completed",
            "response": response_object(&model, cfg.chunks)
        });
        events.push(Ok(Event::default()
            .event("response.completed")
            .data(completed.to_string())));

        let delay = cfg.chunk_delay;
        let stream = futures_util::stream::iter(events).then(move |ev| async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            ev
        });
        Sse::new(stream).into_response()
    } else {
        Json(response_object(&model, cfg.chunks)).into_response()
    }
}

async fn responses_input_tokens(Json(_body): Json<serde_json::Value>) -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "response.input_tokens",
        "input_tokens": 5
    }))
}

// ── audio.cpp endpoints ──────────────────────────────────────────────────

/// A tiny valid WAV file: 44-byte PCM header + a short burst of silence.
fn stub_wav() -> Vec<u8> {
    let samples = 1600u32; // 0.1 s of 16 kHz mono s16le silence
    let data_len = samples * 2;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&16000u32.to_le_bytes()); // sample rate
    wav.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.resize(44 + data_len as usize, 0);
    wav
}

fn audio_model(body: &serde_json::Value) -> String {
    body.get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("stub-audio")
        .to_string()
}

async fn audio_speech(
    axum::extract::State(cfg): axum::extract::State<StubConfig>,
    Json(body): Json<serde_json::Value>,
) -> axum::response::Response {
    use base64::Engine as _;
    let model = audio_model(&body);
    let wav = stub_wav();

    let stream_format = body.get("stream_format").and_then(|v| v.as_str());
    let response_format = body.get("response_format").and_then(|v| v.as_str());

    if stream_format == Some("sse") {
        // speech.audio.delta × N (base64 PCM chunks) → speech.audio.done → [DONE]
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        for _ in 0..cfg.chunks {
            let delta = serde_json::json!({
                "type": "speech.audio.delta",
                "model": model,
                "audio": base64::engine::general_purpose::STANDARD.encode(&wav[..64]),
            });
            events.push(Ok(Event::default()
                .event("speech.audio.delta")
                .data(delta.to_string())));
        }
        let done = serde_json::json!({"type": "speech.audio.done", "model": model});
        events.push(Ok(Event::default()
            .event("speech.audio.done")
            .data(done.to_string())));
        events.push(Ok(Event::default().data("[DONE]")));
        let delay = cfg.chunk_delay;
        let stream = futures_util::stream::iter(events).then(move |ev| async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            ev
        });
        Sse::new(stream).into_response()
    } else if response_format == Some("json") {
        Json(serde_json::json!({
            "model": model,
            "audio": base64::engine::general_purpose::STANDARD.encode(&wav),
        }))
        .into_response()
    } else {
        ([(axum::http::header::CONTENT_TYPE, "audio/wav")], wav).into_response()
    }
}

fn transcript_json(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "text": "stub transcript: the quick brown fox",
    })
}

fn transcript_sse(model: String, chunks: usize, delay: Duration) -> axum::response::Response {
    let mut events: Vec<Result<Event, Infallible>> = Vec::new();
    for i in 0..chunks {
        let delta = serde_json::json!({
            "type": "transcript.text.delta",
            "model": model,
            "delta": format!("chunk-{i} "),
        });
        events.push(Ok(Event::default()
            .event("transcript.text.delta")
            .data(delta.to_string())));
    }
    let done = serde_json::json!({
        "type": "transcript.text.done",
        "model": model,
        "text": "stub transcript: the quick brown fox",
    });
    events.push(Ok(Event::default()
        .event("transcript.text.done")
        .data(done.to_string())));
    events.push(Ok(Event::default().data("[DONE]")));
    let stream = futures_util::stream::iter(events).then(move |ev| async move {
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        ev
    });
    Sse::new(stream).into_response()
}

async fn audio_transcriptions(
    axum::extract::State(cfg): axum::extract::State<StubConfig>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> axum::response::Response {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if content_type.starts_with("multipart/form-data") {
        // Minimal multipart field extraction: pull `model` and `stream`
        // out of the raw body (good enough for a test stub).
        let text = String::from_utf8_lossy(&body);
        let field = |name: &str| -> Option<String> {
            let needle = format!("name=\"{name}\"");
            let pos = text.find(&needle)?;
            let after = &text[pos + needle.len()..];
            let start = after.find("\r\n\r\n")? + 4;
            let end = after[start..].find("\r\n")? + start;
            Some(after[start..end].to_string())
        };
        let model = field("model").unwrap_or_else(|| "stub-audio".into());
        if field("stream").as_deref() == Some("true") {
            transcript_sse(model, cfg.chunks, cfg.chunk_delay)
        } else {
            Json(transcript_json(&model)).into_response()
        }
    } else {
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
        let model = audio_model(&body);
        let stream = body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if stream {
            transcript_sse(model, cfg.chunks, cfg.chunk_delay)
        } else {
            Json(transcript_json(&model)).into_response()
        }
    }
}

async fn audio_voices(
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let model = q
        .get("model")
        .cloned()
        .unwrap_or_else(|| "stub-audio".into());
    Json(serde_json::json!({
        "model": model,
        "voices": [
            {"voice_id": "alba"},
            {"voice_id": "cosette"}
        ]
    }))
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
