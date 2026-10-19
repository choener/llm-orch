// ── Stub llama.cpp / sd.cpp backend ─────────────────────────────────────────────
//
// A minimal HTTP server that mimics the llama.cpp and stable-diffusion.cpp
// endpoints llm-orch uses, so integration tests can exercise the full spawn
// → forward → stream path without a real GPU or model.
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
//   POST /v1/images/generations   → canned base64 image JSON (sd.cpp stub)
//   POST /v1/images/edits         → canned base64 image JSON
//   POST /sdcpp/v1/vid_gen        → 202 + job creation (async video)
//   GET  /sdcpp/v1/jobs/{id}      → completed job (200) or pending (202)
//   POST /sdcpp/v1/img_gen        → 202 + job creation (async image)
//   GET  /sdcpp/v1/capabilities   → canned capability JSON
//   POST /sdapi/v1/txt2img        → canned sdapi image response
//   POST /sdapi/v1/img2img        → canned sdapi image response
//   GET  /test/last-auth          → test support: Authorization header of
//                                   the most recent request (JSON)
//                                   (`{"authorization": "…" | null}`)
//   --fail-status N               → test support: every generation request
//                                   answers with status N and a JSON error
//                                   body (`/health` stays healthy), so
//                                   upstream-fault handling is testable
//   --job-pending-secs N          → test support: async sdcpp jobs report
//                                   `generating` (no result) for the first N
//                                   seconds after creation, so polling
//                                   clients really poll
//   --stop-serving-after N        → test support: stop answering HTTP N seconds
//                                   after the first job is created while the
//                                   process stays alive — an "alive but deaf"
//                                   backend (busy generator, dead listener)
//   --exit-after N                → test support: exit the process N seconds
//                                   after the first job is created
//
// Flags control the canned response: number of chunks, per-chunk delay
// (slow streaming for drain/timeout tests).  audiocpp_server-style flags
// (--config, --host, --backend, …) are accepted and ignored so the stub
// can be spawned with an audio-style cmd line.  sd-server-style flags
// (--diffusion-model, --vae, --clip-l, --backend, …) are accepted and ignored.

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
    /// llama.cpp backend selector (`ROCm0,ROCm1,...`) or an audiocpp
    /// device number; accepted and ignored by the test stub.
    #[arg(long)]
    device: Option<String>,
    /// Speculative-decoding draft model path — accepted and ignored.
    #[arg(long)]
    spec_draft_model: Option<String>,
    /// Speculative-decoding draft device pin — accepted and ignored so the
    /// stub survives llm-orch's automatic `--spec-draft-device` injection.
    #[arg(long)]
    spec_draft_device: Option<String>,
    /// Speculative-decoding type (`draft-mtp`, `ngram-cache`, …) —
    /// accepted and ignored.
    #[arg(long)]
    spec_type: Option<String>,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long)]
    busy_timeout_ms: Option<u64>,
    #[arg(long)]
    model_spec_override: Option<String>,
    /// Write the process argv (program excluded, one argument per line) to
    /// this file at startup — lets integration tests assert the exact
    /// command line llm-orch assembled.
    #[arg(long)]
    argv_log: Option<String>,
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
    // ── sd-server (stable-diffusion.cpp) compatibility flags (accepted, ignored) ──
    #[arg(long)]
    diffusion_model: Option<String>,
    #[arg(long)]
    vae: Option<String>,
    #[arg(long)]
    clip_l: Option<String>,
    #[arg(long)]
    t5xxl: Option<String>,
    #[arg(long)]
    clip_g: Option<String>,
    #[arg(long)]
    control_net_dir: Option<String>,
    #[arg(long)]
    lora_model_dir: Option<String>,
    #[arg(long)]
    hires_upscalers_dir: Option<String>,
    #[arg(long)]
    diffusion_dtype: Option<String>,
    #[arg(long)]
    sampling_steps: Option<usize>,
    #[arg(long)]
    w: Option<u32>,
    #[arg(long)]
    h: Option<u32>,

    // ── test support ────────────────────────────────────────────────────
    /// Respond to every generation request (everything except `/health` and
    /// `/test/*`) with this HTTP status and a JSON error body, simulating a
    /// broken backend.  0 (default) disables the injection.
    #[arg(long, default_value_t = 0)]
    fail_status: u16,
    /// Test support: async sdcpp jobs report `generating` (no result) for this
    /// many seconds after the job is created instead of completing immediately,
    /// so polling clients really poll.
    #[arg(long, default_value_t = 0)]
    job_pending_secs: u64,
    /// Test support: stop serving HTTP this many seconds after the first sdcpp
    /// job is created, while the process keeps running — a backend that is alive
    /// but deaf.  Never fires if no job is created.
    #[arg(long, default_value_t = 0)]
    stop_serving_after: u64,
    /// Test support: exit the process this many seconds after the first sdcpp
    /// job is created.  Never fires if no job is created.
    #[arg(long, default_value_t = 0)]
    exit_after: u64,
}

#[derive(Clone)]
struct StubConfig {
    chunks: usize,
    chunk_delay: Duration,
    /// Test support: Authorization header of the most recent request
    /// (recorded by middleware for every endpoint), exposed via
    /// `GET /test/last-auth`.
    last_auth: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Test support: when non-zero, every generation request fails with this
    /// status code (see `--fail-status`).
    fail_status: u16,
    /// Test support: async jobs stay `generating` this long (see
    /// `--job-pending-secs`); 0 means they complete immediately.
    job_pending_secs: u64,
    /// Test support: set once the first sdcpp job is created, and the reference
    /// point of `--job-pending-secs`, `--stop-serving-after` and `--exit-after`
    /// (so a failure always lands in the middle of polling, never during
    /// startup).
    job_created: tokio::sync::watch::Sender<Option<std::time::Instant>>,
}

impl StubConfig {
    /// When the first sdcpp job was created, if any.
    fn job_created_at(&self) -> Option<std::time::Instant> {
        *self.job_created.borrow()
    }

    /// Record that an sdcpp job was just created (first one wins).
    fn mark_job_created(&self) {
        if self.job_created_at().is_none() {
            let _ = self.job_created.send(Some(std::time::Instant::now()));
        }
    }

    /// Wait until the first sdcpp job is created.
    async fn wait_for_job(&self) {
        if self.job_created_at().is_some() {
            return;
        }
        let mut rx = self.job_created.subscribe();
        let _ = rx.wait_for(Option::is_some).await;
    }

    /// Whether an async sdcpp job should still report `generating`.
    fn job_is_pending(&self) -> bool {
        self.job_pending_secs > 0
            && self
                .job_created_at()
                .is_some_and(|t| t.elapsed() < Duration::from_secs(self.job_pending_secs))
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Some(path) = &cli.argv_log {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        if let Err(e) = std::fs::write(path, argv.join("\n")) {
            eprintln!("argv-log: failed to write {path}: {e}");
        }
    }
    let cfg = StubConfig {
        chunks: cli.chunks,
        chunk_delay: Duration::from_millis(cli.chunk_delay_ms),
        last_auth: std::sync::Arc::new(std::sync::Mutex::new(None)),
        fail_status: cli.fail_status,
        job_pending_secs: cli.job_pending_secs,
        job_created: tokio::sync::watch::channel(None).0,
    };

    if cli.exit_after > 0 {
        let cfg_exit = cfg.clone();
        tokio::spawn(async move {
            cfg_exit.wait_for_job().await;
            tokio::time::sleep(Duration::from_secs(cli.exit_after)).await;
            std::process::exit(0);
        });
    }

    let cfg_shutdown = cfg.clone();
    let app = Router::new()
        .route("/test/last-auth", get(test_last_auth))
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/responses", post(responses))
        .route("/v1/responses/input_tokens", post(responses_input_tokens))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/audio/voices", get(audio_voices))
        .route("/v1/images/generations", post(images_generations))
        .route("/v1/images/edits", post(images_edits))
        .route("/sdcpp/v1/vid_gen", post(sdcpp_vid_gen))
        .route("/sdcpp/v1/jobs/{id}", get(sdcpp_job_status))
        .route("/sdcpp/v1/img_gen", post(sdcpp_img_gen))
        .route("/sdcpp/v1/capabilities", get(sdcpp_capabilities))
        .route("/sdapi/v1/txt2img", post(sdapi_txt2img))
        .route("/sdapi/v1/img2img", post(sdapi_img2img))
        .route_layer(axum::middleware::from_fn_with_state(
            cfg.clone(),
            record_auth_middleware,
        ))
        .with_state(cfg);

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", cli.port))
        .await
        .expect("bind failed");
    // `--stop-serving-after` closes the listener but leaves the process
    // running, which is how a busy-but-alive backend looks from the outside.
    // The timer starts when the first job is created, so the deafness lands
    // while a client is polling, not while the server is starting up.
    let stop_after = cli.stop_serving_after;
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            if stop_after == 0 {
                std::future::pending::<()>().await;
            }
            cfg_shutdown.wait_for_job().await;
            tokio::time::sleep(Duration::from_secs(stop_after)).await;
        })
        .await
        .unwrap();
    if cli.stop_serving_after > 0 {
        std::future::pending::<()>().await;
    }
}

async fn health() -> &'static str {
    "ok"
}

/// Test support: the Authorization header of the most recent request.
async fn test_last_auth(
    axum::extract::State(cfg): axum::extract::State<StubConfig>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "authorization": *cfg.last_auth.lock().unwrap() }))
}

/// Test support middleware: record the Authorization header of every
/// request into the shared state (absent header → `None`).
async fn record_auth_middleware(
    axum::extract::State(cfg): axum::extract::State<StubConfig>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    // Test endpoints must not clobber the recorded value (their own
    // header-less GET would reset it before the handler reads it).
    if req.uri().path().starts_with("/test/") {
        return next.run(req).await;
    }
    *cfg.last_auth.lock().unwrap() = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    // Injected backend failure: answers every generation endpoint with an
    // error so llm-orch's upstream-fault handling can be tested.  Readiness
    // (`/health`) stays healthy so the instance still spawns normally.
    if cfg.fail_status != 0 && req.uri().path() != "/health" {
        let status = axum::http::StatusCode::from_u16(cfg.fail_status)
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        return (
            status,
            Json(serde_json::json!({
                "error": format!("injected stub failure ({})", cfg.fail_status)
            })),
        )
            .into_response();
    }
    next.run(req).await
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

// ── sd.cpp stub endpoints ────────────────────────────────────────────────────

/// Canned base64-encoded 1x1 red PNG used by all image generation stubs.
fn canned_image_b64() -> &'static str {
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
}

/// Canned video container bytes (minimal valid WebM header).
fn canned_video_b64() -> &'static str {
    "GkXfo59ChoEBQveBAULygQRC84EIQoKf2VibU92hSUH2AACA3RDaU6CvAACAmGx0Y29kBGZpcnN0AIB0b3VjaARnZW5yAQJ3Y4RlY29kZQ=="
}

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

/// Shared counter for sdcpp job IDs.
static SDCPP_JOB_SEQ: AtomicU64 = AtomicU64::new(1);

async fn images_generations(Json(body): Json<serde_json::Value>) -> Json<serde_json::Value> {
    let _model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("stub-sd");
    let n = body.get("n").and_then(|v| v.as_i64()).unwrap_or(1);
    let output_format = body
        .get("output_format")
        .and_then(|v| v.as_str())
        .unwrap_or("png");
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let mut data = Vec::new();
    for _ in 0..n {
        data.push(serde_json::json!({
            "b64_json": canned_image_b64()
        }));
    }

    Json(serde_json::json!({
        "created": now,
        "data": data,
        "output_format": output_format
    }))
}

async fn images_edits(
    _body: axum::body::Bytes,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let resp = serde_json::json!({
        "created": now,
        "data": [{"b64_json": canned_image_b64()}],
        "output_format": "png"
    });
    (axum::http::StatusCode::OK, Json(resp))
}

async fn sdcpp_vid_gen(
    axum::extract::State(cfg): axum::extract::State<StubConfig>,
    Json(_body): Json<serde_json::Value>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    cfg.mark_job_created();
    let id = format!(
        "job_stub_vid_{}",
        SDCPP_JOB_SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let resp = serde_json::json!({
        "id": id,
        "kind": "vid_gen",
        "status": "queued",
        "created": now,
        "poll_url": format!("/sdcpp/v1/jobs/{}", id)
    });
    (axum::http::StatusCode::ACCEPTED, Json(resp))
}

async fn sdcpp_img_gen(
    axum::extract::State(cfg): axum::extract::State<StubConfig>,
    Json(_body): Json<serde_json::Value>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    cfg.mark_job_created();
    let id = format!(
        "job_stub_img_{}",
        SDCPP_JOB_SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let resp = serde_json::json!({
        "id": id,
        "kind": "img_gen",
        "status": "queued",
        "created": now,
        "poll_url": format!("/sdcpp/v1/jobs/{}", id)
    });
    (axum::http::StatusCode::ACCEPTED, Json(resp))
}

async fn sdcpp_job_status(
    axum::extract::Path(id): axum::extract::Path<String>,
    axum::extract::State(cfg): axum::extract::State<StubConfig>,
) -> Json<serde_json::Value> {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let kind = if id.contains("vid") {
        "vid_gen"
    } else {
        "img_gen"
    };
    if cfg.job_is_pending() {
        return Json(serde_json::json!({
            "id": id,
            "kind": kind,
            "status": "generating",
            "created": now - 10,
            "started": now - 8,
            "queue_position": 0,
            "error": null
        }));
    }
    Json(serde_json::json!({
        "id": id,
        "kind": kind,
        "status": "completed",
        "created": now - 10,
        "started": now - 8,
        "completed": now,
        "queue_position": 0,
        "result": {
            "b64_json": if kind == "vid_gen" { canned_video_b64() } else { canned_image_b64() },
            "output_format": if kind == "vid_gen" { "webm" } else { "png" },
            "mime_type": if kind == "vid_gen" { "video/webm" } else { "image/png" },
            "frame_count": if kind == "vid_gen" { serde_json::json!(33) } else { serde_json::Value::Null },
            "fps": if kind == "vid_gen" { serde_json::json!(16) } else { serde_json::Value::Null },
            "images": if kind == "img_gen" {
                serde_json::json!([{"index": 0, "b64_json": canned_image_b64()}])
            } else {
                serde_json::Value::Null
            }
        },
        "error": null
    }))
}

async fn sdcpp_capabilities() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "supported_modes": ["img_gen", "vid_gen"],
        "defaults_by_mode": {
            "img_gen": {
                "width": 1024,
                "height": 1024,
                "sample_params": {
                    "sample_steps": 28
                }
            },
            "vid_gen": {
                "width": 832,
                "height": 480,
                "video_frames": 33,
                "fps": 16
            }
        },
        "output_formats_by_mode": {
            "img_gen": ["png", "jpeg", "webp"],
            "vid_gen": ["webm", "webp", "avi"]
        },
        "features_by_mode": {
            "img_gen": {
                "init_image": true,
                "lora": true,
                "cache": true
            },
            "vid_gen": {
                "init_image": true,
                "lora": true
            }
        },
        "samplers": [{"name": "euler_a"}, {"name": "euler"}],
        "loras": [],
        "limits": {
            "min_width": 64,
            "max_width": 4096,
            "min_height": 64,
            "max_height": 4096,
            "max_batch_count": 8,
            "max_queue_size": 10
        }
    }))
}

async fn sdapi_txt2img(Json(_body): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "images": [canned_image_b64()],
        "parameters": {},
        "info": ""
    }))
}

async fn sdapi_img2img(Json(_body): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "images": [canned_image_b64()],
        "parameters": {},
        "info": ""
    }))
}
