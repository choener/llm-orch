// ── OpenAI-compatible request/response types ─────────────────────────────────
//
// These types are the wire format for `/v1/chat/completions` and
// `/v1/completions`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use std::collections::HashMap;

// ── Chat completions request ─────────────────────────────────────────────────

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<MessageContent>,
    /// Optional name for the participant (used for multi-party conversations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls issued by the assistant. Kept as a raw `Value` — modelling
    /// the inner tool_call structure would just be another lossy trap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    /// Links a `role:"tool"` result message back to its tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Pass through any other message-level fields verbatim
    /// (`reasoning_content`, `tool_responses`, `refusal`, and anything future).
    /// Without this, the proxy silently drops them and breaks tool calling.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Message content — can be a plain string or an array of content parts
/// (for multimodal models).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// A content part in a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    /// Catch-all for content part types we don't model explicitly
    /// (`input_audio`, video, ...). Passed through verbatim so an unknown
    /// part type can't fail deserialization of the entire request.
    #[serde(untagged)]
    Other(Map<String, Value>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// The request body for `POST /v1/chat/completions`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    /// Model name (or alias).
    pub model: String,
    /// The conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Whether to stream the response.
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub stream: bool,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Number>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<Number>,
    /// Number of completions to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Whether to stream log probabilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequences>,
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<Number>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<Number>,
    /// Random seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Response format specification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    /// Tool choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// Additional parameters passed through to the backend.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Stop sequences — can be a single string or an array.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

// ── Chat completions response ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

// ── Completions request/response (legacy /v1/completions) ────────────────────

/// The request body for `POST /v1/completions`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionRequest {
    /// Model name (or alias).
    pub model: String,
    /// The prompt text.
    pub prompt: PromptContent,
    /// Whether to stream.
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub stream: bool,
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Number>,
    /// Nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<Number>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequences>,
    /// Additional parameters passed through to the backend.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Prompt content — can be a single string or an array of strings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PromptContent {
    Text(String),
    Array(Vec<String>),
}

// ── /v1/models response ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelEntry>,
}

/// llm-orch's default capability set: a plain text model.
impl Default for Architecture {
    fn default() -> Self {
        Self::text()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    /// Context length (tokens).
    pub context_length: Option<usize>,
    /// Capability advertisement (mirrors llama.cpp's `architecture`).
    pub architecture: Architecture,
}

/// Per-model capability advertisement, mirroring the `architecture` object
/// llama.cpp publishes in `/v1/models`: `input_modalities` lists what a
/// model reads, `output_modalities` what it produces.
///
/// A model whose `output_modalities` contains `"decisions"` is a *native
/// decision model*: it answers `POST /v1/systemone` with a probability per
/// option instead of generating text.  Detection is by membership —
/// `output_modalities.includes("decisions")` — and unknown values must be
/// tolerated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Architecture {
    #[serde(default)]
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub output_modalities: Vec<String>,
}

/// The capability values llm-orch understands, in the vocabulary llama.cpp
/// puts in its `architecture` object:
///
/// - `decisions` — an *output* capability: a native decision model, served
///   by `POST /v1/systemone`.
/// - `image`, `audio`, `video` — *input* modalities: what the model can
///   read besides text (vision decision models, omni models).
///
/// Speech *output* (`audio` in `output_modalities`) is deliberately not
/// modelled: llm-orch routes TTS and ASR through the `/v1/audio/*`
/// endpoints, so a client never needs the advertisement to find them.
pub const KNOWN_CAPABILITIES: [&str; 4] = ["decisions", "image", "audio", "video"];

/// Input modalities, in the order they are advertised.
const INPUT_MODALITIES: [&str; 3] = ["image", "audio", "video"];

impl Architecture {
    /// Plain text model — the default for every llm-orch model.
    pub fn text() -> Self {
        Self {
            input_modalities: vec!["text".into()],
            output_modalities: vec!["text".into()],
        }
    }

    /// The advertisement implied by a model's declared capabilities.
    /// Validation guarantees every entry is in [`KNOWN_CAPABILITIES`];
    /// anything else is ignored here, never invented.
    pub fn from_capabilities(capabilities: &[String]) -> Self {
        let has = |cap: &str| capabilities.iter().any(|c| c == cap);
        let mut input = vec!["text"];
        for m in INPUT_MODALITIES {
            if has(m) {
                input.push(m);
            }
        }
        let mut output = vec!["text"];
        if has("decisions") {
            output.push("decisions");
        }
        Self {
            input_modalities: input.into_iter().map(str::to_string).collect(),
            output_modalities: output.into_iter().map(str::to_string).collect(),
        }
    }

    /// The inverse: llm-orch capabilities implied by a peer's advertised
    /// architecture.  Values outside [`KNOWN_CAPABILITIES`] are dropped —
    /// they belong to the peer, and guessing them here would invent
    /// llm-orch capabilities.  `text` is implicit on both sides.
    pub fn capabilities(&self) -> Vec<String> {
        let mut caps: Vec<String> = Vec::new();
        if self.output_modalities.iter().any(|m| m == "decisions") {
            caps.push("decisions".into());
        }
        for m in INPUT_MODALITIES {
            if self.input_modalities.iter().any(|i| i == m) {
                caps.push(m.into());
            }
        }
        caps
    }
}

// ── /v1/info and /admin/status response ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoResponse {
    pub models: Vec<ModelInfo>,
    pub aliases: Vec<AliasInfo>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gpus: Vec<GpuStatus>,
    pub metrics: SchedulerMetrics,
    /// Cluster-peering observability (empty / omitted without peers).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peers: Vec<crate::peers::PeerStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub context_length: usize,
    pub instance_count: usize,
    pub max_instances: usize,
    pub queue_depth_used: usize,
    pub queue_depth_max: usize,
    pub queue: QueueStatus,
    pub blocked: bool,
    /// 1 / 5 / 15-minute EMA of concurrent in-flight requests.
    pub load_m1: f64,
    pub load_m5: f64,
    pub load_m15: f64,
    /// 1 / 5 / 15-minute EMA of request completion rate (req/min).
    /// Counted at slot release regardless of outcome (release rate).
    pub req_rate_m1: f64,
    pub req_rate_m5: f64,
    pub req_rate_m15: f64,
    /// Total completed requests since daemon start (release-counted).
    pub completions_total: u64,
    /// Last N completion records (ring buffer, newest first).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub recent_completions: Vec<CompletionRecord>,
    /// Backend kind: "llama.cpp" or "sd.cpp" (determined by program basename).
    pub backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasInfo {
    pub name: String,
    pub targets: Vec<String>,
}

/// Aggregate scheduler counters for diagnosing queue starvation and VRAM
/// migration churn.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerMetrics {
    pub queue_enqueued_total: u64,
    pub queue_assigned_total: u64,
    pub queue_cancelled_total: u64,
    pub queue_expired_total: u64,
    pub migration_started_total: u64,
    pub migration_completed_total: u64,
    pub migration_failed_total: u64,
    pub reservation_cleanup_total: u64,
}

// ── Queue/migration/request status ──────────────────────────────────────────

/// State exposed for an accepted request's scheduler lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestState {
    Queued,
    WaitingForSlot,
    WaitingForMigration,
    WaitingForSpawn,
    LoadingReplacement,
    WaitingForDrain,
    LoadingTarget,
    Assigned,
    Running,
    Completed,
    Expired,
    Cancelled,
    Shutdown,
    Failed,
}

/// Phase exposed for a shared make-room migration transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationPhase {
    Planning,
    Reserving,
    LoadingReplacement,
    WaitingForDrain,
    Handoff,
    LoadingTarget,
    Completed,
    RolledBack,
    Cancelled,
    Failed,
}

/// Aggregate queue counts used by model and admin status responses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueueStatus {
    pub waiting_for_slot: usize,
    pub waiting_for_migration: usize,
    pub waiting_for_spawn: usize,
}

/// Status for a shared migration operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStatus {
    pub migration_id: u64,
    pub logical_model: String,
    pub phase: MigrationPhase,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub victim_instance: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub victim_gpu: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub destination_gpu: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_gpu: Vec<String>,
    pub active_victim_slots: usize,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub reservation_peak_vram: HashMap<String, u64>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub reservation_final_vram: HashMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

/// Request status returned by the authenticated request-status endpoint.
///
/// Timestamps use the same string representation as `CompletionRecord`; the
/// scheduler owns formatting wall-clock values while duration remains a
/// numeric millisecond value for clients and metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestStatus {
    pub request_id: String,
    pub logical_model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_model: Option<String>,
    pub state: RequestState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue: Option<QueueStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_position: Option<usize>,
    pub enqueued_at: String,
    pub wait_duration_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub migration_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub migration_phase: Option<MigrationPhase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub victim_instance: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub victim_gpu: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub destination_gpu: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_gpu: Vec<String>,
    pub active_victim_slots: usize,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub reservation_peak_vram: HashMap<String, u64>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub reservation_final_vram: HashMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

// ── GPU status (for /admin/status) ───────────────────────────────────────────

/// Summary of a single GPU, derived from the raw sysfs snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatus {
    pub index: usize,
    pub pci_slot: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_vendor: Option<String>,
    /// Cluster peering: the node this GPU belongs to (`None` = this
    /// instance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    /// VRAM used / total in bytes.
    pub vram_used_bytes: u64,
    pub vram_total_bytes: u64,
    /// VRAM utilisation percentage (0–100).
    pub vram_util_pct: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature_c: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_w: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_busy_pct: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sclk_mhz: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mclk_mhz: Option<u64>,
}

// ── /v1/embeddings request/response ─────────────────────────────────────────

/// Input to the embeddings endpoint — a single string or an array of strings.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Request body for `POST /v1/embeddings`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub index: u32,
    pub embedding: EmbeddingVector,
}

// ── /v1/rerank request ──────────────────────────────────────────────────────

/// A single document in a rerank request.
///
/// llama.cpp (Jina-compatible) accepts plain strings as well as arbitrary
/// objects (e.g. `{"text": "…"}`); objects are serialized before scoring.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum RerankDocument {
    Text(String),
    Object(Map<String, Value>),
}

/// Request body for `POST /v1/rerank`.
///
/// Jina-compatible, matching llama.cpp's `/v1/rerank` (also exposed by the
/// backend as `/rerank` and `/reranking`).  The response is passed through
/// verbatim as `serde_json::Value` — llama.cpp returns
/// `{results: [{index, relevance_score, document?}], model, usage?}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<RerankDocument>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

// ── Image generation types (stable-diffusion.cpp) ────────────────────────────

/// Request body for `POST /v1/images/generations` (OpenAI-compatible).
///
/// Known OpenAI fields are modelled explicitly; extension fields (lora,
/// sample_params, vae_tiling_params, cache_mode, etc.) are passed through
/// verbatim via `extra` — sd-server accepts them as native sdcpp fields.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImagesGenerationRequest {
    pub prompt: String,
    /// Model name (or alias).  Required for llm-orch routing.
    #[serde(default)]
    pub model: Option<String>,
    /// Number of images to generate (sd-server: batch_count).
    #[serde(default = "default_one")]
    pub n: u32,
    /// Image size in `WIDTHxHEIGHT` format, e.g. "1024x1024".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// Output image format: png, jpeg, or webp.
    #[serde(default = "default_output_format")]
    pub output_format: String,
    /// Output compression level (0–100).
    #[serde(default = "default_max_compression")]
    pub output_compression: u32,
    /// Any additional fields (negative_prompt, lora, sample_params, seed,
    /// width, height, cfg_scale, sd_cpp_extra_args in prompt, …) are passed
    /// through verbatim to sd-server.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn default_one() -> u32 {
    1
}
fn default_output_format() -> String {
    "png".into()
}
fn default_max_compression() -> u32 {
    100
}

/// Response body for `POST /v1/images/generations`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagesGenerationResponse {
    pub created: i64,
    pub data: Vec<ImageObject>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
}

/// A single generated image (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageObject {
    /// Base64-encoded image bytes.
    #[serde(rename = "b64_json")]
    pub b64_json: String,
}

/// Multipart form fields for `POST /v1/images/edits`.
///
/// sd-server expects multipart/form-data with `prompt`, `image[]`, `mask`,
/// `n`, `size`, `output_format`, `output_compression` fields plus an
/// optional `model` field (for llm-orch routing).
///
/// Additional sdcpp-native fields can be embedded as `sd_cpp_extra_args`
/// inside `prompt` or passed as a separate `params` JSON field.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImagesEditRequest {
    pub prompt: String,
    /// Model name (or alias) for llm-orch routing.
    #[serde(default)]
    pub model: Option<String>,
    /// Number of images to generate.
    #[serde(default = "default_one")]
    pub n: u32,
    /// Image size in `WIDTHxHEIGHT` format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// Output format: png or jpeg.
    #[serde(default = "default_output_format")]
    pub output_format: String,
    /// Output compression level.
    #[serde(default = "default_max_compression")]
    pub output_compression: u32,
    /// Optional JSON params object for sdcpp-native extension fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Map<String, Value>>,
}

// ── Video generation types (stable-diffusion.cpp) ────────────────────────────

/// Request body for `POST /v1/video/generations`.
///
/// This is a new llm-orch endpoint (not standard OpenAI).  The request is
/// forwarded to sd-server's `/sdcpp/v1/vid_gen`; llm-orch polls the async
/// job and returns the completed video as raw bytes.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VideoGenerationRequest {
    pub prompt: String,
    /// Model name (or alias) for llm-orch routing.
    pub model: Option<String>,
    /// Number of video frames (default varies by model, sd-server sets it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_frames: Option<u32>,
    /// Target frames-per-second.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Video width.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Video height.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// Seed (-1 = random).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Output container format: webm, webp, or avi.
    #[serde(default = "default_video_output_format")]
    pub output_format: String,
    /// Extension fields passed verbatim to sd-server.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn default_video_output_format() -> String {
    "webm".into()
}

/// Response body for `POST /v1/video/generations` (llm-orch internal).
/// The handler decodes the base64 container and returns raw bytes instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenerationResponse {
    pub created: i64,
    pub data: Vec<VideoObject>,
}

/// A single generated video (llm-orch internal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoObject {
    /// Base64-encoded container bytes.
    #[serde(rename = "b64_json")]
    pub b64_json: String,
    /// MIME type of the container (video/webm, image/webp, video/x-msvideo).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Effective playback FPS.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Actual decoded frame count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_count: Option<u32>,
    /// Selected output container format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
}

/// sdcpp-native async job status (parsed from sd-server's GET /sdcpp/v1/jobs/{id}).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SdcppJobStatus {
    pub id: String,
    pub kind: String,
    pub status: String,
    #[serde(default)]
    pub created: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<i64>,
    #[serde(default)]
    pub queue_position: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<SdcppJobResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<SdcppJobError>,
}

/// Result of a completed sdcpp job.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SdcppJobResult {
    /// Base64-encoded output (image or video container).
    #[serde(rename = "b64_json")]
    pub b64_json: Option<String>,
    /// Output format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    /// MIME type for video results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// FPS for video results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Frame count for video results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_count: Option<u32>,
    /// Individual images (for img_gen).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<SdcppJobImage>,
}

/// A single image in a completed sdcpp job result.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SdcppJobImage {
    pub index: u32,
    #[serde(rename = "b64_json")]
    pub b64_json: String,
}

/// Error details for a failed sdcpp job.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SdcppJobError {
    pub code: String,
    pub message: String,
}

// ── Admin actions ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct AdminModelAction {
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminResponse {
    pub status: String,
    pub message: String,
}

// ── Completion tracking (for /admin/status) ──────────────────────────────────

/// A single completed request, stored in a per-model ring buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRecord {
    /// Unix timestamp with milliseconds.
    pub ts: String,
    pub request_id: String,
    pub instance_id: String,
    pub api_user: String,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub cached_tokens: u64,
    pub duration_ms: u64,
}

// ── Serialization helpers ────────────────────────────────────────────────────

/// Used by `#[serde(skip_serializing_if = "bool_is_false")]` to omit
/// `stream: false` from the serialized output, matching the behavior of
/// transparent proxies that don't add default-valued fields.
fn bool_is_false(b: &bool) -> bool {
    !*b
}

// ── Round-trip idempotency tests ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_status_enums_use_stable_snake_case_wire_names() {
        assert_eq!(
            serde_json::to_string(&RequestState::WaitingForMigration).unwrap(),
            "\"waiting_for_migration\""
        );
        assert_eq!(
            serde_json::from_str::<MigrationPhase>("\"loading_target\"").unwrap(),
            MigrationPhase::LoadingTarget
        );
    }

    #[test]
    fn request_status_round_trips_without_optional_payload_fields() {
        let status = RequestStatus {
            request_id: "req-1".into(),
            logical_model: "large".into(),
            resolved_model: Some("large-v1".into()),
            state: RequestState::WaitingForDrain,
            queue: Some(QueueStatus {
                waiting_for_slot: 1,
                waiting_for_migration: 2,
                waiting_for_spawn: 0,
            }),
            queue_position: Some(2),
            enqueued_at: "2025-01-01T00:00:00.000Z".into(),
            wait_duration_ms: 1200,
            deadline: None,
            migration_id: Some(7),
            migration_phase: Some(MigrationPhase::WaitingForDrain),
            victim_instance: Some("small@1#0".into()),
            victim_gpu: vec!["0000:01:00.0".into()],
            destination_gpu: vec!["0000:02:00.0".into()],
            target_gpu: vec!["0000:01:00.0".into()],
            active_victim_slots: 1,
            reservation_peak_vram: HashMap::from([("0000:02:00.0".into(), 30)]),
            reservation_final_vram: HashMap::from([("0000:01:00.0".into(), 90)]),
            failure_reason: None,
        };
        let encoded = serde_json::to_value(&status).unwrap();
        assert!(encoded.get("failure_reason").is_none());
        assert!(encoded.get("deadline").is_none());
        let decoded: RequestStatus = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.state, RequestState::WaitingForDrain);
    }

    #[test]
    fn chat_request_round_trip_omits_stream_when_false() {
        let input =
            r#"{"model":"m","temperature":0,"messages":[{"role":"system","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&req).unwrap();
        // stream=false must be absent (not serialized)
        assert!(
            !output.contains("\"stream\":"),
            "stream should be absent: {}",
            output
        );
        // stream_options must be absent when None
        assert!(
            !output.contains("stream_options"),
            "stream_options should be absent: {}",
            output
        );
        // temperature:0 must stay 0 (integer), not 0.0
        assert!(
            output.contains("\"temperature\":0"),
            "temperature should be 0 int: {}",
            output
        );
    }

    #[test]
    fn chat_request_preserves_extra_field_order() {
        let input = r#"{"model":"m","messages":[],"prompt_cache_key":"abc","custom_flag":true,"temperature":0.7}"#;
        let req: ChatCompletionRequest = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&req).unwrap();
        let val: serde_json::Value = serde_json::from_str(&output).unwrap();
        let keys: Vec<&str> = val
            .as_object()
            .unwrap()
            .keys()
            .map(|s| s.as_str())
            .collect();
        // Declaration order: model, messages, temperature, top_p, ..., extra fields in insertion order
        assert_eq!(keys[0], "model");
        assert_eq!(keys[1], "messages");
        // prompt_cache_key and custom_flag are flattened extra — order preserved
        let pk_pos = keys.iter().position(|&k| k == "prompt_cache_key").unwrap();
        let cf_pos = keys.iter().position(|&k| k == "custom_flag").unwrap();
        assert!(
            pk_pos < cf_pos,
            "prompt_cache_key should come before custom_flag"
        );
    }

    #[test]
    fn chat_request_stream_true_is_serialized() {
        let input = r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&req).unwrap();
        assert!(
            output.contains("\"stream\":true"),
            "stream=true must be present: {}",
            output
        );
    }

    #[test]
    fn chat_request_preserves_message_level_tool_fields() {
        // Regression test: tool calling used to break because ChatMessage
        // silently dropped `tool_calls`, `tool_call_id`, and any other
        // message-level field during the deserialize/re-serialize round-trip.
        let tool_calls = serde_json::json!([{
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}
        }]);
        let input = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": tool_calls,
                    "reasoning_content": "I should run ls"
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt"}
            ]
        });
        let req: ChatCompletionRequest = serde_json::from_value(input.clone()).unwrap();
        let output = serde_json::to_value(&req).unwrap();
        // Every message-level field must survive verbatim.
        assert_eq!(
            output, input,
            "message-level fields were lost in round-trip"
        );
    }

    #[test]
    fn chat_request_tolerates_unknown_content_part_types() {
        // Regression test: an unknown content part type (e.g. input_audio)
        // must not fail deserialization of the entire request, and must be
        // passed through verbatim.
        let input = serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe"},
                    {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
                ]
            }]
        });
        let req: ChatCompletionRequest = serde_json::from_value(input.clone()).unwrap();
        let output = serde_json::to_value(&req).unwrap();
        assert_eq!(output, input, "unknown content part was lost in round-trip");
    }

    #[test]
    fn completion_request_round_trip() {
        let input = r#"{"model":"m","prompt":"hello","temperature":0,"top_p":1}"#;
        let req: CompletionRequest = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&req).unwrap();
        assert!(
            !output.contains("\"stream\":"),
            "stream should be absent: {}",
            output
        );
    }

    #[test]
    fn rerank_request_accepts_string_and_object_documents() {
        // Jina/llama.cpp allow plain strings or arbitrary objects as documents.
        let input = serde_json::json!({
            "model": "reranker",
            "query": "what is rust",
            "documents": [
                "a systems language",
                {"text": "a metal oxide"}
            ],
            "top_n": 2,
            "unknown_future_field": {"nested": true}
        });
        let req: RerankRequest = serde_json::from_value(input.clone()).unwrap();
        let output = serde_json::to_value(&req).unwrap();
        assert_eq!(
            output, input,
            "rerank request fields were lost in round-trip"
        );
    }

    #[test]
    fn rerank_request_omits_optional_fields() {
        let input = r#"{"model":"m","query":"q","documents":["a"]}"#;
        let req: RerankRequest = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&req).unwrap();
        assert!(
            !output.contains("top_n"),
            "top_n should be absent: {}",
            output
        );
        assert!(
            !output.contains("return_documents"),
            "return_documents should be absent: {}",
            output
        );
    }
}
