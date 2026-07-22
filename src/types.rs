// ── OpenAI-compatible request/response types ─────────────────────────────────
//
// These types are the wire format for `/v1/chat/completions` and
// `/v1/completions`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};

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

/// A non-streaming chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// A streaming chat completion chunk (SSE `data:` payload).
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

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

/// A non-streaming completions response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

// ── /v1/models response ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    /// Context length (tokens).
    pub context_length: Option<usize>,
}

// ── /v1/info and /admin/status response ──────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct InfoResponse {
    pub models: Vec<ModelInfo>,
    pub aliases: Vec<AliasInfo>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gpus: Vec<GpuStatus>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub context_length: usize,
    pub instance_count: usize,
    pub max_instances: usize,
    pub queue_depth_used: usize,
    pub queue_depth_max: usize,
    pub blocked: bool,
    /// 1 / 5 / 15-minute EMA of concurrent in-flight requests.
    pub load_m1: f64,
    pub load_m5: f64,
    pub load_m15: f64,
    /// 1 / 5 / 15-minute EMA of request completion rate (req/min).
    pub req_rate_m1: f64,
    pub req_rate_m5: f64,
    pub req_rate_m15: f64,
    /// Total completed requests since daemon start.
    pub completions_total: u64,
    /// Last N completion records (ring buffer, newest first).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub recent_completions: Vec<CompletionRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AliasInfo {
    pub name: String,
    pub target: String,
    pub has_system_prompt: bool,
}

// ── GPU status (for /admin/status) ───────────────────────────────────────────

/// Summary of a single GPU, derived from the raw sysfs snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct GpuStatus {
    pub index: usize,
    pub pci_slot: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_vendor: Option<String>,
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

// ── Admin actions ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct AdminModelAction {
    pub model: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminResponse {
    pub status: String,
    pub message: String,
}

// ── Completion tracking (for /admin/status) ──────────────────────────────────

/// A single completed request, stored in a per-model ring buffer.
#[derive(Debug, Clone, Serialize)]
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
    fn chat_request_round_trip_omits_stream_when_false() {
        let input = r#"{"model":"m","temperature":0,"messages":[{"role":"system","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&req).unwrap();
        // stream=false must be absent (not serialized)
        assert!(!output.contains("\"stream\":"), "stream should be absent: {}", output);
        // stream_options must be absent when None
        assert!(!output.contains("stream_options"), "stream_options should be absent: {}", output);
        // temperature:0 must stay 0 (integer), not 0.0
        assert!(output.contains("\"temperature\":0"), "temperature should be 0 int: {}", output);
    }

    #[test]
    fn chat_request_preserves_extra_field_order() {
        let input = r#"{"model":"m","messages":[],"prompt_cache_key":"abc","custom_flag":true,"temperature":0.7}"#;
        let req: ChatCompletionRequest = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&req).unwrap();
        let val: serde_json::Value = serde_json::from_str(&output).unwrap();
        let keys: Vec<&str> = val.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        // Declaration order: model, messages, temperature, top_p, ..., extra fields in insertion order
        assert_eq!(keys[0], "model");
        assert_eq!(keys[1], "messages");
        // prompt_cache_key and custom_flag are flattened extra — order preserved
        let pk_pos = keys.iter().position(|&k| k == "prompt_cache_key").unwrap();
        let cf_pos = keys.iter().position(|&k| k == "custom_flag").unwrap();
        assert!(pk_pos < cf_pos, "prompt_cache_key should come before custom_flag");
    }

    #[test]
    fn chat_request_stream_true_is_serialized() {
        let input = r#"{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&req).unwrap();
        assert!(output.contains("\"stream\":true"), "stream=true must be present: {}", output);
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
        let req: ChatCompletionRequest =
            serde_json::from_value(input.clone()).unwrap();
        let output = serde_json::to_value(&req).unwrap();
        // Every message-level field must survive verbatim.
        assert_eq!(output, input, "message-level fields were lost in round-trip");
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
        let req: ChatCompletionRequest =
            serde_json::from_value(input.clone()).unwrap();
        let output = serde_json::to_value(&req).unwrap();
        assert_eq!(output, input, "unknown content part was lost in round-trip");
    }

    #[test]
    fn completion_request_round_trip() {
        let input = r#"{"model":"m","prompt":"hello","temperature":0,"top_p":1}"#;
        let req: CompletionRequest = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&req).unwrap();
        assert!(!output.contains("\"stream\":"), "stream should be absent: {}", output);
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
        assert_eq!(output, input, "rerank request fields were lost in round-trip");
    }

    #[test]
    fn rerank_request_omits_optional_fields() {
        let input = r#"{"model":"m","query":"q","documents":["a"]}"#;
        let req: RerankRequest = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&req).unwrap();
        assert!(!output.contains("top_n"), "top_n should be absent: {}", output);
        assert!(!output.contains("return_documents"), "return_documents should be absent: {}", output);
    }
}
