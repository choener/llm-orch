// ── Per-model debug logging (JSONL) ─────────────────────────────────────────
//
// When a model config includes `debug_log: "/path/to/io.jsonl"`, every
// request/response pair is written to that file as one JSON object per line
// (JSONL).  Lines are flushed immediately so the file is tail-able even
// during streaming.
//
// Concurrency: multiple requests targeting the same model share one writer,
// protected by a std::sync::Mutex.  Writes are a single serde + write + flush
// per entry, so the lock is held for microseconds.

use serde::Serialize;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ── Shared registry ──────────────────────────────────────────────────────────

/// Lazily-opened, shared-debug-log writers keyed by file path.
///
/// `Arc<DebugLoggers>` lives in `AppState` so every handler can log without
/// needing to hold a config lock.
pub struct DebugLoggers {
    writers: Mutex<HashMap<PathBuf, Arc<Mutex<BufWriter<File>>>>>,
}

impl DebugLoggers {
    pub fn new() -> Self {
        Self {
            writers: Mutex::new(HashMap::new()),
        }
    }

    /// Write one JSON line to the debug log at `path`.
    ///
    /// Opens the file in append+create mode on first use.  Panics on I/O
    /// errors — debug logging is best-effort, and panicking surfaces the
    /// problem immediately instead of silently dropping data.
    pub fn write_line(&self, path: &Path, entry: &DebugLogEntry) {
        let writer = {
            let mut writers = self.writers.lock().unwrap();
            writers
                .entry(path.to_path_buf())
                .or_insert_with(|| {
                    let file = OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                        .unwrap_or_else(|e| {
                            panic!("failed to open debug log {:?}: {}", path, e)
                        });
                    Arc::new(Mutex::new(BufWriter::new(file)))
                })
                .clone()
        };

        let mut w = writer.lock().unwrap();
        serde_json::to_writer(&mut *w, entry)
            .expect("failed to serialize debug log entry");
        w.write_all(b"\n")
            .expect("failed to write debug log newline");
        w.flush()
            .expect("failed to flush debug log");
    }
}

// ── Log entry ────────────────────────────────────────────────────────────────

/// One line in a per-model debug log (JSONL).
#[derive(Debug, Serialize)]
pub struct DebugLogEntry {
    /// Unix timestamp with milliseconds, e.g. `"1721059200.123"`.
    pub ts: String,
    /// Correlates the request and response lines.
    pub request_id: String,
    /// The resolved model name (after alias lookup).
    pub model: String,
    /// The alias name, if the request was routed through an alias.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    /// The backend instance that handled this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    /// `"request"` or `"response"`.
    pub dir: String,
    /// `true` for streaming responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// The exact body forwarded to / received from the backend.
    /// For streaming responses this is `{"chunks": ["…","…"]}`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<serde_json::Value>,
    /// Token usage (extracted from the response when available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
    /// Wall-clock milliseconds for this request or this response chunk set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Non-empty when the backend returned an error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ── Stream context ───────────────────────────────────────────────────────────

/// Carried into a streaming response so the accumulated chunks can be logged
/// when the stream ends (in `Drop`).
pub struct DebugStreamContext {
    pub loggers: Arc<DebugLoggers>,
    pub path: PathBuf,
    pub request_id: String,
    pub model_name: String,
    pub alias: Option<String>,
    pub instance_id: String,
    pub t0: Instant,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Current time as a Unix timestamp with milliseconds, e.g. `"1721059200.123"`.
pub fn ts_now() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}", dur.as_secs(), dur.subsec_millis())
}