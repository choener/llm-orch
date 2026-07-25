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
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;
use tracing::warn;

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
    /// Opens the file in append+create mode on first use.  **Best-effort**:
    /// every failure — lock poisoning, open/write/flush errors — is logged
    /// via `tracing::warn!` and otherwise ignored.  Debug logging must
    /// never take down a request, least of all from `SseForwarder::drop`,
    /// where a panic-in-drop could also poison the shared writer mutexes
    /// and cascade into unrelated requests.
    pub fn write_line(&self, path: &Path, entry: &DebugLogEntry) {
        let writer = {
            // Recover from poisoning: one panicking request must not
            // disable debug logging for every later request.
            let mut writers = self
                .writers
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            match writers.entry(path.to_path_buf()) {
                std::collections::hash_map::Entry::Occupied(e) => Some(e.get().clone()),
                std::collections::hash_map::Entry::Vacant(e) => {
                    match OpenOptions::new().create(true).append(true).open(path) {
                        Ok(file) => {
                            let w = Arc::new(Mutex::new(BufWriter::new(file)));
                            e.insert(w.clone());
                            Some(w)
                        }
                        Err(err) => {
                            // Not cached: retried on the next entry, so a
                            // transient failure (permissions fixed, directory
                            // created later) recovers automatically.
                            warn!(
                                path = %path.display(),
                                error = %err,
                                "debug log: open failed, dropping entry"
                            );
                            None
                        }
                    }
                }
            }
        };
        let Some(writer) = writer else {
            return;
        };

        // Serialize before touching the writer: a failed `to_writer` could
        // leave a partial line in the buffer and corrupt later entries.
        let mut line = match serde_json::to_string(entry) {
            Ok(l) => l,
            Err(e) => {
                warn!(
                    path = %path.display(),
                    error = %e,
                    "debug log: serialization failed, dropping entry"
                );
                return;
            }
        };
        line.push('\n');

        let mut w = writer.lock().unwrap_or_else(PoisonError::into_inner);
        if let Err(e) = w.write_all(line.as_bytes()).and_then(|()| w.flush()) {
            warn!(
                path = %path.display(),
                error = %e,
                "debug log: write/flush failed, dropping entry"
            );
        }
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
/// when the stream ends (in `Drop`) and completion records can be written.
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_entry() -> DebugLogEntry {
        DebugLogEntry {
            ts: ts_now(),
            request_id: "req-1".into(),
            model: "m".into(),
            alias: None,
            instance_id: None,
            dir: "request".into(),
            stream: None,
            body: None,
            usage: None,
            duration_ms: None,
            error: None,
        }
    }

    fn temp_log_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("llm-orch-debug-log-test-{}-{}", tag, uuid::Uuid::new_v4()))
    }

    #[test]
    fn write_line_appends_valid_jsonl() {
        let path = temp_log_path("happy");
        let loggers = DebugLoggers::new();
        loggers.write_line(&path, &test_entry());
        loggers.write_line(&path, &test_entry());

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let v: serde_json::Value =
                serde_json::from_str(line).expect("each line must be valid JSON");
            assert_eq!(v["request_id"], "req-1");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_line_never_panics_on_unopenable_path() {
        // A misconfigured path (parent directory does not exist) must drop
        // the entry with a warning, not panic — this runs from
        // `SseForwarder::drop`, where a panic would poison shared mutexes.
        let path = temp_log_path("missing-parent").join("no-such-dir").join("x.jsonl");
        let loggers = DebugLoggers::new();
        loggers.write_line(&path, &test_entry());
        // Retried on the next call (failure is not cached) — still no panic.
        loggers.write_line(&path, &test_entry());
    }

    #[test]
    fn write_line_recovers_from_poisoned_writer() {
        // Poison the per-file writer mutex by panicking while holding it;
        // the next write must recover the guard and still log.
        let path = temp_log_path("poison");
        let loggers = Arc::new(DebugLoggers::new());
        loggers.write_line(&path, &test_entry());

        let writer = {
            let writers = loggers.writers.lock().unwrap();
            writers.get(&path).unwrap().clone()
        };
        let w2 = writer.clone();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = w2.lock().unwrap();
            panic!("deliberate poisoning");
        }));
        assert!(writer.lock().is_err(), "writer mutex must be poisoned");

        loggers.write_line(&path, &test_entry());
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            content.lines().count(),
            2,
            "write after poisoning must still append"
        );
        std::fs::remove_file(&path).ok();
    }
}