// ── Token / model statistics ───────────────────────────────────────────────
//
// Opt-in via the top-level `stats:` config section.  Keeps per-model
// counters in two sets:
//
// - `total` — cumulative; restored from the stats file at startup, so it
//   survives restarts (the "bragging" total).
// - `boot`  — zeroed at every startup (ephemeral since-boot view).
//
// Both are updated after every served request — requests whose backend
// reported no usage count without token totals (warned).  The
// JSON file is rewritten (atomic temp + rename) on each update, following
// the project's atomic-file-replace convention so external readers (e.g.
// Home Assistant polling `/v1/stats`, or the file directly) never see a
// partial write.
//
// Concurrency: a short-lived `std::sync::Mutex`.  The file write happens
// *after* the lock is released, so a slow disk never blocks other
// requests' counter updates.  Persistence is best-effort: failures are
// logged via `tracing::warn!` and never take down a request (the
// `debug_log` convention).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use tracing::warn;

/// One model's counters for one set (`total` or `boot`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCounters {
    /// Served requests (token totals may be zero when the backend
    /// reported no usage).
    pub requests: u64,
    /// Prompt (input) tokens.
    pub prompt_tokens: u64,
    /// Prompt tokens served from the KV cache.
    pub cached_tokens: u64,
    /// Generated (output) tokens.
    pub completion_tokens: u64,
    /// Streaming requests for which TTFT was measured.
    pub ttft_count: u64,
    /// Sum of TTFTs in milliseconds.
    pub ttft_sum_ms: u64,
    /// Fastest TTFT in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_min_ms: Option<u64>,
    /// Slowest TTFT in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_max_ms: Option<u64>,
    /// Sum of end-to-end request durations in milliseconds.
    pub duration_sum_ms: u64,
    /// Mean end-to-end request duration in milliseconds (derived).
    pub duration_avg_ms: u64,
    /// Unix seconds of the most recent recorded request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_active: Option<u64>,
}

impl ModelCounters {
    /// Fold one recorded request into the counters.
    pub fn apply(
        &mut self,
        prompt_tokens: Option<u64>,
        cached_tokens: Option<u64>,
        completion_tokens: Option<u64>,
        duration_ms: u64,
        ttft_ms: Option<u64>,
    ) {
        self.requests += 1;
        self.prompt_tokens += prompt_tokens.unwrap_or(0);
        self.cached_tokens += cached_tokens.unwrap_or(0);
        self.completion_tokens += completion_tokens.unwrap_or(0);
        self.duration_sum_ms += duration_ms;
        self.duration_avg_ms = self.duration_sum_ms / self.requests;
        if let Some(ttft) = ttft_ms {
            self.ttft_count += 1;
            self.ttft_sum_ms += ttft;
            self.ttft_min_ms = Some(self.ttft_min_ms.map_or(ttft, |m| m.min(ttft)));
            self.ttft_max_ms = Some(self.ttft_max_ms.map_or(ttft, |m| m.max(ttft)));
        }
        self.last_active = Some(now_secs());
    }
}

/// The on-disk / endpoint JSON document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsFile {
    /// Schema version (currently 1).
    pub version: u32,
    /// Unix seconds of the first startup (restored from the file if present).
    pub started_at: u64,
    /// Unix seconds of the most recent recorded request (or startup).
    pub updated_at: u64,
    /// Per-model counters, keyed by (resolved) model name.
    pub models: HashMap<String, StatsModel>,
}

/// The two counter sets for one model.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct StatsModel {
    /// Cumulative counters (persisted across restarts).
    pub total: ModelCounters,
    /// Counters since this process started.
    pub boot: ModelCounters,
}

impl StatsModel {
    /// Sum another node's counters into this one (cluster-peering stats
    /// union — bare names sum over every node serving the model).
    pub fn merge(&mut self, other: &StatsModel) {
        self.total.merge(&other.total);
        self.boot.merge(&other.boot);
    }
}

impl ModelCounters {
    /// Sum another set of counters into this one.  Minima/maxima combine
    /// accordingly; `duration_avg_ms` is recomputed from the sums.
    pub fn merge(&mut self, other: &ModelCounters) {
        let requests = self.requests + other.requests;
        self.requests = requests;
        self.prompt_tokens += other.prompt_tokens;
        self.cached_tokens += other.cached_tokens;
        self.completion_tokens += other.completion_tokens;
        self.ttft_count += other.ttft_count;
        self.ttft_sum_ms += other.ttft_sum_ms;
        self.ttft_min_ms = match (self.ttft_min_ms, other.ttft_min_ms) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.ttft_max_ms = match (self.ttft_max_ms, other.ttft_max_ms) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        self.duration_sum_ms += other.duration_sum_ms;
        self.duration_avg_ms = self.duration_sum_ms.checked_div(requests).unwrap_or(0);
        self.last_active = self.last_active.max(other.last_active);
    }
}

/// Unix seconds since the epoch.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// In-memory state plus the persistence path.
struct StatsState {
    /// `None` when stats are disabled (no config section).
    path: Option<PathBuf>,
    started_at: u64,
    totals: HashMap<String, ModelCounters>,
    boot: HashMap<String, ModelCounters>,
}

/// Shared statistics store; lives in `AppState`.
pub struct StatsStore {
    state: Mutex<StatsState>,
}

impl StatsStore {
    /// Create the store, optionally restoring `total` counters from an
    /// existing stats file.  A missing or unreadable file is not an
    /// error: the file is created on the first recorded request (or
    /// written on demand), and a corrupt file just resets `total`.
    pub fn new(stats_file: Option<&PathBuf>) -> Self {
        let (started_at, totals) = match stats_file {
            Some(path) => match std::fs::read_to_string(path)
                .ok()
                .and_then(|s| serde_json::from_str::<StatsFile>(&s).ok())
            {
                Some(doc) => (
                    doc.started_at.max(1),
                    doc.models.into_iter().map(|(k, v)| (k, v.total)).collect(),
                ),
                None => (now_secs().max(1), HashMap::new()),
            },
            None => (now_secs().max(1), HashMap::new()),
        };
        Self {
            state: Mutex::new(StatsState {
                path: stats_file.cloned(),
                started_at,
                totals,
                boot: HashMap::new(),
            }),
        }
    }

    /// Retarget (or disable/enable) the persistence path on config
    /// hot-reload.  Counter state is kept; only the destination file
    /// changes.
    pub fn set_path(&self, new_path: Option<&PathBuf>) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.path = new_path.cloned();
        drop(state);
        if new_path.is_some() {
            // Publish the current state immediately so the new file is
            // not empty after a reload.
            self.persist();
        }
    }

    /// The current persistence path (for tests / diagnostics).
    pub fn path(&self) -> Option<PathBuf> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .path
            .clone()
    }

    /// Record one served request and rewrite the JSON file atomically.
    /// No-op when stats are disabled.
    pub fn record(
        &self,
        model: &str,
        prompt_tokens: Option<u64>,
        cached_tokens: Option<u64>,
        completion_tokens: Option<u64>,
        duration_ms: u64,
        ttft_ms: Option<u64>,
    ) {
        let doc = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if state.path.is_none() {
                return;
            }
            state.totals.entry(model.to_owned()).or_default().apply(
                prompt_tokens,
                cached_tokens,
                completion_tokens,
                duration_ms,
                ttft_ms,
            );
            state.boot.entry(model.to_owned()).or_default().apply(
                prompt_tokens,
                cached_tokens,
                completion_tokens,
                duration_ms,
                ttft_ms,
            );
            StatsFile {
                version: 1,
                started_at: state.started_at,
                updated_at: now_secs(),
                models: state
                    .totals
                    .iter()
                    .map(|(name, c)| {
                        (
                            name.clone(),
                            StatsModel {
                                total: *c,
                                boot: state.boot.get(name).copied().unwrap_or_default(),
                            },
                        )
                    })
                    .collect(),
            }
        };
        self.persist_with(doc);
    }

    /// Snapshot of the current document (for the `/v1/stats` endpoint).
    pub fn snapshot(&self) -> Option<StatsFile> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.path.as_ref().map(|_| StatsFile {
            version: 1,
            started_at: state.started_at,
            updated_at: now_secs(),
            models: state
                .totals
                .iter()
                .map(|(name, c)| {
                    (
                        name.clone(),
                        StatsModel {
                            total: *c,
                            boot: state.boot.get(name).copied().unwrap_or_default(),
                        },
                    )
                })
                .collect(),
        })
    }

    /// Rewrite the file atomically (temp in the same directory + rename).
    /// Best-effort: failures warn, never fail the caller.
    fn persist(&self) {
        let Some(doc) = self.snapshot() else {
            return;
        };
        self.persist_with(doc);
    }

    fn persist_with(&self, doc: StatsFile) {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(path) = state.path.clone() else {
            return;
        };
        drop(state);

        let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) else {
            warn!("stats: no parent directory for {}", path.display());
            return;
        };
        let json = match serde_json::to_string_pretty(&doc) {
            Ok(j) => j,
            Err(e) => {
                warn!("stats: serialization failed, dropping write: {}", e);
                return;
            }
        };
        let tmp = parent.join(format!(
            ".{}-{}.tmp",
            path.file_name()
                .map(|f| f.to_string_lossy())
                .unwrap_or_default(),
            std::process::id()
        ));
        match std::fs::write(&tmp, json) {
            Ok(()) => match std::fs::rename(&tmp, &path) {
                Ok(()) => {}
                Err(e) => {
                    warn!("stats: rename failed, dropping write: {}", e);
                    let _ = std::fs::remove_file(&tmp);
                }
            },
            Err(e) => warn!("stats: write failed, dropping write: {}", e),
        }
    }
}

impl Default for StatsStore {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Convenience for handlers: build an [`Arc<StatsStore>`] from config.
pub fn from_config(stats: &Option<crate::config::StatsConfig>) -> Arc<StatsStore> {
    Arc::new(StatsStore::new(stats.as_ref().map(|s| &s.file)))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "llm-orch-stats-test-{}-{}",
            tag,
            uuid::Uuid::new_v4()
        ))
    }

    #[test]
    fn record_updates_both_sets_and_writes_file() {
        let path = temp_path("basic");
        let store = StatsStore::new(Some(&path));
        store.record("model-a", Some(100), Some(40), Some(50), 1200, Some(150));
        store.record("model-a", Some(10), None, Some(5), 800, Some(350));
        store.record("model-b", None, None, Some(7), 500, None);

        let doc: StatsFile =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let a = &doc.models["model-a"];
        assert_eq!(a.total.requests, 2);
        assert_eq!(a.total.prompt_tokens, 110);
        assert_eq!(a.total.cached_tokens, 40);
        assert_eq!(a.total.completion_tokens, 55);
        assert_eq!(a.total.ttft_count, 2);
        assert_eq!(a.total.ttft_min_ms, Some(150));
        assert_eq!(a.total.ttft_max_ms, Some(350));
        assert_eq!(a.total.duration_avg_ms, 1000);
        assert_eq!(a.boot, a.total);
        assert_eq!(doc.models["model-b"].total.completion_tokens, 7);
        assert_eq!(doc.models["model-b"].total.ttft_count, 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn request_without_usage_still_counts() {
        // Backends that report no usage (some OpenAI-compatible
        // embedding providers) must still advance requests/duration —
        // they just contribute no token totals.
        let path = temp_path("no-usage");
        let store = StatsStore::new(Some(&path));
        store.record("embedder", None, None, None, 42, None);
        store.record("embedder", Some(7), None, None, 8, None);

        let doc: StatsFile =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let m = &doc.models["embedder"];
        assert_eq!(m.total.requests, 2);
        assert_eq!(m.total.prompt_tokens, 7);
        assert_eq!(m.total.completion_tokens, 0);
        assert_eq!(m.total.duration_sum_ms, 50);
        assert_eq!(m.total.duration_avg_ms, 25);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn restores_total_from_file_and_zeroes_boot() {
        let path = temp_path("restore");
        let store = StatsStore::new(Some(&path));
        store.record("model-a", Some(100), None, Some(50), 1000, None);

        // Simulate a restart: fresh store, same file.
        let restarted = StatsStore::new(Some(&path));
        let before = restarted.snapshot().unwrap();
        assert_eq!(before.models["model-a"].total.requests, 1);
        assert_eq!(before.models["model-a"].boot.requests, 0);

        restarted.record("model-a", Some(10), None, Some(5), 100, None);
        let after = restarted.snapshot().unwrap();
        assert_eq!(after.models["model-a"].total.requests, 2);
        assert_eq!(after.models["model-a"].total.prompt_tokens, 110);
        assert_eq!(after.models["model-a"].boot.requests, 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn corrupt_file_resets_totals_without_panicking() {
        let path = temp_path("corrupt");
        std::fs::write(&path, "{not json").unwrap();
        let store = StatsStore::new(Some(&path));
        assert_eq!(store.snapshot().unwrap().models.len(), 0);
        store.record("m", Some(1), None, Some(1), 10, None);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn disabled_store_is_noop() {
        let store = StatsStore::new(None);
        store.record("model-a", Some(10), None, Some(5), 100, None);
        assert!(store.snapshot().is_none());
        assert!(store.path().is_none());
    }

    #[test]
    fn set_path_retargers_and_publishes() {
        let old = temp_path("retarget-old");
        let new = temp_path("retarget-new");
        let store = StatsStore::new(Some(&old));
        store.record("m", Some(10), None, Some(5), 100, None);
        store.set_path(Some(&new));
        assert!(std::fs::metadata(&new).is_ok());
        let doc: StatsFile = serde_json::from_str(&std::fs::read_to_string(&new).unwrap()).unwrap();
        assert_eq!(doc.models["m"].total.requests, 1);
        std::fs::remove_file(&old).ok();
        std::fs::remove_file(&new).ok();
    }

    #[test]
    fn missing_parent_dir_does_not_panic() {
        let path = temp_path("noparent").join("no-such-dir").join("stats.json");
        let store = StatsStore::new(Some(&path));
        store.record("m", Some(10), None, Some(5), 100, None);
        store.set_path(Some(&path));
    }
}
