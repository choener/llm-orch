// ── Token / model statistics ───────────────────────────────────────────────
//
// Opt-in via the top-level `stats:` config section.  Keeps per-model
// counters in two sets:
//
// - `total` — cumulative; restored from the stats file at startup, so it
//   survives restarts (the "bragging" total).
// - `boot`  — zeroed at every startup (ephemeral since-boot view).
//
// The same counters are additionally recorded per apikey label (per-user
// per-model matrix plus a per-user aggregate across models), so accounts
// can benchmark their own usage; the label is known at auth time
// (`ApiKey.label`) and threaded into every recording call.
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
    /// Per-user counters, keyed by apikey label.  Optional since schema
    /// version 1 grew the field: files written by older binaries load
    /// without it, and it is omitted while empty.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub users: HashMap<String, StatsUser>,
}

/// The two counter sets for one model.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct StatsModel {
    /// Cumulative counters (persisted across restarts).
    pub total: ModelCounters,
    /// Counters since this process started.
    pub boot: ModelCounters,
}

/// One user's counters (apikey label): a per-model breakdown plus an
/// aggregate across models, each in the `total`/`boot` two-set shape.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StatsUser {
    /// Aggregate cumulative counters across models.
    pub total: ModelCounters,
    /// Aggregate counters across models since this process started.
    pub boot: ModelCounters,
    /// Per-model breakdown (`total`/`boot` per model).
    #[serde(default)]
    pub models: HashMap<String, StatsModel>,
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
    /// Per-user counters, keyed by apikey label.
    users: HashMap<String, UserState>,
}

/// In-memory per-user counters: per-model breakdown plus aggregate.
#[derive(Default)]
struct UserState {
    /// Cumulative per-model counters.
    totals: HashMap<String, ModelCounters>,
    /// Per-model counters since this process started.
    boot: HashMap<String, ModelCounters>,
    /// Cumulative aggregate across models.
    agg_total: ModelCounters,
    /// Aggregate across models since this process started.
    agg_boot: ModelCounters,
}

impl StatsState {
    /// The full stats document for the current state (`updated_at` = now).
    fn doc(&self) -> StatsFile {
        StatsFile {
            version: 1,
            started_at: self.started_at,
            updated_at: now_secs(),
            models: self.models_doc(),
            users: self.users_doc(),
        }
    }

    fn models_doc(&self) -> HashMap<String, StatsModel> {
        self.totals
            .iter()
            .map(|(name, c)| {
                (
                    name.clone(),
                    StatsModel {
                        total: *c,
                        boot: self.boot.get(name).copied().unwrap_or_default(),
                    },
                )
            })
            .collect()
    }

    fn users_doc(&self) -> HashMap<String, StatsUser> {
        self.users
            .iter()
            .map(|(label, u)| {
                (
                    label.clone(),
                    StatsUser {
                        total: u.agg_total,
                        boot: u.agg_boot,
                        models: u
                            .totals
                            .iter()
                            .map(|(name, c)| {
                                (
                                    name.clone(),
                                    StatsModel {
                                        total: *c,
                                        boot: u.boot.get(name).copied().unwrap_or_default(),
                                    },
                                )
                            })
                            .collect(),
                    },
                )
            })
            .collect()
    }
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
        let (started_at, totals, users) = match stats_file {
            Some(path) => match std::fs::read_to_string(path)
                .ok()
                .and_then(|s| serde_json::from_str::<StatsFile>(&s).ok())
            {
                Some(doc) => (
                    doc.started_at.max(1),
                    doc.models.into_iter().map(|(k, v)| (k, v.total)).collect(),
                    // User counters restore like the per-model ones: `total`
                    // survives, `boot` zeroes (files without a `users` map —
                    // written by older binaries — restore empty).
                    doc.users
                        .into_iter()
                        .map(|(label, u)| {
                            (
                                label,
                                UserState {
                                    totals: u
                                        .models
                                        .into_iter()
                                        .map(|(k, v)| (k, v.total))
                                        .collect(),
                                    boot: HashMap::new(),
                                    agg_total: u.total,
                                    agg_boot: ModelCounters::default(),
                                },
                            )
                        })
                        .collect(),
                ),
                None => (now_secs().max(1), HashMap::new(), HashMap::new()),
            },
            None => (now_secs().max(1), HashMap::new(), HashMap::new()),
        };
        Self {
            state: Mutex::new(StatsState {
                path: stats_file.cloned(),
                started_at,
                totals,
                boot: HashMap::new(),
                users,
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
    ///
    /// Updates three places: the top-level per-model counters (unless
    /// `count_model` is false — cluster peering: a peer-forwarded request
    /// never counts into the local per-model metrics, the leaf's numbers
    /// are authoritative there), the user's per-model entry, and the
    /// user's aggregate across models.  The model key is always the
    /// master-side (effective-config) model name — the name the client
    /// requested.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        user: &str,
        model: &str,
        count_model: bool,
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
            if count_model {
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
            }
            // Per-user counters always record on the authenticating node
            // (docs/006, decision 2) — including peer-forwarded requests.
            let u = state.users.entry(user.to_owned()).or_default();
            u.totals.entry(model.to_owned()).or_default().apply(
                prompt_tokens,
                cached_tokens,
                completion_tokens,
                duration_ms,
                ttft_ms,
            );
            u.boot.entry(model.to_owned()).or_default().apply(
                prompt_tokens,
                cached_tokens,
                completion_tokens,
                duration_ms,
                ttft_ms,
            );
            u.agg_total.apply(
                prompt_tokens,
                cached_tokens,
                completion_tokens,
                duration_ms,
                ttft_ms,
            );
            u.agg_boot.apply(
                prompt_tokens,
                cached_tokens,
                completion_tokens,
                duration_ms,
                ttft_ms,
            );
            state.doc()
        };
        self.persist_with(doc);
    }

    /// Snapshot of the current document (for the `/v1/stats` endpoint).
    pub fn snapshot(&self) -> Option<StatsFile> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.path.as_ref().map(|_| state.doc())
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

// ── Shared response assembly ──────────────────────────────────────────────

/// `GET /v1/stats` response (also the MQTT `{prefix}/{node}/stats`
/// payload): per-model token counters plus the latest GPU metrics
/// snapshot.
#[derive(Debug, serde::Serialize)]
pub struct StatsResponse {
    /// `true` when the `stats:` config section is set.
    pub enabled: bool,
    /// Schema version (currently 1).
    pub version: u32,
    /// Unix seconds of the first startup (restored from the stats file
    /// if present; 0 when disabled).
    pub started_at: u64,
    /// Unix seconds of the most recent recorded request (0 when
    /// disabled).
    pub updated_at: u64,
    /// Per-model counters: `total` (across restarts) and `boot` (since
    /// this process started).
    pub models: HashMap<String, StatsModel>,
    /// Per-user counters (apikey label → per-model breakdown + aggregate),
    /// local node only.  Cluster peering: leaf `users` maps are NOT
    /// merged — labels are per-node namespaces (each node has its own
    /// apikeys file), so bare-name merging could conflate distinct
    /// users (docs/006, decision 2).
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub users: HashMap<String, StatsUser>,
    /// Latest GPU metrics snapshot (empty on CPU-only hosts).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gpus: Vec<crate::types::GpuStatus>,
}

/// Build the full stats response: local counters + GPU snapshot + the
/// concurrent cluster-peer union.
///
/// Single code path shared by the HTTP endpoint
/// (`handlers::stats_endpoint`) and the MQTT publisher (`mqtt.rs`) so
/// the wire payload can never drift from the endpoint.
///
/// Cluster peering (docs/006, decision 9): peers' stats are fetched
/// concurrently — 10 s per-peer timeout, last-known fallback,
/// singleflight per peer; the master never counts forwarded requests
/// itself, the leaves' numbers are authoritative for their models.
pub async fn build_response(
    stats: &StatsStore,
    gpu: &tokio::sync::RwLock<Vec<crate::gpu::GpuMetrics>>,
    peers: &crate::peers::PeerRegistry,
) -> StatsResponse {
    let gpu_metrics = gpu.read().await.clone();
    let mut gpus: Vec<crate::types::GpuStatus> = gpu_metrics
        .iter()
        .map(|g| {
            let vram_util_pct = if g.vram_total_bytes > 0 {
                (g.vram_used_bytes as f64 / g.vram_total_bytes as f64) * 100.0
            } else {
                0.0
            };
            crate::types::GpuStatus {
                index: g.index,
                pci_slot: g.pci_slot.clone(),
                vram_vendor: g.vram_vendor.clone(),
                node: None,
                vram_used_bytes: g.vram_used_bytes,
                vram_total_bytes: g.vram_total_bytes,
                vram_util_pct,
                temperature_c: g.temperature_c,
                power_w: g.power_w,
                gpu_busy_pct: g.gpu_busy_pct,
                sclk_mhz: g.sclk_mhz,
                mclk_mhz: g.mclk_mhz,
            }
        })
        .collect();
    let doc = stats.snapshot();
    let stats_enabled = doc.is_some();
    let (version, started_at) = doc
        .as_ref()
        .map(|d| (d.version, d.started_at))
        .unwrap_or((1, 0));
    let mut updated_at = doc.as_ref().map(|d| d.updated_at).unwrap_or(0);
    // Local users map: passed through verbatim, never merged with peers'
    // (labels are per-node namespaces — see `StatsResponse::users`).
    let (local_models, users) = doc.map(|d| (d.models, d.users)).unwrap_or_default();
    let peer_names = peers.peer_names();
    let mut peer_models: Vec<(String, HashMap<String, StatsModel>)> = Vec::new();
    if !peer_names.is_empty() {
        let fetched =
            futures_util::future::join_all(peer_names.iter().map(|p| peers.peer_stats(p))).await;
        for (peer, result) in peer_names.iter().zip(fetched) {
            let Some((value, _fetched_at)) = result else {
                continue;
            };
            if let Some(ts) = value.get("updated_at").and_then(|v| v.as_u64()) {
                updated_at = updated_at.max(ts);
            }
            if let Ok(mut peer_gpus) =
                serde_json::from_value::<Vec<crate::types::GpuStatus>>(value["gpus"].clone())
            {
                for gpu in &mut peer_gpus {
                    gpu.node = Some(peer.clone());
                }
                gpus.extend(peer_gpus);
            }
            if let Ok(models) =
                serde_json::from_value::<HashMap<String, StatsModel>>(value["models"].clone())
            {
                peer_models.push((peer.clone(), models));
            }
        }
    }

    // ── Union model map ───────────────────────────────────────────────
    // - node-qualified entries (`<node>/<model>` for local models,
    //   `<peer>/<model>` for leaf models) carry that node's numbers;
    // - bare entries are the sum over every node serving the name.
    let mut models: HashMap<String, StatsModel> = local_models;
    // Self-qualification (`<node>/<model>` copies of local entries) is a
    // gateway feature: a node without peers emits bare-only stats so a
    // consuming master re-exports each entry exactly once (no nested
    // `cortana/cortana/...` qualification).
    let is_gateway = !peer_names.is_empty();
    if is_gateway && let Some(node) = peers.node_name() {
        let qualified: Vec<(String, StatsModel)> = models
            .iter()
            .map(|(name, m)| (format!("{node}/{name}"), *m))
            .collect();
        for (name, m) in qualified {
            models.insert(name, m);
        }
    }
    for (peer, pmodels) in peer_models {
        for (leaf_name, m) in pmodels {
            // Qualified leaf entry: the leaf's numbers verbatim.
            models.insert(format!("{peer}/{leaf_name}"), m);
            // Bare entry: sum over every node serving the name.
            let bare = models.entry(leaf_name).or_default();
            bare.merge(&m);
        }
    }
    // `users` also counts towards `enabled`: a pure gateway serves every
    // request via peers (no local per-model counters) but still records
    // per-user counters for what it authenticated.
    let enabled = stats_enabled || !models.is_empty() || !users.is_empty();

    StatsResponse {
        enabled,
        version,
        started_at,
        updated_at,
        models,
        users,
        gpus,
    }
}

/// The peers' raw `StatsResponse` documents currently cached by the
/// registry (`(peer_name, doc)`), used by the MQTT publisher to
/// re-export leaf stats without extra fetches.
pub async fn peer_stat_docs(
    peers: &crate::peers::PeerRegistry,
) -> Vec<(String, serde_json::Value)> {
    let mut out = Vec::new();
    for peer in peers.peer_names() {
        if let Some((value, _)) = peers.peer_stats(&peer).await {
            out.push((peer, value));
        }
    }
    out
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
        store.record(
            "u",
            "model-a",
            true,
            Some(100),
            Some(40),
            Some(50),
            1200,
            Some(150),
        );
        store.record(
            "u",
            "model-a",
            true,
            Some(10),
            None,
            Some(5),
            800,
            Some(350),
        );
        store.record("u", "model-b", true, None, None, Some(7), 500, None);

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
        store.record("u", "embedder", true, None, None, None, 42, None);
        store.record("u", "embedder", true, Some(7), None, None, 8, None);

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
        store.record("u", "model-a", true, Some(100), None, Some(50), 1000, None);

        // Simulate a restart: fresh store, same file.
        let restarted = StatsStore::new(Some(&path));
        let before = restarted.snapshot().unwrap();
        assert_eq!(before.models["model-a"].total.requests, 1);
        assert_eq!(before.models["model-a"].boot.requests, 0);

        restarted.record("u", "model-a", true, Some(10), None, Some(5), 100, None);
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
        store.record("u", "m", true, Some(1), None, Some(1), 10, None);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn disabled_store_is_noop() {
        let store = StatsStore::new(None);
        store.record("u", "model-a", true, Some(10), None, Some(5), 100, None);
        assert!(store.snapshot().is_none());
        assert!(store.path().is_none());
    }

    #[test]
    fn set_path_retargers_and_publishes() {
        let old = temp_path("retarget-old");
        let new = temp_path("retarget-new");
        let store = StatsStore::new(Some(&old));
        store.record("u", "m", true, Some(10), None, Some(5), 100, None);
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
        store.record("u", "m", true, Some(10), None, Some(5), 100, None);
        store.set_path(Some(&path));
    }

    #[test]
    fn record_updates_user_matrix_and_aggregate() {
        let path = temp_path("users");
        let store = StatsStore::new(Some(&path));
        store.record(
            "alice",
            "model-a",
            true,
            Some(100),
            None,
            Some(50),
            1000,
            None,
        );
        store.record(
            "alice",
            "model-b",
            true,
            Some(10),
            None,
            Some(5),
            500,
            Some(80),
        );
        store.record("bob", "model-a", true, Some(7), None, None, 42, None);

        let doc: StatsFile =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let alice = &doc.users["alice"];
        // Aggregate across models.
        assert_eq!(alice.total.requests, 2);
        assert_eq!(alice.total.prompt_tokens, 110);
        assert_eq!(alice.total.completion_tokens, 55);
        assert_eq!(alice.total.duration_sum_ms, 1500);
        assert_eq!(alice.total.duration_avg_ms, 750);
        assert_eq!(alice.total.ttft_count, 1);
        assert_eq!(alice.total.ttft_min_ms, Some(80));
        // Per-model breakdown.
        assert_eq!(alice.models["model-a"].total.requests, 1);
        assert_eq!(alice.models["model-a"].total.prompt_tokens, 100);
        assert_eq!(alice.models["model-a"].total.ttft_count, 0);
        assert_eq!(alice.models["model-b"].total.requests, 1);
        assert_eq!(alice.models["model-b"].total.ttft_count, 1);
        // Within one boot, boot mirrors total — aggregate and matrix.
        assert_eq!(alice.boot, alice.total);
        assert_eq!(alice.models["model-a"].boot, alice.models["model-a"].total);
        // A second user's row is independent.
        let bob = &doc.users["bob"];
        assert_eq!(bob.total.requests, 1);
        assert_eq!(bob.total.prompt_tokens, 7);
        assert_eq!(bob.models["model-a"].total.requests, 1);
        assert_eq!(bob.models["model-a"].total.completion_tokens, 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn peer_model_request_counts_per_user_only() {
        // A peer-forwarded request (count_model=false) never touches the
        // top-level per-model counters — the leaf's numbers are
        // authoritative for its models — but the authenticating node still
        // counts it per-user, keyed by the master-side model name
        // (docs/006, decisions 2 and 9).
        let path = temp_path("peer-user");
        let store = StatsStore::new(Some(&path));
        store.record(
            "alice",
            "cortana/qwen",
            false,
            Some(100),
            None,
            Some(50),
            1000,
            None,
        );

        let doc: StatsFile =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(
            !doc.models.contains_key("cortana/qwen"),
            "per-model counters must skip peer-forwarded requests"
        );
        let alice = &doc.users["alice"];
        assert_eq!(alice.total.requests, 1);
        assert_eq!(alice.total.prompt_tokens, 100);
        assert_eq!(alice.models["cortana/qwen"].total.requests, 1);
        assert_eq!(alice.models["cortana/qwen"].total.completion_tokens, 50);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn restores_user_totals_and_zeroes_boot() {
        let path = temp_path("users-restore");
        let store = StatsStore::new(Some(&path));
        store.record(
            "alice",
            "model-a",
            true,
            Some(100),
            None,
            Some(50),
            1000,
            None,
        );

        // Simulate a restart: fresh store, same file.
        let restarted = StatsStore::new(Some(&path));
        let before = restarted.snapshot().unwrap();
        let alice = &before.users["alice"];
        assert_eq!(alice.total.requests, 1);
        assert_eq!(alice.total.prompt_tokens, 100);
        assert_eq!(alice.boot.requests, 0, "boot user counters reset");
        assert_eq!(alice.models["model-a"].total.requests, 1);
        assert_eq!(alice.models["model-a"].boot.requests, 0);

        restarted.record("alice", "model-a", true, Some(10), None, Some(5), 100, None);
        let after = restarted.snapshot().unwrap();
        let alice = &after.users["alice"];
        assert_eq!(alice.total.requests, 2);
        assert_eq!(alice.total.prompt_tokens, 110);
        assert_eq!(alice.boot.requests, 1);
        assert_eq!(alice.models["model-a"].total.prompt_tokens, 110);
        assert_eq!(alice.models["model-a"].boot.requests, 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn old_file_without_users_loads() {
        // Schema stays version 1: `users` is optional, so files written
        // by older binaries load without it.
        let path = temp_path("old-file");
        let counters = r#"{"requests":3,"prompt_tokens":30,"cached_tokens":0,"completion_tokens":9,"ttft_count":0,"ttft_sum_ms":0,"duration_sum_ms":900,"duration_avg_ms":300}"#;
        std::fs::write(
            &path,
            format!(
                r#"{{"version":1,"started_at":100,"updated_at":200,"models":{{"m":{{"total":{counters},"boot":{counters}}}}}}}"#
            ),
        )
        .unwrap();
        let store = StatsStore::new(Some(&path));
        let doc = store.snapshot().unwrap();
        assert_eq!(doc.version, 1);
        assert_eq!(doc.models["m"].total.requests, 3);
        assert!(doc.users.is_empty());

        // New records populate `users` next to the restored models.
        store.record("alice", "m", true, Some(10), None, Some(5), 100, None);
        let doc: StatsFile =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(doc.models["m"].total.requests, 4);
        assert_eq!(doc.users["alice"].total.requests, 1);
        assert_eq!(doc.users["alice"].models["m"].total.requests, 1);
        std::fs::remove_file(&path).ok();
    }
}
