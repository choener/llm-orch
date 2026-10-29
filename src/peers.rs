// ── Cluster peering ─────────────────────────────────────────────────────────
//
// Star-topology peering (see `docs/006-cluster-peering.md`): this instance
// (the master) re-exports the models of further llm-orch instances (leaf
// peers).  The master is a thin shim: peers are ordinary llm-orch
// instances that see a single ordinary client holding one apikey.
//
// Core mechanics:
// - Peers are *config* (hot-reloadable); their model catalogs are runtime
//   state behind a TTL, refreshed lazily — no background poller.
// - Peer models are materialized as remote `ModelConfig` entries named
//   `<peer>/<model>` (pinned form) plus a bare-name alias when the name is
//   unique and does not collide with a local model or user alias.  The
//   materialized entries ride the existing remote-model machinery:
//   health probe provisioning, forwarding, queueing.
// - The leaf-side model name is carried in `ModelConfig::upstream_model`
//   and rewritten into request bodies at the forwarding boundary.

use crate::config::{
    self, AliasConfig, AliasPolicy, Config, MakeRoomMode, ModelConfig, PeerConfig,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Timeout for a single peer catalog / stats fetch.  A slow leaf must not
/// hang request-path refreshes or stats queries.
const PEER_FETCH_TIMEOUT: Duration = Duration::from_secs(10);

/// A model entry discovered in a peer's catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerModel {
    /// Leaf-side model name (as the peer itself knows it).
    pub name: String,
    /// Declared context length, when the peer reports one.
    pub context_length: Option<usize>,
    /// Capabilities the peer advertised (from its `architecture` object);
    /// carried through so this node's `/v1/models` keeps advertising them.
    pub capabilities: Vec<String>,
}

/// A peer's last successfully fetched model catalog.
#[derive(Debug, Clone)]
pub struct Catalog {
    pub models: Vec<PeerModel>,
    pub fetched_at: Instant,
}

#[derive(Debug, Clone)]
struct PeerState {
    config: PeerConfig,
    /// Last successful catalog fetch (`None` = never fetched).
    catalog: Option<Catalog>,
    /// Last catalog fetch error (cleared on success).  Kept for
    /// observability only — never blocks serving the cached catalog.
    last_error: Option<String>,
}

impl PeerState {
    fn new(config: PeerConfig) -> Self {
        Self {
            config,
            catalog: None,
            last_error: None,
        }
    }
}

/// Master-side registry of peers, catalogs, and forwarded-request IDs.
///
/// `reconcile` installs the peer set from the (file-level) config; the
/// catalog fetchers and `effective_config` are the working core.  Interior
/// mutability via `std::sync::RwLock` — critical sections are short and
/// never held across `.await`.
pub struct PeerRegistry {
    client: reqwest::Client,
    inner: RwLock<Inner>,
    /// Serializes concurrent catalog refreshes per peer (singleflight):
    /// the request path refreshes on resolution misses, and a slow leaf
    /// must not accumulate parallel fetches.
    fetch_locks: RwLock<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Same singleflight idea for the on-demand stats/info fetches.
    stats_locks: RwLock<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Last-known peer `/v1/stats` responses (served when a fetch times
    /// out or another query is already in flight).
    stats_cache: RwLock<HashMap<String, (serde_json::Value, Instant)>>,
    /// Runtime handles (shared config + instance manager), installed once
    /// at startup.  Holding them here lets the request path and the
    /// forwarding boundary trigger materialization / refresh without
    /// threading `AppState` through every helper.
    runtime: RwLock<Option<RuntimeHandles>>,
}

#[derive(Clone)]
struct RuntimeHandles {
    cfg: Arc<tokio::sync::RwLock<Config>>,
    manager: Arc<crate::scheduler::InstanceManager>,
}

struct Inner {
    node_name: Option<String>,
    peers: HashMap<String, PeerState>,
    key_files: HashMap<String, String>,
}

impl PeerRegistry {
    pub fn new(client: reqwest::Client) -> Self {
        Self {
            client,
            inner: RwLock::new(Inner {
                node_name: None,
                peers: HashMap::new(),
                key_files: HashMap::new(),
            }),
            fetch_locks: RwLock::new(HashMap::new()),
            stats_locks: RwLock::new(HashMap::new()),
            stats_cache: RwLock::new(HashMap::new()),
            runtime: RwLock::new(None),
        }
    }

    /// Install the runtime handles (shared config + instance manager).
    /// Called once at startup, after the manager and config exist.
    pub fn install_runtime(
        &self,
        cfg: Arc<tokio::sync::RwLock<Config>>,
        manager: Arc<crate::scheduler::InstanceManager>,
    ) {
        *self.runtime.write().unwrap() = Some(RuntimeHandles { cfg, manager });
    }

    /// Install the peer set from the (file-level) config.  Existing
    /// catalogs for surviving peers are kept; removed peers drop theirs.
    pub fn reconcile(&self, cfg: &Config) {
        let mut inner = self.inner.write().unwrap();
        inner.node_name = cfg.node_name.clone();
        inner.key_files = config::load_key_files(&cfg.key_files).unwrap_or_default();
        let new_peers: HashMap<String, PeerState> = cfg
            .peers
            .iter()
            .map(|p| {
                let existing = inner.peers.get(&p.name);
                let state = match existing {
                    // Keep the cached catalog; the TTL check decides
                    // whether it is fresh enough to serve.
                    Some(old) if old.config.url == p.url => PeerState {
                        config: p.clone(),
                        catalog: old.catalog.clone(),
                        last_error: old.last_error.clone(),
                    },
                    _ => PeerState::new(p.clone()),
                };
                (p.name.clone(), state)
            })
            .collect();
        inner.peers = new_peers;
    }

    pub fn node_name(&self) -> Option<String> {
        self.inner.read().unwrap().node_name.clone()
    }

    /// Peer names currently configured.
    pub fn peer_names(&self) -> Vec<String> {
        self.inner.read().unwrap().peers.keys().cloned().collect()
    }

    /// A peer's configuration (with `remote_auth` resolved against the
    /// current key-file contents).
    pub fn resolved_peer(&self, name: &str) -> Option<(PeerConfig, Option<String>)> {
        let inner = self.inner.read().unwrap();
        let state = inner.peers.get(name)?;
        Some((
            state.config.clone(),
            state
                .config
                .resolved_auth(&inner.key_files)
                .filter(|a| !a.is_empty()),
        ))
    }

    fn peer_state(&self, name: &str) -> Option<PeerState> {
        self.inner.read().unwrap().peers.get(name).cloned()
    }

    /// Resolve a peer's `remote_auth` against the current key-file
    /// contents, with the read guard strictly scoped inside this helper
    /// (never held across an `.await`).
    fn current_auth(&self, config: &PeerConfig) -> Option<String> {
        let inner = self.inner.read().unwrap();
        config
            .resolved_auth(&inner.key_files)
            .filter(|a| !a.is_empty())
    }

    fn fetch_lock(&self, peer: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.fetch_locks.write().unwrap();
        locks
            .entry(peer.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    fn stats_lock(&self, peer: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.stats_locks.write().unwrap();
        locks
            .entry(peer.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    // ── Catalog fetching ───────────────────────────────────────────────────

    /// Fetch a peer's model catalog (forced — bypasses the TTL).  Returns
    /// `true` when the catalog content changed (including a first fetch).
    pub async fn refresh_catalog(&self, peer: &str) -> bool {
        let state = match self.peer_state(peer) {
            Some(s) => s,
            None => return false,
        };
        // Singleflight: concurrent resolution misses share one fetch.
        let lock = self.fetch_lock(peer);
        let _guard = lock.lock().await;
        // Another waiter may have completed the fetch while we queued.
        if let Some(s) = self.peer_state(peer)
            && let Some(catalog) = &s.catalog
            && catalog.fetched_at.elapsed() < Duration::from_secs(s.config.catalog_ttl_secs)
        {
            return false;
        }

        let auth = self.current_auth(&state.config);

        let url = format!("{}/v1/models", state.config.url.trim_end_matches('/'));
        let fetched = fetch_model_catalog(&self.client, &url, auth.as_deref()).await;
        let mut inner = self.inner.write().unwrap();
        let slot = match inner.peers.get_mut(peer) {
            Some(s) => s,
            None => return false,
        };
        match fetched {
            Ok(models) => {
                let models = apply_filters(&state.config, &models);
                let changed = slot
                    .catalog
                    .as_ref()
                    .map(|c| c.models != models)
                    .unwrap_or(true);
                slot.last_error = None;
                slot.catalog = Some(Catalog {
                    models,
                    fetched_at: Instant::now(),
                });
                debug!(peer = %peer, "peer catalog refreshed");
                changed
            }
            Err(e) => {
                slot.last_error = Some(e.clone());
                warn!(peer = %peer, error = %e, "peer catalog fetch failed — keeping cached catalog");
                false
            }
        }
    }

    /// Refresh every peer whose catalog is older than its TTL.  Returns
    /// `true` when at least one catalog changed.
    pub async fn refresh_stale_catalogs(&self) -> bool {
        let stale: Vec<String> = {
            let inner = self.inner.read().unwrap();
            inner
                .peers
                .iter()
                .filter(|(_, s)| {
                    s.catalog
                        .as_ref()
                        .map(|c| {
                            c.fetched_at.elapsed() >= Duration::from_secs(s.config.catalog_ttl_secs)
                        })
                        .unwrap_or(true)
                })
                .map(|(name, _)| name.clone())
                .collect()
        };
        let mut changed = false;
        for peer in stale {
            changed |= self.refresh_catalog(&peer).await;
        }
        changed
    }

    /// Refresh all catalogs regardless of TTL (pre-hot-reload path).
    pub async fn refresh_all_catalogs(&self) -> bool {
        let names = self.peer_names();
        let mut changed = false;
        for peer in names {
            changed |= self.refresh_catalog(&peer).await;
        }
        changed
    }

    // ── Materialization ───────────────────────────────────────────────────

    /// Build the effective config: local (user) models and aliases plus
    /// materialized peer entries.  `base` may itself be an effective
    /// config — virtual entries are filtered out by their `peer` marker
    /// before regeneration, so this is idempotent.
    pub fn effective_config(&self, base: &Config) -> Config {
        let inner = self.inner.read().unwrap();

        let local_models: Vec<&ModelConfig> =
            base.models.iter().filter(|m| m.peer.is_none()).collect();
        let local_names: HashSet<&str> = local_models.iter().map(|m| m.name.as_str()).collect();
        let local_alias_names: HashSet<&str> = base
            .aliases
            .iter()
            .filter(|a| a.peer.is_none())
            .map(|a| a.name.as_str())
            .collect();

        let mut virtual_models: Vec<ModelConfig> = Vec::new();
        // bare leaf model name → qualified peer model names serving it.
        let mut bare_targets: HashMap<&str, Vec<String>> = HashMap::new();

        let node_prefix = inner.node_name.as_ref().map(|n| format!("{n}/"));

        for (peer_name, state) in &inner.peers {
            let Some(catalog) = &state.catalog else {
                continue;
            };
            for model in &catalog.models {
                // Cycle guard (docs/006, topology non-goal): never
                // re-export a model that already carries our own node
                // qualification — that means the peer re-exported our
                // own models back at us.
                if let Some(prefix) = &node_prefix
                    && model.name.starts_with(prefix.as_str())
                {
                    warn!(
                        peer = %peer_name,
                        model = %model.name,
                        "peer re-exports our own node-qualified model — skipping (cycle guard)"
                    );
                    continue;
                }
                let qualified = format!("{peer_name}/{}", model.name);
                if local_names.contains(qualified.as_str()) {
                    // A local model cannot contain '/', so this cannot
                    // happen for names from YAML — only for double
                    // materialization, which the peer marker prevents.
                    continue;
                }
                virtual_models.push(ModelConfig {
                    name: qualified,
                    cmd: state.config.url.clone(),
                    remote_auth: state.config.remote_auth.clone(),
                    upstream_model: Some(model.name.clone()),
                    peer: Some(peer_name.clone()),
                    context_length: model.context_length.unwrap_or(4096),
                    capabilities: model.capabilities.clone(),
                    max_instances: 1,
                    max_concurrent: state.config.max_concurrent,
                    // Remote instances are health-probe registry entries;
                    // the leaf owns the real lifecycle.  Generous queue so
                    // requests are not double-held by both layers.
                    idle_ttl: 3600,
                    queue_depth: 256,
                    ..model_defaults()
                });
                if !local_names.contains(model.name.as_str())
                    && !local_alias_names.contains(model.name.as_str())
                {
                    bare_targets
                        .entry(model.name.as_str())
                        .or_default()
                        .push(format!("{peer_name}/{}", model.name));
                }
            }
        }

        let mut virtual_aliases: Vec<AliasConfig> = bare_targets
            .into_iter()
            .map(|(bare, mut targets)| {
                // Deterministic order regardless of peer map iteration.
                targets.sort();
                AliasConfig {
                    name: bare.to_string(),
                    targets,
                    policy: AliasPolicy::PreferLoaded,
                    make_room: MakeRoomMode::None,
                    drain_timeout: 300,
                    peer: Some(String::new()),
                }
            })
            .collect();

        let mut cfg = base.clone();
        cfg.models = local_models.into_iter().cloned().collect();
        cfg.models.extend(virtual_models);
        cfg.aliases = base
            .aliases
            .iter()
            .filter(|a| a.peer.is_none())
            .cloned()
            .collect();
        virtual_aliases.sort_by(|a, b| a.name.cmp(&b.name));
        cfg.aliases.extend(virtual_aliases);
        cfg
    }

    /// Refresh stale catalogs; materialize when anything changed.
    /// Called from the request path on resolution misses and from
    /// `/v1/models` when a catalog TTL expired.
    pub async fn refresh_and_materialize(&self) {
        if self.refresh_stale_catalogs().await {
            self.rematerialize().await;
        }
    }

    /// Swap the materialized effective config into the shared config and
    /// instance manager.  `reconcile_config` is idempotent for unchanged
    /// model sets.  No-op when runtime handles are not installed (unit
    /// tests).
    pub async fn rematerialize(&self) {
        let Some(runtime) = self.runtime.read().unwrap().clone() else {
            return;
        };
        let effective = {
            let current = runtime.cfg.read().await;
            self.effective_config(&current)
        };
        runtime.manager.reconcile_config(&effective).await;
        let model_count = effective.models.len();
        *runtime.cfg.write().await = effective;
        debug!(models = model_count, "effective config materialized");
    }

    /// Fire-and-forget refresh of one peer's catalog + materialization,
    /// used after a leaf answered 404 (its model set drifted).  The
    /// current request still fails; subsequent ones recover.
    pub fn schedule_refresh(self: &Arc<Self>, peer: &str) {
        let registry = Arc::clone(self);
        let peer = peer.to_string();
        tokio::spawn(async move {
            registry.refresh_catalog(&peer).await;
            registry.rematerialize().await;
        });
    }

    // ── Stats / info union ────────────────────────────────────────────────

    /// Fetch a peer's `/v1/stats` on demand.  Singleflight per peer
    /// (a concurrent query rides on the in-flight fetch via the cached
    /// value); a fetch timeout or error serves the last-known response.
    pub async fn peer_stats(&self, peer: &str) -> Option<(serde_json::Value, Instant)> {
        let state = self.peer_state(peer)?;
        let lock = self.stats_lock(peer);
        let Ok(_guard) = lock.try_lock() else {
            // Another query is already fetching — serve last-known.
            return self.stats_cache.read().unwrap().get(peer).cloned();
        };
        let auth = self.current_auth(&state.config);
        let url = format!("{}/v1/stats", state.config.url.trim_end_matches('/'));
        let fetched = tokio::time::timeout(
            PEER_FETCH_TIMEOUT,
            fetch_json(&self.client, &url, auth.as_deref()),
        )
        .await;
        match fetched {
            Ok(Ok(value)) => {
                let entry = (value, Instant::now());
                self.stats_cache
                    .write()
                    .unwrap()
                    .insert(peer.to_string(), entry.clone());
                Some(entry)
            }
            Ok(Err(e)) => {
                warn!(peer = %peer, error = %e, "peer stats fetch failed — serving last-known");
                self.stats_cache.read().unwrap().get(peer).cloned()
            }
            Err(_) => {
                warn!(peer = %peer, "peer stats fetch timed out — serving last-known");
                self.stats_cache.read().unwrap().get(peer).cloned()
            }
        }
    }

    /// Fetch a peer's `/v1/info` on demand (same singleflight semantics,
    /// separate lock so stats and info queries do not serialize each
    /// other).  Not cached beyond the in-flight window — `/v1/info` is a
    /// debugging endpoint; a failed fetch just omits the peer.
    pub async fn peer_info(&self, peer: &str) -> Option<crate::types::InfoResponse> {
        let state = self.peer_state(peer)?;
        let lock = self.stats_lock(peer);
        let Ok(_guard) = lock.try_lock() else {
            return None;
        };
        let auth = self.current_auth(&state.config);
        let url = format!("{}/v1/info", state.config.url.trim_end_matches('/'));
        match tokio::time::timeout(
            PEER_FETCH_TIMEOUT,
            fetch_json(&self.client, &url, auth.as_deref()),
        )
        .await
        {
            Ok(Ok(value)) => serde_json::from_value(value)
                .map_err(|e| {
                    warn!(peer = %peer, error = %e, "peer /v1/info response unparsable");
                    e
                })
                .ok(),
            Ok(Err(e)) => {
                warn!(peer = %peer, error = %e, "peer info fetch failed");
                None
            }
            Err(_) => {
                warn!(peer = %peer, "peer info fetch timed out");
                None
            }
        }
    }
}

/// Baseline field values for materialized peer models (everything the
/// `..model_defaults()` spread does not spell out explicitly).
fn model_defaults() -> ModelConfig {
    ModelConfig {
        name: String::new(),
        context_length: 4096,
        max_instances: 1,
        max_concurrent: 4,
        priority: 0,
        vram: 0,
        ram: 0,
        idle_ttl: 300,
        queue_depth: 10,
        cmd: String::new(),
        capabilities: Vec::new(),
        vulkan_devices: Default::default(),
        cuda_devices: Default::default(),
        rocm_devices: Default::default(),
        gpus: 1,
        debug_log: None,
        autoscale: None,
        spawn_timeout_secs: None,
        remote_auth: None,
        upstream_model: None,
        peer: None,
    }
}

/// Fetch and parse a peer's `/v1/models` (OpenAI list format, with the
/// optional `context_length` extension llm-orch itself emits).  The
/// peer's node-qualified self-listing (announced via the
/// `x-llm-orch-node` response header) is filtered out so it is not
/// re-exported under a second qualification.
async fn fetch_model_catalog(
    client: &reqwest::Client,
    url: &str,
    auth: Option<&str>,
) -> Result<Vec<PeerModel>, String> {
    let mut builder = client.get(url).timeout(PEER_FETCH_TIMEOUT);
    if let Some(auth) = auth {
        builder = builder.header("Authorization", auth);
    }
    let resp = builder
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("peer returned {status}"));
    }
    let peer_node = resp
        .headers()
        .get("x-llm-orch-node")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("invalid JSON: {e}"))?;
    let data = body
        .pointer("/data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| "response has no data array".to_string())?;
    Ok(parse_catalog_data(data, peer_node.as_deref()))
}

/// Turn a peer's `/v1/models` `data` array into catalog entries.
///
/// Besides the name and llm-orch's `context_length` extension, this is
/// where a peer's capability advertisement (`architecture`, llama.cpp
/// shape) becomes llm-orch's own `capabilities`, so this node keeps
/// advertising a leaf's decision models.
fn parse_catalog_data(data: &[serde_json::Value], peer_node: Option<&str>) -> Vec<PeerModel> {
    data.iter()
        .filter_map(|entry| {
            let name = entry.get("id")?.as_str()?.to_string();
            // The peer's own node-qualified listing is a display alias
            // for clients, not a distinct model — re-exporting it would
            // nest the qualification.
            if let Some(node) = &peer_node
                && name.starts_with(&format!("{node}/"))
            {
                return None;
            }
            let context_length = entry
                .get("context_length")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            // Capabilities ride along: the peer's `architecture` object is
            // the only statement about a model the peer may never have
            // loaded, and this node republishes it.  Values llm-orch does
            // not model (`audio` output, future modalities) are dropped by
            // `Architecture::capabilities`, never guessed at.
            let capabilities = entry
                .get("architecture")
                .and_then(|a| serde_json::from_value::<crate::types::Architecture>(a.clone()).ok())
                .map(|a| a.capabilities())
                .unwrap_or_default();
            Some(PeerModel {
                name,
                context_length,
                capabilities,
            })
        })
        .collect()
}

/// Fetch a JSON document from a peer endpoint.
async fn fetch_json(
    client: &reqwest::Client,
    url: &str,
    auth: Option<&str>,
) -> Result<serde_json::Value, String> {
    let mut builder = client.get(url).timeout(PEER_FETCH_TIMEOUT);
    if let Some(auth) = auth {
        builder = builder.header("Authorization", auth);
    }
    let resp = builder
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("peer returned {status}"));
    }
    resp.json().await.map_err(|e| format!("invalid JSON: {e}"))
}

// ── Model-name resolution ───────────────────────────────────────────────────

/// Strip client-side path prefixes from a requested model name:
/// - a leading `<node_name>/` (this instance's own node qualification),
/// - a leading `<peer>/` is *kept* (pinned peer model),
/// - any other leading segment is an opaque client provider prefix
///   (e.g. pi's `llm/`) and is stripped — repeatedly.
///
/// Returns the name to look up as a local model / alias / virtual peer
/// model.
pub fn strip_client_prefixes(
    requested: &str,
    node_name: Option<&str>,
    peers: &[PeerConfig],
) -> String {
    let mut name = requested;
    while let Some((head, rest)) = name.split_once('/') {
        if Some(head) == node_name {
            // Our own qualification — strip and continue scanning.
            name = rest;
            continue;
        }
        if peers.iter().any(|p| p.name == head) {
            // Pinned peer form — keep as-is.
            break;
        }
        // Unknown first segment: a client provider prefix.  Strip.
        name = rest;
    }
    name.to_string()
}

/// Per-peer observability snapshot for `/admin/status` and logs.
pub fn peer_status_snapshot(registry: &PeerRegistry) -> Vec<PeerStatus> {
    let inner = registry.inner.read().unwrap();
    inner
        .peers
        .values()
        .map(|state| PeerStatus {
            name: state.config.name.clone(),
            url: state.config.url.clone(),
            catalog_age_secs: state
                .catalog
                .as_ref()
                .map(|c| c.fetched_at.elapsed().as_secs()),
            catalog_model_count: state.catalog.as_ref().map(|c| c.models.len()),
            last_error: state.last_error.clone(),
        })
        .collect()
}

/// Observability record for one peer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PeerStatus {
    pub name: String,
    pub url: String,
    pub catalog_age_secs: Option<u64>,
    pub catalog_model_count: Option<usize>,
    pub last_error: Option<String>,
}

/// Apply a peer's `include`/`exclude` glob patterns to a fetched catalog.
/// Patterns support `*` (any run of characters) and `?` (one character);
/// no pattern = no filtering.
pub(crate) fn apply_filters(config: &PeerConfig, models: &[PeerModel]) -> Vec<PeerModel> {
    models
        .iter()
        .filter(|m| match &config.include {
            Some(patterns) if !patterns.is_empty() => {
                patterns.iter().any(|p| glob_match(p, &m.name))
            }
            _ => true,
        })
        .filter(|m| match &config.exclude {
            Some(patterns) if !patterns.is_empty() => {
                !patterns.iter().any(|p| glob_match(p, &m.name))
            }
            _ => true,
        })
        .cloned()
        .collect()
}

/// Minimal glob matcher: `*` matches any run of characters, `?` matches
/// exactly one.  All other characters match literally.
pub(crate) fn glob_match(pattern: &str, text: &str) -> bool {
    fn go(p: &[u8], t: &[u8]) -> bool {
        match (p.first(), t.first()) {
            (None, None) => true,
            (Some(b'*'), _) => {
                // Collapse consecutive stars, then try every split.
                let rest = &p[1..];
                if go(rest, t) {
                    return true;
                }
                (0..t.len()).any(|i| go(rest, &t[i..])) || go(rest, &[])
            }
            (Some(b'?'), Some(_)) => go(&p[1..], &t[1..]),
            (Some(a), Some(b)) if a == b => go(&p[1..], &t[1..]),
            _ => false,
        }
    }
    go(pattern.as_bytes(), text.as_bytes())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(name: &str) -> PeerConfig {
        PeerConfig {
            name: name.into(),
            url: format!("http://{name}:8080"),
            remote_auth: Some("Bearer k".into()),
            catalog_ttl_secs: 60,
            max_concurrent: 8,
            include: None,
            exclude: None,
        }
    }

    #[test]
    fn strips_own_node_and_opaque_prefixes() {
        let peers = vec![peer("cortana")];
        assert_eq!(
            strip_client_prefixes("qwen-flash", Some("cheyenne"), &peers),
            "qwen-flash"
        );
        assert_eq!(
            strip_client_prefixes("llm/qwen-flash", Some("cheyenne"), &peers),
            "qwen-flash"
        );
        assert_eq!(
            strip_client_prefixes("cheyenne/qwen-flash", Some("cheyenne"), &peers),
            "qwen-flash"
        );
        assert_eq!(
            strip_client_prefixes("llm/cortana/qwen-flash", Some("cheyenne"), &peers),
            "cortana/qwen-flash"
        );
        assert_eq!(
            strip_client_prefixes("cortana/qwen-flash", Some("cheyenne"), &peers),
            "cortana/qwen-flash"
        );
        // Deeply nested provider prefixes.
        assert_eq!(
            strip_client_prefixes("a/b/cortana/qwen-flash", Some("cheyenne"), &peers),
            "cortana/qwen-flash"
        );
    }

    #[test]
    fn keeps_peer_prefix_even_without_node_name() {
        let peers = vec![peer("cortana")];
        assert_eq!(
            strip_client_prefixes("llm/cortana/qwen", None, &peers),
            "cortana/qwen"
        );
    }

    fn base_config(node: Option<&str>) -> Config {
        let yaml = format!(
            r#"
server: {{}}
apikeys_file: apikeys.txt
node_name: {}
"#,
            node.unwrap_or("~")
        );
        serde_yaml_ng::from_str(&yaml).unwrap()
    }

    #[test]
    fn parse_catalog_data_carries_decision_capability() {
        // A leaf advertises llama.cpp's `architecture`; the master must keep
        // advertising it for the materialized model, while tolerating
        // values llm-orch does not model (speech output, future modalities).
        let data = serde_json::json!([
            {
                "id": "openjev",
                "context_length": 4096,
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text", "decisions"]
                }
            },
            {
                "id": "tts",
                "architecture": {"input_modalities": ["text"], "output_modalities": ["audio"]}
            },
            {"id": "plain"},
        ])
        .as_array()
        .unwrap()
        .clone();
        let models = parse_catalog_data(&data, None);
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].name, "openjev");
        assert_eq!(
            models[0].capabilities,
            vec!["decisions".to_string(), "image".to_string()]
        );
        assert_eq!(models[0].context_length, Some(4096));
        assert!(
            models[1].capabilities.is_empty(),
            "speech output is not modelled"
        );
        assert!(models[2].capabilities.is_empty());
    }

    #[test]
    fn effective_config_keeps_a_peer_decision_model_advertised() {
        let mut cfg = base_config(Some("cheyenne"));
        cfg.models = vec![];
        cfg.peers = vec![peer("cortana")];
        let registry = PeerRegistry::new(reqwest::Client::new());
        registry.reconcile(&cfg);
        {
            let mut inner = registry.inner.write().unwrap();
            inner.peers.get_mut("cortana").unwrap().catalog = Some(Catalog {
                models: vec![PeerModel {
                    name: "kev".into(),
                    context_length: Some(4096),
                    capabilities: vec!["decisions".into()],
                }],
                fetched_at: Instant::now(),
            });
        }
        let eff = registry.effective_config(&cfg);
        let kev = eff
            .models
            .iter()
            .find(|m| m.name == "cortana/kev")
            .expect("peer model materialized");
        assert!(
            kev.is_decision_model(),
            "the master must advertise a leaf decision model as one"
        );
        // The bare alias for the unique peer model resolves to it, so the
        // capability is reachable under both names.
        let kev_bare = eff.models.iter().find(|m| m.name == "cortana/kev").unwrap();
        assert_eq!(kev_bare.peer.as_deref(), Some("cortana"));
    }

    #[test]
    fn effective_config_materializes_qualified_and_bare_entries() {
        let mut cfg = base_config(Some("cheyenne"));
        cfg.models = vec![ModelConfig {
            name: "local-dense".into(),
            cmd: "sleep 1".into(),
            ..model_defaults()
        }];
        cfg.peers = vec![peer("cortana")];

        let registry = PeerRegistry::new(reqwest::Client::new());
        registry.reconcile(&cfg);
        // Install a catalog manually.
        {
            let mut inner = registry.inner.write().unwrap();
            inner.peers.get_mut("cortana").unwrap().catalog = Some(Catalog {
                models: vec![
                    PeerModel {
                        name: "qwen-flash".into(),
                        context_length: Some(32768),
                        capabilities: vec![],
                    },
                    PeerModel {
                        name: "local-dense".into(),
                        context_length: None,
                        capabilities: vec![],
                    },
                ],
                fetched_at: Instant::now(),
            });
        }

        let eff = registry.effective_config(&cfg);
        let names: Vec<&str> = eff.models.iter().map(|m| m.name.as_str()).collect();
        // Local model plus one qualified peer entry; the colliding
        // `local-dense` only exists in qualified peer form, no bare alias.
        assert!(names.contains(&"local-dense"));
        assert!(names.contains(&"cortana/qwen-flash"));
        assert!(names.contains(&"cortana/local-dense"));
        assert_eq!(eff.models.len(), 3);

        let q = eff
            .models
            .iter()
            .find(|m| m.name == "cortana/qwen-flash")
            .unwrap();
        assert_eq!(q.upstream_model.as_deref(), Some("qwen-flash"));
        assert_eq!(q.peer.as_deref(), Some("cortana"));
        assert_eq!(q.cmd, "http://cortana:8080");
        assert_eq!(q.context_length, 32768);
        assert_eq!(q.queue_depth, 256);

        // Bare alias exists for the unique peer model, not for the
        // colliding one.
        let bare: Vec<&AliasConfig> = eff.aliases.iter().filter(|a| a.peer.is_some()).collect();
        assert_eq!(bare.len(), 1);
        assert_eq!(bare[0].name, "qwen-flash");
        assert_eq!(bare[0].targets, vec!["cortana/qwen-flash".to_string()]);
    }

    #[test]
    fn effective_config_bare_alias_spans_overlapping_peers() {
        let cfg = base_config(Some("cheyenne"));
        let registry = PeerRegistry::new(reqwest::Client::new());
        {
            let mut inner = registry.inner.write().unwrap();
            for name in ["cortana", "cheyenne2"] {
                inner.peers.insert(
                    name.into(),
                    PeerState {
                        config: peer(name),
                        catalog: Some(Catalog {
                            models: vec![PeerModel {
                                name: "qwen-flash".into(),
                                context_length: None,
                                capabilities: vec![],
                            }],
                            fetched_at: Instant::now(),
                        }),
                        last_error: None,
                    },
                );
            }
        }
        let eff = registry.effective_config(&cfg);
        let bare = eff.aliases.iter().find(|a| a.name == "qwen-flash").unwrap();
        assert_eq!(
            bare.targets,
            vec![
                "cheyenne2/qwen-flash".to_string(),
                "cortana/qwen-flash".to_string()
            ]
        );
    }

    #[test]
    fn cycle_guard_skips_own_node_qualified_models() {
        let mut cfg = base_config(Some("cheyenne"));
        cfg.peers = vec![peer("cortana")];
        let registry = PeerRegistry::new(reqwest::Client::new());
        registry.reconcile(&cfg);
        {
            let mut inner = registry.inner.write().unwrap();
            inner.peers.get_mut("cortana").unwrap().catalog = Some(Catalog {
                models: vec![PeerModel {
                    name: "cheyenne/local-dense".into(),
                    context_length: None,
                    capabilities: vec![],
                }],
                fetched_at: Instant::now(),
            });
        }
        let eff = registry.effective_config(&cfg);
        assert!(
            eff.models
                .iter()
                .all(|m| m.name != "cortana/cheyenne/local-dense")
        );
    }

    #[test]
    fn effective_config_is_idempotent() {
        let mut cfg = base_config(Some("cheyenne"));
        cfg.peers = vec![peer("cortana")];
        let registry = PeerRegistry::new(reqwest::Client::new());
        registry.reconcile(&cfg);
        {
            let mut inner = registry.inner.write().unwrap();
            inner.peers.get_mut("cortana").unwrap().catalog = Some(Catalog {
                models: vec![PeerModel {
                    name: "qwen-flash".into(),
                    context_length: None,
                    capabilities: vec![],
                }],
                fetched_at: Instant::now(),
            });
        }
        let once = registry.effective_config(&cfg);
        let twice = registry.effective_config(&once);
        let names: Vec<String> = twice.models.iter().map(|m| m.name.clone()).collect();
        assert_eq!(
            names.iter().filter(|n| n == &"cortana/qwen-flash").count(),
            1
        );
        let aliases: Vec<String> = twice.aliases.iter().map(|a| a.name.clone()).collect();
        assert_eq!(aliases.iter().filter(|n| n == &"qwen-flash").count(), 1);
    }

    #[test]
    fn include_exclude_filters_catalog() {
        // Filtering happens at fetch time via globs; test the matching
        // logic indirectly through effective_config with a manual
        // catalog (the fetch filter is exercised in integration tests).
        let mut cfg = base_config(Some("cheyenne"));
        cfg.peers = vec![PeerConfig {
            include: Some(vec!["qwen-*".into()]),
            ..peer("cortana")
        }];
        let registry = PeerRegistry::new(reqwest::Client::new());
        registry.reconcile(&cfg);
        {
            let mut inner = registry.inner.write().unwrap();
            let state = inner.peers.get_mut("cortana").unwrap();
            // Apply the same include/exclude filter the fetch uses.
            let all = ["qwen-flash", "llama-dense"];
            let filtered = apply_filters(
                &state.config,
                &all.iter()
                    .map(|n| PeerModel {
                        name: (*n).into(),
                        context_length: None,
                        capabilities: vec![],
                    })
                    .collect::<Vec<_>>(),
            );
            state.catalog = Some(Catalog {
                models: filtered,
                fetched_at: Instant::now(),
            });
        }
        let eff = registry.effective_config(&cfg);
        assert!(eff.models.iter().any(|m| m.name == "cortana/qwen-flash"));
        assert!(!eff.models.iter().any(|m| m.name == "cortana/llama-dense"));
    }
}
