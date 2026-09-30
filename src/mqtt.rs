// ── MQTT stats publishing ─────────────────────────────────────────────────
//
// Opt-in via the top-level `mqtt:` config section (docs/007-mqtt-stats.md).
// Publishes the same documents `GET /v1/stats` serves, under a per-node
// topic tree:
//
//   {prefix}/{node}/status   "online"/"offline" (retained, + LWT offline)
//   {prefix}/{node}/stats    full stats doc + `rates` (delta-derived rates)
//   {prefix}/{node}/models   catalog discovery doc (models + aliases)
//   {prefix}/{node}/gpus     GPU inventory doc (static fields only)
//   {prefix}/{peer}/stats    gateway re-export of each leaf's cached doc
//
// Publishing is change-only: every tick (default 60 s) rebuilds the docs,
// serializes them canonically (keys sorted recursively), and publishes a
// topic only when its payload differs from the last published one.  Every
// successful CONNACK (startup *and* broker reconnect) force-republishes
// all docs, which replaces any periodic heartbeat.
//
// Rates: the publisher keeps the previous per-model `boot` counters
// (requests / prompt / completion) and divides the delta by the elapsed
// wall time.  llm-orch is both the counter source and the publisher, so
// consumers never subtract across a counter reset: a boot-counter decrease
// within a process is clamped (skipped + `warn!`), and the first tick
// after startup simply has no rates.
//
// Failure isolation: a dead or unreachable broker must never affect
// request serving.  All MQTT work lives in one background task; publish
// errors are logged (rate-limited to one warning per disconnect streak)
// and dropped.  Hot-reload reconfigures by respawn — the reload path
// never waits on the broker beyond the bounded shutdown handshake.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rumqttc::{AsyncClient, Event, Incoming, LastWill, MqttOptions, QoS, Transport};
use serde_json::{Value, json};
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::config::{Config, MqttConfig};
use crate::gpu::GpuMetrics;
use crate::peers::PeerRegistry;
use crate::stats::{self, StatsModel, StatsStore};

/// Shared data sources the publisher reads on every tick.
#[derive(Clone)]
pub struct MqttDeps {
    pub stats: Arc<StatsStore>,
    pub gpu: Arc<RwLock<Vec<GpuMetrics>>>,
    pub peers: Arc<PeerRegistry>,
    /// Hot-reloaded *effective* config (peer models materialized in).
    pub config: Arc<RwLock<Config>>,
}

/// Commands into the publisher task.
enum Cmd {
    /// Publish immediately (test seam; normal operation is timer-driven).
    Tick,
    /// Final flush + clean `offline` + DISCONNECT, then exit.
    Shutdown,
}

/// A running publisher task.
pub struct MqttHandle {
    cmd_tx: mpsc::UnboundedSender<Cmd>,
    task: JoinHandle<()>,
    /// Identity the publisher was started with — a reconfigure with an
    /// identical identity is a no-op.
    identity: Identity,
}

/// Everything that changes the broker connection or topic tree.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Identity {
    mqtt: MqttConfig,
    node_name: String,
    /// Resolved password (template after `key_files` substitution), so a
    /// rotated key file applied via config reload forces a respawn.
    password: Option<String>,
}

/// Start/stop glue owned by `main` and shared with the reload task.
pub struct MqttController {
    deps: MqttDeps,
    slot: Mutex<Option<MqttHandle>>,
}

impl MqttController {
    pub fn new(deps: MqttDeps) -> Arc<Self> {
        Arc::new(Self {
            deps,
            slot: Mutex::new(None),
        })
    }

    /// Bring the publisher in line with `cfg`: start, stop, or respawn on
    /// any change to the `mqtt:` section, `node_name`, or the resolved
    /// password.  Never contacts the broker synchronously beyond the
    /// bounded (2 s) shutdown handshake of a replaced task.
    pub async fn apply(&self, cfg: &Config) {
        let new_identity = match (&cfg.mqtt, &cfg.node_name) {
            (Some(mqtt), Some(node)) => Some(Identity {
                mqtt: mqtt.clone(),
                node_name: node.clone(),
                password: mqtt
                    .key
                    .as_ref()
                    .map(|template| resolve_password(cfg, template)),
            }),
            (Some(_), None) => {
                // Config validation normally rejects this; a hot-reload
                // can only deliver validated configs.
                warn!("mqtt: section present but node_name missing — publisher disabled");
                None
            }
            (None, _) => None,
        };

        {
            let guard = self.slot.lock().unwrap();
            if let Some(handle) = guard.as_ref() {
                if Some(&handle.identity) == new_identity.as_ref() {
                    return; // unchanged
                }
            } else if new_identity.is_none() {
                return; // still disabled
            }
        }

        // Stop the old publisher (bounded), then start the new one.
        let old = self.slot.lock().unwrap().take();
        if let Some(handle) = old {
            Self::stop(handle).await;
        }
        if let Some(identity) = new_identity {
            info!(
                broker = %identity.mqtt.broker,
                node = %identity.node_name,
                "mqtt publisher started"
            );
            let (cmd_tx, task) = spawn_publisher(identity.clone(), self.deps.clone());
            *self.slot.lock().unwrap() = Some(MqttHandle {
                cmd_tx,
                task,
                identity,
            });
        }
    }

    /// Graceful shutdown: final publish + `offline` will override, bounded.
    pub async fn shutdown(&self) {
        let old = self.slot.lock().unwrap().take();
        if let Some(handle) = old {
            info!("mqtt publisher shutting down");
            Self::stop(handle).await;
        }
    }

    /// Force an immediate publish tick (test seam; normal operation is
    /// timer-driven via `mqtt.interval`).
    pub fn tick_now(&self) {
        if let Some(handle) = self.slot.lock().unwrap().as_ref() {
            let _ = handle.cmd_tx.send(Cmd::Tick);
        }
    }

    async fn stop(handle: MqttHandle) {
        let MqttHandle { cmd_tx, task, .. } = handle;
        if cmd_tx.send(Cmd::Shutdown).is_ok() {
            let _ = tokio::time::timeout(Duration::from_secs(5), task).await;
        }
    }
}

/// Resolve a `key` template against the config's `key_files` (same
/// mechanism as `peers[].remote_auth`).
fn resolve_password(cfg: &Config, template: &str) -> String {
    let key_files = crate::config::load_key_files(&cfg.key_files).unwrap_or_else(|e| {
        warn!("mqtt: could not read key_files for password ({e}); using raw template");
        HashMap::new()
    });
    crate::config::resolve_remote_auth(&key_files, template)
}

/// Spawn the publisher task for a validated identity.
fn spawn_publisher(
    identity: Identity,
    deps: MqttDeps,
) -> (mpsc::UnboundedSender<Cmd>, JoinHandle<()>) {
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let task = tokio::spawn(run_publisher(identity, deps, cmd_rx));
    (cmd_tx, task)
}

// ── Publisher task ──────────────────────────────────────────────────────────

/// Previous-tick boot counters for rate computation
/// (model → requests/prompt/completion, monotonic within a boot).
type RateState = (std::time::Instant, HashMap<String, (u64, u64, u64)>);

async fn run_publisher(
    identity: Identity,
    deps: MqttDeps,
    mut cmd_rx: mpsc::UnboundedReceiver<Cmd>,
) {
    let Identity {
        mqtt,
        node_name,
        password,
    } = identity;
    let (host, port, tls) = match mqtt.parse_broker() {
        Ok(v) => v,
        Err(e) => {
            warn!("{e} — mqtt publisher not started");
            return;
        }
    };
    let qos = if mqtt.qos == 0 {
        QoS::AtMostOnce
    } else {
        QoS::AtLeastOnce
    };

    let base = format!("{}/{}", mqtt.prefix, node_name);
    let status_topic = format!("{base}/status");

    let client_id = format!("{}-{:x}", node_name, uuid::Uuid::new_v4().simple());
    let mut opts = MqttOptions::new(client_id, &host, port);
    opts.set_keep_alive(Duration::from_secs(60));
    // No persistent broker session: retained messages + force-republish on
    // every CONNACK cover reconnects, and a uuid client id would leave
    // junk sessions on the broker anyway.
    opts.set_clean_session(true);
    if let Some(user) = &mqtt.username {
        opts.set_credentials(user, password.clone().unwrap_or_default());
    } else if let Some(pass) = &password {
        // Many brokers accept password-only auth under an anonymous id;
        // pass it with the generated client id rather than silently.
        opts.set_credentials(node_name.clone(), pass.clone());
    }
    if tls {
        opts.set_transport(Transport::tls_with_default_config());
    }
    // LWT: whoever dies without a handshake (crash, network) goes offline.
    opts.set_last_will(LastWill::new(
        status_topic.clone(),
        "offline",
        QoS::AtLeastOnce,
        true,
    ));

    let (client, mut event_loop) = AsyncClient::new(opts, 64);

    let mut last: HashMap<String, String> = HashMap::new(); // topic → canonical payload
    let mut rates: Option<RateState> = None;
    let mut connected = false;
    let mut logged_disconnect = false;
    let mut next_tick = tokio::time::Instant::now();
    let interval = Duration::from_secs(mqtt.interval.max(1));

    loop {
        tokio::select! {
            cmd = cmd_rx.recv() => {
                match cmd {
                    None | Some(Cmd::Shutdown) => {
                        // Best-effort graceful exit: flush pending docs,
                        // override the retained status with "offline"
                        // (cleaner than waiting on the broker's will
                        // delay), then DISCONNECT.
                        if connected {
                            let _ = publish_docs(
                                &client, &mqtt, qos, &base, &node_name, &deps,
                                &mut last, &mut rates,
                            )
                            .await;
                            let _ = client
                                .publish(
                                    status_topic.clone(),
                                    QoS::AtLeastOnce,
                                    mqtt.retain,
                                    "offline",
                                )
                                .await;
                            let _ = client.disconnect().await;
                            let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
                            loop {
                                tokio::select! {
                                    ev = event_loop.poll() => {
                                        match ev {
                                            Ok(_) => {
                                                if tokio::time::Instant::now() >= deadline { break; }
                                            }
                                            Err(_) => break,
                                        }
                                    }
                                    _ = tokio::time::sleep_until(deadline) => break,
                                }
                            }
                        }
                        debug!("mqtt publisher task exiting");
                        return;
                    }
                    Some(Cmd::Tick) => {
                        next_tick = tokio::time::Instant::now();
                    }
                }
            }
            _ = tokio::time::sleep_until(next_tick), if connected => {
                publish_docs(
                    &client, &mqtt, qos, &base, &node_name, &deps, &mut last, &mut rates,
                )
                .await;
                next_tick = tokio::time::Instant::now() + interval;
            }
            ev = event_loop.poll() => {
                match ev {
                    Ok(Event::Incoming(Incoming::ConnAck(_))) => {
                        connected = true;
                        logged_disconnect = false;
                        info!(broker = %mqtt.broker, "mqtt connected");
                        // Publish the birth status and force a full
                        // republish (replaces any heartbeat).
                        let _ = client
                            .publish(status_topic.clone(), QoS::AtLeastOnce, mqtt.retain, "online")
                            .await;
                        last.clear();
                        rates = None;
                        publish_docs(
                            &client, &mqtt, qos, &base, &node_name, &deps, &mut last,
                            &mut rates,
                        )
                        .await;
                        next_tick = tokio::time::Instant::now() + interval;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        if connected || !logged_disconnect {
                            warn!("mqtt connection error (will keep retrying): {e}");
                            logged_disconnect = true;
                        } else {
                            debug!("mqtt connection error: {e}");
                        }
                        connected = false;
                    }
                }
            }
        }
    }
}

/// Rebuild all documents and publish the ones that changed.
#[allow(clippy::too_many_arguments)] // borrow-per-source, no coherent bundle
async fn publish_docs(
    client: &AsyncClient,
    mqtt: &MqttConfig,
    qos: QoS,
    base: &str,
    node_name: &str,
    deps: &MqttDeps,
    last: &mut HashMap<String, String>,
    rates: &mut Option<RateState>,
) {
    // ── stats (own doc, with rates) ───────────────────────────────────
    let resp = stats::build_response(&deps.stats, &deps.gpu, &deps.peers).await;
    let elapsed = rates.as_ref().map(|(t, _)| t.elapsed()).unwrap_or_default();
    let mut stats_value = match serde_json::to_value(&resp) {
        Ok(v) => v,
        Err(e) => {
            warn!("mqtt: cannot serialize stats doc: {e}");
            return;
        }
    };
    let current: HashMap<String, (u64, u64, u64)> = resp
        .models
        .iter()
        .map(|(name, m)| {
            let StatsModel { boot, .. } = m;
            (
                name.clone(),
                (boot.requests, boot.prompt_tokens, boot.completion_tokens),
            )
        })
        .collect();
    if let Some((_, prev)) = rates.as_ref()
        && let Some(rates_doc) = compute_rates(prev, &current, elapsed)
    {
        stats_value["rates"] = rates_doc;
    }
    *rates = Some((std::time::Instant::now(), current));
    publish_if_changed(
        client,
        mqtt,
        qos,
        &format!("{base}/stats"),
        stats_value,
        last,
    )
    .await;

    // ── models discovery doc ──────────────────────────────────────────
    let cfg = deps.config.read().await;
    let models_doc = build_models_doc(&cfg);
    drop(cfg);
    publish_if_changed(
        client,
        mqtt,
        qos,
        &format!("{base}/models"),
        models_doc,
        last,
    )
    .await;

    // ── gpu inventory doc (static fields only) ────────────────────────
    let gpu_snapshot = deps.gpu.read().await;
    let gpus_doc = build_inventory_doc(node_name, &gpu_snapshot);
    drop(gpu_snapshot);
    publish_if_changed(client, mqtt, qos, &format!("{base}/gpus"), gpus_doc, last).await;

    // ── gateway re-export of leaf stats (cached, no extra fetches) ────
    let peers = crate::stats::peer_stat_docs(&deps.peers).await;
    for (peer, doc) in peers {
        // Re-export under the *peer's* own node name, not ours.
        let topic = format!("{}/{}", mqtt.prefix, peer);
        publish_if_changed(client, mqtt, qos, &format!("{topic}/stats"), doc, last).await;
    }
}

/// Per-model per-minute rates from boot-counter deltas.  Models absent
/// from the previous snapshot are treated as zero-counted (the stats
/// map only contains models with recorded activity), so their first
/// window reports their boot counters as the delta.  Only models with
/// activity in the window are reported; a fully quiet window produces
/// no `rates` section at all (a monotonic `window_secs` would otherwise
/// churn the payload on every tick).  A model whose counters regressed
/// (should be impossible within a boot; defensive) is skipped for this
/// tick.
fn compute_rates(
    prev: &HashMap<String, (u64, u64, u64)>,
    current: &HashMap<String, (u64, u64, u64)>,
    elapsed: Duration,
) -> Option<Value> {
    if elapsed.is_zero() {
        return None;
    }
    let mins = elapsed.as_secs_f64() / 60.0;
    let mut models = serde_json::Map::new();
    for (name, cur) in current {
        // A model missing from the previous snapshot had zero counters
        // (empty maps hold no entries), so its delta is the raw value.
        let (p_req, p_prompt, p_comp) = prev.get(name).copied().unwrap_or((0, 0, 0));
        let (c_req, c_prompt, c_comp) = *cur;
        if c_req < p_req || c_prompt < p_prompt || c_comp < p_comp {
            warn!(model = %name, "mqtt: boot counters regressed; skipping rates this tick");
            continue;
        }
        if c_req == p_req && c_prompt == p_prompt && c_comp == p_comp {
            continue; // quiet model: no rates entry
        }
        models.insert(
            name.clone(),
            json!({
                "requests_per_min": round2((c_req - p_req) as f64 / mins),
                "prompt_tokens_per_min": round2((c_prompt - p_prompt) as f64 / mins),
                "completion_tokens_per_min": round2((c_comp - p_comp) as f64 / mins),
                "tokens_per_min": round2(((c_prompt - p_prompt) + (c_comp - p_comp)) as f64 / mins),
            }),
        );
    }
    if models.is_empty() {
        return None;
    }
    Some(json!({
        "window_secs": elapsed.as_secs(),
        "models": models,
    }))
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Catalog discovery doc from the effective config (peer models are
/// already materialized as `<peer>/<model>` entries by the registry).
fn build_models_doc(cfg: &Config) -> Value {
    let models: Vec<Value> = cfg
        .models
        .iter()
        .map(|m| json!({ "name": m.name, "context_length": m.context_length }))
        .collect();
    let aliases: Vec<Value> = cfg
        .aliases
        .iter()
        .map(|a| json!({ "name": a.name, "targets": a.targets }))
        .collect();
    json!({
        "version": 1,
        "node": cfg.node_name,
        "models": models,
        "aliases": aliases,
    })
}

/// GPU inventory doc: stable identity fields only (live metrics ride the
/// `stats` topic), so it only changes when the hardware/config changes.
fn build_inventory_doc(node: &str, gpus: &[GpuMetrics]) -> Value {
    let list: Vec<Value> = gpus
        .iter()
        .map(|g| {
            json!({
                "index": g.index,
                "pci_slot": g.pci_slot,
                "vram_vendor": g.vram_vendor,
                "vram_total_bytes": g.vram_total_bytes,
            })
        })
        .collect();
    json!({
        "version": 1,
        "node": node,
        "gpus": list,
    })
}

/// Serialize canonically and publish only when the payload changed.
async fn publish_if_changed(
    client: &AsyncClient,
    mqtt: &MqttConfig,
    qos: QoS,
    topic: &str,
    value: Value,
    last: &mut HashMap<String, String>,
) {
    let payload = canonical(&value).to_string();
    if last.get(topic).is_some_and(|p| *p == payload) {
        debug!(topic, "mqtt: doc unchanged, not republishing");
        return;
    }
    match client.try_publish(topic, qos, mqtt.retain, payload.clone()) {
        Ok(()) => {
            last.insert(topic.to_string(), payload);
        }
        Err(e) => {
            debug!(topic, "mqtt publish failed (broker may be down): {e}");
        }
    }
}

/// Deterministic serialization: object keys sorted recursively, so
/// identical content always hashes/compares equal regardless of the
/// HashMap iteration order it was built with.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), canonical(&map[k]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn config_with(yaml_extra: &str) -> Result<Config, String> {
        let yaml = format!(
            r#"
server:
  listen: "127.0.0.1:0"
apikeys_file: /dev/null
{yaml_extra}
"#
        );
        let path =
            std::env::temp_dir().join(format!("llm-orch-mqtt-test-{}.yaml", uuid::Uuid::new_v4()));
        std::fs::write(&path, yaml).unwrap();
        let r = Config::load(&path).map_err(|e| e.to_string());
        std::fs::remove_file(&path).ok();
        r
    }

    #[test]
    fn broker_parsing() {
        let mqtt: MqttConfig = serde_yaml_ng::from_str("broker: host:1883").unwrap();
        assert_eq!(mqtt.parse_broker().unwrap(), ("host".into(), 1883, false));
        let mqtt: MqttConfig = serde_yaml_ng::from_str("broker: mqtt://h").unwrap();
        assert_eq!(mqtt.parse_broker().unwrap(), ("h".into(), 1883, false));
        let mqtt: MqttConfig = serde_yaml_ng::from_str("broker: mqtts://h").unwrap();
        assert_eq!(mqtt.parse_broker().unwrap(), ("h".into(), 8883, true));
        let mqtt: MqttConfig = serde_yaml_ng::from_str("broker: h:8883\ntls: true").unwrap();
        assert_eq!(mqtt.parse_broker().unwrap(), ("h".into(), 8883, true));
        let mqtt: MqttConfig = serde_yaml_ng::from_str("broker: h:abc").unwrap();
        assert!(mqtt.parse_broker().is_err());
        let mqtt: MqttConfig = serde_yaml_ng::from_str("broker: h/p").unwrap();
        assert!(mqtt.parse_broker().is_err());
        // defaults
        assert_eq!(mqtt.prefix, "llmorch");
        assert_eq!(mqtt.interval, 60);
        assert_eq!(mqtt.qos, 1);
        assert!(mqtt.retain);
    }

    #[test]
    fn validation_rules() {
        // mqtt without node_name → rejected.
        let e = config_with("mqtt:\n  broker: h:1883").unwrap_err();
        assert!(e.contains("requires node_name"), "{e}");
        // valid section accepted.
        assert!(config_with("node_name: testnode\nmqtt:\n  broker: h:1883").is_ok());
        // qos > 1 rejected.
        let e = config_with("node_name: n\nmqtt:\n  broker: h\n  qos: 2").unwrap_err();
        assert!(e.contains("qos"), "{e}");
        // wildcard in prefix rejected.
        let e = config_with("node_name: n\nmqtt:\n  broker: h\n  prefix: a/+b").unwrap_err();
        assert!(e.contains("prefix"), "{e}");
        // empty prefix segment rejected.
        let e = config_with("node_name: n\nmqtt:\n  broker: h\n  prefix: \"a/\"").unwrap_err();
        assert!(e.contains("prefix"), "{e}");
        // interval 0 rejected.
        let e = config_with("node_name: n\nmqtt:\n  broker: h\n  interval: 0").unwrap_err();
        assert!(e.contains("interval"), "{e}");
        // empty key rejected; unbalanced placeholder rejected.
        let e = config_with("node_name: n\nmqtt:\n  broker: h\n  key: \"  \"").unwrap_err();
        assert!(e.contains("key"), "{e}");
        let e = config_with("node_name: n\nmqtt:\n  broker: h\n  key: \"{pw\"").unwrap_err();
        assert!(e.contains("placeholder"), "{e}");
    }

    #[test]
    fn canonical_serialization_is_order_insensitive() {
        let mut a = serde_json::Map::new();
        a.insert("z".into(), serde_json::json!(1));
        a.insert("a".into(), serde_json::json!([2, 3]));
        let mut b = serde_json::Map::new();
        b.insert("a".into(), serde_json::json!([2, 3]));
        b.insert("z".into(), serde_json::json!(1));
        assert_ne!(
            serde_json::to_string(&Value::Object(a.clone())).unwrap(),
            serde_json::to_string(&Value::Object(b.clone())).unwrap(),
            "raw maps differ"
        );
        assert_eq!(
            canonical(&Value::Object(a)).to_string(),
            canonical(&Value::Object(b)).to_string()
        );
        // Nested objects too.
        let v = serde_json::json!({"outer": {"b": {"y": 1, "x": 2}, "a": 0}});
        let c = canonical(&v).to_string();
        assert_eq!(c, canonical(&canonical(&v)).to_string(), "idempotent");
        assert!(c.find("\"a\"").unwrap() < c.find("\"b\"").unwrap());
    }

    fn counters(req: u64, prompt: u64, comp: u64) -> (u64, u64, u64) {
        (req, prompt, comp)
    }

    #[test]
    fn rates_from_deltas() {
        let prev = HashMap::from([("m".to_string(), counters(1, 10, 5))]);
        let cur = HashMap::from([
            ("m".to_string(), counters(2, 20, 15)),
            ("new".to_string(), counters(1, 1, 1)),
        ]);
        let rates = compute_rates(&prev, &cur, Duration::from_secs(60)).unwrap();
        assert_eq!(rates["models"]["m"]["requests_per_min"], 1.0);
        assert_eq!(rates["models"]["m"]["prompt_tokens_per_min"], 10.0);
        assert_eq!(rates["models"]["m"]["completion_tokens_per_min"], 10.0);
        assert_eq!(rates["models"]["m"]["tokens_per_min"], 20.0);
        // A model new to the snapshot map had zero counters before, so
        // its first window reports the boot counters as the delta.
        assert_eq!(rates["models"]["new"]["requests_per_min"], 1.0);
        assert_eq!(rates["models"]["new"]["tokens_per_min"], 2.0);
        assert_eq!(rates["window_secs"], 60);
    }

    #[test]
    fn rates_edge_cases() {
        // Zero elapsed → no rates.
        let prev = HashMap::from([("m".to_string(), counters(1, 1, 1))]);
        assert!(compute_rates(&prev, &prev, Duration::ZERO).is_none());
        // Fully quiet window → no rates section (change-only stability).
        assert!(compute_rates(&prev, &prev, Duration::from_secs(30)).is_none());
        // Regressed counters → model skipped → no rates doc.
        let cur = HashMap::from([("m".to_string(), counters(0, 0, 0))]);
        assert!(compute_rates(&prev, &cur, Duration::from_secs(30)).is_none());
    }

    #[test]
    fn models_doc_lists_models_and_aliases() {
        let cfg = config_with(
            "node_name: n\nmodels:\n  - name: alpha\n    cmd: llama-server\n    context_length: 2048\naliases:\n  - name: fast\n    targets: [alpha]\n",
        )
        .unwrap();
        let doc = build_models_doc(&cfg);
        assert_eq!(doc["version"], 1);
        assert_eq!(doc["node"], "n");
        assert_eq!(doc["models"][0]["name"], "alpha");
        assert_eq!(doc["models"][0]["context_length"], 2048);
        assert_eq!(doc["aliases"][0]["targets"][0], "alpha");
    }
}
