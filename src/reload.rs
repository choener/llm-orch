// ── Hot-reload handlers ──────────────────────────────────────────────────────
//
// Atomic reload logic for the config and apikeys files: validate first,
// swap only on success, record the file mtime only after a successful swap
// so rejected or failed reloads are retried on the next file event.

use crate::{apikeys, config, mqtt, peers, scheduler, stats};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// Atomic config reload: validate first, swap only on success.
///
/// The apikeys store is *not* touched here: its path is fixed at startup
/// (`--apikeyfile`) so a config edit can never redirect authentication, and
/// the file itself is hot-reloaded by `handle_apikeys_reload`.
#[allow(clippy::too_many_arguments)]
pub async fn handle_config_reload(
    path: &std::path::Path,
    cfg: &Arc<RwLock<config::Config>>,
    manager: &Arc<scheduler::InstanceManager>,
    stats: &Arc<stats::StatsStore>,
    peers: &Arc<peers::PeerRegistry>,
    mqtt: &Arc<mqtt::MqttController>,
    last_mtime: &Arc<Mutex<Option<std::time::SystemTime>>>,
) {
    // Skip if the file's mtime hasn't changed since our last reload.
    let mtime = match std::fs::metadata(path).and_then(|m| m.modified()) {
        Ok(mt) => mt,
        Err(_) => return,
    };
    {
        let last = last_mtime.lock().await;
        if *last == Some(mtime) {
            debug!("config mtime unchanged, skipping reload");
            return;
        }
    }
    info!("config file changed, reloading…");
    match config::Config::load(path) {
        Ok(new_cfg) => {
            {
                let old = cfg.read().await;
                if new_cfg.server.listen != old.server.listen {
                    warn!(
                        old = %old.server.listen,
                        new = %new_cfg.server.listen,
                        "server.listen changed — requires a restart, keeping the old listener"
                    );
                }
            }

            // Cluster peering (docs/006): refresh the peer catalogs
            // *before* applying the new config, so alias targets can
            // reference peer models and the materialized entries are
            // present from the first request after the reload.  A peer
            // that is down keeps its cached catalog — a dead leaf must
            // not wedge the reload.
            peers.reconcile(&new_cfg);
            peers.refresh_all_catalogs().await;
            let effective = peers.effective_config(&new_cfg);

            // Reconcile the instance manager (model defs, cmd aliases,
            // device maps, port range, keep-alive, crash blocks) before
            // swapping the shared config that handlers validate against.
            manager.reconcile_config(&effective).await;

            // Retarget the stats file if the path changed (counter state
            // is kept; the new file is published immediately).
            let old_stats_path = cfg.read().await.stats.as_ref().map(|s| s.file.clone());
            let new_stats_path = new_cfg.stats.as_ref().map(|s| s.file.clone());
            if old_stats_path != new_stats_path {
                match &new_stats_path {
                    Some(p) => info!(path = %p.display(), "stats file retargeted"),
                    None => info!("stats disabled on reload"),
                }
                stats.set_path(new_stats_path.as_ref());
            }

            // Atomic swap.
            let model_count = effective.models.len();
            *cfg.write().await = effective;
            // Bring the MQTT publisher in line with the new config
            // (start/stop/respawn on `mqtt:` / `node_name` changes).
            // Identity-only work: never waits on the broker beyond the
            // bounded shutdown handshake of a replaced task.
            mqtt.apply(&new_cfg).await;
            // Record the mtime only on success — a rejected or failed
            // reload must be retried on the next file event, not consumed.
            *last_mtime.lock().await = Some(mtime);
            info!("config reloaded successfully — {} model(s)", model_count);
        }
        Err(e) => {
            error!("config reload rejected (keeping old): {}", e);
        }
    }
}

/// Atomic apikeys reload.
pub async fn handle_apikeys_reload(
    path: &std::path::Path,
    apikeys: &Arc<RwLock<apikeys::ApikeysStore>>,
    last_mtime: &Arc<Mutex<Option<std::time::SystemTime>>>,
) {
    let mtime = match std::fs::metadata(path).and_then(|m| m.modified()) {
        Ok(mt) => mt,
        Err(_) => return,
    };
    {
        let last = last_mtime.lock().await;
        if *last == Some(mtime) {
            debug!("apikeys mtime unchanged, skipping reload");
            return;
        }
    }
    info!("apikeys file changed, reloading…");
    match apikeys::ApikeysStore::load(path) {
        Ok(new_keys) => {
            let count = new_keys.len();
            *apikeys.write().await = new_keys;
            // Record the mtime only on success (see config reload).
            *last_mtime.lock().await = Some(mtime);
            info!("apikeys reloaded successfully — {} key(s)", count);
        }
        Err(e) => {
            error!("apikeys reload rejected (keeping old): {}", e);
        }
    }
}
