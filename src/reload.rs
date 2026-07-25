// ── Hot-reload handlers ──────────────────────────────────────────────────────
//
// Atomic reload logic for the config and apikeys files: validate first,
// swap only on success, record the file mtime only after a successful swap
// so rejected or failed reloads are retried on the next file event.

use crate::{apikeys, config, scheduler};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// Atomic config reload: validate first, swap only on success.
pub async fn handle_config_reload(
    path: &std::path::Path,
    cfg: &Arc<RwLock<config::Config>>,
    apikeys: &Arc<RwLock<apikeys::ApikeysStore>>,
    manager: &Arc<scheduler::InstanceManager>,
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
            // Reload apikeys if the path changed.
            {
                let old = cfg.read().await;
                if new_cfg.server.listen != old.server.listen {
                    warn!(
                        old = %old.server.listen,
                        new = %new_cfg.server.listen,
                        "server.listen changed — requires a restart, keeping the old listener"
                    );
                }
                if new_cfg.apikeys_file != old.apikeys_file {
                    match apikeys::ApikeysStore::load(&new_cfg.apikeys_file) {
                        Ok(new_keys) => {
                            info!("apikeys reloaded: {} key(s)", new_keys.len());
                            *apikeys.write().await = new_keys;
                        }
                        Err(e) => {
                            error!("apikeys reload failed (keeping old): {}", e);
                            return;
                        }
                    }
                }
            }

            // Reconcile the instance manager (model defs, cmd aliases,
            // device maps, port range, keep-alive, crash blocks) before
            // swapping the shared config that handlers validate against.
            manager.reconcile_config(&new_cfg).await;

            // Atomic swap.
            let model_count = new_cfg.models.len();
            *cfg.write().await = new_cfg;
            // Record the mtime only on success — a rejected or failed
            // reload must be retried on the next file event, not consumed.
            *last_mtime.lock().await = Some(mtime);
            info!(
                "config reloaded successfully — {} model(s)",
                model_count
            );
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
