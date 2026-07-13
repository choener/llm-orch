use clap::Parser;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

mod apikeys;
mod backend;
mod config;
mod handlers;
mod http_client;
mod instance;
mod port_alloc;
mod scheduler;
mod server;
mod types;
mod watcher;

/// LLM Orch — single-host LLM orchestrator.
#[derive(Parser, Debug)]
#[command(name = "llm-orch", version, about)]
struct Cli {
    /// Path to the main configuration file.
    #[arg(
        short = 'c',
        long = "config",
        default_value = "config.yaml",
        value_hint = clap::ValueHint::FilePath
    )]
    config: std::path::PathBuf,

    /// Validate the configuration file and exit without starting the server.
    /// Exits 0 on valid config, non-zero on errors.
    #[arg(long = "check-config", value_name = "PATH", value_hint = clap::ValueHint::FilePath)]
    check_config: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Tracing ────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // §2 — check-config mode: validate and exit.
    if let Some(check_path) = cli.check_config {
        match config::Config::load(&check_path) {
            Ok(cfg) => {
                println!(
                    "OK: {} model(s), {} alias(es), {} cmd alias(es)",
                    cfg.models.len(),
                    cfg.aliases.len(),
                    cfg.cmd_aliases.len(),
                );
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("ERROR: {}", e);
                std::process::exit(1);
            }
        }
    }

    // ── Bootstrapping §6 ─────────────────────────────────────────────────
    // 1. Load main config.
    info!("loading config: {}", cli.config.display());
    let cfg = config::Config::load(&cli.config)?;
    info!(
        "loaded {} model(s), {} alias(es)",
        cfg.models.len(),
        cfg.aliases.len()
    );

    // 2. Load API keys from the path specified in the config.
    //    If the file doesn't exist, continue with an empty store (fail-closed).
    info!("loading apikeys: {}", cfg.apikeys_file.display());
    let apikeys = match apikeys::ApikeysStore::load(&cfg.apikeys_file) {
        Ok(a) => a,
        Err(e) => {
            warn!("could not load apikeys (fail-closed): {}", e);
            apikeys::ApikeysStore::empty()
        }
    };
    info!(
        "loaded {} apikey(s){}",
        apikeys.len(),
        if apikeys.is_empty() {
            " (fail-closed: all requests denied)"
        } else {
            ""
        }
    );

    // 3. Create the instance manager.
    let manager = Arc::new(scheduler::InstanceManager::new(&cfg));
    info!("instance manager ready");

    // Shared state for hot-reload.
    let shared_cfg = Arc::new(RwLock::new(cfg));
    let shared_apikeys = Arc::new(RwLock::new(apikeys));
    let last_reload_mtime = Arc::new(Mutex::new(None::<std::time::SystemTime>));
    let last_apikeys_mtime = Arc::new(Mutex::new(None::<std::time::SystemTime>));

    // TODO §7: start HTTP server.

    // ── Start HTTP server §7 ───────────────────────────────────────────
    let app_state = server::AppState {
        config: Arc::clone(&shared_cfg),
        apikeys: Arc::clone(&shared_apikeys),
        manager: Arc::clone(&manager),
        client: manager.client().clone(),
    };
    let server_task = tokio::spawn(server::serve(app_state));

    // ── File watchers + reload loop §6 ───────────────────────────────────
    let config_path = cli.config.clone();
    let apikeys_path = shared_cfg.read().await.apikeys_file.clone();
    let (reload_tx, mut reload_rx) = broadcast::channel::<watcher::FileChange>(16);
    let _watcher = match watcher::watch_paths(
        vec![config_path.clone(), apikeys_path.clone()],
        reload_tx,
        Duration::from_secs(2),
    )
    .await
    {
        Ok(wh) => {
            info!("watching config and apikeys files for changes");
            Some(wh)
        }
        Err(e) => {
            warn!("file watcher failed: {}", e);
            None
        }
    };

    tokio::spawn({
        let cfg = Arc::clone(&shared_cfg);
        let apikeys = Arc::clone(&shared_apikeys);
        let manager = Arc::clone(&manager);
        let last_mtime = Arc::clone(&last_reload_mtime);
        let last_ak_mtime = Arc::clone(&last_apikeys_mtime);
        async move {
            while let Ok(event) = reload_rx.recv().await {
                // The watcher already pre-filters to our watched paths.
                // Determine which file changed by comparing to canonical paths.
                let cfg_p = config_path.canonicalize().ok();
                let ak_p = apikeys_path.canonicalize().ok();
                let changed = event.path.canonicalize().ok();

                match (&changed, &cfg_p) {
                    (Some(c), Some(canon)) if c == canon => {
                        handle_config_reload(
                            &event.path, &cfg, &apikeys, &manager, &last_mtime,
                        ).await;
                    }
                    _ => {}
                }
                match (&changed, &ak_p) {
                    (Some(c), Some(canon)) if c == canon => {
                        handle_apikeys_reload(&event.path, &apikeys, &last_ak_mtime).await;
                    }
                    _ => {}
                }
            }
        }
    });

    info!("running — press Ctrl-C to stop");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("shutting down...");
        }
        _ = server_task => {
            info!("server exited");
        }
    }
    manager.shutdown_all().await;
    info!("stopped");

    Ok(())
}

// ── Reload handlers ──────────────────────────────────────────────────────────

/// Atomic config reload: validate first, swap only on success.
async fn handle_config_reload(
    path: &std::path::Path,
    cfg: &Arc<RwLock<config::Config>>,
    apikeys: &Arc<RwLock<apikeys::ApikeysStore>>,
    _manager: &Arc<scheduler::InstanceManager>,
    last_mtime: &Arc<Mutex<Option<std::time::SystemTime>>>,
) {
    // Skip if the file's mtime hasn't changed since our last reload.
    let mtime = match std::fs::metadata(path).and_then(|m| m.modified()) {
        Ok(mt) => mt,
        Err(_) => return,
    };
    {
        let mut last = last_mtime.lock().await;
        if *last == Some(mtime) {
            debug!("config mtime unchanged, skipping reload");
            return;
        }
        *last = Some(mtime);
    }
    info!("config file changed, reloading…");
    match config::Config::load(path) {
        Ok(new_cfg) => {
            // Reload apikeys if the path changed.
            {
                let old = cfg.read().await;
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

            // Atomic swap.
            let model_count = new_cfg.models.len();
            *cfg.write().await = new_cfg;
            info!(
                "config reloaded successfully — {} model(s), server config may have changed",
                model_count
            );
            // TODO: reconcile InstanceManager model list changes.
        }
        Err(e) => {
            error!("config reload rejected (keeping old): {}", e);
        }
    }
}

/// Atomic apikeys reload.
async fn handle_apikeys_reload(
    path: &std::path::Path,
    apikeys: &Arc<RwLock<apikeys::ApikeysStore>>,
    last_mtime: &Arc<Mutex<Option<std::time::SystemTime>>>,
) {
    let mtime = match std::fs::metadata(path).and_then(|m| m.modified()) {
        Ok(mt) => mt,
        Err(_) => return,
    };
    {
        let mut last = last_mtime.lock().await;
        if *last == Some(mtime) {
            debug!("apikeys mtime unchanged, skipping reload");
            return;
        }
        *last = Some(mtime);
    }
    info!("apikeys file changed, reloading…");
    match apikeys::ApikeysStore::load(path) {
        Ok(new_keys) => {
            let count = new_keys.len();
            *apikeys.write().await = new_keys;
            info!("apikeys reloaded successfully — {} key(s)", count);
        }
        Err(e) => {
            error!("apikeys reload rejected (keeping old): {}", e);
        }
    }
}