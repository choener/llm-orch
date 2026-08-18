use clap::Parser;
use llm_orch::{apikeys, config, debug_log, gpu, keepalive, reload, scheduler, server, watcher};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, broadcast};
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

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
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    if std::env::var("LLM_ORCH_LOG_JSON").is_ok() {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(env_filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    }

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

    // 3. Start GPU metrics reader (periodic sysfs polling).
    let (gpu_reader, _gpu_poll_task) = gpu::GpuReader::start(Duration::from_secs(5));
    let gpu_snapshot = gpu_reader.snapshot_arc();
    info!("gpu metrics reader started");

    // 4. Create GPU keep-alive manager.
    let keepalive = keepalive::KeepAliveManager::new(&cfg.keep_alive).map(Arc::new);
    if keepalive.is_some() {
        info!("keep-alive configured");
    }

    // 5. Create the instance manager.
    let (manager, release_rx, crash_rx) =
        scheduler::InstanceManager::new(&cfg, Arc::clone(&gpu_snapshot), keepalive);
    let manager = Arc::new(manager);
    info!("instance manager ready");

    // Spawn the background release-processing task.
    // This task receives model names from Instance::release_slot (via
    // the unbounded channel) and calls record_metrics_event + wake_one.
    // Because the task holds no locks when it receives a message, it can
    // safely acquire instances.read() → metrics.write() without deadlocking
    // with the request-completion Drop path.
    tokio::spawn({
        let mgr = Arc::clone(&manager);
        async move {
            let mut rx = release_rx;
            while let Some(model_name) = rx.recv().await {
                mgr.record_metrics_event(&model_name, 1);
                mgr.wake_one(&model_name);
                // Finish removing draining instances whose last in-flight
                // request just completed.
                mgr.reap_drained(&model_name).await;
            }
        }
    });

    // Spawn the background crash-processing task.
    // Per-instance monitor tasks report unexpected child exits here; the
    // manager unregisters the crashed instance and blocks the model after
    // too many consecutive pre-output crashes.
    tokio::spawn({
        let mgr = Arc::clone(&manager);
        async move {
            let mut rx = crash_rx;
            while let Some(handle) = rx.recv().await {
                mgr.handle_crash(handle).await;
            }
        }
    });

    // Spawn the autoscaler background task.
    // Periodically evaluates per-model load metrics and scales instances
    // up or down with hysteresis to avoid reacting to brief bursts.
    tokio::spawn({
        let mgr = Arc::clone(&manager);
        async move {
            loop {
                mgr.evaluate_autoscale().await;
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        }
    });

    // Shared state for hot-reload.
    let shared_cfg = Arc::new(RwLock::new(cfg));
    let shared_apikeys = Arc::new(RwLock::new(apikeys));
    let last_reload_mtime = Arc::new(Mutex::new(None::<std::time::SystemTime>));
    let last_apikeys_mtime = Arc::new(Mutex::new(None::<std::time::SystemTime>));

    // TODO §7: start HTTP server.

    // ── Start HTTP server §7 ───────────────────────────────────────────
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    info!(
        version = env!("CARGO_PKG_VERSION"),
        "llm-orch starting (types-idempotent, debug-log, autoscale)"
    );
    let debug_loggers = Arc::new(debug_log::DebugLoggers::new());
    let app_state = server::AppState {
        config: Arc::clone(&shared_cfg),
        apikeys: Arc::clone(&shared_apikeys),
        manager: Arc::clone(&manager),
        client: manager.client().clone(),
        gpu: gpu_snapshot,
        debug_loggers,
    };
    let mut server_task = tokio::spawn(server::serve(app_state, shutdown_rx));

    // ── File watchers + reload loop §6 ───────────────────────────────────
    let config_path = cli.config.clone();
    let apikeys_path = shared_cfg.read().await.apikeys_file.clone();
    let (reload_tx, mut reload_rx) = broadcast::channel::<watcher::FileChange>(16);
    let watcher_tx = reload_tx.clone();
    // Behind a lock so the reload task can rebuild the watcher when the
    // apikeys path changes on reload.  Dropping the handle stops watching.
    let watch_handle = Arc::new(Mutex::new(None::<watcher::WatchHandle>));
    *watch_handle.lock().await = match watcher::watch_paths(
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
        let watch_handle = Arc::clone(&watch_handle);
        async move {
            // The apikeys path currently being watched — updated when a
            // config reload points us at a different file.
            let mut watched_apikeys = apikeys_path.clone();
            loop {
                let event = match reload_rx.recv().await {
                    Ok(event) => event,
                    // Bursty edits overflowed the broadcast channel — some
                    // events were lost.  Reloads are idempotent and
                    // mtime-gated, so resync both files and keep going
                    // instead of silently killing hot-reload for the rest
                    // of the process lifetime.
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(
                            skipped = n,
                            "reload events lagged — resyncing watched files"
                        );
                        reload::handle_config_reload(
                            &config_path,
                            &cfg,
                            &apikeys,
                            &manager,
                            &last_mtime,
                        )
                        .await;
                        reload::handle_apikeys_reload(&watched_apikeys, &apikeys, &last_ak_mtime)
                            .await;
                        continue;
                    }
                    // All senders dropped — watcher is gone (shutdown).
                    Err(broadcast::error::RecvError::Closed) => break,
                };
                // The watcher already pre-filters to our watched paths.
                // Determine which file changed by comparing to canonical paths.
                let cfg_p = config_path.canonicalize().ok();
                let ak_p = watched_apikeys.canonicalize().ok();
                let changed = event.path.canonicalize().ok();

                match (&changed, &cfg_p) {
                    (Some(c), Some(canon)) if c == canon => {
                        reload::handle_config_reload(
                            &event.path,
                            &cfg,
                            &apikeys,
                            &manager,
                            &last_mtime,
                        )
                        .await;
                    }
                    _ => {}
                }
                match (&changed, &ak_p) {
                    (Some(c), Some(canon)) if c == canon => {
                        reload::handle_apikeys_reload(&event.path, &apikeys, &last_ak_mtime).await;
                    }
                    _ => {}
                }

                // If a config reload changed the apikeys path, rebuild the
                // watcher to follow the new file (replacing the handle
                // drops the old watcher).
                let current_apikeys = cfg.read().await.apikeys_file.clone();
                if current_apikeys != watched_apikeys {
                    match watcher::watch_paths(
                        vec![config_path.clone(), current_apikeys.clone()],
                        watcher_tx.clone(),
                        Duration::from_secs(2),
                    )
                    .await
                    {
                        Ok(wh) => {
                            info!(
                                "apikeys path changed — now watching {}",
                                current_apikeys.display()
                            );
                            *watch_handle.lock().await = Some(wh);
                            watched_apikeys = current_apikeys;
                        }
                        Err(e) => {
                            warn!("watcher rebuild failed (keeping old watch set): {}", e);
                        }
                    }
                }
            }
            debug!("reload event channel closed — reload task exiting");
        }
    });

    info!("running — press Ctrl-C to stop");

    // ── Graceful shutdown §11 ───────────────────────────────────────────
    //
    // 1. Wait for SIGINT / SIGTERM.
    // 2. Signal axum to stop accepting new connections.
    // 3. Wait for in-flight requests to drain, up to
    //    `server.shutdown_drain_timeout_secs`, then abort the rest.
    // 4. Shut down all backend instances.

    // Wait for shutdown signal.
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigint =
            signal(SignalKind::interrupt()).expect("failed to register SIGINT handler");
        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
        tokio::select! {
            _ = sigint.recv() => info!("received SIGINT"),
            _ = sigterm.recv() => info!("received SIGTERM"),
            // A dead server task (bind failure, panic) must not leave the
            // daemon idling without a listener.
            res = &mut server_task => {
                error!("http server exited unexpectedly: {:?}", res);
                manager.shutdown_all().await;
                return Err("http server exited unexpectedly".into());
            }
        }
    }
    #[cfg(not(unix))]
    {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => info!("received Ctrl-C"),
            res = &mut server_task => {
                error!("http server exited unexpectedly: {:?}", res);
                manager.shutdown_all().await;
                return Err("http server exited unexpectedly".into());
            }
        }
    }

    // Signal axum to stop accepting and drain in-flight — bounded by the
    // configured drain timeout so one hung streaming request can't block
    // shutdown forever (plan §11).
    info!("stopping http server...");
    drop(shutdown_tx);
    let drain_timeout =
        Duration::from_secs(shared_cfg.read().await.server.shutdown_drain_timeout_secs);
    match tokio::time::timeout(drain_timeout, &mut server_task).await {
        Ok(Ok(Ok(()))) => info!("http server drained"),
        Ok(Ok(Err(e))) => warn!("http server stopped with error: {}", e),
        Ok(Err(e)) => warn!("http server task failed: {}", e),
        Err(_) => {
            warn!(
                timeout_secs = drain_timeout.as_secs(),
                "drain timeout exceeded — aborting remaining in-flight connections"
            );
            server_task.abort();
        }
    }

    // Shut down all backend instances.
    info!("shutting down backends...");
    manager.shutdown_all().await;
    info!("backends shut down");
    info!("stopped");

    Ok(())
}
