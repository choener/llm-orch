use clap::{Parser, Subcommand};
use llm_orch::{
    apikeys, config, debug_log, gpu, keepalive, keytool, mqtt, peers, reload, scheduler, server,
    stats, watcher,
};
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

    /// Path to the API keys file (`label: key` per line).
    ///
    /// Deliberately a command-line argument and not a `config.yaml` field:
    /// whoever is allowed to edit the config must not be able to redirect
    /// authentication to a key file of their own. Omit it to run with no
    /// keys at all (default-deny: every request is rejected). The file is
    /// hot-reloaded, but its *path* is fixed for the process lifetime.
    #[arg(
        short = 'k',
        long = "apikeyfile",
        value_name = "PATH",
        value_hint = clap::ValueHint::FilePath
    )]
    apikeyfile: Option<std::path::PathBuf>,

    /// Refuse plaintext apikeys: only `sha256:` digest entries are accepted.
    ///
    /// A command-line guarantee rather than a config field, for the same
    /// reason the key file's path is: whoever may edit config.yaml must not be
    /// able to relax it. A key file mixing both forms still loads without
    /// this flag; set it once every entry is hashed.
    #[arg(long = "require-hashed-apikeys")]
    require_hashed_apikeys: bool,

    /// Validate the configuration file and exit without starting the server.
    /// Exits 0 on valid config, non-zero on errors.  With `--apikeyfile`
    /// the key file is validated too, so CI can check a deployment's whole
    /// wiring.
    #[arg(long = "check-config", value_name = "PATH", value_hint = clap::ValueHint::FilePath)]
    check_config: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

/// Operator tooling for the apikeys file.
///
/// llm-orch only ever *reads* the key file at runtime; these commands are the
/// supported way to write or check it.
#[derive(Subcommand, Debug)]
enum Command {
    /// Mint a new API key: prints the token once (hand it to the client) and
    /// the hashed line to paste into the apikeys file.
    Keygen {
        /// Label identifying the key holder in logs and statistics.
        #[arg(long)]
        label: String,
        /// How many keys to mint (labels get a -1, -2, … suffix).
        #[arg(long, default_value_t = 1)]
        count: usize,
    },
    /// Convert plaintext entries in the apikeys file to `sha256:` digests.
    /// Idempotent and format-preserving; prints the result unless --write.
    HashApikeys {
        #[command(flatten)]
        file: ApikeyFileArg,
        /// Rewrite the file in place (atomic, mode 0600) instead of printing it.
        #[arg(long)]
        write: bool,
    },
    /// Report which label a bearer token maps to. The token is read from
    /// stdin (hashed exactly as sent) so it never appears in the process list.
    VerifyApikey {
        #[command(flatten)]
        file: ApikeyFileArg,
    },
}

/// `--apikeyfile` for the apikeys subcommands (required there).
#[derive(Parser, Debug)]
struct ApikeyFileArg {
    /// Path to the API keys file.
    #[arg(
        short = 'k',
        long = "apikeyfile",
        value_name = "PATH",
        value_hint = clap::ValueHint::FilePath
    )]
    apikeyfile: std::path::PathBuf,
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

    // ── apikeys tooling ─────────────────────────────────────────────────
    // The daemon never writes the key file, so hashing/verification lives here.
    if let Some(command) = &cli.command {
        return run_command(command);
    }

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
                for w in cfg.warnings() {
                    println!("WARNING: {w}");
                }
                // The key file is part of the deployment wiring, so validate
                // it here too — unlike at runtime, a missing/unreadable file
                // is a hard error (a check that passes would deny everything).
                let check_opts = apikeys::ApikeysOptions {
                    require_hashed: cli.require_hashed_apikeys,
                };
                match &cli.apikeyfile {
                    Some(p) => match apikeys::ApikeysStore::load_with(p, check_opts) {
                        Ok(store) => println!(
                            "OK: {} apikey(s) from {} ({} hashed, {} plaintext)",
                            store.len(),
                            p.display(),
                            store.hashed_len(),
                            store.plaintext_len()
                        ),
                        Err(e) => {
                            eprintln!("ERROR: apikeys file {}: {}", p.display(), e);
                            std::process::exit(1);
                        }
                    },
                    None => println!(
                        "WARNING: no --apikeyfile given — every request would be \
                         denied (default-deny)"
                    ),
                }
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
    for w in cfg.warnings() {
        warn!("{w}");
    }

    // 2. Load API keys from --apikeyfile.
    //    No file, or an unreadable one ⇒ empty store (fail-closed).
    let apikeys_path = cli.apikeyfile;
    let apikeys_opts = apikeys::ApikeysOptions {
        require_hashed: cli.require_hashed_apikeys,
    };
    if apikeys_opts.require_hashed {
        info!("apikeys: plaintext entries are refused (--require-hashed-apikeys)");
    }
    let apikeys = match &apikeys_path {
        Some(path) => {
            info!("loading apikeys: {}", path.display());
            match apikeys::ApikeysStore::load_with(path, apikeys_opts) {
                Ok(a) => a,
                Err(e) => {
                    warn!("could not load apikeys (fail-closed): {}", e);
                    apikeys::ApikeysStore::empty()
                }
            }
        }
        None => {
            warn!("no --apikeyfile given — fail-closed: all requests will be denied");
            apikeys::ApikeysStore::empty()
        }
    };
    info!(
        "loaded {} apikey(s) ({} hashed, {} plaintext){}",
        apikeys.len(),
        apikeys.hashed_len(),
        apikeys.plaintext_len(),
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

    // ── Cluster peering (docs/006) ────────────────────────────────
    // The registry owns peer catalogs and materializes peer models as
    // remote ModelConfig entries in the effective (shared) config.  No
    // peers configured ⇒ inert.
    let peers = Arc::new(peers::PeerRegistry::new(manager.client().clone()));
    peers.install_runtime(Arc::clone(&shared_cfg), Arc::clone(&manager));
    if !shared_cfg.read().await.peers.is_empty() {
        info!(
            peers = ?shared_cfg.read().await.peers.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            "cluster peers configured — fetching initial catalogs"
        );
        peers.reconcile(&shared_cfg.read().await.clone());
        peers.refresh_all_catalogs().await;
        peers.rematerialize().await;
    }

    // TODO §7: start HTTP server.

    // ── Start HTTP server §7 ───────────────────────────────────────────
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    info!(
        version = env!("CARGO_PKG_VERSION"),
        "llm-orch starting (types-idempotent, debug-log, autoscale)"
    );
    let debug_loggers = Arc::new(debug_log::DebugLoggers::new());
    let stats = stats::from_config(&shared_cfg.read().await.stats);
    if stats.path().is_some() {
        info!(path = %stats.path().unwrap().display(), "statistics enabled");
    }
    let gpu_rw = Arc::clone(&gpu_snapshot);
    let app_state = server::AppState {
        config: Arc::clone(&shared_cfg),
        apikeys: Arc::clone(&shared_apikeys),
        manager: Arc::clone(&manager),
        client: manager.client().clone(),
        gpu: gpu_snapshot,
        debug_loggers,
        stats: Arc::clone(&stats),
        peers: Arc::clone(&peers),
    };
    let mut server_task = tokio::spawn(server::serve(app_state, shutdown_rx));

    // ── MQTT stats publisher (docs/007) ─────────────────────────────
    // Inert unless the `mqtt:` config section is set.  A dead broker
    // never affects request serving; hot-reload reconfigures via the
    // controller shared with the reload task below.
    let mqtt_ctl = mqtt::MqttController::new(mqtt::MqttDeps {
        stats: Arc::clone(&stats),
        gpu: gpu_rw,
        peers: Arc::clone(&peers),
        config: Arc::clone(&shared_cfg),
    });
    mqtt_ctl.apply(&shared_cfg.read().await.clone()).await;

    // ── File watchers + reload loop §6 ───────────────────────────────────
    let config_path = cli.config.clone();
    let (reload_tx, mut reload_rx) = broadcast::channel::<watcher::FileChange>(16);
    let watcher_tx = reload_tx.clone();
    // Behind a lock so the reload task can rebuild the watcher when the key
    // file set changes on reload.  Dropping the handle stops watching.
    let watch_handle = Arc::new(Mutex::new(None::<watcher::WatchHandle>));
    // Watch the config, the apikeys file (fixed by --apikeyfile), and every
    // key file (their contents feed remote `remote_auth` resolution at
    // request time).
    let mut watch_paths = vec![config_path.clone()];
    if let Some(path) = &apikeys_path {
        watch_paths.push(path.clone());
    }
    watch_paths.extend(shared_cfg.read().await.key_files.values().cloned());
    *watch_handle.lock().await =
        match watcher::watch_paths(watch_paths, reload_tx, Duration::from_secs(2)).await {
            Ok(wh) => {
                info!("watching config, apikeys and key files for changes");
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
        let peers = Arc::clone(&peers);
        let mqtt_ctl = Arc::clone(&mqtt_ctl);
        let last_mtime = Arc::clone(&last_reload_mtime);
        let last_ak_mtime = Arc::clone(&last_apikeys_mtime);
        let apikey_opts = apikeys_opts;
        let watch_handle = Arc::clone(&watch_handle);
        async move {
            // Key file paths currently being watched.
            let mut watched_keyfiles: Vec<std::path::PathBuf> =
                cfg.read().await.key_files.values().cloned().collect();
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
                            &manager,
                            &stats,
                            &peers,
                            &mqtt_ctl,
                            &last_mtime,
                        )
                        .await;
                        if let Some(path) = &apikeys_path {
                            reload::handle_apikeys_reload(
                                path,
                                &apikeys,
                                apikey_opts,
                                &last_ak_mtime,
                            )
                            .await;
                        }
                        continue;
                    }
                    // All senders dropped — watcher is gone (shutdown).
                    Err(broadcast::error::RecvError::Closed) => break,
                };
                // The watcher already pre-filters to our watched paths.
                // Determine which file changed by comparing to canonical paths.
                let cfg_p = config_path.canonicalize().ok();
                let ak_p = apikeys_path.as_ref().and_then(|p| p.canonicalize().ok());
                let changed = event.path.canonicalize().ok();

                match (&changed, &cfg_p) {
                    (Some(c), Some(canon)) if c == canon => {
                        reload::handle_config_reload(
                            &event.path,
                            &cfg,
                            &manager,
                            &stats,
                            &peers,
                            &mqtt_ctl,
                            &last_mtime,
                        )
                        .await;
                    }
                    _ => {}
                }
                match (&changed, &ak_p) {
                    (Some(c), Some(canon)) if c == canon => {
                        reload::handle_apikeys_reload(
                            &event.path,
                            &apikeys,
                            apikey_opts,
                            &last_ak_mtime,
                        )
                        .await;
                    }
                    _ => {}
                }

                // A watched key file changed → re-read its contents into
                // the manager.  The config itself is unchanged, so no
                // reconcile is needed: `remote_auth` resolves against the
                // fresh contents at request time.
                let current_keyfiles = cfg
                    .read()
                    .await
                    .key_files
                    .values()
                    .cloned()
                    .collect::<Vec<_>>();
                if let Some(c) = &changed
                    && current_keyfiles
                        .iter()
                        .any(|p| p.canonicalize().ok().as_ref() == Some(c))
                {
                    let c = cfg.read().await;
                    manager.refresh_key_files(&c);
                    // The MQTT password may reference the rotated key
                    // file — re-resolve and respawn the publisher if it
                    // changed.
                    mqtt_ctl.apply(&c).await;
                }

                // If a config reload changed the key file set, rebuild the
                // watcher to follow the new files (replacing the handle drops
                // the old watcher).  The apikeys path is fixed at startup, so
                // it never enters this comparison.
                let current_keyfiles = {
                    let c = cfg.read().await;
                    c.key_files.values().cloned().collect::<Vec<_>>()
                };
                if current_keyfiles != watched_keyfiles {
                    let mut paths = vec![config_path.clone()];
                    if let Some(path) = &apikeys_path {
                        paths.push(path.clone());
                    }
                    paths.extend(current_keyfiles.clone());
                    match watcher::watch_paths(paths, watcher_tx.clone(), Duration::from_secs(2))
                        .await
                    {
                        Ok(wh) => {
                            info!(
                                "watch set changed — now following {} key file(s)",
                                current_keyfiles.len()
                            );
                            *watch_handle.lock().await = Some(wh);
                            watched_keyfiles = current_keyfiles;
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
    // Stop new scheduler admissions before beginning axum's bounded drain;
    // existing queued/running handlers may still finish or be cancelled by
    // their clients during the configured shutdown window.
    manager.begin_shutdown();
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
    // Final MQTT publish + retained "offline" (bounded).
    mqtt_ctl.shutdown().await;
    info!("stopped");

    Ok(())
}

/// apikeys operator tooling (`keygen`, `hash-apikeys`, `verify-apikey`).
///
/// These are the only supported way to change the key file: the running
/// orchestrator only ever reads it.
fn run_command(command: &Command) -> Result<(), Box<dyn std::error::Error>> {
    fn entries(n: usize) -> &'static str {
        if n == 1 { "entry" } else { "entries" }
    }

    match command {
        Command::Keygen { label, count } => {
            for key in keytool::generate_keys(label, *count) {
                println!(
                    "# token for `{}` — hand it to the client once; it is not stored\n# anywhere and cannot be recovered later:",
                    key.label
                );
                println!("{}", key.token);
                println!("# paste this line into the apikeys file:");
                print!("{}", key.line);
            }
            Ok(())
        }
        Command::HashApikeys { file, write } => {
            let path = &file.apikeyfile;
            let contents = std::fs::read_to_string(path)?;
            let report = match keytool::hash_file_contents(&contents) {
                Ok(report) => report,
                Err(e) => {
                    eprintln!("ERROR: {e}");
                    std::process::exit(1);
                }
            };
            if report.is_noop() {
                println!(
                    "{}: nothing to convert ({} hashed {} already)",
                    path.display(),
                    report.already_hashed,
                    entries(report.already_hashed)
                );
                return Ok(());
            }
            if !*write {
                println!(
                    "# {} plaintext {} in {} would become hashed (re-run with --write):",
                    report.converted.len(),
                    entries(report.converted.len()),
                    path.display()
                );
                print!("{}", report.text);
                return Ok(());
            }
            if let Err(e) = keytool::write_key_file(path, &report.text) {
                eprintln!("ERROR: {e}");
                std::process::exit(1);
            }
            println!(
                "{}: hashed {} ({}) — the plaintext tokens are now unrecoverable;\nhot-reload picks the file up without a restart",
                path.display(),
                entries(report.converted.len()),
                report.converted.join(", ")
            );
            Ok(())
        }
        Command::VerifyApikey { file } => {
            // Default options on purpose: identifying a plaintext entry is
            // useful even on an instance that refuses to serve it.
            let store = match apikeys::ApikeysStore::load(&file.apikeyfile) {
                Ok(store) => store,
                Err(e) => {
                    eprintln!("ERROR: {}: {}", file.apikeyfile.display(), e);
                    std::process::exit(1);
                }
            };
            use std::io::Read;
            let mut raw = String::new();
            std::io::stdin().read_to_string(&mut raw)?;
            if raw.is_empty() {
                eprintln!(
                    "ERROR: no token on stdin — pipe it byte-exact, e.g.\n  \
                     printf '%s' \"$TOKEN\" | llm-orch verify-apikey -k {}",
                    file.apikeyfile.display()
                );
                std::process::exit(2);
            }
            let token = keytool::token_from_stdin(&raw);
            match store.authenticate(token) {
                Some(label) => {
                    println!("matched label: {label}");
                    Ok(())
                }
                None => {
                    // Reporting the digest of what was actually read lets an
                    // operator find the mismatching line (stray newline, lost
                    // space, …) instead of guessing.
                    println!(
                        "no match — sha256 of the token read from stdin: {}",
                        apikeys::sha256_hex(token)
                    );
                    std::process::exit(1);
                }
            }
        }
    }
}
