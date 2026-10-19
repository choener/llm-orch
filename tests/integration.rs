// ── End-to-end integration tests ─────────────────────────────────────────────
//
// Boots a real orchestrator (config, auth, scheduler, HTTP server) against
// the stub llama.cpp backend binary — no GPU or model required.

use futures_util::StreamExt;
use llm_orch::{apikeys, config, debug_log, gpu, reload, scheduler, server, stats};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, oneshot};

const STUB_BIN: &str = env!("CARGO_BIN_EXE_llm-orch-stub-backend");
const ORCH_BIN: &str = env!("CARGO_BIN_EXE_llm-orch");

// ── Helpers ──────────────────────────────────────────────────────────────────

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llm-orch-it-{}-{}", tag, uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_config(dir: &Path, listen_port: u16, models_yaml: &str) -> PathBuf {
    // drain_idle_timeout_secs: 1 — the global idle reaper (the per-model
    // idle_ttl is only a make-room protection window now); test servers
    // must reap idle instances quickly.
    let config_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\napikeys_file: \"apikeys.txt\"\n{models_yaml}"
    );
    let path = dir.join("config.yaml");
    std::fs::write(&path, config_yaml).unwrap();
    path
}

fn models_yaml(cmd_extra: &str, idle_ttl: u64) -> String {
    format!(
        "models:\n  - name: m\n    context_length: 4096\n    cmd: \"{STUB_BIN} --port {{port}}{cmd_extra}\"\n    idle_ttl: {idle_ttl}\n    max_instances: 1\n"
    )
}

struct TestServer {
    url: String,
    manager: Arc<scheduler::InstanceManager>,
    config: Arc<RwLock<config::Config>>,
    apikeys: Arc<RwLock<apikeys::ApikeysStore>>,
    stats: Arc<stats::StatsStore>,
    peers: Arc<llm_orch::peers::PeerRegistry>,
    mqtt: Arc<llm_orch::mqtt::MqttController>,
    _shutdown: oneshot::Sender<()>,
    _dir: PathBuf,
}

/// Boot a full composed server (mirror of main.rs wiring, minus signals
/// and file watchers) on an ephemeral port.
async fn boot(dir: &Path, apikeys_contents: &str, models_yaml: &str) -> TestServer {
    let listen_port = free_port();
    let config_path = write_config(dir, listen_port, models_yaml);
    boot_from_config_path(dir, &config_path, apikeys_contents).await
}

/// Boot from a complete config file, allowing integration tests to provide
/// device maps and static capacities that are independent of host GPU
/// metrics.
async fn boot_from_config_path(
    dir: &Path,
    config_path: &Path,
    apikeys_contents: &str,
) -> TestServer {
    std::fs::write(dir.join("apikeys.txt"), apikeys_contents).unwrap();
    let cfg = config::Config::load(config_path).unwrap();
    let keys = apikeys::ApikeysStore::load(&cfg.apikeys_file)
        .unwrap_or_else(|_| apikeys::ApikeysStore::empty());

    let (gpu_reader, _gpu_task) = gpu::GpuReader::start(Duration::from_secs(5));
    let gpu_snapshot = gpu_reader.snapshot_arc();
    let (manager, release_rx, crash_rx) =
        scheduler::InstanceManager::new(&cfg, Arc::clone(&gpu_snapshot), None);
    let manager = Arc::new(manager);

    // Background release/crash tasks (mirror main.rs).
    tokio::spawn({
        let mgr = Arc::clone(&manager);
        async move {
            let mut rx = release_rx;
            while let Some(model_name) = rx.recv().await {
                mgr.record_metrics_event(&model_name, 1);
                mgr.wake_one(&model_name);
                mgr.reap_drained(&model_name).await;
            }
        }
    });
    tokio::spawn({
        let mgr = Arc::clone(&manager);
        async move {
            let mut rx = crash_rx;
            while let Some(handle) = rx.recv().await {
                mgr.handle_crash(handle).await;
            }
        }
    });

    let shared_cfg = Arc::new(RwLock::new(cfg));
    let shared_keys = Arc::new(RwLock::new(keys));
    let stats_store = Arc::new(stats::StatsStore::new(
        shared_cfg.read().await.stats.as_ref().map(|s| &s.file),
    ));
    // Cluster peering: mirror main.rs wiring (registry + runtime
    // handles + initial materialization).
    let peers = Arc::new(llm_orch::peers::PeerRegistry::new(manager.client().clone()));
    peers.install_runtime(Arc::clone(&shared_cfg), Arc::clone(&manager));
    peers.reconcile(&shared_cfg.read().await.clone());
    // Mirror main.rs: fetch initial catalogs when peers are configured.
    if !shared_cfg.read().await.peers.is_empty() {
        peers.refresh_all_catalogs().await;
    }
    peers.rematerialize().await;
    let state = server::AppState {
        config: Arc::clone(&shared_cfg),
        apikeys: Arc::clone(&shared_keys),
        manager: Arc::clone(&manager),
        client: manager.client().clone(),
        gpu: Arc::clone(&gpu_snapshot),
        debug_loggers: Arc::new(debug_log::DebugLoggers::new()),
        stats: Arc::clone(&stats_store),
        peers: Arc::clone(&peers),
    };
    // Mirror main.rs: inert MQTT publisher unless the config enables it.
    let mqtt_ctl = llm_orch::mqtt::MqttController::new(llm_orch::mqtt::MqttDeps {
        stats: Arc::clone(&stats_store),
        gpu: gpu_snapshot,
        peers: Arc::clone(&peers),
        config: Arc::clone(&shared_cfg),
    });
    mqtt_ctl.apply(&shared_cfg.read().await.clone()).await;
    let (tx, rx) = oneshot::channel();
    let server_task = tokio::spawn(server::serve(state, rx));
    // The server binds immediately on a fresh port; give it a moment.
    tokio::time::sleep(Duration::from_millis(100)).await;
    if server_task.is_finished() {
        panic!("server task exited at startup: {:?}", server_task.await);
    }

    let listen = shared_cfg.read().await.server.listen.clone();
    TestServer {
        url: format!("http://{listen}"),
        manager,
        config: shared_cfg,
        stats: stats_store,
        apikeys: shared_keys,
        peers,
        mqtt: mqtt_ctl,
        _shutdown: tx,
        _dir: dir.to_path_buf(),
    }
}

fn authed(client: &reqwest::Client, url: &str) -> reqwest::RequestBuilder {
    client.post(url).bearer_auth("secret-key")
}

// ── Core happy path ──────────────────────────────────────────────────────────

#[tokio::test]
async fn happy_path_stream_release_and_ttl_unload() {
    let dir = temp_dir("happy");
    let srv = boot(&dir, "tester: secret-key\n", &models_yaml("", 1)).await;
    let client = reqwest::Client::new();

    // Authed streaming request → canned SSE chunks + [DONE].
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("tok-0"), "expected streamed chunks: {body}");
    assert!(
        body.contains("[DONE]"),
        "expected stream terminator: {body}"
    );

    // The stub backend was spawned on demand and is registered.
    assert_eq!(srv.manager.instance_counts().get("m").copied(), Some(1));

    // The completion was recorded (proves the stream ran to completion).
    let completions = srv.manager.recent_completions_snapshot();
    assert_eq!(completions.get("m").map(|c| c.len()), Some(1));

    // The 1 s drain idle timeout evicts the instance — which transitively
    // proves the in-flight slot was released: a leaked slot would keep the
    // instance busy and block the IfIdle removal.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    srv.manager.evaluate_autoscale().await;
    assert_eq!(
        srv.manager.instance_counts().get("m").copied(),
        Some(0),
        "idle instance must be despawned after drain_idle_timeout"
    );
}

#[tokio::test]
async fn queued_stream_emits_sse_comment_before_assignment() {
    let dir = temp_dir("queued-heartbeat");
    let models = models_yaml(" --chunks 20 --chunk-delay-ms 100", 60).replace(
        "max_instances: 1",
        "max_instances: 1\n    max_concurrent: 1",
    );
    let listen_port = free_port();
    let config_path = write_config(&dir, listen_port, &models);
    let config = std::fs::read_to_string(&config_path).unwrap().replace(
        "drain_idle_timeout_secs: 1",
        "drain_idle_timeout_secs: 1\n  queue_heartbeat_secs: 1",
    );
    std::fs::write(&config_path, config).unwrap();
    std::fs::write(dir.join("apikeys.txt"), "tester: secret-key\n").unwrap();

    let cfg = config::Config::load(&config_path).unwrap();
    let keys = apikeys::ApikeysStore::load(&cfg.apikeys_file).unwrap();
    let (gpu_reader, _gpu_task) = gpu::GpuReader::start(Duration::from_secs(5));
    let gpu_snapshot = gpu_reader.snapshot_arc();
    let (manager, release_rx, crash_rx) =
        scheduler::InstanceManager::new(&cfg, Arc::clone(&gpu_snapshot), None);
    let manager = Arc::new(manager);
    let config_state = Arc::new(RwLock::new(cfg));
    let apikeys_state = Arc::new(RwLock::new(keys));
    let peers = Arc::new(llm_orch::peers::PeerRegistry::new(manager.client().clone()));
    peers.install_runtime(Arc::clone(&config_state), Arc::clone(&manager));
    peers.reconcile(&config_state.read().await.clone());
    peers.rematerialize().await;
    let state = server::AppState {
        config: Arc::clone(&config_state),
        apikeys: Arc::clone(&apikeys_state),
        manager: Arc::clone(&manager),
        client: manager.client().clone(),
        gpu: gpu_snapshot,
        debug_loggers: Arc::new(debug_log::DebugLoggers::new()),
        stats: Arc::new(stats::StatsStore::default()),
        peers: Arc::clone(&peers),
    };
    tokio::spawn({
        let manager = Arc::clone(&manager);
        async move {
            let mut rx = release_rx;
            while let Some(model) = rx.recv().await {
                manager.record_metrics_event(&model, 1);
                manager.wake_one(&model);
                manager.reap_drained(&model).await;
            }
        }
    });
    tokio::spawn({
        let manager = Arc::clone(&manager);
        async move {
            let mut rx = crash_rx;
            while let Some(handle) = rx.recv().await {
                manager.handle_crash(handle).await;
            }
        }
    });
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server_task = tokio::spawn(server::serve(state, shutdown_rx));
    let url = format!("http://127.0.0.1:{listen_port}");
    let client = reqwest::Client::new();
    for _ in 0..50 {
        if client.get(format!("{url}/health")).send().await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let first = authed(&client, &format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "first"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);

    let second = authed(&client, &format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "second"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 200);
    assert_eq!(
        second
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream"),
    );
    assert_eq!(
        second
            .headers()
            .get("cache-control")
            .and_then(|value| value.to_str().ok()),
        Some("no-cache"),
    );
    assert_eq!(
        second
            .headers()
            .get("x-accel-buffering")
            .and_then(|value| value.to_str().ok()),
        Some("no"),
    );
    assert!(second.headers().get("x-llm-orch-request-id").is_some());
    let mut body = second.bytes_stream();
    let chunk = tokio::time::timeout(Duration::from_secs(3), body.next())
        .await
        .expect("queued SSE must emit a heartbeat")
        .expect("heartbeat stream must remain open")
        .unwrap();
    let text = String::from_utf8_lossy(&chunk);
    assert!(
        text.starts_with(": llm-orch waiting;"),
        "unexpected heartbeat: {text}"
    );
    assert!(
        !text.contains("data:"),
        "heartbeat must not be an OpenAI data event"
    );

    // Releasing the first stream assigns the queued request.  The response
    // body must then contain the backend's normal chunks and terminator,
    // not a synthetic completion.
    drop(first);
    let mut forwarded = String::new();
    for _ in 0..300 {
        let Some(chunk) = tokio::time::timeout(Duration::from_secs(3), body.next())
            .await
            .expect("assigned backend stream must continue")
            .expect("assigned stream must not end before [DONE]")
            .ok()
        else {
            break;
        };
        forwarded.push_str(&String::from_utf8_lossy(&chunk));
        if forwarded.contains("[DONE]") {
            break;
        }
    }
    assert!(
        forwarded.contains("tok-0"),
        "backend data was not spliced: {forwarded}"
    );
    assert!(
        forwarded.contains("[DONE]"),
        "backend terminator was not preserved: {forwarded}"
    );

    drop(body);

    // An explicitly opted-in client receives a versioned status event while
    // waiting, whereas the default stream above remained comment-only.
    let holder = authed(&client, &format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "holder"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    let status_response = authed(&client, &format!("{url}/v1/chat/completions"))
        .header("X-LLM-Orch-Wait-Events", "1")
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "status"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(status_response.status(), 200);
    let mut status_body = status_response.bytes_stream();
    let status_chunk = tokio::time::timeout(Duration::from_secs(3), status_body.next())
        .await
        .expect("opt-in status stream must emit an event")
        .expect("opt-in status stream must remain open")
        .unwrap();
    let status_text = String::from_utf8_lossy(&status_chunk);
    assert!(status_text.contains("event: llm-orch.status"));
    assert!(status_text.contains("\"version\":1"));
    assert!(status_text.contains("\"state\":\"queued\""));
    drop(status_body);
    drop(holder);

    drop(shutdown_tx);
    let _ = tokio::time::timeout(Duration::from_secs(3), server_task).await;
}

#[tokio::test]
async fn migration_preserves_busy_victim_and_loads_target_after_handoff() {
    let Some(slots) = real_pci_slots(2) else {
        return;
    };
    let dir = temp_dir("migration");
    let listen_port = free_port();
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, migration_cuda_yaml(listen_port, &slots)).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    // Keep both resident models active while the target migration starts.
    let first = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "m1",
            "messages": [{"role": "user", "content": "m1"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);
    let first_task = tokio::spawn(async move { first.text().await.unwrap() });

    let second = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "m2",
            "messages": [{"role": "user", "content": "m2"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 200);
    let second_task = tokio::spawn(async move { second.text().await.unwrap() });
    for _ in 0..300 {
        if srv.manager.instance_counts().get("m1").copied() == Some(1)
            && srv.manager.instance_counts().get("m2").copied() == Some(1)
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let target = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "preferred",
            "messages": [{"role": "user", "content": "large"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(target.status(), 200);
    let mut target_body = target.bytes_stream();
    let heartbeat = tokio::time::timeout(Duration::from_secs(3), target_body.next())
        .await
        .expect("migration wait must emit a heartbeat")
        .expect("migration stream must remain open")
        .unwrap();
    let heartbeat_text = String::from_utf8_lossy(&heartbeat);
    assert!(
        heartbeat_text.contains("llm-orch waiting"),
        "expected migration heartbeat, got: {heartbeat_text}"
    );

    let (first_body, second_body) = tokio::time::timeout(
        Duration::from_secs(10),
        futures_util::future::join(first_task, second_task),
    )
    .await
    .expect("busy resident requests must finish without migration interruption");
    assert!(first_body.unwrap().contains("[DONE]"));
    assert!(second_body.unwrap().contains("[DONE]"));

    let mut target_text = String::new();
    for _ in 0..20 {
        let Some(chunk) = tokio::time::timeout(Duration::from_secs(3), target_body.next())
            .await
            .expect("target stream must continue after handoff")
            .expect("target stream must reach its terminator")
            .ok()
        else {
            break;
        };
        target_text.push_str(&String::from_utf8_lossy(&chunk));
        if target_text.contains("[DONE]") {
            break;
        }
    }
    assert!(
        target_text.contains("tok-0"),
        "target was not loaded: {target_text}"
    );
    assert!(
        target_text.contains("[DONE]"),
        "target terminator missing: {target_text}"
    );
    assert_eq!(srv.manager.instance_counts().get("large").copied(), Some(1));
    assert_eq!(srv.manager.instance_counts().get("m1").copied(), Some(1));
    assert_eq!(srv.manager.instance_counts().get("m2").copied(), Some(1));
    let metrics = srv.manager.scheduler_metrics_snapshot();
    assert!(metrics.migration_started_total >= 1);
    assert!(metrics.migration_completed_total >= 1);
    assert!(metrics.queue_enqueued_total >= 1);
}

#[tokio::test]
async fn non_streaming_chat_completions() {
    let dir = temp_dir("aggregate");
    let srv = boot(&dir, "tester: secret-key\n", &models_yaml("", 60)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body.pointer("/choices/0/message/content")
            .and_then(|v| v.as_str()),
        Some("tok-0 tok-1 tok-2 ")
    );
    assert_eq!(
        body.pointer("/usage/completion_tokens")
            .and_then(|v| v.as_u64()),
        Some(3)
    );
}

// ── Auth ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn fail_closed_auth_denies_everything_without_keys() {
    let dir = temp_dir("failclosed");
    // Empty apikeys file → all endpoints must deny, with or without a key.
    let srv = boot(&dir, "", &models_yaml("", 60)).await;
    let client = reqwest::Client::new();

    for path in ["/v1/models", "/v1/chat/completions", "/admin/status"] {
        let resp = client
            .request(
                if path.contains("chat") {
                    reqwest::Method::POST
                } else {
                    reqwest::Method::GET
                },
                format!("{}{}", srv.url, path),
            )
            .bearer_auth("secret-key")
            .json(&serde_json::json!({"model": "m", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401, "{path} must deny with empty apikeys");
    }
}

#[tokio::test]
async fn wrong_key_is_unauthorized() {
    let dir = temp_dir("wrongkey");
    let srv = boot(&dir, "tester: secret-key\n", &models_yaml("", 60)).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/models", srv.url))
        .bearer_auth("wrong-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    let resp = client
        .get(format!("{}/v1/models", srv.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

// ── Embeddings (§16) ─────────────────────────────────────────────────────────

#[tokio::test]
async fn embeddings_end_to_end() {
    let dir = temp_dir("embeddings");
    let srv = boot(&dir, "tester: secret-key\n", &models_yaml("", 60)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/embeddings", srv.url))
        .json(&serde_json::json!({"model": "m", "input": "hello"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body.pointer("/data/0/embedding")
            .and_then(|v| v.as_array())
            .map(|a| a.len()),
        Some(3)
    );
}

// ── Hot-reload ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn reload_rejects_bad_config_and_keeps_serving_old() {
    let dir = temp_dir("reload");
    let srv = boot(&dir, "tester: secret-key\n", &models_yaml("", 60)).await;
    let config_path = dir.join("config.yaml");
    let last_mtime = Arc::new(Mutex::new(None));

    // Write an invalid config (max_concurrent: 0) → rejected, old kept.
    let bad = std::fs::read_to_string(&config_path)
        .unwrap()
        .replace("idle_ttl: 60", "idle_ttl: 60\n    max_concurrent: 0");
    std::fs::write(&config_path, bad).unwrap();
    reload::handle_config_reload(
        &config_path,
        &srv.config,
        &srv.apikeys,
        &srv.manager,
        &srv.stats,
        &srv.peers,
        &srv.mqtt,
        &last_mtime,
    )
    .await;
    {
        let cfg = srv.config.read().await;
        assert_eq!(
            cfg.models[0].max_concurrent, 4,
            "bad config must be rejected"
        );
    }
    // And the old config still serves.
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/v1/models", srv.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // A subsequent valid edit IS picked up (mtime was not consumed by the
    // rejected reload).
    let good = std::fs::read_to_string(&config_path)
        .unwrap()
        .replace("max_concurrent: 0", "max_concurrent: 8");
    std::fs::write(&config_path, good).unwrap();
    reload::handle_config_reload(
        &config_path,
        &srv.config,
        &srv.apikeys,
        &srv.manager,
        &srv.stats,
        &srv.peers,
        &srv.mqtt,
        &last_mtime,
    )
    .await;
    let cfg = srv.config.read().await;
    assert_eq!(cfg.models[0].max_concurrent, 8, "valid reload must apply");
    assert_eq!(cfg.server.queue_heartbeat_secs, 10);
    drop(cfg);

    // A semantically invalid new scheduler setting is rejected atomically;
    // the last valid configuration remains active.
    let bad_heartbeat = std::fs::read_to_string(&config_path).unwrap().replace(
        "drain_idle_timeout_secs: 1",
        "drain_idle_timeout_secs: 1\n  queue_heartbeat_secs: 0",
    );
    std::fs::write(&config_path, bad_heartbeat).unwrap();
    reload::handle_config_reload(
        &config_path,
        &srv.config,
        &srv.apikeys,
        &srv.manager,
        &srv.stats,
        &srv.peers,
        &srv.mqtt,
        &last_mtime,
    )
    .await;
    let cfg = srv.config.read().await;
    assert_eq!(cfg.server.queue_heartbeat_secs, 10);
    assert_eq!(cfg.models[0].max_concurrent, 8);
}

// ── --check-config ───────────────────────────────────────────────────────────

#[test]
fn check_config_exits_zero_on_valid_nonzero_on_invalid() {
    let dir = temp_dir("checkconfig");
    let config_path = write_config(&dir, free_port(), &models_yaml("", 60));

    let ok = std::process::Command::new(ORCH_BIN)
        .arg("--check-config")
        .arg(&config_path)
        .output()
        .unwrap();
    assert!(ok.status.success(), "valid config must exit 0: {:?}", ok);

    std::fs::write(&config_path, "models: not-a-list\n").unwrap();
    let bad = std::process::Command::new(ORCH_BIN)
        .arg("--check-config")
        .arg(&config_path)
        .output()
        .unwrap();
    assert!(!bad.status.success(), "invalid config must exit non-zero");
    assert!(!bad.stderr.is_empty(), "invalid config must print an error");

    // The scheduler-specific validation must also be enforced by
    // --check-config.
    let scheduler_config = write_config(&dir, free_port(), &models_yaml("", 60));
    let heartbeat_bad = std::fs::read_to_string(&scheduler_config).unwrap().replace(
        "drain_idle_timeout_secs: 1",
        "drain_idle_timeout_secs: 1\n  queue_heartbeat_secs: 0",
    );
    std::fs::write(&scheduler_config, heartbeat_bad).unwrap();
    let bad_heartbeat = std::process::Command::new(ORCH_BIN)
        .arg("--check-config")
        .arg(&scheduler_config)
        .output()
        .unwrap();
    assert!(
        !bad_heartbeat.status.success(),
        "zero queue heartbeat must be rejected"
    );

    let policy_bad = std::fs::read_to_string(&scheduler_config)
        .unwrap()
        .replace("queue_heartbeat_secs: 0", "eviction_policy: priority_lru");
    std::fs::write(&scheduler_config, policy_bad).unwrap();
    let bad_policy = std::process::Command::new(ORCH_BIN)
        .arg("--check-config")
        .arg(&scheduler_config)
        .output()
        .unwrap();
    assert!(
        !bad_policy.status.success(),
        "unknown eviction policy must be rejected"
    );
}

// ── Graceful shutdown ────────────────────────────────────────────────────────

#[cfg(unix)]
#[tokio::test]
async fn sigterm_with_inflight_stream_exits_cleanly() {
    let dir = temp_dir("shutdown");
    let listen_port = free_port();
    // Slow stub (600 chunks × 1 s = 10 min stream), drain timeout 2 s.
    let models = models_yaml(" --chunks 600 --chunk-delay-ms 1000", 60);
    let config_path = write_config(&dir, listen_port, &models);
    // Inject a short drain timeout into the server section.
    let yaml = std::fs::read_to_string(&config_path).unwrap().replace(
        "port_range: \"ephemeral\"",
        "port_range: \"ephemeral\"\n  shutdown_drain_timeout_secs: 2",
    );
    std::fs::write(&config_path, yaml).unwrap();
    std::fs::write(dir.join("apikeys.txt"), "tester: secret-key\n").unwrap();

    // tracing_subscriber::fmt writes to stdout by default.
    let stdout = std::fs::File::create(dir.join("orch.log")).unwrap();
    let mut child = tokio::process::Command::new(ORCH_BIN)
        .arg("--config")
        .arg(&config_path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(stdout))
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .unwrap();

    // Wait for the server to come up.
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{listen_port}");
    let mut up = false;
    for _ in 0..50 {
        if client
            .get(format!("{url}/health"))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            up = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(up, "orchestrator did not start");

    // Fire a slow streaming request (runs ~10 min if left alone).
    let req_url = format!("{url}/v1/chat/completions");
    let req_task = tokio::spawn(async move {
        authed(&client, &req_url)
            .json(&serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }))
            .send()
            .await
    });
    // Let the stream get going (spawn + first chunks).
    tokio::time::sleep(Duration::from_secs(4)).await;

    // SIGTERM → graceful drain (2 s) → abort → backends killed → exit 0.
    let pid = child.id().unwrap() as libc::pid_t;
    unsafe { libc::kill(pid, libc::SIGTERM) };
    let status = tokio::time::timeout(Duration::from_secs(20), child.wait())
        .await
        .unwrap_or_else(|_| {
            let log = std::fs::read_to_string(dir.join("orch.log")).unwrap_or_default();
            let tail: Vec<&str> = log.lines().collect();
            let tail = &tail[tail.len().saturating_sub(20)..];
            panic!(
                "orchestrator must exit within 20s of SIGTERM; log tail:\n{}",
                tail.join("\n")
            );
        })
        .unwrap();
    assert!(status.success(), "expected clean exit, got {status}");

    req_task.abort();
}

// ── audio.cpp endpoints ──────────────────────────────────────────────────────
//
// The stub backend doubles as an audiocpp_server stand-in (accepts
// --config/--backend and serves /v1/audio/*).  The audio model's cmd uses
// the `{name}` placeholder and omits context_length entirely.

fn audio_models_yaml(idle_ttl: u64) -> String {
    format!(
        "models:\n  - name: audio-m\n    cmd: \"{STUB_BIN} --port {{port}} --config /etc/llm-orch/audio/{{name}}.json --host 127.0.0.1 --backend vulkan\"\n    idle_ttl: {idle_ttl}\n    max_instances: 1\n"
    )
}

#[tokio::test]
async fn request_status_endpoint_reports_completed_request() {
    let dir = temp_dir("request-status");
    let srv = boot(&dir, "tester: secret-key\n", &models_yaml("", 60)).await;
    let client = reqwest::Client::new();

    let response = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "status"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let request_id = response
        .headers()
        .get("x-llm-orch-request-id")
        .and_then(|value| value.to_str().ok())
        .expect("completed response must expose request ID")
        .to_owned();

    let status_response = client
        .get(format!("{}/v1/requests/{request_id}", srv.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(status_response.status(), 200);
    let status: serde_json::Value = status_response.json().await.unwrap();
    assert_eq!(status["request_id"], request_id);
    assert_eq!(status["state"], "completed");
    assert_eq!(status["resolved_model"], "m");

    let missing = client
        .get(format!("{}/v1/requests/missing", srv.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404);
}

#[tokio::test]
async fn audio_speech_wav_passthrough() {
    let dir = temp_dir("audio-wav");
    let srv = boot(&dir, "tester: secret-key\n", &audio_models_yaml(60)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/audio/speech", srv.url))
        .json(&serde_json::json!({"model": "audio-m", "input": "hello"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("audio/wav")
    );
    let body = resp.bytes().await.unwrap();
    assert!(body.starts_with(b"RIFF"), "expected WAV bytes: {body:?}");

    // The stub was spawned on demand via the audio-style cmd line.
    assert_eq!(
        srv.manager.instance_counts().get("audio-m").copied(),
        Some(1)
    );

    // The model shows up in /v1/models without an explicit context_length.
    let models: serde_json::Value = client
        .get(format!("{}/v1/models", srv.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(
        ids.contains(&"audio-m"),
        "audio model must be listed: {ids:?}"
    );
}

#[tokio::test]
async fn audio_speech_sse_stream() {
    let dir = temp_dir("audio-sse");
    let srv = boot(&dir, "tester: secret-key\n", &audio_models_yaml(60)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/audio/speech", srv.url))
        .json(&serde_json::json!({
            "model": "audio-m",
            "input": "stream me",
            "stream_format": "sse"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("speech.audio.delta"),
        "expected audio deltas: {body}"
    );
    assert!(
        body.contains("speech.audio.done"),
        "expected done event: {body}"
    );
    assert!(body.contains("[DONE]"), "expected terminator: {body}");
}

#[tokio::test]
async fn audio_transcriptions_json_and_multipart() {
    let dir = temp_dir("audio-asr");
    let srv = boot(&dir, "tester: secret-key\n", &audio_models_yaml(60)).await;
    let client = reqwest::Client::new();

    // JSON path.
    let resp = authed(&client, &format!("{}/v1/audio/transcriptions", srv.url))
        .json(&serde_json::json!({"model": "audio-m", "audio": "/tmp/in.wav"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["text"].as_str().unwrap().contains("stub transcript"));

    // Multipart path (body built manually — keeps reqwest minimal).
    let boundary = "itboundary42";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\naudio-m\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\nRIFFfake\r\n--{boundary}--\r\n"
    );
    let resp = authed(&client, &format!("{}/v1/audio/transcriptions", srv.url))
        .header(
            reqwest::header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["text"].as_str().unwrap().contains("stub transcript"));

    // Multipart with stream=true → SSE transcript deltas.
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\naudio-m\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\nRIFFfake\r\n--{boundary}--\r\n"
    );
    let resp = authed(&client, &format!("{}/v1/audio/transcriptions", srv.url))
        .header(
            reqwest::header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("transcript.text.delta"), "{text}");
    assert!(text.contains("[DONE]"), "{text}");
}

#[tokio::test]
async fn audio_voices_listing() {
    let dir = temp_dir("audio-voices");
    let srv = boot(&dir, "tester: secret-key\n", &audio_models_yaml(60)).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/audio/voices?model=audio-m", srv.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(
        json["voices"]
            .as_array()
            .map(|v| !v.is_empty())
            .unwrap_or(false),
        "expected voices: {json}"
    );

    // Missing model param → 400.
    let resp = client
        .get(format!("{}/v1/audio/voices", srv.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn audio_endpoints_require_auth() {
    let dir = temp_dir("audio-auth");
    let srv = boot(&dir, "tester: secret-key\n", &audio_models_yaml(60)).await;
    let client = reqwest::Client::new();

    for path in ["/v1/audio/speech", "/v1/audio/transcriptions"] {
        let resp = client
            .post(format!("{}{}", srv.url, path))
            .json(&serde_json::json!({"model": "audio-m", "input": "x"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 401, "{path} must require auth");
    }
    let resp = client
        .get(format!("{}/v1/audio/voices?model=audio-m", srv.url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn audio_model_idle_ttl_unload() {
    let dir = temp_dir("audio-ttl");
    let srv = boot(&dir, "tester: secret-key\n", &audio_models_yaml(1)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/audio/speech", srv.url))
        .json(&serde_json::json!({"model": "audio-m", "input": "hi"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        srv.manager.instance_counts().get("audio-m").copied(),
        Some(1)
    );

    tokio::time::sleep(Duration::from_millis(1200)).await;
    srv.manager.evaluate_autoscale().await;
    assert_eq!(
        srv.manager.instance_counts().get("audio-m").copied(),
        Some(0),
        "idle audio instance must be despawned after drain_idle_timeout"
    );
}

// ── Multi-model aliases (docs/003-smart-handling.md) ─────────────────────────

/// First real PCI slot (devices.cuda validation requires the slot to exist
/// on the host).  None when no PCI bus is visible — the test skips.
fn real_pci_slot() -> Option<String> {
    let mut slots: Vec<String> = std::fs::read_dir("/sys/bus/pci/devices")
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    slots.sort();
    slots.into_iter().next()
}

fn real_pci_slots(count: usize) -> Option<Vec<String>> {
    let mut slots: Vec<String> = std::fs::read_dir("/sys/bus/pci/devices")
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    slots.sort();
    (slots.len() >= count).then(|| slots.into_iter().take(count).collect())
}

/// Real DRM GPU PCI slots (what `devices.vulkan` validates against),
/// enumerated like `config.rs`'s `list_drm_pci_slots`.  Empty when the
/// host exposes no DRM cards — the Vulkan device-pin tests then skip.
fn real_drm_slots() -> Vec<String> {
    let mut slots = Vec::new();
    for i in 0..16 {
        let uevent = format!("/sys/class/drm/card{i}/device/uevent");
        if let Ok(contents) = std::fs::read_to_string(uevent) {
            for line in contents.lines() {
                if let Some(slot) = line.strip_prefix("PCI_SLOT_NAME=") {
                    slots.push(slot.trim().to_string());
                }
            }
        }
    }
    slots
}

/// Two resident 30 GB CUDA models share two synthetic 100 GB capacities;
/// the 90 GB target fits only after one resident model is migrated.
fn migration_cuda_yaml(listen_port: u16, slots: &[String]) -> String {
    format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  queue_wait_timeout_secs: 20\n  queue_heartbeat_secs: 1\n  migration_timeout_secs: 15\napikeys_file: \"apikeys.txt\"\ndevices:\n  cuda:\n    0:\n      pci: \"{}\"\n      vram_mb: 100000\n    1:\n      pci: \"{}\"\n      vram_mb: 100000\nmodels:\n  - name: m1\n    context_length: 4096\n    cmd: \"{STUB_BIN} --port {{port}} --chunks 60 --chunk-delay-ms 50\"\n    max_instances: 1\n    max_concurrent: 1\n    vram: 30000\n    cuda_devices: [0, 1]\n  - name: m2\n    context_length: 4096\n    cmd: \"{STUB_BIN} --port {{port}} --chunks 60 --chunk-delay-ms 50\"\n    max_instances: 1\n    max_concurrent: 1\n    vram: 30000\n    cuda_devices: [0, 1]\n  - name: large\n    context_length: 4096\n    cmd: \"{STUB_BIN} --port {{port}}\"\n    max_instances: 1\n    max_concurrent: 1\n    vram: 90000\n    cuda_devices: [0, 1]\naliases:\n  - name: preferred\n    targets: [large]\n    policy: prefer_order\n    make_room: migrate\n",
        slots[0], slots[1]
    )
}

/// Two CUDA models on one 48 GB GPU: m1 takes 20 GB, m2 takes 40 GB — they
/// cannot coexist.  Alias `smart` prefers m2 with spillover to m1.
/// `idle_ttl: 1` makes instances make-room-evictable after ~1 s idle.
fn alias_cuda_yaml(slot: &str) -> String {
    format!(
        "devices:\n  cuda:\n    0:\n      pci: \"{slot}\"\n      vram_mb: 48000\n\
         models:\n\
         \x20 - name: m1\n    context_length: 4096\n    cmd: \"{STUB_BIN} --port {{port}}\"\n    idle_ttl: 1\n    vram: 20000\n    gpus: 1\n    cuda_devices: [0]\n\
         \x20 - name: m2\n    context_length: 4096\n    cmd: \"{STUB_BIN} --port {{port}}\"\n    idle_ttl: 1\n    vram: 40000\n    gpus: 1\n    cuda_devices: [0]\n\
         aliases:\n  - name: smart\n    targets: [m2, m1]\n    policy: prefer_order\n    make_room: evict_idle\n"
    )
}

fn chat_body(model: &str) -> serde_json::Value {
    serde_json::json!({"model": model, "messages": [{"role": "user", "content": "hi"}]})
}

/// Rule proven by this test for **every** forwarding endpoint: a backend fault
/// is a 502 whose body carries the backend's own message, never a bare 500 —
/// both so a client can tell "retry the backend" from "llm-orch is broken",
/// and because the reason is what gets logged.
#[tokio::test]
async fn chat_backend_failure_is_bad_gateway() {
    let dir = temp_dir("chat-fail");
    let srv = boot(
        &dir,
        "tester: secret-key\n",
        &models_yaml(" --fail-status 503", 60),
    )
    .await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("m"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("backend returned 503"),
        "body must carry the backend's status, got: {body}"
    );
    assert!(
        body.contains("injected stub failure"),
        "body must carry the backend's own error message, got: {body}"
    );
}

#[tokio::test]
async fn alias_make_room_evicts_stale_idle_model() {
    let Some(slot) = real_pci_slot() else { return };
    let dir = temp_dir("alias-makeroom");
    let srv = boot(&dir, "tester: secret-key\n", &alias_cuda_yaml(&slot)).await;
    let client = reqwest::Client::new();

    // Load m1 directly (20 GB of 48 GB — m2's 40 GB no longer fits).
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("m1"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(srv.manager.instance_counts().get("m1").copied(), Some(1));

    // m1 becomes evictable once idle past its idle_ttl protection window.
    tokio::time::sleep(Duration::from_millis(1200)).await;

    // The alias prefers m2 → make-room evicts the stale m1 and loads m2.
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("smart"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(srv.manager.instance_counts().get("m2").copied(), Some(1));
    assert_eq!(
        srv.manager
            .instance_counts()
            .get("m1")
            .copied()
            .unwrap_or(0),
        0,
        "stale m1 must be evicted to make room for m2"
    );
}

#[tokio::test]
async fn direct_request_never_makes_room() {
    let Some(slot) = real_pci_slot() else { return };
    let dir = temp_dir("alias-direct");
    let srv = boot(&dir, "tester: secret-key\n", &alias_cuda_yaml(&slot)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("m1"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // A direct (non-alias) request for m2 must NOT evict m1 (decision 9):
    // the spawn simply does not fit and the request fails fast.
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("m2"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503, "m2 does not fit next to m1");
    assert_eq!(
        srv.manager.instance_counts().get("m1").copied(),
        Some(1),
        "m1 must not be evicted by a direct request"
    );
    assert_eq!(
        srv.manager
            .instance_counts()
            .get("m2")
            .copied()
            .unwrap_or(0),
        0
    );
}

/// Two plain models (no device pools) and an alias over both.
fn alias_plain_yaml(targets: &str) -> String {
    format!(
        "models:\n\
         \x20 - name: m1\n    context_length: 4096\n    cmd: \"{STUB_BIN} --port {{port}}\"\n    max_instances: 1\n\
         \x20 - name: m2\n    context_length: 4096\n    cmd: \"{STUB_BIN} --port {{port}}\"\n    max_instances: 1\n\
         aliases:\n  - name: any\n    targets: {targets}\n"
    )
}

#[tokio::test]
async fn alias_spawns_first_candidate_when_nothing_loaded() {
    let dir = temp_dir("alias-first");
    let srv = boot(&dir, "tester: secret-key\n", &alias_plain_yaml("[m1, m2]")).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("any"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        srv.manager.instance_counts().get("m1").copied(),
        Some(1),
        "the first candidate is spawned"
    );
    assert_eq!(srv.manager.instance_counts().get("m2").copied(), None);

    // Second request: warm slot on m1 — m2 still untouched.
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("any"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        srv.manager
            .instance_counts()
            .get("m2")
            .copied()
            .unwrap_or(0),
        0
    );
}

#[tokio::test]
async fn alias_targets_hot_reloaded() {
    let dir = temp_dir("alias-reload");
    let srv = boot(&dir, "tester: secret-key\n", &alias_plain_yaml("[m1]")).await;
    let client = reqwest::Client::new();
    let config_path = dir.join("config.yaml");
    let last_mtime = Arc::new(Mutex::new(None));

    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("any"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(srv.manager.instance_counts().get("m1").copied(), Some(1));

    // Atomically swap the alias target list and reload.
    let updated = std::fs::read_to_string(&config_path)
        .unwrap()
        .replace("targets: [m1]", "targets: [m2]");
    let tmp = dir.join("config.yaml.tmp");
    std::fs::write(&tmp, updated).unwrap();
    std::fs::rename(&tmp, &config_path).unwrap();
    reload::handle_config_reload(
        &config_path,
        &srv.config,
        &srv.apikeys,
        &srv.manager,
        &srv.stats,
        &srv.peers,
        &srv.mqtt,
        &last_mtime,
    )
    .await;

    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("any"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        srv.manager.instance_counts().get("m2").copied(),
        Some(1),
        "the reloaded alias must route to m2"
    );
}

// ── Speculative decoding: --spec-draft-device injection ─────────────────────

/// A symlink named `llama-stub` pointing at the stub backend binary —
/// the `llama-*` basename is what makes llm-orch treat it as a llama.cpp
/// program (offload enforcement, draft-device injection).
fn llama_stub_link(dir: &Path) -> PathBuf {
    let link = dir.join("llama-stub");
    std::os::unix::fs::symlink(STUB_BIN, &link).unwrap();
    link
}

fn draft_models_yaml(llama_stub: &str, argv_logs: &[&Path]) -> String {
    let mut yaml = String::from(
        "devices:\n  cuda:\n    0:\n      pci: \"{}\"\n      vram_mb: 48000\nmodels:\n",
    );
    // No model may declare its own draft-device flag (`--spec-draft-device
    // none`, `-devd …`, …): config validation rejects those — llm-orch owns
    // device placement for main and draft models alike.
    for (i, (name, extra)) in [
        ("spec", "--spec-draft-model draft.gguf"),
        ("spec-type", "--spec-type draft-mtp"),
        ("ngram", "--spec-type ngram-cache"),
        ("plain", ""),
    ]
    .iter()
    .enumerate()
    {
        yaml.push_str(&format!(
            "  - name: {name}\n    context_length: 4096\n    cmd: \"{llama_stub} --port {{port}} --argv-log {} {}\"\n    max_instances: 1\n    max_concurrent: 1\n    vram: 1000\n    cuda_devices: [[0]]\n",
            argv_logs[i].display(),
            extra
        ));
    }
    yaml
}

fn argv_of(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|l| l.to_string())
        .collect()
}

fn draft_device_value(argv: &[String]) -> Option<String> {
    argv.windows(2)
        .find(|w| w[0] == "--spec-draft-device")
        .map(|w| w[1].clone())
}

#[tokio::test]
async fn spawn_injects_spec_draft_device_for_draft_models() {
    let Some(slot) = real_pci_slot() else { return };
    let dir = temp_dir("draft-device");
    let llama_stub = llama_stub_link(&dir);
    let argv_spec = dir.join("argv-spec.txt");
    let argv_spec_type = dir.join("argv-spec-type.txt");
    let argv_ngram = dir.join("argv-ngram.txt");
    let argv_plain = dir.join("argv-plain.txt");

    // The model YAML is built with a placeholder PCI slot; substitute the
    // real one before boot.
    let yaml = draft_models_yaml(
        &llama_stub.display().to_string(),
        &[&argv_spec, &argv_spec_type, &argv_ngram, &argv_plain],
    )
    .replace("pci: \"{}\"", &format!("pci: \"{slot}\""));

    let srv = boot(&dir, "tester: secret-key\n", &yaml).await;
    let client = reqwest::Client::new();
    for model in ["spec", "spec-type", "ngram", "plain"] {
        let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
            .json(&chat_body(model))
            .send()
            .await
            .unwrap();
        if resp.status() != 200 {
            let body = resp.text().await.unwrap_or_default();
            panic!("{model} must be servable, got body: {body}");
        }
    }

    // `spec`: draft model declared → injected to the instance's device
    // (CUDA0 = post-filter rank of the [[0]] placement).
    let argv = argv_of(&argv_spec);
    assert_eq!(
        draft_device_value(&argv).as_deref(),
        Some("CUDA0"),
        "draft device must be injected for draft models: {argv:?}"
    );

    // `spec-type`: internal-MTP-style enabling (`--spec-type draft-mtp`,
    // no draft-model file flag) must also pin the draft context — its
    // default device is llama.cpp's process-default backend, not the
    // model's.
    let argv = argv_of(&argv_spec_type);
    assert_eq!(
        draft_device_value(&argv).as_deref(),
        Some("CUDA0"),
        "draft device must be injected for --spec-type draft-mtp: {argv:?}"
    );

    // `ngram`: self-speculative types have no draft context.
    let argv = argv_of(&argv_ngram);
    assert_eq!(
        draft_device_value(&argv),
        None,
        "no draft-device flag for ngram speculation: {argv:?}"
    );

    // `plain`: no draft model → nothing injected.
    let argv = argv_of(&argv_plain);
    assert_eq!(
        draft_device_value(&argv),
        None,
        "no draft-device flag without a draft model: {argv:?}"
    );
}

// ── Main-device selector: --device injection ────────────────────────────────

/// Value of the first `--device` occurrence in an argv.
fn device_value(argv: &[String]) -> Option<String> {
    argv.windows(2)
        .find(|w| w[0] == "--device")
        .map(|w| w[1].clone())
}

/// Every GPU-pinned llama.cpp model — Vulkan here — must be spawned with
/// an explicit `--device <Prefix><rank>` selector naming the post-filter
/// visible ranks.  Without it llama.cpp would use every device its build
/// enumerates (all backends) and row-split the instance across them,
/// leaking outside the scheduler's placement.
#[tokio::test]
async fn spawn_pins_vulkan_model_with_device_selector() {
    let slots = real_drm_slots();
    if slots.len() < 2 {
        return;
    }
    let dir = temp_dir("device-pin-vulkan");
    let llama_stub = llama_stub_link(&dir);
    let argv_single = dir.join("argv-vk-single.txt");
    let argv_dual = dir.join("argv-vk-dual.txt");
    let yaml = format!(
        "devices:\n  vulkan:\n    0: \"{}\"\n    1: \"{}\"\nmodels:\n  - name: vk-single\n    context_length: 4096\n    cmd: \"{} --port {{port}} --argv-log {}\"\n    max_instances: 1\n    max_concurrent: 1\n    vram: 1000\n    gpus: 1\n    vulkan_devices: [0, 1]\n  - name: vk-dual\n    context_length: 4096\n    cmd: \"{} --port {{port}} --argv-log {}\"\n    max_instances: 1\n    max_concurrent: 1\n    vram: 1000\n    gpus: 2\n    vulkan_devices: [0, 1]\n",
        slots[0],
        slots[1],
        llama_stub.display(),
        argv_single.display(),
        llama_stub.display(),
        argv_dual.display(),
    );

    let srv = boot(&dir, "tester: secret-key\n", &yaml).await;
    let client = reqwest::Client::new();
    for model in ["vk-single", "vk-dual"] {
        let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
            .json(&chat_body(model))
            .send()
            .await
            .unwrap();
        if resp.status() != 200 {
            let body = resp.text().await.unwrap_or_default();
            panic!("{model} must be servable, got body: {body}");
        }
    }

    // gpus: 1 — the scheduler picks one pool device; whichever it is, the
    // post-filter rank is 0, so the pin must be exactly `Vulkan0`.
    let argv = argv_of(&argv_single);
    assert_eq!(
        device_value(&argv).as_deref(),
        Some("Vulkan0"),
        "single-GPU Vulkan model must be pinned to its visible rank: {argv:?}"
    );

    // gpus: 2 — both pool devices are selected (deterministic pool order);
    // the selector names the post-filter ranks of GGML_VK_VISIBLE_DEVICES=0,1.
    let argv = argv_of(&argv_dual);
    assert_eq!(
        device_value(&argv).as_deref(),
        Some("Vulkan0,Vulkan1"),
        "dual-GPU Vulkan model must name both visible ranks: {argv:?}"
    );
}

/// A non-empty placement renumbering must surface as post-filter ranks in
/// the selector: `vulkan_devices: [[1]]` pins host device 1 via
/// `GGML_VK_VISIBLE_DEVICES=1`, where llama.cpp enumerates it as `Vulkan0`.
#[tokio::test]
async fn spawn_device_selector_uses_post_filter_ranks() {
    let slots = real_drm_slots();
    if slots.len() < 2 {
        return;
    }
    let dir = temp_dir("device-pin-ranks");
    let llama_stub = llama_stub_link(&dir);
    let argv_log = dir.join("argv-ranks.txt");
    let yaml = format!(
        "devices:\n  vulkan:\n    0: \"{}\"\n    1: \"{}\"\nmodels:\n  - name: vk-rank\n    context_length: 4096\n    cmd: \"{} --port {{port}} --argv-log {}\"\n    max_instances: 1\n    max_concurrent: 1\n    vram: 1000\n    gpus: 1\n    vulkan_devices: [[1]]\n",
        slots[0],
        slots[1],
        llama_stub.display(),
        argv_log.display(),
    );

    let srv = boot(&dir, "tester: secret-key\n", &yaml).await;
    let client = reqwest::Client::new();
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("vk-rank"))
        .send()
        .await
        .unwrap();
    if resp.status() != 200 {
        let body = resp.text().await.unwrap_or_default();
        panic!("vk-rank must be servable, got body: {body}");
    }

    let argv = argv_of(&argv_log);
    assert_eq!(
        device_value(&argv).as_deref(),
        Some("Vulkan0"),
        "host device 1 is visible rank 0 after GGML_VK_VISIBLE_DEVICES=1: {argv:?}"
    );
}

/// The CUDA namespace gets the same treatment (`--device CUDA0`); this
/// also guards the pre-existing ROCm/CUDA behavior against regressions.
#[tokio::test]
async fn spawn_pins_cuda_model_with_device_selector() {
    let Some(slot) = real_pci_slot() else { return };
    let dir = temp_dir("device-pin-cuda");
    let llama_stub = llama_stub_link(&dir);
    let argv_log = dir.join("argv-cuda.txt");
    let yaml = format!(
        "devices:\n  cuda:\n    0:\n      pci: \"{slot}\"\n      vram_mb: 48000\nmodels:\n  - name: cuda-m\n    context_length: 4096\n    cmd: \"{} --port {{port}} --argv-log {}\"\n    max_instances: 1\n    max_concurrent: 1\n    vram: 1000\n    gpus: 1\n    cuda_devices: [[0]]\n",
        llama_stub.display(),
        argv_log.display(),
    );

    let srv = boot(&dir, "tester: secret-key\n", &yaml).await;
    let client = reqwest::Client::new();
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("cuda-m"))
        .send()
        .await
        .unwrap();
    if resp.status() != 200 {
        let body = resp.text().await.unwrap_or_default();
        panic!("cuda-m must be servable, got body: {body}");
    }

    let argv = argv_of(&argv_log);
    assert_eq!(
        device_value(&argv).as_deref(),
        Some("CUDA0"),
        "CUDA model must be pinned by device name, not env alone: {argv:?}"
    );
}

// ── stable-diffusion.cpp integration tests ──────────────────────────────────

/// Build a YAML snippet for a single sd-server model.
fn sd_models_yaml(idle_ttl: u64) -> String {
    sd_models_yaml_flags(idle_ttl, "")
}

/// Like [`sd_models_yaml`] but with extra stub-backend flags appended to the
/// model command (e.g. `--fail-status 503`).
fn sd_models_yaml_flags(idle_ttl: u64, flags: &str) -> String {
    format!(
        "models:\n  - name: sd-m\n    context_length: 0\n    cmd: \"{STUB_BIN} --port {{port}} --diffusion-model /models/test.safetensors --backend vulkan{flags}\"\n    idle_ttl: {idle_ttl}\n    max_instances: 1\n"
    )
}

#[tokio::test]
async fn sd_image_generation_sync() {
    let dir = temp_dir("sd-img");
    let srv = boot(&dir, "tester: secret-key\n", &sd_models_yaml(60)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/images/generations", srv.url))
        .json(&serde_json::json!({
            "model": "sd-m",
            "prompt": "a cat",
            "n": 1,
            "size": "512x512",
            "output_format": "png"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["created"].as_i64().is_some());
    assert!(body["data"].is_array());
    assert!(!body["data"].as_array().unwrap().is_empty());
    assert!(body["data"][0]["b64_json"].as_str().unwrap().len() > 10);
}

#[tokio::test]
async fn sd_image_generation_unknown_model_404() {
    let dir = temp_dir("sd-img-404");
    let srv = boot(&dir, "tester: secret-key\n", &sd_models_yaml(60)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/images/generations", srv.url))
        .json(&serde_json::json!({
            "model": "nonexistent-sd",
            "prompt": "a cat"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

/// A diffusion backend that answers with an error must surface as **502 Bad
/// Gateway carrying the backend's own message**.  It used to be a bare 500
/// whose reason only existed in the response body — indistinguishable from an
/// llm-orch bug, and invisible in the journal.
#[tokio::test]
async fn sd_image_generation_backend_failure_is_bad_gateway() {
    let dir = temp_dir("sd-img-fail");
    let srv = boot(
        &dir,
        "tester: secret-key\n",
        &sd_models_yaml_flags(60, " --fail-status 503"),
    )
    .await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/images/generations", srv.url))
        .json(&serde_json::json!({
            "model": "sd-m",
            "prompt": "a cat",
            "size": "512x512"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        502,
        "an upstream fault must be 502, not a generic 500"
    );
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("backend error"),
        "body must attribute the fault to the backend, got: {body}"
    );
    assert!(
        body.contains("503"),
        "body must carry the backend's status code, got: {body}"
    );
    assert!(
        body.contains("injected stub failure"),
        "body must carry the backend's own error message, got: {body}"
    );
}

/// The raw passthrough path (`/sdapi/*`, multipart-capable forwarding) must
/// report upstream faults the same way.
#[tokio::test]
async fn sd_passthrough_backend_failure_is_bad_gateway() {
    let dir = temp_dir("sd-sdapi-fail");
    let listen_port = free_port();
    let models = sd_models_yaml_flags(60, " --fail-status 500");
    let cfg_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  default_diffusion_model: \"sd-m\"\napikeys_file: \"apikeys.txt\"\n{models}",
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, cfg_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/sdapi/v1/txt2img", srv.url))
        .json(&serde_json::json!({ "prompt": "a cat", "steps": 20 }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("injected stub failure"),
        "body must carry the backend's own error message, got: {body}"
    );
}

/// A rejected async video job must report the backend's failure too
/// (submit path: sd-server refuses the job outright).
#[tokio::test]
async fn sd_video_generation_backend_failure_is_bad_gateway() {
    let dir = temp_dir("sd-vid-fail");
    let srv = boot(
        &dir,
        "tester: secret-key\n",
        &sd_models_yaml_flags(60, " --fail-status 503"),
    )
    .await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/video/generations", srv.url))
        .json(&serde_json::json!({
            "model": "sd-m",
            "prompt": "a cat walking",
            "video_frames": 33
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("video job submission failed"),
        "body must say the job submission failed, got: {body}"
    );
}

/// A backend that stops answering HTTP while its process keeps running (a
/// busy sd-server) must not cost the client its job: llm-orch keeps polling
/// until the total timeout instead of failing at the first unreachable poll.
#[tokio::test]
async fn sd_video_generation_polls_through_an_unreachable_but_alive_backend() {
    let dir = temp_dir("sd-vid-deaf");
    let listen_port = free_port();
    // The stub closes its listener 1s after the job is created; the job stays
    // pending, so llm-orch is mid-poll when the backend goes deaf.
    let models = sd_models_yaml_flags(60, " --job-pending-secs 30 --stop-serving-after 1");
    let cfg_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\n  backend_total_timeout_secs: 4\napikeys_file: \"apikeys.txt\"\n{models}",
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, cfg_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    let started = std::time::Instant::now();
    let resp = authed(&client, &format!("{}/v1/video/generations", srv.url))
        .json(&serde_json::json!({
            "model": "sd-m",
            "prompt": "a cat walking",
            "video_frames": 33
        }))
        .send()
        .await
        .unwrap();
    let elapsed = started.elapsed();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    // The listener dies ~1s in, well before the 4s total timeout: the request
    // must be decided by the deadline, not by the first failed poll.
    assert_eq!(
        status, 504,
        "a deaf-but-alive backend must be polled until the deadline, got {status}: {body}"
    );
    assert!(
        body.contains("did not complete within 4s"),
        "expected a timeout answer, got: {body}"
    );
    assert!(
        elapsed.as_secs() >= 3,
        "polling gave up after {elapsed:?}, before the total timeout"
    );
}

/// When the backend process itself exits, the job can never complete, so
/// llm-orch stops polling at once instead of burning the whole total timeout.
#[tokio::test]
async fn sd_video_generation_aborts_when_backend_process_exits() {
    let dir = temp_dir("sd-vid-exit");
    let listen_port = free_port();
    let models = sd_models_yaml_flags(60, " --job-pending-secs 30 --exit-after 1");
    let cfg_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\n  backend_total_timeout_secs: 120\napikeys_file: \"apikeys.txt\"\n{models}",
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, cfg_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    let started = std::time::Instant::now();
    let resp = authed(&client, &format!("{}/v1/video/generations", srv.url))
        .json(&serde_json::json!({
            "model": "sd-m",
            "prompt": "a cat walking",
            "video_frames": 33
        }))
        .send()
        .await
        .unwrap();
    let elapsed = started.elapsed();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert_eq!(
        status, 502,
        "a dead backend must fail as a bad gateway, got {status}: {body}"
    );
    assert!(
        body.contains("can no longer complete"),
        "expected the job to be reported as unrecoverable, got: {body}"
    );
    assert!(
        elapsed.as_secs() < 30,
        "a dead backend must be detected early, waited {elapsed:?}"
    );
}

#[tokio::test]
async fn sd_image_edits_sync() {
    let dir = temp_dir("sd-edit");
    let srv = boot(&dir, "tester: secret-key\n", &sd_models_yaml(60)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/images/edits", srv.url))
        .multipart(
            reqwest::multipart::Form::new()
                .text("model", "sd-m")
                .text("prompt", "make it red")
                .text("n", "1")
                .part(
                    "image[]",
                    reqwest::multipart::Part::bytes(vec![0u8; 64]).file_name("test.png"),
                ),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn sd_video_generation_sync() {
    let dir = temp_dir("sd-vid");
    let srv = boot(&dir, "tester: secret-key\n", &sd_models_yaml(60)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/video/generations", srv.url))
        .json(&serde_json::json!({
            "model": "sd-m",
            "prompt": "a cat walking",
            "video_frames": 33,
            "fps": 16,
            "output_format": "webm"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.contains("video/webm"),
        "expected video/webm content-type, got: {ct}"
    );
}

#[tokio::test]
async fn sd_passthrough_sdapi_txt2img() {
    let dir = temp_dir("sd-sdapi-txt2img");
    // Boot with default_diffusion_model set.
    let listen_port = free_port();
    let models = sd_models_yaml(60);
    let cfg_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\n  default_diffusion_model: \"sd-m\"\napikeys_file: \"apikeys.txt\"\n{models}",
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, cfg_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/sdapi/v1/txt2img", srv.url))
        .json(&serde_json::json!({
            "prompt": "a cat",
            "steps": 20
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["images"].is_array(),
        "sdapi txt2img must return images array"
    );
}

#[tokio::test]
async fn sd_passthrough_sdcpp_vid_gen() {
    let dir = temp_dir("sd-sdcpp-vid");
    let listen_port = free_port();
    let models = sd_models_yaml(60);
    let cfg_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\n  default_diffusion_model: \"sd-m\"\napikeys_file: \"apikeys.txt\"\n{models}",
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, cfg_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    // sdcpp v1 vid_gen returns 202 with async job id
    let resp = authed(&client, &format!("{}/sdcpp/v1/vid_gen", srv.url))
        .json(&serde_json::json!({
            "prompt": "a cat walking",
            "video_frames": 33
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 202);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["id"].as_str().is_some());
    assert_eq!(body["status"], "queued");
}

#[tokio::test]
async fn sd_passthrough_default_diffusion_fallback() {
    let dir = temp_dir("sd-default-fallback");
    let listen_port = free_port();
    let models = sd_models_yaml(60);
    let cfg_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\n  default_diffusion_model: \"sd-m\"\napikeys_file: \"apikeys.txt\"\n{models}",
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, cfg_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    // No model field in body — should use default_diffusion_model
    let resp = authed(&client, &format!("{}/sdapi/v1/txt2img", srv.url))
        .json(&serde_json::json!({"prompt": "default model test"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "must fall back to default_diffusion_model"
    );
}

#[tokio::test]
async fn sd_passthrough_x_llm_model_header() {
    let dir = temp_dir("sd-x-llm-model");
    let listen_port = free_port();
    let models = sd_models_yaml(60);
    let cfg_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\napikeys_file: \"apikeys.txt\"\n{models}",
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, cfg_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    // No default_diffusion_model, but set X-LLM-Model header.
    let resp = client
        .get(format!("{}/sdcpp/v1/capabilities", srv.url))
        .header("Authorization", "Bearer secret-key")
        .header("X-LLM-Model", "sd-m")
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "X-LLM-Model header must route to configured model"
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["supported_modes"].is_array(),
        "must return capabilities"
    );
}

#[tokio::test]
async fn sd_passthrough_missing_model_returns_400() {
    let dir = temp_dir("sd-missing-model");
    let listen_port = free_port();
    let models = sd_models_yaml(60);
    let cfg_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\napikeys_file: \"apikeys.txt\"\n{models}",
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, cfg_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    // No default_diffusion_model and no X-LLM-Model → 400.
    let resp = client
        .post(format!("{}/sdapi/v1/txt2img", srv.url))
        .header("Authorization", "Bearer secret-key")
        .json(&serde_json::json!({"prompt": "test"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "no model target must return 400");
}

#[tokio::test]
async fn sd_endpoints_require_auth() {
    let dir = temp_dir("sd-auth");
    let srv = boot(&dir, "tester: secret-key\n", &sd_models_yaml(60)).await;
    let client = reqwest::Client::new();

    for path in [
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/video/generations",
        "/sdapi/v1/txt2img",
        "/sdcpp/v1/capabilities",
    ] {
        let resp = client
            .post(format!("{}{}", srv.url, path))
            .json(&serde_json::json!({"model": "sd-m", "prompt": "test"}))
            .send()
            .await
            .unwrap();
        // llm-orch only ever denies with 401 (see ApiError::Unauthorized);
        // accepting anything else would hide a route that never reaches auth.
        assert_eq!(
            resp.status(),
            401,
            "endpoint {path} without auth should be denied"
        );
    }
}

/// The native sd-server passthrough catch-alls are default-deny too.  This
/// covers the two routes that the OpenAI-compatible tests above do not:
/// the POST catch-alls (`passthrough_sdcpp`/`passthrough_sdapi`) and,
/// importantly, the GET catch-all (`passthrough_sdcpp_get`) that carries the
/// async job polling (`GET /sdcpp/v1/jobs/{id}`) and job cancellation.
#[tokio::test]
async fn sd_native_passthrough_routes_require_auth() {
    let dir = temp_dir("sd-native-auth");
    let srv = boot(&dir, "tester: secret-key\n", &sd_models_yaml(60)).await;
    let client = reqwest::Client::new();

    // Note there is no GET route under /sdapi/v1, so only /sdcpp has both.
    let routes = [
        ("POST", "/sdcpp/v1/vid_gen"),
        ("POST", "/sdcpp/v1/jobs/job_probe/cancel"),
        ("POST", "/sdapi/v1/txt2img"),
        ("GET", "/sdcpp/v1/jobs/job_probe"),
        ("GET", "/sdcpp/v1/capabilities"),
    ];

    for (method, path) in routes {
        let url = format!("{}{}", srv.url, path);
        let make = || {
            client.request(
                reqwest::Method::from_bytes(method.as_bytes()).unwrap(),
                &url,
            )
        };

        let resp = make().send().await.unwrap();
        assert_eq!(
            resp.status(),
            401,
            "{method} {path}: missing credentials must be denied"
        );

        let resp = make()
            .header("Authorization", "Bearer wrong-key")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            401,
            "{method} {path}: wrong key must be denied"
        );

        let resp = make()
            .header("Authorization", "secret-key")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            401,
            "{method} {path}: key without the Bearer scheme must be denied"
        );

        // A valid key must clear auth; the model target or the backend then
        // decides the final status, so any non-401 proves the point.
        let resp = make().bearer_auth("secret-key").send().await.unwrap();
        assert_ne!(
            resp.status(),
            401,
            "{method} {path}: valid key must not be denied by auth"
        );
    }
}

// ── Remote http(s) models ────────────────────────────────────────────────────

/// Boot the stub backend standalone (not under llm-orch management) on a
/// fresh port and wait for its readiness.
async fn boot_standalone_stub() -> (u16, tokio::process::Child) {
    let port = free_port();
    let mut child = tokio::process::Command::new(STUB_BIN)
        .arg("--port")
        .arg(port.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let probe = reqwest::Client::new();
    for _ in 0..300 {
        if probe
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            return (port, child);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    child.kill().await.unwrap();
    panic!("standalone stub never became ready");
}

/// Bring up the stub on an already-chosen port (recovery tests).
async fn start_standalone_stub(port: u16) -> tokio::process::Child {
    let mut child = tokio::process::Command::new(STUB_BIN)
        .arg("--port")
        .arg(port.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let probe = reqwest::Client::new();
    for _ in 0..300 {
        if probe
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            return child;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    child.kill().await.unwrap();
    panic!("standalone stub on port {port} never became ready");
}

fn remote_model_yaml(stub_port: u16, key_file: &str) -> String {
    format!(
        "key_files:\n  deploy_key: \"{key_file}\"\nmodels:\n  - name: r\n    context_length: 4096\n    cmd: \"http://127.0.0.1:{stub_port}\"\n    remote_auth: \"Bearer {{deploy_key}}\"\n    max_instances: 1\n"
    )
}

fn remote_model_yaml_queue0(stub_port: u16, key_file: &str) -> String {
    remote_model_yaml(stub_port, key_file) + "    queue_depth: 0\n"
}

#[tokio::test]
async fn remote_model_forwards_and_injects_resolved_auth() {
    let dir = temp_dir("remote-auth");
    let (port, mut stub) = boot_standalone_stub().await;
    let key_path = dir.join("deploy.key");
    std::fs::write(&key_path, "remote-secret-token\n").unwrap();

    let listen_port = free_port();
    let config_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\napikeys_file: \"apikeys.txt\"\n{}",
        remote_model_yaml(port, &key_path.display().to_string())
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, config_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "r",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "remote chat must succeed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "tok-0 tok-1 tok-2 "
    );

    // The stub must have received the key-file-resolved Authorization
    // header — proving `key_files` + `remote_auth` substitution end to end.
    let echo: serde_json::Value = client
        .get(format!("http://127.0.0.1:{port}/test/last-auth"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        echo["authorization"],
        serde_json::json!("Bearer remote-secret-token"),
        "remote backend must see the resolved Authorization header"
    );

    // A key file content change applies on refresh without any instance
    // replacement (resolution happens at request time).
    std::fs::write(&key_path, "rotated-token\n").unwrap();
    let cfg = srv.config.read().await;
    srv.manager.refresh_key_files(&cfg);
    drop(cfg);
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "r",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let echo: serde_json::Value = client
        .get(format!("http://127.0.0.1:{port}/test/last-auth"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        echo["authorization"],
        serde_json::json!("Bearer rotated-token"),
        "rotated key file content must apply without instance replacement"
    );

    stub.kill().await.unwrap();
}

#[tokio::test]
async fn remote_model_down_fails_fast_then_recovers() {
    let dir = temp_dir("remote-down");
    let key_path = dir.join("deploy.key");
    std::fs::write(&key_path, "tok\n").unwrap();

    // Nothing is listening on `dead_port` yet.
    let dead_port = free_port();
    let listen_port = free_port();
    let config_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\napikeys_file: \"apikeys.txt\"\n{}",
        remote_model_yaml_queue0(dead_port, &key_path.display().to_string())
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, config_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    // queue_depth: 0 → a failed probe surfaces as 429 instead of
    // queueing.  (The zero-depth path also triggers the spawn attempt
    // and waits for it before rejecting.)
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "r",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        429,
        "unreachable remote endpoint must fail fast (body: {})",
        resp.text().await.unwrap_or_default()
    );

    // The model must NOT be blocked by the failed probe: bring the
    // endpoint up.  The next zero-depth request triggers a fresh probe
    // (which now succeeds) and is still rejected per the zero-depth
    // contract; the following request gets the warm instance.
    let mut stub = start_standalone_stub(dead_port).await;
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "r",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        429,
        "second zero-depth request warms the instance"
    );
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "r",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "recovered remote endpoint must serve: {}",
        resp.text().await.unwrap_or_default()
    );

    stub.kill().await.unwrap();
}

#[tokio::test]
async fn remote_model_without_remote_auth_sends_no_header() {
    let dir = temp_dir("remote-noauth");
    let key_path = dir.join("deploy.key");
    std::fs::write(&key_path, "sekret\n").unwrap();

    let (port, mut stub) = boot_standalone_stub().await;
    let listen_port = free_port();
    // Remote model with a configured key file but *no* `remote_auth`:
    // the stub must see no Authorization header at all.
    let config_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\napikeys_file: \"apikeys.txt\"\nkey_files:\n  deploy_key: \"{key}\"\nmodels:\n  - name: r\n    context_length: 4096\n    cmd: \"http://127.0.0.1:{port}\"\n    max_instances: 1\n",
        key = key_path.display()
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, config_yaml).unwrap();
    let srv = boot_from_config_path(&dir, &config_path, "tester: secret-key\n").await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "r",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "remote chat without remote_auth must succeed: {}",
        resp.text().await.unwrap_or_default()
    );

    let echo: serde_json::Value = client
        .get(format!("http://127.0.0.1:{port}/test/last-auth"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        echo["authorization"],
        serde_json::Value::Null,
        "no remote_auth configured — the backend must not see an Authorization header"
    );

    stub.kill().await.unwrap();
}

// ── Token / model statistics ─────────────────────────────────────────────────

/// Boot a server with the `stats:` section enabled, writing to a file in
/// the test directory.
async fn boot_with_stats(dir: &Path, models_yaml: &str) -> (TestServer, PathBuf) {
    let listen_port = free_port();
    let stats_file = dir.join("stats.json");
    let config_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\n  drain_idle_timeout_secs: 1\napikeys_file: \"apikeys.txt\"\nstats:\n  file: \"{}\"\n{models_yaml}",
        stats_file.display()
    );
    let config_path = dir.join("config.yaml");
    std::fs::write(&config_path, config_yaml).unwrap();
    let srv = boot_from_config_path(dir, &config_path, "tester: secret-key\n").await;
    (srv, stats_file)
}

#[tokio::test]
async fn stats_counters_json_file_and_endpoint() {
    let dir = temp_dir("stats");
    let (srv, stats_file) = boot_with_stats(&dir, &models_yaml("", 60)).await;
    let client = reqwest::Client::new();

    // Non-streaming chat completion (stub: prompt=5, completion=3).
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("m"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // The JSON file is written on the first served request.
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&stats_file).unwrap()).unwrap();
    assert_eq!(doc["version"], 1);
    assert!(doc["started_at"].is_number());
    assert!(doc["updated_at"].is_number());
    let total = &doc["models"]["m"]["total"];
    assert_eq!(total["requests"], 1);
    assert_eq!(total["prompt_tokens"], 5);
    assert_eq!(total["completion_tokens"], 3);
    assert_eq!(total["ttft_count"], 0, "non-streaming: no TTFT");
    assert!(total["duration_sum_ms"].is_number());
    assert_eq!(
        &doc["models"]["m"]["boot"], total,
        "single boot: boot counters equal total counters"
    );

    // Streaming chat completion → TTFT is measured.
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert!(body.contains("[DONE]"), "stream must complete: {body}");

    // /v1/stats (auth-gated) reflects both requests.
    let resp = client
        .get(format!("{}/v1/stats", srv.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["enabled"], true);
    assert_eq!(v["version"], 1);
    let m = &v["models"]["m"]["total"];
    assert_eq!(m["requests"], 2);
    assert_eq!(m["prompt_tokens"], 10);
    assert_eq!(m["completion_tokens"], 6);
    assert_eq!(m["ttft_count"], 1, "one streaming request measured");
    assert!(
        m["ttft_min_ms"].is_u64() && m["ttft_max_ms"].is_u64(),
        "ttft min/max must be present after a streaming request"
    );
    assert_eq!(v["models"]["m"]["boot"], v["models"]["m"]["total"]);

    // The JSON file agrees with the endpoint.
    let doc2: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&stats_file).unwrap()).unwrap();
    assert_eq!(doc2["models"]["m"]["total"]["requests"], 2);
    assert_eq!(doc2["models"]["m"]["total"]["ttft_count"], 1);

    // No auth → 401.
    let resp = client
        .get(format!("{}/v1/stats", srv.url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn stats_disabled_reports_enabled_false() {
    let dir = temp_dir("stats-off");
    let srv = boot(&dir, "tester: secret-key\n", &models_yaml("", 60)).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/stats", srv.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["enabled"], false, "no stats: section → enabled=false");
    assert_eq!(v["models"].as_object().map(|o| o.len()), Some(0));
}

#[tokio::test]
async fn stats_hot_reload_retargers_file() {
    let dir = temp_dir("stats-retarget");
    let (srv, stats_file) = boot_with_stats(&dir, &models_yaml("", 60)).await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("m"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Point the stats file at a new path and reload.
    let new_file = dir.join("stats-new.json");
    let config_path = dir.join("config.yaml");
    let updated = std::fs::read_to_string(&config_path).unwrap().replace(
        format!("\"{}\"", stats_file.display()).as_str(),
        format!("\"{}\"", new_file.display()).as_str(),
    );
    std::fs::write(&config_path, updated).unwrap();
    let last_mtime = Arc::new(Mutex::new(None));
    reload::handle_config_reload(
        &config_path,
        &srv.config,
        &srv.apikeys,
        &srv.manager,
        &srv.stats,
        &srv.peers,
        &srv.mqtt,
        &last_mtime,
    )
    .await;

    // The new file is published immediately with the existing counters.
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&new_file).unwrap()).unwrap();
    assert_eq!(doc["models"]["m"]["total"]["requests"], 1);

    // New requests land in the new file.
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("m"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&new_file).unwrap()).unwrap();
    assert_eq!(doc["models"]["m"]["total"]["requests"], 2);
}

#[tokio::test]
async fn stats_config_validation() {
    let dir = temp_dir("stats-validation");
    std::fs::write(dir.join("apikeys.txt"), "tester: secret-key\n").unwrap();

    // Missing parent directory → rejected at config load.
    let bad = format!(
        "server:\n  listen: \"127.0.0.1:{}\"\n  port_range: \"ephemeral\"\napikeys_file: \"apikeys.txt\"\nstats:\n  file: \"no-such-dir/stats.json\"\n",
        free_port()
    );
    let bad_path = dir.join("bad-config.yaml");
    std::fs::write(&bad_path, bad).unwrap();
    assert!(
        config::Config::load(&bad_path).is_err(),
        "stats.file with a missing parent directory must be rejected"
    );
}

// ── Cluster peering (docs/006) ───────────────────────────────────────────────

/// Boot a server on a *fixed* listen port — needed so a master's `peers:`
/// config can point at a leaf before the master boots.
async fn boot_on_port(
    dir: &Path,
    listen_port: u16,
    apikeys_contents: &str,
    yaml: &str,
) -> TestServer {
    let config_path = write_config(dir, listen_port, yaml);
    boot_from_config_path(dir, &config_path, apikeys_contents).await
}

/// A standalone stub backend process on a fixed port (the leaf's remote
/// model target).  Killed when the guard drops.
struct StubProcess(std::process::Child);

impl Drop for StubProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn spawn_stub_on(port: u16) -> StubProcess {
    let child = std::process::Command::new(STUB_BIN)
        .arg("--port")
        .arg(port.to_string())
        .spawn()
        .expect("stub backend spawns");
    let guard = StubProcess(child);
    // Wait for readiness (health endpoint).
    let client = reqwest::Client::new();
    for _ in 0..100 {
        if client
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .is_ok()
        {
            return guard;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("stub backend on port {port} never became healthy");
}

/// Boot the two-node peering topology used by the tests below:
/// master `cheyenne` (local stub-backed `local-dense` model + peer
/// `cortana`) and leaf `cortana` (remote `qwen-flash` model on a
/// standalone stub).  Both have stats enabled.
struct PeeringTopology {
    master: TestServer,
    #[allow(dead_code)]
    leaf: TestServer,
    #[allow(dead_code)]
    _stub: StubProcess,
}

async fn boot_peering() -> PeeringTopology {
    let leaf_port = free_port();
    let stub_port = free_port();
    let _stub = spawn_stub_on(stub_port).await;

    // Leaf: node `cortana`, one remote model served by the stub.
    let leaf_dir = temp_dir("peering-leaf");
    let leaf_yaml = format!(
        "node_name: cortana\n\
stats:\n  file: \"stats.json\"\n\
models:\n  - name: qwen-flash\n    context_length: 8192\n    cmd: \"http://127.0.0.1:{stub_port}\"\n    idle_ttl: 3600\n"
    );
    let leaf = boot_on_port(&leaf_dir, leaf_port, "tester: secret-key\n", &leaf_yaml).await;

    // Master: node `cheyenne`, local model + the leaf as its only peer.
    let master_dir = temp_dir("peering-master");
    let master_yaml = format!(
        "node_name: cheyenne\n\
peers:\n  - name: cortana\n    url: \"http://127.0.0.1:{leaf_port}\"\n    remote_auth: \"Bearer secret-key\"\n    catalog_ttl_secs: 1\n\
stats:\n  file: \"stats.json\"\n\
models:\n  - name: local-dense\n    context_length: 4096\n    cmd: \"{STUB_BIN} --port {{port}}\"\n    idle_ttl: 3600\n"
    );
    let master = boot(&master_dir, "tester: secret-key\n", &master_yaml).await;
    PeeringTopology {
        master,
        leaf,
        _stub,
    }
}

/// The union model listing: bare local, node-qualified local, bare peer
/// alias, and peer-qualified names — and never a client provider prefix.
#[tokio::test]
async fn peering_models_listing_shows_union() {
    let topo = boot_peering().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/models", topo.master.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    for expected in [
        "local-dense",
        "cheyenne/local-dense",
        "qwen-flash",
        "cortana/qwen-flash",
    ] {
        assert!(
            ids.contains(&expected),
            "model listing must contain '{expected}', got {ids:?}"
        );
    }
    assert!(!ids.iter().any(|id| id.starts_with("llm/")));
}

/// End-to-end chat through all naming forms: pinned peer model, bare
/// peer alias, client provider prefix, and node-qualified local model.
/// The stub echoes the model name it received, so the response's `model`
/// proves the full rewrite chain (master → leaf-side name → back).
#[tokio::test]
async fn peering_chat_routes_all_naming_forms() {
    let topo = boot_peering().await;
    let client = reqwest::Client::new();

    for model in [
        "cortana/qwen-flash",
        "qwen-flash",
        "llm/cortana/qwen-flash",
        "llm/qwen-flash",
        "local-dense",
        "cheyenne/local-dense",
    ] {
        let resp = authed(&client, &format!("{}/v1/chat/completions", topo.master.url))
            .json(&chat_body(model))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "request for {model} failed");
        let body: serde_json::Value = resp.json().await.unwrap();
        let returned = body["model"].as_str().unwrap_or_default();
        // The response carries the *resolved* model name (existing alias
        // semantics): bare requests resolve to the qualified peer model.
        match model {
            "qwen-flash" | "llm/qwen-flash" => assert_eq!(returned, "cortana/qwen-flash"),
            "local-dense" | "cheyenne/local-dense" => assert_eq!(returned, "local-dense"),
            // The provider prefix is stripped during resolution.
            "llm/cortana/qwen-flash" => assert_eq!(returned, "cortana/qwen-flash"),
            _ => assert_eq!(returned, model, "model name must round-trip"),
        }
        assert!(
            !body["choices"][0]["message"]["content"].is_null(),
            "stub must produce content for {model}"
        );
    }
}

/// The request-ID mapping: a forwarded request is answerable through the
/// master's `/v1/requests/{id}` (proxied from the leaf).
#[tokio::test]
async fn peering_request_status_proxies_leaf() {
    let topo = boot_peering().await;
    let client = reqwest::Client::new();

    let resp = authed(&client, &format!("{}/v1/chat/completions", topo.master.url))
        .json(&chat_body("cortana/qwen-flash"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let request_id = resp
        .headers()
        .get("x-llm-orch-request-id")
        .and_then(|v| v.to_str().ok())
        .expect("master sets x-llm-orch-request-id")
        .to_string();

    let status = client
        .get(format!("{}/v1/requests/{request_id}", topo.master.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(status.status(), 200, "leaf status must be proxied");
    let body: serde_json::Value = status.json().await.unwrap();
    assert_eq!(body["request_id"].as_str(), Some(request_id.as_str()));
}

/// Admin actions on a peer model are forwarded to the owning leaf.
#[tokio::test]
async fn peering_admin_load_forwards_to_leaf() {
    let topo = boot_peering().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/admin/load", topo.master.url))
        .bearer_auth("secret-key")
        .json(&serde_json::json!({"model": "cortana/qwen-flash"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["status"].as_str(),
        Some("ok"),
        "leaf must accept the load"
    );

    // The leaf actually provisioned its remote model (health-probed).
    assert_eq!(
        topo.leaf
            .manager
            .instance_counts()
            .get("qwen-flash")
            .copied(),
        Some(1)
    );
}

/// The stats union: node-qualified entries carry each node's numbers,
/// bare names sum over every node serving the model.  The master never
/// counts forwarded requests into its own stats.
#[tokio::test]
async fn peering_stats_union_sums_and_qualifies() {
    let topo = boot_peering().await;
    let client = reqwest::Client::new();

    // One local + one forwarded request.
    for model in ["local-dense", "cortana/qwen-flash"] {
        let resp = authed(&client, &format!("{}/v1/chat/completions", topo.master.url))
            .json(&chat_body(model))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    let resp = client
        .get(format!("{}/v1/stats", topo.master.url))
        .bearer_auth("secret-key")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let models = &body["models"];

    // Qualified leaf entry: the leaf's numbers verbatim.
    assert_eq!(models["cortana/qwen-flash"]["total"]["requests"], 1);
    // Bare peer name: the union sum.
    assert_eq!(models["qwen-flash"]["total"]["requests"], 1);
    // Local: bare + node-qualified copies, NOT counting the forward.
    assert_eq!(models["local-dense"]["total"]["requests"], 1);
    assert_eq!(models["cheyenne/local-dense"]["total"]["requests"], 1);
    // The master did not double-count the peer model locally.
    let obj = models.as_object().unwrap();
    assert!(
        !obj.contains_key("cheyenne/cortana/qwen-flash"),
        "no double qualification: {obj:?}"
    );
}

/// An unreachable peer with no catalog yet: requests fail fast with a
/// model-not-found instead of hanging (stale beats empty, empty 404s).
#[tokio::test]
async fn peering_unreachable_peer_fails_fast() {
    let dead_port = free_port();
    let dir = temp_dir("peering-dead");
    let yaml = format!(
        "node_name: cheyenne\n\
peers:\n  - name: cortana\n    url: \"http://127.0.0.1:{dead_port}\"\n    remote_auth: \"Bearer secret-key\"\n    catalog_ttl_secs: 3600\n\
models:\n  - name: local-dense\n    context_length: 4096\n    cmd: \"{STUB_BIN} --port {{port}}\"\n    idle_ttl: 3600\n"
    );
    let srv = boot(&dir, "tester: secret-key\n", &yaml).await;
    let client = reqwest::Client::new();

    let t0 = std::time::Instant::now();
    let resp = authed(&client, &format!("{}/v1/chat/completions", srv.url))
        .json(&chat_body("cortana/qwen-flash"))
        .send()
        .await
        .unwrap();
    let elapsed = t0.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(20),
        "unreachable peer must fail fast (took {elapsed:?})"
    );
    assert_eq!(resp.status(), 404);
}
