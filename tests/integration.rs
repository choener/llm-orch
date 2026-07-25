// ── End-to-end integration tests ─────────────────────────────────────────────
//
// Boots a real orchestrator (config, auth, scheduler, HTTP server) against
// the stub llama.cpp backend binary — no GPU or model required.

use llm_orch::{apikeys, config, debug_log, gpu, reload, scheduler, server};
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
    let config_yaml = format!(
        "server:\n  listen: \"127.0.0.1:{listen_port}\"\n  port_range: \"ephemeral\"\napikeys_file: \"apikeys.txt\"\n{models_yaml}"
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
    _shutdown: oneshot::Sender<()>,
    _dir: PathBuf,
}

/// Boot a full composed server (mirror of main.rs wiring, minus signals
/// and file watchers) on an ephemeral port.
async fn boot(dir: &Path, apikeys_contents: &str, models_yaml: &str) -> TestServer {
    let listen_port = free_port();
    let config_path = write_config(dir, listen_port, models_yaml);
    std::fs::write(dir.join("apikeys.txt"), apikeys_contents).unwrap();

    let cfg = config::Config::load(&config_path).unwrap();
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
    let state = server::AppState {
        config: Arc::clone(&shared_cfg),
        apikeys: Arc::clone(&shared_keys),
        manager: Arc::clone(&manager),
        client: manager.client().clone(),
        gpu: gpu_snapshot,
        debug_loggers: Arc::new(debug_log::DebugLoggers::new()),
    };
    let (tx, rx) = oneshot::channel();
    let server_task = tokio::spawn(server::serve(state, rx));
    // The server binds immediately on a fresh port; give it a moment.
    tokio::time::sleep(Duration::from_millis(100)).await;
    if server_task.is_finished() {
        panic!("server task exited at startup: {:?}", server_task.await);
    }

    TestServer {
        url: format!("http://127.0.0.1:{listen_port}"),
        manager,
        config: shared_cfg,
        apikeys: shared_keys,
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
    assert!(body.contains("[DONE]"), "expected stream terminator: {body}");

    // The stub backend was spawned on demand and is registered.
    assert_eq!(srv.manager.instance_counts().get("m").copied(), Some(1));

    // The completion was recorded (proves the stream ran to completion).
    let completions = srv.manager.recent_completions_snapshot();
    assert_eq!(completions.get("m").map(|c| c.len()), Some(1));

    // Idle TTL (1 s) evicts the instance — which transitively proves the
    // in-flight slot was released: a leaked slot would keep the instance
    // busy and block the IfIdle removal.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    srv.manager.evaluate_autoscale().await;
    assert_eq!(
        srv.manager.instance_counts().get("m").copied(),
        Some(0),
        "idle instance must be despawned after TTL"
    );
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
        body.pointer("/choices/0/message/content").and_then(|v| v.as_str()),
        Some("tok-0 tok-1 tok-2 ")
    );
    assert_eq!(
        body.pointer("/usage/completion_tokens").and_then(|v| v.as_u64()),
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
    reload::handle_config_reload(&config_path, &srv.config, &srv.apikeys, &srv.manager, &last_mtime)
        .await;
    {
        let cfg = srv.config.read().await;
        assert_eq!(cfg.models[0].max_concurrent, 4, "bad config must be rejected");
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
    reload::handle_config_reload(&config_path, &srv.config, &srv.apikeys, &srv.manager, &last_mtime)
        .await;
    let cfg = srv.config.read().await;
    assert_eq!(cfg.models[0].max_concurrent, 8, "valid reload must apply");
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
    let yaml = std::fs::read_to_string(&config_path)
        .unwrap()
        .replace(
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
            panic!("orchestrator must exit within 20s of SIGTERM; log tail:\n{}", tail.join("\n"));
        })
        .unwrap();
    assert!(status.success(), "expected clean exit, got {status}");

    req_task.abort();
}
