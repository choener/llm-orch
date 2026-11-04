// ── MQTT publisher integration tests ───────────────────────────────────────
//
// Boots the publisher against the stub MQTT broker
// (`llm-orch-stub-mqtt`) — no external broker required.  Covers the wire
// contract: birth status, the doc topics on connect, change-only
// republishing, rate computation, and the graceful "offline".  No HTTP
// server is involved; the publisher is composed directly, mirroring the
// main.rs wiring.

use llm_orch::{config, gpu, mqtt, peers, stats};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

const STUB_MQTT_BIN: &str = env!("CARGO_BIN_EXE_llm-orch-stub-mqtt");

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("llm-orch-mqtt-it-{}-{}", tag, uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Stub broker process + its capture API; killed on drop.
struct StubBroker {
    child: std::process::Child,
    http: String,
}

impl StubBroker {
    fn spawn(mqtt_port: u16, http_port: u16) -> Self {
        let child = std::process::Command::new(STUB_MQTT_BIN)
            .args([
                "--mqtt-port",
                &mqtt_port.to_string(),
                "--http-port",
                &http_port.to_string(),
            ])
            .spawn()
            .expect("spawn stub broker");
        Self {
            child,
            http: format!("http://127.0.0.1:{http_port}"),
        }
    }

    async fn messages(&self, topic: &str) -> Vec<serde_json::Value> {
        let url = format!("{}/messages?topic={}", self.http, topic);
        let resp = reqwest::get(&url).await.unwrap();
        assert!(resp.status().is_success(), "GET {url} failed");
        resp.json().await.unwrap()
    }

    /// Poll until `predicate(messages)` holds or time runs out.
    async fn wait_for<F>(&self, topic: &str, predicate: F) -> Vec<serde_json::Value>
    where
        F: Fn(&[serde_json::Value]) -> bool,
    {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let msgs = self.messages(topic).await;
            if predicate(&msgs) {
                return msgs;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "timeout waiting on topic {topic}; last seen: {msgs:?}"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn wait_ready(&self) {
        let client = reqwest::Client::new();
        for _ in 0..100 {
            if client
                .get(format!("{}/messages", self.http))
                .send()
                .await
                .is_ok()
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("stub broker HTTP endpoint did not start");
    }
}

impl Drop for StubBroker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Write a config enabling stats + mqtt against `broker_port`.
fn write_config(dir: &Path, broker_port: u16) -> PathBuf {
    let path = dir.join("config.yaml");
    std::fs::write(
        &path,
        format!(
            "server:\n  listen: \"127.0.0.1:0\"\nnode_name: testnode\nstats:\n  file: \"{}\"\nmqtt:\n  broker: \"127.0.0.1:{broker_port}\"\n  prefix: llmtest\n  interval: 60\n",
            dir.join("stats.json").display(),
        ),
    )
    .unwrap();
    path
}

#[tokio::test]
async fn mqtt_publishes_docs_change_only_with_rates() {
    let stub_port = free_port();
    let http_port = free_port();
    let stub = StubBroker::spawn(stub_port, http_port);
    stub.wait_ready().await;

    let dir = temp_dir("basic");
    let config_path = write_config(&dir, stub_port);
    let cfg = config::Config::load(&config_path).unwrap();

    // Compose the publisher directly (mirror of main.rs wiring).
    let stats_store = stats::from_config(&cfg.stats);
    let (gpu_reader, _gpu_task) = gpu::GpuReader::start(Duration::from_secs(3600));
    let ctl = mqtt::MqttController::new(mqtt::MqttDeps {
        stats: Arc::clone(&stats_store),
        gpu: gpu_reader.snapshot_arc(),
        peers: Arc::new(peers::PeerRegistry::new(reqwest::Client::new())),
        config: Arc::new(tokio::sync::RwLock::new(cfg.clone())),
    });
    ctl.apply(&cfg).await;

    // ── birth status + initial docs on connect ────────────────────────
    stub.wait_for("llmtest/testnode/status", |m| {
        m.iter().any(|x| x["payload"] == "online")
    })
    .await;
    for topic in [
        "llmtest/testnode/stats",
        "llmtest/testnode/models",
        "llmtest/testnode/gpus",
    ] {
        stub.wait_for(topic, |m| !m.is_empty()).await;
    }

    let models = stub.messages("llmtest/testnode/models").await;
    let doc: serde_json::Value =
        serde_json::from_str(models[0]["payload"].as_str().unwrap()).unwrap();
    assert_eq!(doc["node"], "testnode");
    assert_eq!(doc["version"], 1);
    // Retained flag honored (config default: retain true).
    assert_eq!(models[0]["retain"], true);
    // QoS 1 (config default).
    assert_eq!(models[0]["qos"], 1);

    // ── change-only: ticks without changes must not republish ─────────
    let count_before = stub.messages("llmtest/testnode/models").await.len();
    ctl.tick_now();
    tokio::time::sleep(Duration::from_millis(400)).await;
    ctl.tick_now();
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(
        stub.messages("llmtest/testnode/models").await.len(),
        count_before,
        "unchanged docs must not be republished"
    );

    // ── recorded request → new stats doc with positive rates ──────────
    stats_store.record("m", Some(10), None, Some(5), 100, None);
    ctl.tick_now();
    let msgs = stub
        .wait_for("llmtest/testnode/stats", |m| {
            m.iter().any(|x| {
                serde_json::from_str::<serde_json::Value>(x["payload"].as_str().unwrap())
                    .map(|v| v.get("rates").is_some())
                    .unwrap_or(false)
            })
        })
        .await;
    let last = msgs.last().unwrap();
    let stats_doc: serde_json::Value =
        serde_json::from_str(last["payload"].as_str().unwrap()).unwrap();
    assert_eq!(stats_doc["enabled"], true);
    assert_eq!(stats_doc["models"]["m"]["boot"]["requests"], 1);
    assert!(
        stats_doc["rates"]["models"]["m"]["requests_per_min"]
            .as_f64()
            .unwrap()
            > 0.0,
        "rates must be positive after a recorded request: {stats_doc}"
    );

    // A window with no activity drops the `rates` section (one
    // republish), after which the payload is stable again.
    ctl.tick_now();
    stub.wait_for("llmtest/testnode/stats", |m| {
        m.last().is_some_and(|newest| {
            newest["payload"]
                .as_str()
                .and_then(|p| serde_json::from_str::<serde_json::Value>(p).ok())
                .is_some_and(|v| v.get("rates").is_none())
        })
    })
    .await;
    let stats_count = stub.messages("llmtest/testnode/stats").await.len();
    ctl.tick_now();
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(
        stub.messages("llmtest/testnode/stats").await.len(),
        stats_count,
        "stats doc must not be republished without changes"
    );

    // ── graceful shutdown publishes retained "offline" ────────────────
    ctl.shutdown().await;
    stub.wait_for("llmtest/testnode/status", |m| {
        m.iter().any(|x| x["payload"] == "offline")
    })
    .await;
}

#[tokio::test]
async fn mqtt_disabled_without_config_section() {
    // No mqtt: section → apply is a no-op even when called; shutdown
    // completes instantly; the broker sees no connection at all.
    let stub_port = free_port();
    let http_port = free_port();
    let stub = StubBroker::spawn(stub_port, http_port);
    stub.wait_ready().await;

    let dir = temp_dir("disabled");
    let path = dir.join("config.yaml");
    std::fs::write(
        &path,
        "server:\n  listen: \"127.0.0.1:0\"\nnode_name: testnode\n",
    )
    .unwrap();
    let cfg = config::Config::load(&path).unwrap();

    let (gpu_reader, _gpu_task) = gpu::GpuReader::start(Duration::from_secs(3600));
    let ctl = mqtt::MqttController::new(mqtt::MqttDeps {
        stats: stats::from_config(&cfg.stats),
        gpu: gpu_reader.snapshot_arc(),
        peers: Arc::new(peers::PeerRegistry::new(reqwest::Client::new())),
        config: Arc::new(tokio::sync::RwLock::new(cfg.clone())),
    });
    ctl.apply(&cfg).await;
    ctl.tick_now();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(stub.messages("llmtest/testnode/status").await.is_empty());
    ctl.shutdown().await;
}
