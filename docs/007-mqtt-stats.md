# MQTT stats publishing

Status: **implemented**.

## Problem statement

Polling `GET /v1/stats` works for a single node with a reachable HTTP
endpoint, but a Home Assistant / MQTT-based home-automation stack prefers
push: retained topics that survive subscriber restarts, per-node topic
trees, and rate sensors (tokens/min) that a poller must derive itself.
An external connector script was rejected: it needs a second API key,
polls on a fixed cadence regardless of activity, guesses node names, and
computes rates by subtracting successive counters — which breaks on every
restart (the subtraction crosses a reset boundary).

## Design decisions

- **llm-orch publishes MQTT itself.**  It already holds both datasets
  (stats store, GPU metrics), the node name (`node_name`), and the peers'
  cached stats documents.  One background task replaces the connector
  script, its API key, and its naming guesswork; config hot-reload means
  new models/GPUs push their own discovery records the moment they
  appear.  A dead broker never affects request serving.
- **One shared assembly path.**  `stats::build_response` produces the
  `GET /v1/stats` document; the MQTT publisher consumes the same
  function, so the wire payload can never drift from the endpoint.
- **JSON documents, not flat topics.**  Four topics carry complete JSON
  documents (schema `version: 1`).  Home Assistant maps fields via
  `json_attributes`; HA MQTT Discovery (per-entity config topics) is
  deferred to a later enhancement and can be added without changing
  these documents.
- **Change-only publishing.**  Every tick (default **60 s**) rebuilds all
  docs, serializes each canonically (object keys sorted recursively, so
  HashMap iteration order can't fake a change) and publishes a topic only
  when its bytes differ from the last publish.  Every successful CONNACK
  — startup *and* broker reconnect — force-republishes all docs; retained
  messages + republish-on-connect replace any periodic heartbeat.
- **Rate counters without the reset problem.**  The publisher keeps the
  previous per-model `boot` counters and divides the delta by the elapsed
  window.  llm-orch is both counter source and publisher, so no consumer
  ever subtracts across a restart: the first window after startup
  compares against the boot-time snapshot, a regressed counter is
  clamped and skipped (with a `warn!`), and models missing from the
  previous snapshot are treated as zero-counted (the stats map only
  holds models with activity).  A window with no activity publishes no
  `rates` section at all — a monotonic `window_secs` would otherwise
  churn the payload every tick.
- **Credentials follow the peer-key convention.**  `mqtt.key` is a
  template with `{name}` placeholders resolved from the existing
  `key_files:` map (rotated key files re-resolve on hot-reload).
  `node_name` is **required** when `mqtt:` is set — validated at load,
  so `--check-config` catches topic-naming ambiguity up front.
- **No persistent broker session.**  Clean session + uuid client id:
  retained messages and the force-republish on connect cover reconnects,
  and a rotating client id would leave junk sessions on the broker
  anyway.
- **Graceful `offline`.**  A Last Will (retained `offline`) covers
  crashes; a clean shutdown explicitly publishes `offline` (after a
  final flush) so consumers don't wait for the broker's will delay.

## Topic tree and schemas

All under `{mqtt.prefix}` (default `llmorch`), per node:

| Topic | Payload |
| --- | --- |
| `{prefix}/{node}/status` | `online` / `offline` (retained string) |
| `{prefix}/{node}/stats` | the `GET /v1/stats` document (`enabled`, `version`, `started_at`, `updated_at`, `models` union incl. node-qualified entries, `gpus` live metrics) plus the optional `rates` block |
| `{prefix}/{node}/models` | `{version, node, models: [{name, context_length}], aliases: [{name, targets}]}` from the effective config (peer models appear as `<peer>/<model>`) |
| `{prefix}/{node}/gpus` | `{version, node, gpus: [{index, pci_slot, vram_vendor, vram_total_bytes}]}` — static inventory only, so it changes only with hardware/config |
| `{prefix}/{peer}/stats` | **gateway re-export**: each configured peer's cached `StatsResponse` document, republished under the peer's own name (no extra fetches — the same 10 s cache the stats endpoint uses) |

`rates` block (present only when the last window had activity):

```json
{
  "rates": {
    "window_secs": 60,
    "models": {
      "qwen3-32b": {
        "requests_per_min": 0.42,
        "prompt_tokens_per_min": 311.0,
        "completion_tokens_per_min": 88.5,
        "tokens_per_min": 399.5
      }
    }
  }
}
```

## Configuration

```yaml
node_name: cheyenne           # required by mqtt:

mqtt:
  broker: "mqtt.home.lan:1883"   # host:port | mqtt://… | mqtts://…
  prefix: llmorch                # MQTT path segments, no + # or spaces
  interval: 60                   # seconds between change checks
  qos: 1                         # 0 or 1
  retain: true
  username: llm-orch             # optional
  key: "{mqtt_password}"         # template, resolved from key_files
  tls: false                     # force TLS on plain host:port

key_files:
  mqtt_password: /etc/llm-orch/mqtt.pass   # keep it 0400
```

## Failure and reload behavior

- Broker down: connection errors retried by rumqttc's backoff; the
  warning is logged once per disconnect streak; publishes are dropped
  (the connect-republish catches everything up later).  Nothing on the
  request path changes.
- Config reload: any change to `mqtt:`, `node_name`, or the resolved
  password stops the old task (bounded 5 s graceful handshake) and
  starts a new one; removing the section stops publishing entirely.
  The reload path never waits on the broker beyond that handshake.
- Watched `key_files` rotation re-resolves `mqtt.key` (the publisher
  respawns only if the resolved password actually changed).

## Implementation map

- `src/mqtt.rs` — publisher task (`run_publisher`), controller
  (`MqttController::apply`/`shutdown`), doc builders, rate computation,
  canonical serialization.
- `src/stats.rs` — `build_response` / `peer_stat_docs` (shared with
  `handlers::stats_endpoint`).
- `src/config.rs` — `MqttConfig` + validation (`parse_broker`).
- `src/main.rs` / `src/reload.rs` — startup wiring, hot-reload diff,
  shutdown flush.
- `src/bin/llm-orch-stub-mqtt.rs` — minimal MQTT 3.1.1 broker for
  integration tests (`tests/mqtt.rs`).
