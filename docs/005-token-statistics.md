# Token / model statistics

Status: **implemented** (issue #1).

## Problem statement

Operators want to see, per model, how many tokens have been served —
prompt / cached / completion — plus request and time-to-first-token (TTFT)
statistics, both *cumulative* and *since boot*, and they want that data
available to external systems (Home Assistant) without scraping logs.

## Design decisions

- **JSON counter file, not SQLite.**  The use case is "totals for bragging"
  plus external polling, not queryable history.  A small JSON file keeps the
  dependency surface at zero (no native `libsqlite3` in the nix flake), costs
  O(1) per request, and follows the project's atomic temp-file + `mv`
  convention so readers never see a partial write.  Per-request history is
  already available via the optional `debug_log` JSONL; SQLite is deferred
  until a real query use case appears.
- **MQTT is opt-in, polling stays the default.**  `GET /v1/stats` is
  polled by Home Assistant's built-in HTTP/REST integration (with an
  `Authorization` header, so it works behind the nginx TLS proxy), which
  maps JSON fields to sensors via templates.  The endpoint always returns
  200 — `enabled: false` when the `stats:` section is absent — so pollers
  never handle error states.  An optional `mqtt:` section additionally
  *pushes* the same document (plus delta-derived rates) to an MQTT
  broker; see `docs/007-mqtt-stats.md`.
- **Two counter sets per model.**  `total` is restored from the JSON file at
  startup (survives restarts); `boot` is zeroed at every startup.  Both are
  updated after every served request.  Requests whose backend reported no
  usage (uninstrumented providers — e.g. some OpenAI-compatible embedding
  servers) are still counted as requests and duration, with zero token
  contributions; a `tracing::warn!` flags the missing usage so token
  counters are never silently short.  Failed requests (backend error
  responses, truncated streams) are not counted.
- **Best-effort persistence.**  File-write failures are `tracing::warn!`ed and
  swallowed — statistics must never fail or slow down a request.

## Configuration

```yaml
stats:
  file: /var/lib/llm-orch/stats.json   # parent directory must exist
```

Absent = feature off.  Relative paths resolve against the config file's
directory.  The path is hot-reloadable: counters are kept, the new file is
published immediately with the existing state.

## JSON format (version 1)

```json
{
  "version": 1,
  "started_at": 1751880000,
  "updated_at": 1751966400,
  "models": {
    "qwen3-32b": {
      "total": { "requests": 12, "prompt_tokens": 480, "cached_tokens": 96,
                 "completion_tokens": 360, "ttft_count": 8,
                 "ttft_sum_ms": 920, "ttft_min_ms": 80, "ttft_max_ms": 240,
                 "duration_sum_ms": 72000, "duration_avg_ms": 6000,
                 "last_active": 1751966400 },
      "boot":  { "…": "same shape" }
    }
  }
}
```

`started_at` is the first startup (restored from the file if present).

## Endpoint

`GET /v1/stats` (auth-gated like the other `/v1/*` endpoints):

- the counters document above (as `models`, plus `enabled`, `started_at`,
  `updated_at`), and
- `gpus` — the latest `GpuMetrics` snapshot (VRAM, busy %, temperature,
  power, clocks) in the same `GpuStatus` shape as `/v1/info`.

## Recording points

- **Non-streaming**: `record_stats()` in `src/handlers.rs` on the
  aggregate-response path of chat/completions, responses, completions,
  embeddings, and rerank.  Usage shapes handled: `prompt_tokens` /
  `completion_tokens` (chat, completions, embeddings, rerank) and
  `input_tokens` / `output_tokens` (Responses API); cached tokens from
  `*_tokens_details/cached_tokens`.
- **Streaming**: `SseForwarder` (chat, responses, completions) timestamps the
  first non-empty SSE data chunk (TTFT proxy) and records on clean stream
  end (`[DONE]`) — with the trailing usage chunk when present, otherwise
  request/duration only.  Truncated streams and client disconnects are not
  recorded, matching the completion ring-buffer rule.  Only successful
  completions are counted.

## Code layout

- `src/stats.rs` — `StatsStore` (counters, restore, atomic persistence,
  `set_path` for reload), `ModelCounters`, `StatsFile`.
- `src/config.rs` — `StatsConfig` + validation (non-empty path, existing
  parent directory).
- `src/handlers.rs` — `record_stats()`, `stats_endpoint()`, `SseForwarder`
  TTFT/usage recording.
- `src/reload.rs` — stats path retarget on config reload.
- `src/server.rs` — `AppState.stats`, `/v1/stats` route.

## Test coverage

- Unit (`src/stats.rs`): counter accumulation, file write/restore (total
  persists, boot zeroes), corrupt-file recovery, disabled no-op, path
  retarget, missing-parent robustness.
- Integration (`tests/integration.rs`): end-to-end non-streaming + streaming
  requests update the JSON file and `/v1/stats`; TTFT is recorded for
  streaming; `enabled: false` when unconfigured; 401 without auth; hot-reload
  retarget publishes the new file; config validation rejects a missing parent
  directory.
