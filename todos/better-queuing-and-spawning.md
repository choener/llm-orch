# Better queuing and spawning — plan

## Goal

Make queue depth drive instance spawning: fix EMA recording on queue entry/assignment, and redefine autoscale thresholds as explicit integral counts of active requests (`in_flight + queued`) compared against a configurable EMA window, so bursts absorbed by `queue_depth` trigger a second instance when the queue stays deep.

## Current behavior (verified against code, Nov 2025)

- `max_concurrent` = per-instance slot cap on *orchestrator-accepted* requests (`instance.rs` `has_capacity` / `try_acquire`; enforced in `acquire_ready_slot`, `wake_one`, rebinding). Also substituted as `{max_concurrent}` into `cmd` (conventionally `--parallel {max_concurrent}`). Feeds autoscale capacity math. `in_flight` includes requests sitting in llama.cpp's own queue.
- `queue_depth` = per-model cap on the orchestrator FIFO of requests that couldn't grab any slot (429 at cap; `0` = no-queue special mode that starts a spawn demand, blocks until it finishes, then 429s). Heartbeats every `queue_heartbeat_secs` while parked. Deadline via `queue_wait_timeout_secs`.
- EMA (`ModelMetrics::tick`, tickless, 60/300/900s windows for m1/m5/m15) over `active = in_flight + queued` — correct formula, but only updated on slot acquire/release, NOT on queue entry or wake assignment (bugs below).
- Autoscale today: fractions. `should_spawn` (request path): `load_m5 > scale_up_at × max_concurrent × n`. `evaluate_autoscale` (30s tick): up `load_m5 > scale_up_at × max_concurrent × n`, down `load_m15 < scale_down_at × max_concurrent × (n−1)`.

### Bugs / issues found

1. `register_pending` (scheduler.rs:2686) never calls `record_metrics_event` after pushing a queue entry — queued requests are invisible to the EMA until the next release event or the 30s autoscaler tick.
2. `wake_one` (scheduler.rs:2917) never calls `record_metrics_event` on successful queue→slot assignment — EMA under-reports during queue drain.
3. `acquire_ready_slot` (scheduler.rs:2489) does call it (`record_metrics_event(model_name, 0)`) — inconsistent signal depending on admission path.
4. Autoscale thresholds are fractions of `max_concurrent × n`; queue depth only enters via the EMA, which has both the recording bugs and minutes of convergence lag (300s window) for bursts.
5. `queue_depth: 0` has split semantics (no-queue mode + test reliance at tests/integration.rs:7651) — **out of scope, leave untouched**.
6. `max_instances` defaults to 1 — autoscale enabled without raising it silently never scales up.

## Agreed design (decisions from discussion)

- `max_concurrent` stays "requests committed to the backend pipeline" (backend free to use `--parallel -1`, `--parallel {max_concurrent}`, or `--parallel 1`).
- `queue_depth` stays "requests llm-orch is willing to park with heartbeats".
- EMA = EMA over `active = in_flight + queued` (formula unchanged).
- **Scale-up rule**: `EMA(scale_up_value) > U_n` where `U_n = scale_up_at + (n−1) × max_concurrent`. At full saturation this is "spawn when queue-EMA > `scale_up_at − max_concurrent`" — the queue depth drives spawning, uniformly at any n.
- **Scale-down rule**: `EMA(scale_down_value) < D_n` where `D_n = scale_down_at + (n−2) × max_concurrent` (n ≥ 2). With `scale_down_at = max_concurrent` (safe value): "drop only when the remaining instances stay strictly below 100%".
- **`scale_up_at` / `scale_down_at` become integral counts** (`usize`), required in the autoscale block (no f64 defaults).
- **Two global (server-level) options**: `scale_up_value: m1|m5|m15`, `scale_down_value: m1|m5|m15`, both default `m1`. Fast reaction for bursty workloads; qwen-27b-style ~10s cold starts make quick down/up cycles cheap; freeing VRAM for other models quickly is worth more than hysteresis.
- **Config validation** (errors): `scale_up_at > max_concurrent`, `scale_down_at >= max_concurrent`, `scale_down_at < scale_up_at` (strict — guarantees a non-empty hysteresis band, prevents up/down overlap).
- **Config validation** (warnings): `autoscale.enabled && max_instances <= 1` (scale-up physically impossible); `autoscale.enabled && queue_depth == 0` (queue never fills ⇒ up-rule can never fire: `active ≤ n × max_concurrent < U_n`). New `Config::warnings()` mechanism surfaced in `--check-config` output and startup log.

## Tasks

### 1. Metric recording fixes (prerequisites)
- [x] `register_pending`: call `self.record_metrics_event(&model_name, 0)` after the queue push (after the inner lock scope closes)
- [x] `wake_one`: call `self.record_metrics_event(&logical_model, 0)` after a successful assignment transfer
- [x] Add scheduler unit tests: EMA/`load_m1` rises on enqueue; stays honest on wake assignment

### 2. Config: integral thresholds + EMA windows
- [x] Add `EmaWindow` enum (`M1 | M5 | M15`, serde lowercase `m1|m5|m15`)
- [x] ServerConfig: `scale_up_value: EmaWindow`, `scale_down_value: EmaWindow` (defaults `M1`), in `src/config.rs` next to the other queue/autoscale server fields
- [x] `AutoscaleConfig`: `scale_up_at`/`scale_down_at` `f64` → `usize`, required (remove `default_autoscale_up`/`default_autoscale_down`); update doc comments with the new formulas
- [x] `validate()`: add the three error checks; add `Config::warnings() -> Vec<String>` with the two warnings
- [x] Wire warnings into `main.rs`: print in `--check-config` OK output and `warn!` at startup after config load
- [x] Update config unit tests: remove 0.7/0.4 default assertions, add validation error/warning tests

### 3. Scheduler: new threshold semantics
- [x] InstanceManager: add `RwLock<EmaWindow>` copies `scale_up_value`/`scale_down_value` (pattern of `queue_wait_timeout_secs`: init in `new()`, refresh in the reload path ~line 4328)
- [x] `should_spawn` (request-path gate): replace `load_m5 > scale_up_at × max_concurrent × num_existing` with `EMA(scale_up_value) > scale_up_at + (num_existing−1) × max_concurrent`; cold-start (`num_existing == 0`) and autoscale-disabled paths unchanged
- [x] `evaluate_autoscale`: scale-up uses `U_n` + `scale_up_value` EMA; scale-down uses `D_n = scale_down_at + (n−2) × max_concurrent` + `scale_down_value` EMA; keep cooldown, `spawns_in_flight` dedup, `max_instances` gate, final idle despawn, and the no-autoscale surplus eviction unchanged
- [x] Update scheduler autoscale unit tests to the new semantics (despawn/keep-alive tests at ~5308–5355, 5576, 7516 — unchanged, they test the no-autoscale despawn paths); add unit tests for U_n/D_n math and the should_spawn gate
- [x] Update integration tests that reference autoscale (2 mentions) if needed; run full `cargo test`

### 4. Docs
- [x] `config.example.yaml`: new autoscale block with integral values + `scale_up_value`/`scale_down_value` server options, explained
- [x] README "Autoscaling" section: new formulas, validation rules, warnings, worked example (max_concurrent 2 / queue_depth 10 / scale_up_at 4 → spawn when queue-EMA > 2)

## Notes

- `queue_depth: 0` semantics intentionally untouched (design issue 5) — separate concern; the new warning covers the autoscale interplay.
- `max_instances` default stays 1; the new warning makes the "autoscale enabled but never scales" trap unmissable.
- EMA windows are cheap: `tick` already computes m1/m5/m15 unconditionally; the config just picks which one is compared.
- `load_m1/m5/m15` already exposed in `/v1/info` — no API change.
- Behavior after the change (max_concurrent 2, queue_depth 10, max_instances 4, scale_up_at 4, scale_down_at 2, m1/m1):
  1. Requests 1–2: slots, EMA ≈ 2.
  2. Request 3+: queue fills; with Fix 1 the EMA rises immediately; when queue-EMA > 2 (active-EMA > 4 = U_1), the next request's `should_spawn` gate triggers `ensure_spawn` → instance 2.
  3. If no further requests arrive, the 30s tick fires the same rule.
  4. Instance 2 ready → `wake_one` drains the queue; Fix 2 keeps the EMA honest during the transition.
  5. Burst ends; queue drains; m1 decays below D_2 = 2 within ~1–2 min → scale down, VRAM freed for other models.
