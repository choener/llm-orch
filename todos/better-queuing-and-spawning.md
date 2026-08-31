# Better queuing and spawning

## Problem description

I have:

- `--parallel 1` on the llama.cpp command line — the backend processes one
  request at a time, which keeps pp and tg throughput stable.
- `max_concurrent: 4` on the model — accepts up to 4 concurrent in-flight
  requests onto one instance, **but** llama.cpp processes them one-at-a-time
  via its internal queue, so 3 of them sit in llama.cpp's own slot, not the
  orchestrator's queue.
- `queue_depth: 10` — I want to enqueue up to 10 additional requests while
  waiting for a slot, sending SSE heartbeats so the frontend doesn't time out.

What actually happens: the first 4 requests get a slot immediately (the
orchestrator hands them off to llama.cpp, which internally queues them).
Requests 5+ go into the orchestrator's logical queue.  Because `max_concurrent`
is generous (4), the orchestrator never feels pressure to spawn a second
instance — the 5th request only reaches the orchestrator's queue once all
4 slots are claimed, and by then the autoscaler is behind.

Because `queue_depth` only counts requests that couldn't grab a **slot** at
all, it doesn't participate much in the scaling decision.

The goal: let `queue_depth` absorb bursts (with heartbeats), use
`max_concurrent` as a tighter cap to match `--parallel`, and let the
autoscaler react to queue buildup to spawn another instance when needed.

## What I want

- `--parallel 1` — llama.cpp deals with one request at a time
- `max_concurrent: 2` — one active processing, one in llama.cpp's own queue
- `queue_depth: 10` — ten more requests wait in the orchestrator's queue,
  receiving SSE heartbeats so the frontend doesn't disconnect; no 429s
  unless the queue overflows
- **Autoscale**: when sustained load exceeds a threshold *and free VRAM
  exists*, spawn another instance — the queue depth should drive this,
  not just `max_concurrent × num_instances`

## What's wrong today

### Bug 1: Queue entries don't update the EMA

When `admit()` runs out of slots and calls `register_pending()`, the queue
entry is added but `record_metrics_event` is **never called**.  The EMA
(`load_m5`, `load_m15`) stays at whatever `in_flight` was before queuing.
Queued requests are invisible to autoscale until:

- A **release** event (the active request finishes → the background release
  task calls `record_metrics_event(&model_name, 1)`)
- The **autoscaler tick** runs and calls `force_refresh` — but that only
  happens every 30 seconds.

If the active request(s) are long-running, the autoscaler sits on stale
data for up to 30 seconds and never sees the 5 queued requests piling up.

The code path:

```
register_pending (line ~2686)
  ↓ queue.push_back(entry)
  ↓ scheduler_metrics.queue_enqueued_total.fetch_add(…)
  ↓ wake_one(&model_name)   // tries to assign immediately, but all slots busy
  // NO record_metrics_event call
```

The release processing in `main.rs`:

```rust
while let Some(model_name) = rx.recv().await {
    mgr.record_metrics_event(&model_name, 1);  // only on slot release
    mgr.wake_one(&model_name);
}
```

Only updates the EMA on slot release, not on queue entry.

### Bug 2: `wake_one` also skips `record_metrics_event`

When `wake_one` pops a queued request and successfully calls `try_acquire`,
`in_flight` goes up by 1 and `queued` goes down by 1 — but the EMA isn't
updated to reflect the transition:

```rust
// wake_one — line ~2917
match guard {
    Some(g) => match entry.assignment.send(Ok((logical_model, g))) {
        Ok(()) => {
            // ← record_metrics_event NOT called
            return;
        },
```

The EMA only catches up when the **response finishes** (guard drops →
release_tx → release task).  So after a batch wake-up, the EMA
under-reports load until each request completes.

### Bug 3: `record_metrics_event` isn't called from `acquire_ready_slot` either

Actually, it *is* — but only for the immediate-slot path (no queue):

```rust
// acquire_ready_slot — line ~2489
fn acquire_ready_slot(...) -> Option<SlotGuard> {
    let guard = handle.try_acquire(max_concurrent)?;
    self.record_metrics_event(model_name, 0);  // ← called here
    Some(guard)
}
```

But when a request goes through the queue and later gets assigned via
`wake_one`, `record_metrics_event` is **not called**.  So the EMA
measurement depends on *which code path* admitted the request, producing
an inconsistent signal.

### Design issue 4: Autoscale thresholds use `max_concurrent`, ignoring queue depth

```rust
// should_spawn — line ~1687
let cap = (max_concurrent * num_existing) as f64;
let threshold = a.scale_up_at * cap;
```

```rust
// evaluate_autoscale — line ~4717
let capacity = (cfg.max_concurrent * num_instances) as f64;
if m.load_m5 > a.scale_up_at * capacity { ... }
```

With `max_concurrent: 2` and `num_instances = 1`:

- `capacity = 2.0`
- `threshold = 0.7 × 2.0 = 1.4`

`load_m5` converges toward `in_flight + queued`.  If both slots are full
(2 in-flight, 10 queued), `load_m5 → 12.0`, which trivially exceeds the
threshold.  **But** with Bug 1, queued requests never push the EMA up,
so the autoscaler never reaches this conclusion.

Additionally: `max_instances` defaults to **1**.  Even with correct EMAs,
the autoscaler's cap check blocks every scale-up:

```rust
if num_instances < cfg.max_instances {  // fails when max_instances=1
    ...
```

### Design issue 5: `queue_depth: 0` bypass works but is inelegant

`queue_depth: 0` is documented as "reject all queuing".  But it also has
a special-case fast-path that triggers an immediate spawn attempt before
returning NoCapacity:

```rust
// admit — line ~1620
if queue_cfg.queue_depth == 0 {
    if let Some(demand) = demand {
        match demand {
            AdmissionDemand::Spawn { model_name, cfg } => {
                let mut rx = self.ensure_spawn(&model_name, &cfg);
                let budget = self.spawn_timeout + SPAWN_SERIAL_WINDOW + Duration::from_secs(30);
                let _ = tokio::time::timeout(budget, rx.changed()).await;
            }
            ...
        }
    }
    return Err(AcquireError::NoCapacity);
}
```

This is used in the "no queue, just spawn" pattern.  But the test at
`tests/integration.rs:7651` uses `queue_depth: 0` for the dense model in
make-room scenarios — it treats depth-0 as "never enqueue, let the
fallback candidate serve instead".  Not directly a bug, but worth noting
that `queue_depth: 0` has split semantics.

### Design issue 6: `record_metrics_event` mixes `in_flight + queued` into the EMA, but queued isn't always visible

```rust
// record_metrics_event — line ~4428
let in_flight: usize = self.instances.read().unwrap().get(model_name)
    .map(|list| list.iter().map(|h| h.inner().lock().unwrap().in_flight).sum())
    .unwrap_or(0);
let queued = self.queue_depth(model_name);
let active = in_flight + queued;
```

This _would_ see queued requests correctly — if only `record_metrics_event`
were called when queuing happens.  The formula itself is fine; the problem
is entirely about **when** it's called.

## Proposed fixes

### Fix 1: Record metrics on queue entry

In `register_pending`, after pushing the entry (and after the inner scope
closes), add:

```rust
queue.push_back(entry);
} // close inner scope

self.record_metrics_event(&model_name, 0);  // ADD
self.scheduler_metrics.queue_enqueued_total.fetch_add(1, Ordering::Relaxed);
```

This makes the EMA immediately reflect queued load.  The autoscaler can
see `active = in_flight + queued` at the next tick or force-refresh.

### Fix 2: Record metrics on wake assignment

In `wake_one`, after a successful transfer:

```rust
Ok(()) => {
    self.scheduler_metrics.queue_assigned_total.fetch_add(1, Ordering::Relaxed);
    self.record_metrics_event(&logical_model, 0);  // ADD
    return;
}
```

This keeps the EMA accurate when a request transitions from queue to
in-flight.

### Fix 3: Reconsider autoscale to use queue-depth‑aware thresholds (optional)

The current autoscaler threshold formula `scale_up_at × max_concurrent ×
num_instances` works reasonably well *with* Fix 1+2, because the EMA
includes queued load.  A queue of 10 behind 2 in-flight slots gives
`load_m5 ≈ 12`, which beats `0.7 × 2 = 1.4`.

But an alternative formulation could be:

```
trigger when  queue_depth(model) >= min(config.queue_depth, autoscale.queue_up_threshold)
```

This would scale up not based on the EMA at all, but on raw queue fill.
That's simpler to reason about and removes the EMA convergence delay.
Pushing this too far would give the decision to the config author rather
than the smoothed signal — but it could be an additive second trigger
("scale up if EMA says so **or** queue is more than `queue_up_at: 5` deep").

### Fix 4: `max_instances` must be > 1 for autoscaling to work (documentation)

It's already documented in the YAML defaults (default is 1).  But it's an
easy trap: you enable autoscale, set `max_concurrent: 2`, `queue_depth: 10`,
and forget to set `max_instances: 4` (or whatever your VRAM budget allows).
The autoscaler then faithfully checks `num_instances < cfg.max_instances`
and skips the scale-up every time.  The model seems broken: 429s despite
free VRAM.

## How the corrected system would behave

With `max_concurrent: 2`, `queue_depth: 10`, `max_instances: 4`,
`autoscale.enabled: true`, `autoscale.scale_up_at: 0.7`, and `--parallel 1`:

1. **Request 1** → `acquire_ready_slot`: `in_flight=1`, EMA `active=1`, OK
2. **Request 2** → `acquire_ready_slot`: `in_flight=2`, EMA `active=2`, OK
3. **Request 3** → all slots busy → `register_pending`.  EMA `active=2+1=3`.
   Autoscaler next tick sees `load_m5 > 0.7 × 2 = 1.4` → **spawn instance 2**.
4. **Requests 4–12** → fill queue to depth 10.  Request 13 gets 429.
   EMA keeps rising (`active = 2 in-flight + 10 queued = 12`).
5. **Instance 2 becomes Ready** → `wake_one` assigns queued requests.
   `record_metrics_event` keeps EMA honest during the transition.
6. If `load_m5` stays high → **spawn instance 3**, etc.
7. As queue drains and `load_m15` drops below `scale_down_at × (instances−1) × max_concurrent`
   → scale back down.

The queen's gambit: with `--parallel 1` and `max_concurrent: 2`, the
second in-flight slot is always "waiting at llama.cpp's door".  This is
fine: llama.cpp will pick it up as soon as the previous one finishes,
keeping utilization high.  The orchestrator queue absorbs bursts, and
the autoscaler uses the combined `in_flight + queued` signal to decide
when another instance is worth the VRAM.