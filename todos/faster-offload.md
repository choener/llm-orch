# TODO

## Goal
Implement transactional GPU make-room migration and an OpenAI-compatible waiting queue that keeps client connections alive, while never interrupting in-flight requests; update the Rust code, tests, configuration, and documentation.

## Tasks

### 1. Document the current scheduler and identify integration points
- [x] Record the existing instance states, slot guards, per-model FIFO queues, spawn coordination, VRAM accounting, drain/reap paths, and current queue timeout behavior.
- [x] Identify the exact scheduler, handler, config, and server changes the implementation requires, including interactions with aliases, autoscaling, hot reload, and shutdown.

#### Current implementation baseline

- **Admission path:** text, Responses, embeddings, rerank, and audio handlers resolve an alias to candidate model names and call `InstanceManager::acquire_for_candidates`. Direct model requests are represented as a one-element candidate list with make-room disabled. The returned `SlotGuard` owns the in-flight slot and is held through backend forwarding; dropping it updates activity metrics and wakes a waiter.
- **Instance lifecycle:** `InstanceState` currently has `Loading`, `Ready`, and `Failed`. `Failed` is overloaded for crashed, retired, and draining instances; it means “not routable,” although a draining child can remain alive while `in_flight > 0`. `Instance::in_flight` is the authoritative active-request count. `SlotGuard::drop` decrements it exactly once.
- **Current queue:** queues are per model and FIFO (`VecDeque<(request_id, oneshot::Sender<SlotGuard>)>`). A waiter is queued only when at least one non-`Failed` instance exists, because a release or successful spawn must be able to wake it. `wake_one` acquires the slot under the instance lock before sending the guard, preventing an acquire race. Release notifications and successful spawns wake queues; removal of the last instance drops all queued senders.
- **Current queue deadline:** `QUEUE_WAIT_TIMEOUT` is 120 seconds, but the effective wait is `max(120s, spawn_timeout + serialized_spawn_window + 60s)`. With current defaults (`spawn_timeout = 120s`, serialized window `= 15s`) this is 195 seconds, after which the HTTP request receives `NoCapacity`/429. This is the primary disconnect/timeout behavior to replace with the configured long wait.
- **Current spawning:** `ensure_spawn` deduplicates per-model loads in detached tasks, so a caller disconnecting does not abort a load. `try_spawn` serializes the critical load window with a global semaphore, checks the instance cap, allocates a port, selects GPUs, starts the child, waits for readiness, and registers a `Ready` instance. The spawn path rechecks VRAM and instance limits; feasibility checks are deliberately advisory.
- **Current VRAM accounting:** `select_gpus_inner` keys usage by PCI slot, translates each instance through its model’s backend namespace, attributes each model’s declared `vram` to every GPU it occupies, and includes `Loading` and draining/retired instances unless explicitly excluded by a make-room dry run. It refuses a spawn if the selected pool cannot provide all required GPUs. It does not currently model temporary duplicate instances during migration or reserve a destination GPU for a replacement instance.
- **Current make-room behavior:** aliases already have `make_room` (`none`, `evict_idle`, `drain_surplus`) and `drain_timeout`. `plan_make_room` chooses idle stale victims and, optionally, busy surplus duplicates; `execute_make_room` removes idle victims or marks busy victims `Failed`, waits for them to become idle, and reaps them. This can free a GPU but cannot re-upload the victim to another GPU, redirect its queued requests to a replacement instance, or atomically coordinate victim migration with loading the large model. It also applies only to alias requests in the current design.
- **Current removal guarantees:** `remove_instance` is idle-only. `drain_instance` marks a busy instance unroutable but leaves its child alive until `reap_drained` sees `in_flight == 0`; only shutdown uses force removal. Thus the “never interrupt active request” invariant already exists for normal removal and must remain true for migration.
- **Background coordination:** `main.rs` runs release processing (`record_metrics_event`, `wake_one`, `reap_drained`), crash processing, and 30-second autoscaling. Autoscaling also reaps stranded loading instances and globally reaps idle instances after `drain_idle_timeout`; it must not mistake a migration reservation or queued migration work for ordinary scale-down eligibility.
- **HTTP/SSE boundary:** handlers acquire a slot before returning any response. The backend stream is wrapped in `SseForwarder`, and axum’s `KeepAlive::default()` starts only after the handler has acquired a slot and constructed the SSE response. Consequently, a request waiting inside `acquire_for_candidates` emits no heartbeat today. Non-streaming handlers aggregate JSON and cannot send interim bytes without violating the OpenAI response format.
- **Configuration and reload points:** `ServerConfig` owns global timeouts and is the natural home for a global queue-wait limit; model config owns queue depth, and alias config owns make-room/drain settings. `reconcile_config` atomically updates manager configuration and retires stale spawn configurations while allowing active requests to drain. A future migration must snapshot the resolved request policy, preserve it for the waiting request, and make reloads leave reservations/queues in a valid state.
- **Shutdown:** axum stops accepting new connections and waits only `shutdown_drain_timeout_secs` before aborting remaining HTTP requests; then `shutdown_all` force-removes backends. “Keep queued clients connected” therefore applies during normal scheduling, not past a bounded process shutdown unless shutdown semantics are explicitly expanded.

### 2. Define the make-room and migration state machine
- [x] Specify request states and instance states for `Ready`, `Draining`, migration/loading, failed migration, and completed removal, including the invariants that no active request is killed and no draining instance accepts new work.
- [x] Specify the large-model admission transaction: select a least-recently-used victim, reserve a destination GPU, drain only new work, optionally load the victim on the other GPU, retire the old instance after its active requests finish, free the source GPU, and load the large model there.
- [x] Define VRAM reservations and atomic commit/rollback rules for temporary duplicate instances during migration, including behavior when destination loading fails, the victim has no destination, or another request races the plan.
- [x] Define fairness, serialization, and anti-thrashing rules so concurrent make-room requests cannot evict one another indefinitely; explicitly state that active requests are never interrupted.

#### Migration model

Treat a model instance as a logical service plus a physical backend. Requests are
routed to the logical service, never permanently to a particular GPU or process.
A migration changes the physical backend while preserving the logical model name,
its queue, and its in-flight requests.

Proposed request states:

1. `Admitted`: the request has passed auth/config validation and has a resolved
   model or alias candidate list.
2. `WaitingForCapacity`: no routable slot exists, or the selected model is behind
   a migration barrier. The queue entry owns a cancellation/deadline token and a
   logical model/generation, not an `InstanceHandle`.
3. `Assigned`: the scheduler atomically acquired a slot on a `Ready` instance.
4. `Running`: the handler is forwarding to the backend; the slot guard remains
   alive for the entire aggregate response or stream.
5. `Completed`, `Cancelled`, or `Expired`: the guard is dropped and the queue
   entry, if any, is removed. A waiting request expiring never cancels an
   unrelated migration or an active request.

Proposed explicit instance states (the implementation may initially map these to
existing `Failed`/draining mechanics, but the conceptual distinction is needed):

- `Loading { generation, purpose }`: process is starting. It consumes a
  reservation but accepts no requests. `purpose` distinguishes normal spawn,
  migration replacement, and large-model admission.
- `Ready { generation }`: routable if it has a free slot and is not behind a
  model migration barrier.
- `Draining { migration_id }`: no new slots may be acquired; existing slot
  guards continue to run on the old process. New requests for the logical model
  enter its queue.
- `Retired`: the old process is no longer routable and has no active slots; its
  process/port/GPU reservation can be released. This is a short-lived internal
  state before registry removal.
- `Failed`: startup/crash/removal failed or the process is no longer usable. It
  is never a migration fallback unless a new instance is successfully loaded.

The key invariants are:

- A request whose slot was acquired before `Draining` continues to completion.
  Migration never kills its backend or forcibly drops its slot.
- `Draining` is checked under the same instance/scheduler coordination used for
  slot acquisition. A request racing the transition is either acquired before
  the transition and counts as existing work, or queued after it; it cannot be
  silently lost.
- A draining instance never receives a new backend request.
- A queue is attached to the logical model (and optionally a model generation),
  not to the old process. Once the replacement is ready, queued requests can be
  woken against the replacement.
- Every physical instance and every in-flight load has exactly one VRAM
  reservation. Reservations are released only on abort, failed load cleanup, or
  committed removal.

#### Large-model admission transaction

For the example below, capacities are normalized to 100%:

```text
GPU A: M1 = 30%
GPU B: M2 = 30%
Request: L = 90%
```

The scheduler runs one serialized `make_room` transaction, rather than allowing
each request to independently evict whatever it sees:

1. **Choose a target placement.** Dry-run placement for `L`, including all
   existing reservations and any reservations held by other transitions. In the
   example, `L` can use B only after M2 leaves B.
2. **Choose a victim.** Consider ready instances occupying the target placement,
   excluding `L` itself, already draining instances, and instances needed by an
   earlier transaction. Use least-recently-used ordering (`last_completed` in
   the new model; `last_active` is the current approximation), then lowest
   active-request count, then a stable instance ID. An active victim is eligible
   for *draining*, not for interruption. The first implementation uses LRU;
   priority/policy is a future ordering hook.
3. **Find a destination before changing routing.** For victim M2, find a legal
   placement on A such that the temporary duplicate fits with M1. The placement
   must satisfy M2's device pool, device count, per-device reservation, and
   backend namespace rules. If no such destination exists, this victim is not a
   non-interrupting migration candidate; try another victim/plan or leave `L`
   waiting.
4. **Reserve the whole plan atomically.** Under the scheduler transition lock,
   reserve M2's destination on A and the eventual placement for L on B. Include
   the old M2 reservation on B until the old instance is retired. Once committed,
   competing schedulers see the reservations and cannot claim either placement.
5. **Install a migration barrier.** Mark the old M2 instance `Draining` and
   associate it with `migration_id`. New M2 requests are queued. M1 remains
   routable; requests for unrelated models are unaffected.
6. **Load the replacement.** Start M2 on A while the old M2 remains alive on B.
   This preserves active M2 requests and overlaps model upload with their
   remaining execution. The replacement is not routable until its readiness
   check succeeds.
7. **Wait for a safe handoff.** After the replacement is `Ready`, wait until
   `old_M2.in_flight == 0`. Then atomically switch M2's logical routing
   generation to the new A instance, retire/remove old M2 from B, and wake M2's
   queue. Queued M2 requests are now assigned to A.
8. **Load the preferred large model.** B is now free, so start L there. The L
   request and any coalesced L requests remain in the admission queue until L is
   ready. M2 remains available on A while L loads.
9. **Commit/observe.** On successful L readiness, release temporary migration
   bookkeeping and wake the L queue. The final reservations are M1=30% + M2=30%
   on A and L=90% on B.

This is intentionally a two-phase physical handoff: load the victim replacement
first, then free the source GPU. It avoids taking a model completely offline
while its active request is running and makes the requested “re-upload on the
other GPU” behavior explicit. If a migration replacement is too large to fit
alongside the destination GPU's existing models, the plan must not pretend that
freeing the source GPU is enough; it must wait, choose another victim, or remain
queued.

#### Reservation and rollback rules

Use a scheduler-owned reservation table in addition to the registered-instance
VRAM accounting:

```text
Reservation = { transition_id, purpose, model, PCI-slot set, vram per slot,
                phase, expiry/cleanup token }
```

The projected usage for a device is:

```text
actual registered reservations
+ in-flight normal spawns
+ migration replacement reservations
+ large-model admission reservations
```

A dry run must use the same per-PCI-slot accounting as normal spawn selection,
including multi-GPU reservations on every occupied GPU and all old/draining
instances. It may subtract only reservations owned by the same transaction.
Declared `vram` remains the conservative admission reservation; actual metrics
can reject a stale/unsafe plan but must not allow an unreserved overcommit.

Transaction rules:

- Reserve first, then mark an instance draining. If the reservation cannot be
  acquired, no routing state changes.
- A reservation is not considered free merely because a process is marked
  `Draining`; it is free only after the old process is idle, terminated, and
  removed from the registry.
- Temporary duplicate M2 on A is counted in peak usage while old M2 on B still
  exists. After the handoff, the old reservation is released and B becomes
  available for L.
- If destination loading fails before the old M2 is removed, cancel the
  transaction, release the destination reservation, clear the migration barrier,
  and wake M2's queue against old M2. A backend crash/readiness failure follows
  the normal crash accounting, but must not strand the queue.
- If the replacement succeeds and old M2 is removed but L then fails to load,
  retain the successfully migrated M2 on A; do not immediately migrate it back.
  Release B and either retry L under a bounded admission policy or leave L in
  the waiting queue. This avoids failure-induced ping-pong.
- If the client that triggered the transaction disconnects, the transition is
  infrastructure work and continues until a safe cancellation point. If no
  demand remains and the replacement has not started, it may be cancelled and
  rolled back; once the old instance has been retired, do not undo a successful
  migration solely because the triggering client disappeared.
- On config reload, validate and snapshot new policy first. An in-progress
  transition either completes using its immutable snapshot or is cancelled at a
  safe phase; never apply half of an old placement and half of a new placement.
  A changed device map/model spawn fingerprint must retire affected replacement
  work and release its reservations.

#### Victim selection and fairness

The first policy is `lru`: select the least recently used eligible instance. The
current `Instance::last_active` is updated on acquire and release, so a future
implementation should add a separate `last_completed` timestamp if exact LRU is
needed. Use this ordering within a transaction:

1. eligible idle instances, oldest completion first;
2. eligible busy instances, oldest completion first, then fewest active slots;
3. stable instance ID as a deterministic tie-breaker.

An idle sole instance is migratable if a destination can be reserved; it is not
necessary to have a duplicate because the old process remains alive until the
replacement is ready. A busy sole instance is also migratable under the same
rule: its active requests finish on the old process, while new requests queue.
Never select a model with no valid destination merely to make the large model
fit. If future policy permits destructive idle eviction, that must be a separate
explicit mode, not an accidental migration fallback.

Serialize structural transitions with one admission/migration coordinator (or a
per-GPU lock plus a globally ordered lock acquisition). Coalesce requests for the
same target placement onto one transition and one large-model load. A later
request must observe the transition reservation and queue behind it rather than
starting a competing eviction. FIFO admission order prevents a stream of new
large-model requests from repeatedly replacing the current victim before queued
small-model requests are served. Any transition/drain timeout applies to the
transition or waiting request, never by killing an active backend request.

### 3. Define persistent waiting-queue semantics
- [x] Replace the fixed short queue wait with a configurable maximum queue wait, proposing `server.queue_wait_timeout_secs` with a default of `3600` seconds and an explicit value for unlimited waiting if appropriate.
- [x] Specify FIFO behavior, queue capacity, cancellation when the client disconnects, deadline expiry, migration-triggered wakeups, model unload/config-reload behavior, and `Retry-After`/error behavior after the configured maximum.
- [x] Define how queued requests are associated with a migration/instance-generation so they follow the victim model to its replacement GPU rather than being stranded on the old instance.

#### Queue configuration and lifetime

Add a global server setting:

```yaml
server:
  queue_wait_timeout_secs: 3600  # 0 = no scheduler deadline
  queue_heartbeat_secs: 10        # streaming wait heartbeat interval
```

The default removes the current approximately 195-second effective limit while
still bounding abandoned requests. `0` means the scheduler itself does not
expire a waiter; the connection can remain open until the client/proxy cancels
it or the process shuts down. This is the “indefinite” mode, not a promise that
an external reverse proxy will retain an idle connection forever. The queue
wait deadline is separate from backend request timeouts: backend timing starts
only after a slot is assigned, so a long model upload does not consume the
backend generation budget.

`queue_depth` remains the per-model admission cap. The queue-wait timeout is
server-wide for the first implementation so all API endpoints have consistent
semantics; a per-model override can be added later without changing the queue
entry shape. Validate the heartbeat interval as positive when queue streaming is
enabled, and validate the finite wait timeout as either zero or a supported
positive duration.

#### Queue entry and dispatch model

Replace a queue entry that is only `(id, oneshot<SlotGuard>)` with the equivalent
logical information:

```text
QueueEntry {
    id,
    resolved candidates / selected logical model,
    alias and policy snapshot,
    model generation or migration barrier ID,
    enqueue timestamp,
    optional absolute deadline,
    assignment sender,
    cancellation token,
}
```

The exact Rust representation is open, but it must not store only an
`InstanceHandle`: an instance handle becomes invalid as a routing target when a
model migrates. The assignment message should contain the newly acquired slot
and the selected physical instance, or a terminal queue error.

Admission behavior:

1. Resolve the alias and snapshot its candidate/policy/make-room settings before
   waiting. Do not let a hot reload silently change an already accepted
   request's meaning.
2. Try a ready slot and ordinary spawn first according to the alias policy.
3. If a make-room transaction is active for the selected logical model, or the
   selected instance is `Draining`, enqueue the request behind that barrier.
4. Once queued for a victim model, pin the request to that logical model and
   migration generation. It must follow the replacement instance instead of
   falling through to a different physical GPU or becoming attached to the old
   process. Candidate fallback may be considered before enqueueing, but not
   halfway through a committed migration.
5. When a replacement becomes ready and the handoff commits, wake the model's
   queue. `wake_one` must re-check current generation/state and acquire the slot
   on the replacement under the instance lock before sending it to the waiter.

This means in the example that new M2 requests are waiting while M2 is being
copied to A; after the M2 handoff they are served on A, while the original M2
requests finish on B. M1 remains available unless it is independently selected
as the victim of a separate transaction.

#### FIFO, wakeups, and fairness

Maintain FIFO order within each logical model queue. A waiter that cannot be
served at the instant of a wake is reinserted at the head rather than skipped.
Wakeups are generated by:

- a slot release;
- a successful normal spawn;
- migration replacement readiness and old-instance handoff;
- large-model readiness;
- migration rollback/abort, which makes the old instance routable again;
- config-reload reconciliation when a valid new route is available.

Structural transitions are serialized independently of slot wakeups. A release
must never cause a second migration to start while the first transaction owns
the affected reservations. Requests for the same target model coalesce onto the
same in-flight spawn/migration; they do not each plan a new victim.

A queue should expose separate counts for `waiting_for_slot`,
`waiting_for_migration`, and `waiting_for_spawn`, because “queued” alone is not
enough to diagnose a long wait. Queue metrics should distinguish requests
waiting behind an active request from requests waiting for VRAM.

#### Cancellation and expiry

The HTTP handler must observe request cancellation while it waits for an
assignment. Dropping a request must remove its queue entry promptly, not leave
a dead sender consuming `queue_depth` for up to an hour. Use a queue-ticket
owner whose `Drop` removes the entry by ID, or an equivalent cancellation token
that a queue-maintenance task consumes. A race between cancellation and wake is
safe: if a guard was already acquired for a cancelled receiver, dropping that
guard immediately releases the slot and triggers the normal release path.

At the configured deadline:

- remove the entry if it is still queued;
- record queue wait duration and the reason (`deadline`, `queue_full`,
  `model_removed`, `migration_failed`, or `shutdown`);
- return an error without touching active requests or cancelling the shared
  migration;
- ensure a timed-out request cannot later receive a slot through a stale sender.

Recommended HTTP mapping for compatibility is `429 Too Many Requests` with
`Retry-After` for an immediately full queue, and `504 Gateway Timeout` (or a
clearly documented equivalent) when an accepted request reaches its maximum
wait. Retaining the existing 429 mapping for both is also possible, but the
implementation should distinguish the metrics and error body. If a model is
removed or permanently unavailable, return the existing service-unavailable
error instead of waiting until the deadline.

The deadline never applies to a request after it has acquired a slot. Once
running, existing backend timeout and client-disconnect behavior applies. A
client disconnect never aborts a shared model migration that may be needed by
other queued requests; it only removes that request's waiter.

#### Model removal, reload, and shutdown

If an instance is removed while a queue exists, queued entries must be either
rebound to a committed replacement, moved to an explicitly selected fallback
before the queue is closed, or completed with a terminal error. Dropping all
senders is acceptable only when no routable replacement can ever wake the queue.
Config reload is atomic: queued requests keep their captured policy, while the
manager validates the new configuration and reconciles generations. A changed
spawn fingerprint or removed device must cancel/replan the affected migration
and release reservations; it must not leave a queue pointing at a removed
instance.

During shutdown, stop accepting new work, mark queued requests as shutdown
rather than waiting for the normal one-hour deadline, and allow active requests
to drain under the existing bounded shutdown policy. Never force-kill a backend
that still owns an active request during ordinary migration or queue expiry.

### 4. Research client-compatible heartbeats and status reporting
- [x] Investigate the OpenAI-compatible SSE conventions and the actual streaming parsers/timeout behavior used by pi coding agent, OpenCode, and representative clients.
- [x] Evaluate universally safe transport heartbeats (SSE comment frames) versus optional named status events, custom OpenAI-compatible chunks, headers, and a separate status endpoint; document what each client can display without breaking ordinary clients.
- [x] Recommend a compatibility-first protocol: standard SSE heartbeats for all streaming clients, plus an explicitly negotiated llm-orch status extension if a client opts in; document the unavoidable limitation that a non-streaming JSON response cannot carry interim status bytes while remaining valid OpenAI JSON.

#### Findings from protocol/client research

The relevant protocol is ordinary HTTP Server-Sent Events, not WebSocket:

- OpenAI documents Chat Completions streaming as **data-only SSE** and uses
  `data: [DONE]` as the terminal marker for the legacy Chat/Completions format.
- The SSE format permits comment lines beginning with `:`. MDN documents that
  these lines are ignored by an SSE parser, while the bytes still refresh the
  network connection. A complete comment heartbeat is therefore:

  ```text
  : llm-orch waiting; phase=migration; request=...\n
  \n
  ```

- OpenAI’s JavaScript SDK (`openai-node`, `src/core/streaming.ts`) ignores
  comment lines, joins `data:` fields, parses JSON data events, honors
  `[DONE]`, and supports AbortController cancellation. A comment heartbeat is
  invisible to the SDK’s typed stream and does not become assistant content.
- Pi coding agent’s OpenAI Chat provider
  (`earendil-works/pi/packages/ai/src/api/openai-completions.ts`) consumes the
  OpenAI SDK stream with `for await`, translating recognized text/thinking/tool
  deltas into its own `AssistantMessageEventStream`. Its Responses provider
  similarly processes recognized Responses events. The inspected paths expose
  abort signals and `timeoutMs`, but no generic raw-SSE/comment callback that
  could display an orchestrator status message. Comments keep the connection
  alive but are intentionally not shown in Pi’s conversation UI.
- OpenCode’s OpenAI Chat protocol
  (`anomalyco/opencode/packages/llm/src/protocols/openai-chat.ts`) uses
  `HttpTransport.sseJson`/`Framing.sse` and decodes `data:` payloads into a
  typed OpenAI event schema. Comments are discarded by framing; event names
  are not a general status channel. Its Responses protocol expects the known
  `response.*` event family and has explicit terminal events. There is no
  portable indication that arbitrary orchestrator events would be surfaced in
  the OpenCode TUI.
- A representative official SDK therefore confirms the useful split: comment
  frames are safe transport heartbeats, but arbitrary named events are not a
  safe universal application protocol. Some clients dispatch unknown named
  events correctly, while others try to decode every data payload as an
  OpenAI chunk or validate it against a strict schema.

#### Compatibility matrix

| Mechanism | Ordinary OpenAI clients | Pi / OpenCode display | Recommendation |
| --- | --- | --- | --- |
| SSE `:` comment heartbeat | Ignored by conforming parsers; keeps bytes flowing | Keeps their stream/read timeout from firing; no UI status | **Default** |
| `event: llm-orch.status` + JSON | May be ignored, surfaced raw, or treated as an OpenAI payload; behavior varies | Inspected parsers do not expose a stable status hook | Opt-in only |
| Empty OpenAI chunk with custom `llm_orch` field | Can fail strict schemas; if accepted, usually no visible content | No reliable display hook; may be treated as a no-op or error | Do not use |
| Response headers | Safe and visible to custom HTTP clients only; fixed when headers are sent | Useful to an integration that reads headers, not the normal TUI | Add request ID/state headers where possible |
| Separate authenticated status endpoint | Does not affect the OpenAI stream | A custom Pi/OpenCode integration can poll and render it | Add as an optional control-plane API |
| Assistant text such as “waiting…” | Displays everywhere | Corrupts the model response/transcript and tool protocol | Never use |

#### Recommended wire behavior

For `stream: true`, return the SSE response as soon as the request is admitted
to a queue, rather than waiting for `acquire_for_candidates` before constructing
the response. The response stream has two phases:

1. While waiting, emit only standards-compliant comment heartbeats at
   `queue_heartbeat_secs` (default 10 seconds), with no `data:` payload and no
   `[DONE]` marker. Comments may include a human-readable phase, queue position,
   selected model, and migration ID, but clients that do not understand them
   ignore them safely.
2. Once a slot is assigned, stop the wait-heartbeat timer and splice in the
   normal backend SSE stream. Preserve the backend’s normal OpenAI chunks and
   terminal marker exactly; emit no synthetic assistant text.

Set the usual streaming headers (`Content-Type: text/event-stream`,
`Cache-Control: no-cache`, and, when behind nginx, `X-Accel-Buffering: no`) and
ensure each heartbeat is flushed as a separate chunk. The heartbeat interval
must be comfortably less than the smallest configured client/proxy idle timeout;
10 seconds is a reasonable default but not a guarantee for an unknown proxy.
The existing axum `Sse::keep_alive` is too late by itself because it begins only
after slot acquisition; the waiting stream needs its own pre-assignment timer.

For an explicitly negotiated extension, accept a request header such as:

```text
X-LLM-Orch-Wait-Events: 1
```

Only when this header is present should the server send named
`event: llm-orch.status` frames. Keep them out of the default OpenAI stream. The
payload should be versioned and non-secret, for example:

```text
event: llm-orch.status
data: {"version":1,"state":"waiting_for_migration","model":"M2","queue_position":2,"phase":"loading_replacement"}

```

This is intended for a purpose-built adapter, not as a claim that the stock Pi
or OpenCode clients will render it. A custom adapter can set the opt-in header,
listen for the named event, and translate it into a local “waiting for GPU/model
migration” UI state. It must also ignore comments and continue parsing ordinary
OpenAI data events. Do not put these status objects into `data:` by default,
because the inspected OpenCode Chat schema and some OpenAI-compatible parsers
expect every data event to be a ChatCompletionChunk.

Add `X-LLM-Orch-Request-ID` to the initial response headers when a streaming
response is created, and expose an authenticated `GET /v1/requests/{id}` (or
admin-equivalent) status endpoint for integrations that can poll. The endpoint
can report `queued`, `waiting_for_migration`, `loading_replacement`,
`waiting_for_drain`, `loading_target`, `assigned`, `running`, `completed`,
`expired`, and `cancelled`. This side channel is the only clean way to offer
rich status to a client that cannot consume custom SSE events.

For `stream: false`, there is no compatible way to write an interim heartbeat
body: whitespace/comments would make the body invalid JSON, and HTTP headers
cannot be changed after the response starts. Keep the request future open and
return one valid JSON response when assigned, but document that an external
proxy/client with a hard total or read-idle timeout can still disconnect. Clients
that need reliable long waits should request streaming mode or use the optional
status/job interface. This is a protocol limitation, not something the scheduler
can solve without changing the API contract.

#### Sources

- [OpenAI streaming responses guide](https://developers.openai.com/api/docs/guides/streaming-responses)
- [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
- [Pi OpenAI Completions provider](https://github.com/earendil-works/pi/blob/main/packages/ai/src/api/openai-completions.ts)
- [Pi OpenAI Responses provider](https://github.com/earendil-works/pi/blob/main/packages/ai/src/api/openai-responses.ts)
- [OpenCode OpenAI Chat protocol](https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/protocols/openai-chat.ts)
- [OpenCode OpenAI Responses protocol](https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/protocols/openai-responses.ts)
- [OpenAI Node SSE implementation](https://github.com/openai/openai-node/blob/master/src/core/streaming.ts)

### 5. Write the implementation blueprint and validation plan
- [x] Provide proposed configuration fields, defaults, examples, and backward-compatibility rules for queue waiting, migration/drain deadlines, and the eviction-policy extension point while using LRU as the current policy.
- [x] Map the design to concrete Rust modules/functions and list deterministic unit/integration tests for victim selection, migration success/failure, queued-request reassignment, heartbeats, cancellation, deadlines, fairness, hot reload, and graceful shutdown.
- [x] Record observability requirements: queue/migration states, selected victim and destination, wait duration, heartbeat/status activity, failures, and metrics needed to diagnose starvation or VRAM thrashing.

#### Proposed configuration

Keep the existing per-model `queue_depth` and alias make-room settings. Add these
server-level settings in a backwards-compatible way:

```yaml
server:
  queue_wait_timeout_secs: 3600       # accepted request's maximum queue wait
                                      # 0 means no scheduler deadline
  queue_heartbeat_secs: 10            # SSE wait-comment cadence
  migration_timeout_secs: 0           # 0 means no transition deadline
  eviction_policy: lru                # only lru is implemented initially

models:
  - name: small-a
    queue_depth: 32
    # Existing model lifecycle/VRAM fields remain unchanged.
    vram: 3000
  - name: small-b
    queue_depth: 32
    vram: 3000
  - name: large
    vram: 9000

aliases:
  - name: preferred-large
    targets: [large]
    make_room: migrate             # proposed explicit migration mode
    drain_timeout: 300              # retain for legacy drain_surplus semantics
```

The exact unit of `vram` remains the existing declared MB reservation. The
scheduler continues to account by PCI slot and conservatively charges a
multi-GPU instance on every occupied GPU.

Field semantics:

- `queue_wait_timeout_secs` defaults to 3600 seconds. `0` is the explicit
  unlimited mode. The deadline is per accepted request and includes all waiting
  for capacity, migration, and model loading, but excludes time after slot
  assignment. It never permits killing an active request.
- `queue_heartbeat_secs` defaults to 10 seconds and applies to the pre-assignment
  streaming response. `0` should be rejected rather than silently producing an
  idle connection. The value should be configured below any known proxy read
  timeout.
- `migration_timeout_secs` defaults to `0` for the first implementation: a
  migration may continue while active victim requests finish. If a finite value
  is later selected, expiry may cancel/roll back only before destructive handoff;
  it must never kill the old backend or an active request. Client queue expiry
  remains governed by `queue_wait_timeout_secs`.
- `eviction_policy: lru` makes the current behavior explicit. Future values may
  add `priority_lru` or another policy, but unknown values must fail config
  validation. `ModelConfig.priority` remains reserved until a policy actually
  uses it.
- `make_room: migrate` is the proposed opt-in for the new two-phase behavior.
  Existing `none`, `evict_idle`, and `drain_surplus` keep their current meaning
  for old configurations. An implementation can initially enable migration only
  through a global/default policy if adding the alias enum is too disruptive,
  but direct model requests must not unexpectedly start evicting in v1.

Unknown YAML fields continue to be ignored as today, but new known fields must
be validated atomically during hot reload. Existing configurations without the
new fields retain the current queue and eviction behavior except for any
explicitly chosen new default; the queue timeout default should be documented as
the intentional change from the old short fixed timeout.

#### Future implementation blueprint

**`src/config.rs`**

- Add `queue_wait_timeout_secs`, `queue_heartbeat_secs`,
  `migration_timeout_secs`, and `eviction_policy` to `ServerConfig` with the
  defaults above.
- Add a migration-capable make-room enum or extend the existing alias policy;
  preserve deserialization of current `none`, `evict_idle`, and `drain_surplus`
  values.
- Validate nonzero heartbeat, legal timeout values, and known eviction-policy
  names. Keep validation independent of live GPU state so `--check-config`
  remains deterministic.

**`src/instance.rs`**

- Split the conceptual `Failed`/draining overload into explicit `Draining`,
  `Retired`, and `Failed` states, or introduce an equivalent migration metadata
  field without weakening the current removal guarantees.
- Store a routing generation, migration ID, and a separate last-completion
  timestamp. Keep `SlotGuard` as the sole owner of an active request slot.
- Make `try_acquire` reject `Draining` atomically. Existing guards continue to
  own the old instance until normal completion.

**`src/scheduler.rs`**

- Add an admission/migration coordinator and a reservation table. All placement
  checks, victim transitions, temporary duplicate loads, and commits must pass
  through it. Keep the existing global spawn serialization as a lower-level
  load guard, not as the transaction protocol.
- Refactor `WaitQueue` into logical `QueueEntry` records with model generation,
  migration phase, deadline, cancellation, and an assignment/error channel.
- Refactor `select_gpus_inner`/`spawn_feasible` to include coordinator-owned
  reservations and to return a concrete placement, not only a boolean/list.
- Implement `plan_migration` and `run_migration` as phase-driven operations:
  reserve → barrier victim → load replacement → wait old slots to zero → commit
  handoff/remove old → load large target → wake queues. Every phase transition
  must be idempotent so crash/reload cleanup can safely retry it.
- Retain `ensure_spawn`, `reap_drained`, and `unregister_instance` where their
  ownership guarantees fit. Do not reuse `remove_instance(IfIdle)` for a
  migration handoff unless the reservation/transition lock is held and the
  logical route is already blocked.
- Rework autoscale and idle reaping to skip migration participants and release
  only reservations that belong to completed operations.

**`src/handlers.rs` and `src/server.rs`**

- For streaming requests, construct a response stream immediately after queue
  admission. The stream emits comment heartbeats while awaiting assignment,
  then forwards the normal backend SSE stream and its terminal marker.
- Preserve the existing OpenAI body/chunk format by default. Add the negotiated
  `X-LLM-Orch-Wait-Events: 1` extension only for clients that explicitly ask for
  named status events.
- Return `X-LLM-Orch-Request-ID` on an early streaming response and add an
  authenticated request-status route. Keep non-streaming requests as one valid
  JSON response; do not inject whitespace or status JSON into that body.
- Make queue cancellation observe axum request cancellation and remove the
  corresponding queue entry promptly.

**`src/types.rs` and `src/main.rs`**

- Add serializable queue/request/migration status types for `/v1/requests/{id}`
  and `/admin/status`.
- Extend release/spawn/migration notifications so handoffs wake the logical
  model queue, not only the old instance name. Keep shutdown cancellation
  separate from ordinary deadline expiry.
- Ensure the background tasks can reap abandoned reservations and that shutdown
  first stops new migrations, then handles queued waiters, then drains active
  requests under the existing bounded shutdown policy.

**`src/reload.rs`**

- Keep config validation and swap atomic. Reconcile active migrations against
  the new device/model fingerprints. Cancel only at a safe phase, release all
  reservations, and either restore the old route or queue against a validated
  replacement.

#### Algorithm sketch for the implementation

```text
on request R for logical target L:
    snapshot route policy and create request_id
    if a Ready slot for L exists and no migration barrier applies:
        assign slot and run R
    else:
        register R in the logical queue with deadline/cancellation
        if no compatible transition/load is already active:
            submit a placement demand for L

coordinator for a placement demand:
    serialize with the admission lock
    if L now fits, reserve L and start one deduplicated load
    otherwise enumerate L placements and LRU victims
    for each candidate plan:
        prove victim replacement fits at destination while old victim remains
        prove L fits at source after old victim removal
        reserve both peak and final placements atomically
        mark victim Draining; reject new victim slots
        start victim replacement load on destination
        wait for replacement Ready and old active_count == 0
        commit logical victim route to replacement
        remove old victim and release source reservation
        start/load L on the freed source placement
        commit L when Ready and wake queues
    on pre-handoff failure:
        unmark victim, release reservations, wake old route
    on post-handoff L failure:
        keep successful victim replacement; do not migrate it back
        leave/requeue L according to request deadlines and policy
```

The coordinator should re-check every precondition at each await boundary. A
successful dry-run is only a plan; reservations are what make it safe against
another request or spawn changing VRAM before the process actually starts.

#### Validation and test matrix

Deterministic scheduler/unit tests should cover:

1. **Placement arithmetic:** two 30% instances block a 90% load; a migration
   replacement fits alongside the other 30% instance; peak temporary usage is
   charged; final usage is 30%+30% on one GPU and 90% on the other.
2. **Victim policy:** oldest LRU wins; active count is a tie-breaker; stable IDs
   break exact ties; a victim with no legal destination is skipped; active
   victims are drained rather than killed; no migration selects a failed or
   already-transitioning instance.
3. **Safe handoff:** active victim guards remain usable; new victim requests are
   queued; replacement readiness plus old count reaching zero switches the
   queue to the new instance; queued requests never target the retired process.
4. **Failure paths:** destination readiness failure restores the old route and
   releases reservations; source removal races are idempotent; target-load
   failure keeps a successful victim migration; stale loading reservations are
   reclaimed.
5. **Concurrency:** simultaneous large-model requests coalesce into one
   migration/load; simultaneous requests for the victim join one FIFO queue;
   unrelated model requests continue; no reservation double allocation or
   deadlock occurs.
6. **Queue lifecycle:** FIFO assignment, queue-full rejection, finite deadline,
   unlimited mode, prompt cancellation cleanup, deadline-versus-wake race,
   migration wakeup, model removal, config reload, and shutdown cancellation.
7. **Streaming protocol:** heartbeats begin before backend assignment, occur at
   the configured cadence, are SSE comments, do not create assistant content,
   stop after assignment, preserve normal chunks and `[DONE]`, and survive a
   long model load in a client/proxy test. Opt-in status events are absent by
   default and emitted only with the request header.
8. **Non-streaming protocol:** a queued request eventually returns exactly one
   valid JSON response; expiration returns the documented HTTP error rather
   than malformed partial JSON.
9. **Existing regressions:** current stub-backend integration coverage for
   aliases, slot guards, spawn deduplication, crash handling, autoscaling,
   atomic reload, backend timeouts, and graceful shutdown remains passing.

#### Observability and operations

Expose these fields in `/v1/info`, `/admin/status`, and the request-status route
where appropriate:

```text
request_id, logical_model, resolved_model, queue_state, queue_position,
enqueued_at, wait_duration, deadline, migration_id, migration_phase,
victim_instance, victim_gpu, destination_gpu, target_gpu, active_victim_slots,
reservation_peak_vram, reservation_final_vram, failure_reason
```

Add structured events for:

- migration planned, reservation acquired, victim barrier installed;
- replacement load started/ready/failed;
- victim drain started/completed/timed out;
- logical handoff committed and old GPU released;
- target load started/ready/failed;
- queue enqueue, dequeue, cancellation, deadline expiry, and shutdown;
- reservation cleanup after crash or config reload.

Recommended metrics include queue depth by state, queue wait duration
(histogram), number of expired/cancelled waiters, migration count and duration
by phase, migration failures/rollbacks, victim active-slot count at selection,
peak-vs-final reserved VRAM, model load duration, and per-model starvation or
head-of-line wait time. Include migration IDs and request IDs in logs, but never
include prompts, API keys, or other request contents in status payloads.

Alert-worthy conditions are a reservation leak, a migration stuck past its
configured deadline, repeated destination-load failure, queue depth growing
while no transition is progressing, and repeated migrations of the same models
(ping-pong/thrashing).

## Implementation plan (completed source-code work)

This was the actionable implementation sequence derived from the design above. All
phases are complete and were verified with the source changes and focused tests.

### Phase 0 — Freeze the first implementation contract
- [x] Enable migration through the explicit alias value `make_room: migrate`; do not enable it by a server-wide default or for direct model requests in v1; preserve existing `none`, `evict_idle`, and `drain_surplus` behavior for existing configurations.
- [x] Freeze the wire contract for queueing: `server.queue_wait_timeout_secs` default `3600`, `0` for unlimited scheduler wait, `server.queue_heartbeat_secs` default `10`, and the opt-in `X-LLM-Orch-Wait-Events: 1` header.
- [x] Freeze terminal HTTP behavior: queue-full remains 429 with `Retry-After`; an accepted wait that reaches its deadline returns a documented 504 queue-timeout error; shutdown may allow queued handlers to remain during the configured HTTP drain period rather than waiting for their queue deadlines.
- [x] Freeze the migration invariant in a testable form: active requests remain on the old backend; new victim-model requests queue; replacement readiness precedes old removal; the large model is loaded only after the source reservation is released.

### Phase 1 — Add scheduler state and resource reservations
- [x] Add explicit conceptual instance states/metadata for `Draining`, migration loading, generation, migration ID, and a separate last-completion timestamp, while preserving `SlotGuard` ownership of active slots.
- [x] Implement a scheduler-owned reservation table keyed by transition ID and PCI slot, including purpose, phase, declared VRAM, and cleanup ownership.
- [x] Add one serialized admission/migration coordinator task/command loop as the sole owner of placement reservations and structural transitions; ordinary spawn, autoscaling, admin load, migration, and reload placement operations must submit through it, while the existing spawn semaphore remains only a lower-level process-load guard.
- [x] Add deterministic placement snapshots that return a concrete GPU placement and projected peak/final usage, using the existing PCI-slot and multi-GPU accounting rules; acquire reservations before any await that can start a competing load.
- [x] Add reservation cleanup paths for spawn failure, backend crash, client cancellation before handoff, config reload, timeout, and shutdown.

### Phase 2 — Add configuration and status types
- [x] Add and validate server queue/migration timeout fields and the explicit LRU eviction-policy enum with backward-compatible defaults.
- [x] Add migration-capable alias policy/configuration while retaining compatibility with current scalar alias targets and existing make-room modes.
- [x] Add serializable queue, migration, and request-status types, including request ID, logical/resolved model, phase, queue position, deadline, victim, destination, and failure reason.
- [x] Extend config-check and hot-reload tests for defaults, invalid zero/negative-equivalent values, unknown policy values, and atomic rejection.

### Phase 3 — Implement migration planning and execution
- [x] Implement LRU victim enumeration: idle first, then busy; deterministic completion-time/active-slot/instance-ID tie-breaks; exclude failed, already-transitioning, and destination-ineligible instances.
- [x] Implement a dry-run that proves both sides of the two-phase plan: victim replacement fits while the old victim remains loaded, and the large target fits after the old victim is removed.
- [x] Implement the migration phases `reserve → mark draining → load replacement → wait replacement ready → wait old active count zero → commit logical handoff → remove old instance → load large target → wake queues`.
- [x] Ensure the victim’s replacement may temporarily exceed its ordinary logical instance cap only under the owning migration reservation, and is not routable before readiness.
- [x] Implement pre-handoff rollback that restores the old route when replacement loading fails, and post-handoff failure handling that keeps a successful victim migration instead of immediately migrating it back.
- [x] Coalesce concurrent demands for the same logical target/placement onto one migration and one target spawn; unrelated model requests must continue normally, and a failed target load must not immediately trigger migration back or victim ping-pong.

### Phase 4 — Replace the slot queue with a migration-aware queue
- [x] Replace instance-bound waiters with logical `QueueEntry` records containing resolved route snapshot, model generation/barrier, enqueue time, deadline, assignment channel, and cancellation ownership.
- [x] Expose a scheduler admission operation that registers the queue entry before starting slow spawn or migration work, returns an assignment future/ticket, and submits a coalesced demand even when the model has zero routable instances; streaming handlers must not synchronously await cold-start, spawn, or migration work before returning their response.
- [x] Preserve FIFO order and queue-depth enforcement; distinguish waiting-for-slot, waiting-for-migration, waiting-for-spawn, and terminal states in metrics/status.
- [x] Make client cancellation remove a queued entry promptly and safely release a guard if cancellation races assignment.
- [x] Wake queues on slot release, spawn readiness, replacement readiness, handoff commit, migration rollback, target readiness, config reconciliation, and model removal when a valid fallback exists.
- [x] Ensure queue deadlines affect only waiting requests and never cancel active requests, shared migrations, or backend processes.

### Phase 5 — Add pre-assignment streaming and heartbeats
- [x] Refactor all applicable streaming handlers (Chat Completions, Responses, Completions, and explicit `stream: true` audio requests) so they can return an SSE response before slot assignment and wait inside the response stream for the scheduler assignment.
- [x] Emit only flushed SSE comment heartbeats before assignment, at the configured interval; never emit assistant text, fake OpenAI chunks, or `[DONE]` during the wait.
- [x] Splice the assigned backend stream into the same response, preserving normal OpenAI Chat/Responses event ordering, usage events, error handling, and terminal markers; if the backend cannot be contacted or returns an error after the early SSE response has started, emit the documented SSE error event and release the assignment instead of attempting to change the HTTP status.
- [x] Add `Cache-Control: no-cache`, `X-Accel-Buffering: no`, request ID headers, and any required proxy-flush behavior; bypass or configure the existing compression layer so SSE comments are not buffered, and verify flushing through the supported reverse-proxy setup.
- [x] Add the opt-in named status-event stream only behind `X-LLM-Orch-Wait-Events: 1`; default OpenAI-compatible streams must contain comments plus normal provider data only.

### Phase 6 — Add request status and non-streaming behavior
- [x] Add an authenticated request-status endpoint and expose equivalent aggregate migration/queue information through `/v1/info` and `/admin/status` without exposing prompts or API keys; use a bounded in-memory request registry with documented terminal-state retention and include the request ID on final/error responses so non-streaming clients can discover it.
- [x] Keep non-streaming requests open until assignment or configured expiry, returning exactly one valid JSON response or one documented terminal error; do not inject heartbeat bytes into JSON.
- [x] Document that external proxy/client total or idle timeouts can still terminate non-streaming waits, and recommend streaming mode or the status interface for long waits.
- [x] Verify Pi/OpenCode compatibility with an end-to-end SSE parser test: comments are ignored, normal chunks remain parseable, and opt-in status events are not sent accidentally.

### Phase 7 — Integrate lifecycle, reload, autoscaling, and shutdown
- [x] Make autoscaling and idle reaping skip migration participants, reservations, and logical models behind an active handoff.
- [x] Make hot reload reconcile or safely cancel migrations, release old reservations, and preserve accepted requests’ route snapshots without partial application.
- [x] Make crash/readiness monitoring distinguish migration replacement failure from old-victim failure and wake or fail queues deterministically.
- [x] During shutdown, stop accepting new work and stop initiating new migrations; allow existing queued/running handlers to remain during the configured HTTP drain period so client cancellation or normal queue deadlines can resolve them, then abort remaining HTTP tasks, clean up queued entries and transition reservations, and force-remove backends. This bounded shutdown path is the explicit exception to the ordinary no-interruption migration invariant.

### Phase 8 — Test, observe, and document the implementation
- [x] Add unit tests for 30%/30%/90% placement arithmetic, temporary peak reservations, LRU selection, active-victim draining, tie-breaks, no-destination rejection, and reservation cleanup.
- [x] Add integration tests for successful migration/handoff, queued victim requests, replacement and target load failures, concurrent-demand coalescing, cancellation, deadlines, config reload, and graceful shutdown using the stub backend/fake GPU snapshot.
- [x] Add streaming tests for pre-assignment heartbeats, flush cadence, comment-only compatibility, opt-in status events, preserved `[DONE]`, client disconnect, and non-streaming valid JSON.
- [x] Add structured logs and metrics for migration phases, reservation peak/final VRAM, victim/destination/target devices, queue state/position/wait duration, cancellation/expiry, rollback, and thrashing.
- [x] Update `README.md`, configuration examples, API/status documentation, and operational guidance after the implementation passes the existing build, test, fmt, and clippy checks.

## Notes
- The design/research and implementation tasks above are complete. This todo records the delivered migration, queue, status, observability, and documentation changes.
- Migration is opt-in through alias `make_room: migrate`; direct model requests do not initiate migration in v1. Existing make-room modes retain their current behavior.
- The single coordinator task/command loop owns placement reservations and structural transitions. It must reserve before awaiting loads and must prevent ordinary spawns, autoscaling, admin operations, and reload reconciliation from bypassing those reservations.
- Streaming admission must queue/register the request before slow spawn or migration work so pre-assignment heartbeats can begin even on a cold start. SSE backend failures after the early response begins are represented as SSE errors, and compression must not buffer heartbeats.
- Shutdown is allowed to be delayed by existing request handlers up to `shutdown_drain_timeout_secs`; clients are expected to cancel requests where appropriate. After that bound, remaining HTTP tasks and backends may be forcefully terminated and queue/reservation cleanup must still be deterministic.
- Request statuses retain terminal records for 15 minutes with a 4096-entry bound. Hot reload advances a configuration epoch: transitions roll back before handoff when stale and retain a successful victim handoff afterward.
- Confirmed behavior: existing requests on the victim continue; new requests for it wait; the victim is re-uploaded to the other GPU where possible; the freed GPU then loads the large model.
- The current implementation already has drain/reap primitives and a per-model queue, but its queue wait is bounded by a much shorter fixed timeout. The implementation should reuse those primitives where safe.
- The current eviction policy is LRU for now. The implementation should leave a clean configuration extension point for priorities/policies without making priority part of the first implementation.
