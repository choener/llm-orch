# Smart handling: multi-model aliases and make-room eviction

Status: **implemented**. Line-number references in "Current state" below are
pre-implementation pointers; the mechanics sections describe the shipped
behavior except where "Deviation" notes say otherwise.

Related: `docs/002-gpu-selection.md` already sketched the drain-then-evict cycle
for full GPUs ("`9` is requested while `1`/`2` occupy both cards"); this document
generalizes that idea and ties it to aliases. It also covers the eviction side of
the `docs/TODO.md` item about duplicate unloading.

## Problem statement

An alias today maps 1:1 to a model (`AliasConfig.target`). That is too rigid when
VRAM is the scarce resource:

- The GPU(s) can hold **one dense + one MoE** model, or **two of either**.
- Sometimes two dense instances are loaded. A client (e.g. firecrawl) wants the
  MoE via an alias. The MoE does not fit next to two dense models, the spawn is
  refused, and the request fails — even though the MoE *would* fit if one dense
  instance were unloaded.
- There is currently **no make-room mechanism at all**: when
  `select_gpus_for_model` cannot satisfy a spawn, the request falls through to a
  queue that fails fast (`NoInstances` → `AcquireError::Unavailable`).

So this feature is really two coupled features:

1. **Multi-model aliases**: an alias names an ordered list of candidate models.
2. **Make-room eviction**: when the preferred candidate does not fit, evict or
   drain *other* instances to free VRAM — including busy surplus duplicates.

## Current state (code pointers)

- `AliasConfig { name, target, system_prompt, prompt_template }` —
  `src/config.rs:629`.
- `resolve_alias()` — `src/handlers.rs:1847`, pure config lookup, single target.
- `apply_alias_prompts()` — `src/handlers.rs:1862` (to be removed, see below).
- Request path: `get_or_spawn` — `src/scheduler.rs:440`
  (fast path → autoscale gate → `ensure_spawn`/`try_spawn` → queue).
- Spawn refusal when the device pool can't be satisfied —
  `src/scheduler.rs:862`.
- Eviction primitives that already exist: idle removal (`remove if idle`,
  `src/scheduler.rs:1280`), drain-then-reap for retiring instances
  (`Failed` state + `reap_drained`, `src/scheduler.rs:1361`).
- Queueing is per-model (`enqueue`, `src/scheduler.rs:640`).

## Decisions taken

1. **Make-room eviction is in scope.** Without it the motivating use case
   (MoE wanted while two dense are loaded) still fails.
2. **The target list is an explicit preference ordering.** Later entries are
   spillover for when an earlier entry cannot be loaded or served.
3. **Busy instances may be drained, but only surplus duplicates**: a busy
   instance is a drain candidate only if its model has more than one loaded
   instance (and `max_instances > 1` permits the duplicate to exist at all).
   The last serving instance of a model is never drained — a make-room must
   not take a model fully offline for its own in-flight traffic.
4. **Policy is a single enum, not composable steps.** The fixed selection
   algorithm below has one knob (`prefer_loaded` vs `prefer_order`).
   Explicit per-alias step composition was considered and rejected as
   over-engineering.
5. **`make_room` defaults to `evict_idle`.** Nothing in flight is ever
   interrupted at that level; draining busy surplus duplicates is opt-in via
   `drain_surplus`.
6. **Eviction-free fits beat ordering.** Pass B first looks for *any*
   candidate that fits without make-room (in order), and only then runs
   make-room rounds (again in order). Rationale: keep requests moving;
   evicting for `c1` while `c2` fits for free is wasteful.
7. **Drain waits are bounded by a dedicated per-alias `drain_timeout`,
   default 300s.** Not `spawn_timeout`: that one is hardcoded (120s,
   `src/scheduler.rs:406`) and measures model *load* time, while a drain
   wait is driven by *client stream lengths*, which run for many minutes.
   On timeout the plan for that candidate is abandoned, the loop continues
   (next candidate, then queue); the draining instance stays draining and
   is reaped once idle — nothing to roll back.
8. **Eviction protection splits staleness from reaping.** The per-model
   `idle_ttl` becomes a *protection window* only: an instance idle for
   ≥ its model's `idle_ttl` stops counting as protected (make-room may
   take it) but is **not** automatically reaped anymore. A new global
   `drain_idle_timeout` (default 3600s) is the *actual reaper*: any
   instance idle past it is unconditionally evicted, keeping GPUs able to
   deep-sleep even when no make-room pressure ever arrives. This replaces
   today's TTL-driven active eviction (surplus: `src/scheduler.rs:2010`,
   last instance: `src/scheduler.rs:1975`). Models kept around longer
   (e.g. vision at 600s) automatically get longer make-room protection,
   with no per-model evictability knob. For busy surplus duplicates there
   is no idle time by definition; ordering is least-in-flight, and the
   thrash guard is the `drain_timeout` bound plus "never the last busy
   instance".
9. **Direct (non-alias) requests get no make-room in v1, but share the
   code path.** Internally every request is a candidate list: a direct
   model request is a one-element list with `make_room: none` — exactly
   today's behavior, no new policy surface. Aliases use the full
   multi-candidate path. A global `make_room` default for direct requests
   (doc 002's "load `9` over `2`") can follow later without restructuring.
10. **`system_prompt` / `prompt_template` are removed** from `AliasConfig`.
   They were never used; dropping them simplifies the handlers (no
   `apply_alias_prompts`) and makes aliases purely a routing concern.
   Since `serde` ignores unknown YAML keys by default, old configs keep
   loading; the keys just become inert.

## Configuration schema (proposed)

```yaml
aliases:
  - name: summarize
    targets: [qwen3-moe, qwen3-27b]   # ordered; later entries are spillover
    policy: prefer_loaded             # prefer_loaded (default) | prefer_order
    make_room: evict_idle             # none | evict_idle (default) | drain_surplus
    drain_timeout: 300                # bound on waiting for a busy duplicate
                                      # to drain; default 300s

# global (server-level) knob:
drain_idle_timeout: 3600              # unconditional reap of idle instances
                                      # after this many seconds (deep sleep)
```

- Backward compatibility: scalar `target: <name>` is accepted as sugar for a
  one-element `targets` list (untagged/string-or-seq deserialization).
- Validation: non-empty list; every target must be a known `ModelConfig.name`;
  aliases still cannot target aliases (no chaining); duplicate alias names and
  empty names remain rejected as today.

### The `policy` knob (the "first loaded vs. first fitting" behaviour)

Example: `targets: [moe, dense]`, and a `dense` instance is already loaded and
idle when a request arrives.

- `prefer_loaded` (default): serve from the warm `dense` immediately — zero
  latency, zero eviction churn, but you get the less-preferred model.
- `prefer_order`: ignore the warm `dense`; try to load `moe`, making room if
  needed — you get the best model the hardware can hold, at the cost of load
  latency and evictions. Later entries are used only as spillover when an
  earlier entry truly cannot be loaded.

## Selection algorithm

Request arrives for alias with candidates `C = [c1, c2, ...]` (config order).

### `prefer_loaded`

- **Pass A (warm slot):** scan `C` in order; first candidate with a `Ready`
  instance below `max_concurrent` → acquire slot. Done.
- **Pass B (load):** first scan `C` in order for a candidate that is
  spawn-feasible **without eviction** (dry-run of `select_gpus_for_model` —
  must use the *same* VRAM accounting as `try_spawn`, which deliberately
  includes retiring instances) and whose per-model autoscale gate permits →
  spawn, acquire. Only if *no* candidate fits on its own: scan `C` in order
  again, and for the first candidate where `make_room` allows and an eviction
  plan exists (below) → evict/drain, spawn, acquire. Splitting Pass B into
  these two rounds avoids evicting for `c1` when `c2` would have fit for
  free — eviction is a last resort, not a tie-breaker.
- **Pass C (wait):** enqueue on the first *loaded* candidate in `C`. If no
  candidate is loaded, enqueue on `c1` (which fails fast with `Unavailable`
  if nothing could ever wake it — current behaviour).

### `prefer_order`

Per candidate, in order: warm slot → spawn-if-fits → make-room-and-spawn.
Only advance to the next candidate when all three fail. After the loop,
Pass C as above.  (Implemented exactly like this: the global Pass A warm
scan is skipped — a later candidate's warm slot must not preempt loading
the preferred one.)

### Notes

- Per-candidate spawns reuse the existing per-model machinery
  (`ensure_spawn`, autoscale gates, spawn semaphore, instance caps). The
  multi-target logic only *chooses* the candidate.
- No wake-up re-evaluation in v1: once queued on a candidate, a request stays
  queued there even if another candidate frees up first. Documented
  imperfection; revisit if it hurts in practice.

## Make-room eviction

Goal: free enough VRAM on a suitable device set so candidate `ci` fits.

### Victim ordering (most to least preferred)

1. **Idle surplus duplicates** — any instance, of any model, that is idle
   (no in-flight requests) while its model has ≥ 2 loaded instances.
   Removing it cannot take a model offline. Eligible only once idle for
   ≥ its model's `idle_ttl` (decision 8).
2. **Idle last instances** — an idle instance that is the only one of its
   model. This *does* take the model offline, but nothing is in flight.
   (This is where `lazy-unload` instances — TODO.md item 4 — get reclaimed:
   make-room is the sanctioned pressure that unloads a lazy-kept model.)
3. **Busy surplus duplicates** — only when `make_room: drain_surplus`:
   mark the instance draining (the existing `Failed`/retire mechanism),
   wait for its in-flight requests to finish, reap, then spawn. The model
   keeps serving through its remaining instance(s). Note: a *busy*
   instance has no idle time, so the decision-8 staleness rule does not
   apply to this class — least-in-flight is the ordering.

Never a victim: the last **busy** instance of any model; busy duplicates of
the *requesting* model itself (draining those would be pointless churn);
`Loading` instances (cancelling another spawn is v2 material).

Deviation (implemented): the own-model exclusion for class 3 was made
explicit during implementation — it is listed above.

Within each class: least-recently-used first (last request completion).

### Interaction with the candidate loop

Make-room runs only in the *second* round of Pass B (once every candidate
failed to fit on its own): try to assemble a victim set for `c1` first (in
victim-class order), and only if no plan frees enough VRAM move on to `c2`,
etc. This keeps "explicit ordering, later = spillover" intact where
evictions are concerned — we would rather evict for `c1` than settle for
`c2` — while never evicting when a spillover candidate fits for free.

### Bounded draining

Draining a busy duplicate can take minutes (long streams). The wait is
bounded by the per-alias `drain_timeout` (default 300s — decision 7): if
the drain does not complete in time, abandon the plan for this candidate
and continue the selection loop (next candidate, or queue). The draining
instance is left draining — it will be reaped once idle regardless;
nothing is rolled back.

### Thrashing protection

Without hysteresis, two aliases can ping-pong (A evicts B, next request for B
evicts A) with multi-minute load times each way. Mitigations:

- Idle instances are protected for their model's `idle_ttl` (decision 8) —
  a model anyone used recently is untouchable, which breaks the ping-pong
  for the common case.
- Busy-duplicate drains are bounded by `drain_timeout` and never take the
  last busy instance.
- If ping-ponging still shows up in practice, a follow-up could prefer
  evicting models that are not any alias's first target — deliberately not
  in v1 (over-clever).

## Interactions with existing machinery

- **Autoscale:** unchanged, per-model. A multi-target request that picks `ci`
  goes through `ci`'s normal autoscale gate; surplus-instance spawning of the
  *other* candidates is unaffected.
- **Queueing:** stays per-model. Pass C queues on exactly one candidate.
- **Hot reload:** candidate lists, policy, and make-room mode swap atomically
  with the rest of the config (existing reload machinery). In-flight requests
  keep their resolved candidate.
- **Metrics / debug log:** unchanged per-model granularity — the *resolved*
  model is forwarded, as today. Add tracing for the decision path (candidates
  scanned, why a candidate was skipped, victims chosen, drain waits).
- **`/v1/models`:** aliases still listed once, by alias name. Alias info may
  gain the candidate list.
- **Lazy-unload (TODO.md item 4):** a lazy-kept last instance is an idle
  victim (class 2) once past its `idle_ttl` protection — this is
  intentional and required, otherwise lazy-unload would starve all other
  models. `drain_idle_timeout` bounds it from above as well.
- **Duplicate unloading (TODO.md item 3):** the "drain the less busy 27B, then
  load the MoE" behaviour *is* victim class 3 — this feature implements it for
  alias-driven requests.

## Open questions

None — all settled in v1 (decisions 1–10). Remaining follow-ups, post-v1:
global `make_room` default for direct requests; anti-ping-pong refinement
("prefer evicting non-first-target models") if thrashing shows up.

## Implementation notes (post-implementation)

- `AliasInfo` in `/v1/info` now carries `targets: Vec<String>` (the
  `target` / `has_system_prompt` fields are gone).  `/v1/models` alias
  entries take their `context_length` from the first target.
- `resolve_alias` returns `(candidates, policy, make_room, drain_timeout)`;
  unknown alias/model (no candidate configured) still maps to 404.
- All config knobs validated by `--check-config`; legacy scalar `target:`
  keeps working.

## Implementation outline

1. `src/config.rs`: rework `AliasConfig` (`targets`, `policy`, `make_room`,
   `drain_timeout`; accept scalar `target`; drop
   `system_prompt`/`prompt_template`); global `drain_idle_timeout`;
   validation for empty/unknown/chained targets.
2. `src/handlers.rs`: `resolve_alias` returns the candidate list; remove
   prompt injection; chat/completions/responses handlers call the new
   scheduler entry point.
3. `src/scheduler.rs`: `acquire_for_candidates(candidates, policy,
   make_room)` implementing Passes A–C; spawn-feasibility dry-run sharing
   `select_gpus_for_model`'s accounting; make-room planner (victim classes
   1–3) and executor (idle: existing idle-removal path; busy duplicate:
   mark `Failed`, bounded wait, `reap_drained`).
4. Reaper rework (decision 8): `idle_ttl` stops triggering active
   eviction; new `drain_idle_timeout` reap loop takes over unconditional
   idle eviction.
5. Tracing for decision paths.
6. Tests (stub backend, no GPU): warm-slot preference in alias order;
   skip-full-candidate → spawn fitting one; nothing fits → idle eviction →
   spawn; surplus-drain path; queue fallback on first loaded candidate;
   config validation (empty list, unknown target, alias→alias); scalar
   `target` back-compat; hot-reload swap; removal of prompt injection;
   `drain_idle_timeout` reaping incl. last instance; idle_ttl no longer
   evicting on its own.
7. Update this document and `docs/TODO.md` to match the implementation.
