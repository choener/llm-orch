Cluster peering: one llm-orch acting as a gateway over further llm-orch
instances. Star topology, two hosts today: a dense-model host and a MoE host.

# Problem statement

Two llm-orch instances on two hosts (small dense models / large MoE models).
Running them side by side means either two client-visible endpoints plus an
apikeys-sync problem, or a hand-maintained set of remote-model entries that
drifts when the leaf's model list changes. Peering makes the master a thin
shim: one client endpoint, one apikeys file for users, and a union view of
models and statistics.

# Topology and non-goals

- Star topology: one master (which may also serve local models) and N leaf
  peers. Leaves are ordinary llm-orch instances with their own configs,
  schedulers, and GPUs.
- The master is *not* special to a leaf: from the leaf's point of view the
  master is a single client holding a single apikey. No new leaf-side
  endpoints are required for the v1 feature set.
- Non-goals, accepted deliberately (two hosts, build only what is needed):
  - No cycle detection. Topology must be a star / DAG. If A peers with B and
    B peers with A, discovery loops. Cheap one-line guard nonetheless: skip
    and warn on any discovered model name that already starts with this
    instance's own `node_name` + `/`.
  - No mesh routing, no peer-to-peer failover, no multi-master.
  - No TLS between peers (same rule as everywhere: LAN / WireGuard, or nginx).
  - No per-key roles. Peer keys are ordinary apikeys on the leaf.

# Decisions taken

1. **Peers are config; discovered models are runtime state.** The `peers:`
   section is part of the (hot-reloadable, atomically validated) config. The
   per-peer model catalog fetched from a leaf is in-memory state behind a
   TTL, never written back into config.
2. **One flat peer key per peer** (`remote_auth: "Bearer {peer_key_X}"`,
   resolved from `key_files` exactly like remote models today). The leaf lists
   that one key in its `apikeys.txt`. User keys never leave the master;
   per-user auth and accounting happen only at the master.
3. **Node-qualified naming; client provider prefixes are opaque.** Each
   llm-orch instance has a `node_name` (the master here: `cheyenne`) and
   each peer a `name` (e.g. `cortana`). llm-orch never knows the
   client-side provider namespace — e.g. pi's `<modelprovider>/<optional
   node>/<model>` scheme where the provider is called `llm`; that `llm`
   never appears in any llm-orch config or response.
   - `<node>/<model>` — pinned: `<node>` is either the local node name
     (local model) or a configured peer.
   - Bare `<model>` — local model, or unique leaf model, or smart-routed
     overlap (see decision 5).
   - **Opaque prefix stripping**: if the first segment of an incoming
     `model` string is neither the local node name nor a peer name, it is a
     client-side provider prefix — strip it and resolve the remainder.
     This is what makes `llm/cortana/qwen-flash` (provider `llm`, node
     `cortana`, model `qwen-flash`) work without llm-orch knowing `llm`.
     Known node names always win: a first segment matching a node name is
     treated as a node, never stripped.
   - Overlapping names (model exists locally *and* on a leaf, or on two
     leaves): the bare name resolves local-first; leaf copies are
     reachable under `<peer>/<model>`.
   - `/v1/models` lists both bare and node-qualified names — `qwen-flash`,
     `cheyenne/qwen-flash`, `cortana/qwen-flash` — so pins are
     discoverable. Client provider prefixes never appear.
   - Node names are restricted to `[a-z0-9-]+` since they become part of
     model identifiers.
4. **Lazy TTL refresh, no background poller.** A peer's model catalog is
   refreshed on demand when older than `catalog_ttl_secs` (default 60):
   - before routing a request to a model of that peer;
   - when serving `/v1/models` (stale entries otherwise make the listing
     flap);
   - **before applying a config hot-reload** — alias targets may reference
     peer models, so the catalog is refreshed first, then the new config is
     validated against it. An unreachable peer falls back to its cached
     catalog (reload must not wedge on a dead leaf); a missing cached entry
     fails validation of that alias like any other unknown target.
   - on a 404 from a peer (model vanished between polls): refresh once, retry
     once, then surface the error.
5. **Bare-name overlap routing is minimal.** Local model wins if it exists;
   otherwise, if exactly one peer serves the name, route there. If several
   peers serve it: round-robin over healthy peers, tiebroken by reported
     queue depth from `/v1/info`, with a recent-failure penalty. No autoscale
   machinery on the shim (see decision 8) and no sticky sessions — leaves
   own cache-aware routing internally.
6. **Passthrough with surgical model rewriting.** Bodies are forwarded as
   opaque `serde_json::Value`s (the project's passthrough philosophy), with
   one documented exception: the top-level `model` string is rewritten from
   the master-side name to the leaf's real model name on the way out, and
   the response's top-level `model` is rewritten back. No other field is
   inspected or modified.
7. **Admin endpoints are passed through.** `/admin/load`, `/admin/unload`,
   `/admin/unblock`, `/admin/status` requests whose model maps to a leaf
   model are forwarded to that leaf (model name rewritten). `/admin/status`
   and `/v1/stats` on the *master* return the union view (see below).
   Consequence, accepted: the peer key grants full control of the leaf.
   Two trusted hosts on a private network — fine.
8. **The shim does not autoscale or admit-control leaf models.** Remote
   entries are provisioned by health probe exactly like today's remote
   models; validation already rejects `autoscale` for them (correct). The
   master's per-peer `max_concurrent` acts only as a circuit breaker; real
   queueing, admission (429s), and autoscaling happen on the leaf. The
   master's per-model queue settings for peer models should stay generous
   so requests are not double-held by both layers.
9. **Stats union, fetched on demand, bare-aggregated.** When the master's
   `/v1/stats` or `/v1/info` is queried, it fetches each peer's stats
   concurrently — no background schedule. Per-peer fetch timeout is
   **10 s**; on timeout or error the last-known values are served with an
   age annotation. Every subsequent query attempts a fresh fetch — no
   failure latch, no backoff (a leaf busy with one huge request must not
   be written off) — and concurrent queries share a single in-flight
   fetch per peer so slow leaves don't pile up parallel requests.
   Inference routing is never affected; only the stats endpoints wait.
   The response contains:
   - qualified entries: `cheyenne/<model>` (the master's own
     measurements), `cortana/<model>` (the leaf's reported numbers, labeled
     with the peer name and fetch timestamp);
   - bare entries: for each distinct model name, the **sum** over every
     node serving it (local + peers) — duplicated models add up.
   The master does *not* count forwarded requests into its own per-model
   metrics — the leaf's numbers are authoritative for leaf models;
   aggregation happens only at query time. Peer-state observability (last
   successful fetch, catalog age, in-flight count, recent errors) is
   included in `/admin/status`.
10. **Request-ID mapping.** The master mints its own request IDs (as
    today) and returns them to clients; for forwarded requests it keeps a
    mapping master-ID → leaf-ID and translates `/v1/requests/{id}` lookups
    to the leaf's status endpoint. Leaf-side IDs never surface as
    client-visible identifiers.
    Retention: entries live for `drain_idle_timeout_secs` after request
    completion (default 3600 — by that horizon even loaded models have
    been reaped, so there is nothing left to query) and are pruned lazily
    (checked on insert; no background sweeper). Memory is a non-issue:
    IDs are stored as 16-byte UUIDs, so a retained mapping costs on the
    order of ~100 bytes all-in — 100k retained mappings is ~10 MB, and
    the count is bounded by completed traffic within one drain window
    anyway. No size cap needed.
11. **Failure semantics.** Unreachable peer: catalog stays cached (stale
    beats empty; clients get an honest 502 on request instead of models
    disappearing from `/v1/models`), health-probe failure fails requests
    fast rather than hanging. The stats union degrades to last-known values
    with age annotations.

# Naming and resolution order

For an incoming `model` string (after apikey auth), segments split on `/`:

1. Exact local model name → local.
2. Alias (master config) → resolve target, restart resolution on the result.
3. `<node>/<model>` where `<node>` is the local node name → local model
   `<model>`.
4. `<node>/<model>` where `<node>` is a configured peer → pinned to that
   peer; its catalog must contain `<model>`.
5. `<x>/<rest...>` where `<x>` is not a known node name → client provider
   prefix: strip `<x>`, restart at 1 on `<rest...>` (applies to two- and
   three-segment strings alike, e.g. `llm/cortana/qwen-flash`).
6. Bare `<model>`:
   a. local model exists → local (local-first);
   b. exactly one peer serves it → that peer;
   c. several peers serve it → decision 5 routing;
   d. none → 404-style unknown-model error.

Everything clients see (`/v1/models`, aliases, error messages) uses
master-side names; leaf-side names never leak except inside rewritten
bodies, where they are required by the leaf's own routing.

# Configuration schema (proposed)

```yaml
node_name: cheyenne                # this instance's node name

peers:
  - name: cortana                  # [a-z0-9-]+, part of model names
    url: http://cortana.lan:8080    # leaf llm-orch base URL
    remote_auth: "Bearer {peer_key_cortana}"
    catalog_ttl_secs: 60           # lazy refresh threshold
    max_concurrent: 64             # shim-side circuit breaker
    include: []                    # optional model name patterns to re-export
    exclude: []                    # optional patterns to skip
```

A pure shim has `models: []`; local models plus `peers:` in one config is
the intended common case (the master host also serves its dense models).

# Implementation notes (post-implementation)

Deviations and refinements discovered during implementation:

1. **Nested-qualification guard.** A leaf's `/v1/models` includes its own
   node-qualified names (the client-facing listing). To stop a master from
   re-exporting them as `cortana/cortana/qwen-flash`, every `/v1/models`
   response carries an `x-llm-orch-node` header; the catalog fetch filters
   the peer's self-qualified entries.
2. **Stats/info self-qualification is gateway-only.** The node-qualified
   copies of local entries in `/v1/stats` and `/v1/info` are emitted only
   when peers are configured (a gateway's union view). A peer-less node
   emits bare-only responses, so a consuming master re-exports each entry
   exactly once.
3. **Alias target validation stays offline.** Peer-qualified alias
   targets are validated structurally (the peer must be configured), not
   against the live catalog — so `--check-config` works without network
   access. A stale target surfaces as a request-time 404.
4. **Bare-name overlap routing uses the alias machinery.** A bare name
   served by several peers materializes as an alias over the qualified
   names with the existing `prefer_loaded` policy, not a custom
   round-robin — the existing candidate machinery covers the two-host
   scale; the queue-depth tiebreak can be added later without config
   changes.
5. **Response `model` carries the resolved name.** A bare `qwen-flash`
   request answers with `model: cortana/qwen-flash` — the model that
   actually served it — consistent with pre-existing alias semantics.
6. **404 recovery is scheduled, not inline.** A 404 from a leaf schedules
   a background catalog refresh + rematerialization; the failing request
   still errors. The next request recovers.
7. **The request-ID map stores the model name** alongside (peer, leaf
   id) so the proxied `/v1/requests/{id}` response is rewritten to the
   master's request ID and master-side model names.

# Implementation outline

1. `PeerConfig` + `node_name` in `src/config.rs`; validation (node and
   peer name charset, URL scheme http/https, key file references, peer
   names distinct from the local node name).
2. Peer catalog task per peer: fetch `/v1/models` with the peer key;
   cache behind `catalog_ttl_secs`; lazy refresh triggers as in decision
   4. Reuses `http_client`.
3. Route resolution in the request path (naming table above) feeding the
   existing remote-instance machinery: materialize per-model remote entries
   from the catalog (`try_provision_remote` and forwarding stay untouched),
   remembering the leaf-side name for the rewrite.
4. Body/response `model` rewriting at the forwarding boundary (top-level
   field only).
5. Hot-reload hook: refresh catalogs before validating the new config.
6. Admin passthrough fork in the `/admin/*` handlers.
7. Stats union in `/v1/stats`, `/v1/info`, `/admin/status`: on-demand
   peer fetch with timeout + last-known fallback, qualified and
   bare-summed entries.
8. Request-ID mapping and `/v1/requests/{id}` translation.

Steps 3–4 are the bulk; everything else is small and additive. Existing
remote-model behavior (`cmd: http://...` entries in `models:`) remains
valid and unchanged.

# Open questions

None — the two former minors (ID-map retention, stats fetch timeout) are
specified in decisions 9 and 10.
