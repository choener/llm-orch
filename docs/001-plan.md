llm-orch is a tentative design for a llama-swap replacement. Single-user, single-host
software; not a multi-tenant enterprise product.

# Core

- Default-deny access with apikeys. If no apikeys are configured, deny all access.
  - Apikeys live in a separate file referenced from the main config (`apikeys_file: ...`).
    The file is user-editable at runtime and is hot-reloaded just like the main config.
  - At minimum: one apikey per line, optional label/comment. Revocation = delete the line.
- Hot-reload configuration. On config change (main config *or* apikeys file), reload
  without dropping in-flight requests. Config format is YAML (chosen for comments).
- Config validation on hot-reload. Reject bad configs atomically rather than partially
  applying them and breaking the running system. Old config keeps running on rejection.
- Dry-run / config-check mode: `llm-orch --check-config path.yaml` exits non-zero on
  invalid config without touching a running daemon.
- Graceful shutdown. Drain in-flight requests, finish streaming responses, then kill
  backends.
- NO preload / warm models on startup. GPUs stay offline while idle. First request pays
  the load cost. Not configurable.
- TLS/HTTPS is out of scope -- nginx reverse proxy handles it with name-based routing.
  Corollary: trust `X-Forwarded-For` / `X-Real-IP` only from configured upstream(s),
  otherwise per-key rate limiting can be bypassed.

# Scheduler

All of the items below are facets of one design: when does an instance get spawned,
which instance handles a given request, and when does something get evicted.

## Instance lifecycle

- Multiple instances of the same model can be configured on different GPUs. When
  concurrent requests arrive, llm-orch spawns additional instances up to the configured
  maximum and load-balances across them.
- Per-instance `max_concurrent` (parallel requests). llama.cpp and vLLM differ here;
  the scheduler needs this number to decide when to spawn another instance vs queue.
- TTL-based unloading. Per-model idle timeout (seconds). Models unload automatically
  after N seconds of inactivity to free VRAM. This is the default; keep-alive is the
  exception.
- Deterministic instance IDs (e.g. `qwen3-32b@gpu0`) so logs, metrics, and the activity
  page correlate across restarts.

## Cache-aware routing

- Remember (request -> instance) associations for recent requests. New requests whose
  prompt prefix matches a remembered request are routed to the same instance to reuse
  KV cache.
- Stickiness: short prompts can be freely redistributed. Long prompts stay pinned to
  their current instance. Configurable token-count threshold (single knob; no
  cache-hit-ratio variant for v1).

## Eviction

- When all configured instances are active and a new model request cannot be satisfied,
  evict the lowest-priority instance.
- Priority / eviction policy is a first-class config concept: per-model `priority: int`,
  LRU as tiebreaker.
- Partial VRAM utilization is acceptable. If a small model fits in residual VRAM
  alongside a large model, prefer co-residency over eviction.

## Measurement

- Actual VRAM and GPU utilization read from `/sys` (AMD sysfs nodes; equivalent for
  NVIDIA). Source of truth is real hardware state, not declared model sizes.
- Per-model `vram`/`ram` caps in config feed the scheduler's eviction decisions when
  actual measurement is unavailable or stale.

## Swaps and the KV cache

- A "swap" (moving a model to a different GPU, or evicting+reloading) only happens
  *between requests*, never mid-generation. KV cache loss on swap is accepted -- can't
  magically create more hardware.
- In-flight streaming requests complete on their current instance before any swap.

## Keep-alive

- Per loaded model, optionally configure a keep-alive probe with a delay in seconds to
  prevent the GPU driver from unloading VRAM. Built-in probes preferred (sysfs read for
  AMD, NVML query for NVIDIA) over shelling out to commands.

## Groups / model pools (tentative)

- Sets of models that can coexist vs sets that must swap. llama-swap's grouping is
  unsatisfying because the real constraint is VRAM contention, not arbitrary groups.
  Revisit once the rest of the scheduler is built; may turn out to be unnecessary.

# Backends

- Multi-backend support. llama.cpp, vLLM, TabbyAPI, etc. are pluggable runners.
  Container-based backends need lifecycle hooks beyond `kill` (`cmd` / `cmdStop`
  equivalent).
- Crash handling: a backend instance that exits without producing output gets a
  limited number of restart attempts (configurable, small default e.g. 3). After the
  limit is hit, the *model* is marked blocked and subsequent requests fail fast until
  the block is cleared (manual admin action or config reload).
- Backends that crash *after* producing some output are restarted normally; the
  crash-counter logic only applies to "never produced anything" failures.
- Request cancellation: when a client disconnects mid-stream, propagate cancellation
  to the backend if the backend supports it. Best-effort; not a hard guarantee.

# Request handling

- Request queueing with long timeouts and heartbeat. When all instances of a model are
  busy (or a swap is in progress), queue the request and stream "busy thinking" chunks
  (SSE) so the client doesn't time out.
- Semi-unbounded queue: queue depth is high (e.g. parallel=2, queued=20+) so that real
  workloads with bursty arrival don't get rejected. Above the configured cap, return
  `429` with `Retry-After`.
- Per-API-key / per-model rate limiting.

# API surface

- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`.
- `/v1/embeddings` treated as a distinct workload (no streaming, different ctx-size,
  different backend flags).
- `/v1/models` lists real models *and* aliases, and advertises `context_length`.
- `/v1/info` returns currently loaded models, queue depths, and per-model instance
  counts.  Requires a valid API key.  /health stays open for probes.
- Audio and video generation endpoints planned; separate lifecycle from text models,
  see [Future / maybe](#future--maybe).
- Admin REST API: `POST /admin/load`, `POST /admin/unload`, `GET /admin/status`,
  `POST /admin/unblock` (clears the crash-block from a model).
- Health check endpoint `/health` (open) for systemd / load-balancer probes.

## Aliases

- Multiple names route to the same underlying model.
- Inputs and outputs are tagged with the alias in logs and metrics (metadata only --
  the model never sees the alias name).
- Aliases appear in `/v1/models` so clients can discover them.
- Per-alias system prompt / prompt template injection. Natural extension of alias
  tagging: "alias X also prepends this system prompt."

## Self-service UI key

- The web UI calls llm-orch's own API (e.g. to trigger model loading). Behind a reverse
  proxy, browser CORS and auth headers make this fragile.
- llm-orch auto-generates an internal apikey that the UI uses transparently, so the UI
  works regardless of how it's proxied.

# Observability

## Web UI pages

- Model activity page: per-model status (loaded/idle/evicted/blocked), request count,
  average latency, token throughput, uptime.
- GPU metrics page: VRAM usage per model, GPU utilization %, temperature, power draw.
  One row per GPU. Driven by the same `/sys` reads the scheduler uses.

## Metrics

- Prometheus `/metrics` endpoint. Feeds Grafana / alerting / automated decisions.
- Per-request latency histograms (p50/p95/p99), token throughput, cache hit rates per
  model.

## Logs

- Request logging: write inputs and outputs to log files with automatic rotation and
  size/age-based cleanup. Enabled/disabled via (hot-reloaded) config.
- Structured JSON logging in addition to plain log-to-file. Machine-parseable for
  journald / Loki etc.
- Audit log (separate, append-only): admin actions -- load/unload, apikey file
  changes seen on reload, config reloads, model-block events.

## State persistence

- Summary statistics (request counts, token counts, uptime, cache hit rates,
  crash-block state) are flushed to a JSON file every ~60 seconds so they survive
  daemon restarts.
- Detailed per-request logs are not persisted as state -- those live in the rotating
  log files only.

# Non-goals

- TLS termination (nginx does it).
- Preloading / startup warmup of any kind.
- Multi-tenant isolation, billing, quotas-for-paying-customers, SSO, audit-for-compliance.
  This is single-user software.
- API versioning of llm-orch's own admin endpoints. One person, one deployment, breaking
  changes are fine.

# Future / maybe

- Anthropic API compatibility, if a client I actually use needs it.
- Audio (`/v1/audio/*`) and video generation endpoints. Different lifecycle (long
  jobs, file outputs) so they need their own runner spec; deferred until I actually
  need them.
