// ── Instance manager ─────────────────────────────────────────────────────────
//
// Central scheduler: owns the map of model → running instances, handles
// spawning, slot acquisition, queueing, idle eviction, and shutdown.

use crate::backend::{output_lines, poll_readiness, shutdown_child, spawn_process, Backend, DeviceKind, LlamaCppBackend, ReadyOutcome};
use crate::config::ModelConfig;
use crate::gpu::GpuMetrics;
use crate::types::CompletionRecord;
use crate::http_client;
use crate::instance::{Instance, InstanceHandle, InstanceState, SlotGuard};
use crate::keepalive::KeepAliveManager;
use crate::port_alloc::PortAllocator;

use reqwest::Client;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch, Mutex, Semaphore, oneshot};
use tracing::{debug, error, info, warn};

/// Maximum time a request may wait in the queue for a slot before failing.
/// Without a bound, a lost wakeup (or simply sustained saturation) would
/// park the HTTP request forever.
const QUEUE_WAIT_TIMEOUT: Duration = Duration::from_secs(120);

/// How long a spawn holds the global spawn semaphore while waiting for
/// readiness.  Covers process start + initial load — the window where
/// concurrent loads would thrash the SSD and interleave VRAM allocations.
/// Slow loads keep waiting *without* the semaphore afterwards so one
/// slow model cannot head-of-line block spawns of every other model.
const SPAWN_SERIAL_WINDOW: Duration = Duration::from_secs(15);

/// A parked waiter: unique id (for timeout self-removal) plus the channel
/// used to deliver an already-acquired slot once capacity frees up.  The
/// `SlotGuard` *is* the slot (acquired by `wake_one` under the instance
/// lock), so a woken waiter can never lose the acquire race.
type WaitQueue = VecDeque<(u64, oneshot::Sender<SlotGuard>)>;

/// Why `get_or_spawn` could not provide an instance slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquireError {
    /// The model is crash-blocked.
    Blocked,
    /// Instances exist but all are busy, and the queue is full or the
    /// wait timed out.
    NoCapacity,
    /// No instance could be provided at all: the model is unknown to the
    /// manager (e.g. removed by a concurrent config reload), its spawn
    /// failed (see server logs), or all remaining instances are retiring.
    Unavailable,
}

/// Why `enqueue` failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnqueueError {
    /// The queue is at capacity.
    QueueFull,
    /// No routable instance exists — nothing could ever wake a waiter.
    NoInstances,
    /// The queue wait timed out.
    Timeout,
}

/// How instance removal treats an instance with in-flight requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemovalMode {
    /// Skip busy instances entirely (autoscale / TTL eviction).
    IfIdle,
    /// Mark busy instances draining (unroutable) and finish removal later
    /// via `reap_drained` (admin unload, config removal).
    Drain,
    /// Kill even busy instances (shutdown, reaping).
    Force,
}

// ── Model metrics (tickless EMA) ────────────────────────────────────────────

/// Per-model exponentially-weighted moving averages for load and request rate.
///
/// Updated on every slot acquire / release with the real Δt since the last
/// event — no periodic tick, no polling.  Also forced-refreshed when the
/// info / status endpoints are queried so returned values are never stale.
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    /// 1 / 5 / 15-minute EMA of concurrent in-flight requests (Unix-style load).
    pub load_m1: f64,
    pub load_m5: f64,
    pub load_m15: f64,
    /// 1 / 5 / 15-minute EMA of request completion rate (req / min).
    ///
    /// Counted at slot release, so this is strictly a *release rate*:
    /// successful completions, backend errors, and truncated streams
    /// alike.  That is deliberate — the metric is informational
    /// (dashboards, sizing), not an eviction input, and plumbing
    /// per-request outcomes into the RAII `SlotGuard` drop path isn't
    /// worth the coupling.
    pub req_rate_m1: f64,
    pub req_rate_m5: f64,
    pub req_rate_m15: f64,
    /// Total completed requests since daemon start (release-counted,
    /// see `req_rate_m1`).
    pub completions_total: u64,

    /// Wall-clock time of the most recent activity (acquire, release, or
    /// non-zero in-flight).  Used by the autoscaler for idle-TTL despawn.
    pub last_activity: Instant,

    last_update: Instant,
    last_active: usize,
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            load_m1: 0.0,
            load_m5: 0.0,
            load_m15: 0.0,
            req_rate_m1: 0.0,
            req_rate_m5: 0.0,
            req_rate_m15: 0.0,
            completions_total: 0,
            last_activity: Instant::now(),
            last_update: Instant::now(),
            last_active: 0,
        }
    }
}

impl ModelMetrics {
    /// Advance all EMAs to `now` with `active` as the current in-flight count
    /// and `completions_delta` newly-completed requests since the last tick.
    ///
    /// The load averages decay with the *old* active count (which was present
    /// during the Δt interval), then record the new count for the next tick.
    fn tick(&mut self, now: Instant, active: usize, completions_delta: u64) {
        // Bump last_activity on any non-zero load or completion.
        if active > 0 || completions_delta > 0 {
            self.last_activity = now;
        }

        let dt = now.duration_since(self.last_update).as_secs_f64();
        if dt <= 0.0 {
            self.last_active = active;
            self.completions_total += completions_delta;
            self.last_update = now;
            return;
        }

        let active_f = self.last_active as f64;

        let alpha_1  = 1.0 - (-dt / 60.0_f64).exp();
        let alpha_5  = 1.0 - (-dt / 300.0_f64).exp();
        let alpha_15 = 1.0 - (-dt / 900.0_f64).exp();

        // ── Load averages ───────────────────────────────────────────
        self.load_m1  = self.load_m1  * (1.0 - alpha_1)  + active_f * alpha_1;
        self.load_m5  = self.load_m5  * (1.0 - alpha_5)  + active_f * alpha_5;
        self.load_m15 = self.load_m15 * (1.0 - alpha_15) + active_f * alpha_15;

        // ── Request rate ───────────────────────────────────────────
        if completions_delta > 0 {
            let rate_per_min = (completions_delta as f64) / (dt / 60.0);
            self.req_rate_m1  = self.req_rate_m1  * (1.0 - alpha_1)  + rate_per_min * alpha_1;
            self.req_rate_m5  = self.req_rate_m5  * (1.0 - alpha_5)  + rate_per_min * alpha_5;
            self.req_rate_m15 = self.req_rate_m15 * (1.0 - alpha_15) + rate_per_min * alpha_15;
            self.completions_total += completions_delta;
        } else {
            self.req_rate_m1  *= 1.0 - alpha_1;
            self.req_rate_m5  *= 1.0 - alpha_5;
            self.req_rate_m15 *= 1.0 - alpha_15;
        }

        self.last_update = now;
        self.last_active = active;
    }
}

// ── Spawn-config fingerprint ────────────────────────────────────────────────

/// Fingerprint of the configuration that defines the spawned backend
/// process: the alias-resolved command line (`{port}` deliberately
/// excluded — it differs per instance), the Vulkan device pool, declared
/// VRAM, and context length.  Routing-only fields (`max_instances`,
/// `max_concurrent`, `queue_depth`, `idle_ttl`, `autoscale`, `debug_log`,
/// `priority`, `ram`) are excluded: changing them must not retire running
/// instances.
///
/// Instances whose stored fingerprint no longer matches the current
/// config are retired on reload (marked `Failed`/draining) and replaced
/// on demand.  Runtime-only value — never persisted, so `DefaultHasher`'s
/// version instability is irrelevant.
fn fingerprint_with_aliases(cmd_aliases: &HashMap<String, String>, cfg: &ModelConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut resolved = cfg.cmd.clone();
    for (key, value) in cmd_aliases {
        resolved = resolved.replace(&format!("{{{}}}", key), value);
    }
    let mut devices = cfg.vulkan_devices.clone();
    devices.sort_unstable();
    let mut cuda_devices = cfg.cuda_devices.clone();
    cuda_devices.sort_unstable();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    resolved.hash(&mut h);
    devices.hash(&mut h);
    cuda_devices.hash(&mut h);
    cfg.gpus.hash(&mut h);
    cfg.vram.hash(&mut h);
    cfg.context_length.hash(&mut h);
    h.finish()
}

/// Extract (cuda index → PCI slot) and (cuda index → static VRAM bytes)
/// maps from the config's `devices.cuda` section.
fn cuda_device_maps(
    config: &crate::config::Config,
) -> (HashMap<usize, String>, HashMap<usize, u64>) {
    let Some(devs) = config.devices.as_ref() else {
        return (HashMap::new(), HashMap::new());
    };
    let slots = devs
        .cuda
        .iter()
        .map(|(k, v)| (*k, v.pci.clone()))
        .collect();
    let static_vram = devs
        .cuda
        .iter()
        .filter_map(|(k, v)| v.vram_mb.map(|mb| (*k, mb * 1024 * 1024)))
        .collect();
    (slots, static_vram)
}

// ── Manager ──────────────────────────────────────────────────────────────────

pub struct InstanceManager {
    /// Shared HTTP client for health checks and request forwarding.
    client: Client,

    /// Backend type (currently always llama.cpp).
    backend: LlamaCppBackend,

    /// Port allocator.
    ports: Mutex<PortAllocator>,

    /// Per-model configuration, indexed by model name.
    /// Swapped on config hot-reload (`reconcile_config`).
    model_configs: RwLock<HashMap<String, ModelConfig>>,

    /// Named `{key}` → value fragments from `cmd_aliases`.
    cmd_aliases: RwLock<HashMap<String, String>>,

    /// Running instances, keyed by model name.
    instances: RwLock<HashMap<String, Vec<InstanceHandle>>>,

    /// Per-model wait queues.  When all instances are at capacity, requests
    /// park on a oneshot channel until a slot frees up, the queue is full,
    /// or the wait times out.  Each entry carries a unique id so timed-out
    /// waiters can remove themselves from the queue.
    queues: RwLock<HashMap<String, WaitQueue>>,

    /// Unique id source for queue entries (used for timeout self-removal).
    next_queue_id: AtomicU64,

    /// Sequence number appended to instance IDs (`model@gpu#N`) so every
    /// instance gets a unique ID even when the base `model@gpus` collides
    /// (e.g. multiple CPU instances of one model).
    instance_seq: AtomicU64,

    /// Per-model blocked flag.  A blocked model refuses all requests.
    blocked: RwLock<HashMap<String, bool>>,

    /// Latest GPU metrics snapshot for VRAM-aware scheduling.
    gpu_snapshot: Arc<tokio::sync::RwLock<Vec<GpuMetrics>>>,

    /// Vulkan device index → PCI slot mapping (from config).
    vulkan_slots: RwLock<HashMap<usize, String>>,

    /// Vulkan device index → VRAM limit in bytes (from config, optional).
    vram_limits: RwLock<HashMap<usize, u64>>,

    /// CUDA device index → PCI slot mapping (from config).
    cuda_slots: RwLock<HashMap<usize, String>>,

    /// CUDA device index → static VRAM total in bytes (from config,
    /// optional).  Doubles as a capacity cap and as the fallback total
    /// when nvidia-smi metrics are unavailable for the device.
    cuda_vram_static: RwLock<HashMap<usize, u64>>,

    /// GPU keep-alive manager (None if not configured).
    /// Rebuilt on config hot-reload when the keep-alive section changes.
    keepalive: RwLock<Option<Arc<KeepAliveManager>>>,

    /// Crash limit before a model is blocked.
    crash_limit: usize,

    /// Spawn readiness timeout.
    spawn_timeout: Duration,

    /// Global interval (seconds) between loading-indicator dots sent to
    /// streaming clients while a backend instance is being acquired.
    /// `0` disables globally.  Stored from `server.loading_dots`; the
    /// per-model override is applied in `loading_dots_interval`.
    loading_dots_interval_secs: AtomicU64,

    /// Per-model EMA metrics (load & request rate).
    model_metrics: RwLock<HashMap<String, ModelMetrics>>,

    /// Global semaphore to serialize the critical section of spawn
    /// attempts across models (process start + `SPAWN_SERIAL_WINDOW` of
    /// readiness).  Prevents concurrent model loads from thrashing the
    /// SSD and interleaving VRAM allocations without head-of-line
    /// blocking unrelated models for the full spawn timeout.
    spawn_semaphore: Arc<Semaphore>,

    /// Per-model timestamp of the last autoscale action (cooldown enforcement).
    last_scale_action: RwLock<HashMap<String, Instant>>,

    /// Per-model ring buffer of recent completion records (newest first).
    recent_completions: RwLock<HashMap<String, Vec<CompletionRecord>>>,

    /// Sender side of the release-notification channel.
    /// Cloned into every `Instance` so `release_slot` can notify the
    /// background release-processing task without holding any locks.
    release_tx: mpsc::UnboundedSender<String>,

    /// Sender side of the crash-notification channel.
    /// Cloned into per-instance monitor tasks so an unexpected child exit
    /// can be reported to the background crash-processing task.
    crash_tx: mpsc::UnboundedSender<InstanceHandle>,

    /// Consecutive pre-output crashes per model.  Reset on successful
    /// spawn and on unblock; reaching `crash_limit` blocks the model.
    crash_counts: std::sync::Mutex<HashMap<String, usize>>,

    /// Per-model in-flight spawn tasks.  The `watch::Sender` lives in the
    /// map for the duration of the spawn and is dropped (removing the
    /// entry) when the spawn task finishes; requests subscribe and await
    /// the channel closing instead of driving the spawn themselves, so a
    /// client disconnect can never abort a spawn half-way and strand a
    /// `Loading` instance that nothing would reap.
    spawns_in_flight: std::sync::RwLock<HashMap<String, watch::Sender<()>>>,
}

impl InstanceManager {
    /// Create a new manager plus the receivers for the release channel and
    /// the crash channel.
    ///
    /// The caller must spawn background tasks that drain both receivers:
    ///
    /// - `release_rx` → `record_metrics_event` + `wake_one` per model name;
    /// - `crash_rx` → `handle_crash` per instance handle.
    ///
    /// Both tasks run with no locks held, so they can safely acquire
    /// `instances.read()` → `metrics.write()` without deadlocking with the
    /// request-completion Drop path.
    pub fn new(
        config: &crate::config::Config,
        gpu_snapshot: Arc<tokio::sync::RwLock<Vec<GpuMetrics>>>,
        keepalive: Option<Arc<KeepAliveManager>>,
    ) -> (
        Self,
        mpsc::UnboundedReceiver<String>,
        mpsc::UnboundedReceiver<InstanceHandle>,
    ) {
        let model_configs: HashMap<_, _> = config
            .models
            .iter()
            .map(|m| (m.name.clone(), m.clone()))
            .collect();

        let vulkan_slots = config
            .devices
            .as_ref()
            .map(|d| d.vulkan.clone())
            .unwrap_or_default();

        let vram_limits: HashMap<usize, u64> = config
            .devices
            .as_ref()
            .map(|d| {
                d.vram_limit_mb
                    .iter()
                    .map(|(k, v)| (*k, *v * 1024 * 1024))
                    .collect()
            })
            .unwrap_or_default();

        let (cuda_slots, cuda_vram_static) = cuda_device_maps(config);

        let (release_tx, release_rx) = mpsc::unbounded_channel();
        let (crash_tx, crash_rx) = mpsc::unbounded_channel();

        let mgr = Self {
            client: http_client::build(),
            backend: LlamaCppBackend::default(),
            ports: Mutex::new(PortAllocator::new(config.server.port_range.clone())),
            model_configs: RwLock::new(model_configs),
            cmd_aliases: RwLock::new(config.cmd_aliases.clone()),
            instances: RwLock::new(HashMap::new()),
            queues: RwLock::new(HashMap::new()),
            next_queue_id: AtomicU64::new(0),
            instance_seq: AtomicU64::new(0),
            blocked: RwLock::new(HashMap::new()),
            gpu_snapshot,
            vulkan_slots: RwLock::new(vulkan_slots),
            vram_limits: RwLock::new(vram_limits),
            cuda_slots: RwLock::new(cuda_slots),
            cuda_vram_static: RwLock::new(cuda_vram_static),
            keepalive: RwLock::new(keepalive),
            crash_limit: 3,
            spawn_timeout: Duration::from_secs(120),
            loading_dots_interval_secs: AtomicU64::new(config.server.loading_dots.unwrap_or(0)),
            model_metrics: RwLock::new(HashMap::new()),
            spawn_semaphore: Arc::new(Semaphore::new(1)),
            last_scale_action: RwLock::new(HashMap::new()),
            recent_completions: RwLock::new(HashMap::new()),
            release_tx,
            crash_tx,
            crash_counts: std::sync::Mutex::new(HashMap::new()),
            spawns_in_flight: std::sync::RwLock::new(HashMap::new()),
        };
        (mgr, release_rx, crash_rx)
    }

    // ── get-or-spawn ──────────────────────────────────────────────────────

    /// Acquire an instance slot for `model_name`, spawning a new instance
    /// if necessary.  See `AcquireError` for the failure modes.
    ///
    /// The returned guard already owns one in-flight slot — the caller must
    /// not call `try_acquire` again.  The slot is released automatically
    /// when the guard is dropped.  Use `guard.handle()` to reach the instance.
    ///
    /// Spawns run in a detached background task (see `ensure_spawn`); this
    /// function only *awaits* the spawn's completion.  A client disconnect
    /// that drops this future therefore cannot abort a spawn half-way and
    /// strand a `Loading` instance that nothing would ever reap.
    pub async fn get_or_spawn(self: &Arc<Self>, model_name: &str) -> Result<SlotGuard, AcquireError> {
        if self.is_blocked(model_name) {
            return Err(AcquireError::Blocked);
        }

        // Clone the model config out of the lock — it may be swapped by a
        // config reload at any time, and the guard must not be held across
        // the awaits below.
        let cfg = self
            .model_configs
            .read()
            .unwrap()
            .get(model_name)
            .cloned()
            .ok_or(AcquireError::Unavailable)?;
        let max_concurrent = cfg.max_concurrent;

        // Fast path: find a ready instance with spare capacity.
        if let Some(guard) = self.acquire_ready_slot(model_name, max_concurrent) {
            return Ok(guard);
        }

        // Autoscale spawn gate: only spawn if sustained load exceeds
        // threshold (skip for cold-start, i.e. zero existing instances).
        let should_spawn = if let Some(ref a) = cfg.autoscale {
            if !a.enabled {
                true
            } else {
                // Retiring (`Failed`) instances don't count: from a routing
                // perspective the model has zero instances, so a config
                // change that retired the only instance triggers a cold-start
                // replacement spawn instead of queueing behind the drain.
                let num_existing = {
                    self.instances.read().unwrap()
                        .get(model_name)
                        .map(|l| {
                            l.iter()
                                .filter(|h| {
                                    h.inner().lock().unwrap().state != InstanceState::Failed
                                })
                                .count()
                        })
                        .unwrap_or(0)
                };
                if num_existing == 0 {
                    true // cold start — always spawn immediately
                } else {
                    let load_m5 = self.model_metrics.read().unwrap()
                        .get(model_name)
                        .map(|m| m.load_m5)
                        .unwrap_or(0.0);
                    let cap = (max_concurrent * num_existing) as f64;
                    let threshold = a.scale_up_at * cap;
                    if load_m5 <= threshold {
                        debug!(
                            model = %model_name,
                            load_m5 = %load_m5,
                            threshold = %threshold,
                            "autoscale gate: load too low, queuing"
                        );
                        false
                    } else {
                        true
                    }
                }
            }
        } else {
            true // no autoscale config — spawn immediately
        };

        // Slow path: ensure a spawn is running and await its completion.
        if should_spawn {
            let mut rx = self.ensure_spawn(model_name, &cfg);
            // Bound the wait: spawn timeout + serialized window + margin for
            // semaphore queueing and process setup.
            let budget = self.spawn_timeout + SPAWN_SERIAL_WINDOW + Duration::from_secs(30);
            // The entry's sender is dropped when the spawn task finishes —
            // success or failure — which resolves `changed()` with an error.
            let _ = tokio::time::timeout(budget, rx.changed()).await;
            // Retry the fast path once: after a successful spawn the fresh
            // instance is Ready with spare capacity.  On failure (or if
            // competing requests filled it first) fall through to the
            // queue — which fails fast when no instance exists at all.
            if let Some(guard) = self.acquire_ready_slot(model_name, max_concurrent) {
                return Ok(guard);
            }
        }

        // Queue path: all instances busy and at cap.
        match self.enqueue(model_name, cfg.queue_depth).await {
            Ok(guard) => Ok(guard),
            // Nothing exists that could ever wake a waiter — the spawn
            // failed (see logs) or everything left is retiring.
            Err(EnqueueError::NoInstances) => Err(AcquireError::Unavailable),
            Err(EnqueueError::QueueFull | EnqueueError::Timeout) => {
                Err(AcquireError::NoCapacity)
            }
        }
    }

    /// Ensure at least one instance of `model_name` exists (spawning one
    /// if necessary) without acquiring a request slot and — unlike
    /// `get_or_spawn` — without ever parking on the request queue.
    /// Used by `/admin/load`, which must not hang behind user traffic.
    pub async fn ensure_instance(self: &Arc<Self>, model_name: &str) -> Result<(), AcquireError> {
        if self.is_blocked(model_name) {
            return Err(AcquireError::Blocked);
        }
        let cfg = self
            .model_configs
            .read()
            .unwrap()
            .get(model_name)
            .cloned()
            .ok_or(AcquireError::Unavailable)?;

        // Already serving (or at least registered and routable)?
        if self.find_ready_instance(model_name, cfg.max_concurrent).is_some() {
            return Ok(());
        }

        let mut rx = self.ensure_spawn(model_name, &cfg);
        let budget = self.spawn_timeout + SPAWN_SERIAL_WINDOW + Duration::from_secs(30);
        let _ = tokio::time::timeout(budget, rx.changed()).await;

        if self.find_ready_instance(model_name, cfg.max_concurrent).is_some() {
            Ok(())
        } else {
            Err(AcquireError::Unavailable)
        }
    }

    /// Effective loading-indicator interval for `model_name`, after applying
    /// the per-model override (`models[].loading_dots`) on top of the global
    /// `server.loading_dots` setting.
    ///
    /// Returns `None` when the feature is disabled (either globally `0` /
    /// `None`, or the per-model override is `0`).
    pub fn loading_dots_interval(&self, model_name: &str) -> Option<Duration> {
        let global = self.loading_dots_interval_secs.load(Ordering::Relaxed);
        if global == 0 {
            // Global disabled — per-model override can't re-enable.
            return None;
        }
        let cfg = self.model_configs.read().unwrap();
        match cfg.get(model_name) {
            Some(m) => match m.loading_dots {
                Some(0) => None,                            // per-model override silences dots
                Some(n) => Some(Duration::from_secs(n)),    // per-model override
                None => Some(Duration::from_secs(global)),  // inherit global
            },
            None => None, // model unknown
        }
    }

    /// Fast path of `get_or_spawn`: acquire a slot on the least-loaded
    /// ready instance with spare capacity.
    fn acquire_ready_slot(&self, model_name: &str, max_concurrent: usize) -> Option<SlotGuard> {
        let handle = self.find_ready_instance(model_name, max_concurrent)?;
        let guard = handle.try_acquire(max_concurrent)?;
        self.record_metrics_event(model_name, 0);
        Some(guard)
    }

    /// Subscribe to the completion of `model_name`'s in-flight spawn,
    /// starting the spawn task first if none is running.
    ///
    /// The spawn runs in a detached task so its lifecycle is independent of
    /// any request future.  The returned receiver resolves (with `Err` —
    /// nothing is ever sent) when the spawn task finishes and drops the
    /// entry's sender.
    fn ensure_spawn(self: &Arc<Self>, model_name: &str, cfg: &ModelConfig) -> watch::Receiver<()> {
        {
            let spawns = self.spawns_in_flight.read().unwrap();
            if let Some(tx) = spawns.get(model_name) {
                return tx.subscribe();
            }
        }
        let (tx, rx) = watch::channel(());
        {
            let mut spawns = self.spawns_in_flight.write().unwrap();
            if let Some(existing) = spawns.get(model_name) {
                // Lost the race — another caller registered the spawn first.
                return existing.subscribe();
            }
            spawns.insert(model_name.to_owned(), tx);
        }

        let mgr = Arc::clone(self);
        let name = model_name.to_owned();
        let cfg = cfg.clone();
        tokio::spawn(async move {
            // Removing the entry drops the sender, waking all awaiters.
            // The guard runs on normal completion *and* on panic unwind,
            // so a panicking spawn task can't block future spawns forever.
            let _entry = SpawnEntryGuard {
                mgr: Arc::clone(&mgr),
                model_name: name.clone(),
            };
            if mgr.try_spawn(&name, &cfg).await.is_some() {
                // A fresh instance may be able to serve parked waiters —
                // otherwise queued requests would only be served after the
                // next release event.
                mgr.wake_one(&name);
            }
        });
        rx
    }

    /// Enqueue the caller, waiting for an instance slot to free up.
    /// Fails with `QueueFull` if the queue is at capacity (caller maps to
    /// 429), with `NoInstances` if no routable instance exists (no release
    /// or spawn event could ever wake the parked waiter — fail fast
    /// instead of hanging forever), or with `Timeout` when the wait
    /// exceeds the budget.  On success the received guard already owns the
    /// slot (acquired by `wake_one`) — there is no acquire race on this
    /// side.
    async fn enqueue(
        &self,
        model_name: &str,
        max_depth: usize,
    ) -> Result<SlotGuard, EnqueueError> {
        let (tx, rx) = oneshot::channel();
        let id = self.next_queue_id.fetch_add(1, Ordering::Relaxed);

        {
            // Hold the instances read lock across the check *and* the queue
            // push.  remove_instance takes the instances write lock to remove
            // the last instance and only then drains the queue — so either
            // our push lands before the drain (and is cleared by it) or after
            // the removal (and the check fails).  Without this, a removal
            // could slip in between check and push and park us forever.
            let instances = self.instances.read().unwrap();
            // Only routable instances count: when every remaining instance
            // is retiring (`Failed` — draining or config-retired), no
            // release event can ever serve a parked waiter (retiring
            // instances accept no new slots), so fail fast instead of
            // parking until the reap drains the queue.
            let has_instances = instances
                .get(model_name)
                .map(|l| {
                    l.iter()
                        .any(|h| h.inner().lock().unwrap().state != InstanceState::Failed)
                })
                .unwrap_or(false);
            if !has_instances {
                return Err(EnqueueError::NoInstances);
            }

            let mut queues = self.queues.write().unwrap();
            let queue = queues.entry(model_name.to_owned()).or_default();
            if queue.len() >= max_depth {
                return Err(EnqueueError::QueueFull);
            }
            queue.push_back((id, tx));
        }

        // Close the lost-wakeup race: a slot may have been released between
        // the capacity check in get_or_spawn and our queue push, with its
        // wake_one finding an empty queue.  Re-signal now that we are
        // parked — if capacity exists, the head waiter gets served.
        self.wake_one(model_name);

        // The wait budget must exceed the worst-case spawn (spawn timeout
        // + serialized window + margin): a waiter parked behind a fresh
        // spawn must not time out and 429 just as capacity appears.
        let wait_budget = QUEUE_WAIT_TIMEOUT
            .max(self.spawn_timeout + SPAWN_SERIAL_WINDOW + Duration::from_secs(60));
        let guard = match tokio::time::timeout(wait_budget, rx).await {
            Ok(Ok(guard)) => guard,
            // Queue drained (last instance removed) — fail fast.
            Ok(Err(_)) => return Err(EnqueueError::NoInstances),
            Err(_) => {
                // Timed out — remove our entry if it is still parked.
                // (If wake_one already popped us, the guard it sent is
                // dropped with the receiver, which releases the slot —
                // accounting stays balanced.)
                self.remove_queued(model_name, id);
                return Err(EnqueueError::Timeout);
            }
        };

        // The guard already owns the slot (acquired by `wake_one`).
        self.record_metrics_event(model_name, 0);
        Ok(guard)
    }

    /// Remove a specific queued waiter (used after a queue-wait timeout).
    fn remove_queued(&self, model_name: &str, id: u64) {
        if let Some(q) = self.queues.write().unwrap().get_mut(model_name) {
            q.retain(|(entry_id, _)| *entry_id != id);
        }
    }

    /// If no instances remain for `model_name`, drop all parked queue
    /// senders.  Each waiter's `rx.await` then resolves to `Err`, making the
    /// request fail fast instead of hanging forever — `wake_one` is only
    /// called on release events, which require a live instance.
    fn drain_queue_if_no_instances(&self, model_name: &str) {
        let none_left = self
            .instances
            .read()
            .unwrap()
            .get(model_name)
            .map(|l| l.is_empty())
            .unwrap_or(true);
        if none_left
            && let Some(q) = self.queues.write().unwrap().get_mut(model_name)
        {
            q.clear();
        }
    }

    /// Wake the first queued waiter for `model_name` if an instance has
    /// spare capacity.
    ///
    /// The slot is acquired *here*, under the instance lock, and the
    /// `SlotGuard` itself is sent through the channel — the woken waiter
    /// receives an owned slot, so a competing request can never snatch it
    /// between wake and acquire (the old `LostRace` 429).  Waiters that
    /// vanished (timeout racing the wake) are skipped; a waiter that
    /// can't be served because capacity disappeared in between is
    /// re-parked at the head, preserving FIFO order.
    ///
    /// Public for the background release-processing task (see
    /// `InstanceManager::new`).
    pub fn wake_one(&self, model_name: &str) {
        let cfg = match self.model_configs.read().unwrap().get(model_name).cloned() {
            Some(c) => c,
            None => return,
        };

        loop {
            // Pop the head waiter, if any.  The queues lock is released
            // before touching instances — lock order is instances → queues.
            let entry = {
                let mut queues = self.queues.write().unwrap();
                queues.get_mut(model_name).and_then(|q| q.pop_front())
            };
            let Some((id, tx)) = entry else {
                return;
            };

            let guard = self
                .find_ready_instance(model_name, cfg.max_concurrent)
                .and_then(|h| h.try_acquire(cfg.max_concurrent));
            match guard {
                Some(g) => match tx.send(g) {
                    Ok(()) => return,
                    // Waiter vanished (timeout racing the wake) — the
                    // returned guard drops, releasing the slot; try the
                    // next waiter.  (Costs a rare phantom release event in
                    // the req-rate EMA — informational only.)
                    Err(_) => continue,
                },
                None => {
                    // Capacity disappeared between the release event and
                    // now — re-park at the head and wait for the next one.
                    let mut queues = self.queues.write().unwrap();
                    queues
                        .entry(model_name.to_owned())
                        .or_default()
                        .push_front((id, tx));
                    return;
                }
            }
        }
    }

    /// Find a ready instance for `model` with at least one free slot.
    fn find_ready_instance(
        &self,
        model_name: &str,
        max_concurrent: usize,
    ) -> Option<InstanceHandle> {
        let instances = self.instances.read().unwrap();
        let list = instances.get(model_name)?;
        list.iter()
            .filter(|h| {
                let inst = h.inner().lock().unwrap();
                inst.state == InstanceState::Ready && inst.in_flight < max_concurrent
            })
            .min_by_key(|h| h.inner().lock().unwrap().in_flight)
            .cloned()
    }

    /// Attempt to spawn a new instance for `model`.  Returns `None` if the
    /// per-model instance cap has been reached.
    async fn try_spawn(&self, model_name: &str, cfg: &ModelConfig) -> Option<InstanceHandle> {
        // Serialize the critical section of spawn attempts globally —
        // prevents concurrent model loads from thrashing the SSD and
        // interleaving VRAM.  Released after SPAWN_SERIAL_WINDOW of
        // readiness waiting, not after the full spawn timeout.
        let _permit = self.spawn_semaphore.clone().acquire_owned().await.ok()?;

        // Inside the semaphore: check instance cap.  Retiring instances
        // (`Failed` — draining after admin unload, or retired by a config
        // change) do not count: they are unroutable and reaped once idle,
        // so counting them would wedge models with `max_instances = 1`
        // behind a long-running request with no replacement.  Resource
        // limits are still enforced by the VRAM accounting in
        // `select_gpu_for_model`, which *does* include retiring instances.
        {
            let instances = self.instances.read().unwrap();
            if let Some(list) = instances.get(model_name) {
                let active = list
                    .iter()
                    .filter(|h| h.inner().lock().unwrap().state != InstanceState::Failed)
                    .count();
                if active >= cfg.max_instances {
                    return None;
                }
            }
        }

        // Allocate a port (single lock acquisition).
        //
        // The free-port check is inherently TOCTOU — another process can
        // grab the port between our check and the backend's bind().  The
        // readiness poll detects the resulting instant child exit and
        // fails the spawn fast instead of waiting out the spawn timeout.
        let port = self.ports.lock().await.allocate().await;
        match port {
            Some(p) => debug!(model = %model_name, port = p, "allocated port"),
            None => warn!(model = %model_name, "no port available"),
        }
        let port = port?;

        // Resolve the command string.
        let model_cmd = self.resolve_cmd(cfg, port);
        let parts = shlex::split(&model_cmd);
        if parts.is_none() || parts.as_ref().map(|p| p.is_empty()).unwrap_or(true) {
            warn!(model = %model_name, cmd = %model_cmd, "invalid or empty command after shlex split");
            self.ports.lock().await.free(port);
            return None;
        }
        let parts = parts.unwrap();
        debug!(model = %model_name, parts = ?parts, "parsed command");
        let prog = &parts[0];
        let gpu_indices: Vec<usize> = self.select_gpus_for_model(cfg).await;

        // Safety: when a device pool is configured but the full device
        // set can't be satisfied, fail the spawn instead of launching
        // without GPU restriction (competing on GPUs already occupied by
        // existing instances) or with too few devices (a multi-GPU model
        // would not fit in memory).
        let has_device_pool =
            !cfg.vulkan_devices.is_empty() || !cfg.cuda_devices.is_empty();
        if has_device_pool && gpu_indices.len() < cfg.gpus {
            warn!(
                model = %model_name,
                kind = ?cfg.device_kind(),
                needed = cfg.gpus,
                available = gpu_indices.len(),
                "not enough suitable GPUs — refusing to spawn"
            );
            self.ports.lock().await.free(port);
            return None;
        }

        // Device pinning follows the model's namespace: Vulkan models via
        // GGML_VK_VISIBLE_DEVICES, CUDA models via CUDA_VISIBLE_DEVICES.
        let model_backend = LlamaCppBackend::new(cfg.device_kind());
        let mut args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();
        args.extend(model_backend.gpu_args(&gpu_indices));
        let envs = model_backend.gpu_env(&gpu_indices);

        // Spawn.
        let (child, output) = match spawn_process(prog, &args, &envs, model_name, port).await {
            Ok(pair) => pair,
            Err(e) => {
                warn!(model = %model_name, error = %e, "spawn failed");
                self.ports.lock().await.free(port);
                return None;
            }
        };

        let mut inst = Instance::new(
            model_name,
            gpu_indices.clone(),
            port,
            Some(self.release_tx.clone()),
        );
        inst.config_fingerprint = self.spawn_fingerprint(cfg);
        // Guarantee a unique ID even when the base `model@gpus` collides
        // (multiple CPU instances of the same model).
        inst.id = format!(
            "{}#{}",
            inst.id,
            self.instance_seq.fetch_add(1, Ordering::Relaxed)
        );
        inst.child = Some(child);
        inst.output = Some(output);
        let handle = InstanceHandle::new(inst);

        // Register under model name immediately as Loading.
        {
            let mut instances = self.instances.write().unwrap();
            instances
                .entry(model_name.to_owned())
                .or_default()
                .push(handle.clone());
        }

        // Keep-alive: acquire one reference per occupied GPU.  Released
        // exactly once in unregister_instance, which every instance
        // removal funnels through — refcounting makes concurrent
        // spawn/remove interleavings safe.
        {
            let keepalive = self.keepalive.read().unwrap().clone();
            if let Some(ref ka) = keepalive {
                // Keep-alive is AMD/Vulkan-only; NVIDIA GPUs are kept
                // awake via persistence mode instead.
                if cfg.device_kind() == DeviceKind::Vulkan {
                    let vulkan_slots = self.vulkan_slots.read().unwrap();
                    for vulkan_idx in &gpu_indices {
                        if let Some(slot) = vulkan_slots.get(vulkan_idx) {
                            ka.acquire(slot);
                        }
                    }
                }
            }
        }

        // Spawn intent counts as activity: reset the idle-TTL clock so the
        // autoscaler can't despawn this instance — using stale metrics left
        // over from a previous despawn — while it is still loading.
        self.touch_activity(model_name);

        // Wait for readiness in two phases.  The poll fails fast when
        // the child exits before becoming healthy instead of waiting out
        // the full spawn timeout.
        //
        // Phase 1 (serialized): process start + initial load — fast
        // backends become ready here, dead-on-arrival ones are detected
        // here too.
        let phase1 =
            poll_readiness(&handle, &self.client, &self.backend, SPAWN_SERIAL_WINDOW).await;
        // Release the global spawn lock before the long tail so a slow
        // model can't head-of-line block spawns of every other model.
        drop(_permit);
        // Phase 2 (unserialized): wait out the remaining timeout.
        let outcome = match phase1 {
            ReadyOutcome::TimedOut => {
                let remaining = self.spawn_timeout.saturating_sub(SPAWN_SERIAL_WINDOW);
                poll_readiness(&handle, &self.client, &self.backend, remaining).await
            }
            other => other,
        };
        if outcome != ReadyOutcome::Ready {
            // Distinguish a dead process (pre-output crash — counts toward
            // the model block limit) from a live-but-slow load (no count).
            let child_exited = outcome == ReadyOutcome::ChildExited;
            warn!(model = %model_name, port = port, exited = child_exited, "instance did not become ready — shutting down");
            self.dump_recent_output(&handle);
            let mut child_to_kill = {
                let mut inst_lock = handle.inner().lock().unwrap();
                inst_lock.state = InstanceState::Failed;
                inst_lock.child.take()
            };
            if let Some(ref mut child) = child_to_kill {
                shutdown_child(child, Duration::from_secs(5)).await;
            }
            // Port free, registry removal, queue drain, keep-alive release.
            self.unregister_instance(model_name, &handle, &gpu_indices)
                .await;

            if child_exited {
                self.note_pre_output_crash(model_name);
            }

            return None;
        }

        // A config reload racing the spawn makes the fresh instance stale
        // (or removes its model) before it served a single request.  With
        // zero in-flight requests it can simply be discarded instead of
        // serving old-config traffic until the next reload notices.
        let inst_fingerprint = handle.inner().lock().unwrap().config_fingerprint;
        let current_cfg = self.model_configs.read().unwrap().get(model_name).cloned();
        let still_current = match current_cfg {
            Some(c) => self.spawn_fingerprint(&c) == inst_fingerprint,
            None => false,
        };
        if !still_current {
            warn!(
                model = %model_name,
                port = port,
                "config changed during spawn — discarding fresh instance"
            );
            let mut child = {
                let mut inst = handle.inner().lock().unwrap();
                inst.state = InstanceState::Failed;
                inst.child.take()
            };
            if let Some(ref mut c) = child {
                shutdown_child(c, Duration::from_secs(5)).await;
            }
            self.unregister_instance(model_name, &handle, &gpu_indices)
                .await;
            return None;
        }

        // A successful spawn resets the consecutive pre-output crash counter.
        self.crash_counts.lock().unwrap().remove(model_name);

        // Monitor the child for unexpected exits (crash → unregister/block).
        tokio::spawn(monitor_instance_exit(
            handle.clone(),
            self.crash_tx.clone(),
        ));

        let instance_id = {
            let inst = handle.inner().lock().unwrap();
            inst.id.clone()
        };
        info!(model = %model_name, inst = %instance_id, port = port, "spawn succeeded");

        Some(handle)
    }

    /// Spawn-config fingerprint of `cfg` resolved against the *current*
    /// cmd aliases (see `fingerprint_with_aliases`).
    fn spawn_fingerprint(&self, cfg: &ModelConfig) -> u64 {
        fingerprint_with_aliases(&self.cmd_aliases.read().unwrap(), cfg)
    }

    /// Resolve `cmd_aliases`, `{port}`, and `{context_length}` in the
    /// model's command string.
    fn resolve_cmd(&self, cfg: &ModelConfig, port: u16) -> String {
        let mut resolved = cfg.cmd.clone();
        let cmd_aliases = self.cmd_aliases.read().unwrap();
        for (key, value) in cmd_aliases.iter() {
            let placeholder = format!("{{{}}}", key);
            resolved = resolved.replace(&placeholder, value);
        }
        resolved
            .replace("{port}", &port.to_string())
            .replace("{context_length}", &cfg.context_length.to_string())
    }

    /// Pick the devices for a new instance from the model's device pool
    /// (`vulkan_devices` or `cuda_devices`, per the model's
    /// [`DeviceKind`]).  Returns up to `model_cfg.gpus` distinct devices
    /// (empty when the pool can't satisfy the request — the caller
    /// decides between CPU fallback and spawn refusal).
    ///
    /// Per-GPU accounting is keyed by PCI slot so Vulkan and CUDA
    /// indices can never alias each other.
    async fn select_gpus_for_model(&self, model_cfg: &ModelConfig) -> Vec<usize> {
        let kind = model_cfg.device_kind();
        let pool = match kind {
            DeviceKind::Cuda => &model_cfg.cuda_devices,
            DeviceKind::Vulkan => &model_cfg.vulkan_devices,
        };
        // Clone the (tiny) device maps — std RwLock guards are !Send and
        // must not live across the gpu_snapshot await below.
        let vulkan_slots = self.vulkan_slots.read().unwrap().clone();
        let cuda_slots = self.cuda_slots.read().unwrap().clone();
        let vram_limits = self.vram_limits.read().unwrap().clone();
        let cuda_vram_static = self.cuda_vram_static.read().unwrap().clone();
        let model_kinds: HashMap<String, DeviceKind> = self
            .model_configs
            .read()
            .unwrap()
            .iter()
            .map(|(n, c)| (n.clone(), c.device_kind()))
            .collect();

        let (slots, static_vram) = match kind {
            DeviceKind::Cuda => (&cuda_slots, &cuda_vram_static),
            DeviceKind::Vulkan => (&vulkan_slots, &vram_limits),
        };
        if pool.is_empty() || slots.is_empty() {
            debug!(model = %model_cfg.name, kind = ?kind, "no devices configured");
            return Vec::new();
        }

        // Translate an instance's device index to a PCI slot using its
        // own model's namespace map.  Instances of models no longer in
        // the config are treated as Vulkan (pre-removal semantics).
        let slot_for = |model: &str, idx: usize| -> Option<String> {
            match model_kinds.get(model).copied().unwrap_or(DeviceKind::Vulkan) {
                DeviceKind::Cuda => cuda_slots.get(&idx).cloned(),
                DeviceKind::Vulkan => vulkan_slots.get(&idx).cloned(),
            }
        };

        let gpus = self.gpu_snapshot.read().await;
        debug!(
            model = %model_cfg.name,
            vram_mb = model_cfg.vram,
            kind = ?kind,
            pool = ?pool,
            gpu_count = gpus.len(),
            "selecting GPU"
        );

        let occupied: std::collections::HashSet<String> = {
            let instances = self.instances.read().unwrap();
            if let Some(list) = instances.get(&model_cfg.name) {
                list.iter()
                    .flat_map(|h| {
                        let inst = h.inner().lock().unwrap();
                        inst.gpu_indices.clone()
                    })
                    .filter_map(|idx| slot_for(&model_cfg.name, idx))
                    .collect()
            } else {
                std::collections::HashSet::new()
            }
        };

        let vram_used: HashMap<String, u64> = {
            let instances = self.instances.read().unwrap();
            let model_configs = self.model_configs.read().unwrap();
            let mut used = HashMap::new();
            for (model_name, list) in instances.iter() {
                let model_vram = model_configs.get(model_name)
                    .map(|c| c.vram * 1024 * 1024)
                    .unwrap_or(0);
                for handle in list {
                    let inst = handle.inner().lock().unwrap();
                    // Attribute the full declared VRAM to *every* occupied
                    // GPU — conservative, but prevents oversubscription
                    // when a multi-device instance spans several GPUs.
                    for &idx in &inst.gpu_indices {
                        if let Some(slot) = slot_for(model_name, idx) {
                            *used.entry(slot).or_default() += model_vram;
                        }
                    }
                }
            }
            used
        };

        let model_vram_bytes = model_cfg.vram * 1024 * 1024;
        let mut candidates: Vec<(usize, u64)> = Vec::new();
        for &idx in pool {
            let pci_slot = match slots.get(&idx) {
                Some(s) => s.as_str(),
                None => {
                    debug!(model = %model_cfg.name, device = idx, "slot not in device map");
                    continue;
                }
            };
            if occupied.contains(pci_slot) {
                debug!(model = %model_cfg.name, device = idx, "skipping — already has instance");
                continue;
            }

            let gpu = gpus.iter().find(|g| g.pci_slot == pci_slot);
            // Capacity rule: with metrics, the configured value (Vulkan
            // vram_limit_mb / CUDA vram_mb) caps the reported total.
            // Without metrics, only CUDA devices may fall back to their
            // static vram_mb — NVIDIA exposes no sysfs VRAM, so a
            // missing nvidia-smi must not make the device unusable.
            let capacity = match (gpu, static_vram.get(&idx)) {
                (Some(g), Some(&configured)) => configured.min(g.vram_total_bytes),
                (Some(g), None) => g.vram_total_bytes,
                (None, Some(&configured)) if kind == DeviceKind::Cuda => configured,
                (None, _) => {
                    debug!(model = %model_cfg.name, device = idx, slot = pci_slot, "GPU not in metrics snapshot");
                    continue;
                }
            };

            let used = vram_used.get(pci_slot).copied().unwrap_or(0);
            let free = capacity.saturating_sub(used);
            debug!(
                model = %model_cfg.name, device = idx, slot = pci_slot,
                vram_capacity_mb = capacity / (1024 * 1024),
                vram_used_mb = used / (1024 * 1024),
                vram_free_mb = free / (1024 * 1024),
                model_mb = model_cfg.vram,
            );
            if free < model_vram_bytes {
                debug!(model = %model_cfg.name, device = idx, "insufficient free VRAM");
                continue;
            }
            candidates.push((idx, free));
        }

        let needed = model_cfg.gpus;
        if candidates.len() < needed {
            debug!(
                model = %model_cfg.name,
                needed,
                available = candidates.len(),
                "not enough GPU candidates"
            );
            return Vec::new();
        }

        let instance_counts: HashMap<String, usize> = {
            let instances = self.instances.read().unwrap();
            let mut counts = HashMap::new();
            for (model_name, list) in instances.iter() {
                for handle in list {
                    let inst = handle.inner().lock().unwrap();
                    for &idx in &inst.gpu_indices {
                        if let Some(slot) = slot_for(model_name, idx) {
                            *counts.entry(slot).or_default() += 1;
                        }
                    }
                }
            }
            counts
        };

        // Deterministic packing: least-loaded GPU first, then most free
        // VRAM, then lowest index.  Selection order is also the emission
        // order into GGML_VK_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES —
        // kept stable because device order has semantics in llama.cpp.
        candidates.sort_by(|(a_idx, a_free), (b_idx, b_free)| {
            let count_of = |idx: &usize| {
                slots
                    .get(idx)
                    .and_then(|slot| instance_counts.get(slot))
                    .copied()
                    .unwrap_or(0)
            };
            count_of(a_idx)
                .cmp(&count_of(b_idx))
                .then(b_free.cmp(a_free))
                .then(a_idx.cmp(b_idx))
        });
        let chosen: Vec<usize> = candidates
            .iter()
            .take(needed)
            .map(|(idx, _)| *idx)
            .collect();
        debug!(model = %model_cfg.name, kind = ?kind, devices = ?chosen, "selected GPUs");
        chosen
    }

    // ── blocked flag ─────────────────────────────────────────────────────

    pub fn is_blocked(&self, model_name: &str) -> bool {
        self.blocked
            .read()
            .unwrap()
            .get(model_name)
            .copied()
            .unwrap_or(false)
    }

    pub fn block_model(&self, model_name: &str) {
        self.blocked
            .write()
            .unwrap()
            .insert(model_name.to_owned(), true);
    }

    pub fn unblock_model(&self, model_name: &str) {
        self.blocked.write().unwrap().remove(model_name);
        self.crash_counts.lock().unwrap().remove(model_name);
    }

    // ── instance removal ─────────────────────────────────────────────────

    /// Remove an instance **if it is idle** (autoscale / TTL eviction).
    ///
    /// A busy instance is left completely untouched — still routable — and
    /// the caller simply tries again on a later cycle.  Returns `true` when
    /// the instance was removed.
    async fn remove_instance(&self, model_name: &str, handle: &InstanceHandle) -> bool {
        self.remove_instance_impl(model_name, handle, RemovalMode::IfIdle)
            .await
    }

    /// Remove an instance, **draining** it when busy (admin unload, config
    /// removal).
    ///
    /// A busy instance is marked `Failed` so no new requests are routed to
    /// it, but its child keeps running until the in-flight requests finish;
    /// `reap_drained` then completes the removal.  Returns `true` when the
    /// instance was removed immediately.
    async fn drain_instance(&self, model_name: &str, handle: &InstanceHandle) -> bool {
        self.remove_instance_impl(model_name, handle, RemovalMode::Drain)
            .await
    }

    /// Forcibly remove an instance, killing it even with in-flight
    /// requests.  Only used by shutdown (after the HTTP drain) and by
    /// `reap_drained` — normal removal paths must never interrupt client
    /// requests.
    async fn force_remove_instance(&self, model_name: &str, handle: &InstanceHandle) {
        self.remove_instance_impl(model_name, handle, RemovalMode::Force)
            .await;
    }

    async fn remove_instance_impl(
        &self,
        model_name: &str,
        handle: &InstanceHandle,
        mode: RemovalMode,
    ) -> bool {
        let gpu_indices: Vec<usize> = {
            let inst = handle.inner().lock().unwrap();
            inst.gpu_indices.clone()
        };

        let mut child = {
            let mut inst = handle.inner().lock().unwrap();
            if inst.in_flight > 0 {
                match mode {
                    // Leave the instance fully intact and routable.
                    RemovalMode::IfIdle => return false,
                    // Mark unroutable but keep the child serving the
                    // in-flight requests; reap_drained finishes the job.
                    RemovalMode::Drain => {
                        if inst.state != InstanceState::Failed {
                            debug!(
                                inst = %inst.id,
                                in_flight = inst.in_flight,
                                "instance busy — marked draining"
                            );
                            inst.state = InstanceState::Failed;
                        }
                        return false;
                    }
                    RemovalMode::Force => {}
                }
            }
            inst.state = InstanceState::Failed;
            inst.child.take()
        };
        if let Some(ref mut c) = child {
            shutdown_child(c, Duration::from_secs(5)).await;
        }

        self.unregister_instance(model_name, handle, &gpu_indices).await;
        true
    }

    /// Finish removing draining instances whose requests have completed.
    ///
    /// Draining instances are marked `Failed` but keep their child alive
    /// until their in-flight count reaches zero.  Called on every slot
    /// release (and periodically from the autoscaler) for `model_name`.
    pub async fn reap_drained(&self, model_name: &str) {
        let drained: Vec<InstanceHandle> = {
            let instances = self.instances.read().unwrap();
            instances
                .get(model_name)
                .map(|list| {
                    list.iter()
                        .filter(|h| {
                            let inst = h.inner().lock().unwrap();
                            inst.state == InstanceState::Failed && inst.in_flight == 0
                        })
                        .cloned()
                        .collect()
                })
                .unwrap_or_default()
        };
        for handle in drained {
            debug!(model = %model_name, inst = %handle.id(), "reaping drained instance");
            self.force_remove_instance(model_name, &handle).await;
        }
    }

    /// Remove an instance from the registry: free its port, fail parked
    /// waiters if it was the last instance, and stop keep-alive on GPUs
    /// that are no longer in use.
    ///
    /// The caller must already have terminated (or reaped) the child
    /// process and marked the instance `Failed`.
    ///
    /// Idempotent: if the handle is no longer registered, nothing happens.
    /// Port frees and keep-alive releases must happen exactly once per
    /// instance — the stranded-`Loading` reaper can race the spawn task's
    /// own failure cleanup for the same handle.
    async fn unregister_instance(
        &self,
        model_name: &str,
        handle: &InstanceHandle,
        gpu_indices: &[usize],
    ) {
        // Remove from the registry first and bail out if already gone.
        {
            let mut instances = self.instances.write().unwrap();
            let removed = match instances.get_mut(model_name) {
                Some(list) => {
                    let before = list.len();
                    // Compare handle identity, not the id string — base IDs
                    // collide for same-GPU/CPU instances of one model.
                    list.retain(|h| !Arc::ptr_eq(h.inner(), handle.inner()));
                    list.len() != before
                }
                None => false,
            };
            if !removed {
                debug!(
                    model = %model_name,
                    inst = %handle.id(),
                    "instance already unregistered — skipping cleanup"
                );
                return;
            }
        }

        let port = handle.inner().lock().unwrap().port;
        self.ports.lock().await.free(port);

        self.drain_queue_if_no_instances(model_name);

        // Keep-alive: release this instance's per-GPU reference (acquired
        // at spawn registration).  The task stops when the last instance
        // leaves the GPU — refcounting makes this safe against concurrent
        // spawns landing on the same GPU.
        let keepalive = self.keepalive.read().unwrap().clone();
        if let Some(ref ka) = keepalive {
            // Paired with the acquire at spawn: keep-alive only ever runs
            // for Vulkan-kind models.  A model removed from the config
            // falls back to the Vulkan translation (pre-removal
            // semantics), matching the pre-CUDA behavior.
            let is_cuda = self
                .model_configs
                .read()
                .unwrap()
                .get(model_name)
                .map(|c| c.device_kind() == DeviceKind::Cuda)
                .unwrap_or(false);
            if !is_cuda {
                let vulkan_slots = self.vulkan_slots.read().unwrap();
                for vulkan_idx in gpu_indices {
                    if let Some(slot) = vulkan_slots.get(vulkan_idx) {
                        ka.release(slot);
                    }
                }
            }
        }
    }

    // ── crash handling ───────────────────────────────────────────────────

    /// Handle an unexpected backend exit reported by a monitor task.
    ///
    /// The instance is unregistered (port freed, parked waiters failed,
    /// keep-alive stopped).  A pre-output crash (instance never reached
    /// `Ready`) counts toward the per-model crash limit; reaching it blocks
    /// the model so subsequent requests fail fast.  Post-output crashes do
    /// not count — the model has proven functional and is respawned on
    /// demand (plan §Backends).
    pub async fn handle_crash(&self, handle: InstanceHandle) {
        let (model_name, was_ready, gpu_indices) = {
            let mut inst = handle.inner().lock().unwrap();
            if inst.state == InstanceState::Failed {
                // Already handled by a managed removal path (admin unload,
                // TTL eviction, health-timeout cleanup, shutdown).
                return;
            }
            let was_ready = inst.state == InstanceState::Ready;
            inst.state = InstanceState::Failed;
            // The monitor already reaped the process via try_wait.
            let _ = inst.child.take();
            (
                inst.model_name.clone(),
                was_ready,
                inst.gpu_indices.clone(),
            )
        };
        let (id, port) = {
            let inst = handle.inner().lock().unwrap();
            (inst.id.clone(), inst.port)
        };
        warn!(
            model = %model_name,
            inst = %id,
            port = port,
            was_ready = was_ready,
            "backend instance exited unexpectedly"
        );
        self.dump_recent_output(&handle);

        self.unregister_instance(&model_name, &handle, &gpu_indices).await;

        if !was_ready {
            self.note_pre_output_crash(&model_name);
        }
    }

    /// Log an instance's buffered backend stdout/stderr at `warn!` level.
    /// Called on failure paths (readiness failure, unexpected exit) — this
    /// is where backend init errors (e.g. llama.cpp "CUDA init failed")
    /// surface, since live output is only forwarded at `debug!` level.
    fn dump_recent_output(&self, handle: &InstanceHandle) {
        let (id, buf) = {
            let inst = handle.inner().lock().unwrap();
            (inst.id.clone(), inst.output.clone())
        };
        let Some(buf) = buf else { return };
        let lines = output_lines(&buf);
        if lines.is_empty() {
            return;
        }
        warn!(inst = %id, lines = lines.len(), "recent backend output:");
        for line in lines {
            warn!(inst = %id, "{line}");
        }
    }

    /// Increment the consecutive pre-output crash counter for `model_name`
    /// and block the model when `crash_limit` is reached.
    fn note_pre_output_crash(&self, model_name: &str) {
        let count = {
            let mut counts = self.crash_counts.lock().unwrap();
            let c = counts.entry(model_name.to_owned()).or_insert(0);
            *c += 1;
            *c
        };
        if count >= self.crash_limit {
            error!(
                model = %model_name,
                crashes = count,
                "model blocked after repeated pre-output crashes"
            );
            self.block_model(model_name);
        } else {
            warn!(
                model = %model_name,
                crashes = count,
                limit = self.crash_limit,
                "pre-output crash"
            );
        }
    }

    pub fn instance_counts(&self) -> HashMap<String, usize> {
        let instances = self.instances.read().unwrap();
        instances
            .iter()
            .map(|(model, list)| (model.clone(), list.len()))
            .collect()
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Unload all instances of a model (admin unload, config removal).
    ///
    /// Idle instances are removed immediately; busy ones are marked
    /// draining (unroutable) and finished by `reap_drained` once their
    /// in-flight requests complete — client requests are never
    /// interrupted.  Returns `(removed, draining)`.
    pub async fn unload_model(&self, model_name: &str) -> (usize, usize) {
        let handles: Vec<InstanceHandle> = {
            let instances = self.instances.read().unwrap();
            instances
                .get(model_name)
                .map(|list| list.iter().cloned().collect())
                .unwrap_or_default()
        };

        let mut removed = 0;
        let mut draining = 0;
        for handle in &handles {
            if self.drain_instance(model_name, handle).await {
                removed += 1;
            } else {
                draining += 1;
            }
        }
        if removed > 0 || draining > 0 {
            info!(model = %model_name, removed, draining, "model unload requested");
        }
        (removed, draining)
    }

    // ── shutdown ─────────────────────────────────────────────────────────

    pub async fn shutdown_all(&self) {
        let all: Vec<(String, InstanceHandle)> = {
            let instances = self.instances.read().unwrap();
            instances
                .iter()
                .flat_map(|(model, list)| list.iter().map(move |h| (model.clone(), h.clone())))
                .collect()
        };

        for (model_name, handle) in all {
            // Clear the release channel sender so remove_instance's drop
            // doesn't fire a spurious release event.
            {
                let mut inst = handle.inner().lock().unwrap();
                inst.release_tx = None;
            }
            // Shutdown: the HTTP server has already drained, but kill even
            // busy instances rather than leaking backend processes.
            self.force_remove_instance(&model_name, &handle).await;
        }

        let keepalive = self.keepalive.read().unwrap().clone();
        if let Some(ref ka) = keepalive {
            ka.stop_all();
        }
    }

    // ── config hot-reload ────────────────────────────────────────────────

    /// Reconcile the manager with a reloaded config.
    ///
    /// Swaps all config-derived state (model configs, cmd aliases, device
    /// maps, port range, keep-alive), unloads instances of models that no
    /// longer exist, and clears blocked flags + crash counters — a reload
    /// is the operator's way to recover a model after fixing the underlying
    /// problem (§5).
    ///
    /// Instances of surviving models whose *spawn-relevant* config changed
    /// (see `fingerprint_with_aliases`) are retired: marked `Failed`
    /// (unroutable, same mechanics as draining) so in-flight requests
    /// finish uninterrupted, then reaped by `reap_drained` once idle.
    /// Replacements are spawned on demand under the new config — retired
    /// instances don't count toward the instance cap (see `try_spawn`),
    /// while VRAM accounting still includes them, so a replacement is
    /// spawned exactly when resources allow.  Changes to routing-only
    /// fields (`max_concurrent`, `queue_depth`, `idle_ttl`, autoscale, …)
    /// keep running instances untouched.
    pub async fn reconcile_config(&self, config: &crate::config::Config) {
        let new_model_configs: HashMap<String, ModelConfig> = config
            .models
            .iter()
            .map(|m| (m.name.clone(), m.clone()))
            .collect();

        // ── Unload instances of removed models ───────────────────────
        let removed: Vec<String> = {
            let instances = self.instances.read().unwrap();
            instances
                .keys()
                .filter(|m| !new_model_configs.contains_key(*m))
                .cloned()
                .collect()
        };
        for model in &removed {
            info!(model = %model, "model removed from config — unloading instances");
            self.unload_model(model).await;
        }

        // ── Retire instances of surviving models whose spawn config
        //    changed ─────────────────────────────────────────────────
        // Computed from the *new* config and *new* cmd aliases directly
        // (not the swapped manager state), so this is independent of the
        // swap order below and takes no config locks while holding the
        // instances lock.
        for (model_name, new_cfg) in &new_model_configs {
            let fp = fingerprint_with_aliases(&config.cmd_aliases, new_cfg);
            let stale: Vec<InstanceHandle> = {
                let instances = self.instances.read().unwrap();
                instances
                    .get(model_name)
                    .map(|list| {
                        list.iter()
                            .filter(|h| {
                                let inst = h.inner().lock().unwrap();
                                inst.state != InstanceState::Failed
                                    && inst.config_fingerprint != fp
                            })
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default()
            };
            for handle in stale {
                let (id, in_flight) = {
                    let mut inst = handle.inner().lock().unwrap();
                    inst.state = InstanceState::Failed;
                    (inst.id.clone(), inst.in_flight)
                };
                info!(
                    model = %model_name,
                    inst = %id,
                    in_flight,
                    "spawn config changed — retiring instance (draining)"
                );
            }
        }

        // ── Swap config-derived fields ───────────────────────────────
        *self.model_configs.write().unwrap() = new_model_configs;
        *self.cmd_aliases.write().unwrap() = config.cmd_aliases.clone();
        *self.vulkan_slots.write().unwrap() = config
            .devices
            .as_ref()
            .map(|d| d.vulkan.clone())
            .unwrap_or_default();
        *self.vram_limits.write().unwrap() = config
            .devices
            .as_ref()
            .map(|d| {
                d.vram_limit_mb
                    .iter()
                    .map(|(k, v)| (*k, *v * 1024 * 1024))
                    .collect()
            })
            .unwrap_or_default();
        let (cuda_slots, cuda_vram_static) = cuda_device_maps(config);
        *self.cuda_slots.write().unwrap() = cuda_slots;
        *self.cuda_vram_static.write().unwrap() = cuda_vram_static;
        self.ports
            .lock()
            .await
            .set_range(config.server.port_range.clone());
        self.loading_dots_interval_secs.store(
            config.server.loading_dots.unwrap_or(0),
            Ordering::Relaxed,
        );

        // ── Keep-alive: rebuild when the config changed ──────────────
        let rebuild = match &*self.keepalive.read().unwrap() {
            Some(ka) => !ka.matches(&config.keep_alive),
            None => config.keep_alive.is_some(),
        };
        if rebuild {
            {
                let mut slot = self.keepalive.write().unwrap();
                if let Some(old) = slot.take() {
                    old.stop_all();
                }
                *slot = KeepAliveManager::new(&config.keep_alive).map(Arc::new);
            }
            // Re-acquire keep-alive for GPUs with running instances — the
            // fresh manager starts with no tasks and zero refcounts, so
            // acquire once per instance-GPU pair to rebuild the counts.
            let keepalive = self.keepalive.read().unwrap().clone();
            if let Some(ref ka) = keepalive {
                let in_use_slots: Vec<String> = {
                    let instances = self.instances.read().unwrap();
                    let vulkan_slots = self.vulkan_slots.read().unwrap();
                    let model_configs = self.model_configs.read().unwrap();
                    instances
                        .iter()
                        // Keep-alive is Vulkan-only — CUDA instances never
                        // acquired a reference at spawn.
                        .filter(|(name, _)| {
                            model_configs
                                .get(*name)
                                .map(|c| c.device_kind() == DeviceKind::Vulkan)
                                .unwrap_or(true)
                        })
                        .flat_map(|(_, list)| list)
                        .flat_map(|h| h.inner().lock().unwrap().gpu_indices.clone())
                        .filter_map(|idx| vulkan_slots.get(&idx).cloned())
                        .collect()
                };
                for slot in in_use_slots {
                    ka.acquire(&slot);
                }
            }
            info!("keep-alive manager rebuilt after config reload");
        }

        // ── Clear crash blocks (§5) ──────────────────────────────────
        let had_blocked = !self.blocked.read().unwrap().is_empty();
        self.blocked.write().unwrap().clear();
        self.crash_counts.lock().unwrap().clear();
        if had_blocked {
            info!("crash-blocked models cleared by config reload");
        }
    }

    // ── Metrics ──────────────────────────────────────────────────────

    pub fn queue_depth(&self, model_name: &str) -> usize {
        self.queues
            .read()
            .unwrap()
            .get(model_name)
            .map(|q| q.len())
            .unwrap_or(0)
    }

    /// Record a metrics event for `model_name`.
    /// `completions_delta` is 0 for acquires, 1 for releases (counted
    /// regardless of request outcome — see `ModelMetrics::req_rate_m1`).
    ///
    /// Called from `get_or_spawn` / `enqueue` (acquire path) and from the
    /// background release-processing task (release path — the task the
    /// caller of `InstanceManager::new` must spawn for `release_rx`).
    /// In both cases the caller holds **no locks**, so we can safely
    /// acquire `instances.read()` → `model_metrics.write()` without
    /// deadlock.
    pub fn record_metrics_event(&self, model_name: &str, completions_delta: u64) {
        let in_flight: usize = self
            .instances
            .read()
            .unwrap()
            .get(model_name)
            .map(|list| list.iter().map(|h| h.inner().lock().unwrap().in_flight).sum())
            .unwrap_or(0);
        let queued = self.queue_depth(model_name);
        let active = in_flight + queued;

        let now = Instant::now();
        let mut metrics_map = self.model_metrics.write().unwrap();
        let m = metrics_map.entry(model_name.to_owned()).or_default();
        m.tick(now, active, completions_delta);
    }

    /// Bump `last_activity` for `model_name` without advancing the EMAs.
    ///
    /// Called on spawn intent (when a new instance is registered as Loading)
    /// so the idle-TTL despawn check never acts on a stale timestamp left
    /// over from a previous despawn→respawn cycle.
    pub(crate) fn touch_activity(&self, model_name: &str) {
        let mut metrics_map = self.model_metrics.write().unwrap();
        metrics_map
            .entry(model_name.to_owned())
            .or_default()
            .last_activity = Instant::now();
    }

    /// Force-refresh metrics for all models to `now`.
    ///
    /// Called by `/v1/info` and `/admin/status` to ensure returned load
    /// numbers are never stale.  Collects data under `instances.read()`,
    /// drops it, then acquires `model_metrics.write()` — no lock inversion.
    pub fn force_refresh(&self) {
        let now = Instant::now();

        let snap: Vec<(String, usize)> = {
            let instances = self.instances.read().unwrap();
            let queues = self.queues.read().unwrap();
            instances
                .iter()
                .map(|(model_name, list)| {
                    let in_flight: usize = list
                        .iter()
                        .map(|h| h.inner().lock().unwrap().in_flight)
                        .sum();
                    let queued = queues.get(model_name).map(|q| q.len()).unwrap_or(0);
                    (model_name.clone(), in_flight + queued)
                })
                .collect()
        };

        let mut metrics_map = self.model_metrics.write().unwrap();
        for (model_name, active) in snap {
            let m = metrics_map.entry(model_name).or_default();
            m.tick(now, active, 0);
        }
    }

    pub fn model_metrics_snapshot(&self) -> HashMap<String, ModelMetrics> {
        self.model_metrics.read().unwrap().clone()
    }

    /// Record a completed request in the per-model ring buffer.
    /// Keeps at most 5 entries per model, newest first.
    pub fn record_completion(
        &self,
        model_name: &str,
        record: CompletionRecord,
    ) {
        info!(
            model = %model_name,
            user = %record.api_user,
            prompt = record.prompt_tokens,
            generated = record.generated_tokens,
            cached = record.cached_tokens,
            duration_ms = record.duration_ms,
            "completion recorded"
        );
        let mut completions = self.recent_completions.write().unwrap();
        let entries = completions.entry(model_name.to_owned()).or_default();
        entries.insert(0, record);
        entries.truncate(5);
    }

    /// Snapshot of recent completions per model for the info endpoint.
    pub fn recent_completions_snapshot(
        &self,
    ) -> HashMap<String, Vec<CompletionRecord>> {
        self.recent_completions
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    // ── Autoscaler ────────────────────────────────────────────────────

    /// Evaluate autoscaling decisions for all models.
    ///
    /// Called periodically by a background task.  Uses `load_m5` for
    /// scale-up decisions and `load_m15` for scale-down, with hysteresis
    /// between the two thresholds and a per-model cooldown.
    pub async fn evaluate_autoscale(self: &Arc<Self>) {
        // Advance EMAs to now and bump last_activity for every model with
        // in-flight or queued requests.  Ticks otherwise only happen on
        // acquire/release events, so without this a single long-running
        // request (large context, many minutes) would let last_activity go
        // stale mid-request and trip the idle-TTL despawn below.
        self.force_refresh();
        let now = Instant::now();
        let metrics = self.model_metrics_snapshot();

        // Snapshot configs for the iteration (cheap clone, small structs).
        let configs: Vec<(String, ModelConfig)> = self
            .model_configs
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        for (model_name, cfg) in &configs {
            // Safety net: finish removals of draining instances whose
            // in-flight requests have completed.
            self.reap_drained(model_name).await;

            // Safety net: reap instances stranded in `Loading` far beyond
            // the spawn window (e.g. after a panicked spawn task).  Loading
            // instances can't hold slots, so `last_active` ≈ registration
            // time; a legitimately loading instance never exceeds
            // `spawn_timeout` plus the serialized window by a wide margin.
            let stranded: Vec<InstanceHandle> = {
                let instances = self.instances.read().unwrap();
                instances
                    .get(model_name)
                    .map(|list| {
                        list.iter()
                            .filter(|h| {
                                let inst = h.inner().lock().unwrap();
                                inst.state == InstanceState::Loading
                                    && inst.last_active.elapsed() > self.spawn_timeout * 2
                            })
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default()
            };
            for handle in stranded {
                warn!(
                    model = %model_name,
                    inst = %handle.id(),
                    "reaping instance stranded in Loading state"
                );
                self.force_remove_instance(model_name, &handle).await;
            }

            let num_instances = {
                self.instances.read().unwrap()
                    .get(model_name)
                    .map(|l| l.len())
                    .unwrap_or(0)
            };
            if num_instances == 0 {
                continue;
            }

            let m = match metrics.get(model_name) {
                Some(m) => m,
                None => continue,
            };

            // ── Final despawn (n → 0): idle-TTL on the last instance ────
            // Always checked — does not require autoscale to be enabled.
            // The instance-level check is authoritative: Ready, no
            // in-flight requests, and last_active at least idle_ttl ago —
            // the configured TTL is honored exactly, independent of load
            // EMA decay timescales.  Loading instances (spawn in progress)
            // are excluded because their state != Ready and their
            // last_active is fresh; remove_instance is atomic w.r.t. slot
            // acquisition, so a request racing the despawn is a no-op.
            if num_instances == 1 {
                let mut despawned = false;
                if let Some(h) = self.pick_least_loaded(model_name) {
                    if h.inner().lock().unwrap().is_idle_expired(cfg.idle_ttl) {
                        info!(
                            model = %model_name,
                            ttl = cfg.idle_ttl,
                            "idle TTL expired, despawning last instance"
                        );
                        despawned = self.remove_instance(model_name, &h).await;
                    }
                }
                if despawned {
                    continue;
                }
                // Fall through: autoscale scale-up may still apply.
            }

            // ── Autoscale (n ↔ n±1): requires autoscale.enabled ────────
            let a = match &cfg.autoscale {
                Some(a) if a.enabled => a,
                _ => {
                    // No autoscale: TTL-evict surplus idle instances down
                    // to one (the n → 0 despawn is handled above).
                    let mut n = num_instances;
                    for handle in self.instances_by_load(model_name) {
                        if n <= 1 {
                            break;
                        }
                        if handle.inner().lock().unwrap().is_idle_expired(cfg.idle_ttl) {
                            info!(
                                model = %model_name,
                                inst = %handle.id(),
                                ttl = cfg.idle_ttl,
                                "idle TTL expired, evicting surplus instance"
                            );
                            if self.remove_instance(model_name, &handle).await {
                                n -= 1;
                            }
                        }
                    }
                    continue;
                }
            };

            // ── Cooldown ──────────────────────────────────────────
            {
                let last = self.last_scale_action.read().unwrap();
                if let Some(t) = last.get(model_name) {
                    if now.duration_since(*t).as_secs() < a.cooldown_secs {
                        continue;
                    }
                }
            }

            // ── Scale-up (n → n+1) ─────────────────────────────
            // Proactive: spawn when sustained load exceeds threshold,
            // even if no incoming request triggers the request-path gate.
            if num_instances < cfg.max_instances {
                let capacity = (cfg.max_concurrent * num_instances) as f64;
                if m.load_m5 > a.scale_up_at * capacity {
                    // A request-driven spawn may already be in flight —
                    // don't double-spawn past the cap.
                    if self.spawns_in_flight.read().unwrap().contains_key(model_name) {
                        debug!(
                            model = %model_name,
                            "autoscale: spawn already in flight, skipping scale-up"
                        );
                        continue;
                    }
                    info!(
                        model = %model_name,
                        load_m5 = %m.load_m5,
                        threshold = %(a.scale_up_at * capacity),
                        "autoscale: scaling up"
                    );
                    // Fire-and-forget: the spawn runs in a detached task
                    // (see `ensure_spawn`) which wakes parked waiters on
                    // success; requests subscribe to the entry directly.
                    let _rx = self.ensure_spawn(model_name, cfg);
                    self.last_scale_action.write().unwrap()
                        .insert(model_name.clone(), now);
                    continue;
                }
            }

            // ── Scale-down (n → n−1) ─────────────────────────────
            if num_instances > 1 {
                let reduced_cap = (cfg.max_concurrent * (num_instances - 1)) as f64;
                if m.load_m15 < a.scale_down_at * reduced_cap {
                    // Prefer the least-loaded instance, but skip busy ones
                    // entirely — in-flight requests must never be
                    // interrupted.  remove_instance is atomic with respect
                    // to slot acquisition, so a lost race is a no-op.
                    for handle in self.instances_by_load(model_name) {
                        if self.remove_instance(model_name, &handle).await {
                            info!(
                                model = %model_name,
                                load_m15 = %m.load_m15,
                                threshold = %(a.scale_down_at * reduced_cap),
                                "autoscale: scaled down"
                            );
                            self.last_scale_action.write().unwrap()
                                .insert(model_name.clone(), now);
                            break;
                        }
                    }
                    continue;
                }
            }
        }
    }

    /// Return the instance handle with the fewest in-flight requests for
    /// `model_name`, if any exist.
    fn pick_least_loaded(&self, model_name: &str) -> Option<InstanceHandle> {
        let instances = self.instances.read().unwrap();
        let list = instances.get(model_name)?;
        list.iter()
            .min_by_key(|h| h.inner().lock().unwrap().in_flight)
            .cloned()
    }

    /// Instance handles for `model_name`, sorted by in-flight count
    /// (ascending).
    fn instances_by_load(&self, model_name: &str) -> Vec<InstanceHandle> {
        let instances = self.instances.read().unwrap();
        let mut list: Vec<InstanceHandle> = instances
            .get(model_name)
            .map(|l| l.iter().cloned().collect())
            .unwrap_or_default();
        list.sort_by_key(|h| h.inner().lock().unwrap().in_flight);
        list
    }
}

// ── Spawn bookkeeping ────────────────────────────────────────────────────────

/// RAII guard that removes a model's in-flight spawn entry on drop, waking
/// all requests awaiting the spawn's completion.  Lives in the detached
/// spawn task; runs on normal completion and on panic unwind.
struct SpawnEntryGuard {
    mgr: Arc<InstanceManager>,
    model_name: String,
}

impl Drop for SpawnEntryGuard {
    fn drop(&mut self) {
        self.mgr
            .spawns_in_flight
            .write()
            .unwrap()
            .remove(&self.model_name);
    }
}

// ── Crash monitoring ─────────────────────────────────────────────────────────

/// Poll interval for the per-instance child-exit monitor.
const CRASH_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Watch an instance's child process; on unexpected exit, forward the handle
/// to the manager's crash-processing task via `crash_tx`.
///
/// Exits quietly when the instance is going through a managed removal (state
/// set to `Failed` / child taken by `remove_instance` or shutdown), so only
/// *unexpected* exits are reported.  Managed removal races the report: the
/// crash handler re-checks the state and ignores already-`Failed` instances.
async fn monitor_instance_exit(
    handle: InstanceHandle,
    crash_tx: mpsc::UnboundedSender<InstanceHandle>,
) {
    loop {
        tokio::time::sleep(CRASH_POLL_INTERVAL).await;
        let exited = {
            let mut inst = handle.inner().lock().unwrap();
            if inst.state == InstanceState::Failed {
                return;
            }
            match inst.child.as_mut() {
                None => return,
                Some(child) => match child.try_wait() {
                    Ok(Some(_status)) => true,
                    Ok(None) => false,
                    // Wait error — treat as exit; erring toward unregistering
                    // keeps a wedged instance out of the routing pool.
                    Err(_) => true,
                },
            }
        };
        if exited {
            let _ = crash_tx.send(handle);
            return;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_CONFIG_YAML: &str = r#"
server: {}
apikeys_file: apikeys.txt
models:
  - name: m
    context_length: 4096
    cmd: "sleep 3600"
    idle_ttl: 60
"#;

    fn test_manager() -> InstanceManager {
        let config: crate::config::Config =
            serde_yaml_ng::from_str(TEST_CONFIG_YAML).unwrap();
        let gpu_snapshot = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let (mgr, _release_rx, _crash_rx) =
            InstanceManager::new(&config, gpu_snapshot, None);
        mgr
    }

    fn make_handle(state: InstanceState, in_flight: usize, idle_for: Duration) -> InstanceHandle {
        make_handle_on_gpus(vec![0], state, in_flight, idle_for)
    }

    fn make_handle_on_gpus(
        gpus: Vec<usize>,
        state: InstanceState,
        in_flight: usize,
        idle_for: Duration,
    ) -> InstanceHandle {
        let mut inst = Instance::new("m", gpus, 54321, None);
        inst.state = state;
        inst.in_flight = in_flight;
        inst.last_active = Instant::now() - idle_for;
        InstanceHandle::new(inst)
    }

    /// Stamp the handle with the fingerprint the manager computes for
    /// model "m" under the current config (as `try_spawn` does at spawn).
    fn set_current_fingerprint(mgr: &InstanceManager, handle: &InstanceHandle) {
        let cfg = mgr.model_configs.read().unwrap().get("m").cloned().unwrap();
        let fp = mgr.spawn_fingerprint(&cfg);
        handle.inner().lock().unwrap().config_fingerprint = fp;
    }

    /// Insert a handle and a stale metrics entry (as left over from a
    /// previous despawn→respawn cycle) for model "m".
    fn register_with_stale_metrics(mgr: &InstanceManager, handle: &InstanceHandle) {
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());
        let stale = Instant::now() - Duration::from_secs(3600);
        let mut metrics = mgr.model_metrics.write().unwrap();
        metrics.insert(
            "m".to_owned(),
            ModelMetrics {
                last_activity: stale,
                ..Default::default()
            },
        );
    }

    fn instance_count(mgr: &InstanceManager) -> usize {
        mgr.instances
            .read()
            .unwrap()
            .get("m")
            .map(|l| l.len())
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn autoscale_does_not_despawn_loading_instance_with_stale_metrics() {
        // Regression test for the respawn race: after a TTL despawn, the
        // metrics entry keeps a stale last_activity.  A respawn triggered by
        // a new request registers a Loading instance; the autoscaler must
        // not kill it mid-spawn based on that stale timestamp.
        let mgr = Arc::new(test_manager());
        let handle = make_handle(InstanceState::Loading, 0, Duration::ZERO);
        register_with_stale_metrics(&mgr, &handle);

        mgr.evaluate_autoscale().await;

        assert_eq!(
            instance_count(&mgr),
            1,
            "Loading instance must survive the autoscaler tick"
        );
        let inst = handle.inner().lock().unwrap();
        assert_eq!(inst.state, InstanceState::Loading);
        assert!(inst.child.is_none() || inst.state != InstanceState::Failed);
    }

    #[tokio::test]
    async fn autoscale_keeps_instance_with_in_flight_request_alive() {
        // A long-running request produces no acquire/release events for many
        // minutes.  force_refresh at the start of evaluate_autoscale must
        // bump last_activity (active > 0) so the idle TTL never trips
        // mid-request.
        let mgr = Arc::new(test_manager());
        // Instance itself has been busy on one request for over an hour.
        let handle = make_handle(InstanceState::Ready, 1, Duration::from_secs(3600));
        register_with_stale_metrics(&mgr, &handle);

        mgr.evaluate_autoscale().await;

        assert_eq!(
            instance_count(&mgr),
            1,
            "instance with an in-flight request must not be despawned"
        );
        let last = mgr.model_metrics.read().unwrap()["m"].last_activity;
        assert!(
            last.elapsed().as_secs() < 5,
            "last_activity must have been refreshed by the in-flight request"
        );
    }

    #[tokio::test]
    async fn autoscale_still_despawns_genuinely_idle_instance() {
        // The despawn path itself must keep working: Ready, no in-flight,
        // idle past TTL at both instance and metrics level.
        let mgr = Arc::new(test_manager());
        let handle = make_handle(InstanceState::Ready, 0, Duration::from_secs(3600));
        register_with_stale_metrics(&mgr, &handle);

        mgr.evaluate_autoscale().await;

        assert_eq!(
            instance_count(&mgr),
            0,
            "genuinely idle instance past TTL must be despawned"
        );
    }

    #[tokio::test]
    async fn enqueue_refused_when_no_instances_exist() {
        // Spawn failure with zero instances registered must not park the
        // request forever: nothing would ever wake the waiter.
        let mgr = test_manager();
        let result = mgr.enqueue("m", 8).await;
        assert!(
            matches!(result, Err(EnqueueError::NoInstances)),
            "enqueue with no instances must fail fast"
        );
    }

    #[tokio::test]
    async fn parked_waiter_released_when_last_instance_removed() {
        // A queued request must fail fast when the last instance goes away
        // (despawn / crash / admin unload) instead of hanging forever.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        let (tx, rx) = oneshot::channel();
        mgr.queues
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push_back((0, tx));

        mgr.remove_instance("m", &handle).await;

        assert!(
            rx.await.is_err(),
            "waiter must be released when the last instance is removed"
        );
    }

    #[tokio::test]
    async fn parked_waiter_kept_when_other_instances_remain() {
        // Scale-down (n → n−1) must not disturb queued waiters.
        let mgr = test_manager();
        let h1 = make_handle_on_gpus(vec![0], InstanceState::Ready, 0, Duration::ZERO);
        let h2 = make_handle_on_gpus(vec![1], InstanceState::Ready, 0, Duration::ZERO);
        {
            let mut instances = mgr.instances.write().unwrap();
            let list = instances.entry("m".to_owned()).or_default();
            list.push(h1.clone());
            list.push(h2);
        }

        let (tx, rx) = oneshot::channel::<SlotGuard>();
        mgr.queues
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push_back((0, tx));

        mgr.remove_instance("m", &h1).await;

        assert_eq!(instance_count(&mgr), 1);
        assert!(
            !rx.is_terminated(),
            "waiter must stay parked while instances remain"
        );
    }

    #[tokio::test]
    async fn enqueue_self_wakes_when_capacity_already_free() {
        // Lost-wakeup regression: capacity freed *before* the waiter parks
        // must still serve it — the post-push wake_one re-signals instead
        // of leaving the waiter parked on a free slot.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        let guard = mgr.enqueue("m", 8).await;
        assert!(guard.is_ok(), "waiter must be served by the post-push wake");
        assert_eq!(handle.inner().lock().unwrap().in_flight, 1);
    }

    #[tokio::test]
    async fn wake_one_transfers_acquired_slot_to_waiter() {
        // The wake must transfer an already-acquired slot: the waiter
        // receives a SlotGuard, so it can't lose an acquire race — there
        // is no acquire on its side.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 1, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());
        let (tx, rx) = oneshot::channel::<SlotGuard>();
        mgr.queues
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push_back((0, tx));

        mgr.wake_one("m");

        let guard = rx.await.expect("waiter must receive a slot");
        assert_eq!(
            handle.inner().lock().unwrap().in_flight,
            2,
            "the slot must be acquired atomically with the wake"
        );
        drop(guard);
        assert_eq!(handle.inner().lock().unwrap().in_flight, 1);
    }

    #[tokio::test]
    async fn wake_one_reparks_waiter_at_head_when_capacity_vanishes() {
        // Instance full (in_flight = max_concurrent = 4, the config
        // default): wake_one must re-park the waiter at the head of the
        // queue, preserving FIFO order, instead of dropping it.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 4, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);
        let (tx, rx) = oneshot::channel::<SlotGuard>();
        {
            let mut queues = mgr.queues.write().unwrap();
            let q = queues.entry("m".to_owned()).or_default();
            q.push_back((7, tx));
        }

        mgr.wake_one("m");

        assert_eq!(mgr.queue_depth("m"), 1, "waiter must be re-parked");
        assert!(!rx.is_terminated(), "waiter must stay parked");
        let queues = mgr.queues.read().unwrap();
        assert_eq!(
            queues["m"][0].0, 7,
            "re-parked waiter must be back at the head"
        );
    }

    #[tokio::test]
    async fn remove_queued_drops_only_the_matching_entry() {
        // Timeout path: a timed-out waiter removes its own entry by id;
        // other parked waiters must stay in place.
        let mgr = test_manager();
        let (tx1, rx1) = oneshot::channel::<SlotGuard>();
        let (tx2, rx2) = oneshot::channel::<SlotGuard>();
        {
            let mut queues = mgr.queues.write().unwrap();
            let q = queues.entry("m".to_owned()).or_default();
            q.push_back((1, tx1));
            q.push_back((2, tx2));
        }

        mgr.remove_queued("m", 1);

        assert!(rx1.await.is_err(), "removed waiter's sender must be dropped");
        assert!(!rx2.is_terminated(), "other waiter must stay parked");
        assert_eq!(mgr.queue_depth("m"), 1);
    }

    #[tokio::test]
    async fn pre_output_crashes_block_model_after_limit() {
        // crash_limit = 3 consecutive pre-output crashes → model blocked
        // and the crashed instances unregistered.
        let mgr = test_manager();
        for _ in 0..3 {
            let handle = make_handle(InstanceState::Loading, 0, Duration::ZERO);
            mgr.instances
                .write()
                .unwrap()
                .entry("m".to_owned())
                .or_default()
                .push(handle.clone());
            mgr.handle_crash(handle).await;
        }
        assert!(mgr.is_blocked("m"), "model must be blocked after 3 pre-output crashes");
        assert_eq!(instance_count(&mgr), 0, "crashed instances must be unregistered");
    }

    #[tokio::test]
    async fn post_output_crashes_do_not_block_model() {
        // Instances that reached Ready before crashing must not count
        // toward the crash-block limit (plan §Backends).
        let mgr = test_manager();
        for _ in 0..5 {
            let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
            mgr.instances
                .write()
                .unwrap()
                .entry("m".to_owned())
                .or_default()
                .push(handle.clone());
            mgr.handle_crash(handle).await;
        }
        assert!(!mgr.is_blocked("m"), "post-output crashes must not block the model");
        assert_eq!(instance_count(&mgr), 0, "crashed instances must be unregistered");
    }

    #[tokio::test]
    async fn crash_of_already_failed_instance_is_ignored() {
        // A monitor report racing a managed removal (state already Failed)
        // must be a no-op — no unregister, no crash count.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Failed, 0, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());
        mgr.handle_crash(handle).await;
        assert!(!mgr.is_blocked("m"));
        assert_eq!(instance_count(&mgr), 1, "already-failed instance must be left alone");
    }

    #[tokio::test]
    async fn unblock_resets_crash_counter() {
        // Two crashes, unblock (resets the counter), two more crashes —
        // with limit 3 the model must not be blocked.
        let mgr = test_manager();
        for _ in 0..2 {
            let handle = make_handle(InstanceState::Loading, 0, Duration::ZERO);
            mgr.instances
                .write()
                .unwrap()
                .entry("m".to_owned())
                .or_default()
                .push(handle.clone());
            mgr.handle_crash(handle).await;
        }
        mgr.unblock_model("m");
        for _ in 0..2 {
            let handle = make_handle(InstanceState::Loading, 0, Duration::ZERO);
            mgr.instances
                .write()
                .unwrap()
                .entry("m".to_owned())
                .or_default()
                .push(handle.clone());
            mgr.handle_crash(handle).await;
        }
        assert!(
            !mgr.is_blocked("m"),
            "counter reset on unblock — two fresh crashes must not reach the limit"
        );
    }

    const TEST_CONFIG_YAML_B: &str = r#"
server: {}
apikeys_file: apikeys.txt
models:
  - name: n
    context_length: 4096
    cmd: "sleep 3600"
    idle_ttl: 60
"#;

    #[tokio::test]
    async fn reconcile_unloads_removed_models_and_swaps_configs() {
        // Hot-reload must reach the manager: removed models are unloaded,
        // the new model config becomes spawnable, crash blocks are cleared.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);
        mgr.block_model("m");

        let cfg_b: crate::config::Config =
            serde_yaml_ng::from_str(TEST_CONFIG_YAML_B).unwrap();
        mgr.reconcile_config(&cfg_b).await;

        assert_eq!(instance_count(&mgr), 0, "removed model's instances must be unloaded");
        assert!(!mgr.model_configs.read().unwrap().contains_key("m"));
        assert!(mgr.model_configs.read().unwrap().contains_key("n"));
        assert!(!mgr.is_blocked("m"), "reload must clear blocked flags");
    }

    #[tokio::test]
    async fn reconcile_keeps_surviving_models_instances() {
        // Models still present in the reloaded config with an unchanged
        // spawn fingerprint keep their running instances.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        set_current_fingerprint(&mgr, &handle);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        let cfg: crate::config::Config =
            serde_yaml_ng::from_str(TEST_CONFIG_YAML).unwrap();
        mgr.reconcile_config(&cfg).await;

        assert_eq!(instance_count(&mgr), 1, "surviving model's instance must be kept");
        assert_eq!(
            handle.inner().lock().unwrap().state,
            InstanceState::Ready,
            "unchanged spawn config must keep the instance routable"
        );
    }

    #[tokio::test]
    async fn reconcile_retires_stale_instances_and_reaps_after_drain() {
        // A change to a spawn-relevant field (cmd) retires the running
        // instance: marked unroutable immediately, but the in-flight
        // request finishes; the instance is reaped once idle.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 1, Duration::ZERO);
        set_current_fingerprint(&mgr, &handle);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        let yaml = TEST_CONFIG_YAML.replace("sleep 3600", "sleep 3601");
        let cfg: crate::config::Config = serde_yaml_ng::from_str(&yaml).unwrap();
        mgr.reconcile_config(&cfg).await;

        assert_eq!(
            handle.inner().lock().unwrap().state,
            InstanceState::Failed,
            "stale instance must be marked unroutable"
        );
        assert_eq!(
            instance_count(&mgr),
            1,
            "stale instance with an in-flight request must keep serving"
        );
        assert!(
            mgr.find_ready_instance("m", 4).is_none(),
            "stale instance must not receive new requests"
        );
        assert!(handle.try_acquire(4).is_none());

        // The in-flight request completes → reap finishes the removal.
        handle.inner().lock().unwrap().in_flight = 0;
        mgr.reap_drained("m").await;
        assert_eq!(instance_count(&mgr), 0, "retired instance must be reaped");
    }

    #[tokio::test]
    async fn reconcile_keeps_instances_when_only_routing_fields_change() {
        // max_concurrent / queue_depth / idle_ttl changes apply to routing
        // immediately and must not retire running instances.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        set_current_fingerprint(&mgr, &handle);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
models:
  - name: m
    context_length: 4096
    cmd: "sleep 3600"
    idle_ttl: 30
    max_concurrent: 8
    queue_depth: 3
"#;
        let cfg: crate::config::Config = serde_yaml_ng::from_str(yaml).unwrap();
        mgr.reconcile_config(&cfg).await;

        assert_eq!(
            handle.inner().lock().unwrap().state,
            InstanceState::Ready,
            "routing-only config changes must not retire instances"
        );
    }

    #[tokio::test]
    async fn spawn_cap_excludes_draining_instances() {
        // A draining/retired instance must not block a replacement spawn:
        // with max_instances = 1 and one Failed instance, try_spawn must
        // still attempt the spawn (observable via the crash counter the
        // instantly-failing "true" command produces).
        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
models:
  - name: m
    context_length: 4096
    cmd: "true"
    idle_ttl: 60
    max_instances: 1
"#;
        let config: crate::config::Config = serde_yaml_ng::from_str(yaml).unwrap();
        let gpu_snapshot = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let (mgr, _release_rx, _crash_rx) =
            InstanceManager::new(&config, gpu_snapshot, None);
        let draining = make_handle(InstanceState::Failed, 1, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(draining);

        let cfg = mgr.model_configs.read().unwrap().get("m").cloned().unwrap();
        // "true" exits instantly → spawn attempt fails fast, but it must
        // have been *attempted* (a cap refusal would not count a crash).
        assert!(mgr.try_spawn("m", &cfg).await.is_none());
        assert_eq!(
            mgr.crash_counts.lock().unwrap().get("m").copied(),
            Some(1),
            "spawn must have been attempted despite the draining instance"
        );
    }

    #[tokio::test]
    async fn enqueue_refused_when_only_draining_instances_exist() {
        // All remaining instances retiring (Failed): no release event can
        // ever serve a parked waiter, so enqueue must fail fast.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Failed, 0, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);

        let result = mgr.enqueue("m", 8).await;
        assert!(
            matches!(result, Err(EnqueueError::NoInstances)),
            "enqueue with only retiring instances must fail fast"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn enqueue_times_out_when_never_woken() {
        // All instances busy (in_flight = max_concurrent = 4, the config
        // default), nothing ever releases: the waiter must eventually
        // fail with Timeout (paused time auto-advances).
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 4, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);

        let result = mgr.enqueue("m", 8).await;
        assert!(matches!(result, Err(EnqueueError::Timeout)), "got: {:?}", result.map(|_| "guard"));
        assert_eq!(mgr.queue_depth("m"), 0, "timed-out waiter must self-remove");
    }

    #[tokio::test]
    async fn get_or_spawn_reports_blocked_and_unknown_models() {
        let mgr = Arc::new(test_manager());
        mgr.block_model("m");
        assert!(matches!(
            mgr.get_or_spawn("m").await,
            Err(AcquireError::Blocked)
        ));
        assert!(matches!(
            mgr.get_or_spawn("nonexistent").await,
            Err(AcquireError::Unavailable)
        ));
    }

    #[tokio::test]
    async fn ensure_instance_reports_spawn_failure_without_queueing() {
        // `cmd: "true"` exits instantly: the spawn fails fast and
        // ensure_instance must report Unavailable (not park on a queue).
        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
models:
  - name: m
    context_length: 4096
    cmd: "true"
    idle_ttl: 60
"#;
        let config: crate::config::Config = serde_yaml_ng::from_str(yaml).unwrap();
        let gpu_snapshot = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let (mgr, _release_rx, _crash_rx) =
            InstanceManager::new(&config, gpu_snapshot, None);
        let mgr = Arc::new(mgr);

        let result = tokio::time::timeout(
            Duration::from_secs(15),
            mgr.ensure_instance("m"),
        )
        .await
        .expect("ensure_instance must not park");
        assert!(matches!(result, Err(AcquireError::Unavailable)));

        mgr.block_model("m");
        assert!(matches!(
            mgr.ensure_instance("m").await,
            Err(AcquireError::Blocked)
        ));
    }

    #[test]
    fn fingerprint_covers_cmd_aliases_devices_vram_context() {
        let cfg: crate::config::Config =
            serde_yaml_ng::from_str(TEST_CONFIG_YAML).unwrap();
        let model = cfg.models[0].clone();
        let no_aliases = HashMap::new();

        let base = fingerprint_with_aliases(&no_aliases, &model);
        // Deterministic.
        assert_eq!(base, fingerprint_with_aliases(&no_aliases, &model));

        // cmd change → different fingerprint.
        let mut m2 = model.clone();
        m2.cmd = "sleep 3601".into();
        assert_ne!(base, fingerprint_with_aliases(&no_aliases, &m2));

        // vulkan_devices change → different fingerprint (order-insensitive).
        let mut m3 = model.clone();
        m3.vulkan_devices = vec![1, 0];
        let mut m4 = model.clone();
        m4.vulkan_devices = vec![0, 1];
        assert_eq!(
            fingerprint_with_aliases(&no_aliases, &m3),
            fingerprint_with_aliases(&no_aliases, &m4)
        );
        assert_ne!(base, fingerprint_with_aliases(&no_aliases, &m3));

        // vram / context_length / gpus changes → different fingerprint.
        let mut m5 = model.clone();
        m5.vram = 1024;
        assert_ne!(base, fingerprint_with_aliases(&no_aliases, &m5));
        let mut m6 = model.clone();
        m6.context_length = 8192;
        assert_ne!(base, fingerprint_with_aliases(&no_aliases, &m6));
        let mut m6b = model.clone();
        m6b.gpus = 2;
        assert_ne!(base, fingerprint_with_aliases(&no_aliases, &m6b));

        // cmd_aliases resolving into the cmd → different fingerprint;
        // unused aliases → same fingerprint.
        let mut m7 = model.clone();
        m7.cmd = "sleep {duration}".into();
        let mut aliases = HashMap::new();
        aliases.insert("duration".to_owned(), "3600".to_owned());
        assert_ne!(
            fingerprint_with_aliases(&no_aliases, &m7),
            fingerprint_with_aliases(&aliases, &m7)
        );
        aliases.insert("unused".to_owned(), "x".to_owned());
        let mut only_unused = HashMap::new();
        only_unused.insert("unused".to_owned(), "x".to_owned());
        assert_eq!(
            fingerprint_with_aliases(&only_unused, &model),
            fingerprint_with_aliases(&aliases, &model)
        );

        // Routing-only fields must not affect the fingerprint.
        let mut m8 = model.clone();
        m8.max_concurrent = 99;
        m8.queue_depth = 99;
        m8.idle_ttl = 99;
        m8.max_instances = 99;
        assert_eq!(base, fingerprint_with_aliases(&no_aliases, &m8));
    }

    #[tokio::test]
    async fn removing_one_of_two_same_id_instances_keeps_the_other() {
        // Base IDs collide for GPU-less models (model@cpu) — removal must
        // be by handle identity, never by id string, or evicting one CPU
        // instance would evict all of them.
        let mgr = test_manager();
        let h1 = make_handle_on_gpus(vec![], InstanceState::Ready, 0, Duration::ZERO);
        let h2 = make_handle_on_gpus(vec![], InstanceState::Ready, 0, Duration::ZERO);
        assert_eq!(h1.id(), h2.id(), "test setup requires colliding base ids");
        {
            let mut instances = mgr.instances.write().unwrap();
            let list = instances.entry("m".to_owned()).or_default();
            list.push(h1.clone());
            list.push(h2.clone());
        }

        mgr.remove_instance("m", &h1).await;

        let instances = mgr.instances.read().unwrap();
        let list = &instances["m"];
        assert_eq!(list.len(), 1, "only the targeted instance may be removed");
        assert!(Arc::ptr_eq(list[0].inner(), h2.inner()));
    }

    #[tokio::test]
    async fn short_idle_ttl_is_honored_for_last_instance() {
        // Regression: the old despawn condition required load_m1 < 0.01,
        // which with τ=60 s takes ~5 min to decay regardless of the
        // configured TTL — an idle_ttl of 30 s was silently clamped.
        // The instance-level idle check honors the TTL exactly.
        let mgr = Arc::new(test_manager()); // idle_ttl = 60 in TEST_CONFIG_YAML
        let handle = make_handle(InstanceState::Ready, 0, Duration::from_secs(120));
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);
        // Fresh metrics with a recent high load would have blocked the
        // old condition; the TTL-based check must not care.
        mgr.touch_activity("m");
        {
            let mut metrics = mgr.model_metrics.write().unwrap();
            let m = metrics.entry("m".to_owned()).or_default();
            m.load_m1 = 42.0;
        }

        mgr.evaluate_autoscale().await;

        assert_eq!(
            instance_count(&mgr),
            0,
            "instance idle past its TTL must be despawned regardless of load EMAs"
        );
    }

    #[tokio::test]
    async fn surplus_idle_instances_evicted_without_autoscale() {
        // Models without autoscale config must still scale down: idle
        // surplus instances are TTL-evicted, one instance is kept.
        let mgr = Arc::new(test_manager());
        let idle = make_handle_on_gpus(vec![0], InstanceState::Ready, 0, Duration::from_secs(3600));
        let busy = make_handle_on_gpus(vec![1], InstanceState::Ready, 1, Duration::ZERO);
        {
            let mut instances = mgr.instances.write().unwrap();
            let list = instances.entry("m".to_owned()).or_default();
            list.push(idle.clone());
            list.push(busy.clone());
        }

        mgr.evaluate_autoscale().await;

        assert_eq!(instance_count(&mgr), 1, "idle surplus instance must be evicted");
        let instances = mgr.instances.read().unwrap();
        assert!(
            Arc::ptr_eq(instances["m"][0].inner(), busy.inner()),
            "the busy instance must be the survivor"
        );
    }

    #[tokio::test]
    async fn surplus_fresh_instances_kept_without_autoscale() {
        // Surplus instances that are NOT idle past TTL must be kept.
        let mgr = Arc::new(test_manager());
        let h1 = make_handle_on_gpus(vec![0], InstanceState::Ready, 0, Duration::ZERO);
        let h2 = make_handle_on_gpus(vec![1], InstanceState::Ready, 0, Duration::ZERO);
        {
            let mut instances = mgr.instances.write().unwrap();
            let list = instances.entry("m".to_owned()).or_default();
            list.push(h1);
            list.push(h2);
        }

        mgr.evaluate_autoscale().await;

        assert_eq!(instance_count(&mgr), 2, "fresh instances must be kept");
    }

    #[tokio::test]
    async fn busy_instance_is_drained_not_killed() {
        // Admin unload of a busy instance must not interrupt the in-flight
        // request: the instance is marked draining (unroutable) and only
        // removed once the request completes.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 1, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        let (removed, draining) = mgr.unload_model("m").await;
        assert_eq!((removed, draining), (0, 1));
        assert_eq!(instance_count(&mgr), 1, "busy instance must survive");
        assert_eq!(
            handle.inner().lock().unwrap().state,
            InstanceState::Failed,
            "busy instance must be marked unroutable"
        );
        // No new slots may be acquired on a draining instance.
        assert!(handle.try_acquire(4).is_none());

        // The in-flight request completes → reap finishes the removal.
        handle.inner().lock().unwrap().in_flight = 0;
        mgr.reap_drained("m").await;
        assert_eq!(instance_count(&mgr), 0, "drained instance must be reaped");
    }

    #[tokio::test]
    async fn if_idle_removal_leaves_busy_instance_untouched() {
        // Autoscale/TTL removal must be a pure no-op for busy instances —
        // no state change, no draining mark, instance stays routable.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 1, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        assert!(!mgr.remove_instance("m", &handle).await);
        assert_eq!(instance_count(&mgr), 1);
        assert_eq!(handle.inner().lock().unwrap().state, InstanceState::Ready);
    }

    // ── Multi-GPU placement ────────────────────────────────────────────

    fn gpu_metrics(index: usize, slot: &str, total_mb: u64) -> crate::gpu::GpuMetrics {
        crate::gpu::GpuMetrics {
            index,
            pci_slot: slot.to_owned(),
            vram_vendor: None,
            vram_total_bytes: total_mb * 1024 * 1024,
            vram_used_bytes: 0,
            temperature_c: None,
            power_w: None,
            gpu_busy_pct: None,
            sclk_mhz: None,
            mclk_mhz: None,
        }
    }

    const MULTI_GPU_YAML: &str = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  vulkan:
    0: "0000:01:00.0"
    1: "0000:02:00.0"
    2: "0000:03:00.0"
    3: "0000:04:00.0"
models:
  - name: big
    context_length: 4096
    cmd: "sleep 3600"
    idle_ttl: 60
    vram: 20000
    gpus: 2
    vulkan_devices: [0, 1, 2, 3]
"#;

    fn multi_gpu_manager() -> InstanceManager {
        let config: crate::config::Config =
            serde_yaml_ng::from_str(MULTI_GPU_YAML).unwrap();
        let snapshot = Arc::new(tokio::sync::RwLock::new(vec![
            gpu_metrics(0, "0000:01:00.0", 48000),
            gpu_metrics(1, "0000:02:00.0", 48000),
            gpu_metrics(2, "0000:03:00.0", 48000),
            gpu_metrics(3, "0000:04:00.0", 48000),
        ]));
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, snapshot, None);
        mgr
    }

    fn register_ready(mgr: &InstanceManager, model: &str, gpus: Vec<usize>, port: u16) {
        let mut inst = Instance::new(model, gpus, port, None);
        inst.state = InstanceState::Ready;
        mgr.instances
            .write()
            .unwrap()
            .entry(model.to_owned())
            .or_default()
            .push(InstanceHandle::new(inst));
    }

    #[tokio::test]
    async fn select_picks_two_distinct_gpus_and_complement_for_second_instance() {
        let mgr = multi_gpu_manager();
        let cfg = mgr.model_configs.read().unwrap().get("big").cloned().unwrap();

        let first = mgr.select_gpus_for_model(&cfg).await;
        assert_eq!(first.len(), 2);
        assert_ne!(first[0], first[1], "devices must be distinct");

        // The second instance must get the complement (same-model exclusion).
        register_ready(&mgr, "big", first.clone(), 54321);
        let second = mgr.select_gpus_for_model(&cfg).await;
        assert_eq!(second.len(), 2);
        assert!(
            second.iter().all(|d| !first.contains(d)),
            "second instance must avoid devices of the first: {first:?} vs {second:?}"
        );
        let mut all: Vec<usize> = first.iter().chain(second.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2, 3], "both instances tile the whole pool");

        // Pool exhausted: a third instance finds no placement.
        register_ready(&mgr, "big", second.clone(), 54322);
        assert!(mgr.select_gpus_for_model(&cfg).await.is_empty());
    }

    #[tokio::test]
    async fn select_respects_per_gpu_vram_shares() {
        // An instance occupying 2×20000 leaves 28000 free on each of its
        // GPUs; a 30000 single-GPU model must avoid those two.
        let mgr = multi_gpu_manager();
        register_ready(&mgr, "big", vec![0, 1], 54321);

        let small: ModelConfig = serde_yaml_ng::from_str(
            "name: small\ncontext_length: 4096\ncmd: \"sleep 1\"\nvram: 30000\ngpus: 1\nvulkan_devices: [0, 1, 2, 3]",
        )
        .unwrap();
        let placement = mgr.select_gpus_for_model(&small).await;
        assert_eq!(placement.len(), 1);
        assert!(
            placement[0] == 2 || placement[0] == 3,
            "must avoid GPUs with only 28000 MB free, got {placement:?}"
        );
    }

    #[tokio::test]
    async fn select_refuses_when_pool_cannot_satisfy_gpus() {
        // gpus: 2, but one device is occupied by the same model — only one
        // candidate remains, so there is no valid placement.
        let mgr = multi_gpu_manager();
        let cfg = mgr.model_configs.read().unwrap().get("big").cloned().unwrap();
        register_ready(&mgr, "big", vec![0, 1], 54321);
        register_ready(&mgr, "big", vec![2], 54322);

        let placement = mgr.select_gpus_for_model(&cfg).await;
        assert!(
            placement.is_empty(),
            "fewer candidates than gpus must yield no placement, got {placement:?}"
        );
    }

    // ── CUDA placement ────────────────────────────────────────────────

    const CUDA_GPU_YAML: &str = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  cuda:
    0:
      pci: "0000:0a:00.0"
    1:
      pci: "0000:0b:00.0"
    2:
      pci: "0000:0c:00.0"
    3:
      pci: "0000:0d:00.0"
models:
  - name: big
    context_length: 4096
    cmd: "sleep 3600"
    idle_ttl: 60
    vram: 20000
    gpus: 2
    cuda_devices: [0, 1, 2, 3]
"#;

    fn cuda_gpu_manager() -> InstanceManager {
        let config: crate::config::Config =
            serde_yaml_ng::from_str(CUDA_GPU_YAML).unwrap();
        let snapshot = Arc::new(tokio::sync::RwLock::new(vec![
            gpu_metrics(0, "0000:0a:00.0", 48000),
            gpu_metrics(1, "0000:0b:00.0", 48000),
            gpu_metrics(2, "0000:0c:00.0", 48000),
            gpu_metrics(3, "0000:0d:00.0", 48000),
        ]));
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, snapshot, None);
        mgr
    }

    #[tokio::test]
    async fn cuda_select_tiles_pool_across_instances() {
        let mgr = cuda_gpu_manager();
        let cfg = mgr.model_configs.read().unwrap().get("big").cloned().unwrap();

        let first = mgr.select_gpus_for_model(&cfg).await;
        assert_eq!(first.len(), 2);
        assert_ne!(first[0], first[1], "devices must be distinct");

        register_ready(&mgr, "big", first.clone(), 54321);
        let second = mgr.select_gpus_for_model(&cfg).await;
        assert_eq!(second.len(), 2);
        assert!(
            second.iter().all(|d| !first.contains(d)),
            "second instance must avoid devices of the first: {first:?} vs {second:?}"
        );

        register_ready(&mgr, "big", second.clone(), 54322);
        assert!(mgr.select_gpus_for_model(&cfg).await.is_empty());
    }

    #[tokio::test]
    async fn cuda_select_respects_per_gpu_vram_shares() {
        // An instance occupying 2×20000 leaves 28000 free on each of its
        // GPUs; a 30000 single-GPU CUDA model must avoid those two.
        let mgr = cuda_gpu_manager();
        register_ready(&mgr, "big", vec![0, 1], 54321);

        let small: ModelConfig = serde_yaml_ng::from_str(
            "name: small\ncontext_length: 4096\ncmd: \"sleep 1\"\nvram: 30000\ngpus: 1\ncuda_devices: [0, 1, 2, 3]",
        )
        .unwrap();
        // Register the small model's config so slot translation works.
        mgr.model_configs
            .write()
            .unwrap()
            .insert("small".to_owned(), small.clone());
        let placement = mgr.select_gpus_for_model(&small).await;
        assert_eq!(placement.len(), 1);
        assert!(
            placement[0] == 2 || placement[0] == 3,
            "must avoid GPUs with only 28000 MB free, got {placement:?}"
        );
    }

    #[tokio::test]
    async fn cuda_select_refuses_when_pool_cannot_satisfy_gpus() {
        let mgr = cuda_gpu_manager();
        let cfg = mgr.model_configs.read().unwrap().get("big").cloned().unwrap();
        register_ready(&mgr, "big", vec![0, 1], 54321);
        register_ready(&mgr, "big", vec![2], 54322);

        assert!(
            mgr.select_gpus_for_model(&cfg).await.is_empty(),
            "fewer candidates than gpus must yield no placement"
        );
    }

    #[tokio::test]
    async fn cuda_static_vram_mb_is_fallback_without_metrics() {
        // No metrics snapshot at all (nvidia-smi absent): placement must
        // use static vram_mb totals from devices.cuda.
        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  cuda:
    0:
      pci: "0000:0a:00.0"
      vram_mb: 24000
    1:
      pci: "0000:0b:00.0"
      vram_mb: 24000
models:
  - name: fits
    context_length: 4096
    cmd: "sleep 3600"
    vram: 20000
    cuda_devices: [0, 1]
  - name: toobig
    context_length: 4096
    cmd: "sleep 3600"
    vram: 30000
    cuda_devices: [0, 1]
"#;
        let config: crate::config::Config = serde_yaml_ng::from_str(yaml).unwrap();
        let snapshot = Arc::new(tokio::sync::RwLock::new(vec![]));
        let (mgr, _r, _c) = InstanceManager::new(&config, snapshot, None);

        let fits = mgr.model_configs.read().unwrap().get("fits").cloned().unwrap();
        let placement = mgr.select_gpus_for_model(&fits).await;
        assert_eq!(placement.len(), 1, "static vram_mb must enable placement without metrics");

        let toobig = mgr.model_configs.read().unwrap().get("toobig").cloned().unwrap();
        assert!(
            mgr.select_gpus_for_model(&toobig).await.is_empty(),
            "model exceeding the static total must find no placement"
        );
    }

    #[tokio::test]
    async fn cuda_static_vram_mb_caps_metrics_total() {
        // Metrics report 48 GB but vram_mb caps usable capacity at 24 GB.
        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  cuda:
    0:
      pci: "0000:0a:00.0"
      vram_mb: 24000
models:
  - name: m
    context_length: 4096
    cmd: "sleep 3600"
    vram: 30000
    cuda_devices: [0]
"#;
        let config: crate::config::Config = serde_yaml_ng::from_str(yaml).unwrap();
        let snapshot = Arc::new(tokio::sync::RwLock::new(vec![
            gpu_metrics(0, "0000:0a:00.0", 48000),
        ]));
        let (mgr, _r, _c) = InstanceManager::new(&config, snapshot, None);

        let cfg = mgr.model_configs.read().unwrap().get("m").cloned().unwrap();
        assert!(
            mgr.select_gpus_for_model(&cfg).await.is_empty(),
            "vram_mb must cap capacity below the metrics-reported total"
        );
    }

    #[tokio::test]
    async fn cuda_device_without_metrics_or_static_vram_is_unusable() {
        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  cuda:
    0:
      pci: "0000:0a:00.0"
models:
  - name: m
    context_length: 4096
    cmd: "sleep 3600"
    vram: 1000
    cuda_devices: [0]
"#;
        let config: crate::config::Config = serde_yaml_ng::from_str(yaml).unwrap();
        let snapshot = Arc::new(tokio::sync::RwLock::new(vec![]));
        let (mgr, _r, _c) = InstanceManager::new(&config, snapshot, None);

        let cfg = mgr.model_configs.read().unwrap().get("m").cloned().unwrap();
        assert!(mgr.select_gpus_for_model(&cfg).await.is_empty());
    }

    #[test]
    fn fingerprint_distinguishes_device_namespaces() {
        let base: ModelConfig = serde_yaml_ng::from_str(
            "name: m\ncontext_length: 4096\ncmd: \"sleep 1\"\nvram: 1000",
        )
        .unwrap();
        let mut vk = base.clone();
        vk.vulkan_devices = vec![0];
        let mut cu = base.clone();
        cu.cuda_devices = vec![0];
        let aliases = HashMap::new();
        assert_ne!(
            fingerprint_with_aliases(&aliases, &vk),
            fingerprint_with_aliases(&aliases, &cu),
            "switching namespaces must retire running instances"
        );
        assert_ne!(
            fingerprint_with_aliases(&aliases, &base),
            fingerprint_with_aliases(&aliases, &cu),
            "adding a cuda pool must change the fingerprint"
        );
    }

    #[test]
    fn resolve_cmd_substitutes_context_length_and_port() {
        let mgr = test_manager();
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: x\ncontext_length: 8192\ncmd: \"run --ctx-size {context_length} --port {port}\"",
        )
        .unwrap();
        assert_eq!(
            mgr.resolve_cmd(&cfg, 9999),
            "run --ctx-size 8192 --port 9999"
        );
    }

    #[test]
    fn touch_activity_resets_idle_clock() {
        let mgr = test_manager();
        let stale = Instant::now() - Duration::from_secs(3600);
        mgr.model_metrics.write().unwrap().insert(
            "m".to_owned(),
            ModelMetrics {
                last_activity: stale,
                ..Default::default()
            },
        );

        mgr.touch_activity("m");

        let last = mgr.model_metrics.read().unwrap()["m"].last_activity;
        assert!(
            last.elapsed().as_secs() < 5,
            "touch_activity must reset last_activity to now"
        );
    }

    #[tokio::test]
    async fn autoscale_reaps_instance_stranded_in_loading() {
        // Safety net: an instance stuck in `Loading` far beyond the spawn
        // window (e.g. after a panicked spawn task) must be force-removed —
        // otherwise it wedges the model by counting toward `max_instances`
        // without ever serving requests.
        let mgr = Arc::new(test_manager());
        let handle = make_handle(InstanceState::Loading, 0, Duration::from_secs(3600));
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);

        mgr.evaluate_autoscale().await;

        assert_eq!(
            instance_count(&mgr),
            0,
            "instance stranded in Loading must be reaped"
        );
    }

    #[tokio::test]
    async fn ensure_spawn_runs_detached_and_clears_entry() {
        // The spawn runs in a detached task: subscribers only await the
        // entry's removal.  `cmd: "true"` exits instantly, so the spawn
        // fails fast (ChildExited) and the task must clear the entry,
        // resolving the receiver.
        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
models:
  - name: m
    context_length: 4096
    cmd: "true"
    idle_ttl: 60
"#;
        let config: crate::config::Config = serde_yaml_ng::from_str(yaml).unwrap();
        let gpu_snapshot = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let (mgr, _release_rx, _crash_rx) =
            InstanceManager::new(&config, gpu_snapshot, None);
        let mgr = Arc::new(mgr);
        let cfg = mgr.model_configs.read().unwrap().get("m").cloned().unwrap();

        let mut rx = mgr.ensure_spawn("m", &cfg);
        assert!(
            mgr.spawns_in_flight.read().unwrap().contains_key("m"),
            "spawn entry must be registered while the spawn runs"
        );

        tokio::time::timeout(Duration::from_secs(15), rx.changed())
            .await
            .expect("spawn entry must be cleared once the spawn task finishes")
            .expect_err("the sender is dropped without ever sending");
        assert!(
            !mgr.spawns_in_flight.read().unwrap().contains_key("m"),
            "spawn entry must be gone after completion"
        );
        assert_eq!(
            instance_count(&mgr),
            0,
            "the failed instance must be unregistered"
        );
    }

    #[tokio::test]
    async fn double_unregister_is_a_no_op() {
        // The stranded-`Loading` reaper can race the spawn task's own
        // failure cleanup for the same handle — the second unregister must
        // not free the port twice or corrupt keep-alive refcounts.
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Failed, 0, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        mgr.unregister_instance("m", &handle, &[0]).await;
        assert_eq!(instance_count(&mgr), 0);
        // Second call: no panic, no state change.
        mgr.unregister_instance("m", &handle, &[0]).await;
        assert_eq!(instance_count(&mgr), 0);
    }
}
