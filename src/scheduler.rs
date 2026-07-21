// ── Instance manager ─────────────────────────────────────────────────────────
//
// Central scheduler: owns the map of model → running instances, handles
// spawning, slot acquisition, queueing, idle eviction, and shutdown.

use crate::backend::{shutdown_child, spawn_process, mark_instance_ready, Backend, LlamaCppBackend};
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
use tokio::sync::{mpsc, Mutex, Semaphore, oneshot};
use tracing::{debug, error, info, warn};

/// Maximum time a request may wait in the queue for a slot before failing.
/// Without a bound, a lost wakeup (or simply sustained saturation) would
/// park the HTTP request forever.
const QUEUE_WAIT_TIMEOUT: Duration = Duration::from_secs(120);

/// A parked waiter: unique id (for timeout self-removal) plus the channel
/// used to deliver an instance handle once a slot frees up.
type WaitQueue = VecDeque<(u64, oneshot::Sender<InstanceHandle>)>;

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
    pub req_rate_m1: f64,
    pub req_rate_m5: f64,
    pub req_rate_m15: f64,
    /// Total completed requests since daemon start.
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

    /// GPU keep-alive manager (None if not configured).
    /// Rebuilt on config hot-reload when the keep-alive section changes.
    keepalive: RwLock<Option<Arc<KeepAliveManager>>>,

    /// Crash limit before a model is blocked.
    crash_limit: usize,

    /// Spawn readiness timeout.
    spawn_timeout: Duration,

    /// Per-model EMA metrics (load & request rate).
    model_metrics: RwLock<HashMap<String, ModelMetrics>>,

    /// Global semaphore to serialize all spawn attempts across models.
    /// Prevents concurrent model loads from thrashing the SSD and
    /// interleaving VRAM allocations.
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

        let (release_tx, release_rx) = mpsc::unbounded_channel();
        let (crash_tx, crash_rx) = mpsc::unbounded_channel();

        let mgr = Self {
            client: http_client::build(),
            backend: LlamaCppBackend,
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
            keepalive: RwLock::new(keepalive),
            crash_limit: 3,
            spawn_timeout: Duration::from_secs(120),
            model_metrics: RwLock::new(HashMap::new()),
            spawn_semaphore: Arc::new(Semaphore::new(1)),
            last_scale_action: RwLock::new(HashMap::new()),
            recent_completions: RwLock::new(HashMap::new()),
            release_tx,
            crash_tx,
            crash_counts: std::sync::Mutex::new(HashMap::new()),
        };
        (mgr, release_rx, crash_rx)
    }

    // ── get-or-spawn ──────────────────────────────────────────────────────

    /// Acquire an instance slot for `model_name`, spawning a new instance
    /// if necessary.  Returns `None` when the model is blocked, all instances
    /// are at capacity, the instance cap is reached and the queue is full.
    ///
    /// The returned guard already owns one in-flight slot — the caller must
    /// not call `try_acquire` again.  The slot is released automatically
    /// when the guard is dropped.  Use `guard.handle()` to reach the instance.
    pub async fn get_or_spawn(&self, model_name: &str) -> Option<SlotGuard> {
        if self.is_blocked(model_name) {
            return None;
        }

        // Clone the model config out of the lock — it may be swapped by a
        // config reload at any time, and the guard must not be held across
        // the awaits below.
        let cfg = self.model_configs.read().unwrap().get(model_name).cloned()?;
        let max_concurrent = cfg.max_concurrent;

        // Fast path: find a ready instance with spare capacity.
        if let Some(handle) = self.find_ready_instance(model_name, max_concurrent) {
            if let Some(guard) = handle.try_acquire(max_concurrent) {
                self.record_metrics_event(model_name, 0);
                return Some(guard);
            }
        }

        // Autoscale spawn gate: only spawn if sustained load exceeds
        // threshold (skip for cold-start, i.e. zero existing instances).
        let should_spawn = if let Some(ref a) = cfg.autoscale {
            if !a.enabled {
                true
            } else {
                let num_existing = {
                    self.instances.read().unwrap()
                        .get(model_name).map(|l| l.len()).unwrap_or(0)
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

        // Slow path: try to spawn a new instance.
        if should_spawn {
            if let Some(handle) = self.try_spawn(model_name, &cfg).await {
                if let Some(guard) = handle.try_acquire(max_concurrent) {
                    self.record_metrics_event(model_name, 0);
                    // Serve a parked waiter with any leftover capacity —
                    // otherwise queued requests would only be served after
                    // the next release event.
                    self.wake_one(model_name);
                    return Some(guard);
                }
            }
        }

        // Queue path: all instances busy and at cap.
        self.enqueue(model_name, cfg.queue_depth, max_concurrent).await
    }

    /// Enqueue the caller, waiting for an instance slot to free up.
    /// Returns `None` if the queue is at capacity (caller should return 429),
    /// if no instance exists (or is loading) — in that case no release or
    /// spawn event could ever wake the parked waiter, so fail fast instead
    /// of hanging forever — or if the wait exceeds `QUEUE_WAIT_TIMEOUT`.
    /// Acquires the slot on the received handle before returning.
    async fn enqueue(
        &self,
        model_name: &str,
        max_depth: usize,
        max_concurrent: usize,
    ) -> Option<SlotGuard> {
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
            let has_instances = instances
                .get(model_name)
                .map(|l| !l.is_empty())
                .unwrap_or(false);
            if !has_instances {
                return None;
            }

            let mut queues = self.queues.write().unwrap();
            let queue = queues.entry(model_name.to_owned()).or_default();
            if queue.len() >= max_depth {
                return None;
            }
            queue.push_back((id, tx));
        }

        // Close the lost-wakeup race: a slot may have been released between
        // the capacity check in get_or_spawn and our queue push, with its
        // wake_one finding an empty queue.  Re-signal now that we are
        // parked — if capacity exists, the head waiter gets served.
        self.wake_one(model_name);

        let handle = match tokio::time::timeout(QUEUE_WAIT_TIMEOUT, rx).await {
            Ok(Ok(handle)) => handle,
            // Queue drained (last instance removed) — fail fast.
            Ok(Err(_)) => return None,
            Err(_) => {
                // Timed out — remove our entry if it is still parked.
                // (If wake_one already popped us, the handle it sent is
                // dropped with the receiver; handle drop is a no-op.)
                self.remove_queued(model_name, id);
                return None;
            }
        };

        if let Some(guard) = handle.try_acquire(max_concurrent) {
            self.record_metrics_event(model_name, 0);
            Some(guard)
        } else {
            None
        }
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

    /// Wake the first queued waiter for `model_name` if an instance is available.
    pub(crate) fn wake_one(&self, model_name: &str) {
        let cfg = match self.model_configs.read().unwrap().get(model_name).cloned() {
            Some(c) => c,
            None => return,
        };

        let handle = self.find_ready_instance(model_name, cfg.max_concurrent);

        if let Some(h) = handle {
            let mut queues = self.queues.write().unwrap();
            if let Some(queue) = queues.get_mut(model_name) {
                while let Some((_id, tx)) = queue.pop_front() {
                    if tx.send(h.clone()).is_ok() {
                        return;
                    }
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
        // Serialize all spawn attempts globally — prevents concurrent
        // model loads from thrashing the SSD and interleaving VRAM.
        let _permit = self.spawn_semaphore.clone().acquire_owned().await.ok()?;

        // Inside the semaphore: check instance cap.
        {
            let instances = self.instances.read().unwrap();
            if let Some(list) = instances.get(model_name) {
                if list.len() >= cfg.max_instances {
                    return None;
                }
            }
        }

        // Allocate a port.
        let port = if self.ports.lock().await.is_ephemeral() {
            let p = self.ports.lock().await.allocate_ephemeral_async().await;
            if p.is_some() {
                debug!(model = %model_name, port = p.unwrap(), "allocated ephemeral port");
            } else {
                warn!(model = %model_name, "no ephemeral port available");
            }
            p?
        } else {
            let p = self.ports.lock().await.allocate_range_sync();
            if p.is_none() {
                warn!(model = %model_name, "no free ports in range");
            }
            p?
        };

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
        let gpu_indices: Vec<usize> = self
            .select_gpu_for_model(cfg)
            .await
            .into_iter()
            .collect();

        // Safety: when vulkan_devices is configured but no suitable GPU
        // was found, fail the spawn instead of launching without GPU
        // restriction (which would make the new instance compete on GPUs
        // already occupied by existing instances).
        if !cfg.vulkan_devices.is_empty() && gpu_indices.is_empty() {
            warn!(
                model = %model_name,
                vulkan_devices = ?cfg.vulkan_devices,
                "no suitable GPU available — refusing to spawn without GPU restriction"
            );
            self.ports.lock().await.free(port);
            return None;
        }

        let mut args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();
        args.extend(self.backend.gpu_args(&gpu_indices));
        let envs = self.backend.gpu_env(&gpu_indices);

        // Spawn.
        let child = match spawn_process(prog, &args, &envs).await {
            Ok(c) => c,
            Err(e) => {
                warn!(model = %model_name, error = %e, "spawn failed");
                self.ports.lock().await.free(port);
                return None;
            }
        };

        let mut inst = Instance::new(
            model_name,
            gpu_indices,
            port,
            Some(self.release_tx.clone()),
        );
        // Guarantee a unique ID even when the base `model@gpus` collides
        // (multiple CPU instances of the same model).
        inst.id = format!(
            "{}#{}",
            inst.id,
            self.instance_seq.fetch_add(1, Ordering::Relaxed)
        );
        inst.child = Some(child);
        let handle = InstanceHandle::new(inst);

        // Register under model name immediately as Loading.
        {
            let mut instances = self.instances.write().unwrap();
            instances
                .entry(model_name.to_owned())
                .or_default()
                .push(handle.clone());
        }

        // Spawn intent counts as activity: reset the idle-TTL clock so the
        // autoscaler can't despawn this instance — using stale metrics left
        // over from a previous despawn — while it is still loading.
        self.touch_activity(model_name);

        // Wait for readiness (with timeout).
        if !mark_instance_ready(&handle, &self.client, &self.backend, self.spawn_timeout).await {
            // Distinguish a dead process (pre-output crash — counts toward
            // the model block limit) from a live-but-slow load (no count).
            let child_exited = {
                let mut inst_lock = handle.inner().lock().unwrap();
                match inst_lock.child.as_mut() {
                    Some(c) => !matches!(c.try_wait(), Ok(None)),
                    None => false,
                }
            };
            warn!(model = %model_name, port = port, exited = child_exited, "health check timeout — shutting down instance");
            let mut child_to_kill = {
                let mut inst_lock = handle.inner().lock().unwrap();
                inst_lock.state = InstanceState::Failed;
                inst_lock.child.take()
            };
            if let Some(ref mut child) = child_to_kill {
                shutdown_child(child, Duration::from_secs(5)).await;
            }
            self.ports.lock().await.free(port);

            {
                let mut instances = self.instances.write().unwrap();
                if let Some(list) = instances.get_mut(model_name) {
                    // Compare handle identity, not the id string — base IDs
                    // collide for same-GPU/CPU instances of one model.
                    list.retain(|h| !Arc::ptr_eq(h.inner(), handle.inner()));
                }
            }
            self.drain_queue_if_no_instances(model_name);

            if child_exited {
                self.note_pre_output_crash(model_name);
            }

            return None;
        }

        // Mark ready.
        {
            let mut inst_lock = handle.inner().lock().unwrap();
            inst_lock.state = InstanceState::Ready;
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

        // Start keep-alive for the GPU(s) this instance occupies.
        let keepalive = self.keepalive.read().unwrap().clone();
        if let Some(ref ka) = keepalive {
            let gpus = {
                let inst = handle.inner().lock().unwrap();
                inst.gpu_indices.clone()
            };
            let vulkan_slots = self.vulkan_slots.read().unwrap();
            for vulkan_idx in &gpus {
                if let Some(slot) = vulkan_slots.get(vulkan_idx) {
                    ka.ensure_running(slot);
                }
            }
        }

        Some(handle)
    }

    /// Resolve `cmd_aliases` and `{port}` in the model's command string.
    fn resolve_cmd(&self, cfg: &ModelConfig, port: u16) -> String {
        let mut resolved = cfg.cmd.clone();
        let cmd_aliases = self.cmd_aliases.read().unwrap();
        for (key, value) in cmd_aliases.iter() {
            let placeholder = format!("{{{}}}", key);
            resolved = resolved.replace(&placeholder, value);
        }
        resolved.replace("{port}", &port.to_string())
    }

    /// Pick a Vulkan device for a new instance from the model's `vulkan_devices` pool.
    async fn select_gpu_for_model(&self, model_cfg: &ModelConfig) -> Option<usize> {
        let vulkan_devices = &model_cfg.vulkan_devices;
        // Clone the (tiny) device maps — std RwLock guards are !Send and
        // must not live across the gpu_snapshot await below.
        let vulkan_slots = self.vulkan_slots.read().unwrap().clone();
        if vulkan_devices.is_empty() || vulkan_slots.is_empty() {
            debug!(model = %model_cfg.name, "no vulkan_devices configured");
            return None;
        }
        let vram_limits = self.vram_limits.read().unwrap().clone();

        let gpus = self.gpu_snapshot.read().await;
        debug!(
            model = %model_cfg.name,
            vram_mb = model_cfg.vram,
            vulkan_pool = ?vulkan_devices,
            gpu_count = gpus.len(),
            "selecting GPU"
        );

        let occupied: std::collections::HashSet<usize> = {
            let instances = self.instances.read().unwrap();
            if let Some(list) = instances.get(&model_cfg.name) {
                list.iter()
                    .filter_map(|h| {
                        let inst = h.inner().lock().unwrap();
                        inst.gpu_indices.first().copied()
                    })
                    .collect()
            } else {
                std::collections::HashSet::new()
            }
        };

        let vram_used: HashMap<usize, u64> = {
            let instances = self.instances.read().unwrap();
            let mut used = HashMap::new();
            for (model_name, list) in instances.iter() {
                let model_vram = self.model_configs.read().unwrap().get(model_name)
                    .map(|c| c.vram * 1024 * 1024)
                    .unwrap_or(0);
                for handle in list {
                    let inst = handle.inner().lock().unwrap();
                    if let Some(&vulkan_idx) = inst.gpu_indices.first() {
                        *used.entry(vulkan_idx).or_default() += model_vram;
                    }
                }
            }
            used
        };

        let model_vram_bytes = model_cfg.vram * 1024 * 1024;
        let mut candidates: Vec<(usize, u64)> = Vec::new();
        for &vulkan_idx in vulkan_devices {
            if occupied.contains(&vulkan_idx) {
                debug!(model = %model_cfg.name, vulkan = vulkan_idx, "skipping — already has instance");
                continue;
            }

            let pci_slot = match vulkan_slots.get(&vulkan_idx) {
                Some(s) => s.as_str(),
                None => {
                    debug!(model = %model_cfg.name, vulkan = vulkan_idx, "slot not in device map");
                    continue;
                }
            };
            let gpu = match gpus.iter().find(|g| g.pci_slot == pci_slot) {
                Some(g) => g,
                None => {
                    debug!(model = %model_cfg.name, vulkan = vulkan_idx, slot = pci_slot, "GPU not in metrics snapshot");
                    continue;
                }
            };

            let used = vram_used.get(&vulkan_idx).copied().unwrap_or(0);
            // Cap effective VRAM at the configured limit (if any) or the
            // sysfs-reported total, whichever is smaller.
            let capacity = vram_limits.get(&vulkan_idx)
                .copied()
                .map(|limit| limit.min(gpu.vram_total_bytes))
                .unwrap_or(gpu.vram_total_bytes);
            let free = capacity.saturating_sub(used);
            debug!(
                model = %model_cfg.name, vulkan = vulkan_idx, slot = pci_slot,
                vram_total_mb = gpu.vram_total_bytes / (1024 * 1024),
                vram_limit_mb = capacity / (1024 * 1024),
                vram_used_mb = used / (1024 * 1024),
                vram_free_mb = free / (1024 * 1024),
                model_mb = model_cfg.vram,
            );
            if free < model_vram_bytes {
                debug!(model = %model_cfg.name, vulkan = vulkan_idx, "insufficient free VRAM");
                continue;
            }
            candidates.push((vulkan_idx, free));
        }

        if candidates.is_empty() {
            debug!(model = %model_cfg.name, "no GPU candidate — falling back to CPU");
            return None;
        }

        let instance_counts: HashMap<usize, usize> = {
            let instances = self.instances.read().unwrap();
            let mut counts = HashMap::new();
            for list in instances.values() {
                for handle in list {
                    let inst = handle.inner().lock().unwrap();
                    if let Some(&vulkan_idx) = inst.gpu_indices.first() {
                        *counts.entry(vulkan_idx).or_default() += 1;
                    }
                }
            }
            counts
        };

        let chosen = candidates
            .into_iter()
            .min_by_key(|(idx, _)| instance_counts.get(idx).copied().unwrap_or(0));

        if let Some((idx, _)) = &chosen {
            debug!(model = %model_cfg.name, vulkan = idx, "selected GPU");
        }
        chosen.map(|(idx, _)| idx)
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

    // ── idle eviction ────────────────────────────────────────────────────

    pub async fn unload_idle(&self) {
        let to_evict: Vec<(String, InstanceHandle)> = {
            let instances = self.instances.read().unwrap();
            let mut candidates = Vec::new();
            for (model, list) in instances.iter() {
                let ttl = self
                    .model_configs
                    .read()
                    .unwrap()
                    .get(model.as_str())
                    .map(|c| c.idle_ttl)
                    .unwrap_or(300);
                for handle in list {
                    if handle.inner().lock().unwrap().is_idle_expired(ttl) {
                        candidates.push((model.clone(), handle.clone()));
                    }
                }
            }
            candidates
        };

        let mut evicted_by_model: HashMap<String, usize> = HashMap::new();
        for (model_name, handle) in to_evict {
            *evicted_by_model.entry(model_name.clone()).or_default() += 1;
            self.remove_instance(&model_name, &handle).await;
        }
        for (model, count) in &evicted_by_model {
            info!(model = %model, count = *count, "unloaded via TTL idle eviction");
        }
    }

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
    async fn unregister_instance(
        &self,
        model_name: &str,
        handle: &InstanceHandle,
        gpu_indices: &[usize],
    ) {
        let port = handle.inner().lock().unwrap().port;
        self.ports.lock().await.free(port);

        {
            let mut instances = self.instances.write().unwrap();
            if let Some(list) = instances.get_mut(model_name) {
                // Compare handle identity, not the id string — base IDs
                // collide for same-GPU/CPU instances of one model.
                list.retain(|h| !Arc::ptr_eq(h.inner(), handle.inner()));
            }
        }
        self.drain_queue_if_no_instances(model_name);

        let keepalive = self.keepalive.read().unwrap().clone();
        if let Some(ref ka) = keepalive {
            for vulkan_idx in gpu_indices {
                let still_in_use = {
                    let instances = self.instances.read().unwrap();
                    instances.values().flatten().any(|h| {
                        let inst = h.inner().lock().unwrap();
                        inst.gpu_indices.contains(vulkan_idx)
                    })
                };
                if !still_in_use {
                    let slot = self.vulkan_slots.read().unwrap().get(vulkan_idx).cloned();
                    if let Some(slot) = slot {
                        ka.stop(&slot);
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

        self.unregister_instance(&model_name, &handle, &gpu_indices).await;

        if !was_ready {
            self.note_pre_output_crash(&model_name);
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
    /// problem (§5).  Running instances of surviving models are kept: a
    /// running process cannot change its command line, so `cmd` changes
    /// only affect future spawns.
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
        self.ports
            .lock()
            .await
            .set_range(config.server.port_range.clone());

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
            // Restart keep-alive for GPUs with running instances — the
            // fresh manager starts with no tasks.
            let keepalive = self.keepalive.read().unwrap().clone();
            if let Some(ref ka) = keepalive {
                let in_use_slots: std::collections::HashSet<String> = {
                    let instances = self.instances.read().unwrap();
                    let vulkan_slots = self.vulkan_slots.read().unwrap();
                    instances
                        .values()
                        .flatten()
                        .flat_map(|h| h.inner().lock().unwrap().gpu_indices.clone())
                        .filter_map(|idx| vulkan_slots.get(&idx).cloned())
                        .collect()
                };
                for slot in in_use_slots {
                    ka.ensure_running(&slot);
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
    /// `completions_delta` is 0 for acquires, 1 for releases.
    ///
    /// Called from `get_or_spawn` / `enqueue` (acquire path) and from the
    /// background release-processing task (release path).  In both cases
    /// the caller holds **no locks**, so we can safely acquire
    /// `instances.read()` → `model_metrics.write()` without deadlock.
    pub(crate) fn record_metrics_event(&self, model_name: &str, completions_delta: u64) {
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
    pub async fn evaluate_autoscale(&self) {
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
            // Despawns when: only one instance exists, load_m1 has decayed
            // below 0.01 (no activity for several minutes), and wall-clock
            // time since last activity exceeds idle_ttl.
            if num_instances == 1
                && m.load_m1 < 0.01
                && now.duration_since(m.last_activity).as_secs() > cfg.idle_ttl
            {
                let handle = self.pick_least_loaded(model_name);
                if let Some(h) = handle {
                    // Re-check the victim instance itself before killing it.
                    // is_idle_expired requires Ready + no in-flight requests
                    // + instance-level idle past TTL, which excludes:
                    //   - Loading instances (spawn in progress — their
                    //     last_active is fresh and state != Ready),
                    //   - instances that acquired a request after the
                    //     metrics snapshot above was taken.
                    let expired = h.inner().lock().unwrap().is_idle_expired(cfg.idle_ttl);
                    if !expired {
                        continue;
                    }
                    let idle_secs = now.duration_since(m.last_activity).as_secs();
                    info!(
                        model = %model_name,
                        idle_secs = %idle_secs,
                        ttl = cfg.idle_ttl,
                        "idle TTL expired, despawning last instance"
                    );
                    self.remove_instance(model_name, &h).await;
                }
                continue;
            }

            // ── Autoscale (n ↔ n±1): requires autoscale.enabled ────────
            let a = match &cfg.autoscale {
                Some(a) if a.enabled => a,
                _ => continue,
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
                    info!(
                        model = %model_name,
                        load_m5 = %m.load_m5,
                        threshold = %(a.scale_up_at * capacity),
                        "autoscale: scaling up"
                    );
                    if self.try_spawn(model_name, cfg).await.is_some() {
                        self.last_scale_action.write().unwrap()
                            .insert(model_name.clone(), now);
                        // A proactive scale-up may serve parked waiters.
                        self.wake_one(model_name);
                        continue;
                    }
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
        let mgr = test_manager();
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
        let mgr = test_manager();
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
        let mgr = test_manager();
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
        let result = mgr.enqueue("m", 8, 4).await;
        assert!(result.is_none(), "enqueue with no instances must fail fast");
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

        let (tx, rx) = oneshot::channel::<InstanceHandle>();
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

        let guard = mgr.enqueue("m", 8, 1).await;
        assert!(guard.is_some(), "waiter must be served by the post-push wake");
        assert_eq!(handle.inner().lock().unwrap().in_flight, 1);
    }

    #[tokio::test]
    async fn remove_queued_drops_only_the_matching_entry() {
        // Timeout path: a timed-out waiter removes its own entry by id;
        // other parked waiters must stay in place.
        let mgr = test_manager();
        let (tx1, rx1) = oneshot::channel::<InstanceHandle>();
        let (tx2, rx2) = oneshot::channel::<InstanceHandle>();
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
        // Models still present in the reloaded config keep their running
        // instances (a running process cannot change its command line).
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);

        let cfg: crate::config::Config =
            serde_yaml_ng::from_str(TEST_CONFIG_YAML).unwrap();
        mgr.reconcile_config(&cfg).await;

        assert_eq!(instance_count(&mgr), 1, "surviving model's instance must be kept");
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
}
