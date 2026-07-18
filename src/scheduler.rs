// ── Instance manager ─────────────────────────────────────────────────────────
//
// Central scheduler: owns the map of model → running instances, handles
// spawning, slot acquisition, queueing, idle eviction, and shutdown.

use crate::backend::{shutdown_child, spawn_process, mark_instance_ready, Backend, LlamaCppBackend};
use crate::config::ModelConfig;
use crate::gpu::GpuMetrics;
use crate::types::CompletionRecord;
use crate::http_client;
use crate::instance::{Instance, InstanceHandle, InstanceState};
use crate::keepalive::KeepAliveManager;
use crate::port_alloc::PortAllocator;

use reqwest::Client;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, Semaphore, oneshot};
use tracing::{debug, info, warn};

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
    model_configs: HashMap<String, ModelConfig>,

    /// Named `{key}` → value fragments from `cmd_aliases`.
    cmd_aliases: HashMap<String, String>,

    /// Running instances, keyed by model name.
    instances: RwLock<HashMap<String, Vec<InstanceHandle>>>,

    /// Per-model wait queues.  When all instances are at capacity, requests
    /// park on a oneshot channel until a slot frees up or the queue is full.
    queues: RwLock<HashMap<String, VecDeque<oneshot::Sender<InstanceHandle>>>>,

    /// Per-model blocked flag.  A blocked model refuses all requests.
    blocked: RwLock<HashMap<String, bool>>,

    /// Latest GPU metrics snapshot for VRAM-aware scheduling.
    gpu_snapshot: Arc<tokio::sync::RwLock<Vec<GpuMetrics>>>,

    /// Vulkan device index → PCI slot mapping (from config).
    vulkan_slots: HashMap<usize, String>,

    /// Vulkan device index → VRAM limit in bytes (from config, optional).
    vram_limits: HashMap<usize, u64>,

    /// GPU keep-alive manager (None if not configured).
    keepalive: Option<Arc<KeepAliveManager>>,

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
}

impl InstanceManager {
    /// Create a new manager and the receiver for the release channel.
    ///
    /// The caller must spawn a background task that drains `release_rx`
    /// and calls `record_metrics_event` + `wake_one` on the manager for
    /// each model name received.  This task runs with no locks held, so
    /// it can safely acquire `instances.read()` → `metrics.write()`
    /// without deadlocking with the request-completion Drop path.
    pub fn new(
        config: &crate::config::Config,
        gpu_snapshot: Arc<tokio::sync::RwLock<Vec<GpuMetrics>>>,
        keepalive: Option<Arc<KeepAliveManager>>,
    ) -> (Self, mpsc::UnboundedReceiver<String>) {
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

        let mgr = Self {
            client: http_client::build(),
            backend: LlamaCppBackend,
            ports: Mutex::new(PortAllocator::new(config.server.port_range.clone())),
            model_configs,
            cmd_aliases: config.cmd_aliases.clone(),
            instances: RwLock::new(HashMap::new()),
            queues: RwLock::new(HashMap::new()),
            blocked: RwLock::new(HashMap::new()),
            gpu_snapshot,
            vulkan_slots,
            vram_limits,
            keepalive,
            crash_limit: 3,
            spawn_timeout: Duration::from_secs(120),
            model_metrics: RwLock::new(HashMap::new()),
            spawn_semaphore: Arc::new(Semaphore::new(1)),
            last_scale_action: RwLock::new(HashMap::new()),
            recent_completions: RwLock::new(HashMap::new()),
            release_tx,
        };
        (mgr, release_rx)
    }

    // ── get-or-spawn ──────────────────────────────────────────────────────

    /// Acquire an instance handle for `model_name`, spawning a new instance
    /// if necessary.  Returns `None` when the model is blocked, all instances
    /// are at capacity, the instance cap is reached and the queue is full.
    ///
    /// The returned handle already has a slot acquired — the caller must
    /// not call `try_acquire` again.  The slot is released automatically
    /// when the handle is dropped.
    pub async fn get_or_spawn(&self, model_name: &str) -> Option<InstanceHandle> {
        if self.is_blocked(model_name) {
            return None;
        }

        let cfg = self.model_configs.get(model_name)?;
        let max_concurrent = cfg.max_concurrent;

        // Fast path: find a ready instance with spare capacity.
        if let Some(handle) = self.find_ready_instance(model_name, max_concurrent) {
            if handle.try_acquire(max_concurrent) {
                self.record_metrics_event(model_name, 0);
                return Some(handle);
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
            if let Some(handle) = self.try_spawn(model_name, cfg).await {
                if handle.try_acquire(max_concurrent) {
                    self.record_metrics_event(model_name, 0);
                    return Some(handle);
                }
            }
        }

        // Queue path: all instances busy and at cap.
        self.enqueue(model_name, cfg.queue_depth, max_concurrent).await
    }

    /// Enqueue the caller, waiting for an instance slot to free up.
    /// Returns `None` if the queue is at capacity (caller should return 429).
    /// Acquires the slot on the received handle before returning.
    async fn enqueue(
        &self,
        model_name: &str,
        max_depth: usize,
        max_concurrent: usize,
    ) -> Option<InstanceHandle> {
        let (tx, rx) = oneshot::channel();

        {
            let mut queues = self.queues.write().unwrap();
            let queue = queues.entry(model_name.to_owned()).or_default();
            if queue.len() >= max_depth {
                return None;
            }
            queue.push_back(tx);
        }

        let handle = rx.await.ok()?;

        if handle.try_acquire(max_concurrent) {
            self.record_metrics_event(model_name, 0);
            Some(handle)
        } else {
            None
        }
    }

    /// Wake the first queued waiter for `model_name` if an instance is available.
    pub(crate) fn wake_one(&self, model_name: &str) {
        let cfg = match self.model_configs.get(model_name) {
            Some(c) => c,
            None => return,
        };

        let handle = self.find_ready_instance(model_name, cfg.max_concurrent);

        if let Some(h) = handle {
            let mut queues = self.queues.write().unwrap();
            if let Some(queue) = queues.get_mut(model_name) {
                while let Some(tx) = queue.pop_front() {
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

        // Wait for readiness (with timeout).
        if !mark_instance_ready(&handle, &self.client, &self.backend, self.spawn_timeout).await {
            warn!(model = %model_name, port = port, "health check timeout — shutting down instance");
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
                    list.retain(|h| h.id() != handle.id());
                }
            }

            return None;
        }

        // Mark ready.
        {
            let mut inst_lock = handle.inner().lock().unwrap();
            inst_lock.state = InstanceState::Ready;
        }

        let instance_id = {
            let inst = handle.inner().lock().unwrap();
            inst.id.clone()
        };
        info!(model = %model_name, inst = %instance_id, port = port, "spawn succeeded");

        // Start keep-alive for the GPU(s) this instance occupies.
        if let Some(ref ka) = self.keepalive {
            let gpus = {
                let inst = handle.inner().lock().unwrap();
                inst.gpu_indices.clone()
            };
            for vulkan_idx in &gpus {
                if let Some(slot) = self.vulkan_slots.get(vulkan_idx) {
                    ka.ensure_running(slot);
                }
            }
        }

        Some(handle)
    }

    /// Resolve `cmd_aliases` and `{port}` in the model's command string.
    fn resolve_cmd(&self, cfg: &ModelConfig, port: u16) -> String {
        let mut resolved = cfg.cmd.clone();
        for (key, value) in &self.cmd_aliases {
            let placeholder = format!("{{{}}}", key);
            resolved = resolved.replace(&placeholder, value);
        }
        resolved.replace("{port}", &port.to_string())
    }

    /// Pick a Vulkan device for a new instance from the model's `vulkan_devices` pool.
    async fn select_gpu_for_model(&self, model_cfg: &ModelConfig) -> Option<usize> {
        let vulkan_devices = &model_cfg.vulkan_devices;
        if vulkan_devices.is_empty() || self.vulkan_slots.is_empty() {
            debug!(model = %model_cfg.name, "no vulkan_devices configured");
            return None;
        }

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
                let model_vram = self.model_configs.get(model_name)
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

            let pci_slot = match self.vulkan_slots.get(&vulkan_idx) {
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
            let capacity = self.vram_limits.get(&vulkan_idx)
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

    #[allow(dead_code)]
    pub fn block_model(&self, model_name: &str) {
        self.blocked
            .write()
            .unwrap()
            .insert(model_name.to_owned(), true);
    }

    #[allow(dead_code)]
    pub fn unblock_model(&self, model_name: &str) {
        self.blocked.write().unwrap().remove(model_name);
    }

    // ── idle eviction ────────────────────────────────────────────────────

    pub async fn unload_idle(&self) {
        let to_evict: Vec<(String, InstanceHandle)> = {
            let instances = self.instances.read().unwrap();
            let mut candidates = Vec::new();
            for (model, list) in instances.iter() {
                let ttl = self
                    .model_configs
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

    async fn remove_instance(&self, model_name: &str, handle: &InstanceHandle) {
        let gpu_indices: Vec<usize> = {
            let inst = handle.inner().lock().unwrap();
            inst.gpu_indices.clone()
        };

        let mut child = {
            let mut inst = handle.inner().lock().unwrap();
            inst.state = InstanceState::Failed;
            inst.child.take()
        };
        if let Some(ref mut c) = child {
            shutdown_child(c, Duration::from_secs(5)).await;
        }

        let port = handle.inner().lock().unwrap().port;
        self.ports.lock().await.free(port);

        {
            let mut instances = self.instances.write().unwrap();
            if let Some(list) = instances.get_mut(model_name) {
                list.retain(|h| h.id() != handle.id());
            }
        }

        if let Some(ref ka) = self.keepalive {
            for vulkan_idx in &gpu_indices {
                let still_in_use = {
                    let instances = self.instances.read().unwrap();
                    instances.values().flatten().any(|h| {
                        let inst = h.inner().lock().unwrap();
                        inst.gpu_indices.contains(vulkan_idx)
                    })
                };
                if !still_in_use {
                    if let Some(slot) = self.vulkan_slots.get(vulkan_idx) {
                        ka.stop(slot);
                    }
                }
            }
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

    pub async fn unload_model(&self, model_name: &str) {
        let handles: Vec<InstanceHandle> = {
            let instances = self.instances.read().unwrap();
            instances
                .get(model_name)
                .map(|list| list.iter().cloned().collect())
                .unwrap_or_default()
        };

        let count = handles.len();
        for handle in &handles {
            self.remove_instance(model_name, handle).await;
        }
        if count > 0 {
            info!(model = %model_name, count = count, "unloaded via admin");
        }
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
            self.remove_instance(&model_name, &handle).await;
        }

        if let Some(ref ka) = self.keepalive {
            ka.stop_all();
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
        let now = Instant::now();
        let metrics = self.model_metrics_snapshot();

        // Snapshot configs for the iteration (cheap clone, small structs).
        let configs: Vec<(String, ModelConfig)> = self
            .model_configs
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        for (model_name, cfg) in &configs {
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
                        continue;
                    }
                }
            }

            // ── Scale-down (n → n−1) ─────────────────────────────
            if num_instances > 1 {
                let reduced_cap = (cfg.max_concurrent * (num_instances - 1)) as f64;
                if m.load_m15 < a.scale_down_at * reduced_cap {
                    let victim = self.pick_least_loaded(model_name);
                    if let Some(handle) = victim {
                        info!(
                            model = %model_name,
                            load_m15 = %m.load_m15,
                            threshold = %(a.scale_down_at * reduced_cap),
                            "autoscale: scaling down"
                        );
                        self.remove_instance(model_name, &handle).await;
                        self.last_scale_action.write().unwrap()
                            .insert(model_name.clone(), now);
                        continue;
                    }
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
}
