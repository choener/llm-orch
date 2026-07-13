// ── Instance manager ─────────────────────────────────────────────────────────
//
// Central scheduler: owns the map of model → running instances, handles
// spawning, slot acquisition, queueing, idle eviction, and shutdown.

use crate::backend::{shutdown_child, spawn_process, mark_instance_ready, Backend, LlamaCppBackend};
use crate::config::ModelConfig;
use crate::http_client;
use crate::instance::{Instance, InstanceHandle, InstanceState};
use crate::port_alloc::PortAllocator;

use reqwest::Client;
use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;
use std::time::Duration;
use tokio::sync::{Mutex, oneshot};

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

    /// Crash limit before a model is blocked.
    crash_limit: usize,

    /// Spawn readiness timeout.
    spawn_timeout: Duration,
}

impl InstanceManager {
    /// Create a new manager from the loaded config.
    pub fn new(config: &crate::config::Config) -> Self {
        let model_configs: HashMap<_, _> = config
            .models
            .iter()
            .map(|m| (m.name.clone(), m.clone()))
            .collect();

        Self {
            client: http_client::build(),
            backend: LlamaCppBackend,
            ports: Mutex::new(PortAllocator::new(config.server.port_range.clone())),
            model_configs,
            cmd_aliases: config.cmd_aliases.clone(),
            instances: RwLock::new(HashMap::new()),
            queues: RwLock::new(HashMap::new()),
            blocked: RwLock::new(HashMap::new()),
            crash_limit: 3,
            spawn_timeout: Duration::from_secs(120),
        }
    }

    // ── get-or-spawn ──────────────────────────────────────────────────────

    /// Acquire an instance handle for `model_name`, spawning a new instance
    /// if necessary.  Returns `None` when the model is blocked, all instances
    /// are at capacity, the instance cap is reached and the queue is full.
    pub async fn get_or_spawn(&self, model_name: &str) -> Option<InstanceHandle> {
        // Fast path: blocked model.
        if self.is_blocked(model_name) {
            return None;
        }

        let cfg = self.model_configs.get(model_name)?;

        // Fast path: find a ready instance with spare capacity.
        if let Some(handle) = self.find_ready_instance(model_name, cfg.max_concurrent) {
            return Some(handle);
        }

        // Slow path: try to spawn a new instance.
        if let Some(handle) = self.try_spawn(model_name, cfg).await {
            return Some(handle);
        }

        // Queue path: all instances busy and at cap.
        self.enqueue(model_name, cfg.queue_depth).await
    }

    /// Enqueue the caller, waiting for an instance slot to free up.
    /// Returns `None` if the queue is at capacity (caller should return 429).
    async fn enqueue(&self, model_name: &str, max_depth: usize) -> Option<InstanceHandle> {
        let (tx, rx) = oneshot::channel();

        {
            let mut queues = self.queues.write().unwrap();
            let queue = queues.entry(model_name.to_owned()).or_default();
            if queue.len() >= max_depth {
                return None; // queue full → 429
            }
            queue.push_back(tx);
        }

        // Park until a slot frees up and we get an instance handle.
        // If the sender was dropped (shutdown), this returns an error.
        rx.await.ok()
    }

    /// Wake the first queued waiter for `model_name` if an instance is available.
    fn wake_one(&self, model_name: &str) {
        let cfg = match self.model_configs.get(model_name) {
            Some(c) => c,
            None => return,
        };

        // Find a ready instance with capacity.
        let handle = self.find_ready_instance(model_name, cfg.max_concurrent);

        if let Some(h) = handle {
            let mut queues = self.queues.write().unwrap();
            if let Some(queue) = queues.get_mut(model_name) {
                while let Some(tx) = queue.pop_front() {
                    if tx.send(h.clone()).is_ok() {
                        return;
                    }
                    // Receiver dropped — try next waiter.
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
        // Pick the least-loaded instance.
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
        // Check instance cap.
        {
            let instances = self.instances.read().unwrap();
            if let Some(list) = instances.get(model_name) {
                if list.len() >= cfg.max_instances {
                    return None;
                }
            }
        }

        // Allocate a port.
        // With `tokio::sync::Mutex`, the guard is `Send`, so it's safe to
        // hold across `.await` points.
        let port = if self.ports.lock().await.is_ephemeral() {
            self.ports.lock().await.allocate_ephemeral_async().await?
        } else {
            self.ports.lock().await.allocate_range_sync()
                .ok_or(())
                .ok()?
        };

        // Resolve the command string.
        let model_cmd = self.resolve_cmd(cfg, port);
        let parts = shlex::split(&model_cmd)?;
        if parts.is_empty() {
            self.ports.lock().await.free(port);
            return None;
        }
        let prog = &parts[0];
        let gpu_indices = vec![]; // TODO: GPU allocation
        let mut args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();
        args.extend(self.backend.gpu_args(&gpu_indices));
        let envs = self.backend.gpu_env(&gpu_indices);

        // Spawn.
        let child = spawn_process(prog, &args, &envs).await.ok()?;

        let mut inst = Instance::new(model_name, gpu_indices, port);
        inst.child = Some(child);
        let handle = InstanceHandle::new(inst);

        // Wait for readiness (with timeout).
        if !mark_instance_ready(&handle, &self.client, &self.backend, self.spawn_timeout).await {
            // Failed to become ready — shut down and return.
            // IMPORTANT: extract the child from the instance and drop the
            // `MutexGuard` *before* awaiting shutdown_child, because
            // `std::sync::MutexGuard` is `!Send`.
            let mut child_to_kill = {
                let mut inst_lock = handle.inner().lock().unwrap();
                inst_lock.state = InstanceState::Failed;
                inst_lock.child.take()
            };
            // MutexGuard dropped — safe to `.await`.
            if let Some(ref mut child) = child_to_kill {
                shutdown_child(child, Duration::from_secs(5)).await;
            }
            self.ports.lock().await.free(port);
            return None;
        }

        // Register under model name.
        {
            let mut instances = self.instances.write().unwrap();
            instances
                .entry(model_name.to_owned())
                .or_default()
                .push(handle.clone());
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

    // ── blocked flag ─────────────────────────────────────────────────────

    /// Check whether a model is blocked.
    pub fn is_blocked(&self, model_name: &str) -> bool {
        self.blocked
            .read()
            .unwrap()
            .get(model_name)
            .copied()
            .unwrap_or(false)
    }

    /// Block a model (called when crash limit exhausted).
    #[allow(dead_code)]
    pub fn block_model(&self, model_name: &str) {
        self.blocked
            .write()
            .unwrap()
            .insert(model_name.to_owned(), true);
    }

    /// Unblock a model (called by /admin/unblock or config reload).
    #[allow(dead_code)]
    pub fn unblock_model(&self, model_name: &str) {
        self.blocked.write().unwrap().remove(model_name);
    }

    // ── idle eviction ────────────────────────────────────────────────────

    /// Evict instances that have been idle beyond their model's `idle_ttl`.
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

        for (model_name, handle) in to_evict {
            self.remove_instance(&model_name, &handle).await;
        }
    }

    /// Remove a specific instance: kill its subprocess, free its port, and
    /// drop it from the instance list.
    async fn remove_instance(&self, model_name: &str, handle: &InstanceHandle) {
        // Shut down the process.
        let mut child = {
            let mut inst = handle.inner().lock().unwrap();
            inst.state = InstanceState::Failed;
            inst.child.take()
        };
        if let Some(ref mut c) = child {
            shutdown_child(c, Duration::from_secs(5)).await;
        }

        // Free the port.
        let port = handle.inner().lock().unwrap().port;
        self.ports.lock().await.free(port);

        // Remove from instance list.
        let mut instances = self.instances.write().unwrap();
        if let Some(list) = instances.get_mut(model_name) {
            list.retain(|h| h.id() != handle.id());
        }
    }

    /// Return per-model instance counts (for /v1/info).
    pub fn instance_counts(&self) -> HashMap<String, usize> {
        let instances = self.instances.read().unwrap();
        instances
            .iter()
            .map(|(model, list)| (model.clone(), list.len()))
            .collect()
    }

    /// Return the shared HTTP client (for handlers that need it).
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Unload all instances of a specific model.
    pub async fn unload_model(&self, model_name: &str) {
        let handles: Vec<InstanceHandle> = {
            let instances = self.instances.read().unwrap();
            instances
                .get(model_name)
                .map(|list| list.iter().cloned().collect())
                .unwrap_or_default()
        };

        for handle in handles {
            self.remove_instance(model_name, &handle).await;
        }
    }

    // ── shutdown ─────────────────────────────────────────────────────────

    /// Gracefully drain all instances.
    pub async fn shutdown_all(&self) {
        let all: Vec<(String, InstanceHandle)> = {
            let instances = self.instances.read().unwrap();
            instances
                .iter()
                .flat_map(|(model, list)| list.iter().map(move |h| (model.clone(), h.clone())))
                .collect()
        };

        for (model_name, handle) in all {
            self.remove_instance(&model_name, &handle).await;
        }
    }
}
