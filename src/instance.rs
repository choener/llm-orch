use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::process::Child;
use tokio::sync::mpsc;

// ── Instance state ───────────────────────────────────────────────────────────

/// Lifecycle state of a backend instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceState {
    /// Subprocess spawned, waiting for `/health` to return 200.
    Loading,
    /// Health check passed, accepting requests.
    Ready,
    /// Health check or spawn failed, or crashed without producing output.
    Failed,
}

// ── Instance ─────────────────────────────────────────────────────────────────

/// A running backend process managed by the scheduler.
///
/// Wrapped in `Arc<Mutex<…>>` so an `InstanceHandle` can share ownership.
/// In-flight slot lifecycle is tied to `SlotGuard`, not to handle clones.
pub struct Instance {
    /// Base ID, e.g. `"qwen3-32b@0,1"` (`"model@gpus"`).
    ///
    /// Not unique on its own — GPU-less models all get `"model@cpu"`.
    /// The manager appends a `#seq` suffix at spawn time to make it
    /// unique; removal paths compare handle identity, never the id string.
    pub id: String,

    /// The model this instance serves.
    pub model_name: String,

    /// Allocated port the backend listens on.
    pub port: u16,

    /// GPU device indices this instance occupies.
    pub gpu_indices: Vec<usize>,

    /// Current lifecycle state.
    pub state: InstanceState,

    /// Subprocess handle (`None` before spawn, or after the process exits).
    pub child: Option<Child>,

    /// Ring buffer of the backend's recent stdout/stderr lines.  Set by
    /// the manager at spawn time; dumped at `warn!` level when the
    /// instance fails readiness or crashes.
    pub output: Option<crate::backend::OutputBuffer>,

    /// Number of in-flight requests currently being processed.
    pub in_flight: usize,

    /// Timestamp of the last request completion (for TTL tracking).
    pub last_active: Instant,

    /// Channel sender for notifying the manager on slot release.
    /// `None` in tests or after shutdown.
    pub release_tx: Option<mpsc::UnboundedSender<String>>,

    /// Fingerprint of the spawn-relevant config (resolved `cmd`, selected
    /// device pool, `vram`, `context_length`) this instance was launched
    /// with.  Set by the manager at spawn time; `0` means "unset" (tests).
    /// On config reload, instances whose fingerprint no longer matches are
    /// retired (marked `Failed`/draining) and replaced on demand.
    pub config_fingerprint: u64,
}

impl Instance {
    /// Create a new instance descriptor.  The subprocess has not been spawned yet.
    pub fn new(
        model_name: &str,
        gpu_indices: Vec<usize>,
        port: u16,
        release_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Self {
        let id = format!("{}@{}", model_name, display_gpus(&gpu_indices));
        Self {
            id,
            model_name: model_name.to_owned(),
            port,
            gpu_indices,
            state: InstanceState::Loading,
            child: None,
            output: None,
            in_flight: 0,
            last_active: Instant::now(),
            release_tx,
            config_fingerprint: 0,
        }
    }

    /// Mark this instance as ready (health check passed).
    pub fn mark_ready(&mut self) {
        self.state = InstanceState::Ready;
    }

    /// Increment the in-flight counter and bump the activity timestamp.
    pub fn acquire_slot(&mut self) {
        self.in_flight += 1;
        self.last_active = Instant::now();
    }

    /// Decrement the in-flight counter.  Called by `SlotGuard::drop()`.
    ///
    /// Sends a release notification on the metrics channel only when this
    /// release actually freed a slot (`in_flight` was > 0 before decrement).
    /// The background task that receives this message runs with no locks
    /// held, so it can safely acquire `instances.read()` → `metrics.write()`
    /// without deadlocking with the request-completion Drop path.
    pub fn release_slot(&mut self) {
        let was_active = self.in_flight > 0;
        self.in_flight = self.in_flight.saturating_sub(1);
        self.last_active = Instant::now();
        if was_active {
            if let Some(ref tx) = self.release_tx {
                let _ = tx.send(self.model_name.clone());
            }
        }
    }

    /// Whether this instance has spare capacity for another request.
    pub fn has_capacity(&self, max_concurrent: usize) -> bool {
        self.state == InstanceState::Ready && self.in_flight < max_concurrent
    }

    /// Whether the instance has been idle longer than `ttl_seconds`.
    pub fn is_idle_expired(&self, ttl_seconds: u64) -> bool {
        self.in_flight == 0
            && self.state == InstanceState::Ready
            && self.last_active.elapsed().as_secs() >= ttl_seconds
    }
}

// ── Instance handle (RAII slot tracking) ─────────────────────────────────────

/// A cloneable, reference-counted handle to a running instance.
///
/// Handles are freely cloneable and their `Drop` is a **no-op** — internal
/// copies (instance lists, queues, scheduler temporaries) must never affect
/// slot accounting.  In-flight slots are owned by `SlotGuard`, created only
/// by `try_acquire`, which releases the slot exactly once on drop.
#[derive(Clone)]
pub struct InstanceHandle {
    inner: Arc<Mutex<Instance>>,
}

impl InstanceHandle {
    pub fn new(instance: Instance) -> Self {
        Self {
            inner: Arc::new(Mutex::new(instance)),
        }
    }

    /// Unique identifier of this instance (e.g. `"qwen3-32b@0,1"`).
    pub fn id(&self) -> String {
        self.inner.lock().unwrap().id.clone()
    }

    /// Acquire a slot on the instance.  Returns a RAII guard that releases
    /// the slot on drop, or `None` if no capacity was available.
    pub fn try_acquire(&self, max_concurrent: usize) -> Option<SlotGuard> {
        let mut inst = self.inner.lock().unwrap();
        if inst.has_capacity(max_concurrent) {
            inst.acquire_slot();
            Some(SlotGuard::new(self.clone()))
        } else {
            None
        }
    }

    /// Direct access to the inner mutex (for state transitions).
    pub fn inner(&self) -> &Arc<Mutex<Instance>> {
        &self.inner
    }
}

// ── Slot guard (RAII slot ownership) ─────────────────────────────────────────

/// RAII guard representing one acquired in-flight slot on an instance.
///
/// Created exclusively by `InstanceHandle::try_acquire` — exactly one guard
/// per acquired slot.  Dropping the guard releases the slot.  The guard is
/// deliberately **not** `Clone`: a guard models unique ownership of a slot,
/// so slot releases stay balanced with acquisitions no matter how often the
/// underlying `InstanceHandle` is cloned.
pub struct SlotGuard {
    handle: InstanceHandle,
}

impl SlotGuard {
    fn new(handle: InstanceHandle) -> Self {
        Self { handle }
    }

    /// Access the underlying instance handle (e.g. to read id/port).
    pub fn handle(&self) -> &InstanceHandle {
        &self.handle
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if let Ok(mut inst) = self.handle.inner.lock() {
            inst.release_slot();
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn display_gpus(indices: &[usize]) -> String {
    if indices.is_empty() {
        "cpu".into()
    } else {
        indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_id_format() {
        let inst = Instance::new("qwen3-32b", vec![0, 1], 9001, None);
        assert_eq!(inst.id, "qwen3-32b@0,1");
    }

    #[test]
    fn instance_id_cpu() {
        let inst = Instance::new("tiny-model", vec![], 9002, None);
        assert_eq!(inst.id, "tiny-model@cpu");
    }

    #[test]
    fn handle_acquire_release() {
        let mut inst = Instance::new("test", vec![0], 9003, None);
        inst.mark_ready();
        let handle = InstanceHandle::new(inst);
        let guard = handle.try_acquire(4).expect("slot available");
        {
            let inst = handle.inner.lock().unwrap();
            assert_eq!(inst.in_flight, 1);
        }
        drop(guard);
        {
            let inst = handle.inner.lock().unwrap();
            assert_eq!(inst.in_flight, 0);
        }
    }

    #[test]
    fn handle_no_capacity() {
        let mut inst = Instance::new("test", vec![0], 9004, None);
        inst.mark_ready();
        let handle = InstanceHandle::new(inst);
        // max_concurrent = 1, acquire one slot
        let _guard = handle.try_acquire(1).expect("first slot available");
        // second acquire should fail
        assert!(handle.try_acquire(1).is_none());
    }

    #[test]
    fn handle_clone_drop_does_not_release_slot() {
        // Regression test: handle clones (instance lists, queues, scheduler
        // temporaries) must never affect slot accounting.  Only dropping the
        // SlotGuard releases the slot.
        let mut inst = Instance::new("test", vec![0], 9005, None);
        inst.mark_ready();
        let handle = InstanceHandle::new(inst);
        let guard = handle.try_acquire(4).expect("slot available");
        for _ in 0..10 {
            let clone = handle.clone();
            drop(clone);
        }
        assert_eq!(handle.inner.lock().unwrap().in_flight, 1);
        let probe = handle.clone();
        drop(handle);
        assert_eq!(probe.inner.lock().unwrap().in_flight, 1);
        drop(guard);
        assert_eq!(probe.inner.lock().unwrap().in_flight, 0);
    }
}
