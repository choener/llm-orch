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
/// Wrapped in `Arc<Mutex<…>>` so an `InstanceHandle` can share ownership and
/// release the in-flight slot on drop.
pub struct Instance {
    /// Deterministic ID, e.g. `"qwen3-32b@0,1"`.
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

    /// Number of in-flight requests currently being processed.
    pub in_flight: usize,

    /// Timestamp of the last request completion (for TTL tracking).
    pub last_active: Instant,

    /// How many times this instance has crashed *before* producing output.
    /// Reset on first successful health check.
    pub crash_count: usize,

    /// Channel sender for notifying the manager on slot release.
    /// `None` in tests or after shutdown.
    pub release_tx: Option<mpsc::UnboundedSender<String>>,
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
            in_flight: 0,
            last_active: Instant::now(),
            crash_count: 0,
            release_tx,
        }
    }

    /// Mark this instance as ready (health check passed).
    pub fn mark_ready(&mut self) {
        self.state = InstanceState::Ready;
        self.crash_count = 0;
    }

    /// Mark this instance as failed.
    pub fn mark_failed(&mut self) {
        self.state = InstanceState::Failed;
    }

    /// Increment the in-flight counter and bump the activity timestamp.
    pub fn acquire_slot(&mut self) {
        self.in_flight += 1;
        self.last_active = Instant::now();
    }

    /// Decrement the in-flight counter.  Called by `InstanceHandle::drop()`.
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

    /// Called after the child process exits.  Returns `true` if this was a
    /// "zero-output crash" — the instance never reached Ready, so the crash
    /// counter should be incremented and the model potentially blocked.
    pub fn on_exit(&mut self) -> bool {
        self.state = InstanceState::Failed;
        if self.crash_count == 0 && self.child.is_some() {
            false
        } else {
            self.crash_count += 1;
            true
        }
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
/// When the last handle referring to a particular request context is dropped,
/// `release_slot()` is called automatically, preventing leaked in-flight counts.
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

    /// Acquire a slot on the instance.  Returns `true` if capacity was available.
    pub fn try_acquire(&self, max_concurrent: usize) -> bool {
        let mut inst = self.inner.lock().unwrap();
        if inst.has_capacity(max_concurrent) {
            inst.acquire_slot();
            true
        } else {
            false
        }
    }

    /// Release the in-flight slot explicitly (normally done by Drop).
    pub fn release(&self) {
        self.inner.lock().unwrap().release_slot();
    }

    /// Return a clone of the inner `Arc` for sharing across tasks.
    pub fn clone_arc(&self) -> Arc<Mutex<Instance>> {
        Arc::clone(&self.inner)
    }

    /// Direct access to the inner mutex (for state transitions).
    pub fn inner(&self) -> &Arc<Mutex<Instance>> {
        &self.inner
    }
}

impl Drop for InstanceHandle {
    fn drop(&mut self) {
        if let Ok(mut inst) = self.inner.lock() {
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
        assert!(handle.try_acquire(4));
        {
            let inst = handle.inner.lock().unwrap();
            assert_eq!(inst.in_flight, 1);
        }
        handle.release();
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
        assert!(handle.try_acquire(1));
        // second acquire should fail
        assert!(!handle.try_acquire(1));
    }
}
