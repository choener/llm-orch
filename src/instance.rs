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
    /// No new requests are accepted, but existing slot guards may continue
    /// using the backend until it becomes idle.
    Draining,
    /// The backend is no longer serving requests and is awaiting registry
    /// removal.  This is a short-lived internal state.
    Retired,
    /// Health check or spawn failed, or the backend crashed and is no longer
    /// usable.
    Failed,
}

/// Why a backend instance is being loaded.
///
/// Keeping the purpose on the instance lets the scheduler distinguish an
/// ordinary on-demand spawn from the temporary duplicate used by a future
/// migration transaction, without making the process routable before it is
/// ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceLoadPurpose {
    /// Normal request-driven or autoscaling spawn.
    Normal,
    /// Temporary replacement for an existing instance during migration.
    MigrationReplacement,
    /// Target model load performed as part of a make-room transaction.
    LargeModelAdmission,
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

    /// Purpose of the load that created this instance.
    pub load_purpose: InstanceLoadPurpose,

    /// Logical routing generation served by this physical instance.
    ///
    /// Generation `0` is the initial/default generation.  Migration and
    /// reload coordination may assign later generations before an instance
    /// becomes the active logical route.
    pub generation: u64,

    /// Transition that owns this instance, if it participates in one.
    pub migration_id: Option<u64>,

    /// Subprocess handle (`None` before spawn, or after the process exits).
    pub child: Option<Child>,

    /// Podman container name, when the backend runs inside a container.
    /// `None` for native process backends.  Set at spawn time; used during
    /// shutdown to call `podman stop <name>` before killing the child.
    pub container_name: Option<String>,

    /// Ring buffer of the backend's recent stdout/stderr lines.  Set by
    /// the manager at spawn time; dumped at `warn!` level when the
    /// instance fails readiness or crashes.
    pub output: Option<crate::backend::OutputBuffer>,

    /// Number of in-flight requests currently being processed.
    pub in_flight: usize,

    /// Timestamp of the last request activity (acquire or release, for TTL
    /// tracking).
    pub last_active: Instant,

    /// Timestamp of the last slot release.  Kept separate from
    /// `last_active` so migration victim ordering can use completion history
    /// rather than the most recent request start.
    pub last_completed: Instant,

    /// Channel sender for notifying the manager on slot release.
    /// `None` in tests or after shutdown.
    pub release_tx: Option<mpsc::UnboundedSender<String>>,

    /// Fingerprint of the spawn-relevant config (resolved `cmd`, selected
    /// device pool, `vram`, `context_length`) this instance was launched
    /// with.  Set by the manager at spawn time; `0` means "unset" (tests).
    /// On config reload, instances whose fingerprint no longer matches are
    /// retired (marked `Draining`) and replaced on demand.
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
        Self::new_with_purpose(
            model_name,
            gpu_indices,
            port,
            release_tx,
            InstanceLoadPurpose::Normal,
        )
    }

    /// Create an instance descriptor with an explicit load purpose.
    pub fn new_with_purpose(
        model_name: &str,
        gpu_indices: Vec<usize>,
        port: u16,
        release_tx: Option<mpsc::UnboundedSender<String>>,
        load_purpose: InstanceLoadPurpose,
    ) -> Self {
        let id = format!("{}@{}", model_name, display_gpus(&gpu_indices));
        let now = Instant::now();
        Self {
            id,
            model_name: model_name.to_owned(),
            port,
            gpu_indices,
            state: InstanceState::Loading,
            load_purpose,
            generation: 0,
            migration_id: None,
            child: None,
            container_name: None,
            output: None,
            in_flight: 0,
            last_active: now,
            last_completed: now,
            release_tx,
            config_fingerprint: 0,
        }
    }

    /// Mark this instance as ready (health check passed).
    pub fn mark_ready(&mut self) {
        self.state = InstanceState::Ready;
    }

    /// Mark this instance as draining.  Existing slot guards remain valid,
    /// but future acquisitions reject the instance atomically.
    pub fn mark_draining(&mut self, migration_id: Option<u64>) {
        self.state = InstanceState::Draining;
        self.migration_id = migration_id;
    }

    /// Mark this instance as retired and ready for registry removal.
    pub fn mark_retired(&mut self) {
        self.state = InstanceState::Retired;
    }

    /// Whether this instance is already draining.
    pub fn is_draining(&self) -> bool {
        self.state == InstanceState::Draining
    }

    /// Clear migration ownership once the instance becomes the active logical
    /// route or a pre-handoff rollback restores the old route.
    pub fn clear_migration(&mut self) {
        self.migration_id = None;
    }

    /// Mark this instance as failed and remove any transition association.
    pub fn mark_failed(&mut self) {
        self.state = InstanceState::Failed;
        self.migration_id = None;
    }

    /// Whether the instance can receive a new request slot.  A ready
    /// migration replacement remains behind its barrier until handoff clears
    /// `migration_id`.
    pub fn is_routable(&self) -> bool {
        self.state == InstanceState::Ready && self.migration_id.is_none()
    }

    /// Whether a load or ready instance counts against the ordinary logical
    /// instance cap.  A migration replacement is a temporary duplicate and
    /// is counted through its coordinator reservation instead.
    pub fn counts_toward_instance_cap(&self) -> bool {
        matches!(self.state, InstanceState::Loading | InstanceState::Ready)
            && !(self.migration_id.is_some()
                && self.load_purpose == InstanceLoadPurpose::MigrationReplacement)
    }

    /// Whether a release or readiness event can eventually wake a queued
    /// request.  Loading is included because its spawn completion may make a
    /// slot available; draining and terminal states cannot accept work.
    pub fn can_wake_waiters(&self) -> bool {
        matches!(self.state, InstanceState::Loading | InstanceState::Ready)
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
        let now = Instant::now();
        self.last_active = now;
        if was_active {
            self.last_completed = now;
            if let Some(ref tx) = self.release_tx {
                let _ = tx.send(self.model_name.clone());
            }
        }
    }

    /// Whether this instance has spare capacity for another request.
    pub fn has_capacity(&self, max_concurrent: usize) -> bool {
        self.is_routable() && self.in_flight < max_concurrent
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
    fn migration_metadata_and_draining_state() {
        let mut inst = Instance::new_with_purpose(
            "test",
            vec![1],
            9003,
            None,
            InstanceLoadPurpose::MigrationReplacement,
        );
        assert_eq!(inst.state, InstanceState::Loading);
        assert_eq!(inst.load_purpose, InstanceLoadPurpose::MigrationReplacement);
        assert_eq!(inst.generation, 0);
        assert_eq!(inst.migration_id, None);

        inst.mark_draining(Some(7));
        assert!(inst.is_draining());
        assert_eq!(inst.migration_id, Some(7));
        assert!(!inst.is_routable());
        assert!(!inst.counts_toward_instance_cap());
        assert!(!inst.can_wake_waiters());

        inst.mark_ready();
        assert!(!inst.is_routable());
        assert!(!inst.has_capacity(4));
        assert_eq!(inst.migration_id, Some(7));
        inst.clear_migration();
        assert!(inst.is_routable());
        assert!(inst.has_capacity(4));
    }

    #[test]
    fn migration_replacement_is_barriered_until_handoff() {
        let mut inst = Instance::new_with_purpose(
            "test",
            vec![0],
            9003,
            None,
            InstanceLoadPurpose::MigrationReplacement,
        );
        inst.mark_ready();
        inst.migration_id = Some(1);
        let handle = InstanceHandle::new(inst);

        assert!(!handle.inner.lock().unwrap().counts_toward_instance_cap());
        assert!(handle.try_acquire(4).is_none());

        handle.inner.lock().unwrap().clear_migration();
        assert!(handle.try_acquire(4).is_some());
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
            assert!(inst.last_completed >= inst.last_active);
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
