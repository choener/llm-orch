// ── Instance manager ─────────────────────────────────────────────────────────
//
// Central scheduler: owns the map of model → running instances, handles
// spawning, slot acquisition, queueing, idle eviction, and shutdown.

use crate::backend::{
    Backend, ContainerBackend, DeviceKind, LlamaCppBackend, ReadyOutcome,
    StableDiffusionCppBackend, inject_container_args, output_lines, poll_readiness, shutdown_child,
    shutdown_container, spawn_process,
};
use crate::config::{
    AliasPolicy, DevicePool, EmaWindow, MakeRoomMode, ModelConfig, Runner, load_key_files,
    resolve_remote_auth,
};
use crate::gpu::GpuMetrics;
use crate::http_client;
use crate::instance::{Instance, InstanceHandle, InstanceLoadPurpose, InstanceState, SlotGuard};
use crate::keepalive::KeepAliveManager;
use crate::port_alloc::PortAllocator;
use crate::types::{CompletionRecord, QueueStatus, RequestState, RequestStatus};

use reqwest::Client;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore, mpsc, oneshot, watch};
use tracing::{debug, error, info, warn};

/// How long a spawn holds the global spawn semaphore while waiting for
/// readiness.  Covers process start + initial load — the window where
/// concurrent loads would thrash the SSD and interleave VRAM allocations.
/// Slow loads keep waiting *without* the semaphore afterwards so one
/// slow model cannot head-of-line block spawns of every other model.
const SPAWN_SERIAL_WINDOW: Duration = Duration::from_secs(15);

/// Timeout for the single `/health` probe of a remote endpoint during
/// provisioning.  A reachable endpoint answers quickly; anything slower is
/// treated as down (the request queues or fails, and a later request
/// re-probes).
const REMOTE_PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Snapshot of the logical route captured when a request enters a queue.
/// Physical instance handles are deliberately absent: migration may replace
/// every backend for the model before this route is assigned.
#[derive(Debug, Clone)]
struct QueueRouteSnapshot {
    logical_model: String,
    candidates: Vec<String>,
    policy: AliasPolicy,
    make_room: MakeRoomMode,
    drain_timeout_secs: u64,
    max_concurrent: usize,
    generation: u64,
    migration_id: Option<u64>,
    request_id: Option<String>,
}

/// Terminal conditions delivered through a queue assignment channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueueTerminalError {
    Deadline,
    ModelRemoved,
    MigrationFailed,
    Shutdown,
}

/// A logical queued request and its assignment channel.
struct QueueEntry {
    id: u64,
    route: QueueRouteSnapshot,
    enqueued_at: Instant,
    deadline: Option<Instant>,
    assignment: oneshot::Sender<Result<(String, SlotGuard), QueueTerminalError>>,
    cancelled: Arc<AtomicBool>,
}

type WaitQueue = VecDeque<QueueEntry>;

/// Owns a queued request while its assignment future is alive.  Dropping
/// the ticket marks the entry cancelled and removes it immediately, so a
/// disconnected client does not consume queue depth until the next wake.
struct QueueTicket {
    manager: Arc<InstanceManager>,
    model_name: String,
    id: u64,
    cancelled: Arc<AtomicBool>,
    expired: Arc<AtomicBool>,
    request_id: Option<String>,
}

impl QueueTicket {
    fn mark_expired(&self) {
        self.expired.store(true, Ordering::Release);
        self.cancelled.store(true, Ordering::Release);
        self.manager
            .scheduler_metrics
            .queue_expired_total
            .fetch_add(1, Ordering::Relaxed);
        if let Some(request_id) = &self.request_id {
            self.manager.update_request_state(
                request_id,
                RequestState::Expired,
                None,
                Some("queue wait deadline expired"),
            );
        }
    }
}

impl Drop for QueueTicket {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        if self.manager.remove_queued(&self.model_name, self.id)
            && !self.expired.load(Ordering::Acquire)
        {
            self.manager
                .scheduler_metrics
                .queue_cancelled_total
                .fetch_add(1, Ordering::Relaxed);
            if let Some(request_id) = &self.request_id {
                self.manager.update_request_state(
                    request_id,
                    RequestState::Cancelled,
                    None,
                    Some("client disconnected while queued"),
                );
            }
            debug!(model = %self.model_name, queue_id = self.id, "queue entry cancelled");
        }
    }
}

/// The result of registering a slow admission request.
pub(crate) struct PendingAssignment {
    receiver: oneshot::Receiver<Result<(String, SlotGuard), QueueTerminalError>>,
    deadline: Option<Instant>,
    _ticket: QueueTicket,
}

impl PendingAssignment {
    pub(crate) async fn wait(self) -> Result<(String, SlotGuard), AcquireError> {
        let PendingAssignment {
            receiver,
            deadline,
            _ticket: ticket,
        } = self;
        let result = match deadline {
            Some(deadline) => {
                let wait = deadline.saturating_duration_since(Instant::now());
                match tokio::time::timeout(wait, receiver).await {
                    Ok(result) => result,
                    Err(_) => {
                        ticket.mark_expired();
                        return Err(AcquireError::QueueTimeout);
                    }
                }
            }
            None => receiver.await,
        };
        match result {
            Ok(Ok((model_name, guard))) => Ok((model_name, guard)),
            Ok(Err(QueueTerminalError::Deadline)) => {
                ticket.mark_expired();
                Err(AcquireError::QueueTimeout)
            }
            Err(_) => {
                if let Some(request_id) = &ticket.request_id {
                    ticket.manager.update_request_state(
                        request_id,
                        RequestState::Cancelled,
                        None,
                        Some("queue assignment sender closed"),
                    );
                }
                Err(AcquireError::NoCapacity)
            }
            Ok(Err(
                QueueTerminalError::ModelRemoved
                | QueueTerminalError::MigrationFailed
                | QueueTerminalError::Shutdown,
            )) => {
                if let Some(request_id) = &ticket.request_id {
                    ticket.manager.update_request_state(
                        request_id,
                        RequestState::Failed,
                        None,
                        Some("queued model became unavailable"),
                    );
                }
                Err(AcquireError::Unavailable)
            }
        }
    }
}

/// Either an immediately acquired slot or a queue-backed assignment future.
pub(crate) enum Admission {
    Assigned(String, SlotGuard),
    Waiting(PendingAssignment),
}

/// Slow work to start after a queue entry has been registered.
enum AdmissionDemand {
    Spawn {
        model_name: String,
        cfg: ModelConfig,
    },
    LegacyMakeRoom {
        model_name: String,
        cfg: ModelConfig,
        candidates: Vec<String>,
        mode: MakeRoomMode,
        drain_timeout_secs: u64,
    },
    Migration {
        model_name: String,
        cfg: ModelConfig,
    },
}

/// The operation that owns a GPU reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationPurpose {
    /// A request-driven or autoscaling backend spawn.
    NormalSpawn,
    /// A temporary replacement backend for a migration.
    MigrationReplacement,
    /// The target backend load of a make-room admission transaction.
    LargeModelAdmission,
}

/// Lifecycle phase of a GPU reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationPhase {
    /// Placement has been selected but process startup has not begun.
    Planned,
    /// The reservation is held while a backend is loading.
    Loading,
    /// A migration replacement is ready while the old backend drains.
    WaitingForDrain,
    /// The logical route handoff or target admission has committed.
    Committed,
}

/// Cleanup metadata attached to every reservation entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservationCleanup {
    /// Token used by the owning coordinator to release a whole reservation
    /// set idempotently, even if a load-failure cleanup races another path.
    pub token: u64,
    /// Optional stale-operation deadline.  The coordinator owns enforcing
    /// this deadline; the table only stores it as cleanup metadata.
    pub expires_at: Option<Instant>,
}

/// One reservation request, potentially covering multiple PCI slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservationRequest {
    pub transition_id: u64,
    pub purpose: ReservationPurpose,
    pub model: String,
    pub pci_slots: Vec<String>,
    pub vram_per_slot: u64,
    pub phase: ReservationPhase,
    pub cleanup: ReservationCleanup,
}

/// One per-PCI-slot reservation entry.
///
/// A multi-GPU placement creates one entry per PCI slot so projected usage
/// follows the same conservative accounting rule as registered instances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuReservation {
    pub transition_id: u64,
    pub purpose: ReservationPurpose,
    pub model: String,
    pub pci_slot: String,
    pub vram_per_slot: u64,
    pub phase: ReservationPhase,
    pub cleanup: ReservationCleanup,
}

/// A deterministic physical placement together with projected VRAM usage.
///
/// The current selector produces identical peak and final projections because
/// it represents one ordinary spawn.  Migration planning will use the same
/// shape for a multi-phase transaction where peak usage includes both the old
/// and replacement backends while final usage does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementSnapshot {
    /// Backend device indices in deterministic launch order.
    pub device_indices: Vec<usize>,
    /// PCI slots corresponding to `device_indices` in the same order.
    pub pci_slots: Vec<String>,
    /// Projected per-slot VRAM after the placement commits.
    pub projected_final_vram: HashMap<String, u64>,
    /// Maximum per-slot VRAM during the placement operation.
    pub projected_peak_vram: HashMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReservationKey {
    transition_id: u64,
    pci_slot: String,
}

/// Errors from an atomic reservation attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReservationError {
    /// The same PCI slot occurred more than once in one placement.
    DuplicateSlot(String),
    /// Another reservation already owns this PCI slot.
    AlreadyReserved {
        transition_id: u64,
        pci_slot: String,
    },
    /// The coordinator task has stopped, so no new reservation can be made.
    CoordinatorUnavailable,
}

/// Scheduler-owned table of temporary GPU reservations.
///
/// The table deliberately does not decide whether a placement fits.  The
/// coordinator performs capacity checks and then uses `reserve_placement` to
/// commit all slots atomically; this type guarantees that a partial
/// multi-GPU reservation can never be visible.
#[derive(Debug, Default)]
pub struct ReservationTable {
    entries: HashMap<ReservationKey, GpuReservation>,
}

impl ReservationTable {
    /// Reserve every slot in a placement atomically.
    pub fn reserve_placement(
        &mut self,
        request: ReservationRequest,
    ) -> Result<(), ReservationError> {
        self.reserve_batch(vec![request])
    }

    /// Reserve several placements as one atomic transaction.  This is needed
    /// by migration, which reserves both the temporary victim replacement and
    /// the eventual target placement before changing either route.
    pub fn reserve_batch(
        &mut self,
        requests: Vec<ReservationRequest>,
    ) -> Result<(), ReservationError> {
        let mut seen = HashSet::new();
        for request in &requests {
            for pci_slot in &request.pci_slots {
                if !seen.insert(pci_slot.clone()) {
                    return Err(ReservationError::DuplicateSlot(pci_slot.clone()));
                }
                if let Some(existing) = self
                    .entries
                    .values()
                    .find(|reservation| reservation.pci_slot == *pci_slot)
                {
                    return Err(ReservationError::AlreadyReserved {
                        transition_id: existing.transition_id,
                        pci_slot: pci_slot.clone(),
                    });
                }
            }
        }

        for request in requests {
            for pci_slot in request.pci_slots {
                let key = ReservationKey {
                    transition_id: request.transition_id,
                    pci_slot: pci_slot.clone(),
                };
                self.entries.insert(
                    key,
                    GpuReservation {
                        transition_id: request.transition_id,
                        purpose: request.purpose,
                        model: request.model.clone(),
                        pci_slot,
                        vram_per_slot: request.vram_per_slot,
                        phase: request.phase,
                        cleanup: request.cleanup,
                    },
                );
            }
        }
        Ok(())
    }

    /// Update the phase for all entries owned by a transition.
    ///
    /// Returns `true` when at least one entry was updated.  Empty CPU
    /// placements therefore have no phase entry to update.
    pub fn set_phase(&mut self, transition_id: u64, phase: ReservationPhase) -> bool {
        let mut updated = false;
        for reservation in self.entries.values_mut() {
            if reservation.transition_id == transition_id {
                reservation.phase = phase;
                updated = true;
            }
        }
        updated
    }

    /// Return projected reserved VRAM by PCI slot, optionally excluding one
    /// transition's own reservations for a dry-run.
    pub fn usage_by_slot(&self, excluding_transition: Option<u64>) -> HashMap<String, u64> {
        let mut usage = HashMap::new();
        for reservation in self.entries.values() {
            if excluding_transition == Some(reservation.transition_id) {
                continue;
            }
            *usage.entry(reservation.pci_slot.clone()).or_default() += reservation.vram_per_slot;
        }
        usage
    }

    /// Snapshot current entries without exposing the table's internal keys.
    pub fn snapshot(&self) -> Vec<GpuReservation> {
        self.entries.values().cloned().collect()
    }

    /// Release all entries for a transition.  Safe to call repeatedly.
    pub fn release_transition(&mut self, transition_id: u64) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, reservation| reservation.transition_id != transition_id);
        before - self.entries.len()
    }

    /// Release all entries with a cleanup token.  Safe to call repeatedly and
    /// useful when cleanup is driven by a lease rather than a transition ID.
    pub fn release_cleanup_token(&mut self, token: u64) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, reservation| reservation.cleanup.token != token);
        before - self.entries.len()
    }

    /// Release reservations for one model.  Used when a config reload removes
    /// a model before any associated transition can reach a commit point.
    pub fn release_model(&mut self, model: &str) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, reservation| reservation.model != model);
        before - self.entries.len()
    }

    /// Release every reservation.  Shutdown uses this after stopping loads.
    pub fn release_all(&mut self) -> usize {
        let count = self.entries.len();
        self.entries.clear();
        count
    }

    /// Remove entries whose cleanup deadline has passed.
    pub fn release_expired(&mut self, now: Instant) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, reservation| {
            reservation
                .cleanup
                .expires_at
                .map(|deadline| deadline > now)
                .unwrap_or(true)
        });
        before - self.entries.len()
    }

    /// Return the next cleanup deadline, if any.
    pub fn next_expiry(&self) -> Option<Instant> {
        self.entries
            .values()
            .filter_map(|reservation| reservation.cleanup.expires_at)
            .min()
    }
}

/// Commands processed by the single scheduler admission coordinator.
///
/// The coordinator owns the mutable reservation table and transition ID
/// source.  Callers never mutate reservation state directly; they submit a
/// command and await the result, so reservation commits are serialized even
/// while backend startup is awaiting readiness elsewhere.
enum CoordinatorCommand {
    AllocateTransition {
        reply: oneshot::Sender<u64>,
    },
    ReservePlacement {
        request: ReservationRequest,
        reply: oneshot::Sender<Result<(), ReservationError>>,
    },
    ReserveBatch {
        requests: Vec<ReservationRequest>,
        reply: oneshot::Sender<Result<(), ReservationError>>,
    },
    SetPhase {
        transition_id: u64,
        phase: ReservationPhase,
        reply: oneshot::Sender<bool>,
    },
    Usage {
        excluding_transition: Option<u64>,
        reply: oneshot::Sender<HashMap<String, u64>>,
    },
    Snapshot {
        reply: oneshot::Sender<Vec<GpuReservation>>,
    },
    ReleaseTransition {
        transition_id: u64,
        reply: oneshot::Sender<usize>,
    },
    ReleaseToken {
        token: u64,
        reply: oneshot::Sender<usize>,
    },
    ReleaseModel {
        model: String,
        reply: oneshot::Sender<usize>,
    },
    ReleaseAll {
        reply: oneshot::Sender<usize>,
    },
}

/// Handle used by the manager to submit commands to the coordinator task.
#[derive(Clone)]
struct CoordinatorHandle {
    tx: mpsc::UnboundedSender<CoordinatorCommand>,
}

impl CoordinatorHandle {
    fn start() -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel();
        // `InstanceManager::new` is also used by a few synchronous unit
        // tests that only exercise pure helpers.  There is no useful
        // reservation work to perform without a runtime, so close the
        // command receiver there rather than panicking during construction.
        if tokio::runtime::Handle::try_current().is_err() {
            drop(rx);
            return Self { tx };
        }
        let coordinator = async move {
            let mut reservations = ReservationTable::default();
            let mut next_transition_id = 0u64;
            loop {
                let command = if let Some(deadline) = reservations.next_expiry() {
                    let wait = deadline.saturating_duration_since(Instant::now());
                    tokio::select! {
                        command = rx.recv() => command,
                        _ = tokio::time::sleep(wait) => {
                            reservations.release_expired(Instant::now());
                            continue;
                        }
                    }
                } else {
                    rx.recv().await
                };
                let Some(command) = command else {
                    break;
                };
                match command {
                    CoordinatorCommand::AllocateTransition { reply } => {
                        let id = next_transition_id;
                        next_transition_id = next_transition_id.wrapping_add(1);
                        let _ = reply.send(id);
                    }
                    CoordinatorCommand::ReservePlacement { request, reply } => {
                        let _ = reply.send(reservations.reserve_placement(request));
                    }
                    CoordinatorCommand::ReserveBatch { requests, reply } => {
                        let _ = reply.send(reservations.reserve_batch(requests));
                    }
                    CoordinatorCommand::SetPhase {
                        transition_id,
                        phase,
                        reply,
                    } => {
                        let _ = reply.send(reservations.set_phase(transition_id, phase));
                    }
                    CoordinatorCommand::Usage {
                        excluding_transition,
                        reply,
                    } => {
                        let _ = reply.send(reservations.usage_by_slot(excluding_transition));
                    }
                    CoordinatorCommand::Snapshot { reply } => {
                        let _ = reply.send(reservations.snapshot());
                    }
                    CoordinatorCommand::ReleaseTransition {
                        transition_id,
                        reply,
                    } => {
                        let _ = reply.send(reservations.release_transition(transition_id));
                    }
                    CoordinatorCommand::ReleaseToken { token, reply } => {
                        let _ = reply.send(reservations.release_cleanup_token(token));
                    }
                    CoordinatorCommand::ReleaseModel { model, reply } => {
                        let _ = reply.send(reservations.release_model(&model));
                    }
                    CoordinatorCommand::ReleaseAll { reply } => {
                        let _ = reply.send(reservations.release_all());
                    }
                }
            }
        };
        tokio::spawn(coordinator);
        Self { tx }
    }

    async fn allocate_transition_id(&self) -> Option<u64> {
        let (reply, response) = oneshot::channel();
        self.tx
            .send(CoordinatorCommand::AllocateTransition { reply })
            .ok()?;
        response.await.ok()
    }

    async fn reserve_placement(&self, request: ReservationRequest) -> Result<(), ReservationError> {
        let (reply, response) = oneshot::channel();
        self.tx
            .send(CoordinatorCommand::ReservePlacement { request, reply })
            .map_err(|_| ReservationError::CoordinatorUnavailable)?;
        response
            .await
            .map_err(|_| ReservationError::CoordinatorUnavailable)?
    }

    async fn reserve_batch(
        &self,
        requests: Vec<ReservationRequest>,
    ) -> Result<(), ReservationError> {
        let (reply, response) = oneshot::channel();
        self.tx
            .send(CoordinatorCommand::ReserveBatch { requests, reply })
            .map_err(|_| ReservationError::CoordinatorUnavailable)?;
        response
            .await
            .map_err(|_| ReservationError::CoordinatorUnavailable)?
    }

    async fn set_phase(&self, transition_id: u64, phase: ReservationPhase) -> bool {
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(CoordinatorCommand::SetPhase {
                transition_id,
                phase,
                reply,
            })
            .is_err()
        {
            return false;
        }
        response.await.unwrap_or(false)
    }

    async fn usage(&self, excluding_transition: Option<u64>) -> HashMap<String, u64> {
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(CoordinatorCommand::Usage {
                excluding_transition,
                reply,
            })
            .is_err()
        {
            return HashMap::new();
        }
        response.await.unwrap_or_default()
    }

    async fn snapshot(&self) -> Vec<GpuReservation> {
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(CoordinatorCommand::Snapshot { reply })
            .is_err()
        {
            return Vec::new();
        }
        response.await.unwrap_or_default()
    }

    async fn release_transition(&self, transition_id: u64) -> usize {
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(CoordinatorCommand::ReleaseTransition {
                transition_id,
                reply,
            })
            .is_err()
        {
            return 0;
        }
        response.await.unwrap_or(0)
    }

    async fn release_token(&self, token: u64) -> usize {
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(CoordinatorCommand::ReleaseToken { token, reply })
            .is_err()
        {
            return 0;
        }
        response.await.unwrap_or(0)
    }

    async fn release_model(&self, model: String) -> usize {
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(CoordinatorCommand::ReleaseModel { model, reply })
            .is_err()
        {
            return 0;
        }
        response.await.unwrap_or(0)
    }

    async fn release_all(&self) -> usize {
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(CoordinatorCommand::ReleaseAll { reply })
            .is_err()
        {
            return 0;
        }
        response.await.unwrap_or(0)
    }
}

/// Why `get_or_spawn` could not provide an instance slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquireError {
    /// The model is crash-blocked.
    Blocked,
    /// Instances exist but all are busy, and the queue is full.
    NoCapacity,
    /// An accepted waiter reached its configured maximum queue wait.
    QueueTimeout,
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
    #[cfg(test)]
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

// ── Request status registry ──────────────────────────────────────────────────

const REQUEST_STATUS_MAX_ENTRIES: usize = 4096;
const REQUEST_STATUS_RETENTION: Duration = Duration::from_secs(15 * 60);

#[derive(Default)]
struct SchedulerCounters {
    queue_enqueued_total: AtomicU64,
    queue_assigned_total: AtomicU64,
    queue_cancelled_total: AtomicU64,
    queue_expired_total: AtomicU64,
    migration_started_total: AtomicU64,
    migration_completed_total: AtomicU64,
    migration_failed_total: AtomicU64,
    reservation_cleanup_total: AtomicU64,
}

impl SchedulerCounters {
    fn snapshot(&self) -> crate::types::SchedulerMetrics {
        crate::types::SchedulerMetrics {
            queue_enqueued_total: self.queue_enqueued_total.load(Ordering::Relaxed),
            queue_assigned_total: self.queue_assigned_total.load(Ordering::Relaxed),
            queue_cancelled_total: self.queue_cancelled_total.load(Ordering::Relaxed),
            queue_expired_total: self.queue_expired_total.load(Ordering::Relaxed),
            migration_started_total: self.migration_started_total.load(Ordering::Relaxed),
            migration_completed_total: self.migration_completed_total.load(Ordering::Relaxed),
            migration_failed_total: self.migration_failed_total.load(Ordering::Relaxed),
            reservation_cleanup_total: self.reservation_cleanup_total.load(Ordering::Relaxed),
        }
    }
}

struct RequestStatusRecord {
    status: RequestStatus,
    created_at: Instant,
    terminal_at: Option<Instant>,
}

fn request_state_is_terminal(state: RequestState) -> bool {
    matches!(
        state,
        RequestState::Completed
            | RequestState::Expired
            | RequestState::Cancelled
            | RequestState::Shutdown
            | RequestState::Failed
    )
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

        let alpha_1 = 1.0 - (-dt / 60.0_f64).exp();
        let alpha_5 = 1.0 - (-dt / 300.0_f64).exp();
        let alpha_15 = 1.0 - (-dt / 900.0_f64).exp();

        // ── Load averages ───────────────────────────────────────────
        self.load_m1 = self.load_m1 * (1.0 - alpha_1) + active_f * alpha_1;
        self.load_m5 = self.load_m5 * (1.0 - alpha_5) + active_f * alpha_5;
        self.load_m15 = self.load_m15 * (1.0 - alpha_15) + active_f * alpha_15;

        // ── Request rate ───────────────────────────────────────────
        if completions_delta > 0 {
            let rate_per_min = (completions_delta as f64) / (dt / 60.0);
            self.req_rate_m1 = self.req_rate_m1 * (1.0 - alpha_1) + rate_per_min * alpha_1;
            self.req_rate_m5 = self.req_rate_m5 * (1.0 - alpha_5) + rate_per_min * alpha_5;
            self.req_rate_m15 = self.req_rate_m15 * (1.0 - alpha_15) + rate_per_min * alpha_15;
            self.completions_total += completions_delta;
        } else {
            self.req_rate_m1 *= 1.0 - alpha_1;
            self.req_rate_m5 *= 1.0 - alpha_5;
            self.req_rate_m15 *= 1.0 - alpha_15;
        }

        self.last_update = now;
        self.last_active = active;
    }
}

// ── Spawn-config fingerprint ────────────────────────────────────────────────

/// Fingerprint of the configuration that defines the spawned backend
/// process: the alias-resolved command line (`{port}` deliberately
/// excluded — it differs per instance), the selected device pool, declared
/// VRAM, and context length.  Routing-only fields (`max_instances`,
/// `max_concurrent`, `queue_depth`, `idle_ttl`, `autoscale`, `debug_log`,
/// `priority`, `ram`) are excluded: changing them must not retire running
/// instances.
///
/// Instances whose stored fingerprint no longer matches the current
/// config are retired on reload (marked `Draining`) and replaced
/// on demand.  Runtime-only value — never persisted, so `DefaultHasher`'s
/// version instability is irrelevant.
fn fingerprint_with_aliases(cmd_aliases: &HashMap<String, String>, cfg: &ModelConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut resolved = cfg.cmd.clone();
    for (key, value) in cmd_aliases {
        resolved = resolved.replace(&format!("{{{}}}", key), value);
    }
    let devices = cfg.vulkan_devices.canonical_indices();
    let cuda_devices = cfg.cuda_devices.canonical_indices();
    let rocm_devices = cfg.rocm_devices.canonical_indices();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    resolved.hash(&mut h);
    devices.hash(&mut h);
    cuda_devices.hash(&mut h);
    rocm_devices.hash(&mut h);
    cfg.gpus.hash(&mut h);
    cfg.vram.hash(&mut h);
    cfg.context_length.hash(&mut h);
    // For remote models `resolved` is the base URL and `remote_auth` is
    // the auth template — both define the provisioned instance, so
    // changing either retires existing instances on config reload.
    cfg.remote_auth.hash(&mut h);
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
    let slots = devs.cuda.iter().map(|(k, v)| (*k, v.pci.clone())).collect();
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

    /// Port allocator.
    ports: Mutex<PortAllocator>,

    /// Per-model configuration, indexed by model name.
    /// Swapped on config hot-reload (`reconcile_config`).
    model_configs: RwLock<HashMap<String, ModelConfig>>,

    /// Named `{key}` → value fragments from `cmd_aliases`.
    cmd_aliases: RwLock<HashMap<String, String>>,

    /// Resolved `key_files` contents (`name → trimmed file contents`),
    /// used to resolve `remote_auth` templates at request time.  Re-read
    /// on config hot-reload and when the key files themselves change.
    key_files: RwLock<HashMap<String, String>>,

    /// Running instances, keyed by model name.
    instances: RwLock<HashMap<String, Vec<InstanceHandle>>>,

    /// Per-model wait queues.  When all instances are at capacity, requests
    /// park on a oneshot channel until a slot frees up, the queue is full,
    /// or the wait times out.  Each entry carries a unique id so timed-out
    /// waiters can remove themselves from the queue.
    queues: RwLock<HashMap<String, WaitQueue>>,

    /// Unique id source for queue entries (used for timeout self-removal).
    next_queue_id: AtomicU64,

    /// Single serialized admission coordinator.  It owns transition IDs and
    /// the reservation table, while backend startup continues outside the
    /// command loop so unrelated loads are not blocked on readiness.
    coordinator: CoordinatorHandle,

    /// Incremented after each validated config reconciliation.  Migration
    /// tasks capture the epoch and roll back at safe pre-handoff points if a
    /// reload changes their immutable placement/config snapshot.
    config_epoch: AtomicU64,

    /// Set when shutdown begins; prevents new admissions and migrations while
    /// allowing already-running handlers to drain until the HTTP bound.
    shutdown_requested: AtomicBool,

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

    /// ROCm device index → PCI slot mapping (from config).
    rocm_slots: RwLock<HashMap<usize, String>>,

    /// GPU keep-alive manager (None if not configured).
    /// Rebuilt on config hot-reload when the keep-alive section changes.
    keepalive: RwLock<Option<Arc<KeepAliveManager>>>,

    /// Crash limit before a model is blocked.
    crash_limit: usize,

    /// Spawn readiness timeout.
    spawn_timeout: Duration,

    /// Global unconditional idle reap timeout (seconds) — the "deep sleep"
    /// bound: any instance idle past it is evicted.  The per-model
    /// `idle_ttl` is only a make-room protection window (an instance idle
    /// for less than that is never evicted to make room); this timeout is
    /// what actually unloads idle models.  Swapped on config hot-reload.
    /// See `docs/003-smart-handling.md`, decision 8.
    drain_idle_timeout_secs: RwLock<u64>,

    /// Maximum scheduler queue wait in seconds; zero means no scheduler
    /// deadline.  Backend timeouts begin only after assignment.
    queue_wait_timeout_secs: RwLock<u64>,

    /// EMA window used for autoscale scale-up decisions (server-global,
    /// hot-reloaded).  `m1` reacts to bursts within a minute.
    scale_up_value: RwLock<EmaWindow>,

    /// EMA window used for autoscale scale-down decisions (server-global,
    /// hot-reloaded).
    scale_down_value: RwLock<EmaWindow>,

    /// Maximum migration transition duration; zero means no transition
    /// deadline.  Queue wait deadlines remain separate.
    migration_timeout_secs: RwLock<u64>,

    /// Per-model EMA metrics (load & request rate).
    model_metrics: RwLock<HashMap<String, ModelMetrics>>,

    /// Bounded in-memory request lifecycle registry used by the authenticated
    /// request-status endpoint.  Prompts and API keys are never stored.
    request_statuses: std::sync::RwLock<HashMap<String, RequestStatusRecord>>,

    /// Low-cardinality scheduler counters exposed through info/status.
    scheduler_metrics: SchedulerCounters,

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

    /// Per-target in-flight migration tasks.  All callers for the same
    /// logical target subscribe to one transition rather than planning
    /// competing evictions.
    migrations_in_flight: std::sync::RwLock<HashMap<String, watch::Sender<()>>>,
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
    /// Load `key_files` contents, downgrading read failures to a warning
    /// (a missing key degrades one remote model's auth instead of taking
    /// the whole service down; `--check-config` already verified
    /// existence at load time).
    fn load_key_files_or_warn(config: &crate::config::Config) -> HashMap<String, String> {
        match load_key_files(&config.key_files) {
            Ok(keys) => {
                if !keys.is_empty() {
                    info!(count = keys.len(), "key_files loaded");
                }
                keys
            }
            Err(e) => {
                warn!("key_files load failed (continuing without): {e}");
                HashMap::new()
            }
        }
    }

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
        let rocm_slots = config
            .devices
            .as_ref()
            .map(|d| d.rocm.clone())
            .unwrap_or_default();

        let (release_tx, release_rx) = mpsc::unbounded_channel();
        let (crash_tx, crash_rx) = mpsc::unbounded_channel();

        let mgr = Self {
            client: http_client::build(),
            ports: Mutex::new(PortAllocator::new(config.server.port_range.clone())),
            model_configs: RwLock::new(model_configs),
            cmd_aliases: RwLock::new(config.cmd_aliases.clone()),
            key_files: RwLock::new(Self::load_key_files_or_warn(config)),
            instances: RwLock::new(HashMap::new()),
            queues: RwLock::new(HashMap::new()),
            next_queue_id: AtomicU64::new(0),
            coordinator: CoordinatorHandle::start(),
            config_epoch: AtomicU64::new(0),
            shutdown_requested: AtomicBool::new(false),
            instance_seq: AtomicU64::new(0),
            blocked: RwLock::new(HashMap::new()),
            gpu_snapshot,
            vulkan_slots: RwLock::new(vulkan_slots),
            vram_limits: RwLock::new(vram_limits),
            cuda_slots: RwLock::new(cuda_slots),
            cuda_vram_static: RwLock::new(cuda_vram_static),
            rocm_slots: RwLock::new(rocm_slots),
            keepalive: RwLock::new(keepalive),
            crash_limit: 3,
            spawn_timeout: Duration::from_secs(120),
            drain_idle_timeout_secs: RwLock::new(config.server.drain_idle_timeout_secs),
            queue_wait_timeout_secs: RwLock::new(config.server.queue_wait_timeout_secs),
            scale_up_value: RwLock::new(config.server.scale_up_value),
            scale_down_value: RwLock::new(config.server.scale_down_value),
            migration_timeout_secs: RwLock::new(config.server.migration_timeout_secs),
            model_metrics: RwLock::new(HashMap::new()),
            request_statuses: std::sync::RwLock::new(HashMap::new()),
            scheduler_metrics: SchedulerCounters::default(),
            spawn_semaphore: Arc::new(Semaphore::new(1)),
            last_scale_action: RwLock::new(HashMap::new()),
            recent_completions: RwLock::new(HashMap::new()),
            release_tx,
            crash_tx,
            crash_counts: std::sync::Mutex::new(HashMap::new()),
            spawns_in_flight: std::sync::RwLock::new(HashMap::new()),
            migrations_in_flight: std::sync::RwLock::new(HashMap::new()),
        };
        (mgr, release_rx, crash_rx)
    }

    // ── reservation bookkeeping ───────────────────────────────────────────

    /// Allocate a unique ID for an admission or migration transition.
    pub async fn allocate_transition_id(&self) -> Option<u64> {
        self.coordinator.allocate_transition_id().await
    }

    /// Atomically reserve all PCI slots in one placement.
    ///
    /// Capacity and device-pool eligibility are checked by the coordinator;
    /// this method only commits the reservation set in the coordinator task.
    pub async fn reserve_placement(
        &self,
        request: ReservationRequest,
    ) -> Result<(), ReservationError> {
        self.coordinator.reserve_placement(request).await
    }

    /// Atomically reserve all placements belonging to one transition.
    pub async fn reserve_placements(
        &self,
        requests: Vec<ReservationRequest>,
    ) -> Result<(), ReservationError> {
        self.coordinator.reserve_batch(requests).await
    }

    /// Update all reservation entries belonging to a transition.
    pub async fn set_reservation_phase(&self, transition_id: u64, phase: ReservationPhase) -> bool {
        self.coordinator.set_phase(transition_id, phase).await
    }

    /// Return projected reservation usage by PCI slot.
    pub async fn reservation_usage(
        &self,
        excluding_transition: Option<u64>,
    ) -> HashMap<String, u64> {
        self.coordinator.usage(excluding_transition).await
    }

    /// Return a snapshot of all current reservation entries.
    pub async fn reservation_snapshot(&self) -> Vec<GpuReservation> {
        self.coordinator.snapshot().await
    }

    /// Release every reservation owned by a transition.  Safe to call from
    /// multiple cleanup paths.
    pub async fn release_reservations(&self, transition_id: u64) -> usize {
        let released = self.coordinator.release_transition(transition_id).await;
        if released > 0 {
            self.scheduler_metrics
                .reservation_cleanup_total
                .fetch_add(released as u64, Ordering::Relaxed);
        }
        released
    }

    /// Release every reservation carrying a cleanup token.  Safe to call
    /// repeatedly when a load-failure, timeout, or shutdown cleanup races.
    pub async fn release_reservations_by_token(&self, token: u64) -> usize {
        let released = self.coordinator.release_token(token).await;
        if released > 0 {
            self.scheduler_metrics
                .reservation_cleanup_total
                .fetch_add(released as u64, Ordering::Relaxed);
        }
        released
    }

    /// Release reservations associated with a model during removal or reload.
    pub async fn release_model_reservations(&self, model: &str) -> usize {
        let released = self.coordinator.release_model(model.to_owned()).await;
        if released > 0 {
            self.scheduler_metrics
                .reservation_cleanup_total
                .fetch_add(released as u64, Ordering::Relaxed);
        }
        released
    }

    /// Release all reservations during final shutdown cleanup.
    pub async fn release_all_reservations(&self) -> usize {
        let released = self.coordinator.release_all().await;
        if released > 0 {
            self.scheduler_metrics
                .reservation_cleanup_total
                .fetch_add(released as u64, Ordering::Relaxed);
        }
        released
    }

    // ── request status and queue metrics ─────────────────────────────────

    /// Register an accepted request before scheduler admission begins.
    /// Stop new request admission and infrastructure migrations.
    pub fn begin_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::Release);
    }

    fn is_shutting_down(&self) -> bool {
        self.shutdown_requested.load(Ordering::Acquire)
    }

    pub fn register_request(&self, request_id: &str, logical_model: &str) {
        let mut statuses = self.request_statuses.write().unwrap();
        statuses.insert(
            request_id.to_owned(),
            RequestStatusRecord {
                status: RequestStatus {
                    request_id: request_id.to_owned(),
                    logical_model: logical_model.to_owned(),
                    resolved_model: None,
                    state: RequestState::Queued,
                    queue: None,
                    queue_position: None,
                    enqueued_at: crate::debug_log::ts_now(),
                    wait_duration_ms: 0,
                    deadline: None,
                    migration_id: None,
                    migration_phase: None,
                    victim_instance: None,
                    victim_gpu: Vec::new(),
                    destination_gpu: Vec::new(),
                    target_gpu: Vec::new(),
                    active_victim_slots: 0,
                    reservation_peak_vram: HashMap::new(),
                    reservation_final_vram: HashMap::new(),
                    failure_reason: None,
                },
                created_at: Instant::now(),
                terminal_at: None,
            },
        );
        self.prune_request_statuses_locked(&mut statuses);
    }

    /// Update request lifecycle state without storing request contents.
    pub fn update_request_state(
        &self,
        request_id: &str,
        state: RequestState,
        resolved_model: Option<&str>,
        failure_reason: Option<&str>,
    ) {
        let mut statuses = self.request_statuses.write().unwrap();
        let Some(record) = statuses.get_mut(request_id) else {
            return;
        };
        record.status.state = state;
        if let Some(model) = resolved_model {
            record.status.resolved_model = Some(model.to_owned());
        }
        if let Some(reason) = failure_reason {
            record.status.failure_reason = Some(reason.to_owned());
        }
        record.status.wait_duration_ms = record.created_at.elapsed().as_millis() as u64;
        if request_state_is_terminal(state) {
            record.terminal_at.get_or_insert_with(Instant::now);
        }
        self.prune_request_statuses_locked(&mut statuses);
    }

    /// Return one request status, pruning terminal entries past retention.
    pub fn request_status(&self, request_id: &str) -> Option<RequestStatus> {
        let mut statuses = self.request_statuses.write().unwrap();
        self.prune_request_statuses_locked(&mut statuses);
        statuses.get(request_id).map(|record| record.status.clone())
    }

    fn prune_request_statuses_locked(&self, statuses: &mut HashMap<String, RequestStatusRecord>) {
        let now = Instant::now();
        statuses.retain(|_, record| {
            record
                .terminal_at
                .map(|terminal| now.duration_since(terminal) < REQUEST_STATUS_RETENTION)
                .unwrap_or(true)
        });
        while statuses.len() > REQUEST_STATUS_MAX_ENTRIES {
            let oldest = statuses
                .iter()
                .min_by_key(|(_, record)| record.terminal_at.unwrap_or(record.created_at))
                .map(|(id, _)| id.clone());
            let Some(id) = oldest else { break };
            statuses.remove(&id);
        }
    }

    /// Return aggregate scheduler counters.
    pub fn scheduler_metrics_snapshot(&self) -> crate::types::SchedulerMetrics {
        self.scheduler_metrics.snapshot()
    }

    /// Count queued entries by the reason currently blocking assignment.
    pub fn queue_status(&self, model_name: &str) -> QueueStatus {
        let loading = self
            .instances
            .read()
            .unwrap()
            .get(model_name)
            .map(|instances| {
                instances
                    .iter()
                    .any(|handle| handle.inner().lock().unwrap().state == InstanceState::Loading)
            })
            .unwrap_or(false);
        let migration = self
            .migrations_in_flight
            .read()
            .unwrap()
            .contains_key(model_name);
        let queues = self.queues.read().unwrap();
        let mut result = QueueStatus::default();
        for queue in queues.values() {
            for entry in queue {
                if entry.route.logical_model != model_name
                    || entry.cancelled.load(Ordering::Acquire)
                {
                    continue;
                }
                if migration || entry.route.migration_id.is_some() {
                    result.waiting_for_migration += 1;
                } else if loading {
                    result.waiting_for_spawn += 1;
                } else {
                    result.waiting_for_slot += 1;
                }
            }
        }
        result
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
    pub async fn get_or_spawn(
        self: &Arc<Self>,
        model_name: &str,
    ) -> Result<SlotGuard, AcquireError> {
        // Direct (non-alias) requests are a one-element candidate list with
        // make-room disabled — decision 9 of docs/003-smart-handling.md.
        self.acquire_for_candidates(
            std::slice::from_ref(&model_name.to_owned()),
            AliasPolicy::PreferLoaded,
            MakeRoomMode::None,
            0,
        )
        .await
        .map(|(_, guard)| guard)
    }

    /// Register an admission and return either an immediate slot or a
    /// queue-backed assignment future.  Slow spawn, legacy make-room, and
    /// migration work is started only after the queue entry exists.
    pub(crate) async fn admit(
        self: &Arc<Self>,
        candidates: &[String],
        policy: AliasPolicy,
        make_room: MakeRoomMode,
        drain_timeout: u64,
        request_id: Option<&str>,
    ) -> Result<Admission, AcquireError> {
        if self.is_shutting_down() {
            return Err(AcquireError::Unavailable);
        }
        let mut resolved: Vec<(String, ModelConfig)> = Vec::new();
        let mut any_blocked = false;
        {
            let configs = self.model_configs.read().unwrap();
            for name in candidates {
                if self.is_blocked(name) {
                    any_blocked = true;
                    debug!(model = %name, "candidate skipped — model blocked");
                    continue;
                }
                match configs.get(name) {
                    Some(cfg) => resolved.push((name.clone(), cfg.clone())),
                    None => debug!(model = %name, "candidate skipped — unknown model"),
                }
            }
        }
        if resolved.is_empty() {
            return Err(if any_blocked {
                AcquireError::Blocked
            } else {
                AcquireError::Unavailable
            });
        }

        let mut demand = None;
        let mut queue_choice = None;
        match policy {
            AliasPolicy::PreferLoaded => {
                for (name, cfg) in &resolved {
                    if let Some(guard) = self.acquire_ready_slot(name, cfg.max_concurrent) {
                        return Ok(Admission::Assigned(name.clone(), guard));
                    }
                }
                for (name, cfg) in &resolved {
                    if self.should_spawn(name, cfg)
                        && self.spawn_feasible(cfg, &HashSet::new()).await
                    {
                        demand = Some(AdmissionDemand::Spawn {
                            model_name: name.clone(),
                            cfg: cfg.clone(),
                        });
                        queue_choice = Some((name.clone(), cfg.clone()));
                        break;
                    }
                }
                if demand.is_none() {
                    for (name, cfg) in &resolved {
                        if !self.should_spawn(name, cfg) {
                            continue;
                        }
                        if self.spawn_feasible(cfg, &HashSet::new()).await {
                            continue;
                        }
                        demand = match make_room {
                            mode if mode.is_migration() => Some(AdmissionDemand::Migration {
                                model_name: name.clone(),
                                cfg: cfg.clone(),
                            }),
                            mode if mode.is_legacy_eviction() => {
                                Some(AdmissionDemand::LegacyMakeRoom {
                                    model_name: name.clone(),
                                    cfg: cfg.clone(),
                                    candidates: candidates.to_vec(),
                                    mode,
                                    drain_timeout_secs: drain_timeout,
                                })
                            }
                            _ => None,
                        };
                        if demand.is_some() {
                            queue_choice = Some((name.clone(), cfg.clone()));
                            break;
                        }
                    }
                }
            }
            AliasPolicy::PreferOrder => {
                for (name, cfg) in &resolved {
                    if let Some(guard) = self.acquire_ready_slot(name, cfg.max_concurrent) {
                        return Ok(Admission::Assigned(name.clone(), guard));
                    }
                    if !self.should_spawn(name, cfg) {
                        continue;
                    }
                    let feasible = self.spawn_feasible(cfg, &HashSet::new()).await;
                    demand = if feasible {
                        Some(AdmissionDemand::Spawn {
                            model_name: name.clone(),
                            cfg: cfg.clone(),
                        })
                    } else if make_room.is_migration() {
                        Some(AdmissionDemand::Migration {
                            model_name: name.clone(),
                            cfg: cfg.clone(),
                        })
                    } else if make_room.is_legacy_eviction() {
                        Some(AdmissionDemand::LegacyMakeRoom {
                            model_name: name.clone(),
                            cfg: cfg.clone(),
                            candidates: candidates.to_vec(),
                            mode: make_room,
                            drain_timeout_secs: drain_timeout,
                        })
                    } else {
                        None
                    };
                    if demand.is_some() {
                        queue_choice = Some((name.clone(), cfg.clone()));
                        break;
                    }
                }
            }
        }

        let (queue_name, queue_cfg) = queue_choice.unwrap_or_else(|| {
            resolved
                .iter()
                .find(|(name, _)| self.has_routable_instance(name))
                .cloned()
                .unwrap_or_else(|| resolved[0].clone())
        });
        // If no demand can be started and no routable instance exists, the
        // request cannot ever be woken.  This preserves fail-fast behavior
        // for direct requests whose model does not fit and has no migration
        // policy, while spawnable cold starts still queue below.
        if demand.is_none() && !self.has_routable_instance(&queue_name) {
            return Err(AcquireError::Unavailable);
        }

        // A zero-depth queue rejects waiting immediately, but preserve the
        // existing behavior of triggering a fitting normal spawn before the
        // rejection is returned.  No request has been accepted in this
        // branch, so it does not violate the pre-registration rule.
        if queue_cfg.queue_depth == 0 {
            if let Some(demand) = demand {
                match demand {
                    AdmissionDemand::Spawn { model_name, cfg } => {
                        // Preserve the legacy zero-depth behavior: trigger
                        // the fitting load and let it finish before reporting
                        // that this request cannot wait in the queue.
                        let mut rx = self.ensure_spawn(&model_name, &cfg);
                        let budget =
                            self.spawn_timeout + SPAWN_SERIAL_WINDOW + Duration::from_secs(30);
                        let _ = tokio::time::timeout(budget, rx.changed()).await;
                    }
                    other => self.start_admission_demand(other),
                }
            }
            return Err(AcquireError::NoCapacity);
        }

        let route = QueueRouteSnapshot {
            logical_model: queue_name.clone(),
            candidates: resolved.iter().map(|(name, _)| name.clone()).collect(),
            policy,
            make_room,
            drain_timeout_secs: drain_timeout,
            max_concurrent: queue_cfg.max_concurrent,
            generation: self.model_generation(&queue_name),
            migration_id: self.model_migration_id(&queue_name),
            request_id: request_id.map(str::to_owned),
        };
        let pending = self
            .register_pending(route, queue_cfg.queue_depth, true)
            .map_err(|error| match error {
                EnqueueError::QueueFull => AcquireError::NoCapacity,
                #[cfg(test)]
                EnqueueError::Timeout => AcquireError::QueueTimeout,
                EnqueueError::NoInstances => AcquireError::Unavailable,
            })?;

        if let Some(demand) = demand {
            self.start_admission_demand(demand);
        }
        Ok(Admission::Waiting(pending))
    }

    /// Acquire an instance slot for one of `candidates` (in preference
    /// order), returning the chosen model name alongside the slot guard.
    pub async fn acquire_for_candidates(
        self: &Arc<Self>,
        candidates: &[String],
        policy: AliasPolicy,
        make_room: MakeRoomMode,
        drain_timeout: u64,
    ) -> Result<(String, SlotGuard), AcquireError> {
        match self
            .admit(candidates, policy, make_room, drain_timeout, None)
            .await?
        {
            Admission::Assigned(model_name, guard) => Ok((model_name, guard)),
            Admission::Waiting(pending) => pending.wait().await,
        }
    }

    /// Autoscale spawn gate: only spawn if the load EMA exceeds the
    /// scale-up threshold (always true for cold-start — zero existing
    /// instances — and when autoscale is disabled or absent).
    ///
    /// Threshold: `scale_up_at + (n−1) × max_concurrent` for n existing
    /// instances.  Since `in_flight ≤ n × max_concurrent`, the rule can
    /// only fire with a non-empty queue — "spawn when the queue EMA
    /// exceeds `scale_up_at − max_concurrent`".
    fn should_spawn(&self, model_name: &str, cfg: &ModelConfig) -> bool {
        let max_concurrent = cfg.max_concurrent;
        if let Some(ref a) = cfg.autoscale {
            if !a.enabled {
                true
            } else {
                // Retiring (`Failed`) instances don't count: from a routing
                // perspective the model has zero instances, so a config
                // change that retired the only instance triggers a cold-start
                // replacement spawn instead of queueing behind the drain.
                let num_existing = {
                    self.instances
                        .read()
                        .unwrap()
                        .get(model_name)
                        .map(|l| {
                            l.iter()
                                .filter(|h| h.inner().lock().unwrap().counts_toward_instance_cap())
                                .count()
                        })
                        .unwrap_or(0)
                };
                if num_existing == 0 {
                    true // cold start — always spawn immediately
                } else {
                    let load =
                        self.load_for_window(model_name, *self.scale_up_value.read().unwrap());
                    let threshold = (a.scale_up_at + max_concurrent * (num_existing - 1)) as f64;
                    if load <= threshold {
                        debug!(
                            model = %model_name,
                            load = %load,
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
        }
    }

    /// Current load-EMA value for the given window (load = `in_flight` +
    /// queued requests).  `0.0` when the model has no metrics entry yet.
    fn load_for_window(&self, model_name: &str, window: EmaWindow) -> f64 {
        let metrics = self.model_metrics.read().unwrap();
        let m = match metrics.get(model_name) {
            Some(m) => m,
            None => return 0.0,
        };
        match window {
            EmaWindow::M1 => m.load_m1,
            EmaWindow::M5 => m.load_m5,
            EmaWindow::M15 => m.load_m15,
        }
    }

    /// Spawn-feasibility dry-run: would `try_spawn` get past the instance
    /// cap and the device-pool check right now, assuming the instances in
    /// `excluding` (make-room victims) were gone?
    ///
    /// Racy by nature — another spawn may grab the VRAM between this check
    /// and the actual spawn; `try_spawn` re-checks under the spawn
    /// semaphore and simply fails then, which the caller handles.
    async fn spawn_feasible(&self, cfg: &ModelConfig, excluding: &HashSet<String>) -> bool {
        // Instance cap (mirrors try_spawn): retiring (`Draining`/`Failed`)
        // instances and planned make-room victims don't count.
        {
            let instances = self.instances.read().unwrap();
            if let Some(list) = instances.get(&cfg.name) {
                let active = list
                    .iter()
                    .filter(|h| {
                        let inst = h.inner().lock().unwrap();
                        inst.counts_toward_instance_cap() && !excluding.contains(&inst.id)
                    })
                    .count();
                if active >= cfg.max_instances {
                    return false;
                }
            }
        }
        // Without a device pool there is nothing to check — the spawn
        // proceeds unpinned (CPU or unrestricted GPUs).
        if !cfg.has_device_pool() {
            return true;
        }
        let available = if excluding.is_empty() {
            self.select_gpus_for_model(cfg).await.len()
        } else {
            self.select_gpus_inner(cfg, excluding).await.len()
        };
        available >= cfg.gpus
    }

    /// Whether the model has at least one routable (non-retiring)
    /// instance — i.e. a release or spawn event could wake a queued waiter.
    fn has_routable_instance(&self, model_name: &str) -> bool {
        self.instances
            .read()
            .unwrap()
            .get(model_name)
            .map(|l| {
                l.iter()
                    .any(|h| h.inner().lock().unwrap().can_wake_waiters())
            })
            .unwrap_or(false)
    }

    /// Current logical generation for a model.  Queued requests store this
    /// value as a diagnostic snapshot; physical routing is re-evaluated at
    /// assignment time.
    fn model_generation(&self, model_name: &str) -> u64 {
        self.instances
            .read()
            .unwrap()
            .get(model_name)
            .map(|list| {
                list.iter()
                    .map(|handle| handle.inner().lock().unwrap().generation)
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    }

    /// Return the migration barrier currently attached to a logical model.
    fn model_migration_id(&self, model_name: &str) -> Option<u64> {
        self.instances
            .read()
            .unwrap()
            .get(model_name)
            .and_then(|list| {
                list.iter()
                    .find_map(|handle| handle.inner().lock().unwrap().migration_id)
            })
    }

    /// A queued route may be assigned only when no newer/different migration
    /// barrier is active.  Generation changes alone are safe because the
    /// queue is logical rather than instance-bound.
    fn route_allows_assignment(&self, route: &QueueRouteSnapshot) -> bool {
        match (
            route.migration_id,
            self.model_migration_id(&route.logical_model),
        ) {
            (None, Some(_)) | (Some(_), Some(_)) => false,
            (_, None) => true,
        }
    }

    // ── make-room eviction (docs/003-smart-handling.md) ──────────────────

    /// Enumerate migration candidates in deterministic LRU order.
    ///
    /// Unlike legacy make-room, migration may move a sole instance, but only
    /// when a complete replacement placement can already be proven on a
    /// different PCI-slot set.  This method is intentionally a snapshot:
    /// callers must re-check the destination and reserve it before changing
    /// routing.
    async fn enumerate_migration_victims(
        &self,
        requesting_model: &str,
    ) -> Vec<MigrationVictimCandidate> {
        let configs = self.model_configs.read().unwrap().clone();
        let mut candidates = Vec::new();
        let instances: Vec<(String, InstanceHandle)> = {
            let instances = self.instances.read().unwrap();
            instances
                .iter()
                .flat_map(|(model_name, list)| {
                    list.iter()
                        .map(|handle| (model_name.clone(), handle.clone()))
                        .collect::<Vec<_>>()
                })
                .collect()
        };

        for (model_name, handle) in instances {
            if model_name == requesting_model {
                continue;
            }
            let Some(cfg) = configs.get(&model_name) else {
                continue;
            };
            let (
                instance_id,
                state,
                in_flight,
                last_completed,
                gpu_indices,
                migration_id,
                generation,
            ) = {
                let inst = handle.inner().lock().unwrap();
                (
                    inst.id.clone(),
                    inst.state,
                    inst.in_flight,
                    inst.last_completed,
                    inst.gpu_indices.clone(),
                    inst.migration_id,
                    inst.generation,
                )
            };
            if state != InstanceState::Ready || migration_id.is_some() || gpu_indices.is_empty() {
                continue;
            }
            let Some(source_pci_slots) = self.pci_slots_for_instance(&model_name, &gpu_indices)
            else {
                continue;
            };
            let blocked_slots: HashSet<String> = source_pci_slots.iter().cloned().collect();
            let mut excluding = HashSet::new();
            excluding.insert(instance_id.clone());
            let Some(destination) = self
                .select_placement_snapshot_with_blocked(cfg, &excluding, &blocked_slots)
                .await
            else {
                continue;
            };
            candidates.push(MigrationVictimCandidate {
                model_name,
                handle,
                instance_id,
                generation,
                source_pci_slots,
                destination,
                last_completed,
                in_flight,
            });
        }

        candidates.sort_by(|a, b| {
            let class = |in_flight: usize| usize::from(in_flight > 0);
            class(a.in_flight)
                .cmp(&class(b.in_flight))
                .then(a.last_completed.cmp(&b.last_completed))
                .then(a.in_flight.cmp(&b.in_flight))
                .then(a.instance_id.cmp(&b.instance_id))
        });
        candidates
    }

    /// Dry-run a transactional migration for `target_cfg`.
    ///
    /// Each candidate already has a destination proven to fit on a different
    /// PCI-slot set.  The target placement is then checked with the old
    /// victim excluded and the replacement's VRAM added as virtual usage,
    /// ensuring the temporary duplicate is charged before the source can be
    /// considered free.
    async fn plan_migration(&self, target_cfg: &ModelConfig) -> Option<MigrationPlan> {
        let configs = self.model_configs.read().unwrap().clone();
        for victim in self.enumerate_migration_victims(&target_cfg.name).await {
            let Some(victim_cfg) = configs.get(&victim.model_name) else {
                continue;
            };
            let replacement_vram = victim_cfg.vram * 1024 * 1024;
            let extra_vram: HashMap<String, u64> = victim
                .destination
                .pci_slots
                .iter()
                .map(|slot| (slot.clone(), replacement_vram))
                .collect();
            let mut excluding = HashSet::new();
            excluding.insert(victim.instance_id.clone());
            let Some(target) = self
                .select_placement_snapshot_with_blocked_and_extra(
                    target_cfg,
                    &excluding,
                    &HashSet::new(),
                    &extra_vram,
                )
                .await
            else {
                continue;
            };
            if !target
                .pci_slots
                .iter()
                .any(|slot| victim.source_pci_slots.contains(slot))
            {
                continue;
            }
            return Some(MigrationPlan {
                target_model: target_cfg.name.clone(),
                target,
                victim,
            });
        }
        None
    }

    /// Execute one transactional victim migration followed by the target
    /// load.  The old backend remains alive while the replacement loads and
    /// while existing slot guards finish; only a zero-in-flight old backend
    /// is removed.
    pub(crate) async fn run_migration(
        &self,
        target_cfg: &ModelConfig,
    ) -> Result<(), MigrationError> {
        let transition_epoch = self.config_epoch.load(Ordering::Acquire);
        let plan = self
            .plan_migration(target_cfg)
            .await
            .ok_or(MigrationError::NoPlan)?;
        debug_assert_eq!(plan.target_model, target_cfg.name);
        if self.config_epoch.load(Ordering::Acquire) != transition_epoch {
            return Err(MigrationError::ConfigChanged);
        }
        let migration_id = self
            .allocate_transition_id()
            .await
            .ok_or(MigrationError::ReservationUnavailable)?;
        info!(
            migration_id,
            target = %plan.target_model,
            victim = %plan.victim.instance_id,
            destination = ?plan.victim.destination.pci_slots,
            target_gpu = ?plan.target.pci_slots,
            "migration planned"
        );
        let migration_timeout_secs = *self.migration_timeout_secs.read().unwrap();
        let expires_at = (migration_timeout_secs > 0)
            .then(|| Instant::now() + Duration::from_secs(migration_timeout_secs));
        let cleanup = ReservationCleanup {
            token: migration_id,
            expires_at,
        };
        let configs = self.model_configs.read().unwrap().clone();
        let Some(victim_cfg) = configs.get(&plan.victim.model_name) else {
            return Err(MigrationError::VictimChanged);
        };
        let reservation_requests = vec![
            ReservationRequest {
                transition_id: migration_id,
                purpose: ReservationPurpose::MigrationReplacement,
                model: plan.victim.model_name.clone(),
                pci_slots: plan.victim.destination.pci_slots.clone(),
                vram_per_slot: victim_cfg.vram * 1024 * 1024,
                phase: ReservationPhase::Planned,
                cleanup,
            },
            ReservationRequest {
                transition_id: migration_id,
                purpose: ReservationPurpose::LargeModelAdmission,
                model: target_cfg.name.clone(),
                pci_slots: plan.target.pci_slots.clone(),
                vram_per_slot: target_cfg.vram * 1024 * 1024,
                phase: ReservationPhase::Planned,
                cleanup,
            },
        ];
        if self.reserve_placements(reservation_requests).await.is_err() {
            warn!(migration_id, "migration reservation rejected");
            return Err(MigrationError::ReservationUnavailable);
        }
        info!(migration_id, "migration reservations acquired");
        if self.config_epoch.load(Ordering::Acquire) != transition_epoch {
            self.release_reservations(migration_id).await;
            return Err(MigrationError::ConfigChanged);
        }

        let old_generation = plan.victim.generation;
        let victim_marked = {
            let mut victim = plan.victim.handle.inner().lock().unwrap();
            if victim.state != InstanceState::Ready || victim.migration_id.is_some() {
                false
            } else {
                victim.mark_draining(Some(migration_id));
                true
            }
        };
        if !victim_marked {
            self.release_reservations(migration_id).await;
            return Err(MigrationError::VictimChanged);
        }
        if self.config_epoch.load(Ordering::Acquire) != transition_epoch {
            self.rollback_migration(&plan, migration_id, None).await;
            return Err(MigrationError::ConfigChanged);
        }
        info!(migration_id, victim = %plan.victim.instance_id, "migration victim draining");
        self.set_reservation_phase(migration_id, ReservationPhase::Loading)
            .await;

        let replacement = self
            .try_spawn_with_options(
                &plan.victim.model_name,
                victim_cfg,
                SpawnOptions {
                    requested_gpu_indices: Some(plan.victim.destination.device_indices.clone()),
                    load_purpose: InstanceLoadPurpose::MigrationReplacement,
                    migration_id: Some(migration_id),
                    reservation_id: Some(migration_id),
                    enforce_instance_cap: false,
                    retain_reservation: true,
                },
            )
            .await;
        let Some(replacement) = replacement else {
            self.rollback_migration(&plan, migration_id, None).await;
            return Err(MigrationError::ReplacementLoadFailed);
        };
        replacement.inner().lock().unwrap().generation = old_generation + 1;
        info!(
            migration_id,
            replacement = %replacement.id(),
            destination = ?plan.victim.destination.pci_slots,
            "migration replacement ready"
        );
        if self.config_epoch.load(Ordering::Acquire) != transition_epoch {
            self.rollback_migration(&plan, migration_id, Some(&replacement))
                .await;
            return Err(MigrationError::ConfigChanged);
        }
        self.set_reservation_phase(migration_id, ReservationPhase::WaitingForDrain)
            .await;

        let deadline = expires_at;
        loop {
            let (state, active, owner) = {
                let victim = plan.victim.handle.inner().lock().unwrap();
                (victim.state, victim.in_flight, victim.migration_id)
            };
            if state != InstanceState::Draining || owner != Some(migration_id) {
                self.rollback_migration(&plan, migration_id, Some(&replacement))
                    .await;
                return Err(MigrationError::VictimChanged);
            }
            if active == 0 {
                break;
            }
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                self.rollback_migration(&plan, migration_id, Some(&replacement))
                    .await;
                return Err(MigrationError::ReplacementLoadFailed);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let replacement_ready = {
            let replacement = replacement.inner().lock().unwrap();
            replacement.state == InstanceState::Ready
                && replacement.migration_id == Some(migration_id)
        };
        if !replacement_ready {
            self.rollback_migration(&plan, migration_id, Some(&replacement))
                .await;
            return Err(MigrationError::ReplacementLoadFailed);
        }
        if self.config_epoch.load(Ordering::Acquire) != transition_epoch {
            self.rollback_migration(&plan, migration_id, Some(&replacement))
                .await;
            return Err(MigrationError::ConfigChanged);
        }

        self.set_reservation_phase(migration_id, ReservationPhase::Committed)
            .await;
        {
            let mut replacement_inst = replacement.inner().lock().unwrap();
            replacement_inst.clear_migration();
            replacement_inst.generation = old_generation + 1;
        }
        info!(migration_id, victim = %plan.victim.instance_id, "migration handoff committing");
        self.force_remove_instance(&plan.victim.model_name, &plan.victim.handle)
            .await;
        self.set_reservation_phase(migration_id, ReservationPhase::Loading)
            .await;

        let config_changed = self.config_epoch.load(Ordering::Acquire) != transition_epoch;
        let deadline_expired = deadline.is_some_and(|limit| Instant::now() >= limit);
        if config_changed || deadline_expired {
            self.retain_migrated_replacement_after_target_failure(
                migration_id,
                &plan.victim.model_name,
                &replacement,
            )
            .await;
            return Err(if config_changed {
                MigrationError::ConfigChanged
            } else {
                MigrationError::TargetLoadFailed
            });
        }

        info!(migration_id, target = %target_cfg.name, placement = ?plan.target.pci_slots, "migration target loading");
        let target = self
            .try_spawn_with_options(
                &target_cfg.name,
                target_cfg,
                SpawnOptions {
                    requested_gpu_indices: Some(plan.target.device_indices.clone()),
                    load_purpose: InstanceLoadPurpose::LargeModelAdmission,
                    migration_id: Some(migration_id),
                    reservation_id: Some(migration_id),
                    enforce_instance_cap: true,
                    retain_reservation: true,
                },
            )
            .await;
        let Some(target) = target else {
            self.retain_migrated_replacement_after_target_failure(
                migration_id,
                &plan.victim.model_name,
                &replacement,
            )
            .await;
            return Err(MigrationError::TargetLoadFailed);
        };
        target.inner().lock().unwrap().clear_migration();
        self.set_reservation_phase(migration_id, ReservationPhase::Committed)
            .await;
        self.release_reservations(migration_id).await;
        self.wake_one(&plan.victim.model_name);
        self.wake_one(&target_cfg.name);
        Ok(())
    }

    /// Roll back a pre-handoff migration: stop a ready replacement if one
    /// exists, restore the old logical route, release all reservations, and
    /// wake requests waiting on the victim model.
    async fn rollback_migration(
        &self,
        plan: &MigrationPlan,
        migration_id: u64,
        replacement: Option<&InstanceHandle>,
    ) {
        info!(migration_id, victim = %plan.victim.instance_id, "migration rolling back");
        if let Some(replacement) = replacement {
            self.force_remove_instance(&plan.victim.model_name, replacement)
                .await;
        }
        {
            let mut victim = plan.victim.handle.inner().lock().unwrap();
            if victim.state == InstanceState::Draining && victim.migration_id == Some(migration_id)
            {
                victim.mark_ready();
                victim.clear_migration();
            }
        }
        self.release_reservations(migration_id).await;
        self.wake_one(&plan.victim.model_name);
    }

    /// Preserve a successful victim migration when the subsequent target
    /// load fails.  The replacement becomes the normal logical route rather
    /// than being migrated back, preventing failure-induced ping-pong.
    async fn retain_migrated_replacement_after_target_failure(
        &self,
        migration_id: u64,
        model_name: &str,
        replacement: &InstanceHandle,
    ) {
        info!(migration_id, model = %model_name, "migration target load failed; retaining replacement");
        replacement.inner().lock().unwrap().clear_migration();
        self.release_reservations(migration_id).await;
        self.wake_one(model_name);
    }

    /// Plan the victim set that frees enough VRAM for `cfg` to spawn.
    /// Victims are accumulated in class order (idle surplus duplicates →
    /// idle last instances → busy surplus duplicates), least-recently-used
    /// first within a class, re-checking feasibility after each addition.
    /// `Some(plan)` — possibly empty — means spawning works after the plan
    /// is executed; `None` means no allowed eviction gets there.
    async fn plan_make_room(
        &self,
        cfg: &ModelConfig,
        mode: MakeRoomMode,
    ) -> Option<Vec<MakeRoomVictim>> {
        if !mode.is_legacy_eviction() {
            return None;
        }
        let mut excluding: HashSet<String> = HashSet::new();
        let mut plan = Vec::new();
        loop {
            if self.spawn_feasible(cfg, &excluding).await {
                return Some(plan);
            }
            match self.next_make_room_victim(&cfg.name, mode, &excluding) {
                Some(victim) => {
                    debug!(
                        model = %cfg.name,
                        victim_model = %victim.model_name,
                        victim_inst = %victim.handle.id(),
                        drain = victim.drain,
                        "make-room: planned victim"
                    );
                    excluding.insert(victim.handle.id());
                    plan.push(victim);
                }
                None => return None,
            }
        }
    }

    /// Pick the next make-room victim not already in `excluding`, in
    /// victim-class order.  Idle instances (classes 1–2) are only eligible
    /// once idle past their model's `idle_ttl` protection window.  Busy
    /// duplicates (class 3, `drain_surplus` only) are never taken from the
    /// requesting model itself and never leave a model without a serving
    /// instance.
    fn next_make_room_victim(
        &self,
        requesting_model: &str,
        mode: MakeRoomMode,
        excluding: &HashSet<String>,
    ) -> Option<MakeRoomVictim> {
        struct Snap {
            model_name: String,
            handle: InstanceHandle,
            last_active: Instant,
            in_flight: usize,
            ready: bool,
        }
        let mut snaps: Vec<Snap> = Vec::new();
        {
            let instances = self.instances.read().unwrap();
            for (model_name, list) in instances.iter() {
                for h in list {
                    let inst = h.inner().lock().unwrap();
                    if matches!(
                        inst.state,
                        InstanceState::Failed | InstanceState::Draining | InstanceState::Retired
                    ) || inst.migration_id.is_some()
                        || excluding.contains(&inst.id)
                    {
                        continue;
                    }
                    snaps.push(Snap {
                        model_name: model_name.clone(),
                        handle: h.clone(),
                        last_active: inst.last_active,
                        in_flight: inst.in_flight,
                        ready: inst.state == InstanceState::Ready,
                    });
                }
            }
        }
        // Non-retiring instances per model (excluding planned victims) —
        // the "surplus" reference count.
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for s in &snaps {
            *counts.entry(s.model_name.as_str()).or_default() += 1;
        }
        let idle_ttl_of = |model_name: &str| {
            self.model_configs
                .read()
                .unwrap()
                .get(model_name)
                .map(|c| c.idle_ttl)
                .unwrap_or(0)
        };
        // Idle victims (classes 1–2): Ready, no in-flight, past their
        // model's protection window; least-recently-used first.
        let mut idle: Vec<&Snap> = snaps
            .iter()
            .filter(|s| {
                s.ready
                    && s.in_flight == 0
                    && s.last_active.elapsed().as_secs() >= idle_ttl_of(&s.model_name)
            })
            .collect();
        idle.sort_by_key(|s| s.last_active);
        // Class 1: idle surplus duplicates of any model.
        if let Some(s) = idle.iter().find(|s| counts[s.model_name.as_str()] >= 2) {
            return Some(MakeRoomVictim {
                model_name: s.model_name.clone(),
                handle: s.handle.clone(),
                drain: false,
            });
        }
        // Class 2: idle last instances.
        if let Some(s) = idle.first() {
            return Some(MakeRoomVictim {
                model_name: s.model_name.clone(),
                handle: s.handle.clone(),
                drain: false,
            });
        }
        // Class 3: busy surplus duplicates (drain_surplus only) — never
        // from the requesting model itself, never the last serving
        // instance of a model; least-busy first.
        if mode == MakeRoomMode::DrainSurplus
            && let Some(s) = snaps
                .iter()
                .filter(|s| {
                    s.model_name != requesting_model
                        && s.in_flight > 0
                        && counts[s.model_name.as_str()] >= 2
                })
                .min_by_key(|s| s.in_flight)
        {
            return Some(MakeRoomVictim {
                model_name: s.model_name.clone(),
                handle: s.handle.clone(),
                drain: true,
            });
        }
        None
    }

    /// Execute a make-room plan: evict idle victims immediately, drain
    /// busy duplicates with a per-victim bounded wait (`drain_timeout`
    /// seconds).  Victims are re-validated implicitly — `remove_instance`
    /// is a no-op when the instance became busy in the meantime, and a
    /// timed-out drain leaves the instance draining (reaped once idle);
    /// the subsequent spawn attempt simply fails and the caller advances.
    async fn execute_make_room(&self, plan: &[MakeRoomVictim], drain_timeout: u64) {
        for victim in plan {
            if victim.drain {
                info!(
                    model = %victim.model_name,
                    inst = %victim.handle.id(),
                    "make-room: draining busy surplus instance"
                );
                self.drain_instance(&victim.model_name, &victim.handle)
                    .await;
                let deadline = Instant::now() + Duration::from_secs(drain_timeout);
                loop {
                    self.reap_drained(&victim.model_name).await;
                    let gone = !self
                        .instances
                        .read()
                        .unwrap()
                        .get(&victim.model_name)
                        .map(|l| {
                            l.iter()
                                .any(|h| Arc::ptr_eq(h.inner(), victim.handle.inner()))
                        })
                        .unwrap_or(false);
                    if gone {
                        break;
                    }
                    if Instant::now() >= deadline {
                        warn!(
                            model = %victim.model_name,
                            inst = %victim.handle.id(),
                            "make-room: drain timed out — continuing anyway"
                        );
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            } else {
                info!(
                    model = %victim.model_name,
                    inst = %victim.handle.id(),
                    "make-room: evicting idle instance"
                );
                self.remove_instance(&victim.model_name, &victim.handle)
                    .await;
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
        if self
            .find_ready_instance(model_name, cfg.max_concurrent)
            .is_some()
        {
            return Ok(());
        }

        let mut rx = self.ensure_spawn(model_name, &cfg);
        let budget = self.spawn_timeout + SPAWN_SERIAL_WINDOW + Duration::from_secs(30);
        let _ = tokio::time::timeout(budget, rx.changed()).await;

        if self
            .find_ready_instance(model_name, cfg.max_concurrent)
            .is_some()
        {
            Ok(())
        } else {
            Err(AcquireError::Unavailable)
        }
    }

    /// Whether any queue contains a waiter logically routed to `model_name`.
    /// Entries may remain under their original alias candidate key after a
    /// fallback rebind, so checking only `queues[model_name]` is insufficient
    /// for FIFO enforcement.
    fn has_logical_waiters(&self, model_name: &str) -> bool {
        self.queues.read().unwrap().values().any(|queue| {
            queue
                .iter()
                .any(|entry| entry.route.logical_model == model_name)
        })
    }

    /// Fast path of `get_or_spawn`: acquire a slot on the least-loaded
    /// ready instance with spare capacity, but never bypass an older logical
    /// queue entry.
    fn acquire_ready_slot(&self, model_name: &str, max_concurrent: usize) -> Option<SlotGuard> {
        if self.has_logical_waiters(model_name) {
            return None;
        }
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

    /// Subscribe to one in-flight migration for `model_name`, starting it
    /// if necessary.  The detached task owns the transition so a caller
    /// disconnecting while waiting cannot abort shared infrastructure work.
    fn ensure_migration(
        self: &Arc<Self>,
        model_name: &str,
        cfg: &ModelConfig,
    ) -> watch::Receiver<()> {
        if self.is_shutting_down() {
            let (tx, rx) = watch::channel(());
            drop(tx);
            return rx;
        }
        {
            let migrations = self.migrations_in_flight.read().unwrap();
            if let Some(tx) = migrations.get(model_name) {
                return tx.subscribe();
            }
        }
        let (tx, rx) = watch::channel(());
        {
            let mut migrations = self.migrations_in_flight.write().unwrap();
            if let Some(existing) = migrations.get(model_name) {
                return existing.subscribe();
            }
            migrations.insert(model_name.to_owned(), tx);
        }
        self.scheduler_metrics
            .migration_started_total
            .fetch_add(1, Ordering::Relaxed);

        let mgr = Arc::clone(self);
        let target = model_name.to_owned();
        let cfg = cfg.clone();
        tokio::spawn(async move {
            let _entry = MigrationEntryGuard {
                mgr: Arc::clone(&mgr),
                target_model: target.clone(),
            };
            match mgr.run_migration(&cfg).await {
                Ok(()) => {
                    mgr.scheduler_metrics
                        .migration_completed_total
                        .fetch_add(1, Ordering::Relaxed);
                }
                Err(error) => {
                    mgr.scheduler_metrics
                        .migration_failed_total
                        .fetch_add(1, Ordering::Relaxed);
                    debug!(model = %target, ?error, "migration transition finished without admission");
                    if !mgr.has_routable_instance(&target) {
                        mgr.fail_queue(&target, QueueTerminalError::MigrationFailed);
                    }
                }
            }
        });
        rx
    }

    /// Start slow admission work after its logical queue is registered.
    fn start_admission_demand(self: &Arc<Self>, demand: AdmissionDemand) {
        match demand {
            AdmissionDemand::Spawn { model_name, cfg } => {
                self.start_spawn_demand(&model_name, &cfg, None);
            }
            AdmissionDemand::LegacyMakeRoom {
                model_name,
                cfg,
                candidates,
                mode,
                drain_timeout_secs,
            } => self.ensure_legacy_make_room(
                &model_name,
                &cfg,
                &candidates,
                mode,
                drain_timeout_secs,
            ),
            AdmissionDemand::Migration { model_name, cfg } => {
                let _rx = self.ensure_migration(&model_name, &cfg);
            }
        }
    }

    /// Start a normal spawn after its logical queue is registered.
    fn start_spawn_demand(
        self: &Arc<Self>,
        model_name: &str,
        cfg: &ModelConfig,
        candidates: Option<Vec<String>>,
    ) {
        let mgr = Arc::clone(self);
        let model = model_name.to_owned();
        let cfg = cfg.clone();
        tokio::spawn(async move {
            let mut rx = mgr.ensure_spawn(&model, &cfg);
            let _ = rx.changed().await;
            if mgr.has_routable_instance(&model) {
                mgr.wake_one(&model);
            } else if let Some(candidates) = candidates
                && mgr.rebind_queue_to_fallback(&model, &candidates)
            {
                // Fallback rebinding already woke the logical queue.
            } else {
                mgr.fail_queue(&model, QueueTerminalError::ModelRemoved);
            }
        });
    }

    /// Start the legacy destructive make-room path after its logical queue is
    /// already registered.  Unlike migration, this path is intentionally
    /// kept separate for backwards compatibility with old aliases.
    fn ensure_legacy_make_room(
        self: &Arc<Self>,
        model_name: &str,
        cfg: &ModelConfig,
        candidates: &[String],
        mode: MakeRoomMode,
        drain_timeout_secs: u64,
    ) {
        let mgr = Arc::clone(self);
        let model = model_name.to_owned();
        let candidates = candidates.to_vec();
        let cfg = cfg.clone();
        tokio::spawn(async move {
            match mgr.plan_make_room(&cfg, mode).await {
                Some(plan) => mgr.execute_make_room(&plan, drain_timeout_secs).await,
                None => {
                    if !mgr.has_routable_instance(&model) {
                        mgr.fail_queue(&model, QueueTerminalError::ModelRemoved);
                    }
                    return;
                }
            }
            mgr.start_spawn_demand(&model, &cfg, Some(candidates));
        });
    }

    /// Register a logical queue entry.  `allow_empty` is used by the new
    /// admission path so a cold-start request can wait while its detached
    /// spawn or migration demand is running; legacy callers retain the
    /// fail-fast behavior when no instance exists.
    fn register_pending(
        self: &Arc<Self>,
        route: QueueRouteSnapshot,
        max_depth: usize,
        allow_empty: bool,
    ) -> Result<PendingAssignment, EnqueueError> {
        let (assignment, receiver) = oneshot::channel();
        let id = self.next_queue_id.fetch_add(1, Ordering::Relaxed);
        let model_name = route.logical_model.clone();
        debug!(
            model = %model_name,
            candidates = ?route.candidates,
            policy = ?route.policy,
            make_room = ?route.make_room,
            drain_timeout_secs = route.drain_timeout_secs,
            generation = route.generation,
            migration_id = ?route.migration_id,
            "queue route snapshot created"
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        let enqueued_at = Instant::now();
        let queue_wait = *self.queue_wait_timeout_secs.read().unwrap();
        let deadline = (queue_wait > 0).then(|| enqueued_at + Duration::from_secs(queue_wait));
        let expired = Arc::new(AtomicBool::new(false));
        let ticket = QueueTicket {
            manager: Arc::clone(self),
            model_name: model_name.clone(),
            id,
            cancelled: Arc::clone(&cancelled),
            expired: Arc::clone(&expired),
            request_id: route.request_id.clone(),
        };
        let entry = QueueEntry {
            id,
            route,
            enqueued_at,
            deadline,
            assignment,
            cancelled,
        };

        {
            // Hold the instances read lock across the check *and* the queue
            // push.  remove_instance takes the instances write lock to remove
            // the last instance and only then drains the queue — so either
            // our push lands before the drain (and is cleared by it) or after
            // the removal (and the check fails).
            let instances = self.instances.read().unwrap();
            let has_instances = instances
                .get(&model_name)
                .map(|list| {
                    list.iter()
                        .any(|h| h.inner().lock().unwrap().can_wake_waiters())
                })
                .unwrap_or(false);
            if !allow_empty && !has_instances {
                return Err(EnqueueError::NoInstances);
            }

            let mut queues = self.queues.write().unwrap();
            let queue = queues.entry(model_name.clone()).or_default();
            if queue.len() >= max_depth {
                return Err(EnqueueError::QueueFull);
            }
            queue.push_back(entry);
        }
        self.scheduler_metrics
            .queue_enqueued_total
            .fetch_add(1, Ordering::Relaxed);

        // Make the queue entry visible to the load EMA immediately: the
        // autoscaler's scale-up rules are driven by `in_flight + queued`,
        // so a burst parked in the queue must move the EMA now, not at the
        // next release event or the next 30s autoscaler tick.
        self.record_metrics_event(&model_name, 0);

        // Close the lost-wakeup race: a slot may have been released between
        // the capacity check and our queue push, with wake_one finding an
        // empty queue.  Re-signal now that we are parked.
        self.wake_one(&model_name);

        Ok(PendingAssignment {
            receiver,
            deadline,
            _ticket: ticket,
        })
    }

    /// Enqueue the caller through the logical queue and await assignment.
    #[cfg(test)]
    async fn enqueue(
        self: &Arc<Self>,
        route: QueueRouteSnapshot,
        max_depth: usize,
    ) -> Result<SlotGuard, EnqueueError> {
        match self.register_pending(route, max_depth, false)?.wait().await {
            Ok((_, guard)) => Ok(guard),
            Err(AcquireError::NoCapacity) => Err(EnqueueError::QueueFull),
            Err(AcquireError::QueueTimeout) => Err(EnqueueError::Timeout),
            Err(AcquireError::Unavailable | AcquireError::Blocked) => {
                Err(EnqueueError::NoInstances)
            }
        }
    }

    /// Remove a specific queued waiter (used after a queue-wait timeout).
    fn remove_queued(&self, model_name: &str, id: u64) -> bool {
        self.queues
            .write()
            .unwrap()
            .get_mut(model_name)
            .map(|queue| {
                let before = queue.len();
                queue.retain(|entry| entry.id != id);
                queue.len() != before
            })
            .unwrap_or(false)
    }

    /// Complete all queued entries with a terminal scheduler error.
    fn fail_queue(&self, model_name: &str, error: QueueTerminalError) {
        let entries: Vec<QueueEntry> = self
            .queues
            .write()
            .unwrap()
            .get_mut(model_name)
            .map(|queue| queue.drain(..).collect())
            .unwrap_or_default();
        for entry in entries {
            let _ = entry.assignment.send(Err(error));
        }
    }

    /// Complete every queued request during final shutdown.
    fn fail_all_queues(&self) {
        let entries: Vec<QueueEntry> = self
            .queues
            .write()
            .unwrap()
            .values_mut()
            .flat_map(|queue| queue.drain(..))
            .collect();
        for entry in entries {
            let _ = entry.assignment.send(Err(QueueTerminalError::Shutdown));
        }
    }

    /// Rebind queued entries to the first later alias candidate with a
    /// routable instance.  The queue remains stored under its original key
    /// so its cancellation ticket can still remove entries by ID; wake_one
    /// follows each entry's logical route when assigning it.
    fn rebind_queue_to_fallback(&self, source_model: &str, candidates: &[String]) -> bool {
        let Some(source_index) = candidates.iter().position(|name| name == source_model) else {
            return false;
        };
        let Some(target_model) = candidates
            .iter()
            .skip(source_index + 1)
            .find(|name| self.has_routable_instance(name))
            .cloned()
        else {
            return false;
        };
        let Some(target_cfg) = self
            .model_configs
            .read()
            .unwrap()
            .get(&target_model)
            .cloned()
        else {
            return false;
        };
        let generation = self.model_generation(&target_model);
        let migration_id = self.model_migration_id(&target_model);
        let mut queues = self.queues.write().unwrap();
        let Some(queue) = queues.get_mut(source_model) else {
            return false;
        };
        if queue.is_empty() {
            return false;
        }
        for entry in queue.iter_mut() {
            entry.route.logical_model = target_model.clone();
            entry.route.max_concurrent = target_cfg.max_concurrent;
            entry.route.generation = generation;
            entry.route.migration_id = migration_id;
        }
        drop(queues);
        self.wake_one(source_model);
        true
    }

    /// If no instances remain for `model_name`, complete all parked
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
        if none_left {
            // An alias queue may still be serviceable through a later
            // candidate.  Rebind before declaring the logical model removed;
            // direct model queues have no fallback and are completed below.
            let candidates = self
                .queues
                .read()
                .unwrap()
                .get(model_name)
                .and_then(|queue| queue.front())
                .map(|entry| entry.route.candidates.clone());
            let fallback = candidates
                .as_deref()
                .is_some_and(|candidates| self.rebind_queue_to_fallback(model_name, candidates));
            if !fallback {
                self.fail_queue(model_name, QueueTerminalError::ModelRemoved);
            }
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
        if !self.model_configs.read().unwrap().contains_key(model_name) {
            return;
        }

        loop {
            // Pop the head waiter, if any.  The queues lock is released
            // before touching instances — lock order is instances → queues.
            let entry = {
                let mut queues = self.queues.write().unwrap();
                queues.get_mut(model_name).and_then(|q| q.pop_front())
            };
            let Some(entry) = entry else {
                return;
            };

            if entry.cancelled.load(Ordering::Acquire) {
                continue;
            }
            if entry
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
            {
                self.scheduler_metrics
                    .queue_expired_total
                    .fetch_add(1, Ordering::Relaxed);
                let _ = entry.assignment.send(Err(QueueTerminalError::Deadline));
                continue;
            }
            if !self.route_allows_assignment(&entry.route) {
                let mut queues = self.queues.write().unwrap();
                queues
                    .entry(model_name.to_owned())
                    .or_default()
                    .push_front(entry);
                return;
            }

            let logical_model = entry.route.logical_model.clone();
            let max_concurrent = entry.route.max_concurrent;
            debug!(
                model = %logical_model,
                queue_id = entry.id,
                generation = entry.route.generation,
                queued_for_ms = entry.enqueued_at.elapsed().as_millis() as u64,
                "considering queued assignment"
            );
            let guard = self
                .find_ready_instance(&logical_model, max_concurrent)
                .and_then(|h| h.try_acquire(max_concurrent));
            match guard {
                Some(g) => match entry.assignment.send(Ok((logical_model.clone(), g))) {
                    Ok(()) => {
                        self.scheduler_metrics
                            .queue_assigned_total
                            .fetch_add(1, Ordering::Relaxed);
                        // The request transitioned queue → in-flight; keep
                        // the load EMA honest during the queue drain.
                        self.record_metrics_event(&logical_model, 0);
                        return;
                    }
                    // Waiter vanished (cancellation racing the wake) — the
                    // returned guard drops, releasing the slot; try the next
                    // waiter.
                    Err(_) => continue,
                },
                None => {
                    // Capacity disappeared between the release event and
                    // now — re-park at the head and wait for the next one.
                    let mut queues = self.queues.write().unwrap();
                    queues
                        .entry(model_name.to_owned())
                        .or_default()
                        .push_front(entry);
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
                inst.state == InstanceState::Ready
                    && inst.migration_id.is_none()
                    && inst.in_flight < max_concurrent
            })
            .min_by_key(|h| h.inner().lock().unwrap().in_flight)
            .cloned()
    }

    /// Attempt to spawn a normal instance for `model`.  Returns `None` if the
    /// per-model instance cap has been reached.
    async fn try_spawn(&self, model_name: &str, cfg: &ModelConfig) -> Option<InstanceHandle> {
        self.try_spawn_with_options(model_name, cfg, SpawnOptions::normal())
            .await
    }

    /// Spawn an instance with coordinator-supplied placement and lifecycle
    /// metadata.  Migration uses this path with a pre-reserved placement and
    /// temporarily bypasses the ordinary logical instance cap.
    async fn try_spawn_with_options(
        &self,
        model_name: &str,
        cfg: &ModelConfig,
        options: SpawnOptions,
    ) -> Option<InstanceHandle> {
        // Remote endpoints have no local process, port, or GPU placement:
        // provisioning is a single bounded health probe, handled by a
        // dedicated (much cheaper) path.
        if cfg.is_remote(&self.cmd_aliases.read().unwrap()) {
            return self.try_provision_remote(model_name, cfg, options).await;
        }

        // Serialize the critical section of spawn attempts globally —
        // prevents concurrent model loads from thrashing the SSD and
        // interleaving VRAM.  Released after SPAWN_SERIAL_WINDOW of
        // readiness waiting, not after the full spawn timeout.
        let _permit = self.spawn_semaphore.clone().acquire_owned().await.ok()?;

        // Inside the semaphore: check instance cap.  Retiring instances
        // (`Draining`/`Failed` — after admin unload or config change) do not
        // count: they are unroutable and reaped once idle, so counting them
        // would wedge models with `max_instances = 1` behind a long-running
        // request with no replacement.  Resource limits are still enforced
        // by the VRAM accounting in `select_gpus_for_model`, which *does*
        // include retiring instances.
        if options.enforce_instance_cap {
            let instances = self.instances.read().unwrap();
            if let Some(list) = instances.get(model_name) {
                let active = list
                    .iter()
                    .filter(|h| h.inner().lock().unwrap().counts_toward_instance_cap())
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
        let placement = match options.requested_gpu_indices.as_ref() {
            Some(indices) => self.explicit_placement_snapshot(cfg, indices),
            None => self.select_placement_snapshot(cfg, &HashSet::new()).await,
        };
        let gpu_indices: Vec<usize> = placement
            .as_ref()
            .map(|snapshot| snapshot.device_indices.clone())
            .unwrap_or_default();

        // Safety: when a device pool is configured but the full device
        // set can't be satisfied, fail the spawn instead of launching
        // without GPU restriction (competing on GPUs already occupied by
        // existing instances) or with too few devices (a multi-GPU model
        // would not fit in memory).
        if cfg.has_device_pool() && placement.is_none() {
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

        // Commit a short-lived reservation before process startup for a
        // normal spawn.  A migration supplies an already-committed
        // transaction reservation and retains it until the transition
        // decides whether to roll back or commit.
        let reservation_id = match (options.reservation_id, placement) {
            (Some(transition_id), Some(_)) => Some(transition_id),
            (Some(_), None) => {
                warn!(model = %model_name, "migration placement is no longer valid");
                self.ports.lock().await.free(port);
                return None;
            }
            (None, Some(placement)) if !placement.device_indices.is_empty() => {
                let Some(transition_id) = self.allocate_transition_id().await else {
                    warn!(model = %model_name, "reservation coordinator unavailable");
                    self.ports.lock().await.free(port);
                    return None;
                };
                let cleanup = ReservationCleanup {
                    token: transition_id,
                    expires_at: Some(
                        Instant::now()
                            + self.spawn_timeout
                            + SPAWN_SERIAL_WINDOW
                            + Duration::from_secs(30),
                    ),
                };
                match self
                    .reserve_placement(ReservationRequest {
                        transition_id,
                        purpose: ReservationPurpose::NormalSpawn,
                        model: model_name.to_owned(),
                        pci_slots: placement.pci_slots,
                        vram_per_slot: cfg.vram * 1024 * 1024,
                        phase: ReservationPhase::Loading,
                        cleanup,
                    })
                    .await
                {
                    Ok(()) => Some(transition_id),
                    Err(error) => {
                        debug!(
                            model = %model_name,
                            transition_id,
                            ?error,
                            "spawn placement reservation unavailable"
                        );
                        self.ports.lock().await.free(port);
                        return None;
                    }
                }
            }
            _ => None,
        };

        // Select the backend implementation based on the program basename.
        // sd.cpp models use `StableDiffusionCppBackend` (no --device flag,
        // only the visibility env var); containers use `ContainerBackend`
        // (GPU passthrough via --device flags, no env-var / selector
        // injection); everything else uses the standard llama.cpp backend.
        let prog_basename = prog.rsplit('/').next().unwrap_or(prog);
        let is_container = cfg.runner(&self.cmd_aliases.read().unwrap()) == Runner::Container;
        let model_backend: Box<dyn Backend> = if is_container {
            Box::new(ContainerBackend)
        } else if prog_basename.starts_with("sd-server") {
            Box::new(StableDiffusionCppBackend::new(cfg.device_kind()))
        } else {
            Box::new(LlamaCppBackend::new(cfg.device_kind()))
        };
        let mut args: Vec<String> = parts[1..].iter().map(|s| s.to_string()).collect();
        let mut container_name: Option<String> = None;
        if is_container {
            // Container: inject --name <unique> and --rm after "podman run".
            // GPU arg/env injection is skipped — the container manages its
            // own device selection via --device /dev/dri etc.
            let cname = format!("llm-orch-{model_name}-{port}");
            inject_container_args(&mut args, &cname);
            container_name = Some(cname);
        } else {
            args.extend(model_backend.gpu_args(&gpu_indices));
            // CPU-only spawn: enforce no GPU offload for llama.cpp models.
            // No-op for non-llama programs (basename check inside).
            if gpu_indices.is_empty() {
                LlamaCppBackend::enforce_cpu_offload(prog, &mut args);
            }
            // Speculative decoding: pin the draft model to the same device(s)
            // the instance is placed on.  No-op for non-llama programs.
            LlamaCppBackend::inject_spec_draft_device(
                prog,
                &mut args,
                cfg.device_kind(),
                &gpu_indices,
            );
        }
        let envs = model_backend.gpu_env(&gpu_indices);

        // Spawn.
        let (child, output) = match spawn_process(prog, &args, &envs, model_name, port).await {
            Ok(pair) => pair,
            Err(e) => {
                warn!(model = %model_name, error = %e, "spawn failed");
                if !options.retain_reservation
                    && let Some(transition_id) = reservation_id
                {
                    self.release_reservations(transition_id).await;
                }
                self.ports.lock().await.free(port);
                return None;
            }
        };

        let mut inst = Instance::new_with_purpose(
            model_name,
            gpu_indices.clone(),
            port,
            Some(self.release_tx.clone()),
            options.load_purpose,
        );
        inst.migration_id = options.migration_id;
        inst.config_fingerprint = self.spawn_fingerprint(cfg);
        // Guarantee a unique ID even when the base `model@gpus` collides
        // (multiple CPU instances of the same model).
        inst.id = format!(
            "{}#{}",
            inst.id,
            self.instance_seq.fetch_add(1, Ordering::Relaxed)
        );
        inst.child = Some(child);
        inst.container_name = container_name;
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
        if !options.retain_reservation
            && let Some(transition_id) = reservation_id
        {
            self.release_reservations(transition_id).await;
        }

        // Keep-alive: acquire one reference per occupied GPU.  Released
        // exactly once in unregister_instance, which every instance
        // removal funnels through — refcounting makes concurrent
        // spawn/remove interleavings safe.
        {
            let keepalive = self.keepalive.read().unwrap().clone();
            if let Some(ref ka) = keepalive {
                // Keep-alive applies to AMD devices exposed through either
                // Vulkan or ROCm; NVIDIA GPUs are kept awake via persistence
                // mode instead.
                match cfg.device_kind() {
                    DeviceKind::Vulkan => {
                        let slots = self.vulkan_slots.read().unwrap();
                        for idx in &gpu_indices {
                            if let Some(slot) = slots.get(idx) {
                                ka.acquire(slot);
                            }
                        }
                    }
                    DeviceKind::Rocm => {
                        let slots = self.rocm_slots.read().unwrap();
                        for idx in &gpu_indices {
                            if let Some(slot) = slots.get(idx) {
                                ka.acquire(slot);
                            }
                        }
                    }
                    DeviceKind::Cuda => {}
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
        let model_timeout = Duration::from_secs(cfg.spawn_timeout_secs.unwrap_or(120));
        let phase1 =
            poll_readiness(&handle, &self.client, &*model_backend, SPAWN_SERIAL_WINDOW).await;
        // Release the global spawn lock before the long tail so a slow
        // model can't head-of-line block spawns of every other model.
        drop(_permit);
        // Phase 2 (unserialized): wait out the remaining timeout.
        let outcome = match phase1 {
            ReadyOutcome::TimedOut => {
                let remaining = model_timeout.saturating_sub(SPAWN_SERIAL_WINDOW);
                poll_readiness(&handle, &self.client, &*model_backend, remaining).await
            }
            other => other,
        };
        if outcome != ReadyOutcome::Ready {
            // Distinguish a dead process (pre-output crash — counts toward
            // the model block limit) from a live-but-slow load (no count).
            let child_exited = outcome == ReadyOutcome::ChildExited;
            warn!(model = %model_name, port = port, exited = child_exited, "instance did not become ready — shutting down");
            self.dump_recent_output(&handle);
            let (mut child_to_kill, cname) = {
                let mut inst_lock = handle.inner().lock().unwrap();
                inst_lock.mark_failed();
                (inst_lock.child.take(), inst_lock.container_name.clone())
            };
            if let Some(ref cname) = cname {
                shutdown_container(cname, Duration::from_secs(5)).await;
            }
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
            let (mut child, cname) = {
                let mut inst = handle.inner().lock().unwrap();
                inst.mark_failed();
                (inst.child.take(), inst.container_name.clone())
            };
            if let Some(ref cname) = cname {
                shutdown_container(cname, Duration::from_secs(5)).await;
            }
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
        tokio::spawn(monitor_instance_exit(handle.clone(), self.crash_tx.clone()));

        let instance_id = {
            let inst = handle.inner().lock().unwrap();
            inst.id.clone()
        };
        info!(model = %model_name, inst = %instance_id, port = port, "spawn succeeded");

        Some(handle)
    }

    /// Provision a remote http(s) endpoint: verify it answers
    /// `GET {base}/health` within [`REMOTE_PROBE_TIMEOUT`], then register
    /// the instance as Ready.  No child process, port, GPU reservation,
    /// keep-alive, or crash monitor is involved.
    ///
    /// A failed probe simply leaves the model without an instance — the
    /// request queues (when `queue_depth > 0`) or fails `Unavailable`,
    /// and a later request re-probes.  Probes do not count toward the
    /// pre-output crash block: the endpoint is not ours to restart.
    async fn try_provision_remote(
        &self,
        model_name: &str,
        cfg: &ModelConfig,
        options: SpawnOptions,
    ) -> Option<InstanceHandle> {
        // Serialize against concurrent provisioning of the same model so
        // the instance cap below stays race-free (same rationale as the
        // process spawn path, without the SSD-thrashing concern).
        let _permit = self.spawn_semaphore.clone().acquire_owned().await.ok()?;

        if options.enforce_instance_cap {
            let instances = self.instances.read().unwrap();
            if let Some(list) = instances.get(model_name) {
                let active = list
                    .iter()
                    .filter(|h| h.inner().lock().unwrap().counts_toward_instance_cap())
                    .count();
                if active >= cfg.max_instances {
                    return None;
                }
            }
        }

        // `resolve_cmd` skips the `{port}` substitution for remote models
        // (see `resolve_model_cmd`), so the marker port 0 is safe.
        let base_url = self.resolve_cmd(cfg, 0);
        let health_url = format!("{}/health", base_url.trim_end_matches('/'));

        let healthy =
            match tokio::time::timeout(REMOTE_PROBE_TIMEOUT, self.client.get(&health_url).send())
                .await
            {
                Ok(Ok(resp)) => resp.status().is_success(),
                other => {
                    warn!(
                        model = %model_name,
                        url = %health_url,
                        outcome = ?other,
                        "remote health probe failed or timed out"
                    );
                    false
                }
            };
        if !healthy {
            return None;
        }

        let log_url = base_url.clone();
        let mut inst = Instance::new_remote(
            model_name,
            base_url,
            cfg.remote_auth.clone(),
            Some(self.release_tx.clone()),
            options.load_purpose,
        );
        inst.migration_id = options.migration_id;
        inst.config_fingerprint = self.spawn_fingerprint(cfg);
        inst.id = format!(
            "{}#{}",
            inst.id,
            self.instance_seq.fetch_add(1, Ordering::Relaxed)
        );
        inst.mark_ready();
        let handle = InstanceHandle::new(inst);

        {
            let mut instances = self.instances.write().unwrap();
            instances
                .entry(model_name.to_owned())
                .or_default()
                .push(handle.clone());
        }

        self.touch_activity(model_name);
        // A successful probe resets any stale crash-counter state.
        self.crash_counts.lock().unwrap().remove(model_name);

        let instance_id = handle.id();
        info!(model = %model_name, inst = %instance_id, url = %log_url, "remote instance provisioned");

        Some(handle)
    }

    /// Spawn-config fingerprint of `cfg` resolved against the *current*
    /// cmd aliases (see `fingerprint_with_aliases`).
    fn spawn_fingerprint(&self, cfg: &ModelConfig) -> u64 {
        fingerprint_with_aliases(&self.cmd_aliases.read().unwrap(), cfg)
    }

    /// Replace the resolved `key_files` contents (hot-reload of the key
    /// files themselves).
    pub fn set_key_files(&self, key_files: HashMap<String, String>) {
        *self.key_files.write().unwrap() = key_files;
    }

    /// Re-read `key_files` from `config` (name → path mapping from the
    /// config, contents from disk).  On read failure the previous values
    /// are kept — a transient I/O error must not break auth for working
    /// models.
    pub fn refresh_key_files(&self, config: &crate::config::Config) {
        match load_key_files(&config.key_files) {
            Ok(keys) => {
                self.set_key_files(keys);
                debug!("key_files refreshed");
            }
            Err(e) => warn!("key_files reload failed (keeping old): {e}"),
        }
    }

    /// Resolve the `remote_auth` template of a remote instance against the
    /// current `key_files` contents.  Returns `None` for local instances
    /// or models without `remote_auth`.
    pub fn resolve_auth(&self, handle: &InstanceHandle) -> Option<String> {
        let template = handle.auth_template()?;
        let keys = self.key_files.read().unwrap();
        Some(resolve_remote_auth(&keys, &template))
    }

    /// Resolve `cmd_aliases`, `{port}`, `{context_length}`, `{max_concurrent}`,
    /// and `{name}` in the model's command string.  `{name}` substitutes
    /// the model's config name, enabling generic cmd templates such as
    /// `audiocpp_server --config /etc/llm-orch/audio/{name}.json …`.
    fn resolve_cmd(&self, cfg: &ModelConfig, port: u16) -> String {
        crate::config::resolve_model_cmd(&self.cmd_aliases.read().unwrap(), cfg, port)
    }

    /// Pick the devices for a new instance from the model's device pool
    /// (`vulkan_devices`, `cuda_devices`, or `rocm_devices`, per the
    /// model's [`DeviceKind`]).  For a flat pool this returns up to
    /// `model_cfg.gpus` distinct devices; for explicit placements it
    /// returns exactly one configured alternative.  Empty when the pool
    /// can't satisfy the request — the caller decides between CPU
    /// fallback and spawn refusal.
    ///
    /// Per-GPU accounting is keyed by PCI slot so Vulkan, CUDA, and ROCm
    /// indices can never alias each other.
    pub(crate) async fn select_gpus_for_model(&self, model_cfg: &ModelConfig) -> Vec<usize> {
        self.select_gpus_inner(model_cfg, &HashSet::new()).await
    }

    /// GPU selection with a set of instance IDs to ignore (planned
    /// make-room victims) — their VRAM and GPU occupancy are treated as
    /// already freed.  Used by `plan_make_room` to simulate evictions.
    async fn select_gpus_inner(
        &self,
        model_cfg: &ModelConfig,
        excluding: &HashSet<String>,
    ) -> Vec<usize> {
        self.select_placement_snapshot(model_cfg, excluding)
            .await
            .map(|snapshot| snapshot.device_indices)
            .unwrap_or_default()
    }

    /// Translate an instance's backend indices through its model namespace.
    /// Returning `None` for an incomplete mapping makes that instance
    /// ineligible for non-interrupting migration rather than guessing at a
    /// physical destination.
    fn pci_slots_for_instance(&self, model_name: &str, indices: &[usize]) -> Option<Vec<String>> {
        let kind = self
            .model_configs
            .read()
            .unwrap()
            .get(model_name)
            .map(|cfg| cfg.device_kind())
            .unwrap_or(DeviceKind::Vulkan);
        let slots = match kind {
            DeviceKind::Cuda => self.cuda_slots.read().unwrap().clone(),
            DeviceKind::Rocm => self.rocm_slots.read().unwrap().clone(),
            DeviceKind::Vulkan => self.vulkan_slots.read().unwrap().clone(),
        };
        indices
            .iter()
            .map(|index| slots.get(index).cloned())
            .collect()
    }

    /// Build a placement snapshot for coordinator-supplied device indices.
    /// The migration planner has already performed the VRAM proof; this
    /// method validates that the indices still belong to the model's pool
    /// and still have PCI mappings before startup begins.
    fn explicit_placement_snapshot(
        &self,
        model_cfg: &ModelConfig,
        indices: &[usize],
    ) -> Option<PlacementSnapshot> {
        let pool = match model_cfg.device_kind() {
            DeviceKind::Cuda => &model_cfg.cuda_devices,
            DeviceKind::Rocm => &model_cfg.rocm_devices,
            DeviceKind::Vulkan => &model_cfg.vulkan_devices,
        };
        // Flat pool: any `gpus`-sized subset.  Explicit placements: the
        // indices must match one alternative exactly (order included — the
        // migration planner hands back the selector's own emission order).
        let valid = match pool {
            DevicePool::Pool(devices) => {
                indices.len() == model_cfg.gpus && indices.iter().all(|i| devices.contains(i))
            }
            DevicePool::Placements(placements) => placements.iter().any(|p| p == indices),
        };
        if !valid {
            return None;
        }
        let pci_slots = self.pci_slots_for_config(model_cfg, indices)?;
        Some(PlacementSnapshot {
            device_indices: indices.to_vec(),
            pci_slots,
            projected_final_vram: HashMap::new(),
            projected_peak_vram: HashMap::new(),
        })
    }

    /// Translate backend indices through a model config's device namespace.
    fn pci_slots_for_config(
        &self,
        model_cfg: &ModelConfig,
        indices: &[usize],
    ) -> Option<Vec<String>> {
        let slots = match model_cfg.device_kind() {
            DeviceKind::Cuda => self.cuda_slots.read().unwrap().clone(),
            DeviceKind::Rocm => self.rocm_slots.read().unwrap().clone(),
            DeviceKind::Vulkan => self.vulkan_slots.read().unwrap().clone(),
        };
        indices
            .iter()
            .map(|index| slots.get(index).cloned())
            .collect()
    }

    /// Return a concrete deterministic placement plus projected final and
    /// peak per-PCI-slot usage.
    async fn select_placement_snapshot(
        &self,
        model_cfg: &ModelConfig,
        excluding: &HashSet<String>,
    ) -> Option<PlacementSnapshot> {
        self.select_placement_snapshot_with_blocked_and_extra(
            model_cfg,
            excluding,
            &HashSet::new(),
            &HashMap::new(),
        )
        .await
    }

    /// Return a placement while treating `blocked_slots` as unavailable.
    /// Migration uses this to prove that a victim replacement is on a
    /// different physical GPU set from the old backend.
    async fn select_placement_snapshot_with_blocked(
        &self,
        model_cfg: &ModelConfig,
        excluding: &HashSet<String>,
        blocked_slots: &HashSet<String>,
    ) -> Option<PlacementSnapshot> {
        self.select_placement_snapshot_with_blocked_and_extra(
            model_cfg,
            excluding,
            blocked_slots,
            &HashMap::new(),
        )
        .await
    }

    /// Return a placement with virtual per-slot VRAM added to the projected
    /// current usage.  The migration dry-run uses this for a replacement
    /// backend that has not been registered as an instance yet.
    async fn select_placement_snapshot_with_blocked_and_extra(
        &self,
        model_cfg: &ModelConfig,
        excluding: &HashSet<String>,
        blocked_slots: &HashSet<String>,
        extra_vram: &HashMap<String, u64>,
    ) -> Option<PlacementSnapshot> {
        let kind = model_cfg.device_kind();
        let pool = match kind {
            DeviceKind::Cuda => &model_cfg.cuda_devices,
            DeviceKind::Rocm => &model_cfg.rocm_devices,
            DeviceKind::Vulkan => &model_cfg.vulkan_devices,
        };
        // Clone the (tiny) device maps — std RwLock guards are !Send and
        // must not live across the gpu_snapshot await below.
        let vulkan_slots = self.vulkan_slots.read().unwrap().clone();
        let cuda_slots = self.cuda_slots.read().unwrap().clone();
        let rocm_slots = self.rocm_slots.read().unwrap().clone();
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
            DeviceKind::Rocm => (&rocm_slots, &vram_limits),
            DeviceKind::Vulkan => (&vulkan_slots, &vram_limits),
        };
        if pool.is_empty() || slots.is_empty() {
            debug!(model = %model_cfg.name, kind = ?kind, "no devices configured");
            return None;
        }

        // Translate an instance's device index to a PCI slot using its
        // own model's namespace map.  Instances of models no longer in
        // the config are treated as Vulkan (pre-removal semantics).
        let slot_for = |model: &str, idx: usize| -> Option<String> {
            match model_kinds
                .get(model)
                .copied()
                .unwrap_or(DeviceKind::Vulkan)
            {
                DeviceKind::Cuda => cuda_slots.get(&idx).cloned(),
                DeviceKind::Rocm => rocm_slots.get(&idx).cloned(),
                DeviceKind::Vulkan => vulkan_slots.get(&idx).cloned(),
            }
        };

        let gpus = self.gpu_snapshot.read().await.clone();
        let reservation_used = self.reservation_usage(None).await;
        debug!(
            model = %model_cfg.name,
            vram_mb = model_cfg.vram,
            kind = ?kind,
            pool = ?pool,
            gpu_count = gpus.len(),
            "selecting GPU"
        );

        let occupied: HashSet<String> = {
            let instances = self.instances.read().unwrap();
            if let Some(list) = instances.get(&model_cfg.name) {
                list.iter()
                    .filter(|h| !excluding.contains(&h.inner().lock().unwrap().id))
                    .flat_map(|h| {
                        let inst = h.inner().lock().unwrap();
                        inst.gpu_indices.clone()
                    })
                    .filter_map(|idx| slot_for(&model_cfg.name, idx))
                    .collect()
            } else {
                HashSet::new()
            }
        };

        let vram_used: HashMap<String, u64> = {
            let instances = self.instances.read().unwrap();
            let model_configs = self.model_configs.read().unwrap();
            let mut used = reservation_used;
            for (pci_slot, vram) in extra_vram {
                *used.entry(pci_slot.clone()).or_default() += *vram;
            }
            // Reservations cover in-flight loads that have not yet become
            // registered instances.  Registered Loading/Ready/Draining
            // instances are added below using the normal physical-accounting
            // path.
            for (model_name, list) in instances.iter() {
                let model_vram = model_configs
                    .get(model_name)
                    .map(|c| c.vram * 1024 * 1024)
                    .unwrap_or(0);
                for handle in list {
                    let inst = handle.inner().lock().unwrap();
                    if excluding.contains(&inst.id) {
                        continue;
                    }
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
        for idx in pool.iter_indices() {
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
            if blocked_slots.contains(pci_slot) {
                debug!(model = %model_cfg.name, device = idx, "skipping — blocked by migration source");
                continue;
            }

            let gpu = gpus.iter().find(|g| g.pci_slot == pci_slot);
            // Capacity rule: with metrics, the configured value
            // (Vulkan/ROCm vram_limit_mb or CUDA vram_mb) caps the
            // reported total.  Without metrics, only CUDA devices may
            // fall back to their static vram_mb — NVIDIA exposes no sysfs
            // VRAM, so a missing nvidia-smi must not make the device
            // unusable.
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
        let instance_counts: HashMap<String, usize> = {
            let instances = self.instances.read().unwrap();
            let mut counts = HashMap::new();
            for (model_name, list) in instances.iter() {
                for handle in list {
                    let inst = handle.inner().lock().unwrap();
                    if excluding.contains(&inst.id) {
                        continue;
                    }
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
        // order into GGML_VK_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES /
        // HIP_VISIBLE_DEVICES — kept stable because device order has
        // semantics in llama.cpp.
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
        let count_of = |idx: &usize| {
            slots
                .get(idx)
                .and_then(|slot| instance_counts.get(slot))
                .copied()
                .unwrap_or(0)
        };
        let candidate_free: HashMap<usize, u64> = candidates.iter().copied().collect();
        let chosen: Vec<usize> = match pool.as_placements() {
            // Explicit placements: the instance must land on exactly one
            // alternative, every device of which is feasible (not occupied,
            // not blocked, enough free VRAM).  Among feasible alternatives
            // pick the best with the same spirit as the pool packing:
            // least-loaded GPUs first, then most free VRAM, then lowest
            // indices.  Deterministic.
            Some(placements) => {
                let mut best: Option<(&Vec<usize>, usize, u64, usize)> = None; // (devices, max count, min free, idx sum)
                for placement in placements {
                    if placement.is_empty() {
                        continue; // rejected by config validation
                    }
                    if placement
                        .iter()
                        .any(|idx| !candidate_free.contains_key(idx))
                    {
                        debug!(
                            model = %model_cfg.name,
                            placement = ?placement,
                            "placement not fully feasible — skipping"
                        );
                        continue;
                    }
                    let max_count = placement.iter().map(count_of).max().unwrap_or(0);
                    let min_free = placement
                        .iter()
                        .map(|idx| candidate_free[idx])
                        .min()
                        .unwrap_or(0);
                    let idx_sum: usize = placement.iter().sum();
                    let better = match best {
                        None => true,
                        Some((_, b_max_count, b_min_free, b_idx_sum)) => {
                            max_count < b_max_count
                                || (max_count == b_max_count && min_free > b_min_free)
                                || (max_count == b_max_count
                                    && min_free == b_min_free
                                    && idx_sum < b_idx_sum)
                        }
                    };
                    if better {
                        best = Some((placement, max_count, min_free, idx_sum));
                    }
                }
                match best {
                    Some((placement, _, _, _)) => placement.clone(),
                    None => {
                        debug!(
                            model = %model_cfg.name,
                            placements = placements.len(),
                            "no placement alternative fully feasible"
                        );
                        return None;
                    }
                }
            }
            // Legacy flat pool: any `gpus`-sized subset, packed greedily.
            None => {
                if candidates.len() < needed {
                    debug!(
                        model = %model_cfg.name,
                        needed,
                        available = candidates.len(),
                        "not enough GPU candidates"
                    );
                    return None;
                }
                candidates
                    .iter()
                    .take(needed)
                    .map(|(idx, _)| *idx)
                    .collect()
            }
        };
        let pci_slots: Vec<String> = chosen
            .iter()
            .filter_map(|idx| slots.get(idx).cloned())
            .collect();
        if pci_slots.len() != chosen.len() {
            return None;
        }

        let mut projected_final_vram = vram_used;
        for pci_slot in &pci_slots {
            *projected_final_vram.entry(pci_slot.clone()).or_default() += model_vram_bytes;
        }
        let projected_peak_vram = projected_final_vram.clone();
        debug!(model = %model_cfg.name, kind = ?kind, devices = ?chosen, "selected GPUs");
        Some(PlacementSnapshot {
            device_indices: chosen,
            pci_slots,
            projected_final_vram,
            projected_peak_vram,
        })
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
    /// A busy instance is marked `Draining` so no new requests are routed to
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
        let (gpu_indices, cname) = {
            let inst = handle.inner().lock().unwrap();
            (inst.gpu_indices.clone(), inst.container_name.clone())
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
                        if !inst.is_draining() {
                            debug!(
                                inst = %inst.id,
                                in_flight = inst.in_flight,
                                "instance busy — marked draining"
                            );
                            inst.mark_draining(None);
                        }
                        return false;
                    }
                    RemovalMode::Force => {}
                }
            }
            inst.mark_failed();
            inst.child.take()
        };
        // Container shutdown: stop the container first, then reap the
        // podman client process.  For native subprocesses, SIGTERM
        // on the child PID is sufficient.
        if let Some(ref cname) = cname {
            shutdown_container(cname, Duration::from_secs(5)).await;
        }
        if let Some(ref mut c) = child {
            shutdown_child(c, Duration::from_secs(5)).await;
        }

        self.unregister_instance(model_name, handle, &gpu_indices)
            .await;
        self.drain_queue_if_no_instances(model_name);
        true
    }

    /// Finish removing draining instances whose requests have completed.
    ///
    /// Draining instances keep their child alive until their in-flight count
    /// reaches zero.  Called on every slot release (and periodically from the
    /// autoscaler) for `model_name`.
    pub async fn reap_drained(&self, model_name: &str) {
        let drained: Vec<InstanceHandle> = {
            let instances = self.instances.read().unwrap();
            instances
                .get(model_name)
                .map(|list| {
                    list.iter()
                        .filter(|h| {
                            let inst = h.inner().lock().unwrap();
                            inst.state == InstanceState::Draining
                                && inst.migration_id.is_none()
                                && inst.in_flight == 0
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
    /// process and marked the instance non-routable.
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

        let (port, is_remote) = {
            let inst = handle.inner().lock().unwrap();
            (inst.port, inst.is_remote())
        };
        // Remote instances never took a port from the allocator.
        if !is_remote {
            self.ports.lock().await.free(port);
        }

        // Keep-alive: release this instance's per-GPU reference (acquired
        // at spawn registration).  The task stops when the last instance
        // leaves the GPU — refcounting makes this safe against concurrent
        // spawns landing on the same GPU.
        let keepalive = self.keepalive.read().unwrap().clone();
        if let Some(ref ka) = keepalive {
            // Paired with the acquire at spawn: keep-alive applies to
            // Vulkan- and ROCm-kind AMD models.  A model removed from the
            // config falls back to Vulkan translation, matching the
            // pre-CUDA semantics.
            let kind = self
                .model_configs
                .read()
                .unwrap()
                .get(model_name)
                .map(|c| c.device_kind())
                .unwrap_or(DeviceKind::Vulkan);
            match kind {
                DeviceKind::Vulkan => {
                    let slots = self.vulkan_slots.read().unwrap();
                    for idx in gpu_indices {
                        if let Some(slot) = slots.get(idx) {
                            ka.release(slot);
                        }
                    }
                }
                DeviceKind::Rocm => {
                    let slots = self.rocm_slots.read().unwrap();
                    for idx in gpu_indices {
                        if let Some(slot) = slots.get(idx) {
                            ka.release(slot);
                        }
                    }
                }
                DeviceKind::Cuda => {}
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
        let (model_name, was_ready, gpu_indices, migration_id, cname) = {
            let mut inst = handle.inner().lock().unwrap();
            if matches!(inst.state, InstanceState::Failed | InstanceState::Retired) {
                // Already handled by a managed removal path (admin unload,
                // TTL eviction, health-timeout cleanup, shutdown).
                return;
            }
            let was_ready = matches!(inst.state, InstanceState::Ready | InstanceState::Draining);
            let migration_id = inst.migration_id;
            let cname = inst.container_name.clone();
            inst.mark_failed();
            // The monitor already reaped the process via try_wait.
            let _ = inst.child.take();
            (
                inst.model_name.clone(),
                was_ready,
                inst.gpu_indices.clone(),
                migration_id,
                cname,
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

        // Ensure the container is stopped even if the podman client
        // exited (container may still be running).
        if let Some(ref cname) = cname {
            shutdown_container(cname, Duration::from_secs(5)).await;
        }

        self.unregister_instance(&model_name, &handle, &gpu_indices)
            .await;
        self.drain_queue_if_no_instances(&model_name);

        if let Some(migration_id) = migration_id {
            // A replacement crash cannot leave its transition reservation
            // occupying the destination forever.  The migration coordinator
            // will observe the failed instance and decide whether to roll
            // back the logical route.
            self.release_reservations(migration_id).await;
        }

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
                .map(|list| list.to_vec())
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
        self.begin_shutdown();
        self.fail_all_queues();
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
        self.release_all_reservations().await;
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
            self.release_model_reservations(model).await;
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
                                !matches!(
                                    inst.state,
                                    InstanceState::Failed | InstanceState::Retired
                                ) && inst.migration_id.is_none()
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
                    inst.mark_draining(None);
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
        self.config_epoch.fetch_add(1, Ordering::AcqRel);
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
        *self.rocm_slots.write().unwrap() = config
            .devices
            .as_ref()
            .map(|d| d.rocm.clone())
            .unwrap_or_default();
        *self.drain_idle_timeout_secs.write().unwrap() = config.server.drain_idle_timeout_secs;
        *self.queue_wait_timeout_secs.write().unwrap() = config.server.queue_wait_timeout_secs;
        *self.scale_up_value.write().unwrap() = config.server.scale_up_value;
        *self.scale_down_value.write().unwrap() = config.server.scale_down_value;
        *self.migration_timeout_secs.write().unwrap() = config.server.migration_timeout_secs;
        // Re-read key file contents (paths may have changed; contents may
        // have changed on disk without a config edit).  Read failures keep
        // the previous values (see refresh_key_files).
        self.refresh_key_files(config);
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
            // Re-acquire keep-alive for GPUs with running instances — the
            // fresh manager starts with no tasks and zero refcounts, so
            // acquire once per instance-GPU pair to rebuild the counts.
            let keepalive = self.keepalive.read().unwrap().clone();
            if let Some(ref ka) = keepalive {
                let in_use_slots: Vec<String> = {
                    let instances = self.instances.read().unwrap();
                    let vulkan_slots = self.vulkan_slots.read().unwrap();
                    let rocm_slots = self.rocm_slots.read().unwrap();
                    let model_configs = self.model_configs.read().unwrap();
                    let mut slots_in_use = Vec::new();
                    // Keep-alive applies to AMD instances exposed via
                    // Vulkan or ROCm; CUDA instances never acquired a
                    // reference at spawn.
                    for (name, list) in instances.iter() {
                        let kind = model_configs
                            .get(name)
                            .map(|c| c.device_kind())
                            .unwrap_or(DeviceKind::Vulkan);
                        let slots = match kind {
                            DeviceKind::Vulkan => &*vulkan_slots,
                            DeviceKind::Rocm => &*rocm_slots,
                            DeviceKind::Cuda => continue,
                        };
                        for handle in list {
                            let indices = handle.inner().lock().unwrap().gpu_indices.clone();
                            for idx in indices {
                                if let Some(slot) = slots.get(&idx) {
                                    slots_in_use.push(slot.clone());
                                }
                            }
                        }
                    }
                    slots_in_use
                };
                for slot in in_use_slots {
                    ka.acquire(&slot);
                }
            }
            info!("keep-alive manager rebuilt after config reload");
        }

        // A valid reload may make a previously queued logical route
        // assignable (for example by changing max_concurrent or restoring a
        // model).  Re-check each surviving queue after the atomic swap.
        for model in &config.models {
            self.wake_one(&model.name);
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
            .map(|list| {
                list.iter()
                    .map(|h| h.inner().lock().unwrap().in_flight)
                    .sum()
            })
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
    pub fn record_completion(&self, model_name: &str, record: CompletionRecord) {
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
    pub fn recent_completions_snapshot(&self) -> HashMap<String, Vec<CompletionRecord>> {
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
        let drain_idle_timeout = *self.drain_idle_timeout_secs.read().unwrap();
        let reserved_models: HashSet<String> = self
            .reservation_snapshot()
            .await
            .into_iter()
            .map(|reservation| reservation.model)
            .collect();
        let active_migrations: HashSet<String> = self
            .migrations_in_flight
            .read()
            .unwrap()
            .keys()
            .cloned()
            .collect();

        // Snapshot configs for the iteration (cheap clone, small structs).
        let configs: Vec<(String, ModelConfig)> = self
            .model_configs
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        for (model_name, cfg) in &configs {
            let instance_in_migration = self
                .instances
                .read()
                .unwrap()
                .get(model_name)
                .map(|instances| {
                    instances
                        .iter()
                        .any(|handle| handle.inner().lock().unwrap().migration_id.is_some())
                })
                .unwrap_or(false);
            if reserved_models.contains(model_name)
                || active_migrations.contains(model_name)
                || instance_in_migration
            {
                debug!(model = %model_name, "autoscale: skipping migration participant or reserved model");
                continue;
            }

            // Safety net: finish removals of draining instances whose
            // in-flight requests have completed.
            self.reap_drained(model_name).await;

            // Safety net: reap instances stranded in `Loading` far beyond
            // the spawn window (e.g. after a panicked spawn task).  Loading
            // instances can't hold slots, so `last_active` ≈ registration
            // time; a legitimately loading instance never exceeds
            // `spawn_timeout` plus the serialized window by a wide margin.
            // Use the same per-model override as the spawn path so slow
            // loaders (e.g. vLLM with CUDA graph capture, minutes) are not
            // reaped mid-load by the smaller global default.
            let model_spawn_timeout = Duration::from_secs(cfg.spawn_timeout_secs.unwrap_or(120));
            let stranded: Vec<InstanceHandle> = {
                let instances = self.instances.read().unwrap();
                instances
                    .get(model_name)
                    .map(|list| {
                        list.iter()
                            .filter(|h| {
                                let inst = h.inner().lock().unwrap();
                                inst.state == InstanceState::Loading
                                    && inst.migration_id.is_none()
                                    && inst.last_active.elapsed() > model_spawn_timeout * 2
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

            // Remote instances are never idle-evicted: the endpoint is not
            // ours to unload, and the "instance" is only a passthrough
            // registration.  (Autoscaling is rejected for remote models
            // at config load, so only the global idle-reap logic below
            // needs guarding.)
            if cfg.is_remote(&self.cmd_aliases.read().unwrap()) {
                continue;
            }

            let num_instances = {
                self.instances
                    .read()
                    .unwrap()
                    .get(model_name)
                    .map(|l| l.len())
                    .unwrap_or(0)
            };
            if num_instances == 0 {
                continue;
            }

            if !metrics.contains_key(model_name) {
                continue;
            }

            // ── Final despawn (n → 0): global idle reap on the last
            //    instance ────────────────────────────────────────────────
            // Always checked — does not require autoscale to be enabled.
            // The instance-level check is authoritative: Ready, no
            // in-flight requests, and last_active at least
            // drain_idle_timeout ago — the timeout is honored exactly,
            // independent of load EMA decay timescales.  Loading instances
            // (spawn in progress) are excluded because their state !=
            // Ready and their last_active is fresh; remove_instance is
            // atomic w.r.t. slot acquisition, so a request racing the
            // despawn is a no-op.
            if num_instances == 1 {
                let mut despawned = false;
                if let Some(h) = self.pick_least_loaded(model_name)
                    && h.inner()
                        .lock()
                        .unwrap()
                        .is_idle_expired(drain_idle_timeout)
                {
                    info!(
                        model = %model_name,
                        drain_idle_timeout,
                        "drain idle timeout expired, despawning last instance"
                    );
                    despawned = self.remove_instance(model_name, &h).await;
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
                    // No autoscale: evict surplus idle instances down to
                    // one once past the global drain idle timeout (the
                    // n → 0 despawn is handled above).
                    let mut n = num_instances;
                    for handle in self.instances_by_load(model_name) {
                        if n <= 1 {
                            break;
                        }
                        if handle
                            .inner()
                            .lock()
                            .unwrap()
                            .is_idle_expired(drain_idle_timeout)
                        {
                            info!(
                                model = %model_name,
                                inst = %handle.id(),
                                drain_idle_timeout,
                                "drain idle timeout expired, evicting surplus instance"
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
                if let Some(t) = last.get(model_name)
                    && now.duration_since(*t).as_secs() < a.cooldown_secs
                {
                    continue;
                }
            }

            // ── Scale-up (n → n+1) ─────────────────────────────
            // Proactive: spawn when the load EMA exceeds the scale-up
            // threshold, even if no incoming request triggers the
            // request-path gate.  Threshold `U_n = scale_up_at +
            // (n−1) × max_concurrent` — can only fire with a non-empty
            // queue (see `should_spawn` docs).
            if num_instances < cfg.max_instances {
                let up_window = *self.scale_up_value.read().unwrap();
                let load = self.load_for_window(model_name, up_window);
                let threshold = (a.scale_up_at + cfg.max_concurrent * (num_instances - 1)) as f64;
                if load > threshold {
                    // A request-driven spawn may already be in flight —
                    // don't double-spawn past the cap.
                    if self
                        .spawns_in_flight
                        .read()
                        .unwrap()
                        .contains_key(model_name)
                    {
                        debug!(
                            model = %model_name,
                            "autoscale: spawn already in flight, skipping scale-up"
                        );
                        continue;
                    }
                    info!(
                        model = %model_name,
                        up_window = ?up_window,
                        load = %load,
                        threshold = %threshold,
                        "autoscale: scaling up"
                    );
                    // Fire-and-forget: the spawn runs in a detached task
                    // (see `ensure_spawn`) which wakes parked waiters on
                    // success; requests subscribe to the entry directly.
                    let _rx = self.ensure_spawn(model_name, cfg);
                    self.last_scale_action
                        .write()
                        .unwrap()
                        .insert(model_name.clone(), now);
                    continue;
                }
            }

            // ── Scale-down (n → n−1) ─────────────────────────────
            if num_instances > 1 {
                let down_window = *self.scale_down_value.read().unwrap();
                let load = self.load_for_window(model_name, down_window);
                // Threshold `D_n = scale_down_at + (n−2) × max_concurrent`:
                // with `scale_down_at = max_concurrent` the remaining
                // (n−1) instances stay strictly below 100% utilization.
                let threshold = (a.scale_down_at + cfg.max_concurrent * (num_instances - 2)) as f64;
                if load < threshold {
                    // Prefer the least-loaded instance, but skip busy ones
                    // entirely — in-flight requests must never be
                    // interrupted.  remove_instance is atomic with respect
                    // to slot acquisition, so a lost race is a no-op.
                    for handle in self.instances_by_load(model_name) {
                        if self.remove_instance(model_name, &handle).await {
                            info!(
                                model = %model_name,
                                down_window = ?down_window,
                                load = %load,
                                threshold = %threshold,
                                "autoscale: scaled down"
                            );
                            self.last_scale_action
                                .write()
                                .unwrap()
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
            .map(|l| l.to_vec())
            .unwrap_or_default();
        list.sort_by_key(|h| h.inner().lock().unwrap().in_flight);
        list
    }
}

/// One migration victim candidate with a currently proven destination.
///
/// The destination is still advisory until the coordinator reserves it; the
/// candidate is intentionally cheap to discard and recompute if placement
/// changes before the migration transaction starts.
struct MigrationVictimCandidate {
    model_name: String,
    handle: InstanceHandle,
    instance_id: String,
    generation: u64,
    source_pci_slots: Vec<String>,
    destination: PlacementSnapshot,
    last_completed: Instant,
    in_flight: usize,
}

/// A complete dry-run migration plan.  Reservations are acquired only by
/// the next execution phase after the plan has been selected.
struct MigrationPlan {
    target_model: String,
    target: PlacementSnapshot,
    victim: MigrationVictimCandidate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MigrationError {
    NoPlan,
    ReservationUnavailable,
    VictimChanged,
    ConfigChanged,
    ReplacementLoadFailed,
    TargetLoadFailed,
}

/// Parameters for one backend spawn attempt.
struct SpawnOptions {
    /// Explicit placement for migration replacement; normal spawns select it.
    requested_gpu_indices: Option<Vec<usize>>,
    load_purpose: InstanceLoadPurpose,
    migration_id: Option<u64>,
    /// Existing coordinator reservation, if this load is part of a larger
    /// transaction.  Normal spawns create a short-lived reservation below.
    reservation_id: Option<u64>,
    enforce_instance_cap: bool,
    retain_reservation: bool,
}

impl SpawnOptions {
    fn normal() -> Self {
        Self {
            requested_gpu_indices: None,
            load_purpose: InstanceLoadPurpose::Normal,
            migration_id: None,
            reservation_id: None,
            enforce_instance_cap: true,
            retain_reservation: false,
        }
    }
}

/// One make-room eviction step (docs/003-smart-handling.md, victim
/// classes 1–3): idle victims are removed outright; busy surplus
/// duplicates are drained (bounded by the alias's `drain_timeout`).
struct MakeRoomVictim {
    model_name: String,
    handle: InstanceHandle,
    drain: bool,
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

/// RAII cleanup for a detached migration task.
struct MigrationEntryGuard {
    mgr: Arc<InstanceManager>,
    target_model: String,
}

impl Drop for MigrationEntryGuard {
    fn drop(&mut self) {
        self.mgr
            .migrations_in_flight
            .write()
            .unwrap()
            .remove(&self.target_model);
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
/// crash handler re-checks the state and ignores already-terminal instances.
async fn monitor_instance_exit(
    handle: InstanceHandle,
    crash_tx: mpsc::UnboundedSender<InstanceHandle>,
) {
    loop {
        tokio::time::sleep(CRASH_POLL_INTERVAL).await;
        let exited = {
            let mut inst = handle.inner().lock().unwrap();
            if matches!(inst.state, InstanceState::Failed | InstanceState::Retired) {
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
server:
  drain_idle_timeout_secs: 60
apikeys_file: apikeys.txt
models:
  - name: m
    context_length: 4096
    cmd: "sleep 3600"
    idle_ttl: 60
"#;

    fn test_manager() -> InstanceManager {
        let config: crate::config::Config = serde_yaml_ng::from_str(TEST_CONFIG_YAML).unwrap();
        let gpu_snapshot = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, gpu_snapshot, None);
        mgr
    }

    /// Manager for a model with autoscale enabled under the integral
    /// semantics: max_concurrent 2, scale_up_at 4, scale_down_at 2,
    /// max_instances 4 (so scale-up is possible), m1/m1 EMA windows.
    const AUTOSCALE_CONFIG_YAML: &str = r#"
server:
  drain_idle_timeout_secs: 60
apikeys_file: apikeys.txt
models:
  - name: m
    context_length: 4096
    cmd: "sleep 3600"
    idle_ttl: 60
    max_instances: 4
    max_concurrent: 2
    queue_depth: 10
    autoscale:
      enabled: true
      scale_up_at: 4
      scale_down_at: 2
      cooldown_secs: 120
"#;

    fn autoscale_manager() -> InstanceManager {
        let config: crate::config::Config = serde_yaml_ng::from_str(AUTOSCALE_CONFIG_YAML).unwrap();
        let gpu_snapshot = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, gpu_snapshot, None);
        mgr
    }

    /// Seed the m1 load EMA for model "m" (used to drive autoscale
    /// decisions deterministically without real-time sleeps).
    fn seed_load_m1(mgr: &InstanceManager, load_m1: f64) {
        let mut metrics = mgr.model_metrics.write().unwrap();
        let m = metrics.entry("m".to_owned()).or_default();
        m.load_m1 = load_m1;
        // Decay toward a high baseline so a non-zero dt in later ticks
        // can never pull the EMA below the seeded value.
        m.last_active = 10;
    }

    type TestQueueReceiver = oneshot::Receiver<Result<(String, SlotGuard), QueueTerminalError>>;

    fn test_queue_route(mgr: &InstanceManager) -> QueueRouteSnapshot {
        QueueRouteSnapshot {
            logical_model: "m".to_owned(),
            candidates: vec!["m".to_owned()],
            policy: AliasPolicy::PreferLoaded,
            make_room: MakeRoomMode::None,
            drain_timeout_secs: 0,
            max_concurrent: 4,
            generation: mgr.model_generation("m"),
            migration_id: mgr.model_migration_id("m"),
            request_id: None,
        }
    }

    fn test_queue_entry(mgr: &InstanceManager, id: u64) -> (QueueEntry, TestQueueReceiver) {
        let (assignment, receiver) = oneshot::channel();
        (
            QueueEntry {
                id,
                route: test_queue_route(mgr),
                enqueued_at: Instant::now(),
                deadline: None,
                assignment,
                cancelled: Arc::new(AtomicBool::new(false)),
            },
            receiver,
        )
    }

    #[test]
    fn request_status_registry_tracks_and_returns_bounded_records() {
        let mgr = test_manager();
        mgr.register_request("req-1", "alias");
        mgr.update_request_state("req-1", RequestState::Assigned, Some("m"), None);
        let status = mgr.request_status("req-1").unwrap();
        assert_eq!(status.request_id, "req-1");
        assert_eq!(status.resolved_model.as_deref(), Some("m"));
        assert_eq!(status.state, RequestState::Assigned);

        mgr.update_request_state("req-1", RequestState::Completed, Some("m"), None);
        assert_eq!(
            mgr.request_status("req-1").unwrap().state,
            RequestState::Completed
        );
        assert!(mgr.request_status("missing").is_none());
    }

    #[test]
    fn reservation_table_is_atomic_and_accounts_each_slot() {
        let mut table = ReservationTable::default();
        let slots = vec!["0000:01:00.0".to_owned(), "0000:02:00.0".to_owned()];
        let cleanup = ReservationCleanup {
            token: 11,
            expires_at: None,
        };

        table
            .reserve_placement(ReservationRequest {
                transition_id: 1,
                purpose: ReservationPurpose::MigrationReplacement,
                model: "victim".to_owned(),
                pci_slots: slots.clone(),
                vram_per_slot: 30,
                phase: ReservationPhase::Planned,
                cleanup,
            })
            .unwrap();

        let usage = table.usage_by_slot(None);
        assert_eq!(usage.get(&slots[0]), Some(&30));
        assert_eq!(usage.get(&slots[1]), Some(&30));
        assert_eq!(table.snapshot().len(), 2);

        table.set_phase(1, ReservationPhase::Loading);
        assert!(
            table
                .snapshot()
                .iter()
                .all(|reservation| reservation.phase == ReservationPhase::Loading)
        );

        let conflicting = vec![slots[1].clone(), "0000:03:00.0".to_owned()];
        assert!(matches!(
            table.reserve_placement(ReservationRequest {
                transition_id: 2,
                purpose: ReservationPurpose::LargeModelAdmission,
                model: "target".to_owned(),
                pci_slots: conflicting.clone(),
                vram_per_slot: 90,
                phase: ReservationPhase::Planned,
                cleanup: ReservationCleanup {
                    token: 22,
                    expires_at: None,
                },
            }),
            Err(ReservationError::AlreadyReserved { .. })
        ));
        assert!(!table.usage_by_slot(None).contains_key(&conflicting[1]));
        assert_eq!(
            table.snapshot().len(),
            2,
            "conflicts must not partially reserve"
        );

        assert_eq!(table.release_cleanup_token(11), 2);
        assert_eq!(table.release_cleanup_token(11), 0);
        assert!(table.snapshot().is_empty());
    }

    #[test]
    fn reservation_table_reserves_batch_atomically() {
        let mut table = ReservationTable::default();
        table
            .reserve_batch(vec![
                ReservationRequest {
                    transition_id: 3,
                    purpose: ReservationPurpose::MigrationReplacement,
                    model: "victim".to_owned(),
                    pci_slots: vec!["0000:01:00.0".to_owned()],
                    vram_per_slot: 30,
                    phase: ReservationPhase::Loading,
                    cleanup: ReservationCleanup {
                        token: 3,
                        expires_at: None,
                    },
                },
                ReservationRequest {
                    transition_id: 3,
                    purpose: ReservationPurpose::LargeModelAdmission,
                    model: "target".to_owned(),
                    pci_slots: vec!["0000:02:00.0".to_owned()],
                    vram_per_slot: 90,
                    phase: ReservationPhase::Planned,
                    cleanup: ReservationCleanup {
                        token: 3,
                        expires_at: None,
                    },
                },
            ])
            .unwrap();
        assert_eq!(table.snapshot().len(), 2);
        assert_eq!(table.usage_by_slot(None)["0000:01:00.0"], 30);
        assert_eq!(table.usage_by_slot(None)["0000:02:00.0"], 90);

        assert!(
            table
                .reserve_batch(vec![
                    ReservationRequest {
                        transition_id: 4,
                        purpose: ReservationPurpose::MigrationReplacement,
                        model: "other".to_owned(),
                        pci_slots: vec!["0000:02:00.0".to_owned()],
                        vram_per_slot: 1,
                        phase: ReservationPhase::Planned,
                        cleanup: ReservationCleanup {
                            token: 4,
                            expires_at: None,
                        },
                    },
                    ReservationRequest {
                        transition_id: 4,
                        purpose: ReservationPurpose::LargeModelAdmission,
                        model: "other-target".to_owned(),
                        pci_slots: vec!["0000:03:00.0".to_owned()],
                        vram_per_slot: 1,
                        phase: ReservationPhase::Planned,
                        cleanup: ReservationCleanup {
                            token: 4,
                            expires_at: None,
                        },
                    },
                ])
                .is_err()
        );
        assert_eq!(
            table.snapshot().len(),
            2,
            "batch conflicts must not partially apply"
        );
    }

    #[test]
    fn reservation_table_reaps_expired_entries() {
        let mut table = ReservationTable::default();
        let now = Instant::now();
        table
            .reserve_placement(ReservationRequest {
                transition_id: 1,
                purpose: ReservationPurpose::NormalSpawn,
                model: "model".to_owned(),
                pci_slots: vec!["0000:01:00.0".to_owned()],
                vram_per_slot: 10,
                phase: ReservationPhase::Loading,
                cleanup: ReservationCleanup {
                    token: 1,
                    expires_at: Some(now - Duration::from_secs(1)),
                },
            })
            .unwrap();
        assert_eq!(table.release_expired(now), 1);
        assert!(table.snapshot().is_empty());
    }

    #[test]
    fn reservation_table_rejects_duplicate_slots_without_mutation() {
        let mut table = ReservationTable::default();
        let slots = vec!["0000:01:00.0".to_owned(), "0000:01:00.0".to_owned()];
        assert_eq!(
            table.reserve_placement(ReservationRequest {
                transition_id: 1,
                purpose: ReservationPurpose::NormalSpawn,
                model: "model".to_owned(),
                pci_slots: slots.clone(),
                vram_per_slot: 10,
                phase: ReservationPhase::Planned,
                cleanup: ReservationCleanup {
                    token: 1,
                    expires_at: None,
                },
            }),
            Err(ReservationError::DuplicateSlot(slots[0].clone()))
        );
        assert!(table.snapshot().is_empty());
    }

    #[tokio::test]
    async fn coordinator_serializes_reservation_commands() {
        let mgr = test_manager();
        let transition_id = mgr.allocate_transition_id().await.unwrap();
        let slots = vec!["0000:01:00.0".to_owned()];
        mgr.reserve_placement(ReservationRequest {
            transition_id,
            purpose: ReservationPurpose::NormalSpawn,
            model: "model".to_owned(),
            pci_slots: slots.clone(),
            vram_per_slot: 25,
            phase: ReservationPhase::Loading,
            cleanup: ReservationCleanup {
                token: transition_id,
                expires_at: None,
            },
        })
        .await
        .unwrap();

        assert_eq!(mgr.reservation_usage(None).await.get(&slots[0]), Some(&25));
        assert!(
            mgr.set_reservation_phase(transition_id, ReservationPhase::Committed)
                .await
        );
        assert_eq!(
            mgr.reservation_snapshot().await[0].phase,
            ReservationPhase::Committed
        );
        assert_eq!(mgr.release_reservations(transition_id).await, 1);
        assert!(mgr.reservation_snapshot().await.is_empty());

        let expiring_id = mgr.allocate_transition_id().await.unwrap();
        mgr.reserve_placement(ReservationRequest {
            transition_id: expiring_id,
            purpose: ReservationPurpose::NormalSpawn,
            model: "model".to_owned(),
            pci_slots: vec!["0000:02:00.0".to_owned()],
            vram_per_slot: 10,
            phase: ReservationPhase::Loading,
            cleanup: ReservationCleanup {
                token: expiring_id,
                expires_at: Some(Instant::now() + Duration::from_millis(25)),
            },
        })
        .await
        .unwrap();
        assert_eq!(mgr.reservation_snapshot().await.len(), 1);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(mgr.reservation_snapshot().await.is_empty());
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
        let last = Instant::now() - idle_for;
        inst.last_active = last;
        inst.last_completed = last;
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
        // bump last_activity (active > 0) so the idle reap never trips
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
        // idle past drain_idle_timeout at both instance and metrics level.
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
    async fn should_spawn_gate_tracks_integral_scale_up_threshold() {
        // The request-path spawn gate must use U_n = scale_up_at +
        // (n−1) × max_concurrent with the configured EMA window: with one
        // instance (max_concurrent 2, scale_up_at 4) the boundary is 4,
        // with two instances it is 6 — a deeper queue is required before
        // a third instance is worth the VRAM.
        let mgr = autoscale_manager();
        let cfg = mgr.model_configs.read().unwrap().get("m").cloned().unwrap();
        assert!(cfg.autoscale.as_ref().unwrap().enabled);
        let handle = make_handle(InstanceState::Ready, 2, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);

        // n=1: threshold = 4 + 2×0 = 4.
        seed_load_m1(&mgr, 3.5);
        assert!(
            !mgr.should_spawn("m", &cfg),
            "load below U_1 must not spawn"
        );
        seed_load_m1(&mgr, 4.5);
        assert!(mgr.should_spawn("m", &cfg), "load above U_1 must spawn");

        // n=2: threshold = 4 + 2×1 = 6.  A queue that justified the second
        // instance (load 5) must not yet justify a third.
        let handle2 = make_handle(InstanceState::Ready, 2, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle2);
        seed_load_m1(&mgr, 5.5);
        assert!(
            !mgr.should_spawn("m", &cfg),
            "load below U_2 must not spawn"
        );
        seed_load_m1(&mgr, 6.5);
        assert!(mgr.should_spawn("m", &cfg), "load above U_2 must spawn");
    }

    #[tokio::test]
    async fn autoscale_scales_up_when_queue_drives_load_above_u_n() {
        // Tick-driven scale-up: one saturated instance (2 in-flight) with
        // a queue deep enough that the active EMA exceeds U_1 = 4 must
        // trigger ensure_spawn — no request path involved.
        let mgr = Arc::new(autoscale_manager());
        let handle = make_handle(InstanceState::Ready, 2, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);
        seed_load_m1(&mgr, 4.5);

        mgr.evaluate_autoscale().await;

        assert!(
            mgr.spawns_in_flight.read().unwrap().contains_key("m"),
            "scale-up must fire when the load EMA exceeds U_1"
        );
    }

    #[tokio::test]
    async fn autoscale_scales_down_when_load_fits_remaining_capacity() {
        // Two idle instances, load EMA below D_2 = scale_down_at + 0 = 2:
        // one instance (capacity 2) comfortably carries the load, so the
        // autoscaler must shed one and free its VRAM.
        let mgr = Arc::new(autoscale_manager());
        for _ in 0..2 {
            let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
            mgr.instances
                .write()
                .unwrap()
                .entry("m".to_owned())
                .or_default()
                .push(handle);
        }
        seed_load_m1(&mgr, 1.0);

        mgr.evaluate_autoscale().await;

        assert_eq!(
            instance_count(&mgr),
            1,
            "load below D_2 must scale down to a single instance"
        );
    }

    #[tokio::test]
    async fn enqueue_refused_when_no_instances_exist() {
        // Spawn failure with zero instances registered must not park the
        // request forever: nothing would ever wake the waiter.
        let mgr = Arc::new(test_manager());
        let result = mgr.enqueue(test_queue_route(&mgr), 8).await;
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

        let (entry, rx) = test_queue_entry(&mgr, 0);
        mgr.queues
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push_back(entry);

        mgr.remove_instance("m", &handle).await;

        assert!(
            matches!(rx.await, Ok(Err(QueueTerminalError::ModelRemoved))),
            "waiter must be completed when the last instance is removed"
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

        let (entry, rx) = test_queue_entry(&mgr, 0);
        mgr.queues
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push_back(entry);

        mgr.remove_instance("m", &h1).await;

        assert_eq!(instance_count(&mgr), 1);
        assert!(
            !rx.is_terminated(),
            "waiter must stay parked while instances remain"
        );
    }

    #[tokio::test]
    async fn shutdown_completes_queued_assignments_and_releases_state() {
        let mgr = Arc::new(test_manager());
        let handle = make_handle(InstanceState::Ready, 4, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);
        let admission = mgr
            .admit(
                &cands(&["m"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::None,
                0,
                None,
            )
            .await
            .unwrap();
        let pending = match admission {
            Admission::Waiting(pending) => pending,
            Admission::Assigned(_, _) => panic!("full instance must queue"),
        };
        assert_eq!(mgr.queue_depth("m"), 1);

        mgr.shutdown_all().await;

        assert!(matches!(
            pending.wait().await,
            Err(AcquireError::Unavailable)
        ));
        assert_eq!(mgr.queue_depth("m"), 0);
        assert!(mgr.reservation_snapshot().await.is_empty());
    }

    #[tokio::test]
    async fn queue_depth_and_blocking_state_counts_are_distinct() {
        let mgr = Arc::new(test_manager());
        let full = make_handle(InstanceState::Ready, 4, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(full);

        let first = mgr
            .register_pending(test_queue_route(&mgr), 1, true)
            .unwrap();
        assert_eq!(mgr.queue_status("m").waiting_for_slot, 1);
        assert_eq!(mgr.queue_status("m").waiting_for_spawn, 0);
        assert_eq!(mgr.queue_status("m").waiting_for_migration, 0);
        assert!(matches!(
            mgr.register_pending(test_queue_route(&mgr), 1, true),
            Err(EnqueueError::QueueFull)
        ));
        drop(first);

        let spawn_mgr = Arc::new(test_manager());
        let loading = make_handle(InstanceState::Loading, 0, Duration::ZERO);
        spawn_mgr
            .instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(loading);
        let spawn_pending = spawn_mgr
            .register_pending(test_queue_route(&spawn_mgr), 2, true)
            .unwrap();
        assert_eq!(spawn_mgr.queue_status("m").waiting_for_spawn, 1);
        drop(spawn_pending);

        let migration_mgr = Arc::new(test_manager());
        let (migration_tx, _migration_rx) = watch::channel(());
        migration_mgr
            .migrations_in_flight
            .write()
            .unwrap()
            .insert("m".to_owned(), migration_tx);
        let migration_pending = migration_mgr
            .register_pending(test_queue_route(&migration_mgr), 2, true)
            .unwrap();
        assert_eq!(migration_mgr.queue_status("m").waiting_for_migration, 1);
        drop(migration_pending);
    }

    #[tokio::test(start_paused = true)]
    async fn queue_deadline_expires_without_touching_active_slots() {
        let mgr = Arc::new(test_manager());
        mgr.register_request("req-expired", "m");
        *mgr.queue_wait_timeout_secs.write().unwrap() = 1;
        let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        let mut active_guards = Vec::new();
        for _ in 0..4 {
            active_guards.push(handle.try_acquire(4).expect("active slot available"));
        }
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        let admission = mgr
            .admit(
                &cands(&["m"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::None,
                0,
                Some("req-expired"),
            )
            .await
            .unwrap();
        let pending = match admission {
            Admission::Waiting(pending) => pending,
            Admission::Assigned(_, _) => panic!("full instance must queue"),
        };
        assert_eq!(mgr.queue_depth("m"), 1);
        assert!(matches!(
            pending.wait().await,
            Err(AcquireError::QueueTimeout)
        ));
        assert_eq!(
            mgr.request_status("req-expired").unwrap().state,
            RequestState::Expired
        );
        assert_eq!(mgr.queue_depth("m"), 0);
        assert_eq!(handle.inner().lock().unwrap().in_flight, 4);
        drop(active_guards);
    }

    #[tokio::test]
    async fn autoscale_skips_migration_participants() {
        let mgr = Arc::new(test_manager());
        let handle = make_handle(InstanceState::Ready, 0, Duration::from_secs(3600));
        handle.inner().lock().unwrap().migration_id = Some(42);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        mgr.evaluate_autoscale().await;
        assert_eq!(instance_count(&mgr), 1);
        assert_eq!(handle.inner().lock().unwrap().state, InstanceState::Ready);
    }

    #[tokio::test]
    async fn dropped_pending_assignment_removes_queue_entry_promptly() {
        let mgr = Arc::new(test_manager());
        mgr.register_request("req-cancel", "m");
        let handle = make_handle(InstanceState::Ready, 4, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);

        let admission = mgr
            .admit(
                &cands(&["m"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::None,
                0,
                Some("req-cancel"),
            )
            .await
            .unwrap();
        let pending = match admission {
            Admission::Waiting(pending) => pending,
            Admission::Assigned(_, _) => panic!("full instance must queue"),
        };
        assert_eq!(mgr.queue_depth("m"), 1);
        drop(pending);
        assert_eq!(mgr.queue_depth("m"), 0);
        assert_eq!(
            mgr.request_status("req-cancel").unwrap().state,
            RequestState::Cancelled
        );
        let metrics = mgr.scheduler_metrics_snapshot();
        assert_eq!(metrics.queue_enqueued_total, 1);
        assert_eq!(metrics.queue_cancelled_total, 1);
    }

    #[tokio::test]
    async fn cancelled_receiver_releases_raced_assignment_slot() {
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 3, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());
        let (entry, receiver) = test_queue_entry(&mgr, 19);
        mgr.queues
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push_back(entry);
        drop(receiver);

        mgr.wake_one("m");
        assert_eq!(
            handle.inner().lock().unwrap().in_flight,
            3,
            "a guard sent to a cancelled receiver must be dropped"
        );
        assert_eq!(mgr.queue_depth("m"), 0);
    }

    #[tokio::test]
    async fn enqueue_self_wakes_when_capacity_already_free() {
        // Lost-wakeup regression: capacity freed *before* the waiter parks
        // must still serve it — the post-push wake_one re-signals instead
        // of leaving the waiter parked on a free slot.
        let mgr = Arc::new(test_manager());
        let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());

        let guard = mgr.enqueue(test_queue_route(&mgr), 8).await;
        assert!(guard.is_ok(), "waiter must be served by the post-push wake");
        assert_eq!(handle.inner().lock().unwrap().in_flight, 1);
    }

    #[tokio::test]
    async fn enqueue_records_metrics_event() {
        // Regression test: queue entries must be visible to the load EMA
        // the moment they are parked, otherwise the autoscaler sits on
        // stale data (up to 30s) and never sees a burst building up.
        let mgr = Arc::new(test_manager());
        mgr.register_request("req-metrics", "m");
        // Instance at capacity: in_flight == max_concurrent == 4.
        let handle = make_handle(InstanceState::Ready, 4, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);
        // Seed metrics as if the last tick was 60s ago with zero load.
        {
            let mut metrics = mgr.model_metrics.write().unwrap();
            metrics.insert(
                "m".to_owned(),
                ModelMetrics {
                    last_update: Instant::now() - Duration::from_secs(60),
                    last_active: 0,
                    ..Default::default()
                },
            );
        }

        let admission = mgr
            .admit(
                &cands(&["m"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::None,
                0,
                Some("req-metrics"),
            )
            .await
            .unwrap();
        let pending = match admission {
            Admission::Waiting(pending) => pending,
            Admission::Assigned(_, _) => panic!("full instance must queue"),
        };
        assert_eq!(mgr.queue_depth("m"), 1);

        // The enqueue must have recorded a metrics event: active =
        // 4 in-flight + 1 queued.  `last_active` is the snapshot the next
        // EMA tick decays toward, so it proves the event fired.
        let m = mgr.model_metrics.read().unwrap().get("m").cloned().unwrap();
        assert_eq!(
            m.last_active, 5,
            "queue entry must be recorded in the metrics snapshot"
        );

        // And the next tick must move the m1 EMA toward the queued load
        // (dt ≈ 60s ⇒ alpha ≈ 0.63, so load_m1 jumps to ≈ 3.2).  The
        // enqueue's own tick already consumed the 60s seed above, so
        // rewind the clock to simulate the next tick a minute later.
        {
            let mut metrics = mgr.model_metrics.write().unwrap();
            metrics.get_mut("m").unwrap().last_update = Instant::now() - Duration::from_secs(60);
        }
        mgr.force_refresh();
        let m = mgr.model_metrics.read().unwrap().get("m").cloned().unwrap();
        assert!(
            m.load_m1 > 2.0,
            "m1 EMA must reflect queued load (got {})",
            m.load_m1
        );
        drop(pending);
    }

    #[tokio::test]
    async fn wake_one_records_metrics_event_on_assignment() {
        // Regression test: a queue → in-flight transition must update the
        // load EMA, otherwise the EMA under-reports during a queue drain
        // (the old value lingers until each response completes).
        let mgr = test_manager();
        let handle = make_handle(InstanceState::Ready, 1, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());
        {
            let mut metrics = mgr.model_metrics.write().unwrap();
            metrics.insert(
                "m".to_owned(),
                ModelMetrics {
                    last_update: Instant::now() - Duration::from_secs(60),
                    last_active: 1,
                    ..Default::default()
                },
            );
        }
        let (entry, rx) = test_queue_entry(&mgr, 0);
        mgr.queues
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push_back(entry);

        mgr.wake_one("m");

        let guard = rx
            .await
            .expect("waiter must receive an assignment")
            .expect("assignment must succeed");
        assert_eq!(handle.inner().lock().unwrap().in_flight, 2);
        // active transitioned 1 in-flight + 1 queued → 2 in-flight + 0 queued.
        let m = mgr.model_metrics.read().unwrap().get("m").cloned().unwrap();
        assert_eq!(
            m.last_active, 2,
            "assignment must record the active transition"
        );
        drop(guard);
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
        let (entry, rx) = test_queue_entry(&mgr, 0);
        mgr.queues
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push_back(entry);

        mgr.wake_one("m");

        let guard = rx
            .await
            .expect("waiter must receive an assignment")
            .expect("assignment must succeed");
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
        let (entry, rx) = test_queue_entry(&mgr, 7);
        {
            let mut queues = mgr.queues.write().unwrap();
            let q = queues.entry("m".to_owned()).or_default();
            q.push_back(entry);
        }

        mgr.wake_one("m");

        assert_eq!(mgr.queue_depth("m"), 1, "waiter must be re-parked");
        assert!(!rx.is_terminated(), "waiter must stay parked");
        let queues = mgr.queues.read().unwrap();
        assert_eq!(
            queues["m"][0].id, 7,
            "re-parked waiter must be back at the head"
        );
    }

    #[tokio::test]
    async fn remove_queued_drops_only_the_matching_entry() {
        // Timeout path: a timed-out waiter removes its own entry by id;
        // other parked waiters must stay in place.
        let mgr = test_manager();
        let (entry1, rx1) = test_queue_entry(&mgr, 1);
        let (entry2, rx2) = test_queue_entry(&mgr, 2);
        {
            let mut queues = mgr.queues.write().unwrap();
            let q = queues.entry("m".to_owned()).or_default();
            q.push_back(entry1);
            q.push_back(entry2);
        }

        mgr.remove_queued("m", 1);

        assert!(
            rx1.await.is_err(),
            "removed waiter's sender must be dropped"
        );
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
        assert!(
            mgr.is_blocked("m"),
            "model must be blocked after 3 pre-output crashes"
        );
        assert_eq!(
            instance_count(&mgr),
            0,
            "crashed instances must be unregistered"
        );
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
        assert!(
            !mgr.is_blocked("m"),
            "post-output crashes must not block the model"
        );
        assert_eq!(
            instance_count(&mgr),
            0,
            "crashed instances must be unregistered"
        );
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
        assert_eq!(
            instance_count(&mgr),
            1,
            "already-failed instance must be left alone"
        );
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
        mgr.reserve_placement(ReservationRequest {
            transition_id: 91,
            purpose: ReservationPurpose::LargeModelAdmission,
            model: "m".to_owned(),
            pci_slots: vec!["0000:01:00.0".to_owned()],
            vram_per_slot: 1,
            phase: ReservationPhase::Planned,
            cleanup: ReservationCleanup {
                token: 91,
                expires_at: None,
            },
        })
        .await
        .unwrap();

        let cfg_b: crate::config::Config = serde_yaml_ng::from_str(TEST_CONFIG_YAML_B).unwrap();
        mgr.reconcile_config(&cfg_b).await;

        assert_eq!(
            instance_count(&mgr),
            0,
            "removed model's instances must be unloaded"
        );
        assert!(!mgr.model_configs.read().unwrap().contains_key("m"));
        assert!(mgr.model_configs.read().unwrap().contains_key("n"));
        assert!(!mgr.is_blocked("m"), "reload must clear blocked flags");
        assert!(mgr.reservation_snapshot().await.is_empty());
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

        let cfg: crate::config::Config = serde_yaml_ng::from_str(TEST_CONFIG_YAML).unwrap();
        mgr.reconcile_config(&cfg).await;

        assert_eq!(
            instance_count(&mgr),
            1,
            "surviving model's instance must be kept"
        );
        assert_eq!(
            handle.inner().lock().unwrap().state,
            InstanceState::Ready,
            "unchanged spawn config must keep the instance routable"
        );
    }

    #[tokio::test]
    async fn reload_preserves_accepted_queue_route_snapshot() {
        let mgr = Arc::new(test_manager());
        let handle = make_handle(InstanceState::Ready, 4, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);
        let mut route = test_queue_route(&mgr);
        route.policy = AliasPolicy::PreferOrder;
        route.make_room = MakeRoomMode::Migrate;
        route.generation = 17;
        let pending = mgr.register_pending(route, 8, true).unwrap();

        let yaml = TEST_CONFIG_YAML.replace("idle_ttl: 60", "idle_ttl: 60\n    max_concurrent: 8");
        let cfg: crate::config::Config = serde_yaml_ng::from_str(&yaml).unwrap();
        mgr.reconcile_config(&cfg).await;

        let queued = mgr.queues.read().unwrap();
        let route = &queued["m"][0].route;
        assert_eq!(route.policy, AliasPolicy::PreferOrder);
        assert_eq!(route.make_room, MakeRoomMode::Migrate);
        assert_eq!(route.generation, 17);
        assert_eq!(route.max_concurrent, 4);
        drop(queued);
        drop(pending);
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
            InstanceState::Draining,
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
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, gpu_snapshot, None);
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
        let mgr = Arc::new(test_manager());
        let handle = make_handle(InstanceState::Failed, 0, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);

        let result = mgr.enqueue(test_queue_route(&mgr), 8).await;
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
        let mgr = Arc::new(test_manager());
        let handle = make_handle(InstanceState::Ready, 4, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);

        let result = mgr.enqueue(test_queue_route(&mgr), 8).await;
        assert!(
            matches!(result, Err(EnqueueError::Timeout)),
            "got: {:?}",
            result.map(|_| "guard")
        );
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
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, gpu_snapshot, None);
        let mgr = Arc::new(mgr);

        let result = tokio::time::timeout(Duration::from_secs(15), mgr.ensure_instance("m"))
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
        let cfg: crate::config::Config = serde_yaml_ng::from_str(TEST_CONFIG_YAML).unwrap();
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
        m3.vulkan_devices = DevicePool::Pool(vec![1, 0]);
        let mut m4 = model.clone();
        m4.vulkan_devices = DevicePool::Pool(vec![0, 1]);
        assert_eq!(
            fingerprint_with_aliases(&no_aliases, &m3),
            fingerprint_with_aliases(&no_aliases, &m4)
        );
        assert_ne!(base, fingerprint_with_aliases(&no_aliases, &m3));

        // rocm_devices change → different fingerprint (namespace and order).
        let mut rocm = model.clone();
        rocm.rocm_devices = DevicePool::Pool(vec![1, 0]);
        let mut rocm_reordered = model.clone();
        rocm_reordered.rocm_devices = DevicePool::Pool(vec![0, 1]);
        assert_eq!(
            fingerprint_with_aliases(&no_aliases, &rocm),
            fingerprint_with_aliases(&no_aliases, &rocm_reordered)
        );
        assert_ne!(base, fingerprint_with_aliases(&no_aliases, &rocm));

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
    async fn short_drain_idle_timeout_is_honored_for_last_instance() {
        // Regression: the old despawn condition required load_m1 < 0.01,
        // which with τ=60 s takes ~5 min to decay regardless of the
        // configured timeout — a short idle timeout was silently clamped.
        // The instance-level idle check honors the timeout exactly.
        let mgr = Arc::new(test_manager()); // drain_idle_timeout = 60 in TEST_CONFIG_YAML
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
            "instance idle past drain_idle_timeout must be despawned regardless of load EMAs"
        );
    }

    #[tokio::test]
    async fn surplus_idle_instances_evicted_without_autoscale() {
        // Models without autoscale config must still scale down: surplus
        // instances idle past drain_idle_timeout are evicted, one is kept.
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

        assert_eq!(
            instance_count(&mgr),
            1,
            "idle surplus instance must be evicted"
        );
        let instances = mgr.instances.read().unwrap();
        assert!(
            Arc::ptr_eq(instances["m"][0].inner(), busy.inner()),
            "the busy instance must be the survivor"
        );
    }

    #[tokio::test]
    async fn surplus_fresh_instances_kept_without_autoscale() {
        // Surplus instances that are NOT idle past drain_idle_timeout must
        // be kept.
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
            InstanceState::Draining,
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

    const MIGRATION_VICTIM_YAML: &str = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  vulkan:
    0: "0000:01:00.0"
    1: "0000:02:00.0"
    2: "0000:03:00.0"
    3: "0000:04:00.0"
models:
  - name: large
    context_length: 4096
    cmd: "sleep 3600"
    vram: 40000
    vulkan_devices: [0, 1, 2, 3]
  - name: idle-old
    context_length: 4096
    cmd: "sleep 3600"
    vram: 20000
    vulkan_devices: [0, 1, 2, 3]
  - name: idle-new
    context_length: 4096
    cmd: "sleep 3600"
    vram: 20000
    vulkan_devices: [0, 1, 2, 3]
  - name: busy
    context_length: 4096
    cmd: "sleep 3600"
    vram: 20000
    vulkan_devices: [0, 1, 2, 3]
  - name: transitioning
    context_length: 4096
    cmd: "sleep 3600"
    vram: 20000
    vulkan_devices: [0, 1, 2, 3]
  - name: stuck
    context_length: 4096
    cmd: "sleep 3600"
    vram: 20000
    vulkan_devices: [0]
"#;

    const MIGRATION_PLAN_YAML: &str = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  vulkan:
    0: "0000:01:00.0"
    1: "0000:02:00.0"
models:
  - name: large
    context_length: 4096
    cmd: "sleep 3600"
    vram: 90000
    vulkan_devices: [0, 1]
  - name: resident
    context_length: 4096
    cmd: "sleep 3600"
    vram: 30000
    vulkan_devices: [0, 1]
  - name: victim
    context_length: 4096
    cmd: "sleep 3600"
    vram: 30000
    vulkan_devices: [0, 1]
"#;

    fn multi_gpu_manager() -> InstanceManager {
        let config: crate::config::Config = serde_yaml_ng::from_str(MULTI_GPU_YAML).unwrap();
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

    fn register_aged_ready(
        mgr: &InstanceManager,
        model: &str,
        gpus: Vec<usize>,
        port: u16,
        in_flight: usize,
        completed_ago: Duration,
    ) -> InstanceHandle {
        let mut inst = Instance::new(model, gpus, port, None);
        inst.state = InstanceState::Ready;
        inst.in_flight = in_flight;
        let completed = Instant::now() - completed_ago;
        inst.last_completed = completed;
        inst.last_active = completed;
        let handle = InstanceHandle::new(inst);
        mgr.instances
            .write()
            .unwrap()
            .entry(model.to_owned())
            .or_default()
            .push(handle.clone());
        handle
    }

    #[tokio::test]
    async fn migration_demands_for_one_target_are_coalesced() {
        let mgr = Arc::new(test_manager());
        let cfg = mgr.model_configs.read().unwrap().get("m").cloned().unwrap();
        let mut first = mgr.ensure_migration("m", &cfg);
        let mut second = mgr.ensure_migration("m", &cfg);
        assert_eq!(mgr.migrations_in_flight.read().unwrap().len(), 1);
        assert!(first.changed().await.is_err());
        assert!(second.changed().await.is_err());
        assert!(mgr.migrations_in_flight.read().unwrap().is_empty());
    }

    #[tokio::test]
    async fn migration_rollback_restores_old_route_and_releases_reservations() {
        let mgr = test_manager();
        let old = make_handle(InstanceState::Draining, 1, Duration::ZERO);
        old.inner().lock().unwrap().migration_id = Some(9);
        let replacement = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        {
            let mut instance = replacement.inner().lock().unwrap();
            instance.load_purpose = InstanceLoadPurpose::MigrationReplacement;
            instance.migration_id = Some(9);
        }
        {
            let mut instances = mgr.instances.write().unwrap();
            let list = instances.entry("m".to_owned()).or_default();
            list.push(old.clone());
            list.push(replacement.clone());
        }
        mgr.reserve_placement(ReservationRequest {
            transition_id: 9,
            purpose: ReservationPurpose::MigrationReplacement,
            model: "m".to_owned(),
            pci_slots: vec!["0000:01:00.0".to_owned()],
            vram_per_slot: 1,
            phase: ReservationPhase::WaitingForDrain,
            cleanup: ReservationCleanup {
                token: 9,
                expires_at: None,
            },
        })
        .await
        .unwrap();

        let plan = MigrationPlan {
            target_model: "target".to_owned(),
            target: PlacementSnapshot {
                device_indices: vec![],
                pci_slots: vec![],
                projected_final_vram: HashMap::new(),
                projected_peak_vram: HashMap::new(),
            },
            victim: MigrationVictimCandidate {
                model_name: "m".to_owned(),
                handle: old.clone(),
                instance_id: old.id(),
                generation: 0,
                source_pci_slots: vec!["0000:02:00.0".to_owned()],
                destination: PlacementSnapshot {
                    device_indices: vec![],
                    pci_slots: vec![],
                    projected_final_vram: HashMap::new(),
                    projected_peak_vram: HashMap::new(),
                },
                last_completed: Instant::now(),
                in_flight: 1,
            },
        };

        mgr.rollback_migration(&plan, 9, Some(&replacement)).await;

        let (old_state, old_migration, old_in_flight) = {
            let old_state = old.inner().lock().unwrap();
            (old_state.state, old_state.migration_id, old_state.in_flight)
        };
        assert_eq!(old_state, InstanceState::Ready);
        assert_eq!(old_migration, None);
        assert_eq!(old_in_flight, 1, "active request must remain intact");
        assert_eq!(instance_count(&mgr), 1);
        assert!(mgr.reservation_snapshot().await.is_empty());
        assert_eq!(
            replacement.inner().lock().unwrap().state,
            InstanceState::Failed
        );
    }

    #[tokio::test]
    async fn migration_participant_crash_releases_transition_and_unblocks_state() {
        let mgr = Arc::new(test_manager());
        let replacement = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        {
            let mut instance = replacement.inner().lock().unwrap();
            instance.load_purpose = InstanceLoadPurpose::MigrationReplacement;
            instance.migration_id = Some(27);
        }
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(replacement.clone());
        mgr.reserve_placement(ReservationRequest {
            transition_id: 27,
            purpose: ReservationPurpose::MigrationReplacement,
            model: "m".to_owned(),
            pci_slots: vec!["0000:01:00.0".to_owned()],
            vram_per_slot: 1,
            phase: ReservationPhase::Loading,
            cleanup: ReservationCleanup {
                token: 27,
                expires_at: None,
            },
        })
        .await
        .unwrap();

        mgr.handle_crash(replacement).await;

        assert_eq!(registered_count(&mgr, "m"), 0);
        assert!(mgr.reservation_snapshot().await.is_empty());
        assert!(
            !mgr.is_blocked("m"),
            "a ready migration crash is post-output"
        );
    }

    #[tokio::test]
    async fn post_handoff_target_failure_keeps_replacement_routable() {
        let mgr = test_manager();
        let replacement = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        {
            let mut instance = replacement.inner().lock().unwrap();
            instance.load_purpose = InstanceLoadPurpose::MigrationReplacement;
            instance.migration_id = Some(12);
        }
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(replacement.clone());
        mgr.reserve_placement(ReservationRequest {
            transition_id: 12,
            purpose: ReservationPurpose::LargeModelAdmission,
            model: "target".to_owned(),
            pci_slots: vec!["0000:01:00.0".to_owned()],
            vram_per_slot: 1,
            phase: ReservationPhase::Loading,
            cleanup: ReservationCleanup {
                token: 12,
                expires_at: None,
            },
        })
        .await
        .unwrap();

        mgr.retain_migrated_replacement_after_target_failure(12, "m", &replacement)
            .await;

        let (state, migration_id) = {
            let instance = replacement.inner().lock().unwrap();
            (instance.state, instance.migration_id)
        };
        assert_eq!(state, InstanceState::Ready);
        assert_eq!(migration_id, None);
        assert_eq!(instance_count(&mgr), 1);
        assert!(mgr.reservation_snapshot().await.is_empty());
    }

    #[tokio::test]
    async fn migration_dry_run_charges_replacement_before_target() {
        let config: crate::config::Config = serde_yaml_ng::from_str(MIGRATION_PLAN_YAML).unwrap();
        let snapshot = Arc::new(tokio::sync::RwLock::new(vec![
            gpu_metrics(0, "0000:01:00.0", 100_000),
            gpu_metrics(1, "0000:02:00.0", 100_000),
        ]));
        let mgr = {
            let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, snapshot, None);
            mgr
        };
        register_aged_ready(
            &mgr,
            "resident",
            vec![0],
            54321,
            0,
            Duration::from_secs(100),
        );
        let victim =
            register_aged_ready(&mgr, "victim", vec![1], 54322, 0, Duration::from_secs(300));
        let target_cfg = mgr
            .model_configs
            .read()
            .unwrap()
            .get("large")
            .cloned()
            .unwrap();

        let plan = mgr.plan_migration(&target_cfg).await.unwrap();
        assert_eq!(plan.victim.handle.id(), victim.id());
        assert_eq!(plan.victim.source_pci_slots, vec!["0000:02:00.0"]);
        assert_eq!(plan.victim.destination.pci_slots, vec!["0000:01:00.0"]);
        assert_eq!(
            plan.victim.destination.projected_peak_vram,
            plan.victim.destination.projected_final_vram
        );
        assert_eq!(
            plan.victim.destination.projected_peak_vram["0000:01:00.0"],
            60_000 * 1024 * 1024,
            "replacement must fit alongside the resident model"
        );
        assert_eq!(plan.target.pci_slots, vec!["0000:02:00.0"]);
        assert_eq!(plan.target_model, "large");
        assert_eq!(
            plan.target.projected_final_vram["0000:01:00.0"],
            60_000 * 1024 * 1024
        );
        assert_eq!(
            plan.target.projected_final_vram["0000:02:00.0"],
            90_000 * 1024 * 1024
        );

        let mut too_large = target_cfg;
        too_large.vram = 100_001;
        assert!(mgr.plan_migration(&too_large).await.is_none());
    }

    #[tokio::test]
    async fn migration_victims_are_lru_ordered_and_destination_safe() {
        let config: crate::config::Config = serde_yaml_ng::from_str(MIGRATION_VICTIM_YAML).unwrap();
        let snapshot = Arc::new(tokio::sync::RwLock::new(vec![
            gpu_metrics(0, "0000:01:00.0", 48_000),
            gpu_metrics(1, "0000:02:00.0", 48_000),
            gpu_metrics(2, "0000:03:00.0", 48_000),
            gpu_metrics(3, "0000:04:00.0", 48_000),
        ]));
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, snapshot, None);

        let idle_old = register_aged_ready(
            &mgr,
            "idle-old",
            vec![0],
            54321,
            0,
            Duration::from_secs(300),
        );
        let idle_new = register_aged_ready(
            &mgr,
            "idle-new",
            vec![1],
            54322,
            0,
            Duration::from_secs(100),
        );
        let busy = register_aged_ready(&mgr, "busy", vec![2], 54323, 1, Duration::from_secs(1000));
        let transitioning = register_aged_ready(
            &mgr,
            "transitioning",
            vec![3],
            54324,
            0,
            Duration::from_secs(400),
        );
        transitioning.inner().lock().unwrap().migration_id = Some(9);
        register_aged_ready(&mgr, "stuck", vec![0], 54325, 0, Duration::from_secs(500));

        let candidates = mgr.enumerate_migration_victims("large").await;
        let names: Vec<&str> = candidates
            .iter()
            .map(|candidate| candidate.model_name.as_str())
            .collect();
        assert_eq!(names, vec!["idle-old", "idle-new", "busy"]);
        assert_eq!(candidates[0].handle.id(), idle_old.id());
        assert_eq!(candidates[1].handle.id(), idle_new.id());
        assert_eq!(candidates[2].handle.id(), busy.id());
        for candidate in &candidates {
            assert!(
                candidate
                    .source_pci_slots
                    .iter()
                    .all(|slot| !candidate.destination.pci_slots.contains(slot)),
                "destination must not reuse source slots"
            );
        }
    }

    #[tokio::test]
    async fn selection_observes_coordinator_reservations() {
        let mgr = multi_gpu_manager();
        let transition_id = mgr.allocate_transition_id().await.unwrap();
        let reserved_slot = "0000:01:00.0".to_owned();
        mgr.reserve_placement(ReservationRequest {
            transition_id,
            purpose: ReservationPurpose::LargeModelAdmission,
            model: "target".to_owned(),
            pci_slots: vec![reserved_slot.clone()],
            vram_per_slot: 48_000 * 1024 * 1024,
            phase: ReservationPhase::Loading,
            cleanup: ReservationCleanup {
                token: transition_id,
                expires_at: None,
            },
        })
        .await
        .unwrap();

        let cfg = mgr
            .model_configs
            .read()
            .unwrap()
            .get("big")
            .cloned()
            .unwrap();
        let placement = mgr.select_gpus_for_model(&cfg).await;
        assert_eq!(placement.len(), 2);
        assert!(
            !placement.contains(&0),
            "reserved PCI slot must not be selected"
        );

        assert_eq!(mgr.release_reservations(transition_id).await, 1);
    }

    #[tokio::test]
    async fn placement_snapshot_reports_concrete_devices_and_projected_usage() {
        let mgr = multi_gpu_manager();
        let cfg = mgr
            .model_configs
            .read()
            .unwrap()
            .get("big")
            .cloned()
            .unwrap();

        let snapshot = mgr
            .select_placement_snapshot(&cfg, &HashSet::new())
            .await
            .unwrap();
        assert_eq!(snapshot.device_indices.len(), 2);
        assert_eq!(snapshot.pci_slots.len(), 2);
        assert_eq!(snapshot.projected_peak_vram, snapshot.projected_final_vram);
        for pci_slot in &snapshot.pci_slots {
            assert_eq!(
                snapshot.projected_final_vram.get(pci_slot),
                Some(&(20_000 * 1024 * 1024))
            );
        }
    }

    #[tokio::test]
    async fn select_picks_two_distinct_gpus_and_complement_for_second_instance() {
        let mgr = multi_gpu_manager();
        let cfg = mgr
            .model_configs
            .read()
            .unwrap()
            .get("big")
            .cloned()
            .unwrap();

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
        let cfg = mgr
            .model_configs
            .read()
            .unwrap()
            .get("big")
            .cloned()
            .unwrap();
        register_ready(&mgr, "big", vec![0, 1], 54321);
        register_ready(&mgr, "big", vec![2], 54322);

        let placement = mgr.select_gpus_for_model(&cfg).await;
        assert!(
            placement.is_empty(),
            "fewer candidates than gpus must yield no placement, got {placement:?}"
        );
    }

    // ── explicit placement lists (nested form) ────────────────────────

    #[tokio::test]
    async fn select_uses_explicit_dual_gpu_placement() {
        // [[0,1]]: one placement spanning both GPUs — the instance must
        // land on exactly that pair.
        let mgr = multi_gpu_manager();
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: dual\ncontext_length: 4096\ncmd: \"sleep 1\"\nvram: 20000\nvulkan_devices: [[0,1]]",
        )
        .unwrap();
        let placement = mgr.select_gpus_for_model(&cfg).await;
        assert_eq!(placement, vec![0, 1], "dual placement must use both GPUs");
    }

    #[tokio::test]
    async fn select_avoids_occupied_single_gpu_placement() {
        // [[0],[1]]: single-GPU alternatives, either one is fine.  With GPU
        // 0 already carrying an instance of the model, GPU 1 is chosen.
        let mgr = multi_gpu_manager();
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: single\ncontext_length: 4096\ncmd: \"sleep 1\"\nvram: 20000\nvulkan_devices: [[0],[1]]",
        )
        .unwrap();
        register_ready(&mgr, "single", vec![0], 54321);
        let placement = mgr.select_gpus_for_model(&cfg).await;
        assert_eq!(placement, vec![1]);
    }

    #[tokio::test]
    async fn select_picks_best_heterogeneous_placement() {
        // [[0],[1,2]]: one single-GPU and one dual-GPU alternative.  Both
        // empty, [0] wins on lower index sum; once GPU 0 is occupied, the
        // instance spans 1 and 2.
        let mgr = multi_gpu_manager();
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: mixed\ncontext_length: 4096\ncmd: \"sleep 1\"\nvram: 20000\nvulkan_devices: [[0],[1,2]]",
        )
        .unwrap();
        assert_eq!(mgr.select_gpus_for_model(&cfg).await, vec![0]);

        register_ready(&mgr, "mixed", vec![0], 54321);
        assert_eq!(mgr.select_gpus_for_model(&cfg).await, vec![1, 2]);

        // All alternatives infeasible → no placement.
        register_ready(&mgr, "mixed", vec![1, 2], 54322);
        assert!(mgr.select_gpus_for_model(&cfg).await.is_empty());
    }

    #[tokio::test]
    async fn explicit_placement_snapshot_rejects_non_alternative() {
        // Migration may only place an instance on one of the configured
        // alternatives — a random 2-subset of the union is not valid.
        let mgr = multi_gpu_manager();
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: mixed\ncontext_length: 4096\ncmd: \"sleep 1\"\nvram: 20000\nvulkan_devices: [[0],[1,2]]",
        )
        .unwrap();
        assert!(
            mgr.explicit_placement_snapshot(&cfg, &[1, 2]).is_some(),
            "configured alternative must be accepted"
        );
        assert!(
            mgr.explicit_placement_snapshot(&cfg, &[2, 1]).is_none(),
            "permutation of an alternative is not a configured placement"
        );
        assert!(
            mgr.explicit_placement_snapshot(&cfg, &[0, 1]).is_none(),
            "cross-alternative subset must be rejected"
        );
    }

    #[test]
    fn fingerprint_ignores_placement_structure_and_order() {
        // The spawned process depends on the device set, not on how the
        // config groups it into placement alternatives.
        let base: ModelConfig = serde_yaml_ng::from_str(
            "name: m\ncontext_length: 4096\ncmd: \"sleep 1\"\nvram: 1000\nvulkan_devices: [[0],[1]]",
        )
        .unwrap();
        let aliases = HashMap::new();
        let f = fingerprint_with_aliases(&aliases, &base);

        let mut reordered = base.clone();
        reordered.vulkan_devices = DevicePool::Placements(vec![vec![1], vec![0]]);
        assert_eq!(f, fingerprint_with_aliases(&aliases, &reordered));

        // The device set is what defines the process: the same union
        // grouped as a dual-GPU placement or as two single-GPU
        // alternatives has the same fingerprint (gpus stays 1).
        let mut dual = base.clone();
        dual.vulkan_devices = DevicePool::Placements(vec![vec![0, 1]]);
        assert_eq!(f, fingerprint_with_aliases(&aliases, &dual));
        let mut flat = base.clone();
        flat.vulkan_devices = DevicePool::Pool(vec![0, 1]);
        assert_eq!(f, fingerprint_with_aliases(&aliases, &flat));

        // A different span (gpus: 2) changes the spawned process and must
        // change the fingerprint.
        let mut flat_dual = base.clone();
        flat_dual.vulkan_devices = DevicePool::Pool(vec![0, 1]);
        flat_dual.gpus = 2;
        assert_ne!(f, fingerprint_with_aliases(&aliases, &flat_dual));
    }

    // ── ROCm placement ────────────────────────────────────────────────

    const ROCM_GPU_YAML: &str = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  rocm:
    0: "0000:0a:00.0"
    1: "0000:0b:00.0"
    2: "0000:0c:00.0"
    3: "0000:0d:00.0"
models:
  - name: big
    context_length: 4096
    cmd: "sleep 3600"
    idle_ttl: 60
    vram: 20000
    gpus: 2
    rocm_devices: [0, 1, 2, 3]
"#;

    fn rocm_gpu_manager() -> InstanceManager {
        let config: crate::config::Config = serde_yaml_ng::from_str(ROCM_GPU_YAML).unwrap();
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
    async fn rocm_select_tiles_pool_across_instances() {
        let mgr = rocm_gpu_manager();
        let cfg = mgr
            .model_configs
            .read()
            .unwrap()
            .get("big")
            .cloned()
            .unwrap();

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
    async fn rocm_select_respects_per_gpu_vram_shares() {
        let mgr = rocm_gpu_manager();
        register_ready(&mgr, "big", vec![0, 1], 54321);

        let small: ModelConfig = serde_yaml_ng::from_str(
            "name: small\ncontext_length: 4096\ncmd: \"sleep 1\"\nvram: 30000\ngpus: 1\nrocm_devices: [0, 1, 2, 3]",
        )
        .unwrap();
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
    async fn rocm_and_vulkan_share_vram_accounting_by_pci_slot() {
        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  vulkan:
    0: "0000:0a:00.0"
  rocm:
    0: "0000:0a:00.0"
models:
  - name: vulkan-model
    context_length: 4096
    cmd: "sleep 3600"
    vram: 30000
    vulkan_devices: [0]
  - name: rocm-model
    context_length: 4096
    cmd: "sleep 3600"
    vram: 30000
    rocm_devices: [0]
"#;
        let config: crate::config::Config = serde_yaml_ng::from_str(yaml).unwrap();
        let snapshot = Arc::new(tokio::sync::RwLock::new(vec![gpu_metrics(
            0,
            "0000:0a:00.0",
            48000,
        )]));
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, snapshot, None);
        let rocm_cfg = mgr
            .model_configs
            .read()
            .unwrap()
            .get("rocm-model")
            .cloned()
            .unwrap();
        register_ready(&mgr, "vulkan-model", vec![0], 54321);

        assert!(
            mgr.select_gpus_for_model(&rocm_cfg).await.is_empty(),
            "ROCm placement must account for Vulkan VRAM on the same PCI slot"
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
        let config: crate::config::Config = serde_yaml_ng::from_str(CUDA_GPU_YAML).unwrap();
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
        let cfg = mgr
            .model_configs
            .read()
            .unwrap()
            .get("big")
            .cloned()
            .unwrap();

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
        let cfg = mgr
            .model_configs
            .read()
            .unwrap()
            .get("big")
            .cloned()
            .unwrap();
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

        let fits = mgr
            .model_configs
            .read()
            .unwrap()
            .get("fits")
            .cloned()
            .unwrap();
        let placement = mgr.select_gpus_for_model(&fits).await;
        assert_eq!(
            placement.len(),
            1,
            "static vram_mb must enable placement without metrics"
        );

        let toobig = mgr
            .model_configs
            .read()
            .unwrap()
            .get("toobig")
            .cloned()
            .unwrap();
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
        let snapshot = Arc::new(tokio::sync::RwLock::new(vec![gpu_metrics(
            0,
            "0000:0a:00.0",
            48000,
        )]));
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
        let base: ModelConfig =
            serde_yaml_ng::from_str("name: m\ncontext_length: 4096\ncmd: \"sleep 1\"\nvram: 1000")
                .unwrap();
        let mut vk = base.clone();
        vk.vulkan_devices = DevicePool::Pool(vec![0]);
        let mut cu = base.clone();
        cu.cuda_devices = DevicePool::Pool(vec![0]);
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
    fn resolve_cmd_substitutes_max_concurrent() {
        let mgr = test_manager();
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: x\nmax_concurrent: 8\ncmd: \"run --parallel {max_concurrent}\"",
        )
        .unwrap();
        assert_eq!(mgr.resolve_cmd(&cfg, 9999), "run --parallel 8");
    }

    #[test]
    fn resolve_cmd_substitutes_name() {
        let mgr = test_manager();
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: qwen3-tts\ncmd: \"audiocpp_server --config /etc/llm-orch/audio/{name}.json --port {port}\"",
        )
        .unwrap();
        assert_eq!(
            mgr.resolve_cmd(&cfg, 9123),
            "audiocpp_server --config /etc/llm-orch/audio/qwen3-tts.json --port 9123"
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
    async fn autoscale_does_not_reap_loading_within_per_model_spawn_timeout() {
        // A model with a large `spawn_timeout_secs` (e.g. vLLM doing CUDA
        // graph capture over several minutes) must not be reaped by the
        // stranded-in-Loading safety net while it is still within its own
        // spawn window — the reaper must use the per-model timeout, not the
        // smaller global default.
        let yaml = r#"
server:
  drain_idle_timeout_secs: 60
apikeys_file: apikeys.txt
models:
  - name: m
    context_length: 4096
    cmd: "sleep 3600"
    idle_ttl: 60
    spawn_timeout_secs: 600
"#;
        let config: crate::config::Config = serde_yaml_ng::from_str(yaml).unwrap();
        let gpu_snapshot = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, gpu_snapshot, None);
        let mgr = Arc::new(mgr);
        // Loading for 240 s: beyond the global default (120*2=240 boundary
        // would reap it) but well within 600*2 for this model.
        let handle = make_handle(InstanceState::Loading, 0, Duration::from_secs(240));
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle);

        mgr.evaluate_autoscale().await;

        assert_eq!(
            instance_count(&mgr),
            1,
            "instance within its per-model spawn timeout must not be reaped"
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
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, gpu_snapshot, None);
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

    // ── Multi-model alias routing (docs/003-smart-handling.md) ───────────

    /// Two spawn-failing models (`true` exits instantly), no device pools.
    const TWO_MODEL_YAML: &str = r#"
server: {}
apikeys_file: apikeys.txt
models:
  - name: m1
    context_length: 4096
    cmd: "true"
  - name: m2
    context_length: 4096
    cmd: "true"
"#;

    fn two_model_manager() -> InstanceManager {
        let config: crate::config::Config = serde_yaml_ng::from_str(TWO_MODEL_YAML).unwrap();
        let gpu_snapshot = Arc::new(tokio::sync::RwLock::new(Vec::new()));
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, gpu_snapshot, None);
        mgr
    }

    /// Three 48 GB GPUs; `moe` needs 30 GB (pool 0–1), `dense` needs
    /// 20 GB (pool 0–2).  Two dense instances on GPUs 0–1 leave only
    /// 28 GB free there, so `moe` cannot fit without eviction — while
    /// `dense` can still spawn on the free GPU 2.
    const MAKEROOM_YAML: &str = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  vulkan:
    0: "0000:01:00.0"
    1: "0000:02:00.0"
    2: "0000:03:00.0"
models:
  - name: moe
    context_length: 4096
    cmd: "true"
    idle_ttl: 60
    vram: 30000
    gpus: 1
    vulkan_devices: [0, 1]
  - name: dense
    context_length: 4096
    cmd: "true"
    idle_ttl: 60
    vram: 20000
    gpus: 1
    max_instances: 3
    queue_depth: 0
    vulkan_devices: [0, 1, 2]
"#;

    fn makeroom_manager() -> InstanceManager {
        let config: crate::config::Config = serde_yaml_ng::from_str(MAKEROOM_YAML).unwrap();
        let snapshot = Arc::new(tokio::sync::RwLock::new(vec![
            gpu_metrics(0, "0000:01:00.0", 48000),
            gpu_metrics(1, "0000:02:00.0", 48000),
            gpu_metrics(2, "0000:03:00.0", 48000),
        ]));
        let (mgr, _release_rx, _crash_rx) = InstanceManager::new(&config, snapshot, None);
        mgr
    }

    fn cands(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn register_inst(
        mgr: &InstanceManager,
        model: &str,
        gpus: Vec<usize>,
        in_flight: usize,
        idle_for: Duration,
    ) -> InstanceHandle {
        let mut inst = Instance::new(model, gpus, 54321, None);
        inst.state = InstanceState::Ready;
        inst.in_flight = in_flight;
        inst.last_active = Instant::now() - idle_for;
        let h = InstanceHandle::new(inst);
        mgr.instances
            .write()
            .unwrap()
            .entry(model.to_owned())
            .or_default()
            .push(h.clone());
        h
    }

    fn registered_count(mgr: &InstanceManager, model: &str) -> usize {
        mgr.instances
            .read()
            .unwrap()
            .get(model)
            .map(|l| l.len())
            .unwrap_or(0)
    }

    fn crash_count(mgr: &InstanceManager, model: &str) -> usize {
        mgr.crash_counts
            .lock()
            .unwrap()
            .get(model)
            .copied()
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn candidates_warm_slot_follows_order() {
        let mgr = Arc::new(two_model_manager());
        register_inst(&mgr, "m1", vec![], 0, Duration::ZERO);
        register_inst(&mgr, "m2", vec![], 0, Duration::ZERO);

        let (name, guard) = mgr
            .acquire_for_candidates(
                &cands(&["m1", "m2"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::EvictIdle,
                300,
            )
            .await
            .unwrap();
        assert_eq!(name, "m1", "first candidate with a warm slot wins");
        drop(guard);

        let (name, _guard) = mgr
            .acquire_for_candidates(
                &cands(&["m2", "m1"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::EvictIdle,
                300,
            )
            .await
            .unwrap();
        assert_eq!(name, "m2", "reversed order reverses the pick");
    }

    #[tokio::test]
    async fn candidates_skip_unknown_and_blocked() {
        let mgr = Arc::new(two_model_manager());
        register_inst(&mgr, "m2", vec![], 0, Duration::ZERO);

        // Unknown candidate is skipped, the known one serves.
        let (name, _guard) = mgr
            .acquire_for_candidates(
                &cands(&["nope", "m2"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::EvictIdle,
                300,
            )
            .await
            .unwrap();
        assert_eq!(name, "m2");

        // All unknown → Unavailable.
        let res = mgr
            .acquire_for_candidates(
                &cands(&["nope"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::EvictIdle,
                300,
            )
            .await;
        assert!(matches!(res, Err(AcquireError::Unavailable)));

        // All blocked → Blocked (single-model get_or_spawn semantics).
        mgr.block_model("m2");
        let res = mgr
            .acquire_for_candidates(
                &cands(&["nope", "m2"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::EvictIdle,
                300,
            )
            .await;
        assert!(matches!(res, Err(AcquireError::Blocked)));
    }

    #[tokio::test]
    async fn candidates_spawn_failure_falls_through_to_unavailable() {
        // Nothing loaded; both candidates fail to spawn (cmd "true" exits
        // instantly).  Pass C queues on the first candidate, which fails
        // fast — nothing could ever wake the waiter.
        let mgr = Arc::new(two_model_manager());
        let res = tokio::time::timeout(
            Duration::from_secs(15),
            mgr.acquire_for_candidates(
                &cands(&["m1", "m2"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::EvictIdle,
                300,
            ),
        )
        .await
        .expect("must not park");
        assert!(matches!(res, Err(AcquireError::Unavailable)));
    }

    #[tokio::test(start_paused = true)]
    async fn candidates_queue_on_first_loaded_candidate() {
        // m1 blocked (skipped), m2 loaded but full.  Pass C must queue on
        // m2 — the loaded candidate — where the wait times out
        // (NoCapacity), rather than on m1, which would fail fast
        // (Unavailable).
        let mgr = Arc::new(two_model_manager());
        mgr.block_model("m1");
        register_inst(&mgr, "m2", vec![], 4, Duration::ZERO); // max_concurrent = 4

        let res = mgr
            .acquire_for_candidates(
                &cands(&["m1", "m2"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::EvictIdle,
                300,
            )
            .await;
        assert!(
            matches!(res, Err(AcquireError::QueueTimeout)),
            "queued on the loaded candidate → timeout, got: {:?}",
            res.map(|_| "guard")
        );
    }

    #[tokio::test]
    async fn spawn_feasible_respects_instance_cap() {
        let mgr = test_manager(); // model "m", no device pool, max_instances = 1
        let cfg = mgr.model_configs.read().unwrap().get("m").cloned().unwrap();

        assert!(mgr.spawn_feasible(&cfg, &HashSet::new()).await);

        let handle = make_handle(InstanceState::Ready, 0, Duration::ZERO);
        mgr.instances
            .write()
            .unwrap()
            .entry("m".to_owned())
            .or_default()
            .push(handle.clone());
        assert!(
            !mgr.spawn_feasible(&cfg, &HashSet::new()).await,
            "at cap → infeasible"
        );

        // Retiring (Failed) instances don't count toward the cap.
        handle.inner().lock().unwrap().state = InstanceState::Failed;
        assert!(mgr.spawn_feasible(&cfg, &HashSet::new()).await);
    }

    #[tokio::test]
    async fn plan_make_room_evicts_stale_idle_surplus() {
        let mgr = makeroom_manager();
        let moe = mgr
            .model_configs
            .read()
            .unwrap()
            .get("moe")
            .cloned()
            .unwrap();
        // Two stale dense instances tile both GPUs (28 GB free each) —
        // moe (30 GB) fits nowhere.
        let older = register_inst(&mgr, "dense", vec![0], 0, Duration::from_secs(7200));
        let _newer = register_inst(&mgr, "dense", vec![1], 0, Duration::from_secs(3600));
        assert!(!mgr.spawn_feasible(&moe, &HashSet::new()).await);

        let plan = mgr
            .plan_make_room(&moe, MakeRoomMode::EvictIdle)
            .await
            .expect("one eviction must suffice");
        assert_eq!(plan.len(), 1, "one freed GPU fits the moe");
        assert!(!plan[0].drain, "idle victim is removed, not drained");
        assert_eq!(plan[0].model_name, "dense");
        assert!(
            Arc::ptr_eq(plan[0].handle.inner(), older.inner()),
            "least-recently-used victim first"
        );
    }

    #[tokio::test]
    async fn plan_make_room_respects_protection_window() {
        let mgr = makeroom_manager();
        let moe = mgr
            .model_configs
            .read()
            .unwrap()
            .get("moe")
            .cloned()
            .unwrap();
        // Fresh dense instances (idle_ttl = 60 s protection) must never
        // be evicted to make room.
        register_inst(&mgr, "dense", vec![0], 0, Duration::ZERO);
        register_inst(&mgr, "dense", vec![1], 0, Duration::ZERO);

        assert!(
            mgr.plan_make_room(&moe, MakeRoomMode::EvictIdle)
                .await
                .is_none(),
            "instances inside their idle_ttl window are protected"
        );
    }

    #[tokio::test]
    async fn plan_make_room_drains_busy_surplus_only_when_allowed() {
        let mgr = makeroom_manager();
        let moe = mgr
            .model_configs
            .read()
            .unwrap()
            .get("moe")
            .cloned()
            .unwrap();
        // Both dense instances busy — no idle victim exists.
        let less_busy = register_inst(&mgr, "dense", vec![0], 1, Duration::ZERO);
        let _more_busy = register_inst(&mgr, "dense", vec![1], 2, Duration::ZERO);

        assert!(
            mgr.plan_make_room(&moe, MakeRoomMode::EvictIdle)
                .await
                .is_none(),
            "evict_idle must not touch busy instances"
        );

        let plan = mgr
            .plan_make_room(&moe, MakeRoomMode::DrainSurplus)
            .await
            .expect("draining one busy duplicate frees a GPU");
        assert_eq!(plan.len(), 1);
        assert!(plan[0].drain, "busy duplicate is drained, not removed");
        assert!(
            Arc::ptr_eq(plan[0].handle.inner(), less_busy.inner()),
            "least-busy duplicate is drained first"
        );
    }

    #[tokio::test]
    async fn plan_make_room_never_drains_last_serving_instance() {
        let mgr = makeroom_manager();
        let moe = mgr
            .model_configs
            .read()
            .unwrap()
            .get("moe")
            .cloned()
            .unwrap();
        // A single busy dense instance: draining it would take the model
        // offline for its own in-flight traffic.  The moe instance keeps
        // GPU 1 occupied (and is itself busy, but own-model drains are
        // off-limits).
        register_inst(&mgr, "dense", vec![0], 1, Duration::ZERO);
        register_inst(&mgr, "moe", vec![1], 1, Duration::ZERO);
        assert!(!mgr.spawn_feasible(&moe, &HashSet::new()).await);

        assert!(
            mgr.plan_make_room(&moe, MakeRoomMode::DrainSurplus)
                .await
                .is_none(),
            "the last busy instance of a model is never a victim"
        );
    }

    #[tokio::test]
    async fn plan_make_room_never_drains_own_model() {
        let mgr = makeroom_manager();
        let moe = mgr
            .model_configs
            .read()
            .unwrap()
            .get("moe")
            .cloned()
            .unwrap();
        // Two busy moe duplicates occupy both GPUs (18 GB free each).
        register_inst(&mgr, "moe", vec![0], 1, Duration::ZERO);
        register_inst(&mgr, "moe", vec![1], 1, Duration::ZERO);
        assert!(!mgr.spawn_feasible(&moe, &HashSet::new()).await);

        assert!(
            mgr.plan_make_room(&moe, MakeRoomMode::DrainSurplus)
                .await
                .is_none(),
            "busy duplicates of the requesting model itself are off-limits"
        );
    }

    #[tokio::test]
    async fn acquire_make_room_evicts_then_attempts_spawn() {
        // End-to-end Pass B round 2: moe does not fit (28 GB free per
        // GPU), a stale idle dense instance is evicted, then the moe spawn
        // is attempted (and fails — cmd "true").
        let mgr = Arc::new(makeroom_manager());
        let stale = register_inst(&mgr, "dense", vec![0], 0, Duration::from_secs(3600));
        let busy = register_inst(&mgr, "dense", vec![1], 4, Duration::ZERO); // full

        let res = tokio::time::timeout(
            Duration::from_secs(15),
            mgr.acquire_for_candidates(
                &cands(&["moe"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::EvictIdle,
                300,
            ),
        )
        .await
        .expect("must not park");
        assert!(matches!(res, Err(AcquireError::Unavailable)));
        assert_eq!(crash_count(&mgr, "moe"), 1, "moe spawn must be attempted");
        assert_eq!(registered_count(&mgr, "dense"), 1, "stale instance evicted");
        assert!(
            mgr.instances.read().unwrap()["dense"]
                .iter()
                .any(|h| Arc::ptr_eq(h.inner(), busy.inner())),
            "busy instance untouched"
        );
        assert!(
            !mgr.instances.read().unwrap()["dense"]
                .iter()
                .any(|h| Arc::ptr_eq(h.inner(), stale.inner())),
            "stale instance gone"
        );
    }

    #[tokio::test]
    async fn acquire_prefer_loaded_spawns_fitting_candidate_without_eviction() {
        // Decision 6: a spillover candidate that fits *without* eviction
        // beats evicting for the preferred candidate.  Both dense
        // instances are full (28 GB free per GPU): moe (30 GB) cannot fit
        // without eviction, dense (20 GB, max_instances 3) can — with
        // make_room disabled, dense must be spawned and nothing evicted.
        let mgr = Arc::new(makeroom_manager());
        register_inst(&mgr, "dense", vec![0], 4, Duration::from_secs(3600));
        register_inst(&mgr, "dense", vec![1], 4, Duration::from_secs(3600));

        let res = tokio::time::timeout(
            Duration::from_secs(15),
            mgr.acquire_for_candidates(
                &cands(&["moe", "dense"]),
                AliasPolicy::PreferLoaded,
                MakeRoomMode::None,
                300,
            ),
        )
        .await
        .expect("must not park");
        // dense spawn fails (cmd "true"); Pass C queues on dense (loaded,
        // queue_depth 0) → NoCapacity.
        assert!(matches!(res, Err(AcquireError::NoCapacity)));
        assert_eq!(crash_count(&mgr, "dense"), 1, "dense spawn attempted");
        assert_eq!(crash_count(&mgr, "moe"), 0, "moe spawn never attempted");
        assert_eq!(registered_count(&mgr, "dense"), 2, "nothing evicted");
    }

    #[tokio::test]
    async fn acquire_prefer_order_makes_room_for_first_candidate() {
        // Two stale idle dense instances tile the moe pool.  prefer_order
        // tries moe fully — including make-room eviction — before
        // advancing to dense: one dense instance is evicted and the moe
        // spawn attempted; when it fails (cmd "true"), the remaining
        // dense instance serves via its warm slot.
        let mgr = Arc::new(makeroom_manager());
        register_inst(&mgr, "dense", vec![0], 0, Duration::from_secs(3600));
        register_inst(&mgr, "dense", vec![1], 0, Duration::from_secs(3600));

        let (name, _guard) = tokio::time::timeout(
            Duration::from_secs(15),
            mgr.acquire_for_candidates(
                &cands(&["moe", "dense"]),
                AliasPolicy::PreferOrder,
                MakeRoomMode::EvictIdle,
                300,
            ),
        )
        .await
        .expect("must not park")
        .expect("dense must serve after the failed moe spawn");
        assert_eq!(name, "dense", "spillover candidate serves at the end");
        assert_eq!(crash_count(&mgr, "moe"), 1, "moe spawn attempted first");
        assert_eq!(
            registered_count(&mgr, "dense"),
            1,
            "one stale dense instance evicted to make room for moe"
        );
    }

    #[tokio::test]
    async fn execute_make_room_drain_completes_when_victim_goes_idle() {
        let mgr = makeroom_manager();
        let moe = mgr
            .model_configs
            .read()
            .unwrap()
            .get("moe")
            .cloned()
            .unwrap();
        let victim = register_inst(&mgr, "dense", vec![0], 1, Duration::ZERO);
        register_inst(&mgr, "dense", vec![1], 2, Duration::ZERO);

        let plan = mgr
            .plan_make_room(&moe, MakeRoomMode::DrainSurplus)
            .await
            .expect("a busy duplicate must be drainable");
        assert_eq!(plan.len(), 1);

        // Simulate the victim's in-flight request completing shortly after
        // the drain starts.
        let v = victim.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            v.inner().lock().unwrap().in_flight = 0;
        });

        tokio::time::timeout(Duration::from_secs(10), mgr.execute_make_room(&plan, 5))
            .await
            .expect("drain must complete well within the timeout");
        assert_eq!(
            registered_count(&mgr, "dense"),
            1,
            "drained victim reaped once idle"
        );
        assert!(
            mgr.spawn_feasible(&moe, &HashSet::new()).await,
            "moe fits after the drain"
        );
    }

    #[tokio::test]
    async fn execute_make_room_drain_timeout_leaves_instance_draining() {
        let mgr = makeroom_manager();
        let moe = mgr
            .model_configs
            .read()
            .unwrap()
            .get("moe")
            .cloned()
            .unwrap();
        let victim = register_inst(&mgr, "dense", vec![0], 1, Duration::ZERO); // never idles
        register_inst(&mgr, "dense", vec![1], 2, Duration::ZERO);

        let plan = mgr
            .plan_make_room(&moe, MakeRoomMode::DrainSurplus)
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(10), mgr.execute_make_room(&plan, 1))
            .await
            .expect("a stuck drain must be abandoned after drain_timeout");
        assert_eq!(
            victim.inner().lock().unwrap().state,
            InstanceState::Draining,
            "timed-out drain leaves the instance draining (reaped once idle)"
        );
        assert_eq!(registered_count(&mgr, "dense"), 2, "nothing reaped yet");
    }
}
