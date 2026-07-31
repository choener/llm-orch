// ── Backend trait ────────────────────────────────────────────────────────────
//
// Each backend variant (llama.cpp Vulkan, llama.cpp ROCm, sd.cpp, …) knows
// how to translate a set of GPU device indices into the CLI flags and/or
// environment variables its binary understands.  Process spawning and
// health-checking are shared helpers that sit *outside* the trait — only the
// backend-specific plumbing lives here.

/// What a backend implementation needs to provide.
pub trait Backend: Send + Sync {
    /// CLI arguments injected for GPU device selection.
    ///
    /// Example for llama.cpp: `["--device", "0,1"]`
    fn gpu_args(&self, indices: &[usize]) -> Vec<String>;

    /// Environment variables set for GPU device selection.
    ///
    /// Example for stable-diffusion-cpp:
    /// `[("GGML_VK_VISIBLE_DEVICES", "0,1")]`
    fn gpu_env(&self, indices: &[usize]) -> Vec<(String, String)>;

    /// Health-check URL for a backend listening on `port`.
    fn health_url(&self, port: u16) -> String {
        format!("http://127.0.0.1:{port}/health")
    }
}

// ── llama.cpp backend ────────────────────────────────────────────────────────

/// Device namespace a llama.cpp build uses for GPU selection.
///
/// Determines which environment variable device indices are pinned
/// through.  A model's kind follows its configured device pool
/// (`vulkan_devices` vs `cuda_devices`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceKind {
    /// Vulkan backend — pinned via `GGML_VK_VISIBLE_DEVICES`.
    #[default]
    Vulkan,
    /// CUDA backend — pinned via `CUDA_VISIBLE_DEVICES`.
    Cuda,
}

/// Backend for llama.cpp built with Vulkan, ROCm, or CUDA.
///
/// GPU selection uses `GGML_VK_VISIBLE_DEVICES` (Vulkan) or
/// `CUDA_VISIBLE_DEVICES` (CUDA) depending on [`DeviceKind`].  The indices
/// passed to `gpu_env` are logical backend device indices, not sysfs card
/// numbers.  Their order is preserved — it has semantics in llama.cpp
/// (tensor split order).
pub struct LlamaCppBackend {
    kind: DeviceKind,
}

impl LlamaCppBackend {
    pub fn new(kind: DeviceKind) -> Self {
        Self { kind }
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new(DeviceKind::Vulkan)
    }
}

impl Backend for LlamaCppBackend {
    fn gpu_args(&self, _indices: &[usize]) -> Vec<String> {
        Vec::new()
    }

    fn gpu_env(&self, indices: &[usize]) -> Vec<(String, String)> {
        if indices.is_empty() {
            return Vec::new();
        }
        let var = match self.kind {
            DeviceKind::Vulkan => "GGML_VK_VISIBLE_DEVICES",
            DeviceKind::Cuda => "CUDA_VISIBLE_DEVICES",
        };
        let list = indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        vec![(var.into(), list)]
    }
}

// ── Process helpers (shared) ─────────────────────────────────────────────────

use std::process::Stdio;
use std::time::Duration;
use reqwest::Client;
use crate::instance::InstanceHandle;

/// Spawn a backend subprocess with the given program, arguments, and environment.
///
/// `kill_on_drop(true)` ensures the child is reaped when the handle drops.
/// Additionally, `PR_SET_PDEATHSIG` is set on Unix so the child receives
/// SIGTERM if llm-orch itself is killed (including SIGKILL).
pub async fn spawn_process(
    prog: &str,
    args: &[String],
    envs: &[(String, String)],
) -> std::io::Result<tokio::process::Child> {
    let mut cmd = tokio::process::Command::new(prog);
    cmd.args(args);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.kill_on_drop(true);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());

    // Die with the parent — even if the parent is SIGKILL'd.
    #[cfg(unix)]
    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }

    cmd.spawn()
}

// ── Readiness detection ──────────────────────────────────────────────────────

/// Outcome of a readiness poll phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyOutcome {
    /// `/health` returned success — the instance was marked `Ready`.
    Ready,
    /// The child process exited before becoming healthy.
    ChildExited,
    /// Neither ready nor dead within the deadline.
    TimedOut,
}

/// Poll the instance's `/health` endpoint until it returns 200, the child
/// process exits, or the timeout expires.
///
/// Checking `try_wait` on every iteration fails the spawn fast when the
/// backend dies instantly (bad command, port collision, missing model
/// file) instead of waiting out the full timeout.
pub async fn poll_readiness(
    handle: &InstanceHandle,
    client: &Client,
    backend: &dyn Backend,
    timeout: Duration,
) -> ReadyOutcome {
    let url = {
        let inst = handle.inner().lock().unwrap();
        backend.health_url(inst.port)
    };
    let start = tokio::time::Instant::now();
    loop {
        if start.elapsed() >= timeout {
            return ReadyOutcome::TimedOut;
        }
        // Early exit: is the process already dead?
        {
            let mut inst = handle.inner().lock().unwrap();
            if let Some(child) = inst.child.as_mut() {
                if !matches!(child.try_wait(), Ok(None)) {
                    return ReadyOutcome::ChildExited;
                }
            }
        }
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                handle.inner().lock().unwrap().mark_ready();
                return ReadyOutcome::Ready;
            }
            _ => tokio::time::sleep(Duration::from_millis(200)).await,
        }
    }
}

// ── Shutdown ─────────────────────────────────────────────────────────────────

/// Gracefully shut down a backend child process.
///
/// Sends SIGTERM, waits up to `drain_timeout`, then escalates to SIGKILL
/// if the process is still alive.  Returns once the process has exited.
///
/// Note: `tokio::process::Child::start_kill()` is SIGKILL on Unix — the
/// graceful phase must go through `libc::kill` directly.
pub async fn shutdown_child(child: &mut tokio::process::Child, drain_timeout: Duration) {
    // child.id() returns None once the process has been reaped.  Never
    // default to 0 here: kill(0, …) would signal our *own* process group.
    let pid = match child.id() {
        Some(pid) => pid,
        None => return, // already reaped
    };

    // Graceful phase: SIGTERM on Unix; on other platforms start_kill is
    // the only option (it maps to TerminateProcess / SIGKILL).
    #[cfg(unix)]
    let term_sent = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) } == 0;
    #[cfg(not(unix))]
    let term_sent = child.start_kill().is_ok();

    if !term_sent {
        // Process likely already gone — reap it and bail out.
        let _ = tokio::time::timeout(drain_timeout, child.wait()).await;
        return;
    }

    let result = tokio::time::timeout(drain_timeout, child.wait()).await;
    match result {
        Ok(Ok(_status)) => {
            tracing::debug!(pid, "backend exited after SIGTERM");
        }
        _ => {
            // Timeout or error — force kill.
            tracing::warn!(pid, "backend did not exit after SIGTERM, sending SIGKILL");
            let _ = child.kill().await;
            // Wait with a timeout — stuck D-state processes may never exit.
            let _ = tokio::time::timeout(Duration::from_secs(10), child.wait()).await;
        }
    }
}
// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::Instance;

    #[test]
    fn gpu_env_pins_vulkan_devices() {
        let backend = LlamaCppBackend::new(DeviceKind::Vulkan);
        assert_eq!(
            backend.gpu_env(&[0, 2]),
            vec![("GGML_VK_VISIBLE_DEVICES".to_string(), "0,2".to_string())]
        );
    }

    #[test]
    fn gpu_env_pins_cuda_devices() {
        let backend = LlamaCppBackend::new(DeviceKind::Cuda);
        assert_eq!(
            backend.gpu_env(&[1, 0]),
            // Order is preserved — it has semantics in llama.cpp.
            vec![("CUDA_VISIBLE_DEVICES".to_string(), "1,0".to_string())]
        );
    }

    #[test]
    fn gpu_env_empty_pool_sets_nothing() {
        for kind in [DeviceKind::Vulkan, DeviceKind::Cuda] {
            assert!(LlamaCppBackend::new(kind).gpu_env(&[]).is_empty());
        }
    }

    #[tokio::test]
    async fn poll_readiness_detects_instant_child_exit() {
        // A child that exits immediately must fail readiness fast —
        // not after the full spawn timeout.
        let child = spawn_process("true", &[], &[]).await.unwrap();
        let mut inst = Instance::new("m", vec![], 9, None);
        inst.child = Some(child);
        let handle = InstanceHandle::new(inst);
        let client = Client::new();
        let backend = LlamaCppBackend::default();

        let t0 = std::time::Instant::now();
        let outcome =
            poll_readiness(&handle, &client, &backend, Duration::from_secs(30)).await;
        assert_eq!(outcome, ReadyOutcome::ChildExited);
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "instant child exit must be detected fast, took {:?}",
            t0.elapsed()
        );
    }

    #[tokio::test]
    async fn poll_readiness_times_out_for_unresponsive_backend() {
        // A live child that never serves /health must time out, not be
        // mistaken for a crash.
        let child = spawn_process("sleep", &["30".into()], &[]).await.unwrap();
        let mut inst = Instance::new("m", vec![], 9, None);
        inst.child = Some(child);
        let handle = InstanceHandle::new(inst);
        let client = Client::builder()
            .connect_timeout(Duration::from_millis(100))
            .build()
            .unwrap();
        let backend = LlamaCppBackend::default();

        let outcome =
            poll_readiness(&handle, &client, &backend, Duration::from_millis(600)).await;
        assert_eq!(outcome, ReadyOutcome::TimedOut);
    }
}
