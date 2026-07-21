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

/// Backend for llama.cpp built with Vulkan, ROCm, or CUDA.
///
/// GPU selection uses `GGML_VK_VISIBLE_DEVICES` (Vulkan) — other backends
/// may use different env vars or CLI flags.  The indices passed to `gpu_env`
/// are logical backend device indices, not sysfs card numbers.
pub struct LlamaCppBackend;

impl Backend for LlamaCppBackend {
    fn gpu_args(&self, _indices: &[usize]) -> Vec<String> {
        Vec::new()
    }

    fn gpu_env(&self, indices: &[usize]) -> Vec<(String, String)> {
        if indices.is_empty() {
            return Vec::new();
        }
        let list = indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        vec![("GGML_VK_VISIBLE_DEVICES".into(), list)]
    }
}

// ── Process helpers (shared) ─────────────────────────────────────────────────

use std::process::Stdio;
use std::time::Duration;
use reqwest::Client;
use crate::instance::InstanceHandle;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

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

/// Poll a backend instance's `/health` endpoint until it returns 200 or the
/// timeout expires.
pub async fn wait_until_ready(
    client: &Client,
    health_url: &str,
    timeout: Duration,
) -> bool {
    let start = tokio::time::Instant::now();
    loop {
        if start.elapsed() >= timeout {
            return false;
        }
        match client.get(health_url).send().await {
            Ok(resp) if resp.status().is_success() => return true,
            _ => tokio::time::sleep(Duration::from_millis(200)).await,
        }
    }
}

/// Poll the instance's health endpoint and update its state accordingly.
/// Returns `true` if the instance became ready.
pub async fn mark_instance_ready(
    handle: &InstanceHandle,
    client: &Client,
    backend: &dyn Backend,
    timeout: Duration,
) -> bool {
    let url = {
        let inst = handle.inner().lock().unwrap();
        backend.health_url(inst.port)
    };

    if wait_until_ready(client, &url, timeout).await {
        handle.inner().lock().unwrap().mark_ready();
        true
    } else {
        handle.inner().lock().unwrap().mark_failed();
        false
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