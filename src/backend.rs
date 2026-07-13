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
/// GPU selection is done via `--device` (comma-separated indices).  No
/// environment variables are set — the binary auto-detects its backend.
pub struct LlamaCppBackend;

impl Backend for LlamaCppBackend {
    fn gpu_args(&self, indices: &[usize]) -> Vec<String> {
        if indices.is_empty() {
            return Vec::new();
        }
        let list = indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        vec!["--device".into(), list]
    }

    fn gpu_env(&self, _indices: &[usize]) -> Vec<(String, String)> {
        Vec::new()
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
/// Sends SIGTERM, waits up to `drain_timeout`, then sends SIGKILL if the
/// process is still alive.  Returns once the process has exited.
pub async fn shutdown_child(child: &mut tokio::process::Child, drain_timeout: Duration) {
    let pid = child.id().unwrap_or(0);

    // Try graceful shutdown.
    if let Err(e) = child.start_kill() {
        tracing::warn!(pid, error = %e, "SIGTERM failed");
        return; // process already gone
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
            let _ = child.wait().await;
        }
    }
}