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