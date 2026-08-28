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
/// (`vulkan_devices`, `cuda_devices`, or `rocm_devices`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceKind {
    /// Vulkan backend — pinned via `GGML_VK_VISIBLE_DEVICES`.
    #[default]
    Vulkan,
    /// CUDA backend — pinned via `CUDA_VISIBLE_DEVICES`.
    Cuda,
    /// ROCm/HIP backend — pinned via `HIP_VISIBLE_DEVICES`.
    Rocm,
}

/// Backend for llama.cpp built with Vulkan, ROCm, or CUDA.
///
/// GPU selection uses `GGML_VK_VISIBLE_DEVICES` (Vulkan),
/// `CUDA_VISIBLE_DEVICES` (CUDA), or `HIP_VISIBLE_DEVICES` (ROCm)
/// depending on [`DeviceKind`].  ROCm additionally receives an explicit
/// `--device ROCmN` argument because a binary built with both ROCm and
/// Vulkan otherwise may select the Vulkan copy of the same AMD GPU.  The
/// indices passed to `gpu_env` are logical backend device indices, not sysfs
/// card numbers.  Their order is preserved — it has semantics in llama.cpp
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

impl LlamaCppBackend {
    /// Enforce CPU-only execution for a deviceless llama.cpp spawn.
    ///
    /// llama.cpp's default is to offload as many layers as fit, so a
    /// CPU-only model must explicitly pass `--n-gpu-layers 0`.  The flag
    /// is appended when the resolved command line declares no offload
    /// count; an existing declaration is left untouched (config
    /// validation already rejects non-zero values for CPU-only models, so
    /// a present value is either 0 or the config was never validated —
    /// either way appending a duplicate flag would be wrong).
    ///
    /// Only llama.cpp programs are touched — the program basename must
    /// start with `llama-`.  Other backends (e.g. `audiocpp_server`)
    /// select their compute device through their own interface and must
    /// never receive llama.cpp flags.
    pub fn enforce_cpu_offload(program: &str, args: &mut Vec<String>) {
        let basename = program.rsplit('/').next().unwrap_or(program);
        if !basename.starts_with("llama-") {
            return;
        }
        if llama_offload_decl(args) != OffloadDecl::Absent {
            return;
        }
        args.extend(["--n-gpu-layers".to_string(), "0".to_string()]);
    }
}

impl Backend for LlamaCppBackend {
    fn gpu_args(&self, indices: &[usize]) -> Vec<String> {
        if self.kind != DeviceKind::Rocm || indices.is_empty() {
            return Vec::new();
        }

        // HIP_VISIBLE_DEVICES restricts and renumbers the visible devices.
        // Use those post-filter ranks in --device rather than the original
        // host indices: HIP_VISIBLE_DEVICES=2,4 exposes ROCm0 and ROCm1.
        let devices = indices
            .iter()
            .enumerate()
            .map(|(rank, _)| format!("ROCm{rank}"))
            .collect::<Vec<_>>()
            .join(",");
        vec!["--device".to_string(), devices]
    }

    fn gpu_env(&self, indices: &[usize]) -> Vec<(String, String)> {
        if indices.is_empty() {
            return Vec::new();
        }
        let var = match self.kind {
            DeviceKind::Vulkan => "GGML_VK_VISIBLE_DEVICES",
            DeviceKind::Cuda => "CUDA_VISIBLE_DEVICES",
            DeviceKind::Rocm => "HIP_VISIBLE_DEVICES",
        };
        let list = indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        vec![(var.into(), list)]
    }
}

// ── CPU-offload enforcement (llama.cpp command lines) ──────────────────────

/// The outcome of scanning a resolved llama.cpp command line (program
/// excluded) for an explicit GPU-offload declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadDecl {
    /// No offload flag in the command line.
    Absent,
    /// Offload flag with a parsed layer count (`-1` = "as much as fits").
    Layers(i64),
    /// Offload flag present, but its value is missing or not an integer.
    Invalid,
}

/// Offload-layer flags recognized in a llama.cpp command line.
const OFFLOAD_FLAGS: [&str; 3] = ["--n-gpu-layers", "-ngl", "--n-offload"];

/// Scan `args` (the resolved command line without the program) for an
/// explicit GPU-offload declaration.
///
/// Recognizes `--n-gpu-layers`, `-ngl`, and `--n-offload` in both
/// `--flag value` and `--flag=value` forms.  A value that parses as an
/// integer (including negatives like `-1` = "as much as fits") is a value,
/// not another flag.  If the flag appears more than once, the last
/// declaration wins (llama.cpp's usual last-value-wins parsing).
pub fn llama_offload_decl(args: &[String]) -> OffloadDecl {
    let mut last: Option<OffloadDecl> = None;
    let mut i = 0;
    while i < args.len() {
        let (flag, inline) = match args[i].split_once('=') {
            Some((f, v)) => (f, Some(v)),
            None => (args[i].as_str(), None),
        };
        if OFFLOAD_FLAGS.contains(&flag) {
            match inline {
                Some(v) => {
                    last = Some(match v.parse::<i64>() {
                        Ok(n) => OffloadDecl::Layers(n),
                        Err(_) => OffloadDecl::Invalid,
                    });
                }
                None => match args.get(i + 1).and_then(|v| v.parse::<i64>().ok()) {
                    Some(n) => {
                        last = Some(OffloadDecl::Layers(n));
                        i += 1; // consume the value token
                    }
                    None => last = Some(OffloadDecl::Invalid),
                },
            }
        }
        i += 1;
    }
    last.unwrap_or(OffloadDecl::Absent)
}

// ── Process helpers (shared) ─────────────────────────────────────────────────

use crate::instance::InstanceHandle;
use reqwest::Client;
use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::debug;

/// Maximum number of recent backend output lines kept per instance.
/// Dumped at `warn!` level when the instance fails or crashes — this is
/// where llama.cpp's "CUDA init failed" style messages surface.
const OUTPUT_BUFFER_LINES: usize = 200;

/// Ring buffer of a backend process's recent stdout/stderr lines.
/// Shared between the reader tasks (writers) and the instance/scheduler
/// (readers on failure paths).
pub type OutputBuffer = Arc<Mutex<VecDeque<String>>>;

/// Snapshot of the buffered lines, oldest first.
pub fn output_lines(buf: &OutputBuffer) -> Vec<String> {
    buf.lock().unwrap().iter().cloned().collect()
}

/// Forward one stream's lines to `tracing::debug!` and the ring buffer.
fn pipe_output<S>(stream: S, stream_name: &'static str, model: String, port: u16, buf: OutputBuffer)
where
    S: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut lines = BufReader::new(stream).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            debug!(model = %model, port = port, stream = stream_name, "backend| {line}");
            let mut guard = buf.lock().unwrap();
            if guard.len() >= OUTPUT_BUFFER_LINES {
                guard.pop_front();
            }
            guard.push_back(format!("{stream_name}| {line}"));
        }
    });
}

/// Spawn a backend subprocess with the given program, arguments, and
/// environment.  Returns the child plus a ring buffer of its recent
/// stdout/stderr lines (also forwarded live at `debug!` level).
///
/// `kill_on_drop(true)` ensures the child is reaped when the handle drops.
/// Additionally, `PR_SET_PDEATHSIG` is set on Unix so the child receives
/// SIGTERM if llm-orch itself is killed (including SIGKILL).
pub async fn spawn_process(
    prog: &str,
    args: &[String],
    envs: &[(String, String)],
    model: &str,
    port: u16,
) -> std::io::Result<(tokio::process::Child, OutputBuffer)> {
    let mut cmd = tokio::process::Command::new(prog);
    cmd.args(args);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.kill_on_drop(true);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    // Die with the parent — even if the parent is SIGKILL'd.
    #[cfg(unix)]
    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }

    let mut child = cmd.spawn()?;
    let buf: OutputBuffer = Arc::new(Mutex::new(VecDeque::new()));
    if let Some(stdout) = child.stdout.take() {
        pipe_output(stdout, "stdout", model.to_owned(), port, Arc::clone(&buf));
    }
    if let Some(stderr) = child.stderr.take() {
        pipe_output(stderr, "stderr", model.to_owned(), port, Arc::clone(&buf));
    }
    Ok((child, buf))
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
    fn gpu_args_and_env_select_rocm_backend() {
        let backend = LlamaCppBackend::new(DeviceKind::Rocm);
        assert_eq!(
            backend.gpu_args(&[2, 4]),
            vec!["--device".to_string(), "ROCm0,ROCm1".to_string()],
            "--device uses HIP's post-filter visible ranks"
        );
        assert_eq!(
            backend.gpu_env(&[2, 4]),
            vec![("HIP_VISIBLE_DEVICES".to_string(), "2,4".to_string())]
        );
    }

    #[test]
    fn gpu_args_for_non_rocm_backends_are_empty() {
        for kind in [DeviceKind::Vulkan, DeviceKind::Cuda] {
            assert!(LlamaCppBackend::new(kind).gpu_args(&[0, 1]).is_empty());
        }
    }

    #[test]
    fn gpu_env_empty_pool_sets_nothing() {
        for kind in [DeviceKind::Vulkan, DeviceKind::Cuda, DeviceKind::Rocm] {
            let backend = LlamaCppBackend::new(kind);
            assert!(backend.gpu_args(&[]).is_empty());
            assert!(backend.gpu_env(&[]).is_empty());
        }
    }

    fn s(values: &[&str]) -> Vec<String> {
        values.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn offload_decl_scans_flag_forms() {
        use OffloadDecl::*;
        assert_eq!(llama_offload_decl(&s(&[])), Absent);
        assert_eq!(llama_offload_decl(&s(&["--model", "m.gguf"])), Absent);
        assert_eq!(
            llama_offload_decl(&s(&["--n-gpu-layers", "99"])),
            Layers(99)
        );
        assert_eq!(llama_offload_decl(&s(&["--n-gpu-layers", "0"])), Layers(0));
        assert_eq!(llama_offload_decl(&s(&["--n-gpu-layers=42"])), Layers(42));
        assert_eq!(llama_offload_decl(&s(&["-ngl", "99"])), Layers(99));
        // A negative value is a value, not another flag.
        assert_eq!(llama_offload_decl(&s(&["-ngl", "-1"])), Layers(-1));
        assert_eq!(llama_offload_decl(&s(&["--n-offload", "7"])), Layers(7));
        assert_eq!(llama_offload_decl(&s(&["--n-gpu-layers"])), Invalid);
        assert_eq!(llama_offload_decl(&s(&["--n-gpu-layers", "abc"])), Invalid);
        assert_eq!(
            llama_offload_decl(&s(&["--n-gpu-layers", "--model"])),
            Invalid
        );
        // Last declaration wins.
        assert_eq!(
            llama_offload_decl(&s(&["--n-gpu-layers", "99", "-ngl", "0"])),
            Layers(0)
        );
    }

    #[test]
    fn enforce_cpu_offload_appends_for_llama_programs() {
        for prog in ["llama-server", "/opt/llama/llama-server"] {
            let mut args = s(&["--model", "m.gguf"]);
            LlamaCppBackend::enforce_cpu_offload(prog, &mut args);
            assert!(
                args.ends_with(&["--n-gpu-layers".to_string(), "0".to_string()]),
                "{prog}: {args:?}"
            );
        }
    }

    #[test]
    fn enforce_cpu_offload_respects_existing_declaration() {
        // 0: redundant.  99: config validation should have rejected it —
        // either way, never produce a duplicate flag.
        for value in ["0", "99"] {
            let mut args = s(&["--n-gpu-layers", value]);
            LlamaCppBackend::enforce_cpu_offload("llama-server", &mut args);
            assert_eq!(args, s(&["--n-gpu-layers", value]));
        }
    }

    #[test]
    fn enforce_cpu_offload_leaves_other_backends_alone() {
        let mut args = s(&["--config", "x.json", "--backend", "cpu"]);
        LlamaCppBackend::enforce_cpu_offload("audiocpp_server", &mut args);
        assert_eq!(args, s(&["--config", "x.json", "--backend", "cpu"]));
    }

    #[tokio::test]
    async fn poll_readiness_detects_instant_child_exit() {
        // A child that exits immediately must fail readiness fast —
        // not after the full spawn timeout.
        let (child, _out) = spawn_process("true", &[], &[], "test", 0).await.unwrap();
        let mut inst = Instance::new("m", vec![], 9, None);
        inst.child = Some(child);
        let handle = InstanceHandle::new(inst);
        let client = Client::new();
        let backend = LlamaCppBackend::default();

        let t0 = std::time::Instant::now();
        let outcome = poll_readiness(&handle, &client, &backend, Duration::from_secs(30)).await;
        assert_eq!(outcome, ReadyOutcome::ChildExited);
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "instant child exit must be detected fast, took {:?}",
            t0.elapsed()
        );
    }

    #[tokio::test]
    async fn output_buffer_captures_stdout_and_stderr() {
        let (mut child, buf) = spawn_process(
            "sh",
            &["-c".into(), "echo out-line; echo err-line >&2".into()],
            &[],
            "test",
            0,
        )
        .await
        .unwrap();
        child.wait().await.unwrap();
        // Give the reader tasks a moment to drain the pipes.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let lines = output_lines(&buf);
        assert!(
            lines.iter().any(|l| l == "stdout| out-line"),
            "missing stdout line: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l == "stderr| err-line"),
            "missing stderr line: {lines:?}"
        );
    }

    #[tokio::test]
    async fn poll_readiness_times_out_for_unresponsive_backend() {
        // A live child that never serves /health must time out, not be
        // mistaken for a crash.
        let (child, _out) = spawn_process("sleep", &["30".into()], &[], "test", 0)
            .await
            .unwrap();
        let mut inst = Instance::new("m", vec![], 9, None);
        inst.child = Some(child);
        let handle = InstanceHandle::new(inst);
        let client = Client::builder()
            .connect_timeout(Duration::from_millis(100))
            .build()
            .unwrap();
        let backend = LlamaCppBackend::default();

        let outcome = poll_readiness(&handle, &client, &backend, Duration::from_millis(600)).await;
        assert_eq!(outcome, ReadyOutcome::TimedOut);
    }
}
