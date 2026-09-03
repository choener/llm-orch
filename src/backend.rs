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
/// through and which device-name prefix the `--device` selector uses.
/// A model's kind follows its configured device pool
/// (`vulkan_devices`, `cuda_devices`, or `rocm_devices`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceKind {
    /// Vulkan backend — pinned via `GGML_VK_VISIBLE_DEVICES` + `--device VulkanN`.
    #[default]
    Vulkan,
    /// CUDA backend — pinned via `CUDA_VISIBLE_DEVICES` + `--device CUDAN`.
    Cuda,
    /// ROCm/HIP backend — pinned via `HIP_VISIBLE_DEVICES` + `--device ROCmN`.
    Rocm,
}

/// Backend for llama.cpp built with Vulkan, ROCm, or CUDA.
///
/// GPU selection combines a per-backend visibility environment variable
/// (`GGML_VK_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` /
/// `HIP_VISIBLE_DEVICES`, depending on [`DeviceKind`]) with an explicit
/// `--device <Vulkan|CUDA|ROCm><rank>[,…]` selector.  The env var alone
/// filters only its own backend's enumeration — a multi-backend
/// llama-server build (e.g. Vulkan + ROCm) still sees the other backend's
/// GPUs, and without `--device` llama.cpp row-splits an instance across
/// every enumerated accelerator.  The selector names *post-filter ranks*
/// (the env var renumbers the visible devices starting at 0, in list
/// order), never host indices — `GGML_VK_VISIBLE_DEVICES=1` makes that GPU
/// `Vulkan0`.  The indices passed to `gpu_env` are logical backend device
/// indices, not sysfs card numbers.  Their order is preserved — it has
/// semantics in llama.cpp (tensor split order).
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

/// Draft-model flags in a llama.cpp command line: declaring one of these
/// activates speculative decoding with a separate draft model that needs
/// its own device pinning (`--spec-draft-device`).
const DRAFT_MODEL_FLAGS: [&str; 6] = [
    "--spec-draft-model",
    "--model-draft",
    "-md",
    "--spec-draft-hf",
    "--hf-repo-draft",
    "-hfd",
];

/// Draft-device flags: an explicit user declaration that must never be
/// overridden by llm-orch's automatic injection (including `none`, which
/// keeps the draft on the CPU).
///
/// Config validation rejects these in llama.cpp command lines (llm-orch
/// owns draft-device placement); exposed for that check.
pub const DRAFT_DEVICE_FLAGS: [&str; 3] = ["--spec-draft-device", "--device-draft", "-devd"];

/// Main-device selector flags: llama.cpp's `--device` (alias `-dev`, env
/// `LLAMA_ARG_DEVICE`).  llm-orch owns device placement and appends the
/// selector itself; a user declaration is a config error.  Exposed for
/// config validation.
pub const MAIN_DEVICE_FLAGS: [&str; 2] = ["--device", "-dev"];

/// `--spec-type` values that create a separate draft-model context
/// (`ctx_dft`) with its own device selection.  These need the draft device
/// pin even without a `--spec-draft-model` flag — an internal MTP head
/// (Qwen3.6/3.8) lives in the main GGUF but is still loaded into a draft
/// context whose default device is llama.cpp's process-default backend,
/// *not* the main model's pinned device, so it can land on the wrong
/// backend (e.g. HIP) and OOM against the already-loaded model.
/// The `ngram-*` types are self-speculative (no draft model) and must not
/// trigger injection.
const DRAFT_SPEC_TYPES: [&str; 3] = ["draft-simple", "draft-eagle3", "draft-mtp"];

/// Draft offload-layer flags.  An explicit value of `0` pins the draft
/// model to the CPU; injecting a device flag would then be pointless.
const DRAFT_NGL_FLAGS: [&str; 4] = [
    "--spec-draft-ngl",
    "-ngld",
    "--gpu-layers-draft",
    "--n-gpu-layers-draft",
];

/// Backend device-name prefix used by llama.cpp's `--device` and
/// `--spec-draft-device` values (`Vulkan0`, `CUDA0`, `ROCm0`, …).
fn device_name_prefix(kind: DeviceKind) -> &'static str {
    match kind {
        DeviceKind::Vulkan => "Vulkan",
        DeviceKind::Cuda => "CUDA",
        DeviceKind::Rocm => "ROCm",
    }
}

/// Whether `args` contains any of `flags`, in `--flag value` or
/// `--flag=value` form (presence only; values are ignored).
pub fn declares_any_flag(args: &[String], flags: &[&str]) -> bool {
    args.iter().any(|arg| {
        let (flag, _) = arg.split_once('=').unwrap_or((arg.as_str(), ""));
        flags.contains(&flag)
    })
}

/// Whether a `--spec-type` declaration enables draft-model-based
/// speculation: the value (comma-separated list, `--spec-type X` or
/// `--spec-type=X`) contains any of [`DRAFT_SPEC_TYPES`].
fn spec_type_has_draft(args: &[String]) -> bool {
    let mut i = 0;
    while i < args.len() {
        let (flag, inline) = match args[i].split_once('=') {
            Some((f, v)) => (f, Some(v)),
            None => (args[i].as_str(), None),
        };
        if flag == "--spec-type" {
            let value = match inline {
                Some(v) => v,
                None => match args.get(i + 1) {
                    Some(v) => {
                        i += 1; // consume the value token
                        v.as_str()
                    }
                    None => return false,
                },
            };
            if value.split(',').any(|t| DRAFT_SPEC_TYPES.contains(&t)) {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Whether the command line's last draft-offload declaration is an
/// explicit `0` (draft fully on CPU).  Unparseable values (`auto`, …) and
/// absent declarations return `false`.
fn draft_offload_is_zero(args: &[String]) -> bool {
    let mut last: Option<i64> = None;
    let mut i = 0;
    while i < args.len() {
        let (flag, inline) = match args[i].split_once('=') {
            Some((f, v)) => (f, Some(v)),
            None => (args[i].as_str(), None),
        };
        if DRAFT_NGL_FLAGS.contains(&flag) {
            match inline {
                Some(v) => {
                    if let Ok(n) = v.parse::<i64>() {
                        last = Some(n);
                    }
                }
                None => {
                    if let Some(n) = args.get(i + 1).and_then(|v| v.parse::<i64>().ok()) {
                        last = Some(n);
                        i += 1; // consume the value token
                    }
                }
            }
        }
        i += 1;
    }
    last == Some(0)
}

impl LlamaCppBackend {
    /// Pin a speculative-decoding draft model to the same device(s) the
    /// main model instance is placed on.
    ///
    /// llama.cpp runs the draft on the CPU when `--spec-draft-device` is
    /// absent, silently wasting the GPUs the model is pinned to.  When the
    /// command declares a draft model (`--spec-draft-model`, `-md`, …),
    /// this appends `--spec-draft-device <VulkanN|CUDAN|ROCmN,…>` naming
    /// every device of the placement.  Device names use the post-filter
    /// visible namespace (ranks in `indices`, not host indices) — the
    /// visibility env vars renumber devices, exactly like the ROCm
    /// `--device ROCmN` handling above.
    ///
    /// No-op unless all of these hold:
    /// - the program is a llama.cpp program (basename starts with `llama-`),
    /// - the instance is GPU-pinned (`indices` non-empty; a CPU-only
    ///   model keeps its draft on the CPU), and
    /// - the command enables draft-model speculation: a draft-model flag
    ///   (`--spec-draft-model`, `-md`, …) or `--spec-type` with a draft-*
    ///   value (`draft-simple`, `draft-eagle3`, `draft-mtp` — including
    ///   internal MTP heads, which live in the main GGUF but still load
    ///   into a separate draft context), and
    /// - the command does not already control the draft device
    ///   (`--spec-draft-device`/`-devd`/`--device-draft`, incl. `none`)
    ///   or pin the draft to the CPU (`--spec-draft-ngl 0`).
    pub fn inject_spec_draft_device(
        program: &str,
        args: &mut Vec<String>,
        kind: DeviceKind,
        indices: &[usize],
    ) {
        let basename = program.rsplit('/').next().unwrap_or(program);
        if !basename.starts_with("llama-") || indices.is_empty() {
            return;
        }
        if !declares_any_flag(args, &DRAFT_MODEL_FLAGS) && !spec_type_has_draft(args) {
            return;
        }
        if declares_any_flag(args, &DRAFT_DEVICE_FLAGS) || draft_offload_is_zero(args) {
            return;
        }
        let prefix = device_name_prefix(kind);
        let devices = (0..indices.len())
            .map(|rank| format!("{prefix}{rank}"))
            .collect::<Vec<_>>()
            .join(",");
        args.extend(["--spec-draft-device".to_string(), devices]);
    }

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
        if indices.is_empty() {
            return Vec::new();
        }

        // The visibility env var restricts and renumbers the visible
        // devices.  Name the post-filter ranks in --device rather than the
        // original host indices: HIP_VISIBLE_DEVICES=2,4 exposes ROCm0 and
        // ROCm1; GGML_VK_VISIBLE_DEVICES=1 exposes Vulkan0.  Without this
        // selector llama.cpp would use every device its build enumerates
        // (including other backends' GPUs) and row-split across them.
        let devices = (0..indices.len())
            .map(|rank| format!("{}{rank}", device_name_prefix(self.kind)))
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

// ── stable-diffusion.cpp backend ─────────────────────────────────────────────

/// Backend for stable-diffusion.cpp (`sd-server`).
///
/// GPU selection uses only the visibility environment variable
/// (`GGML_VK_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`)
/// — sd.cpp does **not** have an equivalent of llama.cpp's `--device`
/// selector, so [`Backend::gpu_args`] returns an empty vector.
///
/// The health endpoint is the default `GET /health`.
pub struct StableDiffusionCppBackend {
    kind: DeviceKind,
}

impl StableDiffusionCppBackend {
    pub fn new(kind: DeviceKind) -> Self {
        Self { kind }
    }
}

impl Backend for StableDiffusionCppBackend {
    fn gpu_args(&self, _indices: &[usize]) -> Vec<String> {
        // sd.cpp does not have a `--device`-style selector — only the
        // visibility env var matters.  No CLI args to inject.
        Vec::new()
    }

    fn gpu_env(&self, indices: &[usize]) -> Vec<(String, String)> {
        if indices.is_empty() {
            return Vec::new();
        }
        // Same env vars as llama.cpp for the same device kinds.
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

    /// sd-server has no `/health` endpoint — use `GET /` which returns 200
    /// once the server is ready.
    fn health_url(&self, port: u16) -> String {
        format!("http://127.0.0.1:{port}/")
    }
}

// ── Container backend (podman) ─────────────────────────────────────────────

/// Backend for models running inside a container.
///
/// Container backends manage their own GPU selection through `--device`
/// flags in the podman command (e.g. `--device /dev/dri --device /dev/kfd`)
/// — no visibility env vars or device-selector flags are injected.  The
/// health check uses the default `GET /health` endpoint (vLLM, TGI, and
/// most inference servers serve this).
///
/// Container lifecycle uses `podman stop` for graceful shutdown rather
/// than POSIX signals on the podman client process (which would orphan
/// the container if the client exits first).
pub struct ContainerBackend;

impl Backend for ContainerBackend {
    fn gpu_args(&self, _indices: &[usize]) -> Vec<String> {
        // Container manages GPU access via `--device /dev/dri` etc. in
        // the podman command — no CLI selector is injected.
        Vec::new()
    }

    fn gpu_env(&self, _indices: &[usize]) -> Vec<(String, String)> {
        // Container discovers GPUs through the mounted device nodes;
        // visibility env vars would have no effect.
        Vec::new()
    }

    fn health_url(&self, port: u16) -> String {
        format!("http://127.0.0.1:{port}/health")
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
            if let Some(child) = inst.child.as_mut()
                && !matches!(child.try_wait(), Ok(None))
            {
                return ReadyOutcome::ChildExited;
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

/// Gracefully shut down a podman container by name.
///
/// Runs `podman stop --time <timeout_secs> <container_name>` to give the
/// container a chance to drain in-flight requests.  If `podman stop` fails
/// (container already gone, podman not available), the error is logged at
/// debug level — the container may already have been cleaned up.
///
/// After this call the podman *client* process still needs to be reaped
/// via `shutdown_child`, but the container itself is gone.
pub async fn shutdown_container(container_name: &str, drain_timeout: Duration) {
    let timeout_secs = drain_timeout.as_secs().max(1);
    let status = tokio::process::Command::new("podman")
        .args(["stop", "--time", &timeout_secs.to_string(), container_name])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await;
    match status {
        Ok(s) if s.success() => {
            tracing::debug!(%container_name, timeout_secs, "container stopped");
        }
        Ok(s) => {
            tracing::debug!(
                %container_name,
                code = s.code(),
                "podman stop returned non-zero — container may already be gone"
            );
        }
        Err(e) => {
            tracing::debug!(
                %container_name,
                error = %e,
                "podman stop failed — container may already be gone or podman not available"
            );
        }
    }
}

/// Inject auto-generated podman arguments into a container command.
///
/// Appends `--name <name>` (unique per instance) and `--rm` (auto-cleanup)
/// right after `podman run` in the argument list.  `--rm` is only injected
/// when not already present.
///
/// Returns the container name for storage on the `Instance`.
pub fn inject_container_args(args: &mut Vec<String>, name: &str) {
    // Find the index of "run" (the subcommand) — insert after it.
    // `podman run [--flags] image [args]`
    if let Some(run_pos) = args.iter().position(|a| a == "run") {
        // Inject --rm if not already present between "run" and the image.
        if !args.iter().any(|a| a == "--rm") {
            args.insert(run_pos + 1, "--rm".to_string());
        }
        // Inject --name after --rm (or after "run" if --rm was already there).
        let insert_at = if args.get(run_pos + 1).map(|s| s.as_str()) == Some("--rm") {
            run_pos + 2
        } else {
            run_pos + 1
        };
        args.insert(insert_at, "--name".to_string());
        args.insert(insert_at + 1, name.to_string());
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
    fn gpu_args_pin_devices_for_all_backends() {
        // --device names the post-filter visible ranks, not host indices:
        // the visibility env var renumbers devices from 0 in list order
        // (GGML_VK_VISIBLE_DEVICES=2,4 → Vulkan0,Vulkan1).
        for (kind, prefix, env) in [
            (DeviceKind::Vulkan, "Vulkan", "GGML_VK_VISIBLE_DEVICES"),
            (DeviceKind::Cuda, "CUDA", "CUDA_VISIBLE_DEVICES"),
            (DeviceKind::Rocm, "ROCm", "HIP_VISIBLE_DEVICES"),
        ] {
            let backend = LlamaCppBackend::new(kind);
            assert_eq!(
                backend.gpu_args(&[2, 4]),
                vec!["--device".to_string(), format!("{prefix}0,{prefix}1")],
                "{kind:?}: --device uses the post-filter visible ranks"
            );
            assert_eq!(
                backend.gpu_env(&[2, 4]),
                vec![(env.to_string(), "2,4".to_string())]
            );
            assert_eq!(
                backend.gpu_args(&[1]),
                vec!["--device".to_string(), format!("{prefix}0")],
                "{kind:?}: a single visible device is rank 0"
            );
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
    fn injects_spec_draft_device_on_draft_model_declaration() {
        let mut args = s(&["--model", "m.gguf", "--spec-draft-model", "draft.gguf"]);
        LlamaCppBackend::inject_spec_draft_device(
            "llama-server",
            &mut args,
            DeviceKind::Vulkan,
            &[0, 2],
        );
        assert!(
            args.ends_with(&[
                "--spec-draft-device".to_string(),
                "Vulkan0,Vulkan1".to_string()
            ]),
            "{args:?}"
        );
    }

    #[test]
    fn injects_spec_draft_device_uses_post_filter_ranks() {
        // Host indices 2,4 become visible ranks 0,1 — the names llama.cpp
        // resolves against must be the post-filter ones.
        for (kind, prefix, env) in [
            (DeviceKind::Vulkan, "Vulkan", "GGML_VK_VISIBLE_DEVICES"),
            (DeviceKind::Cuda, "CUDA", "CUDA_VISIBLE_DEVICES"),
            (DeviceKind::Rocm, "ROCm", "HIP_VISIBLE_DEVICES"),
        ] {
            let mut args = s(&["-md", "draft.gguf"]);
            LlamaCppBackend::inject_spec_draft_device("llama-server", &mut args, kind, &[2, 4]);
            let expected = format!("{prefix}0,{prefix}1");
            assert!(
                args.ends_with(&["--spec-draft-device".to_string(), expected]),
                "{kind:?}: {args:?}"
            );
            // The pinned env var uses the host indices; the flag the ranks.
            let backend = LlamaCppBackend::new(kind);
            assert_eq!(backend.gpu_env(&[2, 4]), vec![(env.into(), "2,4".into())]);
        }
    }

    #[test]
    fn injects_spec_draft_device_recognizes_hf_and_equals_forms() {
        for flag in ["--spec-draft-hf", "--hf-repo-draft", "-hfd"] {
            let mut args = s(&[flag, "user/draft"]);
            LlamaCppBackend::inject_spec_draft_device(
                "llama-server",
                &mut args,
                DeviceKind::Cuda,
                &[0],
            );
            assert!(
                args.ends_with(&["--spec-draft-device".to_string(), "CUDA0".to_string()]),
                "{flag}: {args:?}"
            );
        }
        let mut args = s(&["--spec-draft-model=draft.gguf"]);
        LlamaCppBackend::inject_spec_draft_device(
            "llama-server",
            &mut args,
            DeviceKind::Rocm,
            &[1],
        );
        assert!(
            args.ends_with(&["--spec-draft-device".to_string(), "ROCm0".to_string()]),
            "{args:?}"
        );
    }

    #[test]
    fn spec_draft_injection_respects_explicit_user_control() {
        // Explicit draft-device flags (incl. `none`) are never overridden.
        for flag in ["--spec-draft-device", "--device-draft", "-devd"] {
            let mut args = s(&["-md", "draft.gguf", flag, "Vulkan0"]);
            LlamaCppBackend::inject_spec_draft_device(
                "llama-server",
                &mut args,
                DeviceKind::Vulkan,
                &[0, 1],
            );
            assert_eq!(args, s(&["-md", "draft.gguf", flag, "Vulkan0"]), "{flag}");
        }
        let mut args = s(&["--spec-draft-model", "d.gguf", "--spec-draft-device=none"]);
        LlamaCppBackend::inject_spec_draft_device(
            "llama-server",
            &mut args,
            DeviceKind::Vulkan,
            &[0],
        );
        assert_eq!(
            args,
            s(&["--spec-draft-model", "d.gguf", "--spec-draft-device=none"])
        );

        // Explicit `--spec-draft-ngl 0` pins the draft to the CPU — nothing
        // to inject.  Any other value (or `auto`) still gets the device.
        for zero_form in [
            vec!["-ngld", "0"],
            vec!["--gpu-layers-draft", "0"],
            vec!["--spec-draft-ngl=0"],
        ] {
            let mut expected = s(&["-md", "d.gguf"]);
            expected.extend(zero_form.iter().map(|x| x.to_string()));
            let mut args = expected.clone();
            LlamaCppBackend::inject_spec_draft_device(
                "llama-server",
                &mut args,
                DeviceKind::Vulkan,
                &[0],
            );
            assert_eq!(args, expected, "{zero_form:?}");
        }
        for value in ["4", "auto"] {
            let mut args = s(&["-md", "d.gguf", "--spec-draft-ngl", value]);
            LlamaCppBackend::inject_spec_draft_device(
                "llama-server",
                &mut args,
                DeviceKind::Vulkan,
                &[0],
            );
            assert!(
                args.ends_with(&["--spec-draft-device".to_string(), "Vulkan0".to_string()]),
                "ngl {value}: {args:?}"
            );
        }
    }

    #[test]
    fn injects_spec_draft_device_for_spec_type_draft_variants() {
        // Internal MTP heads (Qwen3.6/3.8) load into their own draft
        // context: `--spec-type draft-mtp` alone must trigger the pin.
        for (flag, value) in [
            ("--spec-type", "draft-mtp"),
            ("--spec-type", "draft-simple"),
            ("--spec-type", "draft-eagle3"),
            ("--spec-type", "draft-mtp,ngram-cache"),
            ("--spec-type=draft-eagle3", ""),
        ] {
            let mut args = s(&["--model", "m.gguf"]);
            if value.is_empty() {
                args.push(flag.to_string());
            } else {
                args.extend([flag.to_string(), value.to_string()]);
            }
            LlamaCppBackend::inject_spec_draft_device(
                "llama-server",
                &mut args,
                DeviceKind::Vulkan,
                &[0],
            );
            assert!(
                args.ends_with(&["--spec-draft-device".to_string(), "Vulkan0".to_string()]),
                "{flag} {value}: {args:?}"
            );
        }
    }

    #[test]
    fn spec_type_ngram_never_injects() {
        // Self-speculative n-gram types have no draft model context.
        for value in [
            "ngram-simple",
            "ngram-cache",
            "ngram-map-k4v,ngram-mod",
            "none",
        ] {
            let mut args = s(&["--spec-type", value]);
            LlamaCppBackend::inject_spec_draft_device(
                "llama-server",
                &mut args,
                DeviceKind::Vulkan,
                &[0],
            );
            assert_eq!(args, s(&["--spec-type", value]), "{value}");
        }
    }

    #[test]
    fn spec_type_respects_explicit_draft_device_and_ngl_zero() {
        let mut args = s(&["--spec-type", "draft-mtp", "--spec-draft-device", "CUDA1"]);
        LlamaCppBackend::inject_spec_draft_device(
            "llama-server",
            &mut args,
            DeviceKind::Vulkan,
            &[0],
        );
        assert_eq!(
            args,
            s(&["--spec-type", "draft-mtp", "--spec-draft-device", "CUDA1"])
        );

        let mut args = s(&["--spec-type", "draft-mtp", "-ngld", "0"]);
        LlamaCppBackend::inject_spec_draft_device(
            "llama-server",
            &mut args,
            DeviceKind::Vulkan,
            &[0],
        );
        assert_eq!(args, s(&["--spec-type", "draft-mtp", "-ngld", "0"]));
    }

    #[test]
    fn spec_draft_injection_noops_without_draft_or_gpus() {
        // No draft model declared → nothing injected.
        let mut args = s(&["--model", "m.gguf"]);
        LlamaCppBackend::inject_spec_draft_device(
            "llama-server",
            &mut args,
            DeviceKind::Vulkan,
            &[0],
        );
        assert_eq!(args, s(&["--model", "m.gguf"]));

        // CPU-only instance (no pinned devices) → draft stays on CPU.
        let mut args = s(&["-md", "d.gguf"]);
        LlamaCppBackend::inject_spec_draft_device(
            "llama-server",
            &mut args,
            DeviceKind::Vulkan,
            &[],
        );
        assert_eq!(args, s(&["-md", "d.gguf"]));

        // Non-llama programs are never touched.
        let mut args = s(&["-md", "d.gguf"]);
        LlamaCppBackend::inject_spec_draft_device(
            "audiocpp_server",
            &mut args,
            DeviceKind::Vulkan,
            &[0],
        );
        assert_eq!(args, s(&["-md", "d.gguf"]));
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

    // ── Container backend tests ─────────────────────────────────────

    #[test]
    fn container_backend_returns_empty_args_and_env() {
        let backend = ContainerBackend;
        assert!(backend.gpu_args(&[]).is_empty());
        assert!(backend.gpu_args(&[0, 1]).is_empty());
        assert!(backend.gpu_env(&[]).is_empty());
        assert!(backend.gpu_env(&[0, 1]).is_empty());
    }

    #[test]
    fn container_backend_health_url() {
        let backend = ContainerBackend;
        assert_eq!(backend.health_url(8080), "http://127.0.0.1:8080/health");
    }

    // ── inject_container_args tests ─────────────────────────────────

    #[test]
    fn inject_container_args_adds_name_and_rm() {
        let mut args = s(&["run", "some-image", "--arg1"]);
        inject_container_args(&mut args, "my-container");
        assert_eq!(
            args,
            s(&["run", "--rm", "--name", "my-container", "some-image", "--arg1"])
        );
    }

    #[test]
    fn inject_container_args_skips_rm_when_already_present() {
        let mut args = s(&["run", "--rm", "some-image"]);
        inject_container_args(&mut args, "my-container");
        assert_eq!(
            args,
            s(&["run", "--rm", "--name", "my-container", "some-image"])
        );
    }

    #[test]
    fn inject_container_args_handles_no_run_subcommand() {
        let mut args = s(&["info"]);
        inject_container_args(&mut args, "my-container");
        // No "run" subcommand → no injection.
        assert_eq!(args, s(&["info"]));
    }
}
