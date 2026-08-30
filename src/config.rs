use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur when loading or validating the configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid YAML: {0}")]
    Yaml(#[from] serde_yaml_ng::Error),

    #[error("config validation failed: {0}")]
    Validation(String),
}

// ---------------------------------------------------------------------------
// Top-level configuration
// ---------------------------------------------------------------------------

/// Top-level configuration, deserialized from YAML.
#[derive(Debug, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub apikeys_file: PathBuf,

    /// Named `{key}` → value fragments substituted into model `cmd` fields.
    #[serde(default)]
    pub cmd_aliases: HashMap<String, String>,

    #[serde(default)]
    pub models: Vec<ModelConfig>,

    #[serde(default)]
    pub aliases: Vec<AliasConfig>,

    /// Device mapping: logical backend indices → PCI slots.
    #[serde(default)]
    pub devices: Option<DevicesConfig>,

    /// GPU keep-alive configuration to prevent driver autosuspend.
    #[serde(default)]
    pub keep_alive: Option<KeepAliveConfig>,
}

impl Config {
    /// Load and deserialize configuration from a YAML file.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)?;
        let mut config: Self = serde_yaml_ng::from_str(&contents)?;
        // Resolve relative apikeys_file against the config file's directory.
        if config.apikeys_file.is_relative()
            && let Some(parent) = path.parent()
        {
            config.apikeys_file = parent.join(&config.apikeys_file);
        }
        config.validate()?;
        Ok(config)
    }

    /// Check the configuration for semantic errors beyond YAML parsing.
    fn validate(&self) -> Result<(), ConfigError> {
        // --- models ---
        let mut seen_names = HashSet::new();
        for m in &self.models {
            if m.name.is_empty() {
                return Err(ConfigError::Validation(
                    "model name must not be empty".into(),
                ));
            }
            if !seen_names.insert(&m.name) {
                return Err(ConfigError::Validation(format!(
                    "duplicate model name: {}",
                    m.name
                )));
            }
            if m.context_length == 0 {
                return Err(ConfigError::Validation(format!(
                    "model '{}': context_length must be > 0",
                    m.name
                )));
            }
            if m.cmd.trim().is_empty() {
                return Err(ConfigError::Validation(format!(
                    "model '{}': cmd must not be empty",
                    m.name
                )));
            }
            if m.max_instances == 0 {
                return Err(ConfigError::Validation(format!(
                    "model '{}': max_instances must be > 0 (0 would make the model unspawnable)",
                    m.name
                )));
            }
            if m.max_concurrent == 0 {
                return Err(ConfigError::Validation(format!(
                    "model '{}': max_concurrent must be > 0 (0 would reject every request)",
                    m.name
                )));
            }
            // llm-orch owns device placement: it pins the instance with
            // the namespace's visibility env var and appends the matching
            // `--device <rank>,…` (and `--spec-draft-device …`) selector
            // itself.  An operator-declared selector would silently lose
            // to that append and desync VRAM accounting from reality.
            if let Some(declared) = cmd_device_flag_decl(&self.cmd_aliases, m) {
                return Err(ConfigError::Validation(format!(
                    "model '{}': command declares {declared} — device placement is owned by llm-orch (vulkan_devices/cuda_devices/rocm_devices; keep a draft on the CPU with --spec-draft-ngl 0)",
                    m.name
                )));
            }
            if m.gpus == 0 {
                return Err(ConfigError::Validation(format!(
                    "model '{}': gpus must be >= 1",
                    m.name
                )));
            }
            let configured_pools = [
                !m.vulkan_devices.is_empty(),
                !m.cuda_devices.is_empty(),
                !m.rocm_devices.is_empty(),
            ]
            .into_iter()
            .filter(|configured| *configured)
            .count();
            if configured_pools > 1 {
                return Err(ConfigError::Validation(format!(
                    "model '{}': vulkan_devices, cuda_devices, and rocm_devices are mutually exclusive (a model uses exactly one device namespace)",
                    m.name
                )));
            }
            let (pool, ns) = if !m.cuda_devices.is_empty() {
                (&m.cuda_devices, "cuda_devices")
            } else if !m.rocm_devices.is_empty() {
                (&m.rocm_devices, "rocm_devices")
            } else {
                (&m.vulkan_devices, "vulkan_devices")
            };
            match pool {
                DevicePool::Pool(devices) if devices.is_empty() => {
                    // CPU-only model: spanning multiple devices is meaningless.
                    if m.gpus != 1 {
                        return Err(ConfigError::Validation(format!(
                            "model '{}': gpus ({}) exceeds device count (0) (CPU-only models must use gpus: 1)",
                            m.name, m.gpus
                        )));
                    }
                    // A CPU-only llama.cpp model must not request GPU offload in
                    // its command: llama.cpp's default is to offload as many
                    // layers as fit, so the contradiction would silently load
                    // onto GPU 0.  Rejected here, at config load, rather than
                    // guessing at intent at spawn time.
                    if let Some(declared) = cpu_cmd_offload_decl(&self.cmd_aliases, m) {
                        return Err(ConfigError::Validation(format!(
                            "model '{}': CPU-only (no device pool) but its command requests GPU offload ({declared}) — add vulkan_devices/cuda_devices/rocm_devices or drop the flag (use a CPU alias)",
                            m.name
                        )));
                    }
                }
                DevicePool::Pool(devices) => {
                    if m.gpus > devices.len() {
                        return Err(ConfigError::Validation(format!(
                            "model '{}': gpus ({}) exceeds {} count ({})",
                            m.name,
                            m.gpus,
                            ns,
                            devices.len()
                        )));
                    }
                }
                DevicePool::Placements(placements) => {
                    // Each placement's device count is its own span — `gpus`
                    // is meaningless here and must stay at its default of 1.
                    if m.gpus != 1 {
                        return Err(ConfigError::Validation(format!(
                            "model '{}': gpus ({}) must stay 1 when {} uses placement lists (each placement's device count is its span)",
                            m.name, m.gpus, ns
                        )));
                    }
                    for (i, placement) in placements.iter().enumerate() {
                        if placement.is_empty() {
                            return Err(ConfigError::Validation(format!(
                                "model '{}': {} placement {} is empty",
                                m.name, ns, i
                            )));
                        }
                        let mut seen = HashSet::new();
                        for idx in placement {
                            if !seen.insert(idx) {
                                return Err(ConfigError::Validation(format!(
                                    "model '{}': {} placement {} repeats device {}",
                                    m.name, ns, i, idx
                                )));
                            }
                        }
                    }
                }
            }
        }

        // --- aliases ---
        let model_names: HashSet<&str> = self.models.iter().map(|m| m.name.as_str()).collect();
        let alias_names: HashSet<&str> = self.aliases.iter().map(|a| a.name.as_str()).collect();
        let mut seen_aliases = HashSet::new();
        for a in &self.aliases {
            if a.name.is_empty() {
                return Err(ConfigError::Validation(
                    "alias name must not be empty".into(),
                ));
            }
            if !seen_aliases.insert(&a.name) {
                return Err(ConfigError::Validation(format!(
                    "duplicate alias name: {}",
                    a.name
                )));
            }
            if a.targets.is_empty() {
                return Err(ConfigError::Validation(format!(
                    "alias '{}': targets must not be empty",
                    a.name
                )));
            }
            if a.drain_timeout == 0 {
                return Err(ConfigError::Validation(format!(
                    "alias '{}': drain_timeout must be > 0",
                    a.name
                )));
            }
            for t in &a.targets {
                if t.is_empty() {
                    return Err(ConfigError::Validation(format!(
                        "alias '{}': target names must not be empty",
                        a.name
                    )));
                }
                // Chaining check first: an alias target that names another
                // alias would otherwise fail with the less helpful
                // "unknown model" error.
                if alias_names.contains(t.as_str()) {
                    return Err(ConfigError::Validation(format!(
                        "alias '{}' targets alias '{}' — aliases cannot chain",
                        a.name, t
                    )));
                }
                if !model_names.contains(t.as_str()) {
                    return Err(ConfigError::Validation(format!(
                        "alias '{}' targets unknown model '{}'",
                        a.name, t
                    )));
                }
            }
        }

        // --- cmd_aliases ---
        for reserved in ["port", "context_length", "max_concurrent", "name"] {
            if self.cmd_aliases.contains_key(reserved) {
                return Err(ConfigError::Validation(format!(
                    "cmd_aliases: '{}' is a reserved name",
                    reserved
                )));
            }
        }

        // --- queue and migration settings ---
        if self.server.queue_heartbeat_secs == 0 {
            return Err(ConfigError::Validation(
                "server.queue_heartbeat_secs must be > 0".into(),
            ));
        }

        // `0` is the documented unlimited value for both queue waiting and
        // migration transitions; unsigned deserialization rejects negative
        // values before semantic validation reaches this point.

        // --- port range ---
        match &self.server.port_range {
            PortRange::Range { start, end } => {
                if start > end {
                    return Err(ConfigError::Validation(format!(
                        "server.port_range: start ({}) must be <= end ({})",
                        start, end
                    )));
                }
            }
            // The string form means exactly "ephemeral" — anything else is
            // a typo that would silently enable ephemeral allocation.
            PortRange::EphemeralWord(word) => {
                if word != "ephemeral" {
                    return Err(ConfigError::Validation(format!(
                        "server.port_range: unknown value '{}' (expected a start/end range or \"ephemeral\")",
                        word
                    )));
                }
            }
        }

        // --- keep-alive ---
        if let Some(ref ka) = self.keep_alive
            && let Some(ref amd) = ka.amd
        {
            if amd.sleep == 0 {
                return Err(ConfigError::Validation(
                    "keep_alive.amd.sleep must be > 0 (0 would busy-loop the command)".into(),
                ));
            }
            if amd.cmd.trim().is_empty() {
                return Err(ConfigError::Validation(
                    "keep_alive.amd.cmd must not be empty".into(),
                ));
            }
        }

        // --- devices ---
        if let Some(ref devs) = self.devices {
            let pci_slots = list_pci_slots();
            let existing_slots: HashSet<&str> = pci_slots.iter().map(|s| s.as_str()).collect();
            for (idx, slot) in &devs.vulkan {
                if !existing_slots.contains(slot.as_str()) {
                    return Err(ConfigError::Validation(format!(
                        "devices.vulkan.{}: PCI slot '{}' not found in /sys/class/drm/card*/device/",
                        idx, slot
                    )));
                }
            }
            // Check for duplicate slots.
            let mut seen = HashSet::new();
            for slot in devs.vulkan.values() {
                if !seen.insert(slot) {
                    return Err(ConfigError::Validation(format!(
                        "devices.vulkan: PCI slot '{}' assigned to multiple indices",
                        slot
                    )));
                }
            }

            // --- cuda ---
            // NVIDIA GPUs do not register DRM cards, so slots are checked
            // against /sys/bus/pci/devices/ directly.
            let mut seen_cuda = HashSet::new();
            for (idx, cdev) in &devs.cuda {
                if !pci_slot_exists(&cdev.pci) {
                    return Err(ConfigError::Validation(format!(
                        "devices.cuda.{}: PCI slot '{}' not found in /sys/bus/pci/devices/",
                        idx, cdev.pci
                    )));
                }
                if !seen_cuda.insert(&cdev.pci) {
                    return Err(ConfigError::Validation(format!(
                        "devices.cuda: PCI slot '{}' assigned to multiple indices",
                        cdev.pci
                    )));
                }
                if devs.vulkan.values().any(|s| s == &cdev.pci) {
                    return Err(ConfigError::Validation(format!(
                        "devices.cuda.{}: PCI slot '{}' is also assigned in devices.vulkan",
                        idx, cdev.pci
                    )));
                }
            }

            // --- rocm ---
            // ROCm GPUs are AMD DRM devices.  Unlike Vulkan, which may use
            // other DRM vendors too, reject a ROCm mapping that does not
            // identify an AMD card so a typo cannot become an unplaceable
            // model later in the scheduler.
            let amd_slots = list_amd_pci_slots();
            let existing_amd_slots: HashSet<&str> = amd_slots.iter().map(|s| s.as_str()).collect();
            let mut seen_rocm = HashSet::new();
            for (idx, slot) in &devs.rocm {
                if !existing_amd_slots.contains(slot.as_str()) {
                    return Err(ConfigError::Validation(format!(
                        "devices.rocm.{}: PCI slot '{}' not found as an AMD GPU in /sys/class/drm/card*/device/",
                        idx, slot
                    )));
                }
                if !seen_rocm.insert(slot) {
                    return Err(ConfigError::Validation(format!(
                        "devices.rocm: PCI slot '{}' assigned to multiple indices",
                        slot
                    )));
                }
            }
        }

        // --- model vulkan_devices ---
        if let Some(ref devs) = self.devices {
            let valid_indices: HashSet<&usize> = devs.vulkan.keys().collect();
            for m in &self.models {
                for idx in m.vulkan_devices.iter_indices() {
                    if !valid_indices.contains(&idx) {
                        return Err(ConfigError::Validation(format!(
                            "model '{}': vulkan_device {} not defined in devices.vulkan",
                            m.name, idx
                        )));
                    }
                }
            }
        }

        // --- model cuda_devices ---
        {
            let valid_indices: HashSet<usize> = self
                .devices
                .as_ref()
                .map(|d| d.cuda.keys().copied().collect())
                .unwrap_or_default();
            for m in &self.models {
                for idx in m.cuda_devices.iter_indices() {
                    if !valid_indices.contains(&idx) {
                        return Err(ConfigError::Validation(format!(
                            "model '{}': cuda_device {} not defined in devices.cuda",
                            m.name, idx
                        )));
                    }
                }
            }
        }

        // --- model rocm_devices ---
        {
            let valid_indices: HashSet<usize> = self
                .devices
                .as_ref()
                .map(|d| d.rocm.keys().copied().collect())
                .unwrap_or_default();
            for m in &self.models {
                for idx in m.rocm_devices.iter_indices() {
                    if !valid_indices.contains(&idx) {
                        return Err(ConfigError::Validation(format!(
                            "model '{}': rocm_device {} not defined in devices.rocm",
                            m.name, idx
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Server settings
// ---------------------------------------------------------------------------

/// Policy used to order candidates when a make-room transition needs a
/// victim.  LRU is the only implemented policy for now; the enum leaves an
/// explicit validation point for future priority-aware policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionPolicy {
    #[default]
    Lru,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Address and port the HTTP server listens on, e.g. `"127.0.0.1:8080"`.
    #[serde(default = "default_listen")]
    pub listen: String,

    /// Port range for backend instance allocation.
    #[serde(default)]
    pub port_range: PortRange,

    /// Total timeout (seconds) for non-streaming backend requests.
    /// Non-streaming responses only materialize once generation is
    /// complete, so this must cover worst-case prefill + generation —
    /// keep it generous.  `0` disables the timeout.
    #[serde(default = "default_backend_total_timeout")]
    pub backend_total_timeout_secs: u64,

    /// Idle timeout (seconds) for streaming backend requests: maximum gap
    /// between SSE chunks (and until the first response headers) before
    /// the backend is considered hung.  Must cover worst-case prefill
    /// before the first token.  `0` disables the timeout.
    #[serde(default = "default_backend_idle_timeout")]
    pub backend_idle_timeout_secs: u64,

    /// Seconds to wait for in-flight HTTP requests to drain on shutdown
    /// before aborting the remaining connections.
    #[serde(default = "default_shutdown_drain_timeout")]
    pub shutdown_drain_timeout_secs: u64,

    /// Unconditional reap of idle instances after this many seconds without
    /// requests — the "deep sleep" bound.  The per-model `idle_ttl` is only
    /// a protection window for make-room eviction; this global timeout is
    /// what actually unloads idle models.
    /// See `docs/003-smart-handling.md`, decision 8.
    #[serde(default = "default_drain_idle_timeout")]
    pub drain_idle_timeout_secs: u64,

    /// Maximum time an accepted request may wait for a scheduler assignment.
    /// `0` disables the scheduler deadline; external clients/proxies may
    /// still impose their own timeout.
    #[serde(default = "default_queue_wait_timeout")]
    pub queue_wait_timeout_secs: u64,

    /// Interval for pre-assignment SSE comment heartbeats.
    /// Must be greater than zero so queued streaming clients do not receive
    /// an accidentally idle connection.
    #[serde(default = "default_queue_heartbeat")]
    pub queue_heartbeat_secs: u64,

    /// Maximum duration of a migration transition.  `0` means no transition
    /// deadline; active requests are never terminated by this setting.
    #[serde(default)]
    pub migration_timeout_secs: u64,

    /// Victim ordering policy for make-room migrations.
    #[serde(default)]
    pub eviction_policy: EvictionPolicy,
}

fn default_listen() -> String {
    "127.0.0.1:8080".into()
}
fn default_backend_total_timeout() -> u64 {
    900
}
fn default_backend_idle_timeout() -> u64 {
    300
}
fn default_shutdown_drain_timeout() -> u64 {
    60
}
fn default_drain_idle_timeout() -> u64 {
    3600
}
fn default_queue_wait_timeout() -> u64 {
    3600
}
fn default_queue_heartbeat() -> u64 {
    10
}

/// Port allocation strategy for backend instances.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PortRange {
    /// Explicit start-end range.
    Range { start: u16, end: u16 },
    /// Let the OS choose a free port for each instance.
    EphemeralWord(String),
}

impl Default for PortRange {
    fn default() -> Self {
        PortRange::Range {
            start: 9000,
            end: 9100,
        }
    }
}

// ---------------------------------------------------------------------------
// Model definitions
// ---------------------------------------------------------------------------

/// A model's device pool, in one of two YAML forms:
///
/// - flat pool: `[0, 1]` — each instance spans `gpus` devices chosen from
///   the list (legacy form), or
/// - explicit placements: `[[0],[1]]` — each inner list is one complete
///   device set; every instance uses exactly one of them.  `[[0],[1]]` is
///   single-GPU (either one is fine), `[[0,1]]` is dual-GPU (using both).
#[derive(Debug, Clone)]
pub enum DevicePool {
    /// Legacy flat pool: the scheduler picks `gpus` devices per instance.
    Pool(Vec<usize>),
    /// Explicit placement alternatives: each inner list is one complete
    /// device set for an instance (its own span).  `gpus` must stay 1.
    Placements(Vec<Vec<usize>>),
}

impl Default for DevicePool {
    fn default() -> Self {
        DevicePool::Pool(Vec::new())
    }
}

impl DevicePool {
    /// Whether no devices are configured at all (CPU-only model).
    pub fn is_empty(&self) -> bool {
        match self {
            DevicePool::Pool(devices) => devices.is_empty(),
            DevicePool::Placements(placements) => placements.is_empty(),
        }
    }

    /// The flat pool, when configured in the legacy form.
    pub fn as_pool(&self) -> Option<&Vec<usize>> {
        match self {
            DevicePool::Pool(devices) => Some(devices),
            DevicePool::Placements(_) => None,
        }
    }

    /// The explicit placement alternatives, when configured in that form.
    pub fn as_placements(&self) -> Option<&Vec<Vec<usize>>> {
        match self {
            DevicePool::Pool(_) => None,
            DevicePool::Placements(placements) => Some(placements),
        }
    }

    /// Every referenced device index, in configuration order (duplicates
    /// possible across placement alternatives).  Pools are tiny — an owned
    /// copy keeps the signature simple.
    pub fn iter_indices(&self) -> Vec<usize> {
        match self {
            DevicePool::Pool(devices) => devices.clone(),
            DevicePool::Placements(placements) => placements.iter().flatten().copied().collect(),
        }
    }

    /// Order-insensitive, deduplicated index set — used for fingerprinting.
    /// Placement structure does not change the spawned process, only the
    /// device set an instance ends up on does.
    pub fn canonical_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self.iter_indices();
        indices.sort_unstable();
        indices.dedup();
        indices
    }
}

impl<'de> Deserialize<'de> for DevicePool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_yaml_ng::Value::deserialize(deserializer)?;
        let seq = value
            .as_sequence()
            .ok_or_else(|| serde::de::Error::custom("expected a list of device indices"))?;
        match seq.first() {
            // `[]` — no devices (CPU-only).
            None => Ok(DevicePool::Pool(Vec::new())),
            // `[[0],[1]]` — explicit placement alternatives.
            Some(first) if first.is_sequence() => {
                let mut placements = Vec::with_capacity(seq.len());
                for item in seq {
                    let inner = item.as_sequence().ok_or_else(|| {
                        serde::de::Error::custom("placement must be a list of device indices")
                    })?;
                    let mut devices = Vec::with_capacity(inner.len());
                    for device in inner {
                        let index = device.as_u64().ok_or_else(|| {
                            serde::de::Error::custom("device index must be a non-negative integer")
                        })?;
                        devices.push(index as usize);
                    }
                    placements.push(devices);
                }
                Ok(DevicePool::Placements(placements))
            }
            // `[0, 1]` — flat pool (legacy form).
            Some(_) => {
                let mut devices = Vec::with_capacity(seq.len());
                for item in seq {
                    let index = item.as_u64().ok_or_else(|| {
                        serde::de::Error::custom("device index must be a non-negative integer")
                    })?;
                    devices.push(index as usize);
                }
                Ok(DevicePool::Pool(devices))
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Unique model name, used as the identifier in `/v1/models` and alias targets.
    pub name: String,

    /// Maximum context length (tokens).
    /// Meaningless for non-LLM backends (e.g. audio.cpp TTS/ASR models),
    /// which may omit it; only used for `{context_length}` cmd substitution
    /// and `/v1/models` metadata.
    #[serde(default = "default_context_length")]
    pub context_length: usize,

    /// Maximum number of backend instances of this model across all GPUs.
    #[serde(default = "default_max_instances")]
    pub max_instances: usize,

    /// Maximum concurrent requests a single instance can handle.
    /// Also substituted for `{max_concurrent}` in `cmd` at spawn time.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,

    /// Eviction priority — higher values are kept longer.
    /// Reserved for the deferred eviction policy (plan §Eviction);
    /// accepted in config today for forward compatibility.
    #[serde(default)]
    #[allow(dead_code)]
    pub priority: i32,

    /// Declared VRAM usage in MB (scheduler hint).
    #[serde(default)]
    pub vram: u64,

    /// Declared system RAM usage in MB (scheduler hint).
    /// Reserved for the deferred eviction policy (plan §Eviction).
    #[serde(default)]
    #[allow(dead_code)]
    pub ram: u64,

    /// Seconds of inactivity before the instance is unloaded.
    #[serde(default = "default_idle_ttl")]
    pub idle_ttl: u64,

    /// Maximum number of queued requests before returning 429.
    #[serde(default = "default_queue_depth")]
    pub queue_depth: usize,

    /// Subprocess command line.  Shell-style quoting / whitespace separation.
    /// `{alias_name}` placeholders are resolved from `cmd_aliases` at spawn
    /// time; `{port}` is replaced with the allocated port number,
    /// `{context_length}` with the model's declared `context_length`, and
    /// `{max_concurrent}` with the model's declared `max_concurrent`.
    pub cmd: String,

    /// Vulkan devices this model can be placed on (from `devices.vulkan`).
    /// Empty = CPU only (unless `cuda_devices` or `rocm_devices` is set).
    /// Mutually exclusive with the other device namespaces.
    ///
    /// Two forms are accepted:
    /// - flat pool: `[0, 1]` — each instance spans `gpus` devices chosen
    ///   from the pool (legacy form), or
    /// - explicit placements: `[[0],[1]]` — each inner list is one complete
    ///   device set; every instance uses exactly one of them (its own
    ///   span).  `[[0],[1]]` is single-GPU (either one), `[[0,1]]` is
    ///   dual-GPU (using both).  `gpus` must stay 1.
    #[serde(default)]
    pub vulkan_devices: DevicePool,

    /// CUDA devices this model can be placed on (from `devices.cuda`).
    /// When non-empty the model is a CUDA model: placement uses the CUDA
    /// pool and instances are pinned via `CUDA_VISIBLE_DEVICES`.
    /// Mutually exclusive with `vulkan_devices` and `rocm_devices`.
    /// Same two forms as `vulkan_devices`.
    #[serde(default)]
    pub cuda_devices: DevicePool,

    /// ROCm devices this model can be placed on (from `devices.rocm`).
    /// When non-empty the model is a ROCm model: placement uses the ROCm
    /// pool and instances are pinned via `HIP_VISIBLE_DEVICES` plus llama.cpp's
    /// explicit `--device ROCmN` selection.
    /// Mutually exclusive with `vulkan_devices` and `cuda_devices`.
    /// Same two forms as `vulkan_devices`.
    #[serde(default)]
    pub rocm_devices: DevicePool,

    /// Number of devices each instance of this model spans.
    /// The model is split evenly across them (llama.cpp's default tensor
    /// split over `GGML_VK_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` /
    /// `HIP_VISIBLE_DEVICES`). The `vram` value is reserved on **each**
    /// occupied GPU, so `vram: 20000,
    /// gpus: 2` reserves 2×20000 MB.  Asymmetric splits (`--tensor-split`
    /// in `cmd`) are possible; then `vram` is a conservative per-GPU
    /// reservation.  Must be `>= 1` and `<=` the device pool size.
    #[serde(default = "default_gpus")]
    pub gpus: usize,

    /// Optional path for per-request debug I/O logging (JSONL).
    /// When set, every request/response pair for this model is appended
    /// as one JSON object per line.
    #[serde(default)]
    pub debug_log: Option<PathBuf>,

    /// Optional load-based autoscaling configuration.
    /// When set and enabled, instance spawn/despawn decisions use the
    /// load EMA metrics instead of purely instantaneous saturation.
    #[serde(default)]
    pub autoscale: Option<AutoscaleConfig>,
}

impl ModelConfig {
    /// Whether any GPU device pool is configured (Vulkan, CUDA, or ROCm).
    pub fn has_device_pool(&self) -> bool {
        !self.vulkan_devices.is_empty()
            || !self.cuda_devices.is_empty()
            || !self.rocm_devices.is_empty()
    }

    /// Device namespace this model is placed on.  Validation guarantees
    /// that at most one GPU pool is configured; an empty pool means CPU-only.
    pub fn device_kind(&self) -> crate::backend::DeviceKind {
        if !self.cuda_devices.is_empty() {
            crate::backend::DeviceKind::Cuda
        } else if !self.rocm_devices.is_empty() {
            crate::backend::DeviceKind::Rocm
        } else {
            crate::backend::DeviceKind::Vulkan
        }
    }
}

/// Resolve the model's `cmd_aliases` and the runtime placeholders
/// (`{port}`, `{context_length}`, `{max_concurrent}`, `{name}`) in its
/// command string.  `{port}` is the instance's allocated port; config
/// validation calls this with a placeholder (0) since the port is only
/// known at spawn time and never affects flag detection.
pub fn resolve_model_cmd(
    cmd_aliases: &HashMap<String, String>,
    cfg: &ModelConfig,
    port: u16,
) -> String {
    let mut resolved = cfg.cmd.clone();
    for (key, value) in cmd_aliases.iter() {
        let placeholder = format!("{{{}}}", key);
        resolved = resolved.replace(&placeholder, value);
    }
    resolved
        .replace("{port}", &port.to_string())
        .replace("{context_length}", &cfg.context_length.to_string())
        .replace("{max_concurrent}", &cfg.max_concurrent.to_string())
        .replace("{name}", &cfg.name)
}

/// Detect an operator-declared device selector in a model's resolved
/// command line.  Returns a human-readable description of the offending
/// flag (e.g. `--device`) when the command runs a llama.cpp program and
/// declares the main-device selector (`--device`/`-dev`) or a draft-device
/// selector (`--spec-draft-device`/`--device-draft`/`-devd`); `None`
/// otherwise.
///
/// llm-orch owns device placement: it appends `--device <Prefix><rank>,…`
/// and `--spec-draft-device …` from the model's device pool.  An operator
/// declaration would either be silently overridden (llama.cpp is
/// last-value-wins and llm-orch appends last) or mislead about the actual
/// placement — either way VRAM accounting would diverge from reality, so
/// it is rejected at config load.  A CPU-resident draft is declared with
/// `--spec-draft-ngl 0`, not a device flag.
///
/// Non-llama.cpp programs (e.g. `audiocpp_server`) select their compute
/// device through their own interface and are not checked.
fn cmd_device_flag_decl(cmd_aliases: &HashMap<String, String>, m: &ModelConfig) -> Option<String> {
    let resolved = resolve_model_cmd(cmd_aliases, m, 0);
    let parts = shlex::split(&resolved)?;
    let program = parts.first()?;
    let basename = program.rsplit('/').next().unwrap_or(program);
    if !basename.starts_with("llama-") {
        return None;
    }
    let args = &parts[1..];
    use crate::backend::{DRAFT_DEVICE_FLAGS, MAIN_DEVICE_FLAGS, declares_any_flag};
    if let Some(flag) = MAIN_DEVICE_FLAGS
        .iter()
        .find(|flag| declares_any_flag(args, std::slice::from_ref(*flag)))
    {
        return Some((*flag).to_string());
    }
    if declares_any_flag(args, &DRAFT_DEVICE_FLAGS) {
        return Some(
            "a draft-device flag (--spec-draft-device / --device-draft / -devd)".to_string(),
        );
    }
    None
}

/// Detect a GPU-offload request in a CPU-only model's resolved command
/// line.  Returns a human-readable description of the offending
/// declaration (e.g. `--n-gpu-layers 99`) when the command runs a
/// llama.cpp program and declares a non-zero (or unparseable) offload
/// count; `None` otherwise.
///
/// Non-llama.cpp programs (e.g. `audiocpp_server`) select their compute
/// device through their own interface (`server.json`'s `backend` field)
/// and are not checked.
fn cpu_cmd_offload_decl(cmd_aliases: &HashMap<String, String>, m: &ModelConfig) -> Option<String> {
    let resolved = resolve_model_cmd(cmd_aliases, m, 0);
    let parts = shlex::split(&resolved)?;
    let program = parts.first()?;
    let basename = program.rsplit('/').next().unwrap_or(program);
    if !basename.starts_with("llama-") {
        return None;
    }
    use crate::backend::OffloadDecl;
    match crate::backend::llama_offload_decl(&parts[1..]) {
        OffloadDecl::Layers(n) if n != 0 => Some(format!("--n-gpu-layers {n}")),
        OffloadDecl::Invalid => {
            Some("an unparseable offload flag (--n-gpu-layers / -ngl / --n-offload)".to_string())
        }
        _ => None,
    }
}

fn default_context_length() -> usize {
    4096
}
fn default_max_instances() -> usize {
    1
}
fn default_gpus() -> usize {
    1
}
fn default_max_concurrent() -> usize {
    4
}
fn default_idle_ttl() -> u64 {
    300
}
fn default_queue_depth() -> usize {
    10
}

// ── Autoscale configuration ──────────────────────────────────────────────────

/// Load-based autoscaling for a model.
///
/// When enabled, instance spawn/despawn decisions use hysteresis on the
/// per-model load EMA metrics (`load_m5` for scale-up, `load_m15` for
/// scale-down) instead of purely instantaneous saturation + fixed idle TTL.
#[derive(Debug, Clone, Deserialize)]
pub struct AutoscaleConfig {
    /// Enable load-based autoscaling for this model.
    pub enabled: bool,

    /// Fraction of `max_concurrent × current_instances` above which a new
    /// instance is spawned.  Uses `load_m5` (5-minute EMA).
    /// Example: 0.7 means spawn when sustained load exceeds 70% of capacity.
    #[serde(default = "default_autoscale_up")]
    pub scale_up_at: f64,

    /// Fraction of `max_concurrent × (current_instances − 1)` below which
    /// an instance is despawned.  Uses `load_m15` (15-minute EMA).
    /// Example: 0.4 means despawn when the reduced instance count could
    /// handle the load at under 40% of their capacity.
    #[serde(default = "default_autoscale_down")]
    pub scale_down_at: f64,

    /// Minimum seconds between any scale action (up or down).
    #[serde(default = "default_autoscale_cooldown")]
    pub cooldown_secs: u64,
}

fn default_autoscale_up() -> f64 {
    0.7
}
fn default_autoscale_down() -> f64 {
    0.4
}
fn default_autoscale_cooldown() -> u64 {
    120
}

// ---------------------------------------------------------------------------
// Device mapping
// ---------------------------------------------------------------------------

/// Global device index → PCI slot mapping.
///
/// Each backend gets its own namespace (e.g. `vulkan`, `cuda`, `rocm`).
/// The indices here are what the backend visibility variables use — they
/// correspond to the backend's enumeration order, not sysfs card numbers.
#[derive(Debug, Clone, Deserialize)]
pub struct DevicesConfig {
    /// Vulkan device index → PCI slot name.
    #[serde(default)]
    pub vulkan: HashMap<usize, String>,

    /// ROCm device index → PCI slot name.
    #[serde(default)]
    pub rocm: HashMap<usize, String>,

    /// CUDA device index → device definition.
    #[serde(default)]
    pub cuda: HashMap<usize, CudaDeviceConfig>,

    /// Optional per-GPU VRAM limit in MB.
    /// Caps the usable VRAM below the sysfs-reported total, leaving
    /// headroom for driver overhead.  When unset, sysfs total is used.
    #[serde(default)]
    pub vram_limit_mb: HashMap<usize, u64>,
}

/// A single CUDA device definition.
#[derive(Debug, Clone, Deserialize)]
pub struct CudaDeviceConfig {
    /// PCI slot name (e.g. `"0000:65:00.0"`).  Validated against
    /// `/sys/bus/pci/devices/` — NVIDIA GPUs do not register DRM cards,
    /// so the Vulkan `/sys/class/drm` check does not apply.
    pub pci: String,

    /// Static VRAM total in MB.  Doubles as a capacity cap (like
    /// `vram_limit_mb`) and as the fallback total when `nvidia-smi`
    /// metrics are unavailable.  When unset and no metrics are available,
    /// the device cannot satisfy VRAM-aware placement.
    #[serde(default)]
    pub vram_mb: Option<u64>,
}

/// List PCI slot names from `/sys/class/drm/card*/device/uevent`.
fn list_pci_slots() -> Vec<String> {
    list_drm_pci_slots(false)
}

/// List PCI slot names for AMD DRM devices.
fn list_amd_pci_slots() -> Vec<String> {
    list_drm_pci_slots(true)
}

fn list_drm_pci_slots(amd_only: bool) -> Vec<String> {
    let mut slots = Vec::new();
    for i in 0..16 {
        let device = std::path::PathBuf::from(format!("/sys/class/drm/card{}/device", i));
        if amd_only
            && std::fs::read_to_string(device.join("vendor"))
                .map(|vendor| vendor.trim() != "0x1002")
                .unwrap_or(true)
        {
            continue;
        }
        let uevent = device.join("uevent");
        if let Ok(contents) = std::fs::read_to_string(&uevent) {
            for line in contents.lines() {
                if let Some(slot) = line.strip_prefix("PCI_SLOT_NAME=") {
                    slots.push(slot.trim().to_string());
                }
            }
        }
    }
    slots
}

/// Check whether a PCI slot name exists under `/sys/bus/pci/devices/`.
fn pci_slot_exists(slot: &str) -> bool {
    std::path::Path::new("/sys/bus/pci/devices")
        .join(slot)
        .exists()
}

// ---------------------------------------------------------------------------
// Keep-alive
// ---------------------------------------------------------------------------

/// Per-GPU-type keep-alive configuration.
///
/// When at least one model instance is running on a GPU, the configured
/// command is invoked periodically to prevent the kernel driver from
/// auto-suspending the device.
#[derive(Debug, Clone, Deserialize)]
pub struct KeepAliveConfig {
    /// AMD (amdgpu) keep-alive.
    #[serde(default)]
    pub amd: Option<GpuKeepAlive>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GpuKeepAlive {
    /// Command to run.  `{index}` is substituted with the GPU index.
    pub cmd: String,
    /// Seconds between invocations.
    #[serde(default = "default_keepalive_sleep")]
    pub sleep: u64,
}

fn default_keepalive_sleep() -> u64 {
    5
}

// ---------------------------------------------------------------------------
// Aliases
// ---------------------------------------------------------------------------

/// How a multi-target alias picks among its candidate models.
/// See `docs/003-smart-handling.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AliasPolicy {
    /// Serve from any already-loaded candidate (in target order) before
    /// spawning anything; fall back to loading, then to queueing.
    #[default]
    PreferLoaded,
    /// Try candidates strictly in order: warm slot, spawn, make-room —
    /// only advance to the next candidate when all three fail.
    PreferOrder,
}

/// What an alias may evict to make room for a candidate that does not fit.
/// See `docs/003-smart-handling.md`, victim classes 1–3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MakeRoomMode {
    /// Never evict; fail/queue as before.
    None,
    /// Evict idle instances past their model's `idle_ttl` protection window.
    #[default]
    EvictIdle,
    /// Additionally drain busy *surplus duplicate* instances (never the last
    /// serving instance of a model), bounded by the alias's `drain_timeout`.
    DrainSurplus,
    /// Migrate an eligible victim to another GPU before loading the requested
    /// model.  This is an explicit opt-in and is handled by the migration
    /// coordinator rather than the legacy eviction path.
    Migrate,
}

impl MakeRoomMode {
    /// Whether this mode uses the existing destructive/legacy make-room
    /// implementation.  Migration must never silently fall through to it.
    pub fn is_legacy_eviction(self) -> bool {
        matches!(self, Self::EvictIdle | Self::DrainSurplus)
    }

    /// Whether this mode requests the transactional migration path.
    pub fn is_migration(self) -> bool {
        self == Self::Migrate
    }
}

/// Deserialize a field that accepts either a single string or a sequence of
/// strings, normalizing to `Vec<String>`.
fn string_or_seq<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    Ok(match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(s) => vec![s],
        OneOrMany::Many(v) => v,
    })
}

#[derive(Debug, Clone, Deserialize)]
pub struct AliasConfig {
    /// The alias name — appears in `/v1/models` and can be used in API requests.
    pub name: String,

    /// Ordered candidate model names (each must match a `ModelConfig::name`).
    /// Later entries are spillover for when an earlier entry cannot be
    /// served or loaded.  The legacy scalar form `target: <name>` is
    /// accepted as a one-element list.
    #[serde(alias = "target", deserialize_with = "string_or_seq")]
    pub targets: Vec<String>,

    /// Candidate selection policy.
    #[serde(default)]
    pub policy: AliasPolicy,

    /// What may be evicted to make room for a candidate that does not fit.
    #[serde(default)]
    pub make_room: MakeRoomMode,

    /// Bound (seconds) on waiting for a busy surplus duplicate to drain
    /// before giving up on a make-room plan.  Only relevant with
    /// `make_room: drain_surplus`.
    #[serde(default = "default_drain_timeout")]
    pub drain_timeout: u64,
}

fn default_drain_timeout() -> u64 {
    300
}
// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"
server: {}
apikeys_file: apikeys.txt
models:
  - name: m
    context_length: 4096
    cmd: "sleep 3600"
"#;

    fn parse(yaml: &str) -> Result<Config, ConfigError> {
        let cfg: Config = serde_yaml_ng::from_str(yaml).unwrap();
        cfg.validate()?;
        Ok(cfg)
    }

    fn expect_validation_error(yaml: &str, needle: &str) {
        match parse(yaml) {
            Err(ConfigError::Validation(msg)) => assert!(
                msg.contains(needle),
                "error '{msg}' must contain '{needle}'"
            ),
            other => panic!("expected validation error containing '{needle}', got {other:?}"),
        }
    }

    #[test]
    fn base_config_is_valid() {
        parse(BASE).unwrap();
    }

    #[test]
    fn context_length_defaults_when_omitted() {
        // Audio (non-LLM) models may omit context_length entirely.
        let cfg = parse(BASE).unwrap();
        assert_eq!(cfg.models[0].context_length, 4096);
    }

    #[test]
    fn rejects_zero_max_instances() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    max_instances: 0",
        );
        expect_validation_error(&yaml, "max_instances must be > 0");
    }

    #[test]
    fn rejects_zero_max_concurrent() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    max_concurrent: 0",
        );
        expect_validation_error(&yaml, "max_concurrent must be > 0");
    }

    #[test]
    fn rejects_unknown_port_range_word() {
        let yaml = BASE.replace("server: {}", "server:\n  port_range: \"ephemral\"");
        expect_validation_error(&yaml, "unknown value 'ephemral'");
    }

    #[test]
    fn accepts_ephemeral_port_range() {
        let yaml = BASE.replace("server: {}", "server:\n  port_range: \"ephemeral\"");
        parse(&yaml).unwrap();
    }

    #[test]
    fn rejects_zero_keepalive_sleep() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\nkeep_alive:\n  amd:\n    cmd: \"true\"\n    sleep: 0",
        );
        expect_validation_error(&yaml, "keep_alive.amd.sleep must be > 0");
    }

    #[test]
    fn rejects_zero_gpus() {
        let yaml = BASE.replace("cmd: \"sleep 3600\"", "cmd: \"sleep 3600\"\n    gpus: 0");
        expect_validation_error(&yaml, "gpus must be >= 1");
    }

    #[test]
    fn rejects_gpus_exceeding_device_pool() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    gpus: 3\n    vulkan_devices: [0, 1]",
        );
        expect_validation_error(&yaml, "gpus (3) exceeds vulkan_devices count (2)");
    }

    #[test]
    fn rejects_multi_gpu_for_cpu_model() {
        let yaml = BASE.replace("cmd: \"sleep 3600\"", "cmd: \"sleep 3600\"\n    gpus: 2");
        expect_validation_error(&yaml, "CPU-only models must use gpus: 1");
    }

    #[test]
    fn rejects_cpu_llama_model_requesting_offload() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"llama-server --model m.gguf --n-gpu-layers 99\"",
        );
        expect_validation_error(
            &yaml,
            "CPU-only (no device pool) but its command requests GPU offload (--n-gpu-layers 99)",
        );
    }

    #[test]
    fn rejects_cpu_llama_model_auto_offload() {
        // -1 = "offload as much as fits" — a conflict for a CPU-only model.
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"llama-server --model m.gguf -ngl -1\"",
        );
        expect_validation_error(&yaml, "requests GPU offload (--n-gpu-layers -1)");
    }

    #[test]
    fn rejects_cpu_llama_model_offload_equals_form() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"llama-server --model m.gguf --n-gpu-layers=99\"",
        );
        expect_validation_error(&yaml, "requests GPU offload (--n-gpu-layers 99)");
    }

    #[test]
    fn rejects_cpu_llama_model_offload_hidden_in_alias() {
        // The offload flag may hide inside a cmd alias.
        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
cmd_aliases:
  llama: |
    llama-server
      --port {port}
      --n-gpu-layers 99
models:
  - name: m
    context_length: 4096
    cmd: "{llama} --model m.gguf"
"#;
        expect_validation_error(yaml, "requests GPU offload (--n-gpu-layers 99)");
    }

    #[test]
    fn rejects_llama_model_declaring_main_device_flag() {
        // llm-orch appends `--device` from the placement itself; an
        // operator declaration would silently lose to that append.
        for (flag, needle) in [
            ("--device Vulkan0", "--device"),
            ("--device=Vulkan0", "--device"),
            ("-dev ROCm0", "-dev"),
            ("-dev=ROCm0", "-dev"),
        ] {
            let yaml = BASE.replace(
                "cmd: \"sleep 3600\"",
                &format!("cmd: \"llama-server --model m.gguf {flag}\""),
            );
            expect_validation_error(&yaml, &format!("declares {needle}"));
        }
    }

    #[test]
    fn rejects_llama_model_declaring_draft_device_flag() {
        // Same ownership rule for the draft model's device.  `--spec-draft-ngl 0`
        // remains the supported way to keep a draft on the CPU.
        for flag in [
            "--spec-draft-device none",
            "--spec-draft-device=Vulkan0",
            "--device-draft CUDA0",
            "-devd CUDA0",
        ] {
            let yaml = BASE.replace(
                "cmd: \"sleep 3600\"",
                &format!("cmd: \"llama-server --model m.gguf {flag}\""),
            );
            expect_validation_error(&yaml, "declares a draft-device flag");
        }
    }

    #[test]
    fn rejects_device_flag_hidden_in_alias() {
        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
cmd_aliases:
  llama: |
    llama-server
      --port {port}
      --device Vulkan0
models:
  - name: m
    context_length: 4096
    cmd: "{llama} --model m.gguf"
"#;
        expect_validation_error(yaml, "declares --device");
    }

    #[test]
    fn device_flag_check_applies_to_gpu_models_too() {
        // Not just a CPU-only concern: a pooled model declaring its own
        // `--device` would desync VRAM accounting from the actual spawn.
        let yaml = r#"
server: {}
apikeys_file: apikeys.txt
devices:
  cuda:
    0:
      pci: "0000:01:00.0"
      vram_mb: 48000
models:
  - name: m
    context_length: 4096
    cmd: "llama-server --model m.gguf --device CUDA0"
    cuda_devices: [0]
"#;
        // The PCI slot may not exist on this host; the device-flag error
        // must fire regardless (it precedes the devices-section check).
        match parse(yaml) {
            Err(ConfigError::Validation(msg)) => assert!(
                msg.contains("declares --device"),
                "unexpected validation error: {msg}"
            ),
            other => panic!("expected device-flag validation error, got {other:?}"),
        }
    }

    #[test]
    fn accepts_non_llama_program_with_device_like_args() {
        // Non-llama.cpp programs select their compute device through their
        // own interface; llm-orch must not police their flags.
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"audiocpp_server --config server.json --device cuda:0\"",
        );
        parse(&yaml).unwrap();
    }

    #[test]
    fn accepts_llama_model_with_draft_ngl_zero() {
        // The supported way to keep a draft model on the CPU.
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"llama-server --model m.gguf --spec-draft-model d.gguf --spec-draft-ngl 0\"",
        );
        parse(&yaml).unwrap();
    }

    #[test]
    fn accepts_cpu_llama_model_explicit_zero_offload() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"llama-server --model m.gguf --n-gpu-layers 0\"",
        );
        parse(&yaml).unwrap();
    }

    #[test]
    fn accepts_cpu_llama_model_without_offload_flag() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"llama-server --model m.gguf\"",
        );
        parse(&yaml).unwrap();
    }

    #[test]
    fn accepts_cpu_audiocpp_model() {
        // Non-llama.cpp backends select their device through their own
        // interface — the offload check must not apply to them.
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"audiocpp_server --config x.json --backend vulkan\"",
        );
        parse(&yaml).unwrap();
    }

    #[test]
    fn accepts_multi_gpu_model() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    gpus: 2\n    vulkan_devices: [0, 1]",
        );
        parse(&yaml).unwrap();
    }

    // ── explicit placement lists (nested form) ────────────────────────

    #[test]
    fn parses_flat_and_nested_device_pools() {
        // Legacy flat pool.
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: m\ncontext_length: 4096\ncmd: \"sleep 1\"\nvulkan_devices: [0, 1]",
        )
        .unwrap();
        assert!(matches!(cfg.vulkan_devices, DevicePool::Pool(ref d) if d == &vec![0, 1]));

        // Explicit single-GPU alternatives.
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: m\ncontext_length: 4096\ncmd: \"sleep 1\"\nvulkan_devices: [[0],[1]]",
        )
        .unwrap();
        assert!(
            matches!(cfg.vulkan_devices, DevicePool::Placements(ref p) if p == &vec![vec![0], vec![1]])
        );

        // Explicit dual-GPU placement.
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: m\ncontext_length: 4096\ncmd: \"sleep 1\"\nvulkan_devices: [[0,1]]",
        )
        .unwrap();
        assert!(
            matches!(cfg.vulkan_devices, DevicePool::Placements(ref p) if p == &vec![vec![0, 1]])
        );

        // Empty list = no devices (CPU-only).
        let cfg: ModelConfig = serde_yaml_ng::from_str(
            "name: m\ncontext_length: 4096\ncmd: \"sleep 1\"\nvulkan_devices: []",
        )
        .unwrap();
        assert!(cfg.vulkan_devices.is_empty());
        assert!(matches!(cfg.vulkan_devices, DevicePool::Pool(_)));
    }

    #[test]
    fn accepts_nested_single_and_dual_gpu_placements() {
        let slots = real_drm_slots();
        if slots.len() < 2 {
            return;
        }
        for placements in ["[[0],[1]]", "[[0,1]]"] {
            let yaml = BASE
                .replace(
                    "apikeys_file: apikeys.txt",
                    &format!(
                        "apikeys_file: apikeys.txt\ndevices:\n  vulkan:\n    0: \"{}\"\n    1: \"{}\"",
                        slots[0], slots[1]
                    ),
                )
                .replace(
                    "cmd: \"sleep 3600\"",
                    &format!("cmd: \"sleep 3600\"\n    vulkan_devices: {placements}"),
                );
            parse(&yaml).unwrap();
        }
    }

    #[test]
    fn rejects_gpus_with_nested_placements() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    gpus: 2\n    vulkan_devices: [[0],[1]]",
        );
        expect_validation_error(&yaml, "gpus (2) must stay 1");
    }

    #[test]
    fn rejects_empty_and_duplicate_placements() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    vulkan_devices: [[], [0]]",
        );
        expect_validation_error(&yaml, "placement 0 is empty");

        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    vulkan_devices: [[0, 0]]",
        );
        expect_validation_error(&yaml, "placement 0 repeats device 0");
    }

    #[test]
    fn rejects_undefined_device_in_placement() {
        let slots = real_drm_slots();
        if slots.is_empty() {
            return;
        }
        let yaml = BASE
            .replace(
                "apikeys_file: apikeys.txt",
                &format!(
                    "apikeys_file: apikeys.txt\ndevices:\n  vulkan:\n    0: \"{}\"",
                    slots[0]
                ),
            )
            .replace(
                "cmd: \"sleep 3600\"",
                "cmd: \"sleep 3600\"\n    vulkan_devices: [[0],[1]]",
            );
        expect_validation_error(&yaml, "vulkan_device 1 not defined in devices.vulkan");
    }

    #[test]
    fn rejects_reserved_cmd_alias_names() {
        for reserved in ["port", "context_length", "max_concurrent", "name"] {
            let yaml = BASE.replace(
                "apikeys_file: apikeys.txt",
                &format!("apikeys_file: apikeys.txt\ncmd_aliases:\n  {reserved}: \"x\""),
            );
            expect_validation_error(&yaml, &format!("'{reserved}' is a reserved name"));
        }
    }

    // ── CUDA device namespace ─────────────────────────────────────────

    /// Real PCI slot names from /sys/bus/pci/devices, for tests that need
    /// slots passing the existence check.  Every Linux host has some.
    fn real_pci_slots(n: usize) -> Vec<String> {
        let mut slots: Vec<String> = std::fs::read_dir("/sys/bus/pci/devices")
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        slots.sort();
        slots.truncate(n);
        slots
    }

    /// Real DRM PCI slots (what devices.vulkan validates against).
    fn real_drm_slots() -> Vec<String> {
        list_pci_slots()
    }

    /// Real AMD DRM PCI slots (what devices.rocm validates against).
    fn real_amd_slots() -> Vec<String> {
        list_amd_pci_slots()
    }

    #[test]
    fn rejects_model_with_both_device_namespaces() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    vulkan_devices: [0]\n    cuda_devices: [0]",
        );
        expect_validation_error(&yaml, "mutually exclusive");
    }

    #[test]
    fn rejects_model_with_vulkan_and_rocm_namespaces() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    vulkan_devices: [0]\n    rocm_devices: [0]",
        );
        expect_validation_error(&yaml, "mutually exclusive");
    }

    #[test]
    fn rejects_gpus_exceeding_rocm_pool() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    gpus: 3\n    rocm_devices: [0, 1]",
        );
        expect_validation_error(&yaml, "gpus (3) exceeds rocm_devices count (2)");
    }

    #[test]
    fn rejects_rocm_device_without_devices_section() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    rocm_devices: [0]",
        );
        expect_validation_error(&yaml, "rocm_device 0 not defined in devices.rocm");
    }

    #[test]
    fn rejects_undefined_rocm_device_index() {
        let slots = real_amd_slots();
        if slots.is_empty() {
            return;
        }
        let yaml = BASE
            .replace(
                "apikeys_file: apikeys.txt",
                &format!(
                    "apikeys_file: apikeys.txt\ndevices:\n  rocm:\n    0: \"{}\"",
                    slots[0]
                ),
            )
            .replace(
                "cmd: \"sleep 3600\"",
                "cmd: \"sleep 3600\"\n    rocm_devices: [1]",
            );
        expect_validation_error(&yaml, "rocm_device 1 not defined in devices.rocm");
    }

    #[test]
    fn accepts_multi_gpu_rocm_model() {
        let slots = real_amd_slots();
        if slots.len() < 2 {
            return;
        }
        let yaml = BASE
            .replace(
                "apikeys_file: apikeys.txt",
                &format!(
                    "apikeys_file: apikeys.txt\ndevices:\n  rocm:\n    0: \"{}\"\n    1: \"{}\"",
                    slots[0], slots[1]
                ),
            )
            .replace(
                "cmd: \"sleep 3600\"",
                "cmd: \"sleep 3600\"\n    gpus: 2\n    rocm_devices: [0, 1]",
            );
        let cfg = parse(&yaml).unwrap();
        let devs = cfg.devices.unwrap();
        assert_eq!(devs.rocm.len(), 2);
        assert_eq!(
            cfg.models[0].device_kind(),
            crate::backend::DeviceKind::Rocm
        );
    }

    #[test]
    fn rejects_duplicate_rocm_slots() {
        let slots = real_amd_slots();
        if slots.is_empty() {
            return;
        }
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            &format!(
                "apikeys_file: apikeys.txt\ndevices:\n  rocm:\n    0: \"{0}\"\n    1: \"{0}\"",
                slots[0]
            ),
        );
        expect_validation_error(&yaml, "devices.rocm: PCI slot");
    }

    #[test]
    fn rejects_non_amd_rocm_slot() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\ndevices:\n  rocm:\n    0: \"0000:ff:ff.f\"",
        );
        expect_validation_error(&yaml, "not found as an AMD GPU");
    }

    #[test]
    fn rejects_gpus_exceeding_cuda_pool() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    gpus: 3\n    cuda_devices: [0, 1]",
        );
        expect_validation_error(&yaml, "gpus (3) exceeds cuda_devices count (2)");
    }

    #[test]
    fn rejects_cuda_device_without_devices_section() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    cuda_devices: [0]",
        );
        expect_validation_error(&yaml, "cuda_device 0 not defined in devices.cuda");
    }

    #[test]
    fn rejects_undefined_cuda_device_index() {
        let slots = real_pci_slots(1);
        if slots.is_empty() {
            return; // no PCI bus visible in this environment
        }
        let yaml = BASE
            .replace(
                "apikeys_file: apikeys.txt",
                &format!(
                    "apikeys_file: apikeys.txt\ndevices:\n  cuda:\n    0:\n      pci: \"{}\"",
                    slots[0]
                ),
            )
            .replace(
                "cmd: \"sleep 3600\"",
                "cmd: \"sleep 3600\"\n    cuda_devices: [1]",
            );
        expect_validation_error(&yaml, "cuda_device 1 not defined in devices.cuda");
    }

    #[test]
    fn accepts_multi_gpu_cuda_model() {
        let slots = real_pci_slots(2);
        if slots.len() < 2 {
            return; // need two distinct PCI slots for this test
        }
        let yaml = BASE
            .replace(
                "apikeys_file: apikeys.txt",
                &format!(
                    "apikeys_file: apikeys.txt\ndevices:\n  cuda:\n    0:\n      pci: \"{}\"\n      vram_mb: 24576\n    1:\n      pci: \"{}\"",
                    slots[0], slots[1]
                ),
            )
            .replace(
                "cmd: \"sleep 3600\"",
                "cmd: \"sleep 3600\"\n    gpus: 2\n    cuda_devices: [0, 1]",
            );
        let cfg = parse(&yaml).unwrap();
        let devs = cfg.devices.unwrap();
        assert_eq!(devs.cuda[&0].vram_mb, Some(24576));
        assert_eq!(devs.cuda[&1].vram_mb, None);
    }

    #[test]
    fn rejects_duplicate_cuda_slots() {
        let slots = real_pci_slots(1);
        if slots.is_empty() {
            return;
        }
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            &format!(
                "apikeys_file: apikeys.txt\ndevices:\n  cuda:\n    0:\n      pci: \"{0}\"\n    1:\n      pci: \"{0}\"",
                slots[0]
            ),
        );
        expect_validation_error(&yaml, "assigned to multiple indices");
    }

    #[test]
    fn rejects_cuda_slot_also_assigned_in_vulkan() {
        let drm = real_drm_slots();
        if drm.is_empty() {
            return; // no DRM cards in this environment
        }
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            &format!(
                "apikeys_file: apikeys.txt\ndevices:\n  vulkan:\n    0: \"{0}\"\n  cuda:\n    0:\n      pci: \"{0}\"",
                drm[0]
            ),
        );
        expect_validation_error(&yaml, "also assigned in devices.vulkan");
    }

    #[test]
    fn rejects_cuda_slot_not_on_pci_bus() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\ndevices:\n  cuda:\n    0:\n      pci: \"0000:ff:ff.f\"",
        );
        expect_validation_error(&yaml, "not found in /sys/bus/pci/devices/");
    }

    #[test]
    fn rejects_empty_keepalive_cmd() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\nkeep_alive:\n  amd:\n    cmd: \"  \"\n    sleep: 5",
        );
        expect_validation_error(&yaml, "keep_alive.amd.cmd must not be empty");
    }

    // ── Aliases ─────────────────────────────────────────────────────

    #[test]
    fn accepts_scalar_target_as_one_element_targets() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\naliases:\n  - name: a\n    target: m",
        );
        let cfg = parse(&yaml).unwrap();
        assert_eq!(cfg.aliases[0].targets, vec!["m".to_owned()]);
        // Defaults.
        assert_eq!(cfg.aliases[0].policy, AliasPolicy::PreferLoaded);
        assert_eq!(cfg.aliases[0].make_room, MakeRoomMode::EvictIdle);
        assert_eq!(cfg.aliases[0].drain_timeout, 300);
    }

    #[test]
    fn accepts_migration_make_room_mode() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\naliases:\n  - name: a\n    targets: [m]\n    make_room: migrate",
        );
        let cfg = parse(&yaml).unwrap();
        assert_eq!(cfg.aliases[0].make_room, MakeRoomMode::Migrate);
        assert!(cfg.aliases[0].make_room.is_migration());
        assert!(!cfg.aliases[0].make_room.is_legacy_eviction());
    }

    #[test]
    fn accepts_targets_list_with_explicit_settings() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n  - name: m2\n    context_length: 4096\n    cmd: \"sleep 3600\"",
        )
        .replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\naliases:\n  - name: a\n    targets: [m, m2]\n    policy: prefer_order\n    make_room: drain_surplus\n    drain_timeout: 60",
        );
        let cfg = parse(&yaml).unwrap();
        let a = &cfg.aliases[0];
        assert_eq!(a.targets, vec!["m".to_owned(), "m2".to_owned()]);
        assert_eq!(a.policy, AliasPolicy::PreferOrder);
        assert_eq!(a.make_room, MakeRoomMode::DrainSurplus);
        assert_eq!(a.drain_timeout, 60);
    }

    #[test]
    fn rejects_empty_targets_list() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\naliases:\n  - name: a\n    targets: []",
        );
        expect_validation_error(&yaml, "targets must not be empty");
    }

    #[test]
    fn rejects_unknown_alias_target() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\naliases:\n  - name: a\n    targets: [m, nope]",
        );
        expect_validation_error(&yaml, "alias 'a' targets unknown model 'nope'");
    }

    #[test]
    fn rejects_alias_chaining() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\naliases:\n  - name: a\n    target: m\n  - name: b\n    targets: [a]",
        );
        expect_validation_error(&yaml, "aliases cannot chain");
    }

    #[test]
    fn rejects_zero_drain_timeout() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\naliases:\n  - name: a\n    target: m\n    drain_timeout: 0",
        );
        expect_validation_error(&yaml, "drain_timeout must be > 0");
    }

    #[test]
    fn ignores_removed_alias_prompt_keys() {
        // Old configs with system_prompt / prompt_template keep loading —
        // serde ignores the now-unknown keys.
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\naliases:\n  - name: a\n    target: m\n    system_prompt: hi\n    prompt_template: \"{prompt}\"",
        );
        parse(&yaml).unwrap();
    }

    #[test]
    fn queue_and_migration_defaults_are_backward_compatible() {
        let cfg = parse(BASE).unwrap();
        assert_eq!(cfg.server.queue_wait_timeout_secs, 3600);
        assert_eq!(cfg.server.queue_heartbeat_secs, 10);
        assert_eq!(cfg.server.migration_timeout_secs, 0);
        assert_eq!(cfg.server.eviction_policy, EvictionPolicy::Lru);
    }

    #[test]
    fn accepts_unlimited_queue_and_migration_timeouts() {
        let yaml = BASE.replace(
            "server: {}",
            "server:\n  queue_wait_timeout_secs: 0\n  migration_timeout_secs: 0",
        );
        let cfg = parse(&yaml).unwrap();
        assert_eq!(cfg.server.queue_wait_timeout_secs, 0);
        assert_eq!(cfg.server.migration_timeout_secs, 0);
    }

    #[test]
    fn rejects_zero_queue_heartbeat() {
        let yaml = BASE.replace("server: {}", "server:\n  queue_heartbeat_secs: 0");
        expect_validation_error(&yaml, "queue_heartbeat_secs must be > 0");
    }

    #[test]
    fn rejects_unknown_eviction_policy() {
        let yaml = BASE.replace("server: {}", "server:\n  eviction_policy: priority_lru");
        assert!(serde_yaml_ng::from_str::<Config>(&yaml).is_err());
    }

    #[test]
    fn drain_idle_timeout_defaults_to_3600() {
        let cfg = parse(BASE).unwrap();
        assert_eq!(cfg.server.drain_idle_timeout_secs, 3600);
    }
}
