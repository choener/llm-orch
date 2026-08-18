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
        if config.apikeys_file.is_relative() {
            if let Some(parent) = path.parent() {
                config.apikeys_file = parent.join(&config.apikeys_file);
            }
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
            if m.gpus == 0 {
                return Err(ConfigError::Validation(format!(
                    "model '{}': gpus must be >= 1",
                    m.name
                )));
            }
            if !m.vulkan_devices.is_empty() && !m.cuda_devices.is_empty() {
                return Err(ConfigError::Validation(format!(
                    "model '{}': vulkan_devices and cuda_devices are mutually exclusive (a model uses exactly one device namespace)",
                    m.name
                )));
            }
            let (pool_len, ns) = if !m.cuda_devices.is_empty() {
                (m.cuda_devices.len(), "cuda_devices")
            } else {
                (m.vulkan_devices.len(), "vulkan_devices")
            };
            if pool_len == 0 {
                // CPU-only model: spanning multiple devices is meaningless.
                if m.gpus != 1 {
                    return Err(ConfigError::Validation(format!(
                        "model '{}': gpus ({}) exceeds device count (0) (CPU-only models must use gpus: 1)",
                        m.name, m.gpus
                    )));
                }
            } else if m.gpus > pool_len {
                return Err(ConfigError::Validation(format!(
                    "model '{}': gpus ({}) exceeds {} count ({})",
                    m.name, m.gpus, ns, pool_len
                )));
            }
        }

        // --- aliases ---
        let model_names: HashSet<&str> = self.models.iter().map(|m| m.name.as_str()).collect();
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
            if a.target.is_empty() {
                return Err(ConfigError::Validation(format!(
                    "alias '{}': target must not be empty",
                    a.name
                )));
            }
            if !model_names.contains(a.target.as_str()) {
                return Err(ConfigError::Validation(format!(
                    "alias '{}' targets unknown model '{}'",
                    a.name, a.target
                )));
            }
        }

        // --- cmd_aliases ---
        for reserved in ["port", "context_length", "name"] {
            if self.cmd_aliases.contains_key(reserved) {
                return Err(ConfigError::Validation(format!(
                    "cmd_aliases: '{}' is a reserved name",
                    reserved
                )));
            }
        }

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
        if let Some(ref ka) = self.keep_alive {
            if let Some(ref amd) = ka.amd {
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
        }

        // --- model vulkan_devices ---
        if let Some(ref devs) = self.devices {
            let valid_indices: HashSet<&usize> = devs.vulkan.keys().collect();
            for m in &self.models {
                for idx in &m.vulkan_devices {
                    if !valid_indices.contains(idx) {
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
                for idx in &m.cuda_devices {
                    if !valid_indices.contains(idx) {
                        return Err(ConfigError::Validation(format!(
                            "model '{}': cuda_device {} not defined in devices.cuda",
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
    /// time; `{port}` is replaced with the allocated port number and
    /// `{context_length}` with the model's declared `context_length`.
    pub cmd: String,

    /// Vulkan device indices this model can be placed on (from `devices.vulkan`).
    /// Empty = CPU only (unless `cuda_devices` is set).
    /// Mutually exclusive with `cuda_devices`.
    #[serde(default)]
    pub vulkan_devices: Vec<usize>,

    /// CUDA device indices this model can be placed on (from `devices.cuda`).
    /// When non-empty the model is a CUDA model: placement uses the CUDA
    /// pool and instances are pinned via `CUDA_VISIBLE_DEVICES`.
    /// Mutually exclusive with `vulkan_devices`.
    #[serde(default)]
    pub cuda_devices: Vec<usize>,

    /// Number of devices each instance of this model spans.
    /// The model is split evenly across them (llama.cpp's default tensor
    /// split over `GGML_VK_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`):
    /// `vram` is reserved on **each** occupied GPU, so `vram: 20000,
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
    /// Device namespace this model is placed on.  A model is a CUDA model
    /// iff `cuda_devices` is non-empty (validation guarantees the two
    /// pools are mutually exclusive); otherwise it is a Vulkan model
    /// (`vulkan_devices` possibly empty = CPU-only).
    pub fn device_kind(&self) -> crate::backend::DeviceKind {
        if !self.cuda_devices.is_empty() {
            crate::backend::DeviceKind::Cuda
        } else {
            crate::backend::DeviceKind::Vulkan
        }
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
/// Each backend gets its own namespace (e.g. `vulkan`, `cuda`).  The
/// indices here are what `GGML_VK_VISIBLE_DEVICES` /
/// `CUDA_VISIBLE_DEVICES` use — they correspond to the backend's
/// enumeration order, not sysfs card numbers.
#[derive(Debug, Clone, Deserialize)]
pub struct DevicesConfig {
    /// Vulkan device index → PCI slot name.
    #[serde(default)]
    pub vulkan: HashMap<usize, String>,

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
    let mut slots = Vec::new();
    for i in 0..16 {
        let uevent = format!("/sys/class/drm/card{}/device/uevent", i);
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

#[derive(Debug, Clone, Deserialize)]
pub struct AliasConfig {
    /// The alias name — appears in `/v1/models` and can be used in API requests.
    pub name: String,

    /// Target model name (must match a `ModelConfig::name`).
    pub target: String,

    /// Optional system prompt injected when this alias is used.
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Optional prompt template override (e.g. chat format).
    #[serde(default)]
    pub prompt_template: Option<String>,
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
    fn accepts_multi_gpu_model() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    gpus: 2\n    vulkan_devices: [0, 1]",
        );
        parse(&yaml).unwrap();
    }

    #[test]
    fn rejects_reserved_cmd_alias_names() {
        for reserved in ["port", "context_length", "name"] {
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

    #[test]
    fn rejects_model_with_both_device_namespaces() {
        let yaml = BASE.replace(
            "cmd: \"sleep 3600\"",
            "cmd: \"sleep 3600\"\n    vulkan_devices: [0]\n    cuda_devices: [0]",
        );
        expect_validation_error(&yaml, "mutually exclusive");
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
}
