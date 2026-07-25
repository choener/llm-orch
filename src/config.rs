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
                return Err(ConfigError::Validation("model name must not be empty".into()));
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
            if m.vulkan_devices.is_empty() {
                // CPU-only model: spanning multiple devices is meaningless.
                if m.gpus != 1 {
                    return Err(ConfigError::Validation(format!(
                        "model '{}': gpus ({}) exceeds vulkan_devices count (0) (CPU-only models must use gpus: 1)",
                        m.name, m.gpus
                    )));
                }
            } else if m.gpus > m.vulkan_devices.len() {
                return Err(ConfigError::Validation(format!(
                    "model '{}': gpus ({}) exceeds vulkan_devices count ({})",
                    m.name,
                    m.gpus,
                    m.vulkan_devices.len()
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
        if self.cmd_aliases.contains_key("port") {
            return Err(ConfigError::Validation(
                "cmd_aliases: 'port' is a reserved name".into(),
            ));
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
            let existing_slots: HashSet<&str> = pci_slots
                .iter()
                .map(|s| s.as_str())
                .collect();
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
    /// `{alias_name}` placeholders are resolved from `cmd_aliases` at load time;
    /// `{port}` is replaced with the allocated port number before spawning.
    pub cmd: String,

    /// Vulkan device indices this model can be placed on (from `devices.vulkan`).
    /// Empty = CPU only.
    #[serde(default)]
    pub vulkan_devices: Vec<usize>,

    /// Number of Vulkan devices each instance of this model spans.
    /// The model is split evenly across them (llama.cpp's default tensor
    /// split over `GGML_VK_VISIBLE_DEVICES`): `vram` is reserved on **each**
    /// occupied GPU, so `vram: 20000, gpus: 2` reserves 2×20000 MB.
    /// Asymmetric splits (`--tensor-split` in `cmd`) are possible; then
    /// `vram` is a conservative per-GPU reservation.  Must be `>= 1` and
    /// `<= vulkan_devices.len()`.
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

fn default_autoscale_up() -> f64 { 0.7 }
fn default_autoscale_down() -> f64 { 0.4 }
fn default_autoscale_cooldown() -> u64 { 120 }

// ---------------------------------------------------------------------------
// Device mapping
// ---------------------------------------------------------------------------

/// Global device index → PCI slot mapping.
///
/// Each backend gets its own namespace (e.g. `vulkan`).  The indices here
/// are what `GGML_VK_VISIBLE_DEVICES` uses — they correspond to Vulkan's
/// enumeration order, not sysfs card numbers.
#[derive(Debug, Clone, Deserialize)]
pub struct DevicesConfig {
    /// Vulkan device index → PCI slot name.
    #[serde(default)]
    pub vulkan: HashMap<usize, String>,

    /// Optional per-GPU VRAM limit in MB.
    /// Caps the usable VRAM below the sysfs-reported total, leaving
    /// headroom for driver overhead.  When unset, sysfs total is used.
    #[serde(default)]
    pub vram_limit_mb: HashMap<usize, u64>,
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
    fn rejects_zero_max_instances() {
        let yaml = BASE.replace("cmd: \"sleep 3600\"", "cmd: \"sleep 3600\"\n    max_instances: 0");
        expect_validation_error(&yaml, "max_instances must be > 0");
    }

    #[test]
    fn rejects_zero_max_concurrent() {
        let yaml = BASE.replace("cmd: \"sleep 3600\"", "cmd: \"sleep 3600\"\n    max_concurrent: 0");
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
    fn rejects_empty_keepalive_cmd() {
        let yaml = BASE.replace(
            "apikeys_file: apikeys.txt",
            "apikeys_file: apikeys.txt\nkeep_alive:\n  amd:\n    cmd: \"  \"\n    sleep: 5",
        );
        expect_validation_error(&yaml, "keep_alive.amd.cmd must not be empty");
    }
}
