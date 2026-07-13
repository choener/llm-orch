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
        if let PortRange::Range { start, end } = &self.server.port_range {
            if start > end {
                return Err(ConfigError::Validation(format!(
                    "server.port_range: start ({}) must be <= end ({})",
                    start, end
                )));
            }
        }

        Ok(())
    }

    /// Resolve `{alias_name}` references in a model's `cmd` against `cmd_aliases`.
    /// `{port}` is left untouched — resolved at spawn time.
    pub fn resolve_model_cmd(&self, raw_cmd: &str) -> String {
        let mut resolved = raw_cmd.to_owned();
        for (key, value) in &self.cmd_aliases {
            let placeholder = format!("{{{}}}", key);
            resolved = resolved.replace(&placeholder, value);
        }
        resolved
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
}

fn default_listen() -> String {
    "127.0.0.1:8080".into()
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

impl PortRange {
    /// Returns `true` when the OS should pick ports.
    pub fn is_ephemeral(&self) -> bool {
        matches!(self, PortRange::EphemeralWord(s) if s == "ephemeral")
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
    #[serde(default)]
    pub priority: i32,

    /// Declared VRAM usage in MB (scheduler hint).
    #[serde(default)]
    pub vram: u64,

    /// Declared system RAM usage in MB (scheduler hint).
    #[serde(default)]
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
}

fn default_max_instances() -> usize {
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