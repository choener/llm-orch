// ── GPU keep-alive manager ──────────────────────────────────────────────────
//
// Prevents kernel GPU driver auto-suspend when at least one model instance
// runs on a GPU.  Each GPU gets its own background tokio task that runs a
// configurable command on a fixed interval.

use crate::config::{GpuKeepAlive, KeepAliveConfig};
use std::collections::HashMap;
use std::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

/// Manages per-GPU keep-alive background tasks.
pub struct KeepAliveManager {
    /// Active tasks keyed by PCI slot.
    tasks: Mutex<HashMap<String, (JoinHandle<()>, CancellationToken)>>,
    /// PCI slot → sysfs card index (built at construction).
    slot_to_card: HashMap<String, usize>,
    /// Config.
    cfg: GpuKeepAlive,
}

impl KeepAliveManager {
    /// Create a new manager from the keep-alive config.
    ///
    /// Scans `/sys/class/drm/card*/device/uevent` to build a PCI slot →
    /// cardN index map for `{index}` substitution.  Returns `None` if no
    /// keep-alive is configured.
    pub fn new(cfg: &Option<KeepAliveConfig>) -> Option<Self> {
        let amd = cfg.as_ref().and_then(|k| k.amd.clone())?;
        let slot_to_card = build_slot_map();
        Some(Self {
            tasks: Mutex::new(HashMap::new()),
            slot_to_card,
            cfg: amd,
        })
    }

    /// Start keep-alive for the GPU at `pci_slot` if not already running.
    ///
    /// Safe to call multiple times — subsequent calls are no-ops.
    pub fn ensure_running(&self, pci_slot: &str) {
        let card_index = match self.slot_to_card.get(pci_slot) {
            Some(&idx) => idx,
            None => {
                warn!(slot = pci_slot, "keep-alive: PCI slot not in sysfs");
                return;
            }
        };

        let mut tasks = self.tasks.lock().unwrap();
        if tasks.contains_key(pci_slot) {
            return; // already running
        }

        let cmd = self.cfg.cmd.replace("{index}", &card_index.to_string());
        let sleep_secs = self.cfg.sleep;
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let slot = pci_slot.to_owned();

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = cancel_clone.cancelled() => {
                        debug!(slot = %slot, card = card_index, "keep-alive cancelled");
                        break;
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_secs(sleep_secs)) => {
                        match tokio::process::Command::new("sh")
                            .arg("-c")
                            .arg(&cmd)
                            .stdin(std::process::Stdio::null())
                            .stdout(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null())
                            .status()
                            .await
                        {
                            Ok(status) if !status.success() => {
                                warn!(
                                    slot = %slot, card = card_index,
                                    exit_code = status.code(),
                                    "keep-alive command failed"
                                );
                            }
                            Ok(_) => {
                                debug!(slot = %slot, card = card_index, "keep-alive ping");
                            }
                            Err(e) => {
                                warn!(slot = %slot, card = card_index, error = %e, "keep-alive command spawn failed");
                            }
                            _ => {}
                        }
                    }
                }
            }
        });

        tasks.insert(pci_slot.to_owned(), (handle, cancel));
        debug!(slot = pci_slot, card = card_index, "keep-alive started");
    }

    /// Stop keep-alive for the GPU at `pci_slot`.
    pub fn stop(&self, pci_slot: &str) {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some((_handle, cancel)) = tasks.remove(pci_slot) {
            cancel.cancel();
            debug!(slot = pci_slot, "keep-alive stopped");
        }
    }

    /// Whether this manager was built from the same keep-alive config
    /// (used to decide whether a config reload requires a rebuild).
    pub fn matches(&self, cfg: &Option<KeepAliveConfig>) -> bool {
        let amd = cfg.as_ref().and_then(|k| k.amd.as_ref());
        amd == Some(&self.cfg)
    }

    /// Stop all keep-alive tasks.
    pub fn stop_all(&self) {
        let tasks: Vec<_> = {
            let mut guard = self.tasks.lock().unwrap();
            guard.drain().map(|(_, (_, cancel))| cancel).collect()
        };
        for cancel in tasks {
            cancel.cancel();
        }
        debug!("all keep-alive tasks cancelled");
    }
}

/// Scan `/sys/class/drm/card*/device/uevent` and return a
/// PCI slot → cardN index map.
fn build_slot_map() -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for i in 0..16 {
        let uevent = format!("/sys/class/drm/card{}/device/uevent", i);
        if let Ok(contents) = std::fs::read_to_string(&uevent) {
            for line in contents.lines() {
                if let Some(slot) = line.strip_prefix("PCI_SLOT_NAME=") {
                    map.insert(slot.trim().to_string(), i);
                }
            }
        }
    }
    map
}