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
///
/// Usage is refcounted per PCI slot: every instance occupying a GPU holds
/// one `acquire`, released exactly once when the instance is unregistered.
/// The task runs while the count is > 0.  Refcounting makes concurrent
/// spawn/remove interleavings safe — a stop can never take effect while a
/// new instance has just landed on the GPU.
pub struct KeepAliveManager {
    /// Active tasks keyed by PCI slot.
    tasks: Mutex<HashMap<String, (JoinHandle<()>, CancellationToken)>>,
    /// Number of instances currently occupying each GPU (by PCI slot).
    refcounts: Mutex<HashMap<String, usize>>,
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
            refcounts: Mutex::new(HashMap::new()),
            slot_to_card,
            cfg: amd,
        })
    }

    /// Register an instance on the GPU at `pci_slot`.  Starts the
    /// keep-alive task when the first instance lands on the GPU.
    ///
    /// Must be paired with exactly one `release` call per acquire.
    pub fn acquire(&self, pci_slot: &str) {
        let first = {
            let mut counts = self.refcounts.lock().unwrap();
            let count = counts.entry(pci_slot.to_owned()).or_insert(0);
            *count += 1;
            *count == 1
        };
        if first {
            self.start_task(pci_slot);
        }
    }

    /// Unregister an instance from the GPU at `pci_slot`.  Stops the
    /// keep-alive task when the last instance leaves the GPU.
    pub fn release(&self, pci_slot: &str) {
        let last = {
            let mut counts = self.refcounts.lock().unwrap();
            match counts.get_mut(pci_slot) {
                Some(count) if *count > 1 => {
                    *count -= 1;
                    false
                }
                Some(_) => {
                    counts.remove(pci_slot);
                    true
                }
                None => {
                    warn!(slot = pci_slot, "keep-alive: release without acquire");
                    return;
                }
            }
        };
        if last {
            self.stop_task(pci_slot);
        }
    }

    /// Start keep-alive for the GPU at `pci_slot` if not already running.
    fn start_task(&self, pci_slot: &str) {
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
                        }
                    }
                }
            }
        });

        tasks.insert(pci_slot.to_owned(), (handle, cancel));
        debug!(slot = pci_slot, card = card_index, "keep-alive started");
    }

    /// Stop keep-alive for the GPU at `pci_slot`.
    fn stop_task(&self, pci_slot: &str) {
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

    /// Stop all keep-alive tasks and reset refcounts (shutdown / rebuild).
    pub fn stop_all(&self) {
        let tasks: Vec<_> = {
            let mut guard = self.tasks.lock().unwrap();
            guard.drain().map(|(_, (_, cancel))| cancel).collect()
        };
        for cancel in tasks {
            cancel.cancel();
        }
        self.refcounts.lock().unwrap().clear();
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
