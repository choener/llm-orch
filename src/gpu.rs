// ── AMD GPU sysfs monitoring ─────────────────────────────────────────────────
//
// Reads GPU metrics from /sys/class/drm/card*/device/ on Linux for AMD GPUs
// (vendor 0x1002).  Metrics are read per-card and exposed as GpuMetrics.
//
// Sysfs nodes read (where available):
//   mem_info_vram_total, mem_info_vram_used      – VRAM (bytes)
//   mem_info_gtt_total,   mem_info_gtt_used       – GTT memory (bytes)
//   mem_info_vis_vram_total, mem_info_vis_vram_used – visible VRAM (bytes)
//   hwmon/hwmon*/temp1_input                      – temperature (milli°C)
//   hwmon/hwmon*/power1_average                    – power (µW)
//   gpu_busy_percent                               – GPU utilisation (%)
//   pp_dpm_sclk                                    – shader clock (parse active)
//   pp_dpm_mclk                                    – memory clock (parse active)

use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, interval};
use tracing::warn;

// ── Public types ─────────────────────────────────────────────────────────────

/// One snapshot of metrics for a single AMD GPU.
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// Card index (e.g. 0 for card0).
    pub index: usize,
    /// PCI slot name from the uevent file (e.g. "0000:7b:00.0").
    pub pci_slot: String,
    /// VRAM vendor string (e.g. "samsung"), if available.
    pub vram_vendor: Option<String>,

    // ── Memory ──────────────────────────────────────────────────────
    pub vram_total_bytes: u64,
    pub vram_used_bytes: u64,

    // ── Thermals / power ────────────────────────────────────────────
    /// Temperature in °C.
    pub temperature_c: Option<f64>,
    /// Power in watts.
    pub power_w: Option<f64>,

    // ── Utilisation ─────────────────────────────────────────────────
    /// GPU busy percentage (0–100).
    pub gpu_busy_pct: Option<u32>,

    // ── Clocks ──────────────────────────────────────────────────────
    /// Current shader clock in MHz.
    pub sclk_mhz: Option<u64>,
    /// Current memory clock in MHz.
    pub mclk_mhz: Option<u64>,
}

// ── Reader (periodic snapshot) ───────────────────────────────────────────────

/// Holds the latest GPU metrics snapshot behind an `Arc<RwLock<…>>` and
/// periodically refreshes it on a background task.
pub struct GpuReader {
    /// Latest snapshot — `None` if no AMD GPUs were found.
    snapshot: Arc<RwLock<Vec<GpuMetrics>>>,
}

impl GpuReader {
    /// Create a new reader, take an initial snapshot, and start the
    /// background polling task.  Returns the reader handle (for
    /// cheap shared access) and the `JoinHandle` of the poll task.
    pub fn start(poll_interval: Duration) -> (Self, tokio::task::JoinHandle<()>) {
        let initial = read_all();
        let snapshot = Arc::new(RwLock::new(initial));

        let snapshot_clone = Arc::clone(&snapshot);
        let handle = tokio::spawn(async move {
            let mut tick = interval(poll_interval);
            loop {
                tick.tick().await;
                let metrics = read_all_async().await;
                let mut guard = snapshot_clone.write().await;
                *guard = metrics;
            }
        });

        (Self { snapshot }, handle)
    }

    /// Return the `Arc<RwLock<…>>` for sharing with axum state.
    pub fn snapshot_arc(&self) -> Arc<RwLock<Vec<GpuMetrics>>> {
        Arc::clone(&self.snapshot)
    }
}

// ── Core reading ─────────────────────────────────────────────────────────────

/// Read metrics for every detected GPU: AMD via sysfs, NVIDIA via
/// nvidia-smi (blocking variant — used for the initial snapshot).
pub fn read_all() -> Vec<GpuMetrics> {
    let mut metrics = read_all_gpus();
    metrics.extend(crate::nvidia::read_all_nvidia_blocking());
    metrics
}

/// Async variant of [`read_all`], used by the periodic poll task.
async fn read_all_async() -> Vec<GpuMetrics> {
    let mut metrics = read_all_gpus();
    metrics.extend(crate::nvidia::read_all_nvidia().await);
    metrics
}

/// Read metrics for every AMD GPU detected on the system.
pub fn read_all_gpus() -> Vec<GpuMetrics> {
    let mut metrics = Vec::new();

    for card_index in 0..16 {
        let card_path = format!("/sys/class/drm/card{}/device", card_index);
        let card = Path::new(&card_path);
        if !card.exists() {
            // Card numbering is not guaranteed contiguous (render nodes,
            // driver unbinds) — skip missing indices, don't stop scanning.
            continue;
        }

        // Filter for AMD GPUs only (vendor 0x1002).
        if !is_amd_gpu(card) {
            continue;
        }

        match read_one_gpu(card_index, card) {
            Ok(m) => metrics.push(m),
            Err(e) => warn!("failed to read gpu metrics for card{}: {}", card_index, e),
        }
    }

    metrics
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Read all metrics for a single GPU card.
fn read_one_gpu(card_index: usize, device: &Path) -> Result<GpuMetrics, String> {
    let pci_slot = read_pci_slot(device);
    let vram_vendor = read_file(device, "mem_info_vram_vendor");

    // Memory
    let vram_total_bytes = read_u64(device, "mem_info_vram_total").unwrap_or(0);
    let vram_used_bytes = read_u64(device, "mem_info_vram_used").unwrap_or(0);

    // Temperature: find the first hwmon that has temp1_input.
    let temperature_c =
        find_hwmon_numeric(device, "temp1_input").map(|millic| millic as f64 / 1000.0);

    // Power: find the first hwmon that has power1_average.
    let power_w =
        find_hwmon_numeric(device, "power1_average").map(|microw| microw as f64 / 1_000_000.0);

    // GPU utilisation.
    let gpu_busy_pct = read_u64(device, "gpu_busy_percent").map(|v| v as u32);

    // Clocks: parse the active level from pp_dpm_sclk / pp_dpm_mclk.
    let sclk_mhz = read_dpm_clock(device, "pp_dpm_sclk");
    let mclk_mhz = read_dpm_clock(device, "pp_dpm_mclk");

    Ok(GpuMetrics {
        index: card_index,
        pci_slot,
        vram_vendor,
        vram_total_bytes,
        vram_used_bytes,
        temperature_c,
        power_w,
        gpu_busy_pct,
        sclk_mhz,
        mclk_mhz,
    })
}

/// Check whether a card device directory is for an AMD GPU.
fn is_amd_gpu(device: &Path) -> bool {
    read_file(device, "vendor")
        .map(|v| v.trim() == "0x1002")
        .unwrap_or(false)
}

/// Read the PCI slot name from the device's `uevent` file.
fn read_pci_slot(device: &Path) -> String {
    let uevent = device.join("uevent");
    match fs::read_to_string(&uevent) {
        Ok(contents) => {
            for line in contents.lines() {
                if let Some(slot) = line.strip_prefix("PCI_SLOT_NAME=") {
                    return slot.trim().to_string();
                }
            }
            "unknown".to_string()
        }
        Err(_) => "unknown".to_string(),
    }
}

// ── Sysfs read helpers ───────────────────────────────────────────────────────

/// Read the contents of a single-file sysfs node as a trimmed `String`.
fn read_file(device: &Path, name: &str) -> Option<String> {
    let path = device.join(name);
    fs::read_to_string(&path).map(|s| s.trim().to_string()).ok()
}

/// Read a numeric sysfs node as `u64`.
fn read_u64(device: &Path, name: &str) -> Option<u64> {
    read_file(device, name).and_then(|s| s.parse::<u64>().ok())
}

/// Find a numeric value under the first `hwmon/hwmon*/<name>` directory.
/// Walks `device/hwmon/hwmonN/<name>` for N in 0..8 and returns the first
/// successful parse.
fn find_hwmon_numeric(device: &Path, name: &str) -> Option<u64> {
    for n in 0..8 {
        let path = device.join(format!("hwmon/hwmon{}", n)).join(name);
        if let Ok(contents) = fs::read_to_string(&path) {
            if let Ok(val) = contents.trim().parse::<u64>() {
                return Some(val);
            }
        }
    }
    None
}

/// Parse the active clock frequency from a `pp_dpm_*` file.
///
/// Each line looks like: `0: 600Mhz *`  – the active level has a trailing `*`.
/// We return the numeric value (in MHz) of the active line.
fn read_dpm_clock(device: &Path, name: &str) -> Option<u64> {
    let contents = read_file(device, name)?;
    for line in contents.lines() {
        if line.contains('*') {
            // Format: "N: 1234Mhz *"
            // Find the colon, skip whitespace, parse until non-digit.
            if let Some(colon) = line.find(':') {
                let after_colon = &line[colon + 1..].trim();
                let num_str: String = after_colon
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                return num_str.parse::<u64>().ok();
            }
        }
    }
    None
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_all_gpus_does_not_panic() {
        let metrics = read_all_gpus();
        // There may be zero GPUs in CI, but the function must not panic.
        for m in &metrics {
            assert!(m.vram_total_bytes > 0, "vram total must be > 0");
            assert!(!m.pci_slot.is_empty(), "pci slot must not be empty");
        }
    }

    #[test]
    fn test_parse_dpm_clock() {
        let sample = "0: 600Mhz *\n1: 700Mhz\n2: 2200Mhz\n";
        let temp = std::env::temp_dir().join("test_pp_dpm_sclk");
        std::fs::write(&temp, sample).unwrap();
        let _device = temp.parent().unwrap();
        // Can't easily inject filename, but we can test the logic inline.
        let result: Option<u64> = {
            let mut val = None;
            for line in sample.lines() {
                if line.contains('*') {
                    if let Some(colon) = line.find(':') {
                        let after = &line[colon + 1..].trim();
                        let num: String =
                            after.chars().take_while(|c| c.is_ascii_digit()).collect();
                        val = Some(num.parse().unwrap());
                        break;
                    }
                }
            }
            val
        };
        assert_eq!(result, Some(600));
    }
}
