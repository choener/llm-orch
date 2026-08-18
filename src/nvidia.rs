// ── NVIDIA GPU metrics via nvidia-smi ────────────────────────────────────────
//
// NVIDIA GPUs expose no VRAM metrics through sysfs, so metrics are collected
// by periodically shelling out to `nvidia-smi` (always present with the
// proprietary driver) and parsing its CSV output into the shared GpuMetrics
// shape, keyed by PCI slot like the AMD sysfs reader.
//
// Degradation: if `nvidia-smi` is missing or fails, this module contributes
// no entries (a warning is logged once per process); the scheduler then
// falls back to static `vram_mb` totals from `devices.cuda`.

use crate::gpu::GpuMetrics;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, warn};

/// Fields queried from nvidia-smi, in CSV column order.
const QUERY_FIELDS: &str = "pci.bus_id,memory.total,memory.used,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem";

// nvidia-smi failures are expected on non-NVIDIA hosts — warn once.
static FAILURE_WARNED: AtomicBool = AtomicBool::new(false);

fn warn_failure_once(msg: std::fmt::Arguments) {
    if !FAILURE_WARNED.swap(true, Ordering::Relaxed) {
        warn!("nvidia-smi metrics unavailable: {}", msg);
    } else {
        debug!("nvidia-smi metrics unavailable: {}", msg);
    }
}

/// Query nvidia-smi asynchronously (for the periodic poll task).
/// Returns an empty vec on any failure.
pub async fn read_all_nvidia() -> Vec<GpuMetrics> {
    let result = tokio::process::Command::new("nvidia-smi")
        .arg(format!("--query-gpu={QUERY_FIELDS}"))
        .arg("--format=csv,noheader,nounits")
        .output()
        .await;

    match result {
        Ok(out) if out.status.success() => parse_csv(&String::from_utf8_lossy(&out.stdout)),
        Ok(out) => {
            warn_failure_once(format_args!("exit status {}", out.status));
            Vec::new()
        }
        Err(e) => {
            warn_failure_once(format_args!("{e}"));
            Vec::new()
        }
    }
}

/// Query nvidia-smi synchronously (for the initial snapshot at startup).
/// Returns an empty vec on any failure.
pub fn read_all_nvidia_blocking() -> Vec<GpuMetrics> {
    let result = std::process::Command::new("nvidia-smi")
        .arg(format!("--query-gpu={QUERY_FIELDS}"))
        .arg("--format=csv,noheader,nounits")
        .output();

    match result {
        Ok(out) if out.status.success() => parse_csv(&String::from_utf8_lossy(&out.stdout)),
        Ok(out) => {
            warn_failure_once(format_args!("exit status {}", out.status));
            Vec::new()
        }
        Err(e) => {
            warn_failure_once(format_args!("{e}"));
            Vec::new()
        }
    }
}

/// Parse full nvidia-smi CSV output (one GPU per row, no header).
/// Rows with an unparseable PCI bus id or VRAM total are skipped —
/// a missing VRAM total must not masquerade as "0 bytes", or the
/// scheduler would compute zero capacity instead of falling back to
/// static `vram_mb`.
pub fn parse_csv(output: &str) -> Vec<GpuMetrics> {
    output
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .filter_map(|(index, line)| parse_row(index, line))
        .collect()
}

/// Parse one CSV row into GpuMetrics.  `index` is the nvidia-smi
/// enumeration order of the row.
fn parse_row(index: usize, line: &str) -> Option<GpuMetrics> {
    let cols: Vec<&str> = line.split(',').map(str::trim).collect();
    if cols.len() < 8 {
        debug!(
            "nvidia-smi row has {} columns (expected 8): {:?}",
            cols.len(),
            line
        );
        return None;
    }

    let pci_slot = normalize_pci_slot(cols[0])?;
    let vram_total_bytes = parse_mib(cols[1])?;
    let vram_used_bytes = parse_mib(cols[2]).unwrap_or(0);

    Some(GpuMetrics {
        index,
        pci_slot,
        vram_vendor: None,
        vram_total_bytes,
        vram_used_bytes,
        temperature_c: parse_f64(cols[3]),
        gpu_busy_pct: parse_u64(cols[4]).map(|v| v as u32),
        power_w: parse_f64(cols[5]),
        sclk_mhz: parse_u64(cols[6]),
        mclk_mhz: parse_u64(cols[7]),
    })
}

/// Normalize an nvidia-smi PCI bus id (`00000000:65:00.0`, uppercase hex)
/// to the sysfs slot form (`0000:65:00.0`, lowercase hex).
fn normalize_pci_slot(bus_id: &str) -> Option<String> {
    let (domain, rest) = bus_id.split_once(':')?;
    let domain = u32::from_str_radix(domain, 16).ok()?;
    // Validate the remainder: bus:dev.func, all hex.
    let parts: Vec<&str> = rest.split([':', '.']).collect();
    if parts.len() != 3 || parts.iter().any(|p| u32::from_str_radix(p, 16).is_err()) {
        return None;
    }
    Some(format!("{domain:04x}:{}", rest.to_lowercase()))
}

/// Parse a MiB value (from `--format=nounits`) into bytes.
/// Returns None for `N/A` / `[N/A]` / unparseable values.
fn parse_mib(s: &str) -> Option<u64> {
    parse_u64(s).map(|mib| mib * 1024 * 1024)
}

fn parse_u64(s: &str) -> Option<u64> {
    s.parse::<u64>().ok()
}

fn parse_f64(s: &str) -> Option<f64> {
    s.parse::<f64>().ok()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TWO_GPU: &str = "\
00000000:41:00.0, 24576, 512, 34, 0, 25.50, 210, 5001
00000000:C1:00.0, 24576, 12000, 41, 87, 180.25, 1500, 5001
";

    #[test]
    fn parses_two_gpu_output() {
        let gpus = parse_csv(SAMPLE_TWO_GPU);
        assert_eq!(gpus.len(), 2);

        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[0].pci_slot, "0000:41:00.0");
        assert_eq!(gpus[0].vram_total_bytes, 24576 * 1024 * 1024);
        assert_eq!(gpus[0].vram_used_bytes, 512 * 1024 * 1024);
        assert_eq!(gpus[0].temperature_c, Some(34.0));
        assert_eq!(gpus[0].gpu_busy_pct, Some(0));
        assert_eq!(gpus[0].power_w, Some(25.50));
        assert_eq!(gpus[0].sclk_mhz, Some(210));
        assert_eq!(gpus[0].mclk_mhz, Some(5001));

        // Uppercase hex domain/device letters normalize to lowercase.
        assert_eq!(gpus[1].index, 1);
        assert_eq!(gpus[1].pci_slot, "0000:c1:00.0");
        assert_eq!(gpus[1].gpu_busy_pct, Some(87));
    }

    #[test]
    fn skips_row_with_na_vram_total() {
        let out = "00000000:41:00.0, [N/A], 0, 34, 0, 25.50, 210, 5001\n";
        assert!(parse_csv(out).is_empty());
    }

    #[test]
    fn keeps_row_with_na_optional_fields() {
        let out = "00000000:41:00.0, 24576, [N/A], N/A, N/A, [N/A], N/A, N/A\n";
        let gpus = parse_csv(out);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].vram_total_bytes, 24576 * 1024 * 1024);
        assert_eq!(gpus[0].vram_used_bytes, 0);
        assert_eq!(gpus[0].temperature_c, None);
        assert_eq!(gpus[0].gpu_busy_pct, None);
        assert_eq!(gpus[0].power_w, None);
        assert_eq!(gpus[0].sclk_mhz, None);
    }

    #[test]
    fn skips_malformed_rows() {
        let out = "garbage line\n00000000:41:00.0, 24576\n\n";
        assert!(parse_csv(out).is_empty());
    }

    #[test]
    fn empty_output_yields_no_gpus() {
        assert!(parse_csv("").is_empty());
        assert!(parse_csv("\n\n").is_empty());
    }

    #[test]
    fn normalizes_pci_slot_forms() {
        assert_eq!(
            normalize_pci_slot("00000000:65:00.0").as_deref(),
            Some("0000:65:00.0")
        );
        assert_eq!(
            normalize_pci_slot("0000:65:00.0").as_deref(),
            Some("0000:65:00.0")
        );
        assert!(normalize_pci_slot("not-a-slot").is_none());
        assert!(normalize_pci_slot("zzzz:65:00.0").is_none());
    }

    #[test]
    fn blocking_query_does_not_panic_without_nvidia_smi() {
        // On hosts without nvidia-smi (CI, dev machines) this must
        // degrade to an empty vec; on NVIDIA hosts it returns real data.
        let _ = read_all_nvidia_blocking();
    }

    #[tokio::test]
    async fn async_query_does_not_panic_without_nvidia_smi() {
        let _ = read_all_nvidia().await;
    }
}
