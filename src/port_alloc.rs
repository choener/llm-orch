use crate::config::PortRange;
use std::collections::HashSet;
use std::net::TcpListener;
use tokio::net::TcpListener as AsyncTcpListener;

// ── Port allocator ───────────────────────────────────────────────────────────

/// Manages a pool of ports for backend instances.
///
/// The allocation API is split into synchronous and async parts so that the
/// `MutexGuard` (from `std::sync::Mutex<PortAllocator>`) can be dropped before
/// any `.await` — critical for keeping handler futures `Send`.
pub struct PortAllocator {
    /// All ports currently in use.
    used: HashSet<u16>,

    /// Configured allocation strategy.
    range: PortRange,

    /// Next candidate port for sequential scan (range mode only).
    next: u16,
}

impl PortAllocator {
    /// Create a new allocator for the given port range.
    pub fn new(range: PortRange) -> Self {
        let start = match &range {
            PortRange::Range { start, .. } => *start,
            PortRange::EphemeralWord(_) => 0,
        };
        Self {
            used: HashSet::new(),
            range,
            next: start,
        }
    }

    /// Try to allocate a port from the configured range (synchronous).
    ///
    /// Returns `Some(port)` on success, `None` if no ports are available
    /// or the range is exhausted.  The port is marked as used.
    ///
    /// This method does a quick TCP bind check to avoid handing out ports
    /// that are already in use by other processes.
    pub fn allocate_range_sync(&mut self) -> Option<u16> {
        let (start, end) = match &self.range {
            PortRange::Range { start, end } => (*start, *end),
            _ => return None,
        };

        if start > end || start == 0 {
            return None;
        }

        let range_size = (end - start + 1) as usize;
        for _ in 0..range_size {
            if self.next > end {
                self.next = start;
            }
            let candidate = self.next;
            self.next = candidate.wrapping_add(1);

            if self.used.contains(&candidate) {
                continue;
            }
            if !port_is_free(candidate) {
                continue;
            }

            self.used.insert(candidate);
            return Some(candidate);
        }

        None
    }

    /// Try to allocate an ephemeral port (async).
    ///
    /// Returns `Some(port)` on success.  Ephemeral ports are not tracked
    /// in the `used` set — the OS manages them.
    ///
    /// Call this **outside** any `MutexGuard` scope — it's async.
    pub async fn allocate_ephemeral_async(&self) -> Option<u16> {
        let listener = AsyncTcpListener::bind("127.0.0.1:0").await.ok()?;
        let addr = listener.local_addr().ok()?;
        drop(listener);
        Some(addr.port())
    }

    /// Whether this allocator uses ephemeral ports.
    pub fn is_ephemeral(&self) -> bool {
        matches!(self.range, PortRange::EphemeralWord(_))
    }

    /// Release a port back to the pool.
    pub fn free(&mut self, port: u16) {
        self.used.remove(&port);
    }

    /// Number of ports currently in use.
    #[allow(dead_code)]
    pub fn used_count(&self) -> usize {
        self.used.len()
    }
}

/// Quick synchronous check: is a TCP port currently free?
fn port_is_free(port: u16) -> bool {
    let addr = format!("127.0.0.1:{port}");
    // Try to bind synchronously — fast and simple.
    TcpListener::bind(&addr).is_ok()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_free_range() {
        let range = PortRange::Range {
            start: 12000,
            end: 12002,
        };
        let mut alloc = PortAllocator::new(range);

        let p1 = alloc.allocate_range_sync().unwrap();
        let p2 = alloc.allocate_range_sync().unwrap();
        let p3 = alloc.allocate_range_sync().unwrap();
        assert_eq!(alloc.used_count(), 3);
        assert_ne!(p1, p2);
        assert_ne!(p2, p3);

        // Range is full — next allocate should fail.
        assert!(alloc.allocate_range_sync().is_none());

        // Free one and reallocate.
        alloc.free(p2);
        let p4 = alloc.allocate_range_sync().unwrap();
        assert_eq!(p4, p2); // should get the same port back
    }

    #[tokio::test]
    async fn ephemeral_allocates() {
        let range = PortRange::EphemeralWord("ephemeral".into());
        let alloc = PortAllocator::new(range);
        let p = alloc.allocate_ephemeral_async().await.unwrap();
        assert!(p > 0);
        // Ephemeral ports are not tracked in `used`, so free is a no-op
        // but shouldn't panic.
        // (We can't call free without &mut, but that's fine — the test
        // just verifies the allocation succeeded.)
    }

    #[test]
    fn port_is_free_detects_used_port() {
        // Bind a port, check it's not free, then release it.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!port_is_free(port));
        drop(listener);
        assert!(port_is_free(port));
    }
}
