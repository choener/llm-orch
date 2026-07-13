use crate::config::PortRange;
use std::collections::HashSet;
use std::net::TcpListener;
use tokio::net::TcpListener as AsyncTcpListener;

// ── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum PortError {
    #[error("no free ports available in configured range")]
    Exhausted,

    #[error("failed to bind port {0}: {1}")]
    BindFailed(u16, std::io::Error),
}

// ── Port allocator ───────────────────────────────────────────────────────────

/// Manages a pool of ports for backend instances.
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

    /// Allocate a free port.  Returns the port number.
    ///
    /// * **Range mode**: scans the configured range, skipping ports already in
    ///   use and ports that fail a quick TCP bind check.
    /// * **Ephemeral mode**: asks the OS for a free port by binding to 0,
    ///   reads the assigned port, then releases the socket.
    pub async fn allocate(&mut self) -> Result<u16, PortError> {
        match &self.range {
            PortRange::Range { start, end } => self.allocate_range(*start, *end).await,
            PortRange::EphemeralWord(_) => self.allocate_ephemeral().await,
        }
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

    // ── private ──────────────────────────────────────────────────────────

    async fn allocate_range(&mut self, start: u16, end: u16) -> Result<u16, PortError> {
        if start > end || start == 0 {
            return Err(PortError::Exhausted);
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
            return Ok(candidate);
        }

        Err(PortError::Exhausted)
    }

    async fn allocate_ephemeral(&self) -> Result<u16, PortError> {
        // Bind to port 0, read the assigned port, close the socket.
        let listener = AsyncTcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| PortError::BindFailed(0, e))?;
        let addr = listener.local_addr().map_err(|e| PortError::BindFailed(0, e))?;
        drop(listener);
        Ok(addr.port())
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

    #[tokio::test]
    async fn allocate_and_free_range() {
        let range = PortRange::Range {
            start: 12000,
            end: 12002,
        };
        let mut alloc = PortAllocator::new(range);

        let p1 = alloc.allocate().await.unwrap();
        let p2 = alloc.allocate().await.unwrap();
        let p3 = alloc.allocate().await.unwrap();
        assert_eq!(alloc.used_count(), 3);
        assert_ne!(p1, p2);
        assert_ne!(p2, p3);

        // Range is full — next allocate should fail.
        assert!(alloc.allocate().await.is_err());

        // Free one and reallocate.
        alloc.free(p2);
        let p4 = alloc.allocate().await.unwrap();
        assert_eq!(p4, p2); // should get the same port back
    }

    #[tokio::test]
    async fn ephemeral_allocates() {
        let range = PortRange::EphemeralWord("ephemeral".into());
        let mut alloc = PortAllocator::new(range);
        let p = alloc.allocate().await.unwrap();
        assert!(p > 0);
        // Ephemeral ports are not tracked in `used`, so free is a no-op
        // but shouldn't panic.
        alloc.free(p);
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