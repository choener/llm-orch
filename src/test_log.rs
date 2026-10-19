// ── Test-only log capture ────────────────────────────────────────────────────
//
// Lets a test assert that something was actually *logged*, not just returned.
// Observability bugs (an error path that reports a status code but never says
// why) are invisible to return-value assertions.
//
// The capture installs a thread-local dispatcher (`set_default`), so it is safe
// under the parallel test harness and never leaks into other tests' threads.

use std::io::Write;
use std::sync::{Arc, Mutex};
use tracing::dispatcher::{DefaultGuard, Dispatch};

/// A shared in-memory log buffer plus the guard that keeps it installed.
///
/// Drop it to restore the previous dispatcher.
pub struct Capture {
    _guard: DefaultGuard,
    bytes: Arc<Mutex<Vec<u8>>>,
}

#[derive(Clone)]
struct BufferWriter(Arc<Mutex<Vec<u8>>>);

impl Write for BufferWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for BufferWriter {
    type Writer = BufferWriter;
    fn make_writer(&'a self) -> BufferWriter {
        self.clone()
    }
}

impl Capture {
    /// Install a capturing subscriber for the current thread.
    pub fn install() -> Capture {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(BufferWriter(bytes.clone()))
            .with_ansi(false)
            .with_max_level(tracing::Level::TRACE)
            .finish();
        Capture {
            _guard: tracing::dispatcher::set_default(&Dispatch::new(subscriber)),
            bytes,
        }
    }

    /// Every log line captured so far.
    pub fn lines(&self) -> Vec<String> {
        let taken = std::mem::take(&mut *self.bytes.lock().unwrap());
        String::from_utf8_lossy(&taken)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(str::to_owned)
            .collect()
    }
}
