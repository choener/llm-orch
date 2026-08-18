// ── File watcher for config hot-reload ──────────────────────────────────────
//
// Uses the `notify` crate to watch the main config file and the apikeys file.
// On each change, an event is sent through a tokio broadcast channel so the
// main loop can trigger an atomic reload.

use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::broadcast;
use tracing::debug;

/// A debounced file-watch event.
#[derive(Debug, Clone)]
pub struct FileChange {
    /// Which file changed.
    pub path: PathBuf,
}

/// Handle that keeps a watcher alive.  Drop it to stop watching.
pub struct WatchHandle {
    _watcher: Box<dyn Watcher + Send>,
    _debounce_task: tokio::task::JoinHandle<()>,
}

/// Start watching `paths` and send change events on `tx`.
///
/// Events are debounced: multiple writes within `debounce` are collapsed
/// into a single event per path.
///
/// Returns a `WatchHandle` that must be kept alive while watching is desired;
/// dropping it stops the watcher.
pub async fn watch_paths(
    paths: Vec<PathBuf>,
    tx: broadcast::Sender<FileChange>,
    debounce: Duration,
) -> notify::Result<WatchHandle> {
    // Canonicalize watched paths upfront so we can match exactly.
    let mut canonical: Vec<PathBuf> = Vec::with_capacity(paths.len());
    for p in &paths {
        match p.canonicalize() {
            Ok(c) => canonical.push(c),
            Err(_) => canonical.push(p.clone()),
        }
    }
    let paths_clone = paths.clone();

    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<PathBuf>();

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
        match res {
            Ok(event) => {
                debug!(?event.kind, ?event.paths, "raw notify event");
                if matches!(
                    event.kind,
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                ) {
                    for path in &event.paths {
                        // Canonicalize and match against the watched set.
                        let candidate = match path.canonicalize() {
                            Ok(c) => c,
                            Err(_) => continue,
                        };
                        if canonical.contains(&candidate) || paths_clone.contains(path) {
                            debug!("matched watched file: {}", candidate.display());
                            if event_tx.send(candidate).is_err() {
                                break;
                            }
                        }
                    }
                }
            }
            Err(e) => {
                debug!("notify error: {}", e);
            }
        }
    })?;

    for path in &paths {
        if let Some(parent) = path.parent() {
            watcher.watch(parent, RecursiveMode::NonRecursive)?;
        }
        if path.exists() {
            watcher.watch(path, RecursiveMode::NonRecursive)?;
        }
    }

    // Debounce task
    let debounce_task = tokio::spawn(async move {
        let mut pending: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
        loop {
            match event_rx.recv().await {
                Some(path) => {
                    debug!("debounce: received {}", path.display());
                    pending.insert(path);
                    loop {
                        match tokio::time::timeout(debounce, event_rx.recv()).await {
                            Ok(Some(p)) => {
                                debug!("debounce: burst {}", p.display());
                                pending.insert(p);
                            }
                            _ => break,
                        }
                    }
                    for p in pending.drain() {
                        debug!("debounce: broadcasting {}", p.display());
                        let _ = tx.send(FileChange { path: p });
                    }
                }
                None => {
                    debug!("debounce: channel closed");
                    break;
                }
            }
        }
    });

    Ok(WatchHandle {
        _watcher: Box::new(watcher),
        _debounce_task: debounce_task,
    })
}
