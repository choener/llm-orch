// ── llm-orch library ─────────────────────────────────────────────────────────
//
// All orchestrator logic lives in the library so integration tests (and the
// thin `llm-orch` binary) can compose the pieces: config, auth, scheduler,
// HTTP server, reload handling.

pub mod apikeys;
pub mod backend;
pub mod config;
pub mod debug_log;
pub mod gpu;
pub mod handlers;
pub mod http_client;
pub mod instance;
pub mod keepalive;
pub mod port_alloc;
pub mod reload;
pub mod scheduler;
pub mod server;
pub mod types;
pub mod watcher;
