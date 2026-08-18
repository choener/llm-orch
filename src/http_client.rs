use reqwest::Client;
use std::time::Duration;

/// Build a shared `reqwest::Client` tuned for forwarding requests to backend
/// instances (llama.cpp, etc.).
///
/// * Connection pooling is enabled by default in `reqwest`.
/// * A short connect timeout so dead backends fail fast.
/// * **No** overall read timeout at the client level — streaming responses
///   may last minutes.  Per-request total/idle timeouts are enforced by
///   the handlers instead (`forward_request_aggregate`, `SseForwarder`),
///   configured via `server.backend_total_timeout_secs` and
///   `server.backend_idle_timeout_secs`.
pub fn build() -> Client {
    Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .expect("reqwest::Client::builder() should never fail with these options")
}
