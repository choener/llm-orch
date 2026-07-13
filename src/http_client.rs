use reqwest::Client;
use std::time::Duration;

/// Build a shared `reqwest::Client` tuned for forwarding requests to backend
/// instances (llama.cpp, etc.).
///
/// * Connection pooling is enabled by default in `reqwest`.
/// * A short connect timeout so dead backends fail fast.
/// * **No** overall read timeout — streaming responses may last minutes.
pub fn build() -> Client {
    Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .expect("reqwest::Client::builder() should never fail with these options")
}