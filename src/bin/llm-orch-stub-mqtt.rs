// ── Stub MQTT broker (test support) ────────────────────────────────────────
//
// A minimal MQTT 3.1.1 broker for integration tests: accepts TCP
// connections, answers CONNECT with a successful CONNACK, captures every
// PUBLISH (replying PUBACK for QoS 1), answers PINGREQ, and closes on
// DISCONNECT.  Last Will messages are accepted but not fired (tests
// exercise clean sessions only).
//
// Captured messages are exposed over a second (HTTP) port:
//   GET  /messages[?topic=<exact topic>]  → JSON array of
//        {"topic": …, "payload": "…", "qos": …, "retain": …} (oldest first)
//   DELETE /messages                      → clear the capture buffer

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    routing::get,
};
use clap::Parser;
use serde::Serialize;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Parser, Debug)]
#[command(name = "llm-orch-stub-mqtt")]
struct Cli {
    /// MQTT (TCP) port to listen on.
    #[arg(long)]
    mqtt_port: u16,

    /// HTTP port exposing the captured messages.
    #[arg(long)]
    http_port: u16,
}

#[derive(Clone, Serialize)]
struct Captured {
    topic: String,
    payload: String,
    qos: u8,
    retain: bool,
}

#[derive(Clone)]
struct BrokerState {
    messages: Arc<Mutex<Vec<Captured>>>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let state = BrokerState {
        messages: Arc::new(Mutex::new(Vec::new())),
    };

    let app = Router::new()
        .route("/messages", get(get_messages).delete(clear_messages))
        .with_state(state.clone());

    let http_listener = tokio::net::TcpListener::bind(("127.0.0.1", cli.http_port)).await?;
    let mqtt_listener = tokio::net::TcpListener::bind(("127.0.0.1", cli.mqtt_port)).await?;

    tokio::spawn(async move {
        axum::serve(http_listener, app).await.ok();
    });

    while let Ok((stream, _peer)) = mqtt_listener.accept().await {
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = serve_mqtt_connection(stream, state).await {
                eprintln!("stub-mqtt: connection closed: {e}");
            }
        });
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct MessageQuery {
    topic: Option<String>,
}

async fn get_messages(
    State(state): State<BrokerState>,
    Query(q): Query<MessageQuery>,
) -> Json<Vec<Captured>> {
    let all = state.messages.lock().unwrap().clone();
    Json(match q.topic {
        Some(t) => all.into_iter().filter(|m| m.topic == t).collect(),
        None => all,
    })
}

async fn clear_messages(State(state): State<BrokerState>) -> StatusCode {
    state.messages.lock().unwrap().clear();
    StatusCode::NO_CONTENT
}

// ── MQTT 3.1.1 minimal protocol ────────────────────────────────────────────

/// Broker connection handler error type.
type ConnError = Box<dyn std::error::Error + Send + Sync>;

async fn serve_mqtt_connection(
    mut stream: tokio::net::TcpStream,
    state: BrokerState,
) -> Result<(), ConnError> {
    loop {
        // Fixed header: 1 byte type/flags + remaining-length varint.
        let mut first = [0u8; 1];
        if stream.read_exact(&mut first).await.is_err() {
            return Ok(()); // client vanished
        }
        let packet_type = first[0] >> 4;
        let flags = first[0] & 0x0f;
        let remaining = read_varint(&mut stream).await?;
        let mut body = vec![0u8; remaining];
        if remaining > 0 {
            stream.read_exact(&mut body).await?;
        }

        match packet_type {
            // CONNECT → CONNACK accepted.
            1 => {
                stream.write_all(&[0x20, 0x02, 0x00, 0x00]).await?;
            }
            // PUBLISH → capture, PUBACK for QoS 1.
            3 => {
                let qos = (flags >> 1) & 0x03;
                let retain = flags & 0x01 != 0;
                let (topic, payload, packet_id) = parse_publish(&body, qos)?;
                if qos == 1
                    && let Some(pid) = packet_id
                {
                    stream
                        .write_all(&[0x40, 0x02, (pid >> 8) as u8, (pid & 0xff) as u8])
                        .await?;
                }
                state.messages.lock().unwrap().push(Captured {
                    topic,
                    payload: String::from_utf8_lossy(&payload).into_owned(),
                    qos,
                    retain,
                });
            }
            // SUBSCRIBE → SUBACK granting QoS 0 for every filter.
            8 => {
                if body.len() >= 2 {
                    let mut n_filters = 0usize;
                    let mut rest = &body[2..];
                    while rest.len() >= 2 {
                        let tlen = u16::from_be_bytes([rest[0], rest[1]]) as usize;
                        let after_topic = 2 + tlen + 1; // + subscription options byte
                        if rest.len() < after_topic {
                            break;
                        }
                        rest = &rest[after_topic..];
                        n_filters += 1;
                    }
                    let mut resp = vec![0x90, (2 + n_filters) as u8, body[0], body[1]];
                    resp.extend(std::iter::repeat_n(0x00u8, n_filters));
                    stream.write_all(&resp).await?;
                }
            }
            // UNSUBSCRIBE → UNSUBACK.
            10 => {
                if body.len() >= 2 {
                    stream.write_all(&[0xb0, 0x02, body[0], body[1]]).await?;
                }
            }
            // PINGREQ → PINGRESP.
            12 => {
                stream.write_all(&[0xd0, 0x00]).await?;
            }
            // DISCONNECT.
            14 => return Ok(()),
            _ => {}
        }
    }
}

async fn read_varint(stream: &mut tokio::net::TcpStream) -> Result<usize, std::io::Error> {
    let mut multiplier = 1usize;
    let mut value = 0usize;
    loop {
        let mut b = [0u8; 1];
        stream.read_exact(&mut b).await?;
        value += (b[0] & 0x7f) as usize * multiplier;
        if b[0] & 0x80 == 0 {
            return Ok(value);
        }
        multiplier *= 128;
        if multiplier > 128 * 128 * 128 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "malformed remaining length",
            ));
        }
    }
}

fn parse_publish(body: &[u8], qos: u8) -> Result<(String, Vec<u8>, Option<u16>), ConnError> {
    if body.len() < 2 {
        return Err("truncated PUBLISH".into());
    }
    let tlen = u16::from_be_bytes([body[0], body[1]]) as usize;
    if body.len() < 2 + tlen {
        return Err("truncated PUBLISH topic".into());
    }
    let topic = String::from_utf8_lossy(&body[2..2 + tlen]).into_owned();
    let mut offset = 2 + tlen;
    let mut packet_id = None;
    if qos > 0 {
        if body.len() < offset + 2 {
            return Err("truncated PUBLISH packet id".into());
        }
        packet_id = Some(u16::from_be_bytes([body[offset], body[offset + 1]]));
        offset += 2;
    }
    Ok((topic, body[offset..].to_vec(), packet_id))
}
