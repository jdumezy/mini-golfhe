//! mini-golfhe client: holds the secret key. The browser sends it a shot; it
//! encrypts it, has the server simulate it, and decrypts the contacts of the
//! trajectory. The browser draws the path between them.

mod scheme;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use mini_golfhe_course::{Course, inside_polygon};
use mini_golfhe_protocol::{DEFAULT_CLIENT_ADDR, DEFAULT_SERVER_ADDR, Request, Response, read_frame, write_frame};
use scheme::{ClientScheme, Plaintext};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::net::TcpStream;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

struct App {
    scheme: Box<dyn ClientScheme>,
    server_addr: String,
    keygen_ms: f64,
    course: &'static Course,
}

type ApiError = (StatusCode, Json<Value>);

fn error(status: StatusCode, message: impl ToString) -> ApiError {
    (status, Json(json!({ "error": message.to_string() })))
}

/// One request/response round trip with the server, with sizes on the wire.
struct Exchange {
    response: Response,
    sent: usize,
    received: usize,
    elapsed: Duration,
}

async fn exchange(addr: &str, request: &Request) -> std::io::Result<Exchange> {
    let start = Instant::now();
    let mut stream = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(addr))
        .await
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "connection timed out"))??;
    let sent = write_frame(&mut stream, request).await?;
    let (response, received) = read_frame(&mut stream).await?;
    Ok(Exchange { response, sent, received, elapsed: start.elapsed() })
}

// --- GET /api/status -------------------------------------------------------------

#[derive(Serialize)]
struct Status {
    client: ClientStatus,
    server: ServerStatus,
}

#[derive(Serialize)]
struct ClientStatus {
    version: &'static str,
    scheme: &'static str,
    keygen_ms: f64,
    eval_key_bytes: usize,
}

#[derive(Serialize)]
struct ServerStatus {
    addr: String,
    reachable: bool,
    max_contacts: Option<u32>,
    steps: Option<u32>,
    dt: Option<f64>,
    error: Option<String>,
}

async fn status(State(app): State<Arc<App>>) -> Json<Status> {
    let client = ClientStatus {
        version: env!("CARGO_PKG_VERSION"),
        scheme: app.scheme.scheme().name(),
        keygen_ms: app.keygen_ms,
        eval_key_bytes: app.scheme.eval_key().len(),
    };
    let mut server = ServerStatus {
        addr: app.server_addr.clone(),
        reachable: false,
        max_contacts: None,
        steps: None,
        dt: None,
        error: None,
    };
    match exchange(&app.server_addr, &Request::Info).await {
        Ok(Exchange { response: Response::Info(info), .. }) => {
            server.reachable = true;
            server.max_contacts = Some(info.max_contacts);
            server.steps = Some(info.steps);
            server.dt = Some(info.dt);
            if info.course_hash != app.course.hash {
                server.error = Some("the server simulates a different course".into());
            }
        }
        Ok(_) => server.error = Some("unexpected response".into()),
        Err(e) => server.error = Some(e.to_string()),
    }
    Json(Status { client, server })
}

// --- POST /api/shot --------------------------------------------------------------

#[derive(Deserialize)]
struct ShotRequest {
    /// Hole index, from 0.
    hole: usize,
    /// x, y, vx, vy.
    shot: [f64; 4],
}

#[derive(Serialize)]
struct Segment {
    t: f64,
    state: [f64; 4],
}

#[derive(Serialize)]
struct ShotResponse {
    /// The shot, then the state after each contact.
    segments: Vec<Segment>,
    /// Whether all contacts were spent: the last segment was not searched.
    exhausted: bool,
    stats: ShotStats,
}

#[derive(Serialize)]
struct ShotStats {
    scheme: &'static str,
    slots: usize,
    contacts: usize,
    eval_key_bytes: usize,
    shot_bytes: usize,
    trajectory_bytes: usize,
    upload_bytes: usize,
    download_bytes: usize,
    encrypt_ms: f64,
    simulate_ms: f64,
    network_ms: f64,
    decrypt_ms: f64,
    total_ms: f64,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The shot must start on the green, no faster than the fastest shot.
fn validate(course: &Course, req: &ShotRequest) -> Result<(), String> {
    let hole = course.holes.get(req.hole).ok_or_else(|| format!("no hole {}", req.hole))?;
    let [x, y, vx, vy] = req.shot;
    if req.shot.iter().any(|v| !v.is_finite()) {
        return Err("the shot is not finite".into());
    }
    if vx.hypot(vy) > course.physics.max_speed * (1.0 + 1e-9) {
        return Err(format!("the shot is faster than {} m/s", course.physics.max_speed));
    }
    if hole.obstacles.iter().any(|o| o.touches(x, y)) || hole.spec.water.iter().any(|w| inside_polygon(w, x, y)) {
        return Err("the ball is not on the green".into());
    }
    Ok(())
}

async fn shot(State(app): State<Arc<App>>, Json(req): Json<ShotRequest>) -> Result<Json<ShotResponse>, ApiError> {
    let start = Instant::now();
    validate(app.course, &req).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let max_contacts = app.course.physics.max_contacts;

    let t = Instant::now();
    let ciphertext = app.scheme.encrypt_shot(&req.shot);
    let encrypt = t.elapsed();
    let shot_bytes = ciphertext.len();

    let request = Request::Simulate {
        scheme: app.scheme.scheme(),
        eval_key: app.scheme.eval_key().to_vec(),
        hole: req.hole as u32,
        shot: ciphertext,
    };
    let ex = exchange(&app.server_addr, &request)
        .await
        .map_err(|e| error(StatusCode::BAD_GATEWAY, format!("server unreachable at {}: {e}", app.server_addr)))?;
    let (encrypted, simulate_micros) = match ex.response {
        Response::Trajectory { contacts, simulate_micros } => (contacts, simulate_micros),
        Response::Error(e) => return Err(error(StatusCode::BAD_GATEWAY, format!("server: {e}"))),
        Response::Info(_) => return Err(error(StatusCode::BAD_GATEWAY, "unexpected response")),
    };

    let t = Instant::now();
    let contacts =
        app.scheme.decrypt_contacts(&encrypted, max_contacts).map_err(|e| error(StatusCode::BAD_GATEWAY, e))?;
    let decrypt = t.elapsed();

    // Contacts are in order; the first empty slot ends the trajectory.
    let used: Vec<_> = contacts.iter().take_while(|c| c.hit).collect();
    let mut segments = vec![Segment { t: 0.0, state: req.shot }];
    segments.extend(used.iter().map(|c| Segment { t: c.t, state: c.state }));
    let simulate = Duration::from_micros(simulate_micros);

    let stats = ShotStats {
        scheme: app.scheme.scheme().name(),
        slots: contacts.len(),
        contacts: used.len(),
        eval_key_bytes: app.scheme.eval_key().len(),
        shot_bytes,
        trajectory_bytes: encrypted.len(),
        upload_bytes: ex.sent,
        download_bytes: ex.received,
        encrypt_ms: ms(encrypt),
        simulate_ms: ms(simulate),
        network_ms: ms(ex.elapsed.saturating_sub(simulate)),
        decrypt_ms: ms(decrypt),
        total_ms: ms(start.elapsed()),
    };
    eprintln!("hole {}: {} contacts ({:.1} ms)", req.hole + 1, used.len(), stats.total_ms);
    Ok(Json(ShotResponse { segments, exhausted: used.len() == max_contacts, stats }))
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let addr = arg("--listen").unwrap_or_else(|| DEFAULT_CLIENT_ADDR.to_string());
    let server_addr = arg("--server").unwrap_or_else(|| DEFAULT_SERVER_ADDR.to_string());

    let t = Instant::now();
    let scheme: Box<dyn ClientScheme> = Box::new(Plaintext);
    let keygen_ms = ms(t.elapsed());
    eprintln!("{} scheme ready, evaluation key: {} B", scheme.scheme().name(), scheme.eval_key().len());

    let app = Arc::new(App { scheme, server_addr, keygen_ms, course: Course::get() });
    let router = Router::new().route("/api/status", get(status)).route("/api/shot", post(shot)).with_state(app.clone());

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("listening on {addr}, server at {}", app.server_addr);
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await
}

/// Value of a `--name value` command-line argument.
fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
