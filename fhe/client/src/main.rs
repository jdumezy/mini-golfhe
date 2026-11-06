//! mini-golfhe client: holds the secret key. The browser sends it a shot; it
//! encrypts it, has the server simulate it, and decrypts the contacts of the
//! trajectory. The browser draws the path between them.

mod ckks;

use std::io::{BufReader, BufWriter};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use ckks::{CkksClient, CkksStats, Phase};
use mini_golfhe_ckks::Params;
use mini_golfhe_course::{Course, inside_polygon};
use mini_golfhe_protocol::{
    DEFAULT_CLIENT_ADDR, DEFAULT_SERVER_ADDR, Request, Response, Scheme, ServerInfo, SimulateProgress, plaintext,
    read_frame, write_frame,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

struct App {
    scheme: Scheme,
    server_addr: String,
    ckks: Option<Arc<CkksClient>>,
    course: &'static Course,
}

type ApiError = (StatusCode, Json<Value>);

fn error(status: StatusCode, message: impl std::fmt::Display) -> ApiError {
    (status, Json(json!({ "error": format!("{message:#}") })))
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

fn connect(addr: &str) -> Result<TcpStream> {
    let socket = addr.to_socket_addrs()?.next().context("no address")?;
    TcpStream::connect_timeout(&socket, CONNECT_TIMEOUT).with_context(|| format!("server unreachable at {addr}"))
}

/// One request/response round trip with the server, with sizes on the wire.
pub struct Exchange {
    response: Response,
    sent: usize,
    received: usize,
    elapsed_ms: f64,
}

fn exchange(addr: &str, request: &Request) -> Result<Exchange> {
    let start = Instant::now();
    let stream = connect(addr)?;
    let sent = write_frame(&mut BufWriter::new(stream.try_clone()?), request)?;
    let (response, received) = read_frame(&mut BufReader::new(stream))?;
    Ok(Exchange { response, sent, received, elapsed_ms: ms(start) })
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(f).await.expect("blocking task panicked")
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
    backend: &'static str,
    params: Option<String>,
    setup: Option<Phase>,
    keygen_ms: Option<f64>,
}

#[derive(Serialize)]
struct ServerStatus {
    addr: String,
    reachable: bool,
    info: Option<ServerInfo>,
    error: Option<String>,
}

async fn status(State(app): State<Arc<App>>) -> Json<Status> {
    let client = ClientStatus {
        version: env!("CARGO_PKG_VERSION"),
        scheme: app.scheme.name(),
        backend: mini_golfhe_ckks::backend::NAME,
        params: app.ckks.as_ref().map(|c| c.params_name.clone()),
        setup: app.ckks.as_ref().map(|c| c.phase()),
        keygen_ms: app.ckks.as_ref().and_then(|c| *c.keygen_ms.lock().unwrap()),
    };
    let (addr, hash) = (app.server_addr.clone(), app.course.hash);
    let server = blocking(move || match exchange(&addr, &Request::Info) {
        Ok(Exchange { response: Response::Info(info), .. }) => {
            let error = (info.course_hash != hash).then(|| "the server simulates a different course".to_string());
            ServerStatus { addr, reachable: true, info: Some(info), error }
        }
        Ok(_) => ServerStatus { addr, reachable: true, info: None, error: Some("unexpected response".into()) },
        Err(e) => ServerStatus { addr, reachable: false, info: None, error: Some(format!("{e:#}")) },
    })
    .await;
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
    /// Whether the segment starts at a contact (else it continues the last one).
    contact: bool,
}

#[derive(Serialize)]
struct ShotResponse {
    /// The shot, then one segment per contact the server searched.
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
    /// Each contact's decrypted contact weight: near 1 or 0, in between when the comparison was unsure.
    confidence: Vec<f64>,
    upload_bytes: usize,
    download_bytes: usize,
    simulate_ms: f64,
    network_ms: f64,
    total_ms: f64,
    #[serde(flatten)]
    ckks: Option<CkksStats>,
}

/// Time, state, whether there was a contact, and its decrypted weight.
type Decrypted = (f64, [f64; 4], bool, f64);

/// The shot must start on the green, no faster than the fastest shot.
fn validate(course: &Course, req: &ShotRequest) -> Result<()> {
    let hole = course.holes.get(req.hole).with_context(|| format!("no hole {}", req.hole))?;
    let [x, y, vx, vy] = req.shot;
    if req.shot.iter().any(|v| !v.is_finite()) {
        bail!("the shot is not finite");
    }
    if vx.hypot(vy) > course.physics.max_speed * (1.0 + 1e-9) {
        bail!("the shot is faster than {} m/s", course.physics.max_speed);
    }
    if hole.obstacles.iter().any(|o| o.touches(x, y)) || hole.spec.water.iter().any(|w| inside_polygon(w, x, y)) {
        bail!("the ball is not on the green");
    }
    Ok(())
}

fn shot_blocking(app: &App, req: &ShotRequest) -> Result<ShotResponse, ApiError> {
    let start = Instant::now();
    validate(app.course, req).map_err(|e| error(StatusCode::BAD_REQUEST, e))?;
    let gateway = |e: anyhow::Error| error(StatusCode::BAD_GATEWAY, e);
    let max_contacts = app.course.physics.max_contacts;

    let (contacts, ckks_stats, ex): (Vec<Decrypted>, Option<CkksStats>, Exchange) = match (app.scheme, &app.ckks) {
        (Scheme::Ckks, Some(ckks)) => {
            let (decoded, stats, ex) = ckks.shoot(&app.server_addr, req.hole, &req.shot).map_err(gateway)?;
            // The browser follows the grid: the contact steps are whole.
            let mut t = 0.0;
            let dt = app.course.physics.dt;
            let contacts = decoded
                .iter()
                .map(|c| {
                    t += (c.dt / dt).round() * dt;
                    (t, c.state, c.hit > 0.5, c.hit)
                })
                .collect();
            (contacts, Some(stats), ex)
        }
        _ => {
            let request = Request::Simulate {
                scheme: Scheme::Plaintext,
                key_id: 0,
                hole: req.hole as u32,
                shot: plaintext::encode_shot(&req.shot),
            };
            let ex = exchange(&app.server_addr, &request).map_err(gateway)?;
            let contacts = match &ex.response {
                Response::Trajectory { contacts, .. } => {
                    plaintext::decode_contacts(&contacts[0], max_contacts).map_err(|e| gateway(anyhow::anyhow!(e)))?
                }
                Response::Error(e) => return Err(gateway(anyhow::anyhow!("server: {e}"))),
                _ => return Err(gateway(anyhow::anyhow!("unexpected response"))),
            };
            let contacts = contacts.iter().map(|c| (c.t, c.state, c.hit, c.hit as u8 as f64)).collect();
            (contacts, None, ex)
        }
    };
    let Response::Trajectory { simulate_ms, .. } = &ex.response else { unreachable!() };

    let mut segments = vec![Segment { t: 0.0, state: req.shot, contact: false }];
    segments.extend(contacts.iter().map(|&(t, state, contact, _)| Segment { t, state, contact }));
    let stats = ShotStats {
        scheme: app.scheme.name(),
        slots: max_contacts,
        contacts: contacts.iter().filter(|c| c.2).count(),
        confidence: contacts.iter().map(|c| c.3).collect(),
        upload_bytes: ex.sent,
        download_bytes: ex.received,
        simulate_ms: *simulate_ms,
        network_ms: (ex.elapsed_ms - simulate_ms).max(0.0),
        total_ms: ms(start),
        ckks: ckks_stats,
    };
    eprintln!("hole {}: {} contacts ({:.0} ms)", req.hole + 1, stats.contacts, stats.total_ms);
    // After the last contact the server searched nothing.
    Ok(ShotResponse { segments, exhausted: true, stats })
}

async fn shot(State(app): State<Arc<App>>, Json(req): Json<ShotRequest>) -> Result<Json<ShotResponse>, ApiError> {
    blocking(move || shot_blocking(&app, &req)).await.map(Json)
}

// --- GET /api/progress -----------------------------------------------------------

#[derive(Serialize)]
struct Progress {
    step: &'static str,
    server: Option<SimulateProgress>,
}

async fn progress(State(app): State<Arc<App>>) -> Json<Progress> {
    let step = app.ckks.as_ref().map_or("idle", |c| *c.step.lock().unwrap());
    let server = if step == "evaluating" {
        let addr = app.server_addr.clone();
        blocking(move || match exchange(&addr, &Request::Progress) {
            Ok(Exchange { response: Response::Progress(p), .. }) => p,
            _ => None,
        })
        .await
    } else {
        None
    };
    Json(Progress { step, server })
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr = arg("--listen").unwrap_or_else(|| DEFAULT_CLIENT_ADDR.to_string());
    let server_addr = arg("--server").unwrap_or_else(|| DEFAULT_SERVER_ADDR.to_string());
    let scheme = match arg("--scheme").as_deref() {
        None | Some("ckks") => Scheme::Ckks,
        Some("plaintext") => Scheme::Plaintext,
        Some(other) => bail!("unknown scheme {other:?} (ckks or plaintext)"),
    };
    let ckks = (scheme == Scheme::Ckks).then(|| -> Result<_> {
        if !std::env::args().any(|a| a == "--allow-low-memory") {
            mini_golfhe_ckks::system::ensure_memory(mini_golfhe_ckks::system::CLIENT_MIN_MEMORY)?;
        }
        let params = Params::secure()?;
        let threads = mini_golfhe_ckks::system::init_threads();
        eprintln!(
            "{} backend ({threads} threads), {} parameters: preparing keys",
            mini_golfhe_ckks::backend::NAME,
            params.name
        );
        Ok(CkksClient::start(params, server_addr.clone()))
    });
    let ckks = ckks.transpose()?;

    let app = Arc::new(App { scheme, server_addr, ckks, course: Course::get() });
    let router = Router::new()
        .route("/api/status", get(status))
        .route("/api/shot", post(shot))
        .route("/api/progress", get(progress))
        .with_state(app.clone());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("listening on {addr}, server at {}, scheme {}", app.server_addr, app.scheme.name());
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await?;
    Ok(())
}

/// Value of a `--name value` command-line argument.
fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
