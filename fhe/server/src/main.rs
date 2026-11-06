//! mini-golfhe server: receives an evaluation key and an encrypted shot,
//! computes the encrypted contacts of its trajectory. It never holds a secret
//! key.

use std::sync::Arc;
use std::time::Instant;

use mini_golfhe_course::Course;
use mini_golfhe_physics::{Flow, simulate};
use mini_golfhe_protocol::{
    DEFAULT_SERVER_ADDR, PROTOCOL_VERSION, Request, Response, Scheme, ServerInfo, plaintext, read_frame, write_frame,
};
use tokio::net::{TcpListener, TcpStream};

struct Server {
    course: &'static Course,
    /// One flow per hole, by hole index.
    flows: Vec<Flow>,
}

impl Server {
    fn info(&self) -> ServerInfo {
        let p = &self.course.physics;
        ServerInfo {
            version: PROTOCOL_VERSION,
            schemes: vec![Scheme::Plaintext],
            course_hash: self.course.hash,
            holes: self.course.holes.len() as u32,
            max_contacts: p.max_contacts as u32,
            steps: p.steps() as u32,
            dt: p.dt,
        }
    }

    fn simulate(&self, scheme: Scheme, _eval_key: &[u8], hole: u32, shot: &[u8]) -> Response {
        let Some(h) = self.course.holes.get(hole as usize) else {
            return Response::Error(format!("no hole {hole}"));
        };
        let start = Instant::now();
        let contacts = match scheme {
            Scheme::Plaintext => match plaintext::decode_shot(shot) {
                Ok(s) => plaintext::encode_contacts(&simulate(h, &self.flows[h.index], &self.course.physics, &s)),
                Err(e) => return Response::Error(e),
            },
        };
        Response::Trajectory { contacts, simulate_micros: start.elapsed().as_micros() as u64 }
    }
}

async fn handle(server: Arc<Server>, mut stream: TcpStream) -> std::io::Result<()> {
    let peer = stream.peer_addr()?;
    loop {
        let (request, size) = match read_frame::<_, Request>(&mut stream).await {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        let response = match request {
            Request::Info => Response::Info(server.info()),
            Request::Simulate { scheme, eval_key, hole, shot } => {
                let server = server.clone();
                let response = tokio::task::spawn_blocking(move || server.simulate(scheme, &eval_key, hole, &shot))
                    .await
                    .map_err(std::io::Error::other)?;
                if let Response::Trajectory { simulate_micros, .. } = &response {
                    eprintln!("{peer}: {} shot on hole {}, {size} B in, {simulate_micros} µs", scheme.name(), hole + 1);
                }
                response
            }
        };
        write_frame(&mut stream, &response).await?;
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let addr = arg("--listen").unwrap_or_else(|| DEFAULT_SERVER_ADDR.to_string());

    let start = Instant::now();
    let course = Course::get();
    let flows = course.holes.iter().map(|h| Flow::new(&h.terrain, &course.physics)).collect();
    let server = Arc::new(Server { course, flows });
    eprintln!(
        "course ready in {:.2?}: {} holes, {} contacts of {} steps per shot",
        start.elapsed(),
        course.holes.len(),
        course.physics.max_contacts,
        course.physics.steps()
    );

    let listener = TcpListener::bind(&addr).await?;
    eprintln!("listening on {addr}");
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                let server = server.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle(server, stream).await {
                        eprintln!("connection error: {e}");
                    }
                });
            }
            _ = tokio::signal::ctrl_c() => return Ok(()),
        }
    }
}

/// Value of a `--name value` command-line argument.
fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
