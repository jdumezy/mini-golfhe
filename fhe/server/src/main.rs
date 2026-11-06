//! mini-golfhe server: receives evaluation keys and encrypted shots, and
//! evaluates the contact circuit of the hole on them. It never holds a secret
//! key.

use std::io::{self, BufReader, BufWriter, Read};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result};
use mini_golfhe_circuit::{Event, Plan, Prepared, evaluate};
use mini_golfhe_ckks::wire::{ciphertext_from_bytes, ciphertext_to_bytes};
use mini_golfhe_ckks::{CkksEngine, Params, Ring, backend, system};
use mini_golfhe_course::Course;
use mini_golfhe_physics::{Flow, simulate};
use mini_golfhe_protocol::{
    DEFAULT_SERVER_ADDR, PROTOCOL_VERSION, Request, Response, Scheme, ServerInfo, SimulateProgress, Step, plaintext,
    read_frame, write_frame,
};

struct Server {
    course: &'static Course,
    /// One flow per hole, by hole index.
    flows: Vec<Flow>,
    ckks: Ckks,
    progress: Mutex<Option<SimulateProgress>>,
}

struct Ckks {
    /// Every hole's circuit, its plaintexts encoded on the device.
    prepared: Vec<Prepared<CkksEngine>>,
    engine: RwLock<CkksEngine>,
    key_id: Mutex<Option<u64>>,
}

impl Server {
    fn info(&self) -> ServerInfo {
        let p = &self.course.physics;
        ServerInfo {
            version: PROTOCOL_VERSION,
            backend: backend::DEVICE_NAME.to_string(),
            ckks_params: self.ckks.engine.read().unwrap().ring.params.name.clone(),
            key_id: *self.ckks.key_id.lock().unwrap(),
            course_hash: self.course.hash,
            max_contacts: p.max_contacts as u32,
            steps: p.steps() as u32,
            dt: p.dt,
        }
    }

    fn install_keys(&self, reader: &mut impl Read) -> Result<Response> {
        let start = Instant::now();
        // Drop the current keys first: two prepared sets would not fit side by side.
        *self.ckks.key_id.lock().unwrap() = None;
        self.ckks.engine.write().unwrap().set_keys(None);

        // Each key is prepared as it arrives; receiving and preparing overlap.
        let mut counting = CountingReader { inner: reader, bytes: 0 };
        let keys = self.ckks.engine.read().unwrap().prepare_keys_from(&mut counting).context("receiving keys")?;
        self.ckks.engine.write().unwrap().set_keys(Some(keys));
        if backend::ON_GPU {
            let warm = Instant::now();
            self.ckks.engine.read().unwrap().warm_up(&self.ckks.prepared[0]).context("compiling the device kernels")?;
            eprintln!("device kernels compiled in {:.0} ms", ms(warm));
        }
        let key_id = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_nanos() as u64;
        *self.ckks.key_id.lock().unwrap() = Some(key_id);
        let elapsed = ms(start);
        eprintln!(
            "installed evaluation keys {key_id:x}: {:.2} GB received and prepared in {elapsed:.0} ms",
            counting.bytes as f64 / 1e9
        );
        Ok(Response::KeysInstalled { key_id, bytes: counting.bytes, prepare_ms: elapsed })
    }

    fn simulate(&self, scheme: Scheme, key_id: u64, hole: u32, shot: &[u8]) -> Result<Response> {
        let h = self.course.holes.get(hole as usize).with_context(|| format!("no hole {hole}"))?;
        let start = Instant::now();
        match scheme {
            Scheme::Plaintext => {
                let s = plaintext::decode_shot(shot).map_err(anyhow::Error::msg)?;
                let contacts = simulate(h, &self.flows[h.index], &self.course.physics, &s);
                Ok(Response::Trajectory {
                    contacts: vec![plaintext::encode_contacts(&contacts)],
                    simulate_ms: ms(start),
                    refreshes: 0,
                })
            }
            Scheme::Ckks => {
                anyhow::ensure!(
                    *self.ckks.key_id.lock().unwrap() == Some(key_id),
                    "unknown evaluation keys: upload them again"
                );
                let engine = self.ckks.engine.read().unwrap();
                let shot = engine.upload(&ciphertext_from_bytes(&engine.ring, shot)?);
                let prepared = &self.ckks.prepared[h.index];
                let contacts = prepared.max_contacts as u32;
                // Two halves per contact, the second ending with the next state's bootstrap.
                let total = 2.0 * contacts as f64;
                let mut refreshes = 0;
                let report = |contact: usize, step: Step, done: f64| {
                    *self.progress.lock().unwrap() = Some(SimulateProgress {
                        contact: contact as u32,
                        contacts,
                        step,
                        fraction: (done / total).min(0.99),
                        elapsed_ms: ms(start),
                    });
                };
                let result = evaluate(&*engine, prepared, &shot, |e| match e {
                    Event::Contact { index } => report(index, Step::Gaps, 2.0 * index as f64),
                    Event::Refresh { index, state } => {
                        refreshes += 1;
                        report(index, Step::Bootstrapping, 2.0 * index as f64 + if state { 1.9 } else { 0.9 });
                        if !state {
                            report(index, Step::Contact, 2.0 * index as f64 + 1.0)
                        }
                    }
                    Event::Free { .. } => {}
                });
                report(contacts as usize, Step::Output, total);
                let finished = result.and_then(|outputs| {
                    outputs.iter().map(|c| Ok(ciphertext_to_bytes(&engine.download(c)?)?)).collect::<Result<Vec<_>>>()
                });
                *self.progress.lock().unwrap() = None;
                let contacts = finished?;
                eprintln!("ckks shot on hole {} in {:.0} ms ({refreshes} bootstraps)", hole + 1, ms(start));
                Ok(Response::Trajectory { contacts, simulate_ms: ms(start), refreshes })
            }
        }
    }
}

struct CountingReader<R> {
    inner: R,
    bytes: u64,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes += n as u64;
        Ok(n)
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

fn handle(server: &Server, stream: TcpStream) -> io::Result<()> {
    let mut reader = BufReader::with_capacity(1 << 20, stream.try_clone()?);
    let mut writer = BufWriter::new(stream);
    loop {
        let request = match read_frame::<_, Request>(&mut reader) {
            Ok((r, _)) => r,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        let response = match request {
            Request::Info => Ok(Response::Info(server.info())),
            Request::UploadKeys => server.install_keys(&mut reader),
            Request::Simulate { scheme, key_id, hole, shot } => server.simulate(scheme, key_id, hole, &shot),
            Request::Progress => Ok(Response::Progress(server.progress.lock().unwrap().clone())),
        };
        let response = response.unwrap_or_else(|e| {
            eprintln!("error: {e:#}");
            Response::Error(format!("{e:#}"))
        });
        write_frame(&mut writer, &response)?;
    }
}

fn main() -> Result<()> {
    let addr = arg("--listen").unwrap_or_else(|| DEFAULT_SERVER_ADDR.to_string());
    if !flag("--allow-low-memory") {
        system::ensure_memory(system::SERVER_MIN_MEMORY)?;
    }
    let params = Params::secure()?;
    let threads = system::init_threads();

    let start = Instant::now();
    let course = Course::get();
    let flows: Vec<Flow> = course.holes.iter().map(|h| Flow::new(&h.terrain, &course.physics)).collect();
    let name = params.name.clone();
    let engine = CkksEngine::new(Ring::new(params)?)?;
    let prepared = course
        .holes
        .iter()
        .zip(&flows)
        .map(|(h, f)| Prepared::new(&engine, &Plan::new(h, f, &course.physics)?))
        .collect::<Result<Vec<_>>>()?;
    eprintln!(
        "{} backend ({threads} threads), {name} parameters, {} holes ready in {:.1} s",
        backend::DEVICE_NAME,
        course.holes.len(),
        start.elapsed().as_secs_f64()
    );

    let server = Arc::new(Server {
        course,
        flows,
        ckks: Ckks { prepared, engine: RwLock::new(engine), key_id: Mutex::new(None) },
        progress: Mutex::new(None),
    });
    let listener = TcpListener::bind(&addr)?;
    eprintln!("listening on {addr}");
    for stream in listener.incoming() {
        let (stream, server) = (stream?, server.clone());
        std::thread::spawn(move || {
            if let Err(e) = handle(&server, stream) {
                eprintln!("connection error: {e}");
            }
        });
    }
    Ok(())
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// Value of a `--name value` command-line argument.
fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
