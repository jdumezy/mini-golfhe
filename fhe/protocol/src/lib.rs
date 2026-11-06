//! Messages exchanged between the client and the server.
//!
//! Each message is a postcard-encoded frame prefixed by its length as a
//! little-endian u32. Shots and contacts travel as opaque byte strings whose
//! meaning depends on the scheme: the server never needs more than the
//! evaluation keys and the ciphertexts. Evaluation keys, gigabytes with
//! bootstrapping, follow an [`Request::UploadKeys`] frame as a raw stream.

use std::io::{Read, Write};

use mini_golfhe_physics::{Contact, State};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 2;
pub const DEFAULT_CLIENT_ADDR: &str = "127.0.0.1:8787";
pub const DEFAULT_SERVER_ADDR: &str = "127.0.0.1:8788";

const MAX_FRAME_BYTES: usize = 1 << 30;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// No encryption: the contacts are computed in the clear, for comparison.
    Plaintext,
    /// CKKS: the contact circuit on the encrypted shot.
    Ckks,
}

impl Scheme {
    pub fn name(self) -> &'static str {
        match self {
            Scheme::Plaintext => "plaintext",
            Scheme::Ckks => "ckks",
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Request {
    Info,
    /// Followed on the same connection by the evaluation-key stream.
    UploadKeys,
    /// Simulates a shot on a hole. The hole is public; the shot (position and
    /// velocity) is the scheme's ciphertext.
    Simulate {
        scheme: Scheme,
        key_id: u64,
        hole: u32,
        shot: Vec<u8>,
    },
    /// Where the running simulation is, asked on a separate connection.
    Progress,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Response {
    Info(ServerInfo),
    KeysInstalled {
        key_id: u64,
        bytes: u64,
        prepare_ms: f64,
    },
    /// One output per contact (the plaintext scheme sends them all in one).
    Trajectory {
        contacts: Vec<Vec<u8>>,
        simulate_ms: f64,
        refreshes: u32,
    },
    Progress(Option<SimulateProgress>),
    Error(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    /// Gaps and contact indicators of every time step and face.
    Gaps,
    Bootstrapping,
    /// The first contact, summed out of the grid.
    Contact,
    Output,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SimulateProgress {
    pub contact: u32,
    pub contacts: u32,
    pub step: Step,
    /// Estimated fraction of the evaluation done.
    pub fraction: f64,
    pub elapsed_ms: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServerInfo {
    pub version: u32,
    /// The poulpy backend the server evaluates on.
    pub backend: String,
    pub ckks_params: String,
    /// Identifier of the installed evaluation keys, if any.
    pub key_id: Option<u64>,
    /// The course the server simulates, to check the client agrees.
    pub course_hash: u64,
    pub max_contacts: u32,
    /// Grid steps searched for each contact, and their spacing in seconds.
    pub steps: u32,
    pub dt: f64,
}

/// Writes one frame and returns its size on the wire.
pub fn write_frame<W: Write>(w: &mut W, msg: &impl Serialize) -> std::io::Result<usize> {
    let body = postcard::to_stdvec(msg).map_err(std::io::Error::other)?;
    let len = u32::try_from(body.len()).map_err(std::io::Error::other)?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&body)?;
    w.flush()?;
    Ok(4 + body.len())
}

/// Reads one frame and returns it with its size on the wire.
pub fn read_frame<R: Read, T: DeserializeOwned>(r: &mut R) -> std::io::Result<(T, usize)> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(std::io::Error::other(format!("frame too large: {len} bytes")));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    let msg = postcard::from_bytes(&body).map_err(std::io::Error::other)?;
    Ok((msg, 4 + len))
}

/// A writer that counts the bytes going through it.
pub struct Counting<W> {
    pub inner: W,
    pub bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl<W: Write> Write for Counting<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Encodings used by the plaintext scheme: little-endian f64s.
pub mod plaintext {
    use super::*;

    /// hit, t, x, y, vx, vy.
    const CONTACT_VALUES: usize = 6;

    fn floats(bytes: &[u8], n: usize) -> Result<Vec<f64>, String> {
        if bytes.len() != n * 8 {
            return Err(format!("expected {} bytes, got {}", n * 8, bytes.len()));
        }
        Ok(bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect())
    }

    pub fn encode_shot(s: &State) -> Vec<u8> {
        s.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    pub fn decode_shot(bytes: &[u8]) -> Result<State, String> {
        Ok(floats(bytes, 4)?.try_into().unwrap())
    }

    pub fn encode_contacts(contacts: &[Contact]) -> Vec<u8> {
        contacts
            .iter()
            .flat_map(|c| [if c.hit { 1.0 } else { 0.0 }, c.t, c.state[0], c.state[1], c.state[2], c.state[3]])
            .flat_map(f64::to_le_bytes)
            .collect()
    }

    pub fn decode_contacts(bytes: &[u8], max_contacts: usize) -> Result<Vec<Contact>, String> {
        Ok(floats(bytes, max_contacts * CONTACT_VALUES)?
            .as_chunks::<CONTACT_VALUES>()
            .0
            .iter()
            .map(|v| Contact { hit: v[0] > 0.5, t: v[1], state: [v[2], v[3], v[4], v[5]] })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plaintext_round_trips() {
        let shot = [0.5, 1.25, -3.0, 2.0];
        assert_eq!(plaintext::decode_shot(&plaintext::encode_shot(&shot)).unwrap(), shot);
        let contacts = [Contact { hit: true, t: 0.42, state: [1.0, 2.0, 3.0, 4.0] }, Contact::default()];
        assert_eq!(plaintext::decode_contacts(&plaintext::encode_contacts(&contacts), 2).unwrap(), contacts);
        assert!(plaintext::decode_contacts(&[0; 8], 2).is_err());
    }
}
