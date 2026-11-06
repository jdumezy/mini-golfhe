//! Messages exchanged between the client and the server.
//!
//! Each message is a postcard-encoded frame prefixed by its length as a
//! little-endian u32. Shots and trajectories travel as opaque byte strings
//! whose meaning depends on the scheme: the server never needs more than the
//! evaluation key and the ciphertexts.

use mini_golfhe_physics::{Contact, State};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const PROTOCOL_VERSION: u32 = 1;
pub const DEFAULT_CLIENT_ADDR: &str = "127.0.0.1:8787";
pub const DEFAULT_SERVER_ADDR: &str = "127.0.0.1:8788";

/// Frames can carry evaluation keys, which get large under FHE.
const MAX_FRAME_BYTES: usize = 1 << 30;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// No encryption: the "ciphertexts" are the plaintext encodings.
    Plaintext,
}

impl Scheme {
    pub fn name(self) -> &'static str {
        match self {
            Scheme::Plaintext => "plaintext",
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Request {
    Info,
    /// Simulates a shot on a hole. The hole is public; the shot (position and
    /// velocity) is the scheme's ciphertext.
    Simulate {
        scheme: Scheme,
        eval_key: Vec<u8>,
        hole: u32,
        shot: Vec<u8>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Response {
    Info(ServerInfo),
    /// The contacts, always `max_contacts` of them, as the scheme's ciphertext.
    Trajectory {
        contacts: Vec<u8>,
        simulate_micros: u64,
    },
    Error(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServerInfo {
    pub version: u32,
    pub schemes: Vec<Scheme>,
    /// The course the server simulates, to check the client agrees.
    pub course_hash: u64,
    pub holes: u32,
    pub max_contacts: u32,
    /// Grid steps searched for each contact, and their spacing in seconds.
    pub steps: u32,
    pub dt: f64,
}

/// Writes one frame and returns its size on the wire.
pub async fn write_frame<W: AsyncWrite + Unpin>(w: &mut W, msg: &impl Serialize) -> std::io::Result<usize> {
    let body = postcard::to_stdvec(msg).map_err(std::io::Error::other)?;
    let len = u32::try_from(body.len()).map_err(std::io::Error::other)?;
    w.write_all(&len.to_le_bytes()).await?;
    w.write_all(&body).await?;
    w.flush().await?;
    Ok(4 + body.len())
}

/// Reads one frame and returns it with its size on the wire.
pub async fn read_frame<R: AsyncRead + Unpin, T: DeserializeOwned>(r: &mut R) -> std::io::Result<(T, usize)> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len).await?;
    let len = u32::from_le_bytes(len) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(std::io::Error::other(format!("frame too large: {len} bytes")));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).await?;
    let msg = postcard::from_bytes(&body).map_err(std::io::Error::other)?;
    Ok((msg, 4 + len))
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
