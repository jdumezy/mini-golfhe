//! Client side of an encryption scheme: owns the secret key.

use mini_golfhe_physics::{Contact, State};
use mini_golfhe_protocol::{Scheme, plaintext};

pub trait ClientScheme: Send + Sync {
    fn scheme(&self) -> Scheme;

    /// Public key material the server needs to compute on ciphertexts.
    fn eval_key(&self) -> &[u8];

    fn encrypt_shot(&self, shot: &State) -> Vec<u8>;

    fn decrypt_contacts(&self, ciphertext: &[u8], max_contacts: usize) -> Result<Vec<Contact>, String>;
}

/// No encryption at all; stands in until the FHE scheme is wired in.
pub struct Plaintext;

impl ClientScheme for Plaintext {
    fn scheme(&self) -> Scheme {
        Scheme::Plaintext
    }

    fn eval_key(&self) -> &[u8] {
        &[]
    }

    fn encrypt_shot(&self, shot: &State) -> Vec<u8> {
        plaintext::encode_shot(shot)
    }

    fn decrypt_contacts(&self, ciphertext: &[u8], max_contacts: usize) -> Result<Vec<Contact>, String> {
        plaintext::decode_contacts(ciphertext, max_contacts)
    }
}
