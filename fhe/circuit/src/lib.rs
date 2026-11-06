//! The oblivious contact circuit: from an encrypted shot, the contacts of its
//! trajectory, using only additions, products, rotations and public
//! constants, the same number of operations whatever the shot.

pub mod engine;
pub mod eval;
pub mod plan;
pub mod poly;

pub use engine::{Engine, Rotations, SlotSim, rotations};
pub use eval::{Event, Prepared, evaluate};
pub use plan::{Decoded, Plan, decode_contact, encode_state};

#[cfg(test)]
mod tests;
