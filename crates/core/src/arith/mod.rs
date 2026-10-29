//! Fixed-width integer arithmetic used by the factoring algorithms.

// Carry chains read most clearly as indexed loops over parallel limb arrays.
#![allow(clippy::needless_range_loop)]

mod limb;
mod monty;
mod uint;

pub use limb::{LIMB_BITS, Limb};
pub use monty::Monty;
pub use uint::{ParseUintError, Uint};
