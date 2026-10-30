//! Integer factorization algorithms.
//!
//! The plan is a pipeline that removes small factors cheaply (trial division,
//! p±1, ECM) before handing what is left to the quadratic sieve or GNFS.
//! So far [`factor`] runs trial division, Pollard's rho, p−1, p+1 and ECM,
//! with the Baillie–PSW test deciding primality, on the fixed-width integers
//! in [`arith`].

#![no_std]

extern crate alloc;

pub mod arith;
mod batch;
pub mod curve;
pub mod ecm;
mod factor;
pub mod pm1;
pub mod pp1;
pub mod prime;
pub mod primes;
pub mod rho;
mod stage2;

pub use arith::{Monty, Uint};
pub use factor::{Config, Factor, factor, factor_str, factor_str_with, factor_with};
