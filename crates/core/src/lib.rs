//! Integer factorization algorithms.
//!
//! The plan is a pipeline that removes small factors cheaply (trial division,
//! p±1, ECM) before handing what is left to the quadratic sieve or GNFS.
//! So far [`factor`] runs trial division, Pollard's rho and p−1, with the
//! Baillie–PSW test deciding primality, on the fixed-width integers in
//! [`arith`].

#![no_std]

extern crate alloc;

pub mod arith;
mod factor;
pub mod pm1;
pub mod prime;
pub mod primes;
pub mod rho;

pub use arith::{Monty, Uint};
pub use factor::{Factor, factor, factor_str};
