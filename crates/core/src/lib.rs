//! Integer factorization algorithms.
//!
//! The plan is a pipeline that removes small factors cheaply (trial division,
//! p±1, ECM) before handing what is left to the quadratic sieve or GNFS.
//! So far there is the fixed-width arithmetic in [`arith`], Pollard's rho in
//! [`rho`], and trial division on `u64`, which gives the CLI and Wasm crates
//! something to call.

#![no_std]

extern crate alloc;

pub mod arith;
pub mod rho;

use alloc::vec::Vec;

pub use arith::{Monty, Uint};

/// Returns the prime factors of `n` in ascending order, with multiplicity.
///
/// Returns an empty vector for `n < 2`.
pub fn factor(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    if n < 2 {
        return factors;
    }
    while n.is_multiple_of(2) {
        factors.push(2);
        n /= 2;
    }
    let mut d = 3;
    while d <= n / d {
        while n.is_multiple_of(d) {
            factors.push(d);
            n /= d;
        }
        d += 2;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn small_values() {
        assert_eq!(factor(0), vec![]);
        assert_eq!(factor(1), vec![]);
        assert_eq!(factor(2), vec![2]);
        assert_eq!(factor(360), vec![2, 2, 2, 3, 3, 5]);
    }

    #[test]
    fn prime() {
        assert_eq!(factor(1_000_000_007), vec![1_000_000_007]);
    }

    #[test]
    fn semiprime() {
        assert_eq!(
            factor(1_000_000_016_000_000_063),
            vec![1_000_000_007, 1_000_000_009]
        );
    }

    #[test]
    fn largest_u64() {
        assert_eq!(factor(u64::MAX), vec![3, 5, 17, 257, 641, 65537, 6700417]);
    }
}
