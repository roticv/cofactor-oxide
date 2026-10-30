//! Batched GCD checks shared by the p−1, p+1 and (later) ECM stages.
//!
//! Each of these methods produces a long run of values, any of which might
//! share a factor with `n`. Taking a GCD per value would be slow, so they
//! multiply values together and take one GCD per chunk. If a chunk's GCD is
//! `n` itself, several factors turned up in the same chunk, and the chunk is
//! replayed value by value to separate them.

use crate::arith::{Monty, Uint};

/// Number of primes processed between GCD checks.
pub(crate) const CHUNK: usize = 1024;

/// What a batched GCD check found.
pub(crate) enum Check<const LIMBS: usize> {
    /// No factor yet; carry on.
    Continue,
    Found(Uint<LIMBS>),
    /// Every factor of `n` turned up at the same step, so this base and
    /// these bounds can't separate them.
    Fail,
}

/// Checks `gcd(acc, n)`, where `acc` is a product of values that each might
/// share a factor with `n`. If the GCD is `n` itself, `replay` regenerates
/// those values one at a time so they can be checked individually.
pub(crate) fn check<const LIMBS: usize, I: Iterator<Item = Uint<LIMBS>>>(
    monty: &Monty<LIMBS>,
    acc: &Uint<LIMBS>,
    replay: impl FnOnce() -> I,
) -> Check<LIMBS> {
    let n = monty.modulus();
    let g = acc.gcd(n);
    if g == Uint::ONE {
        return Check::Continue;
    }
    if g != *n {
        return Check::Found(g);
    }
    for value in replay() {
        let g = value.gcd(n);
        if g == *n {
            break;
        }
        if g != Uint::ONE {
            return Check::Found(g);
        }
    }
    Check::Fail
}
