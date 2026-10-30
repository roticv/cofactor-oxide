//! Lenstra's elliptic curve method (ECM).
//!
//! ECM is p−1 with a random group. p−1 works in the multiplicative group
//! mod `p`, whose order `p − 1` is fixed: if it isn't smooth, there is
//! nothing to be done. An elliptic curve mod `p` has a group whose order is
//! roughly uniform in `p + 1 ± 2√p` and different for every curve, so when
//! one curve's order isn't smooth, another curve gets a fresh chance.
//!
//! Stage 1 multiplies a point by every prime power up to B1. If the order of
//! the point mod `p` divides that product, the result is the point at
//! infinity mod `p`, so `Z ≡ 0 (mod p)`, and `gcd(Z, n)` reveals `p`.

use alloc::vec::Vec;

use crate::arith::{Monty, Uint};
use crate::batch::{CHUNK, Check, check};
use crate::curve::{Curve, Point};
use crate::primes::Primes;

/// Runs one ECM curve, chosen by Suyama's parametrization with `sigma`,
/// using stage 1 with bound B1.
///
/// Returns a factor of `n` if this curve finds one. A single curve usually
/// doesn't; [`ecm`] runs many. Even `n` greater than 2 returns 2 without
/// searching.
///
/// # Panics
///
/// Panics if `b1` is 2^32 or more.
pub fn ecm_curve<const LIMBS: usize>(n: &Uint<LIMBS>, sigma: u64, b1: u64) -> Option<Uint<LIMBS>> {
    assert!(b1 < 1 << 32, "B1 too large");
    let two = Uint::from_u64(2);
    if !n.is_odd() {
        return (*n > two).then_some(two);
    }
    let monty = Monty::new(*n)?;
    let (curve, point) = match Curve::suyama(&monty, sigma) {
        Ok(found) => found,
        // Setting up the curve needs an inverse, and a failed inverse is
        // already a factor, unless it's all of n.
        Err(g) => return (g != *n).then_some(g),
    };
    stage1(&curve, point, b1)
}

/// Multiplies `point` by each maximal prime power up to `b1`, checking
/// `gcd(Z, n)` once per chunk of primes, and returns the factor found, if
/// any.
fn stage1<const LIMBS: usize>(
    curve: &Curve<LIMBS>,
    mut point: Point<LIMBS>,
    b1: u64,
) -> Option<Uint<LIMBS>> {
    let monty = curve.monty();
    let mut checkpoint = point;
    let mut chunk = Vec::with_capacity(CHUNK);
    let mut primes = Primes::new(2, b1).peekable();
    while let Some(p) = primes.next() {
        let mut q = p;
        while q <= b1 / p {
            q *= p;
        }
        point = curve.mul(&point, q);
        chunk.push(q);
        if chunk.len() == CHUNK || primes.peek().is_none() {
            // Once Z ≡ 0 (mod p) it stays that way, so Z itself is the
            // accumulated value: no product of differences is needed.
            match check(monty, &point.z, || {
                let mut replay = checkpoint;
                chunk.iter().map(move |&q| {
                    replay = curve.mul(&replay, q);
                    replay.z
                })
            }) {
                Check::Continue => {}
                Check::Found(g) => return Some(g),
                Check::Fail => return None,
            }
            checkpoint = point;
            chunk.clear();
        }
    }
    None
}

/// Runs up to `curves` ECM curves with stage-1 bound `b1`, using
/// `sigma = first_sigma, first_sigma + 1, …`, and returns the first factor
/// found.
///
/// Each curve finds a prime `p` exactly when the order of its starting point
/// mod `p` is B1-smooth (counting prime powers), so the number of curves
/// needed depends on how likely that is for the size of `p` and the chosen
/// B1.
pub fn ecm<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    b1: u64,
    curves: u64,
    first_sigma: u64,
) -> Option<Uint<LIMBS>> {
    (first_sigma..first_sigma + curves).find_map(|sigma| ecm_curve(n, sigma, b1))
}
