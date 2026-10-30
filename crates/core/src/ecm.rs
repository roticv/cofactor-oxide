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
//!
//! Stage 2 catches points whose order is B1-smooth apart from one prime
//! `q ≤ B2`, with the same baby-step giant-step search that p+1 uses
//! ([`crate::pp1`]).

use alloc::vec::Vec;

use crate::arith::{Monty, Uint};
use crate::batch::{CHUNK, Check, check};
use crate::curve::{Curve, Point};
use crate::primes::Primes;
use crate::stage2::{Stage2Group, stage2};

/// Runs one ECM curve, chosen by Suyama's parametrization with `sigma`,
/// with stage-1 bound B1 and stage-2 bound B2. Use `b2 <= b1` to skip
/// stage 2.
///
/// Returns a factor of `n` if this curve finds one. A single curve usually
/// doesn't; [`ecm`] runs many. Even `n` greater than 2 returns 2 without
/// searching.
///
/// # Panics
///
/// Panics if `b1` is 2^32 or more.
pub fn ecm_curve<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    sigma: u64,
    b1: u64,
    b2: u64,
) -> Option<Uint<LIMBS>> {
    assert!(b1 < 1 << 32, "B1 too large");
    let two = Uint::from_u64(2);
    if !n.is_odd() {
        return (*n > two).then_some(two);
    }
    let monty = Monty::new(*n)?;
    let (curve, mut point) = match Curve::suyama(&monty, sigma) {
        Ok(found) => found,
        // Setting up the curve needs an inverse, and a failed inverse is
        // already a factor, unless it's all of n.
        Err(g) => return (g != *n).then_some(g),
    };
    match stage1(&curve, &mut point, b1) {
        Check::Continue => stage2(&curve, &point, b1, b2),
        Check::Found(g) => Some(g),
        Check::Fail => None,
    }
}

/// Multiplies `point` in place by each maximal prime power up to `b1`,
/// checking `gcd(Z, n)` once per chunk of primes.
///
/// Returns `Continue` if no factor turned up, leaving the final point for
/// stage 2.
fn stage1<const LIMBS: usize>(
    curve: &Curve<LIMBS>,
    point: &mut Point<LIMBS>,
    b1: u64,
) -> Check<LIMBS> {
    let monty = curve.monty();
    let mut checkpoint = *point;
    let mut chunk = Vec::with_capacity(CHUNK);
    let mut primes = Primes::new(2, b1).peekable();
    while let Some(p) = primes.next() {
        let mut q = p;
        while q <= b1 / p {
            q *= p;
        }
        *point = curve.mul(point, q);
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
                found_or_failed => return found_or_failed,
            }
            checkpoint = *point;
            chunk.clear();
        }
    }
    Check::Continue
}

/// Runs up to `curves` ECM curves with bounds `b1` and `b2`, using
/// `sigma = first_sigma, first_sigma + 1, …`, and returns the first factor
/// found.
///
/// Each curve finds a prime `p` exactly when the order of its starting point
/// mod `p` is B1-smooth (counting prime powers), apart from at most one
/// prime up to B2. So the number of curves needed depends on how likely that
/// is for the size of `p` and the chosen bounds.
pub fn ecm<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    b1: u64,
    b2: u64,
    curves: u64,
    first_sigma: u64,
) -> Option<Uint<LIMBS>> {
    (first_sigma..first_sigma + curves).find_map(|sigma| ecm_curve(n, sigma, b1, b2))
}

/// Curves as a stage-2 group. Points are known only by x-coordinate, which
/// identifies them up to sign, and the x-only formulas need the difference
/// of the points being added, as stage 2 requires.
impl<const LIMBS: usize> Stage2Group<LIMBS> for Curve<LIMBS> {
    type Elem = Point<LIMBS>;

    fn monty(&self) -> &Monty<LIMBS> {
        Curve::monty(self)
    }

    fn double(&self, a: &Point<LIMBS>) -> Point<LIMBS> {
        Curve::double(self, a)
    }

    fn add(&self, a: &Point<LIMBS>, b: &Point<LIMBS>, diff: &Point<LIMBS>) -> Point<LIMBS> {
        Curve::add(self, a, b, diff)
    }

    fn mul(&self, a: &Point<LIMBS>, k: u64) -> Point<LIMBS> {
        Curve::mul(self, a, k)
    }

    /// `Xa·Zb − Xb·Za`, which vanishes mod `p` exactly when the points have
    /// the same x-coordinate mod `p`, so are equal up to sign.
    fn compare(&self, a: &Point<LIMBS>, b: &Point<LIMBS>) -> Uint<LIMBS> {
        let m = Curve::monty(self);
        m.sub(&m.mul(&a.x, &b.z), &m.mul(&b.x, &a.z))
    }
}
