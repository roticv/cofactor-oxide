//! The self-initializing quadratic sieve (SIQS), in progress.
//!
//! The quadratic sieve looks for many `x` where `Q(x) = (ax + b)² − kN` has
//! only small prime factors, all from a *factor base* of primes `p` for which
//! `kN` is a square mod `p` (no other odd prime can divide `Q(x)`). Each such
//! `x` gives a congruence `(ax + b)² ≡ Q(x) (mod N)`. Once there are more of
//! them than primes in the factor base, linear algebra over GF(2) finds a
//! subset whose `Q(x)` multiply to a square `y²`, and with `X` the product
//! of the matching `ax + b`, `X² ≡ y² (mod N)`, so `gcd(X − y, N)` is a
//! factor of `N` about half the time.
//!
//! So far [`qs`] sieves a single polynomial, `a = 1` and `b = ⌈√kN⌉`, whose
//! values grow steadily with `|x|`, which limits it to about 40 digits.

mod base;
mod linalg;
mod modp;
mod params;
mod sieve;

use alloc::vec::Vec;

use crate::arith::{Monty, Uint};

pub use base::{FactorBase, Prime, multiplier};
pub use linalg::dependencies;
pub use modp::{pow_mod, sqrt_mod};
pub use params::{BLOCK_SIZE, Params};
pub use sieve::{Relation, Sieve};

/// Relations beyond the factor-base size (plus one for the sign) to
/// collect. Each is one more dependency, and each dependency splits `N`
/// with probability at least ½.
const EXTRA_RELATIONS: usize = 32;

/// Factors `n` with the quadratic sieve, using `params`.
///
/// Returns a proper factor of `n`, or `None` if every dependency gave a
/// trivial one, which is what happens when `n` is prime or a prime power.
/// Even `n` greater than 2 returns 2.
///
/// # Panics
///
/// Panics if `kN` overflows the integer width for the chosen multiplier `k`
/// (at most 70).
pub fn qs<const LIMBS: usize>(n: &Uint<LIMBS>, params: &Params) -> Option<Uint<LIMBS>> {
    let two = Uint::from_u64(2);
    if !n.is_odd() {
        return (*n > two).then_some(two);
    }
    let proper = |g: Uint<LIMBS>| (g != Uint::ONE && g != *n).then_some(g);
    let fb = match FactorBase::new(n, multiplier(n), params.fb_size) {
        Ok(fb) => fb,
        Err(p) => return proper(p),
    };
    let s = fb.kn.isqrt();
    if s.widening_square() == (fb.kn, Uint::ZERO) {
        return proper(s.gcd(n));
    }

    let mut sieve = Sieve::new(&fb);
    let target = fb.primes.len() + 1 + EXTRA_RELATIONS;
    let mut relations = Vec::new();
    // Blocks at x ≥ 0 and x < 0 in turn, outward from 0.
    for i in 0i64.. {
        let x0 = if i % 2 == 0 { i / 2 } else { -(i + 1) / 2 } * BLOCK_SIZE as i64;
        relations.extend(sieve.block(x0));
        if relations.len() >= target {
            break;
        }
    }
    relations.truncate(target);

    // Column 0 is the sign, column i + 1 the i-th factor-base prime.
    let rows: Vec<Vec<usize>> = relations
        .iter()
        .map(|r| {
            let sign = r.negative.then_some(0);
            sign.into_iter()
                .chain(r.factors.iter().map(|&i| i as usize + 1))
                .collect()
        })
        .collect();
    dependencies(&rows, fb.primes.len() + 1)
        .iter()
        .find_map(|dep| square_root(n, &fb, &relations, dep).and_then(proper))
}

/// For relations whose product is a square, `X = ∏ t` and `Y = √∏ |Q|`
/// satisfy `X² ≡ Y² (mod n)`, and returns `gcd(X − Y, n)`.
fn square_root<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    fb: &FactorBase<LIMBS>,
    relations: &[Relation<LIMBS>],
    dep: &[usize],
) -> Option<Uint<LIMBS>> {
    let monty = Monty::new(*n)?;
    let mut x = monty.one();
    let mut exponents = alloc::vec![0u32; fb.primes.len()];
    for r in dep.iter().map(|&i| &relations[i]) {
        x = monty.mul(&x, &monty.to_monty(&r.t));
        for &i in &r.factors {
            exponents[i as usize] += 1;
        }
    }
    let mut y = monty.one();
    for (prime, &e) in fb.primes.iter().zip(&exponents) {
        debug_assert!(e % 2 == 0, "odd exponent of {}", prime.p);
        let p = monty.to_monty(&Uint::from_u64(prime.p.into()));
        for _ in 0..e / 2 {
            y = monty.mul(&y, &p);
        }
    }
    let (x, y) = (monty.from_monty(&x), monty.from_monty(&y));
    debug_assert_eq!(monty.mul(&x, &x), monty.mul(&y, &y), "X² ≢ Y²");
    // X − Y mod n, with both already reduced.
    let d = n.wrapping_sub(&y).wrapping_add(&x);
    let d = if d >= *n { d.wrapping_sub(n) } else { d };
    Some(d.gcd(n))
}

/// `log₂ x` for `x > 0`, to about 12 significant digits.
///
/// `core` has no logarithm, and the sieve only needs rough ones, so split
/// `x = m·2^e` with `m` in `[1, 2)` and take `ln m = 2·atanh((m − 1)/(m + 1))`
/// from its series, which converges fast because `(m − 1)/(m + 1) ≤ 1/3`.
pub(crate) fn log2(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    let bits = x.to_bits();
    // Normal numbers only, which is all that's passed in here.
    let e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let m = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52));
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let mut term = t;
    let mut sum = 0.0;
    for k in 0..12 {
        sum += term / (2 * k + 1) as f64;
        term *= t2;
    }
    e as f64 + 2.0 * sum * core::f64::consts::LOG2_E
}
