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
//! [`qs`] runs the sieve with many polynomials ([`Sieve`]), keeps relations
//! with one large prime and pairs them up, and finds the dependencies by
//! dense Gaussian elimination, which limits it to factor bases of a few
//! thousand primes for now.

mod base;
mod linalg;
mod modp;
mod params;
mod sieve;

use alloc::collections::{BTreeMap, BTreeSet};
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

/// A row of the matrix: `t² ≡ ±L²·∏ p (mod N)`, from a full relation
/// (`L = 1`) or two partial relations sharing the large prime `L`.
struct Row<const LIMBS: usize> {
    /// `t mod N`, in Montgomery form.
    t: Uint<LIMBS>,
    negative: bool,
    factors: Vec<u32>,
    large: u64,
}

/// Factors `n` with the quadratic sieve, using `params`.
///
/// Returns a proper factor of `n`, or `None` if every dependency gave a
/// trivial one, which is what happens when `n` is prime or a prime power.
/// Even `n` greater than 2 returns 2.
///
/// # Panics
///
/// Panics if `kN` overflows the integer width for the chosen multiplier `k`
/// (at most 70), or if `n` is too small for the factor base.
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
    let monty = Monty::new(*n)?;

    let mut sieve = Sieve::new(&fb, params, 0);
    let target = fb.primes.len() + 1 + EXTRA_RELATIONS;
    let mut rows = Vec::new();
    // The same relation can come from more than one polynomial.
    let mut seen = BTreeSet::new();
    let mut partials: BTreeMap<u64, Relation<LIMBS>> = BTreeMap::new();
    while rows.len() < target {
        for r in sieve.sieve_poly() {
            if r.large == 1 {
                if seen.insert(r.t) {
                    rows.push(Row {
                        t: monty.to_monty(&r.t),
                        negative: r.negative,
                        factors: r.factors,
                        large: 1,
                    });
                }
                continue;
            }
            // Each partial after the first with its large prime pairs up
            // with the first.
            match partials.get(&r.large) {
                None => {
                    partials.insert(r.large, r);
                }
                Some(first) if first.t != r.t => {
                    let mut factors = first.factors.clone();
                    factors.extend(&r.factors);
                    rows.push(Row {
                        t: monty.mul(&monty.to_monty(&first.t), &monty.to_monty(&r.t)),
                        negative: first.negative != r.negative,
                        factors,
                        large: r.large,
                    });
                }
                Some(_) => {}
            }
        }
    }
    rows.truncate(target);

    // Column 0 is the sign, column i + 1 the i-th factor-base prime.
    let matrix: Vec<Vec<usize>> = rows
        .iter()
        .map(|r| {
            let sign = r.negative.then_some(0);
            sign.into_iter()
                .chain(r.factors.iter().map(|&i| i as usize + 1))
                .collect()
        })
        .collect();
    dependencies(&matrix, fb.primes.len() + 1)
        .iter()
        .find_map(|dep| proper(square_root(&monty, &fb, &rows, dep)))
}

/// For rows whose product is a square, `X = ∏ t` and `Y = √∏ |Q|` satisfy
/// `X² ≡ Y² (mod n)`, and returns `gcd(X − Y, n)`.
fn square_root<const LIMBS: usize>(
    monty: &Monty<LIMBS>,
    fb: &FactorBase<LIMBS>,
    rows: &[Row<LIMBS>],
    dep: &[usize],
) -> Uint<LIMBS> {
    let mut x = monty.one();
    let mut y = monty.one();
    let mut exponents = alloc::vec![0u32; fb.primes.len()];
    for r in dep.iter().map(|&i| &rows[i]) {
        x = monty.mul(&x, &r.t);
        y = monty.mul(&y, &monty.to_monty(&Uint::from_u64(r.large)));
        for &i in &r.factors {
            exponents[i as usize] += 1;
        }
    }
    for (prime, &e) in fb.primes.iter().zip(&exponents) {
        debug_assert!(e % 2 == 0, "odd exponent of {}", prime.p);
        let p = monty.to_monty(&Uint::from_u64(prime.p.into()));
        for _ in 0..e / 2 {
            y = monty.mul(&y, &p);
        }
    }
    let n = monty.modulus();
    let (x, y) = (monty.from_monty(&x), monty.from_monty(&y));
    debug_assert_eq!(monty.mul(&x, &x), monty.mul(&y, &y), "X² ≢ Y²");
    // X − Y mod n, with both already reduced.
    let d = n.wrapping_sub(&y).wrapping_add(&x);
    let d = if d >= *n { d.wrapping_sub(n) } else { d };
    d.gcd(n)
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
