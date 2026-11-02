//! Polynomial selection by the base-m method.
//!
//! GNFS needs two polynomials with a common root `m` modulo `N`. Writing
//! `N` in base `m` gives `f` with `f(m) = N`, and the other is the linear
//! `g(x) = x − m`. A good `f` has small values over the sieve region (its
//! size) and many roots modulo small primes, which makes those values more
//! often smooth (its α). Base-m is the simplest method: Kleinjung's and
//! Murphy's methods find much better polynomials, but this is enough to
//! start with.

use alloc::vec::Vec;

use crate::arith::{Limb, Uint};
use crate::factor::root;
use crate::primes::Primes;
use crate::qs::log2;

/// Leading coefficients tried.
const LEADS: u64 = 2000;

/// Values of `m` tried below the largest for each leading coefficient.
const OFFSETS: u64 = 4;

/// Candidates kept by size for scoring by α, which costs more.
const FINALISTS: usize = 64;

/// α counts the roots modulo primes up to this.
const ALPHA_PRIMES: u64 = 200;

/// Two polynomials for GNFS: `f` of degree `d` and `x − m`, with
/// `f(m) = N`.
#[derive(Clone, Debug, PartialEq)]
pub struct Poly {
    /// The coefficients of `f`, lowest degree first.
    pub f: Vec<i128>,
    /// The common root.
    pub m: u128,
    /// How much wider the sieve region should be in `a` than in `b`: the
    /// `s` that balances `|c₀|` against `|c_d|·s^d`.
    pub skew: f64,
    /// `log₂` of the typical size of `f` over the region, with the skew.
    pub size: f64,
    /// Murphy's α in bits: how much smaller than its size a value of `f`
    /// behaves for smoothness, from its roots modulo small primes. Negative
    /// is better.
    pub alpha: f64,
}

/// The degree of `f` for a composite of `bits` bits.
pub fn degree(bits: u32) -> usize {
    match bits {
        ..260 => 3,
        260..370 => 4,
        _ => 5,
    }
}

impl Poly {
    /// Chooses polynomials for `n` by the base-m method: for each leading
    /// coefficient `a`, `m` just below `(n/a)^(1/d)` makes `f`'s leading
    /// coefficient `a`, and the others are `n`'s digits in base `m`,
    /// centered on 0. The smallest by size, then by size plus α, wins.
    ///
    /// # Panics
    ///
    /// Panics if `m` doesn't fit in 96 bits, which happens only past
    /// about 480 bits at degree 5.
    pub fn select<const LIMBS: usize>(n: &Uint<LIMBS>) -> Self {
        let d = degree(n.bits());
        let mut candidates: Vec<Self> = Vec::new();
        for a in 1..=LEADS {
            let (q, _) = n.div_rem(&Uint::from_u64(a));
            let top = root(&q, d as u32);
            for k in 0..OFFSETS {
                let m = top.wrapping_sub(&Uint::from_u64(k));
                if let Some(poly) = Self::base_m(n, &m, d) {
                    candidates.push(poly);
                }
            }
        }
        candidates.sort_by(|x, y| x.size.total_cmp(&y.size));
        candidates.truncate(FINALISTS);
        for c in &mut candidates {
            c.alpha = alpha(&c.f);
        }
        candidates
            .into_iter()
            .min_by(|x, y| (x.size + x.alpha).total_cmp(&(y.size + y.alpha)))
            .expect("no polynomial")
    }

    /// `n`'s digits in base `m`, centered on 0, as a polynomial of degree
    /// `d`, or `None` if the leading digit is 0 or anything overflows.
    pub fn base_m<const LIMBS: usize>(n: &Uint<LIMBS>, m: &Uint<LIMBS>, d: usize) -> Option<Self> {
        let m128 = to_u128(m).filter(|&m| m < 1 << 96 && m > 1)?;
        let mut digits = Vec::with_capacity(d + 1);
        let mut rest = *n;
        for _ in 0..d {
            let (q, r) = rest.div_rem(m);
            digits.push(to_u128(&r)? as i128);
            rest = q;
        }
        let lead = to_u128(&rest).filter(|&c| c > 0 && c < 1 << 96)?;
        digits.push(lead as i128);
        // Center the digits: c − m with a carry keeps f(m) the same.
        let m = m128 as i128;
        for i in 0..d {
            if digits[i] > m / 2 {
                digits[i] -= m;
                digits[i + 1] += 1;
            }
        }
        let (skew, size) = skew_and_size(&digits);
        Some(Self {
            f: digits,
            m: m128,
            skew,
            size,
            alpha: 0.0,
        })
    }

    /// The degree of `f`.
    pub fn degree(&self) -> usize {
        self.f.len() - 1
    }
}

/// The skew `s = |c₀/c_d|^(1/d)`, and `log₂` of the root mean square of
/// the coefficients scaled by it, `c_i·s^(i − d/2)`: about the size of
/// `F(a, b) = b^d·f(a/b)` for `|a| ≈ s·|b|`, divided by `|b|^d·s^(d/2)`,
/// so polynomials of the same degree and skewed region compare fairly.
fn skew_and_size(f: &[i128]) -> (f64, f64) {
    let d = f.len() - 1;
    let c0 = (f[0].unsigned_abs() as f64).max(1.0);
    let cd = (f[d].unsigned_abs() as f64).max(1.0);
    let ls = ((log2(c0) - log2(cd)) / d as f64).max(0.0);
    let half = d as f64 / 2.0;
    let sum: f64 = f
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .map(|(i, &c)| {
            let l = log2(c.unsigned_abs() as f64) + (i as f64 - half) * ls;
            exp2(2.0 * (l - 64.0))
        })
        .sum();
    (exp2(ls), 64.0 + log2(sum / (d + 1) as f64) / 2.0)
}

/// Murphy's α in bits: the sum over small primes `p` of
/// `(1 − q_p·p/(p+1))·log₂ p/(p − 1)`, with `q_p` the roots of `f` modulo
/// `p`, counting a root at infinity when `p` divides the leading
/// coefficient. A random integer has `1/(p − 1)` factors of `p` on
/// average, and a value of `f` about `q_p·p/(p² − 1)`.
pub fn alpha(f: &[i128]) -> f64 {
    let lead = f[f.len() - 1];
    Primes::new(2, ALPHA_PRIMES)
        .map(|p| {
            let mut q = super::modp::roots(f, p).len() as f64;
            if lead.rem_euclid(i128::from(p)) == 0 {
                q += 1.0;
            }
            let p = p as f64;
            (1.0 - q * p / (p + 1.0)) * log2(p) / (p - 1.0)
        })
        .sum()
}

/// `2^x` for `|x| < 1000`: the integer part in the exponent field, the
/// rest by its Taylor series.
pub(crate) fn exp2(x: f64) -> f64 {
    debug_assert!(x.abs() < 1000.0);
    let mut i = x as i64;
    if i as f64 > x {
        i -= 1;
    }
    let t = (x - i as f64) * core::f64::consts::LN_2;
    let (mut term, mut sum) = (1.0, 1.0);
    for k in 1..20 {
        term *= t / f64::from(k);
        sum += term;
    }
    sum * f64::from_bits(((i + 1023) as u64) << 52)
}

// Limb is u32 or u64, so the casts are needed on one target and redundant
// on the other.
#[allow(clippy::unnecessary_cast)]
pub(crate) fn to_u128<const LIMBS: usize>(x: &Uint<LIMBS>) -> Option<u128> {
    if x.bits() > 128 {
        return None;
    }
    let mut r = 0;
    for (i, &l) in x.as_limbs().iter().enumerate() {
        if l != 0 {
            r |= (l as u128) << (i as u32 * Limb::BITS);
        }
    }
    Some(r)
}
