//! Finding a factor of a cofactor that may be two large primes past 2^62
//! or three, below 2^128, which has to be fast: most such cofactors that
//! split turn out to have a prime too large, so the sieve splits many for
//! each relation.
//!
//! P−1 runs first, then any ECM curves, all with small bounds fixed in
//! advance: the prime powers of stage 1 and the pairs of stage 2 are
//! worked out once, and each stage takes one GCD at its end.

use alloc::vec;
use alloc::vec::Vec;

use crate::arith::{Limb, Monty, Uint};
use crate::curve::{Curve, Point};
use crate::prime::miller_rabin;
use crate::primes::Primes;

/// Wide enough for a cofactor of three large primes, each below 2^40.
const LIMBS: usize = 128 / Limb::BITS as usize;
pub(crate) type Cofactor = Uint<LIMBS>;

/// P−1's bounds, then ECM's, for up to `ECM_CURVES` curves.
///
/// At 90 digits, with large primes below 2^34, the cheapest search gave
/// the most relations a second. With two a side, P−1 to (60, 500) and no
/// curves gave 4,570 a second; to (300, 3000) 3,890; with one curve
/// after, 2,720; with eight, 1,370, though that found 12% more relations.
/// With three on the algebraic side it was 1,100 a second against 290
/// with eight curves, which found 58% more. Where sieving each special-q
/// costs more, as with larger numbers, more relations from each could
/// make some curves pay.
const PM1_BOUNDS: (u64, u64) = (60, 500);
const ECM_BOUNDS: (u64, u64) = (150, 3_000);
const ECM_CURVES: u64 = 0;

/// The first ECM curve's sigma: smaller ones give degenerate curves.
const FIRST_SIGMA: u64 = 6;

/// ECM's stage 2 pairs each prime `k·D ± j` up to B2 with `j` coprime to
/// `D` and below `D/2`.
const D: u64 = 210;

/// The work of the search, fixed by its bounds.
#[derive(Clone, Debug)]
pub(crate) struct Splitter {
    /// The prime powers up to each stage-1 bound, multiplied together as
    /// far as a word holds.
    pm1_stage1: Vec<u64>,
    ecm_stage1: Vec<u64>,
    /// The gaps between P−1's stage-2 primes, after the first of them.
    pm1_first: u64,
    pm1_gaps: Vec<u8>,
    /// For ECM's stage 2, the first `k`, and for each `k` from it, a mask
    /// of the `j` (by index among those coprime to `D`) to pair with `k·D`.
    ecm_first_k: u64,
    ecm_pairs: Vec<u64>,
}

/// The odd `j < D/2` coprime to `D`, which index the bits of a mask.
fn babies() -> impl Iterator<Item = u64> {
    (1..D / 2).step_by(2).filter(|j| gcd(*j, D) == 1)
}

impl Splitter {
    pub(crate) fn new() -> Self {
        let (pm1_first, pm1_gaps) = gaps(PM1_BOUNDS.0, PM1_BOUNDS.1);
        let (b1, b2) = ECM_BOUNDS;
        // Each prime in (B1, B2] is k·D ± j: one bit for k, at j.
        let ecm_first_k = (b1 + D / 2) / D;
        let mut ecm_pairs = vec![0u64; ((b2 + D / 2) / D - ecm_first_k + 1) as usize];
        let index: Vec<u64> = babies().collect();
        for p in Primes::new(b1 + 1, b2) {
            let k = (p + D / 2) / D;
            let j = p.abs_diff(k * D);
            if let Some(i) = index.iter().position(|&b| b == j) {
                ecm_pairs[(k - ecm_first_k) as usize] |= 1 << i;
            }
        }
        Self {
            pm1_stage1: powers(PM1_BOUNDS.0),
            ecm_stage1: powers(b1),
            pm1_first,
            pm1_gaps,
            ecm_first_k,
            ecm_pairs,
        }
    }

    /// A proper factor of the odd `n`, or `None` if `n` passes a strong
    /// probable-prime test or the search finds none.
    pub(crate) fn factor(&self, n: &Cofactor) -> Option<Cofactor> {
        let monty = Monty::new(*n)?;
        if miller_rabin(&monty, 2) {
            return None;
        }
        self.pm1(&monty).or_else(|| {
            (FIRST_SIGMA..FIRST_SIGMA + ECM_CURVES).find_map(|sigma| self.ecm(&monty, sigma))
        })
    }

    fn pm1(&self, monty: &Monty<LIMBS>) -> Option<Cofactor> {
        let one = monty.one();
        let pow = |x: &Cofactor, e: u64| monty.pow(x, &Uint::<2>::from_u64(e));
        let mut x = monty.to_monty(&Uint::from_u64(3));
        for &e in &self.pm1_stage1 {
            x = pow(&x, e);
        }
        if let Some(g) = proper(&monty.sub(&x, &one), monty) {
            return Some(g);
        }
        // x^q for each prime q of stage 2, from the last by x^gap.
        let mut steps: Vec<Option<Cofactor>> = vec![None; 256];
        let mut y = pow(&x, self.pm1_first);
        let mut acc = monty.sub(&y, &one);
        for &gap in &self.pm1_gaps {
            let step = *steps[usize::from(gap)].get_or_insert_with(|| pow(&x, gap.into()));
            y = monty.mul(&y, &step);
            acc = monty.mul(&acc, &monty.sub(&y, &one));
        }
        proper(&acc, monty)
    }

    fn ecm(&self, monty: &Monty<LIMBS>, sigma: u64) -> Option<Cofactor> {
        let (curve, mut q) = match Curve::suyama(monty, sigma) {
            Ok(found) => found,
            Err(g) => return (g != *monty.modulus()).then_some(g),
        };
        for &e in &self.ecm_stage1 {
            q = curve.mul(&q, e);
        }
        if let Some(g) = proper(&q.z, monty) {
            return Some(g);
        }
        // Baby steps [j]Q for odd j below D/2, from [2]Q.
        let two = curve.double(&q);
        let mut odd = vec![q, curve.add(&two, &q, &q)];
        while odd.len() < (D / 4) as usize {
            let i = odd.len();
            odd.push(curve.add(&odd[i - 1], &two, &odd[i - 2]));
        }
        let baby: Vec<Point<_>> = babies().map(|j| odd[(j / 2) as usize]).collect();
        // Giant steps [k·D]Q, pairing each with its babies, and the next.
        let step = curve.mul(&q, D);
        let mut giant = curve.mul(&q, self.ecm_first_k * D);
        let mut next = curve.mul(&q, (self.ecm_first_k + 1) * D);
        let mut acc = monty.one();
        for &mask in &self.ecm_pairs {
            let mut bits = mask;
            while bits != 0 {
                let b = &baby[bits.trailing_zeros() as usize];
                bits &= bits - 1;
                let t = monty.sub(&monty.mul(&giant.x, &b.z), &monty.mul(&b.x, &giant.z));
                acc = monty.mul(&acc, &t);
            }
            (giant, next) = (next, curve.add(&next, &step, &giant));
        }
        proper(&acc, monty)
    }
}

/// `gcd(x, n)` if it's a proper factor of `n`.
fn proper(x: &Cofactor, monty: &Monty<LIMBS>) -> Option<Cofactor> {
    let n = monty.modulus();
    let g = x.gcd(n);
    (g != Uint::ONE && g != *n).then_some(g)
}

/// The largest power of each prime up to `b1`, multiplied together as far
/// as 64 bits hold.
fn powers(b1: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut word: u64 = 1;
    for p in Primes::new(2, b1) {
        let mut q = p;
        while q <= b1 / p {
            q *= p;
        }
        match word.checked_mul(q) {
            Some(w) => word = w,
            None => {
                out.push(word);
                word = q;
            }
        }
    }
    out.push(word);
    out
}

/// The first prime in `(b1, b2]`, and the gaps between it and the rest.
fn gaps(b1: u64, b2: u64) -> (u64, Vec<u8>) {
    let primes: Vec<u64> = Primes::new(b1 + 1, b2).collect();
    let gaps = primes.windows(2).map(|w| (w[1] - w[0]) as u8).collect();
    (primes[0], gaps)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
