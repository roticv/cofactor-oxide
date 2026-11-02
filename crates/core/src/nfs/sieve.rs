//! The line siever: for each `b`, sieves `a − b·m` and `F(a, b)` over a
//! range of `a` and keeps the pairs for which both are smooth.
//!
//! Each line is cut into chunks that fit in cache. A chunk is sieved on
//! the rational side first, then on the algebraic side; positions over
//! both thresholds are candidates. The primes from 2^8 up are sieved again
//! to find which of them hit the candidates, and the smaller ones are
//! tried directly.

use alloc::vec;
use alloc::vec::Vec;

use super::base::{FactorBase, Prime};
use super::params::Params;
use super::poly::{Poly, exp2};
use crate::arith::{Limb, Uint};
use crate::qs::log2;

/// Bytes sieved at a time.
const CHUNK: usize = 1 << 16;

/// Primes below this aren't sieved, only tried on candidates.
const SMALL_PRIMES: u32 = 30;

/// Primes from this up are sieved again over a chunk with candidates, to
/// find which candidates they divide; smaller ones are tried directly.
const RESIEVE: u32 = 256;

/// The algebraic threshold is computed once per this many positions.
const SEGMENT: usize = 256;

/// Bits the thresholds allow beyond the large prime, for rounded logs and
/// powers of primes the sieve counts once.
const FUDGE: f64 = 4.0;

/// Wide enough for `F(a, b)` and `a − b·m` up to 400-bit `N`.
pub(crate) type Wide = Uint<{ 512 / Limb::BITS as usize }>;

/// A pair `(a, b)` with `a − b·m` and `F(a, b)` smooth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relation {
    pub a: i64,
    pub b: u64,
    /// Whether `a − b·m` is negative.
    pub negative: bool,
    /// The factor-base primes of `|a − b·m|`, with repetition.
    pub rational: Vec<u32>,
    /// The factor-base primes of `|F(a, b)|`, with repetition. Each stands
    /// for the prime ideal `(p, a/b mod p)`, or the one at infinity if `p`
    /// divides `b`.
    pub algebraic: Vec<u32>,
    /// The large prime of each side, rational then algebraic, or 1 if none.
    pub large: [u64; 2],
}

/// One side's primes and where each next hits the line.
struct Side {
    primes: Vec<Prime>,
    next: Vec<u32>,
    saved: Vec<u32>,
    /// Index of the first prime that's sieved, and of the first resieved.
    first_sieved: usize,
    first_resieved: usize,
    large_bound: u64,
    /// Expected `log₂` from the primes that aren't sieved.
    skipped: f64,
    block: Vec<u8>,
}

impl Side {
    fn new(primes: &[Prime], params: &Params) -> Self {
        let primes: Vec<Prime> = primes.iter().copied().filter(|p| !p.projective()).collect();
        let largest = primes.last().map_or(2, |p| p.p);
        let skipped = primes
            .iter()
            .take_while(|p| p.p < SMALL_PRIMES)
            .map(|p| log2(f64::from(p.p)) / f64::from(p.p - 1))
            .sum();
        Self {
            next: vec![0; primes.len()],
            saved: vec![0; primes.len()],
            first_sieved: primes.partition_point(|p| p.p < SMALL_PRIMES),
            first_resieved: primes.partition_point(|p| p.p < RESIEVE),
            large_bound: u64::from(largest) * u64::from(params.large_mult),
            skipped,
            block: vec![0; CHUNK],
            primes,
        }
    }

    /// Points each prime at its first hit on line `b`, where position `x`
    /// stands for `a = x − w`: `a ≡ b·r (mod p)`.
    fn start_line(&mut self, b: u64, w: u64) {
        for (next, q) in self.next.iter_mut().zip(&self.primes) {
            let p = u64::from(q.p);
            *next = ((b % p * u64::from(q.r) + w) % p) as u32;
        }
    }

    /// Adds each prime's log at its hits in `[start, start + CHUNK)`.
    fn sieve(&mut self, start: u32) {
        self.block.fill(0);
        self.saved.copy_from_slice(&self.next);
        let end = start + CHUNK as u32;
        for i in self.first_sieved..self.primes.len() {
            let Prime { p, log, .. } = self.primes[i];
            let mut x = self.next[i];
            while x < end {
                let cell = &mut self.block[(x - start) as usize];
                *cell = cell.saturating_add(log);
                x += p;
            }
            self.next[i] = x;
        }
    }

    /// Records `(offset, prime)` for each resieved prime that hits one of
    /// the marked offsets of the chunk at `start`.
    fn resieve(&self, start: u32, marked: &[bool], hits: &mut Vec<(u32, u32)>) {
        let end = start + CHUNK as u32;
        for (q, &first) in self.primes[self.first_resieved..]
            .iter()
            .zip(&self.saved[self.first_resieved..])
        {
            let mut x = first;
            while x < end {
                let i = x - start;
                if marked[i as usize] {
                    hits.push((i, q.p));
                }
                x += q.p;
            }
        }
    }
}

/// Sieves lines of `(a, b)` pairs.
pub struct Sieve<'a> {
    poly: &'a Poly,
    w: u32,
    rational: Side,
    algebraic: Side,
    /// Algebraic primes that divide the leading coefficient, for `p | b`.
    projective: Vec<u32>,
    f: Vec<f64>,
    marked: Vec<bool>,
}

impl<'a> Sieve<'a> {
    pub fn new(poly: &'a Poly, fb: &'a FactorBase, params: &Params) -> Self {
        Self {
            poly,
            w: half_width(params.half_width, poly.skew),
            rational: Side::new(&fb.rational, params),
            algebraic: Side::new(&fb.algebraic, params),
            projective: fb
                .algebraic
                .iter()
                .filter(|p| p.projective())
                .map(|p| p.p)
                .collect(),
            f: poly.f.iter().map(|&c| c as f64).collect(),
            marked: vec![false; CHUNK],
        }
    }

    /// Each line sieves `a` in `[−w, w)`.
    pub fn half_width(&self) -> u32 {
        self.w
    }

    /// `log₂ |F(a, b)|`, or 0 where it vanishes.
    fn log_norm(&self, a: f64, b: f64) -> f64 {
        let x = a / b;
        let v = self.f.iter().rev().fold(0.0, |acc, &c| acc * x + c);
        let d = self.f.len() - 1;
        let v = v.abs() * b_pow(b, d);
        if v >= 1.0 { log2(v) } else { 0.0 }
    }

    /// The relations on line `b`, for `a` in `[−w, w)` with `gcd(a, b) = 1`.
    pub fn line(&mut self, b: u64) -> Vec<Relation> {
        let w = self.w;
        self.rational.start_line(b, w.into());
        self.algebraic.start_line(b, w.into());
        let bf = b as f64;
        // |a − b·m| barely changes along the line.
        let rational_log = log2(bf * self.poly.m as f64 + f64::from(w));
        let r_threshold =
            rational_log - log2(self.rational.large_bound as f64) - self.rational.skipped - FUDGE;
        let r_threshold = r_threshold.clamp(0.0, 255.0) as u8;
        let a_slack = log2(self.algebraic.large_bound as f64) + self.algebraic.skipped + FUDGE;
        let mut relations = Vec::new();
        let mut candidates = Vec::new();
        let mut hits = Vec::new();
        for start in (0..2 * w).step_by(CHUNK) {
            self.rational.sieve(start);
            self.algebraic.sieve(start);
            candidates.clear();
            for seg in (0..CHUNK).step_by(SEGMENT) {
                // The smaller end's norm, or no threshold if F changes
                // sign, and so has a root, in between.
                let a0 = f64::from(start) + seg as f64 - f64::from(w);
                let (l0, l1) = (
                    self.log_norm(a0, bf),
                    self.log_norm(a0 + SEGMENT as f64, bf),
                );
                let sign_change = self.f_sign(a0, bf) != self.f_sign(a0 + SEGMENT as f64, bf);
                let a_threshold = if sign_change {
                    0
                } else {
                    (l0.min(l1) - a_slack).clamp(0.0, 255.0) as u8
                };
                for x in seg..seg + SEGMENT {
                    if self.rational.block[x] >= r_threshold
                        && self.algebraic.block[x] >= a_threshold
                    {
                        candidates.push(x as u32);
                    }
                }
            }
            if candidates.is_empty() {
                continue;
            }
            for &x in &candidates {
                self.marked[x as usize] = true;
            }
            hits.clear();
            self.rational.resieve(start, &self.marked, &mut hits);
            let split = hits.len();
            self.algebraic.resieve(start, &self.marked, &mut hits);
            for &x in &candidates {
                self.marked[x as usize] = false;
            }
            let (r_hits, a_hits) = hits.split_at(split);
            for &x in &candidates {
                let a = i64::from(start + x) - i64::from(w);
                if let Some(rel) = self.check(a, b, x, r_hits, a_hits) {
                    relations.push(rel);
                }
            }
        }
        relations
    }

    fn f_sign(&self, a: f64, b: f64) -> bool {
        let x = a / b;
        self.f.iter().rev().fold(0.0, |acc, &c| acc * x + c) < 0.0
    }

    /// Trial-divides the candidate `(a, b)` at offset `x`, given the
    /// resieved primes of its chunk.
    fn check(
        &self,
        a: i64,
        b: u64,
        x: u32,
        r_hits: &[(u32, u32)],
        a_hits: &[(u32, u32)],
    ) -> Option<Relation> {
        if gcd(a.unsigned_abs(), b) != 1 {
            return None;
        }
        // Rational side: a − b·m.
        let bm = u128::from(b).checked_mul(self.poly.m)?;
        let negative = i128::from(a) < bm as i128;
        let v = (i128::from(a) - bm as i128).unsigned_abs();
        let mut v = wide(v);
        let mut rational = Vec::new();
        let small = &self.rational.primes[..self.rational.first_resieved];
        for q in small {
            if divides_line(a, b, q) {
                divide_out(&mut v, q.p, &mut rational);
            }
        }
        for &(_, p) in r_hits.iter().filter(|h| h.0 == x) {
            divide_out(&mut v, p, &mut rational);
        }
        let r_large = cofactor(&v, self.rational.large_bound)?;

        // Algebraic side: F(a, b).
        let mut v = norm(&self.poly.f, a, b);
        let mut algebraic = Vec::new();
        for q in &self.algebraic.primes[..self.algebraic.first_resieved] {
            if divides_line(a, b, q) {
                divide_out(&mut v, q.p, &mut algebraic);
            }
        }
        for &p in &self.projective {
            if b.is_multiple_of(u64::from(p)) {
                divide_out(&mut v, p, &mut algebraic);
            }
        }
        for &(_, p) in a_hits.iter().filter(|h| h.0 == x) {
            divide_out(&mut v, p, &mut algebraic);
        }
        let a_large = cofactor(&v, self.algebraic.large_bound)?;
        Some(Relation {
            a,
            b,
            negative,
            rational,
            algebraic,
            large: [r_large, a_large],
        })
    }
}

/// The half width of a line: `w·√s`, so that the region is about `s`
/// times wider in `a` than the number of lines in `b`, as the skew `s`
/// wants, rounded to whole chunks and kept within `u32` positions.
fn half_width(w: u32, skew: f64) -> u32 {
    let scaled = f64::from(w) * exp2(log2(skew.max(1.0)) / 2.0);
    let half = (CHUNK / 2) as f64;
    ((scaled / half + 0.5) as u32).clamp(1, 1 << 14) * (CHUNK / 2) as u32
}

/// `b^d`.
fn b_pow(b: f64, d: usize) -> f64 {
    (0..d).fold(1.0, |acc, _| acc * b)
}

/// Whether `a ≡ b·r (mod p)`.
fn divides_line(a: i64, b: u64, q: &Prime) -> bool {
    let p = i64::from(q.p);
    let br = (b % q.p as u64 * u64::from(q.r) % q.p as u64) as i64;
    (a.rem_euclid(p) - br).rem_euclid(p) == 0
}

/// Divides every factor of `p` out of `v`, recording each.
// Limb is u32 or u64, so the casts are needed on one target and redundant
// on the other.
#[allow(clippy::unnecessary_cast)]
fn divide_out(v: &mut Wide, p: u32, factors: &mut Vec<u32>) {
    loop {
        let (q, r) = v.div_rem_limb(p as Limb);
        if r != 0 || v.is_zero() {
            return;
        }
        *v = q;
        factors.push(p);
    }
}

/// The cofactor left by trial division if it's 1 or a prime below
/// `bound`, which must be below the square of the largest factor-base
/// prime so that anything that small is prime.
fn cofactor(v: &Wide, bound: u64) -> Option<u64> {
    let c = v.to_u64()?;
    (c != 0 && c < bound).then_some(c)
}

/// `|F(a, b)| = |Σ cᵢ·aⁱ·b^(d−i)|`, exactly.
pub(crate) fn norm(f: &[i128], a: i64, b: u64) -> Wide {
    let d = f.len() - 1;
    let (mut pos, mut neg) = (Wide::ZERO, Wide::ZERO);
    let wa = Wide::from_u64(a.unsigned_abs());
    let wb = Wide::from_u64(b);
    for (i, &c) in f.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let ca = c.unsigned_abs();
        let mut t = wide(ca);
        for _ in 0..i {
            t = t.widening_mul(&wa).0;
        }
        for _ in i..d {
            t = t.widening_mul(&wb).0;
        }
        if (c < 0) != (a < 0 && i % 2 == 1) {
            neg = neg.wrapping_add(&t);
        } else {
            pos = pos.wrapping_add(&t);
        }
    }
    if pos >= neg {
        pos.wrapping_sub(&neg)
    } else {
        neg.wrapping_sub(&pos)
    }
}

fn wide(x: u128) -> Wide {
    Wide::from_u64(x as u64).wrapping_add(&(Wide::from_u64((x >> 64) as u64) << 64))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
