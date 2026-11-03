//! The lattice siever.
//!
//! For each special-q, an ideal `(q, r)` of degree one, the pairs
//! `(a, b)` whose algebraic norm `q` divides form the lattice `a ≡ r·b (mod q)`. With a reduced basis
//! `(a₀, b₀), (a₁, b₁)`, they are `(a, b) = i·(a₀, b₀) + j·(a₁, b₁)`, and the
//! siever covers `i` in `[−I/2, I/2)` and `j` in `[1, J)`: a compact region
//! whose algebraic norms are `q` times smaller than those of a line of `b`
//! as long, so that far more of them are smooth.
//!
//! A factor-base prime `p` with root `R` divides `a − R·b` where
//! `i·(a₀ − R·b₀) + j·(a₁ − R·b₁) ≡ 0 (mod p)`: the lattice `i ≡ ρ·j` in the
//! `(i, j)` plane. Primes below `I` are sieved row by row. Larger ones hit
//! each row at most once, and are walked from hit to hit by Franke and
//! Kleinjung's method. The region is sieved in slabs of whole rows that
//! fit in cache, rational side then algebraic; positions over both
//! thresholds are candidates. The primes that hit them are found by
//! sieving again, or recorded during the walk, and divided out; primes
//! below 2^8 are tried directly.

use alloc::vec;
use alloc::vec::Vec;

use super::base::{FactorBase, Prime};
use super::modp::roots;
use super::params::Params;
use super::poly::Poly;
use crate::arith::{Limb, Uint};
use crate::primes::Primes;
use crate::qs::cofactor::split;
use crate::qs::log2;

/// The most positions sieved at a time.
const SLAB_BITS: u32 = 19;

/// Primes below this aren't sieved, only tried on candidates.
const SMALL_PRIMES: u32 = 30;

/// Primes from this up are sieved again over a slab with candidates, to
/// find which candidates they divide; smaller ones are tried directly.
const RESIEVE: u32 = 256;

/// The most positions along a row that share a threshold.
const SEGMENT: usize = 256;

/// Positions that share a threshold near where a norm changes sign.
const PIECE: usize = 16;

/// Bits the thresholds allow beyond the large prime, for rounded logs and
/// powers of primes the sieve counts once.
const FUDGE: f64 = 4.0;

/// Wide enough for `F(a, b)` and `a − b·m` up to 400-bit `N`.
pub(crate) type Wide = Uint<{ 512 / Limb::BITS as usize }>;

/// A pair `(a, b)` with `G(a, b) = l·a − m·b` and `F(a, b)` smooth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relation {
    pub a: i64,
    pub b: u64,
    /// Whether `G(a, b)` is negative.
    pub negative: bool,
    /// The primes of `|G(a, b)|`, with repetition: factor-base primes and
    /// up to [`Params::large_primes`] larger ones.
    pub rational: Vec<u64>,
    /// The primes of `|F(a, b)|`, with repetition: factor-base primes, the
    /// special-q and up to [`Params::large_primes`] other larger ones. Each
    /// stands for the prime ideal `(p, a/b mod p)`, or the one at infinity
    /// if `p | b`.
    pub algebraic: Vec<u64>,
}

/// The special-q ideals `(q, r)` with `q` in `[lo, hi)`: each prime with
/// each root of `f` modulo it.
pub fn special_qs(poly: &Poly, lo: u64, hi: u64) -> Vec<(u64, u64)> {
    Primes::new(lo, hi)
        .flat_map(|q| roots(&poly.f, q).into_iter().map(move |r| (q, r)))
        .collect()
}

/// A prime below `I`: it hits row `j` where `i ≡ ρ·j (mod p)`.
#[derive(Clone, Copy)]
struct Small {
    p: u32,
    rho: u32,
    log: u8,
}

/// A prime from `I` up, walked from hit to hit. Its next hit is at `pos`,
/// `j·I + x` for `x = i + I/2`. From `(x, j)`, the one after is
/// `(x − a, j + ja)` if `x ≥ a`, else `(x + b, j + jb)` if `x + b < I`, else
/// `(x − a + b, j + ja + jb)`.
#[derive(Clone, Copy)]
struct Walk {
    pos: u64,
    a: u32,
    b: u32,
    ja: u32,
    jb: u32,
    p: u32,
    log: u8,
}

/// One side's factor base and its sieve for the current special-q.
struct Side {
    primes: Vec<Prime>,
    large_bound: u64,
    /// How many primes past the factor base, below `large_bound`, a value
    /// may have.
    large_primes: u32,
    /// The largest factor-base prime squared: any cofactor below it is
    /// prime.
    prime_below: u64,
    /// Expected `log₂` from the primes that aren't sieved.
    skipped: f64,
    small: Vec<Small>,
    walks: Vec<Walk>,
    /// Primes that divide every point of rows `j ≡ 0 (mod p)`.
    rows: Vec<Small>,
    block: Vec<u8>,
    /// `(offset, p)` of the walked primes' hits in the current slab.
    hits: Vec<(u32, u32)>,
}

impl Side {
    fn new(primes: &[Prime], params: &Params, slab: usize) -> Self {
        let largest = primes.last().map_or(2, |p| p.p);
        let skipped = primes
            .iter()
            .take_while(|p| p.p < SMALL_PRIMES)
            .filter(|p| !p.projective())
            .map(|p| log2(f64::from(p.p)) / f64::from(p.p - 1))
            .sum();
        Self {
            primes: primes.to_vec(),
            large_bound: u64::from(largest) * u64::from(params.large_mult),
            large_primes: params.large_primes,
            prime_below: u64::from(largest) * u64::from(largest),
            skipped,
            small: Vec::new(),
            walks: Vec::new(),
            rows: Vec::new(),
            block: vec![0; slab],
            hits: Vec::new(),
        }
    }

    /// Finds each prime's lattice in the `(i, j)` plane for the special-q
    /// basis `u`, `v`, in a region `2^log_i` wide.
    fn start(&mut self, u: (i64, i64), v: (i64, i64), log_i: u32) {
        let width = 1u64 << log_i;
        self.small.clear();
        self.walks.clear();
        self.rows.clear();
        for q in &self.primes {
            if q.p < SMALL_PRIMES {
                continue;
            }
            let p = u64::from(q.p);
            // a − R·b at the basis vectors, or b for the root at infinity.
            let at = |(a, b): (i64, i64)| {
                let b = b.rem_euclid(p as i64) as u64;
                if q.projective() {
                    b
                } else {
                    let a = a.rem_euclid(p as i64) as u64;
                    (a + p - b * u64::from(q.r) % p) % p
                }
            };
            let (alpha, beta) = (at(u), at(v));
            if alpha == 0 {
                if beta != 0 {
                    self.rows.push(Small {
                        p: q.p,
                        rho: 0,
                        log: q.log,
                    });
                }
                continue;
            }
            let rho = (p - beta) * inverse(alpha, p) % p;
            if p < width {
                self.small.push(Small {
                    p: q.p,
                    rho: rho as u32,
                    log: q.log,
                });
            } else {
                self.walks.push(walk(q, rho, log_i));
            }
        }
    }

    /// Sieves the rows `[j0, j0 + rows)` of a region `2^log_i` wide.
    fn sieve(&mut self, j0: u64, rows: usize, log_i: u32) {
        let width = 1usize << log_i;
        let half = width as u64 / 2;
        self.block.fill(0);
        for (r, row) in self.block[..rows * width]
            .chunks_exact_mut(width)
            .enumerate()
        {
            let j = j0 + r as u64;
            for s in &self.small {
                let p = u64::from(s.p);
                let mut x = ((u64::from(s.rho) * (j % p) + half) % p) as usize;
                while x < width {
                    row[x] = row[x].saturating_add(s.log);
                    x += s.p as usize;
                }
            }
            for s in self.rows.iter().filter(|s| j.is_multiple_of(s.p.into())) {
                for cell in row.iter_mut() {
                    *cell = cell.saturating_add(s.log);
                }
            }
        }
        let start = j0 << log_i;
        let end = start + (rows << log_i) as u64;
        let mask = width as u64 - 1;
        self.hits.clear();
        for w in &mut self.walks {
            while w.pos < end {
                let off = (w.pos - start) as usize;
                self.block[off] = self.block[off].saturating_add(w.log);
                self.hits.push((off as u32, w.p));
                let x = w.pos & mask;
                let (a, b) = (u64::from(w.a), u64::from(w.b));
                let (ja, jb) = (u64::from(w.ja) << log_i, u64::from(w.jb) << log_i);
                w.pos = if x >= a {
                    w.pos + ja - a
                } else if x + b < width as u64 {
                    w.pos + jb + b
                } else {
                    w.pos + ja + jb + b - a
                };
            }
        }
    }

    /// Records `(offset, p)` for each prime from [`RESIEVE`] up that hits
    /// one of the marked offsets of the slab at row `j0`.
    fn resieve(
        &self,
        j0: u64,
        rows: usize,
        log_i: u32,
        marked: &[bool],
        out: &mut Vec<(u32, u32)>,
    ) {
        let width = 1usize << log_i;
        let half = width as u64 / 2;
        for r in 0..rows {
            let j = j0 + r as u64;
            let row = &marked[r * width..(r + 1) * width];
            if !row.contains(&true) {
                continue;
            }
            for s in self.small.iter().filter(|s| s.p >= RESIEVE) {
                let p = u64::from(s.p);
                let mut x = ((u64::from(s.rho) * (j % p) + half) % p) as usize;
                while x < width {
                    if row[x] {
                        out.push(((r * width + x) as u32, s.p));
                    }
                    x += s.p as usize;
                }
            }
        }
        out.extend(self.hits.iter().filter(|h| marked[h.0 as usize]));
    }
}

/// The walk of a prime `p ≥ I` over the lattice `i ≡ ρ·j (mod p)`, from the
/// origin to its first hit with `j > 0`.
///
/// The steps come from reducing the basis `(−p, 0), (ρ, 1)` like a
/// continued fraction until `(−a, ja)` and `(b, jb)` have `0 < a, b < I`
/// and `a + b ≥ I`: from any point, the next one up is then one step away.
fn walk(q: &Prime, rho: u64, log_i: u32) -> Walk {
    let width = 1u64 << log_i;
    let (mut a, mut ja, mut b, mut jb) = (u64::from(q.p), 0u64, rho, 1u64);
    if b == 0 {
        // Only i = 0: a can never be taken, b moves up a row.
        a = width;
    } else {
        while a >= width && b >= width {
            if a >= b {
                let k = a / b;
                a -= k * b;
                ja += k * jb;
            } else {
                let k = b / a;
                b -= k * a;
                jb += k * ja;
            }
        }
        if a < width {
            let k = (b + a - width) / a;
            b -= k * a;
            jb += k * ja;
        } else {
            let k = (a + b - width) / b;
            a -= k * b;
            ja += k * jb;
        }
    }
    let cap = |j: u64| j.min(u64::from(u32::MAX >> 1)) as u32;
    let mut w = Walk {
        pos: width / 2,
        a: a as u32,
        b: b as u32,
        ja: cap(ja),
        jb: cap(jb),
        p: q.p,
        log: q.log,
    };
    let x = w.pos;
    w.pos = if x >= a {
        x + (ja << log_i) - a
    } else if x + b < width {
        x + (jb << log_i) + b
    } else {
        x + ((ja + jb) << log_i) + b - a
    };
    w
}

/// Sieves special-q lattices.
pub struct Sieve<'a> {
    poly: &'a Poly,
    log_i: u32,
    log_j: u32,
    rational: Side,
    algebraic: Side,
    f: Vec<f64>,
    marked: Vec<bool>,
}

impl<'a> Sieve<'a> {
    pub fn new(poly: &'a Poly, fb: &'a FactorBase, params: &Params) -> Self {
        let log_i = params.log_i;
        let log_j = log_i - 1;
        let slab = 1usize << SLAB_BITS.min(log_i + log_j).max(log_i);
        Self {
            poly,
            log_i,
            log_j,
            rational: Side::new(&fb.rational, params, slab),
            algebraic: Side::new(&fb.algebraic, params, slab),
            f: poly.f.iter().map(|&c| c as f64).collect(),
            marked: vec![false; slab],
        }
    }

    /// `log₂` of the width and height of the region sieved per special-q.
    pub fn region(&self) -> (u32, u32) {
        (self.log_i, self.log_j)
    }

    /// `log₂ |F(a, b)|`, or 0 where it vanishes, and whether it's negative.
    fn log_norm(&self, a: f64, b: f64) -> (f64, bool) {
        // Horner's rule in a, with the powers of b alongside.
        let (mut v, mut bk) = (0.0, 1.0);
        for &c in self.f.iter().rev() {
            v = v * a + c * bk;
            bk *= b;
        }
        (if v.abs() >= 1.0 { log2(v.abs()) } else { 0.0 }, v < 0.0)
    }

    /// The rational and algebraic thresholds for `len` positions of row
    /// `j` from `x`, for the special-q basis `uv`: the smaller end's norm
    /// less `slack`, or 0 if the norm changes sign, and so vanishes, in
    /// between.
    fn thresholds(
        &self,
        x: usize,
        len: usize,
        j: u64,
        (u, v): ((i64, i64), (i64, i64)),
        (r_slack, a_slack): (f64, f64),
    ) -> (u8, u8) {
        let half = (1i64 << self.log_i) / 2;
        let (m, l) = (self.poly.m as f64, self.poly.l as f64);
        let ends = [x as i64 - half, (x + len) as i64 - half].map(|i| {
            let a = i as f64 * u.0 as f64 + j as f64 * v.0 as f64;
            let b = i as f64 * u.1 as f64 + j as f64 * v.1 as f64;
            let rat = l * a - b * m;
            let rl = if rat.abs() >= 1.0 {
                log2(rat.abs())
            } else {
                0.0
            };
            ((rl, rat < 0.0), self.log_norm(a, b))
        });
        let limit = |(l0, s0): (f64, bool), (l1, s1): (f64, bool), slack: f64| {
            if s0 != s1 {
                0
            } else {
                (l0.min(l1) - slack).clamp(0.0, 255.0) as u8
            }
        };
        (
            limit(ends[0].0, ends[1].0, r_slack),
            limit(ends[0].1, ends[1].1, a_slack),
        )
    }

    /// The relations from the special-q `(q, r)`, for `q` from 2^8 up and
    /// below 2^31. A `q` within the factor base is skipped by the sieve,
    /// whose lattice it divides, and divided out directly.
    pub fn special_q(&mut self, q: u64, r: u64) -> Vec<Relation> {
        let (log_i, log_j) = (self.log_i, self.log_j);
        let width = 1usize << log_i;
        let half = (width / 2) as i64;
        let (u, v) = reduce(q, r, self.poly.skew);
        self.rational.start(u, v, log_i);
        self.algebraic.start(u, v, log_i);
        let r_slack = self.rational.slack();
        let a_slack = self.algebraic.slack() + log2(q as f64);
        let segment = SEGMENT.min(width);
        let rows_per_slab = self.marked.len() >> log_i;
        let mut relations = Vec::new();
        let mut candidates = Vec::new();
        let mut hits = Vec::new();
        for j0 in (0..1u64 << log_j).step_by(rows_per_slab) {
            self.rational.sieve(j0, rows_per_slab, log_i);
            self.algebraic.sieve(j0, rows_per_slab, log_i);
            candidates.clear();
            for row in 0..rows_per_slab {
                let j = j0 + row as u64;
                if j == 0 {
                    continue;
                }
                for seg in (0..width).step_by(segment) {
                    let (r_t, a_t) = self.thresholds(seg, segment, j, (u, v), (r_slack, a_slack));
                    // Where a norm changes sign, and so vanishes, take the
                    // thresholds over shorter pieces.
                    let pieces = if r_t == 0 || a_t == 0 {
                        PIECE.min(segment)
                    } else {
                        segment
                    };
                    for x0 in (seg..seg + segment).step_by(pieces) {
                        let (r_t, a_t) = if pieces == segment {
                            (r_t, a_t)
                        } else {
                            self.thresholds(x0, pieces, j, (u, v), (r_slack, a_slack))
                        };
                        let base = row * width + x0;
                        for x in base..base + pieces {
                            if self.rational.block[x] >= r_t && self.algebraic.block[x] >= a_t {
                                candidates.push(x as u32);
                            }
                        }
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
            self.rational
                .resieve(j0, rows_per_slab, log_i, &self.marked, &mut hits);
            let split = hits.len();
            self.algebraic
                .resieve(j0, rows_per_slab, log_i, &self.marked, &mut hits);
            for &x in &candidates {
                self.marked[x as usize] = false;
            }
            let (r_hits, a_hits) = hits.split_at_mut(split);
            r_hits.sort_unstable();
            a_hits.sort_unstable();
            let (r_hits, a_hits) = (&*r_hits, &*a_hits);
            for &x in &candidates {
                let i = i64::from(x) % width as i64 - half;
                let j = j0 as i64 + i64::from(x >> log_i);
                let (mut a, mut b) = (i * u.0 + j * v.0, i * u.1 + j * v.1);
                if b < 0 {
                    (a, b) = (-a, -b);
                }
                if b == 0 {
                    continue;
                }
                if let Some(rel) = self.check(a, b as u64, q, x, r_hits, a_hits) {
                    relations.push(rel);
                }
            }
        }
        relations
    }

    /// Trial-divides the candidate `(a, b)` at offset `x`, given the
    /// resieved primes of its slab.
    fn check(
        &self,
        a: i64,
        b: u64,
        q: u64,
        x: u32,
        r_hits: &[(u32, u32)],
        a_hits: &[(u32, u32)],
    ) -> Option<Relation> {
        if gcd(a.unsigned_abs(), b) != 1 {
            return None;
        }
        // Rational side: l·a − b·m.
        let bm = i128::try_from(u128::from(b).checked_mul(self.poly.m)?).ok()?;
        let la = i128::try_from(self.poly.l).ok()?.checked_mul(a.into())?;
        let negative = la < bm;
        let mut v = wide(la.checked_sub(bm)?.unsigned_abs());
        let mut rational = Vec::new();
        self.rational.divide(a, b, x, r_hits, &mut v, &mut rational);
        if !self.rational.cofactor(&v, &mut rational) {
            return None;
        }

        // Algebraic side: F(a, b), which q divides.
        let mut v = norm(&self.poly.f, a, b);
        let mut algebraic = Vec::new();
        divide_out(&mut v, q as u32, &mut algebraic);
        self.algebraic
            .divide(a, b, x, a_hits, &mut v, &mut algebraic);
        if !self.algebraic.cofactor(&v, &mut algebraic) {
            return None;
        }
        Some(Relation {
            a,
            b,
            negative,
            rational,
            algebraic,
        })
    }
}

impl Side {
    /// Bits a value's sieved logs may fall short of its size: room for the
    /// large primes, the primes not sieved, and rounding.
    fn slack(&self) -> f64 {
        f64::from(self.large_primes) * log2(self.large_bound as f64) + self.skipped + FUDGE
    }

    /// Whether `v`, what trial division left, is 1 or splits into at most
    /// `large_primes` primes below `large_bound`, which are added to
    /// `factors`. The large-prime bound must be below the square of the
    /// largest factor-base prime, so that anything that small is prime.
    fn cofactor(&self, v: &Wide, factors: &mut Vec<u64>) -> bool {
        let Some(c) = v.to_u64() else {
            return false;
        };
        let bound = self.large_bound;
        match c {
            0 => false,
            1 => true,
            c if c < bound => {
                factors.push(c);
                true
            }
            c if self.large_primes < 2
                || c < self.prime_below
                || u128::from(c) >= u128::from(bound) * u128::from(bound) =>
            {
                false
            }
            c => match split(c) {
                Some([p, q]) if q < bound => {
                    factors.extend([p, q]);
                    true
                }
                _ => false,
            },
        }
    }

    /// Divides the factor-base primes out of `v`, a norm at `(a, b)` and
    /// offset `x`: the small ones by trial, the rest from `hits`, sorted.
    fn divide(
        &self,
        a: i64,
        b: u64,
        x: u32,
        hits: &[(u32, u32)],
        v: &mut Wide,
        factors: &mut Vec<u64>,
    ) {
        for q in self.primes.iter().take_while(|q| q.p < RESIEVE) {
            if divides(a, b, q) {
                divide_out(v, q.p, factors);
            }
        }
        for s in self.rows.iter().filter(|s| s.p >= RESIEVE) {
            divide_out(v, s.p, factors);
        }
        let first = hits.partition_point(|h| h.0 < x);
        for &(_, p) in hits[first..].iter().take_while(|h| h.0 == x) {
            divide_out(v, p, factors);
        }
    }
}

/// A basis of the lattice `a ≡ r·b (mod q)` reduced for the norm
/// `a² + (s·b)²`, `s` the skew, by Lagrange and Gauss's method.
fn reduce(q: u64, r: u64, skew: f64) -> ((i64, i64), (i64, i64)) {
    let s2 = skew * skew;
    let dot =
        |u: (i64, i64), v: (i64, i64)| u.0 as f64 * v.0 as f64 + s2 * (u.1 as f64 * v.1 as f64);
    let (mut u, mut v) = ((q as i64, 0i64), (r as i64, 1i64));
    if dot(u, u) > dot(v, v) {
        core::mem::swap(&mut u, &mut v);
    }
    loop {
        let t = dot(u, v) / dot(u, u);
        let k = if t >= 0.0 {
            (t + 0.5) as i64
        } else {
            (t - 0.5) as i64
        };
        v = (v.0 - k * u.0, v.1 - k * u.1);
        if dot(v, v) >= dot(u, u) {
            return (u, v);
        }
        core::mem::swap(&mut u, &mut v);
    }
}

/// `x⁻¹ mod p`, for `x` coprime to `p`.
fn inverse(x: u64, p: u64) -> u64 {
    let (mut r0, mut r1) = (p as i64, x as i64);
    let (mut t0, mut t1) = (0i64, 1i64);
    while r1 != 0 {
        let k = r0 / r1;
        (r0, r1) = (r1, r0 - k * r1);
        (t0, t1) = (t1, t0 - k * t1);
    }
    t0.rem_euclid(p as i64) as u64
}

/// Whether `p` divides the norm at `(a, b)`: `a ≡ b·r (mod p)`, or `p | b`
/// for the root at infinity.
fn divides(a: i64, b: u64, q: &Prime) -> bool {
    let p = u64::from(q.p);
    if q.projective() {
        return b.is_multiple_of(p);
    }
    let br = (b % p * u64::from(q.r) % p) as i64;
    (a.rem_euclid(p as i64) - br).rem_euclid(p as i64) == 0
}

/// Divides every factor of `p` out of `v`, recording each.
// Limb is u32 or u64, so the casts are needed on one target and redundant
// on the other.
#[allow(clippy::unnecessary_cast)]
fn divide_out(v: &mut Wide, p: u32, factors: &mut Vec<u64>) {
    loop {
        let (q, r) = v.div_rem_limb(p as Limb);
        if r != 0 || v.is_zero() {
            return;
        }
        *v = q;
        factors.push(p.into());
    }
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
