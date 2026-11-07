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
//! thresholds are candidates. Primes larger than a slab are walked once
//! over the whole region, their hits dropped into a bucket per slab, so
//! that a slab doesn't visit the many that miss it. The primes that hit
//! candidates are found from their rows' first hits, or recorded during
//! the walk or in the buckets, and divided out; primes below 2^8 are tried
//! directly.

use alloc::vec;
use alloc::vec::Vec;

use num_bigint::{BigInt, Sign};

use super::base::{CHARACTER_PRIMES, FactorBase, Prime};
use super::cofactor::{Cofactor, Splitter};
use super::modp::{Coeff, roots};
use super::params::Params;
use super::poly::{Poly, to_u128};
use crate::arith::{Limb, Uint};
use crate::primes::Primes;
use crate::qs::cofactor::{probable_prime, split};
use crate::qs::log2;

/// The most positions sieved at a time.
const SLAB_BITS: u32 = 19;

/// Primes below this aren't sieved, only tried on candidates.
const SMALL_PRIMES: u32 = 30;

/// Primes from this up that are sieved row by row are found on a
/// candidate by its distance from the row's first hit, which with few
/// candidates a row costs less than sieving the row again; smaller ones
/// are tried directly.
const RESIEVE: u32 = 256;

/// The most positions along a row that share a threshold.
const SEGMENT: usize = 256;

/// Positions checked at once for candidates.
const CHUNK: usize = 32;

/// Positions that share a threshold near where a norm changes sign.
const PIECE: usize = 16;

/// Primes whose roots are transformed together in [`Side::start`].
const LANES: usize = 4;

/// Bits the thresholds allow beyond the large prime, for rounded logs and
/// powers of primes the sieve counts once.
const FUDGE: f64 = 4.0;

/// Wide enough for `F(a, b)` and `l·a − m·b`: about 300 bits at most for
/// RSA-270-class polynomials.
pub(crate) type Wide = Uint<{ 512 / Limb::BITS as usize }>;

/// A pair `(a, b)` with `G(a, b) = l·a − m·b` and `F(a, b)` smooth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relation {
    pub a: i64,
    pub b: u64,
    /// Whether `G(a, b)` is negative.
    pub negative: bool,
    /// The primes of `|G(a, b)|`, with repetition: factor-base primes and
    /// up to three larger ones, below `2^`[`Params::rational_lpb`].
    pub rational: Vec<u64>,
    /// The primes of `|F(a, b)|`, with repetition: factor-base primes, the
    /// special-q and up to three other larger ones, below
    /// `2^`[`Params::algebraic_lpb`]. Each
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

/// A prime below `I`: it hits row `j` where `i ≡ ρ·j (mod p)`, first at
/// `x = i + I/2` equal to `next` in the row being sieved.
#[derive(Clone, Copy)]
struct Small {
    p: u32,
    rho: u32,
    next: u32,
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

impl Walk {
    /// Moves to the next hit, in a region `2^log_i` wide.
    fn step(&mut self, log_i: u32) {
        let x = self.pos & ((1 << log_i) - 1);
        let (a, b) = (u64::from(self.a), u64::from(self.b));
        // The first case takes the step (−a, ja), the second (b, jb) and
        // the third both: the first where x − a stays in the row or x + b
        // doesn't, the second where x − a doesn't. Chosen by multiplying
        // rather than branching, as which case comes next is close to
        // random.
        let first = x >= a || x + b >= 1 << log_i;
        let second = x < a;
        let da = (u64::from(self.ja) << log_i).wrapping_sub(a) * u64::from(first);
        let db = ((u64::from(self.jb) << log_i) + b) * u64::from(second);
        self.pos = self.pos.wrapping_add(da).wrapping_add(db);
    }
}

/// One side's factor base and its sieve for the current special-q.
struct Side {
    primes: Vec<Prime>,
    /// Primes outside the factor base must be below this.
    large_bound: u64,
    /// `log₂` of the largest cofactor kept, and the cofactor itself.
    mfb: u32,
    cofactor_bound: u128,
    /// The largest factor-base prime, and its square: any cofactor below
    /// that is prime.
    largest: u64,
    prime_below: u64,
    /// Expected `log₂` from the primes that aren't sieved.
    skipped: f64,
    splitter: Splitter,
    small: Vec<Small>,
    /// Walked primes smaller than a slab, walked slab by slab.
    walks: Vec<Walk>,
    /// Primes that divide every point of rows `j ≡ 0 (mod p)`.
    rows: Vec<Small>,
    block: Vec<u8>,
    /// `(offset, p)` of the walked primes' hits in the current slab.
    hits: Vec<(u32, u32)>,
    /// For each slab, the hits of the primes from a slab's size up, walked
    /// once over the whole region: `(offset | log << 24, p)`.
    buckets: Vec<Vec<(u32, u32)>>,
}

impl Side {
    /// The side with factor base `primes`, large primes below `2^lpb` and
    /// cofactors up to `2^mfb`.
    fn new(primes: &[Prime], (lpb, mfb): (u32, u32), slab: usize) -> Self {
        assert!(
            1 << lpb <= CHARACTER_PRIMES,
            "large primes must be below the quadratic characters'"
        );
        let largest = primes.last().map_or(2, |p| p.p);
        let skipped = primes
            .iter()
            .take_while(|p| p.p < SMALL_PRIMES)
            .filter(|p| !p.projective())
            .map(|p| log2(f64::from(p.p)) / f64::from(p.p - 1))
            .sum();
        Self {
            primes: primes.to_vec(),
            large_bound: 1 << lpb,
            mfb,
            cofactor_bound: 1 << mfb.min(127),
            largest: largest.into(),
            prime_below: u64::from(largest) * u64::from(largest),
            skipped,
            splitter: Splitter::new(),
            small: Vec::new(),
            walks: Vec::new(),
            rows: Vec::new(),
            block: vec![0; slab],
            hits: Vec::new(),
            buckets: Vec::new(),
        }
    }

    /// Finds each prime's lattice in the `(i, j)` plane for the special-q
    /// basis `u`, `v`, in a region `2^log_i` wide and `2^log_j` high, and
    /// fills the buckets.
    fn start(&mut self, u: (i64, i64), v: (i64, i64), log_i: u32, log_j: u32) {
        let width = 1u64 << log_i;
        let slab_bits = self.block.len().trailing_zeros();
        let end = 1u64 << (log_i + log_j);
        let slabs = (end >> slab_bits).max(1) as usize;
        self.buckets.resize_with(slabs, Vec::new);
        self.buckets.iter_mut().for_each(Vec::clear);
        self.small.clear();
        self.walks.clear();
        self.rows.clear();
        let first = self.primes.partition_point(|q| q.p < SMALL_PRIMES);
        // In groups, so that the divisions of the inverses and the walks'
        // steps overlap.
        for group in self.primes[first..].chunks(LANES) {
            let mut x = [1u32; LANES];
            let mut ps = [1u32; LANES];
            let mut betas = [0u64; LANES];
            for (l, q) in group.iter().enumerate() {
                let p = u64::from(q.p);
                // a − R·b at the basis vectors, or b for the root at infinity.
                let at = |(a, b): (i64, i64)| {
                    let b = reduce_mod(b, q.p);
                    if q.projective() {
                        b
                    } else {
                        // a + (p − b)·R, below p + p², reduced once.
                        (reduce_mod(a, q.p) + (p - b) * u64::from(q.r)) % p
                    }
                };
                let alpha = at(u);
                (x[l], ps[l], betas[l]) = (alpha as u32, q.p, at(v));
            }
            let inv = inverses(x, ps);
            let mut rhos = [0u32; LANES];
            // The walked primes, for their steps; 0 for the others.
            let mut walked = [0u32; LANES];
            for (l, q) in group.iter().enumerate() {
                let p = u64::from(q.p);
                rhos[l] = ((p - betas[l]) * u64::from(inv[l]) % p) as u32;
                if x[l] != 0 && p >= width {
                    walked[l] = q.p;
                }
            }
            let steps = steps(walked, rhos, log_i);
            for (l, q) in group.iter().enumerate() {
                let (p, beta) = (u64::from(q.p), betas[l]);
                if x[l] == 0 {
                    if beta != 0 {
                        self.rows.push(Small {
                            p: q.p,
                            rho: 0,
                            next: 0,
                            log: q.log,
                        });
                    }
                    continue;
                }
                if p < width {
                    self.small.push(Small {
                        p: q.p,
                        rho: rhos[l],
                        next: (width / 2 % p) as u32,
                        log: q.log,
                    });
                } else if p < 1 << slab_bits {
                    self.walks.push(walk(q, steps[l], log_i));
                } else {
                    let mut w = walk(q, steps[l], log_i);
                    let mask = (1u64 << slab_bits) - 1;
                    let tag = u32::from(q.log) << 24;
                    while w.pos < end {
                        self.buckets[(w.pos >> slab_bits) as usize]
                            .push(((w.pos & mask) as u32 | tag, q.p));
                        w.step(log_i);
                    }
                }
            }
        }
    }

    /// Sieves the rows `[j0, j0 + rows)` of a region `2^log_i` wide, the
    /// slabs in order from `j0 = 0`.
    fn sieve(&mut self, j0: u64, rows: usize, log_i: u32) {
        let width = 1usize << log_i;
        self.block.fill(0);
        for (r, row) in self.block[..rows * width]
            .chunks_exact_mut(width)
            .enumerate()
        {
            let j = j0 + r as u64;
            // On even rows, where even x makes a and b both even, only odd
            // x are sieved.
            let even = j.is_multiple_of(2);
            for s in &mut self.small {
                let p = s.p as usize;
                let mut x = s.next as usize;
                let step = if even {
                    x += (x + 1) % 2 * p;
                    2 * p
                } else {
                    p
                };
                while x < width {
                    row[x] = row[x].saturating_add(s.log);
                    x += step;
                }
                // From row j to j + 1, the first hit moves by ρ mod p.
                s.next += s.rho;
                if s.next >= s.p {
                    s.next -= s.p;
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
        self.hits.clear();
        for w in &mut self.walks {
            while w.pos < end {
                let off = (w.pos - start) as usize;
                self.block[off] = self.block[off].saturating_add(w.log);
                self.hits.push((off as u32, w.p));
                w.step(log_i);
            }
        }
        let slab = (start >> self.block.len().trailing_zeros()) as usize;
        for &(tagged, _) in &self.buckets[slab] {
            let off = (tagged & 0xff_ffff) as usize;
            self.block[off] = self.block[off].saturating_add((tagged >> 24) as u8);
        }
    }

    /// Records `(offset, p)` for each prime from [`RESIEVE`] up that hits
    /// one of the `candidates`, in order, of the slab at row `j0`, which
    /// are the offsets set in `marked`.
    fn resieve(
        &self,
        j0: u64,
        log_i: u32,
        candidates: &[u32],
        marked: &[u64],
        out: &mut Vec<(u32, u32)>,
    ) {
        let width = 1u64 << log_i;
        let half = width / 2;
        let mut rest = candidates;
        while let Some(&first) = rest.first() {
            let r = first >> log_i;
            let n = rest.partition_point(|&x| x >> log_i == r);
            let (row, next) = rest.split_at(n);
            rest = next;
            let j = j0 + u64::from(r);
            let base = r << log_i;
            for s in self.small.iter().filter(|s| s.p >= RESIEVE) {
                let p = u64::from(s.p);
                let x0 = ((u64::from(s.rho) * (j % p) + half) % p) as u32;
                for &x in row {
                    if (x - base) % s.p == x0 {
                        out.push((x, s.p));
                    }
                }
            }
        }
        out.extend(self.hits.iter().filter(|h| is_marked(marked, h.0)));
        let slab = ((j0 << log_i) >> self.block.len().trailing_zeros()) as usize;
        out.extend(
            self.buckets[slab]
                .iter()
                .map(|&(tagged, p)| (tagged & 0xff_ffff, p))
                .filter(|h| is_marked(marked, h.0)),
        );
    }
}

/// The steps `(a, ja, b, jb)` of the walks of primes `pₗ ≥ I` over the
/// lattices `i ≡ ρₗ·j (mod pₗ)`, the lanes in step so that their divisions
/// overlap; lanes with `pₗ = 0` are left alone.
///
/// The steps come from reducing the basis `(−p, 0), (ρ, 1)` like a
/// continued fraction until `(−a, ja)` and `(b, jb)` have `0 < a, b < I`
/// and `a + b ≥ I`: from any point, the next one up is then one step away.
fn steps(p: [u32; LANES], rho: [u32; LANES], log_i: u32) -> [[u32; 4]; LANES] {
    let width = 1u32 << log_i;
    let mut s: [[u32; 4]; LANES] = core::array::from_fn(|l| [p[l], 0, rho[l], 1]);
    for (l, [a, _, b, _]) in s.iter_mut().enumerate() {
        if p[l] != 0 && *b == 0 {
            // Only i = 0: a can never be taken, b moves up a row.
            *a = width;
        }
    }
    let busy = |&[a, _, b, _]: &[u32; 4]| a >= width && b >= width;
    while s.iter().any(busy) {
        for [a, ja, b, jb] in s.iter_mut() {
            if *a >= width && *b >= width {
                if *a >= *b {
                    let k = *a / *b;
                    *a -= k * *b;
                    *ja += k * *jb;
                } else {
                    let k = *b / *a;
                    *b -= k * *a;
                    *jb += k * *ja;
                }
            }
        }
    }
    for (l, [a, ja, b, jb]) in s.iter_mut().enumerate() {
        if p[l] == 0 || rho[l] == 0 {
            continue;
        }
        if *a < width {
            let k = (*b + *a - width) / *a;
            *b -= k * *a;
            *jb += k * *ja;
        } else {
            let k = (*a + *b - width) / *b;
            *a -= k * *b;
            *ja += k * *jb;
        }
    }
    s
}

/// The walk of `q` with steps `(a, ja, b, jb)`, from the origin to its
/// first hit with `j > 0`.
fn walk(q: &Prime, [a, ja, b, jb]: [u32; 4], log_i: u32) -> Walk {
    let cap = |j: u32| j.min(u32::MAX >> 1);
    let mut w = Walk {
        pos: 1 << (log_i - 1),
        a,
        b,
        ja: cap(ja),
        jb: cap(jb),
        p: q.p,
        log: q.log,
    };
    w.step(log_i);
    w
}

/// Sieves special-q lattices.
pub struct Sieve {
    poly: Poly,
    log_i: u32,
    log_j: u32,
    rational: Side,
    algebraic: Side,
    /// `f`'s coefficients, and `g`'s as `(m, l)`, as `f64` for the
    /// thresholds.
    f: Vec<f64>,
    g: (f64, f64),
    /// `f`'s coefficients, and `g`'s as `(m, l)`, exactly, for checking
    /// candidates: `g`'s also in 128 bits when they fit, as they do below
    /// about 480 bits, for speed.
    f_wide: Vec<(Wide, bool)>,
    g_wide: (Wide, Wide),
    g_small: Option<(u128, i128)>,
    marked: Vec<u64>,
}

const TOO_WIDE: &str = "polynomial coefficients too large";

impl Sieve {
    /// A sieve for `poly`, whose coefficients must be below 2^256 so that
    /// the norms fit in 512 bits, as [`parse_poly`](super::parse_poly)
    /// checks.
    pub fn new(poly: &Poly, fb: &FactorBase, params: &Params) -> Self {
        let log_i = params.log_i;
        let log_j = log_i - 1;
        let slab = 1usize << SLAB_BITS.min(log_i + log_j).max(log_i);
        Self {
            poly: poly.clone(),
            log_i,
            log_j,
            rational: Side::new(
                &fb.rational,
                (params.rational_lpb, params.rational_mfb),
                slab,
            ),
            algebraic: Side::new(
                &fb.algebraic,
                (params.algebraic_lpb, params.algebraic_mfb),
                slab,
            ),
            f: poly.f.iter().map(Coeff::to_f64).collect(),
            g: (poly.m.to_f64(), poly.l.to_f64()),
            f_wide: poly.f.iter().map(|c| to_wide(c).expect(TOO_WIDE)).collect(),
            g_wide: (
                to_wide(&poly.m).expect(TOO_WIDE).0,
                to_wide(&poly.l).expect(TOO_WIDE).0,
            ),
            g_small: u128::try_from(&poly.m)
                .ok()
                .zip(i128::try_from(&poly.l).ok()),
            marked: vec![0; slab / 64],
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
        let (m, l) = self.g;
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
    /// below [`CHARACTER_PRIMES`]. A `q` within the factor base is skipped by the sieve,
    /// whose lattice it divides, and divided out directly.
    pub fn special_q(&mut self, q: u64, r: u64) -> Vec<Relation> {
        let (log_i, log_j) = (self.log_i, self.log_j);
        let width = 1usize << log_i;
        let half = (width / 2) as i64;
        let (u, v) = reduce(q, r, self.poly.skew);
        self.rational.start(u, v, log_i, log_j);
        self.algebraic.start(u, v, log_i, log_j);
        let r_slack = self.rational.slack();
        let a_slack = self.algebraic.slack() + log2(q as f64);
        let segment = SEGMENT.min(width);
        let rows_per_slab = (self.marked.len() * 64) >> log_i;
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
                        let rat = &self.rational.block[base..base + pieces];
                        let alg = &self.algebraic.block[base..base + pieces];
                        // Few positions pass, and whether any in a chunk
                        // does is a vectorizable reduction.
                        for (k, (r, a)) in rat.chunks(CHUNK).zip(alg.chunks(CHUNK)).enumerate() {
                            let any = r
                                .iter()
                                .zip(a)
                                .fold(false, |any, (&r, &a)| any | (r >= r_t) & (a >= a_t));
                            if !any {
                                continue;
                            }
                            for (i, (&r, &a)) in r.iter().zip(a).enumerate() {
                                let x = base + k * CHUNK + i;
                                if r >= r_t && a >= a_t && (j % 2 == 1 || x % 2 == 1) {
                                    candidates.push(x as u32);
                                }
                            }
                        }
                    }
                }
            }
            if candidates.is_empty() {
                continue;
            }
            for &x in &candidates {
                self.marked[x as usize / 64] |= 1 << (x % 64);
            }
            hits.clear();
            self.rational
                .resieve(j0, log_i, &candidates, &self.marked, &mut hits);
            let split = hits.len();
            self.algebraic
                .resieve(j0, log_i, &candidates, &self.marked, &mut hits);
            for &x in &candidates {
                self.marked[x as usize / 64] = 0;
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
        let (mut v, negative) = match self.g_small {
            Some((m, l)) => {
                let bm = i128::try_from(u128::from(b).checked_mul(m)?).ok()?;
                let la = l.checked_mul(a.into())?;
                (wide(la.checked_sub(bm)?.unsigned_abs()), la < bm)
            }
            None => rational(&self.g_wide, a, b)?,
        };
        let mut rational = Vec::new();
        self.rational.divide(a, b, x, r_hits, &mut v, &mut rational);
        let r_left = self.rational.cofactor(&v, &mut rational)?;

        // Algebraic side: F(a, b), which q divides.
        let mut v = norm(&self.f_wide, a, b);
        let mut algebraic = Vec::new();
        match u32::try_from(q) {
            Ok(q) => divide_out(&mut v, q, &mut algebraic),
            Err(_) => divide_out_wide(&mut v, q, &mut algebraic),
        }
        self.algebraic
            .divide(a, b, x, a_hits, &mut v, &mut algebraic);
        let a_left = self.algebraic.cofactor(&v, &mut algebraic)?;

        // Splitting cofactors costs the most, so it waits until both sides
        // have passed the cheap tests: most candidates fail one of them.
        for (side, left, factors) in [
            (&self.rational, r_left, &mut rational),
            (&self.algebraic, a_left, &mut algebraic),
        ] {
            if let Some(c) = left
                && !side.large_primes(c, factors)
            {
                return None;
            }
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
    /// cofactor, the primes not sieved, and rounding.
    fn slack(&self) -> f64 {
        f64::from(self.mfb) + self.skipped + FUDGE
    }

    /// The cheap tests of `v`, what trial division left: `None` if it
    /// can't be a product of primes below the large-prime bound and no
    /// larger than the cofactor bound, `Some(None)` if it's 1 or such a
    /// prime, which is added to `factors`, and otherwise `Some(Some(c))`
    /// for [`large_primes`](Self::large_primes) to split.
    fn cofactor(&self, v: &Wide, factors: &mut Vec<u64>) -> Option<Option<u128>> {
        let c = to_u128(v).filter(|&c| c != 0 && c <= self.cofactor_bound)?;
        let (large, b) = (u128::from(self.large_bound), u128::from(self.largest));
        if c == 1 {
            Some(None)
        } else if c < large {
            factors.push(c as u64);
            Some(None)
        } else if c < self.prime_below.into() {
            // A prime too large.
            None
        } else if c < b * b * b
            && (c >= large * large || (c < 1 << 62 && c % 2 == 1 && probable_prime(c as u64)))
        {
            // At most two primes, not both below L, or a prime.
            None
        } else {
            Some(Some(c))
        }
    }

    /// Whether `c > 1`, which no factor-base prime divides, is a product of
    /// primes below the large-prime bound, which are added to `factors`.
    /// The large-prime bound, `L`, must be below `B²`, for `B` the largest
    /// factor-base prime, so that any `c` below it is prime.
    fn large_primes(&self, c: u128, factors: &mut Vec<u64>) -> bool {
        let large = u128::from(self.large_bound);
        if c < large {
            factors.push(c as u64);
            return true;
        }
        // A prime too large.
        if c < self.prime_below.into() {
            return false;
        }
        // Below B³ there are at most two primes, both below L only if c
        // is below L², and then a 64-bit split is quickest if it fits.
        let b = u128::from(self.largest);
        if c < b * b * b {
            if c >= large * large {
                return false;
            }
            if c < 1 << 62 {
                return match split(c as u64) {
                    Some([p, q]) if q < self.large_bound => {
                        factors.extend([p, q]);
                        true
                    }
                    _ => false,
                };
            }
        }
        // Otherwise two primes past 2^62, three or more, or a prime.
        if c.is_multiple_of(2) {
            return false;
        }
        let n = Cofactor::from_u64(c as u64)
            .wrapping_add(&(Cofactor::from_u64((c >> 64) as u64) << 64));
        match self.splitter.factor(&n).and_then(|g| to_u128(&g)) {
            Some(g) if g > 1 && g < c => {
                self.large_primes(g, factors) && self.large_primes(c / g, factors)
            }
            _ => false,
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

/// `x mod p` in `[0, p)`.
fn reduce_mod(x: i64, p: u32) -> u64 {
    let r = x % i64::from(p);
    (if r < 0 { r + i64::from(p) } else { r }) as u64
}

/// `xₗ⁻¹ mod pₗ` for each lane, by Euclid's algorithm in 32 bits, the
/// lanes in step so that their divisions overlap; 0 where `xₗ` is 0. The
/// cofactors of `x` alternate in sign, so only their magnitudes are kept,
/// and the last one's sign comes from the number of steps.
fn inverses(x: [u32; LANES], p: [u32; LANES]) -> [u32; LANES] {
    let (mut r0, mut r1) = (p, x);
    let (mut t0, mut t1) = ([0u32; LANES], [1u32; LANES]);
    let mut odd = [false; LANES];
    while r1.iter().any(|&r| r != 0) {
        for l in 0..LANES {
            if let Some(k) = r0[l].checked_div(r1[l]) {
                (r0[l], r1[l]) = (r1[l], r0[l] - k * r1[l]);
                (t0[l], t1[l]) = (t1[l], t0[l] + k * t1[l]);
                odd[l] = !odd[l];
            }
        }
    }
    // After n steps the cofactor has the sign of (−1)^(n+1).
    core::array::from_fn(|l| {
        if odd[l] || t0[l] == 0 {
            t0[l]
        } else {
            p[l] - t0[l]
        }
    })
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
fn divide_out(v: &mut Wide, p: u32, factors: &mut Vec<u64>) {
    while let (q, 0) = div_small(v, p) {
        if v.is_zero() {
            return;
        }
        *v = q;
        factors.push(p.into());
    }
}

/// [`divide_out`] for a special-q past 2^32, by slower long division,
/// once a relation.
fn divide_out_wide(v: &mut Wide, p: u64, factors: &mut Vec<u64>) {
    let p_wide = Wide::from_u64(p);
    while !v.is_zero() {
        let (q, r) = v.div_rem(&p_wide);
        if !r.is_zero() {
            return;
        }
        *v = q;
        factors.push(p);
    }
}

/// `v/p` and `v mod p`, in 32-bit pieces from the highest nonzero limb, so
/// that each step is a 64-bit division: on 64-bit limbs, dividing limb by
/// limb would take 128-bit ones, which are done in software.
// Limb is u32 or u64, so the casts are needed on one target and redundant
// on the other.
#[allow(clippy::unnecessary_cast)]
fn div_small(v: &Wide, p: u32) -> (Wide, u64) {
    const PIECES: u32 = Limb::BITS / 32;
    let limbs = v.as_limbs();
    let Some(top) = limbs.iter().rposition(|&l| l != 0) else {
        return (Wide::ZERO, 0);
    };
    let p = u64::from(p);
    let mut quotient = [0 as Limb; 512 / Limb::BITS as usize];
    let mut rem = 0u64;
    for i in (0..=top).rev() {
        let mut q = 0u64;
        for k in (0..PIECES).rev() {
            let t = (rem << 32) | ((limbs[i] as u64 >> (32 * k)) & 0xffff_ffff);
            q |= (t / p) << (32 * k);
            rem = t % p;
        }
        quotient[i] = q as Limb;
    }
    (Wide::from_limbs(quotient), rem)
}

/// `|l·a − m·b|` for `g = (m, l)`, and whether it's negative, or `None` if
/// it doesn't fit.
pub(crate) fn rational((m, l): &(Wide, Wide), a: i64, b: u64) -> Option<(Wide, bool)> {
    let (la, high_la) = l.widening_mul(&Wide::from_u64(a.unsigned_abs()));
    let (mb, high_mb) = m.widening_mul(&Wide::from_u64(b));
    if !high_la.is_zero() || !high_mb.is_zero() {
        return None;
    }
    Some(if a < 0 {
        (la.checked_add(&mb)?, true)
    } else if la >= mb {
        (la.wrapping_sub(&mb), false)
    } else {
        (mb.wrapping_sub(&la), true)
    })
}

/// `|F(a, b)| = |Σ cᵢ·aⁱ·b^(d−i)|`, exactly, for `f`'s coefficients
/// `(|cᵢ|, cᵢ < 0)`.
pub(crate) fn norm(f: &[(Wide, bool)], a: i64, b: u64) -> Wide {
    let d = f.len() - 1;
    let (mut pos, mut neg) = (Wide::ZERO, Wide::ZERO);
    let wa = Wide::from_u64(a.unsigned_abs());
    let wb = Wide::from_u64(b);
    for (i, &(c, negative)) in f.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let mut t = c;
        for _ in 0..i {
            t = t.widening_mul(&wa).0;
        }
        for _ in i..d {
            t = t.widening_mul(&wb).0;
        }
        if negative != (a < 0 && i % 2 == 1) {
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

/// `(|c|, c < 0)`, if `|c|` fits in a [`Wide`].
pub(crate) fn to_wide(c: &BigInt) -> Option<(Wide, bool)> {
    if c.bits() >= 512 {
        return None;
    }
    let w = c.iter_u64_digits().rev().fold(Wide::ZERO, |w, d| {
        (w << 64).wrapping_add(&Wide::from_u64(d))
    });
    Some((w, c.sign() == Sign::Minus))
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

/// Whether bit `x` of `marked` is set.
fn is_marked(marked: &[u64], x: u32) -> bool {
    marked[x as usize / 64] >> (x % 64) & 1 != 0
}
