//! Sieving for relations, one polynomial at a time.

use alloc::collections::BTreeSet;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use super::base::{FactorBase, rem};
use super::log2;
use super::modp::pow_mod;
use super::params::{BLOCK_SIZE, Params};
use crate::arith::{Limb, Uint};

/// A congruence `t² ≡ ±L·∏ p (mod N)` with every `p` in the factor base and
/// `L` either 1 or a single larger prime.
#[derive(Clone, Debug)]
pub struct Relation<const LIMBS: usize> {
    pub t: Uint<LIMBS>,
    /// Whether `t² − kN` is negative.
    pub negative: bool,
    /// Indices into the factor base, one per prime factor of `|t² − kN|`
    /// counted with multiplicity, in increasing order.
    pub factors: Vec<u32>,
    /// The prime factor of `t² − kN` outside the factor base, or 1 for a
    /// full relation. Two relations with the same large prime multiply to
    /// one with a square cofactor.
    pub large: u64,
}

/// Marks a prime that isn't sieved, only tried in trial division: 2, the
/// primes dividing `k`, which have a single root, and the primes dividing
/// the current `a`, which have one or none.
const UNSIEVED: u32 = u32::MAX;

/// Target size of each prime in `a`. Smaller primes would make more
/// choices of `a` collide, larger ones leave fewer polynomials per `a`.
const A_PRIME_BITS: f64 = 11.0;

/// Primes below this aren't sieved. Measured at 60 and 66 digits, skipping
/// them saves about a tenth of the time.
const SMALL_PRIMES: u32 = 100;

/// Sums within this many times the largest prime's log of `log₂ |g(x)|`
/// (allowing too for the unsieved primes) get trial-divided. A large
/// prime alone takes up to about twice that log. Measured at 60 digits,
/// 2.0 misses relations and 2.8 tries too many candidates.
const SLACK: f64 = 2.4;

/// `log₂ BLOCK_SIZE`.
const BLOCK_BITS: u32 = BLOCK_SIZE.trailing_zeros();

/// A sign and magnitude.
type Signed<const LIMBS: usize> = (bool, Uint<LIMBS>);

fn signed_add<const LIMBS: usize>(a: Signed<LIMBS>, b: Signed<LIMBS>) -> Signed<LIMBS> {
    match (a.0 == b.0, a.1 >= b.1) {
        (true, _) => (a.0, a.1.wrapping_add(&b.1)),
        (false, true) => (a.0, a.1.wrapping_sub(&b.1)),
        (false, false) => (b.0, b.1.wrapping_sub(&a.1)),
    }
}

/// `log₂ x` for `x > 0`, from its top 64 bits.
fn log2_uint<const LIMBS: usize>(x: &Uint<LIMBS>) -> f64 {
    let shift = x.bits().saturating_sub(64);
    log2((*x >> shift).to_u64().unwrap() as f64) + f64::from(shift)
}

/// The self-initializing quadratic sieve's polynomials and their sieving.
///
/// Each polynomial is `Q(x) = (ax + b)² − kN` for `x` in `[−M, M)`, with `a`
/// a product of `s` factor-base primes `q_j` and `b² ≡ kN (mod a)`, so that
/// `a` divides `Q(x)` and `g(x) = Q(x)/a` is what has to be smooth. With
/// `a ≈ √(2kN)/M`, `|g(x)|` stays below about `M·√(kN/2)` across the whole
/// interval, far smaller than a single polynomial's values get.
///
/// For one `a` there are `2^(s−1)` such `b`, namely `b = B₁ ± B₂ ± … ± B_s`
/// with `B_j² ≡ kN (mod q_j)` and `B_j ≡ 0` modulo the other `q`. Visiting
/// them in Gray-code order changes one sign at a time, which moves each root
/// of `Q` mod `p` by a precomputed `2·B_j/a mod p`: this is the
/// self-initialization.
///
/// The sieve adds `log₂ p` at every `x` where `p` divides `g(x)`. Where the
/// sum comes close to `log₂ |g(x)|`, trial division decides whether `g(x)`
/// is smooth, apart from at most one prime below the large-prime bound.
pub struct Sieve<const LIMBS: usize> {
    /// Shared with the other sieves on the same number.
    fb: Arc<FactorBase<LIMBS>>,
    /// Half the sieve interval.
    m: i64,
    large_bound: u64,
    /// How far below `log₂ |g(x)|` a sum may be and still be tried.
    slack: u32,
    /// Primes before this index aren't sieved: they are slow to sieve and
    /// add little. Trial division still tries them.
    first_sieved: usize,
    /// Primes from this index on are larger than a block, so they hit each
    /// block at most a few times. Instead of every block visiting every one
    /// of them, each polynomial drops their hits into [`Sieve::buckets`].
    first_large: usize,
    /// Runs of large primes as `(first index, end index, log)`, each at
    /// most 2^16 primes long and with one rounded log, so that a hit fits
    /// in 32 bits: the index within its run, and the offset in the block.
    runs: Vec<(usize, usize, u8)>,
    /// For each block, the hits of the large primes in it, as
    /// `(index − first index of its run) << 16 | offset`, run by run.
    buckets: Vec<Vec<u32>>,
    /// For each block, where each run's hits start in its bucket, and
    /// where the last one ends.
    run_starts: Vec<Vec<u32>>,
    block: Vec<u8>,
    rng: u64,
    /// Choices of `a`, by the factor-base indices of their primes.
    used: BTreeSet<Vec<usize>>,

    // The current `a`.
    a: Uint<LIMBS>,
    a_primes: Vec<usize>,
    /// `B_j` for each prime of `a`.
    bs: Vec<Uint<LIMBS>>,
    /// `2·B_j/a mod p`, indexed by `j` then factor-base prime.
    deltas: Vec<Vec<u32>>,

    // The current `b`.
    b: Signed<LIMBS>,
    c: Signed<LIMBS>,
    /// Whether each `B_j` is subtracted from `b`.
    minus: Vec<bool>,
    /// Which `b` of the current `a` this is, in Gray-code order.
    index: u64,
    /// The roots of `Q` mod each factor-base prime, or [`UNSIEVED`], as
    /// offsets `x + M` into the interval.
    roots: Vec<(u32, u32)>,
    /// Where each prime smaller than a block hits next in the current
    /// block, carried over from one block to the next.
    next: Vec<(u32, u32)>,
    /// `⌈2^64/p⌉` for each prime, for remainders by multiplication.
    magic: Vec<u64>,
}

impl<const LIMBS: usize> Sieve<LIMBS> {
    /// Starts sieving with the first polynomial. Different seeds choose
    /// different values of `a`.
    ///
    /// # Panics
    ///
    /// Panics if the factor base has fewer than 4 primes that can go into
    /// `a`.
    pub fn new(fb: Arc<FactorBase<LIMBS>>, params: &Params, seed: u64) -> Self {
        let largest = *fb.primes.last().unwrap();
        let large_bound = u64::from(largest.p) * u64::from(params.large_mult);
        let mut sieve = Self {
            fb: Arc::clone(&fb),
            m: (params.blocks * BLOCK_SIZE) as i64,
            large_bound,
            slack: 0,
            first_sieved: 0,
            first_large: fb.primes.partition_point(|p| (p.p as usize) < BLOCK_SIZE),
            runs: Vec::new(),
            buckets: vec![Vec::new(); 2 * params.blocks],
            run_starts: vec![Vec::new(); 2 * params.blocks],
            block: vec![0; BLOCK_SIZE],
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            used: BTreeSet::new(),
            a: Uint::ONE,
            a_primes: Vec::new(),
            bs: Vec::new(),
            deltas: Vec::new(),
            b: (false, Uint::ZERO),
            c: (false, Uint::ZERO),
            minus: Vec::new(),
            index: 0,
            roots: vec![(UNSIEVED, UNSIEVED); fb.primes.len()],
            next: Vec::new(),
            magic: fb
                .primes
                .iter()
                .map(|p| (u64::MAX / u64::from(p.p)).wrapping_add(1))
                .collect(),
        };
        sieve.first_sieved = fb.primes.partition_point(|p| p.p < SMALL_PRIMES);
        let mut start = sieve.first_large;
        while start < fb.primes.len() {
            let log = fb.primes[start].log;
            let len = fb.primes[start..]
                .iter()
                .take(1 << 16)
                .take_while(|p| p.log == log)
                .count();
            sieve.runs.push((start, start + len, log));
            start += len;
        }
        // What the unsieved primes would add on average: log₂ p from each
        // of two roots, at 1 in p of the positions.
        let k = fb.multiplier;
        let skipped: f64 = fb.primes[..sieve.first_sieved]
            .iter()
            .filter(|p| p.p > 2 && !k.is_multiple_of(p.p))
            .map(|p| 2.0 * f64::from(p.log) / f64::from(p.p - 1))
            .sum();
        sieve.slack = (SLACK * f64::from(largest.log) + skipped) as u32;
        sieve.new_a();
        sieve
    }

    fn random(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// The factor-base indices of a fresh `a` close to `√(2kN)/M`.
    fn choose_a(&mut self) -> Vec<usize> {
        let fb = Arc::clone(&self.fb);
        let k = fb.multiplier;
        let primes = &fb.primes;
        let usable: Vec<usize> = (0..primes.len())
            .filter(|&i| primes[i].p > 2 && !k.is_multiple_of(primes[i].p))
            .collect();
        assert!(usable.len() >= 4, "factor base too small");
        let target = (log2_uint(&self.fb.kn) + 1.0) / 2.0 - log2(self.m as f64);
        let max_bits = log2(f64::from(primes[*usable.last().unwrap()].p)) - 1.0;
        // The fewest primes that keep each within the target size, rounded
        // up by hand: `core` has no `ceil`.
        let s = target / A_PRIME_BITS.min(max_bits);
        let s = (s as usize + usize::from(s > (s as usize) as f64)).clamp(1, usable.len() / 2);
        let per = target / s as f64;
        let bits = |i: usize| log2(f64::from(primes[i].p));
        // Candidates for all but the last prime, within a factor of 2 of
        // the ideal size, widened if that leaves too few.
        let mut width = 1.0;
        let pool = loop {
            let pool: Vec<usize> = usable
                .iter()
                .copied()
                .filter(|&i| (bits(i) - per).abs() <= width)
                .collect();
            if pool.len() >= 2 * s + 4 || pool.len() == usable.len() {
                break pool;
            }
            width += 1.0;
        };
        for attempt in 0.. {
            let mut chosen: Vec<usize> = Vec::with_capacity(s);
            while chosen.len() + 1 < s {
                let i = pool[self.random() as usize % pool.len()];
                if !chosen.contains(&i) {
                    chosen.push(i);
                }
            }
            // The last prime brings the product as close to the target as
            // the factor base allows.
            let rest = target - chosen.iter().map(|&i| bits(i)).sum::<f64>();
            let mut order: Vec<usize> = usable
                .iter()
                .copied()
                .filter(|i| !chosen.contains(i))
                .collect();
            order.sort_by(|&x, &y| (bits(x) - rest).abs().total_cmp(&(bits(y) - rest).abs()));
            // Later attempts take the next-best last prime, so that a
            // small pool still yields new choices.
            let last = order[(attempt / 8).min(order.len() - 1)];
            chosen.push(last);
            chosen.sort_unstable();
            if self.used.insert(chosen.clone()) {
                return chosen;
            }
        }
        unreachable!()
    }

    /// Moves to the first `b` of a fresh `a`.
    fn new_a(&mut self) {
        let fb = Arc::clone(&self.fb);
        self.a_primes = self.choose_a();
        self.a = self.a_primes.iter().fold(Uint::ONE, |a, &i| {
            a.overflowing_mul_limb(fb.primes[i].p as Limb).0
        });
        self.bs = self
            .a_primes
            .iter()
            .map(|&i| {
                let q = fb.primes[i];
                let (rest, _) = self.a.div_rem_limb(q.p as Limb);
                let q64 = u64::from(q.p);
                let inv = pow_mod(rem(&rest, q64), q64 - 2, q64);
                let mut gamma = u64::from(q.sqrt) * inv % q64;
                if gamma > q64 / 2 {
                    gamma = q64 - gamma;
                }
                rest.overflowing_mul_limb(gamma as Limb).0
            })
            .collect();
        self.minus = vec![false; self.bs.len()];
        self.index = 0;
        self.b = self
            .bs
            .iter()
            .fold((false, Uint::ZERO), |b, bj| signed_add(b, (false, *bj)));

        let k = fb.multiplier;
        self.deltas = vec![vec![0; fb.primes.len()]; self.bs.len()];
        for (i, prime) in fb.primes.iter().enumerate() {
            let p = u64::from(prime.p);
            if p == 2 || k.is_multiple_of(prime.p) || self.a_primes.contains(&i) {
                self.roots[i] = (UNSIEVED, UNSIEVED);
                continue;
            }
            let ainv = pow_mod(rem(&self.a, p), p - 2, p);
            for (j, bj) in self.bs.iter().enumerate() {
                self.deltas[j][i] = (2 * rem(bj, p) * ainv % p) as u32;
            }
            // Q(x) ≡ 0 where ax + b ≡ ±sqrt(kN).
            let bp = rem(&self.b.1, p);
            let bp = if self.b.0 { (p - bp) % p } else { bp };
            let s = u64::from(prime.sqrt);
            let m = self.m as u64 % p;
            let r1 = ((s + p - bp) * ainv + m) % p;
            let r2 = ((2 * p - s - bp) * ainv + m) % p;
            self.roots[i] = (r1 as u32, r2 as u32);
        }
    }

    /// Moves to the next polynomial: the next `b` in Gray-code order, or a
    /// fresh `a` once they run out.
    fn next_poly(&mut self) {
        self.index += 1;
        if self.index >= 1 << (self.bs.len() - 1) {
            self.new_a();
            return;
        }
        // B₁ keeps its sign: b and −b give the same values.
        let j = self.index.trailing_zeros() as usize + 1;
        let was_minus = self.minus[j];
        self.minus[j] = !was_minus;
        // Subtracting 2·B_j from b adds 2·B_j/a to each root, and adding
        // it subtracts.
        let two_bj = self.bs[j] << 1;
        self.b = signed_add(self.b, (!was_minus, two_bj));
        for (i, roots) in self.roots.iter_mut().enumerate() {
            if roots.0 == UNSIEVED {
                continue;
            }
            let p = self.fb.primes[i].p;
            let d = if was_minus {
                p - self.deltas[j][i]
            } else {
                self.deltas[j][i]
            };
            let step = |r: u32| {
                let r = r + d;
                if r >= p { r - p } else { r }
            };
            *roots = (step(roots.0), step(roots.1));
        }
    }

    /// The current `a` and `b`, for checking.
    pub fn poly(&self) -> (Uint<LIMBS>, bool, Uint<LIMBS>) {
        (self.a, self.b.0, self.b.1)
    }

    /// `|ax + b|`.
    fn t(&self, x: i64) -> Uint<LIMBS> {
        let ax = self.a.overflowing_mul_limb(x.unsigned_abs() as Limb).0;
        signed_add((x < 0, ax), self.b).1
    }

    /// `c = (b² − kN)/a`, so that `g(x) = ax² + 2bx + c`.
    fn c(&self) -> Signed<LIMBS> {
        let (b2, hi) = self.b.1.widening_square();
        assert!(hi.is_zero(), "b too large");
        let (c, r) = signed_add((false, b2), (true, self.fb.kn))
            .1
            .div_rem(&self.a);
        debug_assert!(r.is_zero(), "a doesn't divide b² − kN");
        (b2 < self.fb.kn, c)
    }

    /// `g(x) = ((ax + b)² − kN)/a = (ax + 2b)·x + c` as its sign and
    /// magnitude.
    fn value(&self, x: i64) -> Signed<LIMBS> {
        let abs_x = x.unsigned_abs() as Limb;
        let ax = self.a.overflowing_mul_limb(abs_x).0;
        let (negative, u) = signed_add((x < 0, ax), (self.b.0, self.b.1 << 1));
        let ux = u.overflowing_mul_limb(abs_x).0;
        signed_add((negative != (x < 0), ux), self.c)
    }

    /// Sieves the current polynomial over `[−M, M)`, then moves to the next
    /// one. Returns the relations found, full and partial.
    pub fn sieve_poly(&mut self) -> Vec<Relation<LIMBS>> {
        self.c = self.c();
        // |g| peaks at the ends of the interval and at its minimum near
        // x = 0, at roughly the same size.
        let peak = self.value(-self.m).1.max(self.value(0).1);
        let threshold = peak.bits().saturating_sub(self.slack);
        let threshold = threshold.min(u32::from(u8::MAX)) as u8;
        let mut relations = Vec::new();
        self.next.clear();
        self.next.extend_from_slice(&self.roots[..self.first_large]);
        self.fill_buckets();
        let mut candidates = Vec::new();
        let mut large = Vec::new();
        for block in 0..self.buckets.len() {
            self.sieve_block(block);
            candidates.clear();
            // Most 64-byte chunks hold no candidate, and their maximum is a
            // vectorizable reduction.
            for (c, chunk) in self.block.as_chunks::<64>().0.iter().enumerate() {
                if chunk.iter().fold(0, |m, &b| m.max(b)) >= threshold {
                    let base = 64 * c as u32;
                    candidates.extend(
                        (0..64)
                            .filter(|&j| chunk[j as usize] >= threshold)
                            .map(|j| base + j),
                    );
                }
            }
            if candidates.is_empty() {
                continue;
            }
            // The large primes dividing each candidate are the ones whose
            // hits landed on it.
            large.clear();
            large.resize(candidates.len(), Vec::new());
            let (bucket, starts) = (&self.buckets[block], &self.run_starts[block]);
            for (r, &(first, _, _)) in self.runs.iter().enumerate() {
                for &hit in &bucket[starts[r] as usize..starts[r + 1] as usize] {
                    let j = hit & 0xffff;
                    if self.block[j as usize] >= threshold
                        && let Ok(c) = candidates.binary_search(&j)
                    {
                        large[c].push(first as u32 + (hit >> 16));
                    }
                }
            }
            let start = (block * BLOCK_SIZE) as u32;
            for (&j, large) in candidates.iter().zip(&large) {
                if let Some(relation) = self.trial_divide(start + j, large) {
                    relations.push(relation);
                }
            }
        }
        self.next_poly();
        relations
    }

    /// Drops every hit of the large primes in `[−M, M)` into the bucket for
    /// its block.
    fn fill_buckets(&mut self) {
        for (bucket, starts) in self.buckets.iter_mut().zip(&mut self.run_starts) {
            bucket.clear();
            starts.clear();
        }
        let len = 2 * self.m as u32;
        for &(first, end, _) in &self.runs {
            for (bucket, starts) in self.buckets.iter().zip(&mut self.run_starts) {
                starts.push(bucket.len() as u32);
            }
            let primes = &self.fb.primes[first..end];
            for (i, (prime, &(r1, r2))) in primes.iter().zip(&self.roots[first..end]).enumerate() {
                if r1 == UNSIEVED {
                    continue;
                }
                let tag = (i as u32) << 16;
                for r in [r1, r2] {
                    // Positions stay below 2^32: the interval and the
                    // primes are both far smaller.
                    let mut pos = r;
                    while pos < len {
                        self.buckets[(pos >> BLOCK_BITS) as usize].push(tag | (pos & 0xffff));
                        pos += prime.p;
                    }
                }
            }
        }
        for (bucket, starts) in self.buckets.iter().zip(&mut self.run_starts) {
            starts.push(bucket.len() as u32);
        }
    }

    /// Adds up the logs of the primes dividing `g(x)` over block `block`,
    /// which is the next one for the primes smaller than a block.
    fn sieve_block(&mut self, block: usize) {
        self.block.fill(0);
        let range = self.first_sieved..self.first_large;
        let primes = &self.fb.primes[range.clone()];
        for (prime, next) in primes.iter().zip(&mut self.next[range]) {
            if next.0 == UNSIEVED {
                continue;
            }
            let p = prime.p as usize;
            for r in [&mut next.0, &mut next.1] {
                let mut j = *r as usize;
                while j < BLOCK_SIZE {
                    self.block[j] = self.block[j].saturating_add(prime.log);
                    j += p;
                }
                *r = (j - BLOCK_SIZE) as u32;
            }
        }
        let (bucket, starts) = (&self.buckets[block], &self.run_starts[block]);
        for (r, &(_, _, log)) in self.runs.iter().enumerate() {
            for &hit in &bucket[starts[r] as usize..starts[r + 1] as usize] {
                let j = (hit & 0xffff) as usize;
                self.block[j] = self.block[j].saturating_add(log);
            }
        }
    }

    /// Factors `a·g(x)` at offset `i = x + M` over the factor base, if
    /// `g(x)` is smooth apart from at most one prime below the large-prime
    /// bound. `large` lists the primes larger than a block that hit `i`.
    fn trial_divide(&self, i: u32, large: &[u32]) -> Option<Relation<LIMBS>> {
        let x = i64::from(i) - self.m;
        let (negative, mut g) = self.value(x);
        if g.is_zero() {
            return None;
        }
        let mut factors = Vec::new();
        let n = self.first_large;
        let primes = self.fb.primes[..n].iter().zip(&self.roots).zip(&self.magic);
        for (index, ((prime, &(r1, r2)), &magic)) in primes.enumerate() {
            let p = prime.p;
            if r1 != UNSIEVED {
                // A sieved prime divides g(x) exactly at its roots, which
                // is cheaper to check than a remainder of g(x). This is
                // Lemire's remainder by multiplication: the low 64 bits
                // of i·⌈2^64/p⌉ are the fraction i/p in fixed point.
                let low = magic.wrapping_mul(u64::from(i));
                let ip = ((u128::from(low) * u128::from(p)) >> 64) as u32;
                if ip != r1 && ip != r2 {
                    continue;
                }
            }
            loop {
                let (quot, r) = g.div_rem_limb(p as Limb);
                if r != 0 {
                    break;
                }
                g = quot;
                factors.push(index as u32);
            }
        }
        for &index in large {
            let p = self.fb.primes[index as usize].p;
            loop {
                let (quot, r) = g.div_rem_limb(p as Limb);
                if r != 0 {
                    break;
                }
                g = quot;
                factors.push(index);
            }
        }
        // Every prime below the largest factor-base prime that can divide
        // g(x) is in the factor base, so a cofactor below the bound, which
        // is less than its square, is prime.
        let large = g.to_u64().filter(|&g| g < self.large_bound)?;
        // And a itself, which divides Q(x) = a·g(x).
        factors.extend(self.a_primes.iter().map(|&i| i as u32));
        factors.sort_unstable();
        Some(Relation {
            t: self.t(x),
            negative,
            factors,
            large,
        })
    }
}
