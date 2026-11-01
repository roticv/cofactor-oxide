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
//! with one or two large primes and combines them through cycles, and finds the dependencies by
//! Gaussian elimination for small matrices and block Lanczos for large
//! ones.

mod base;
mod cofactor;
mod cycles;
mod job;
mod lanczos;
mod linalg;
mod modp;
mod params;
mod sieve;

use core::ops::ControlFlow;

use alloc::collections::BTreeSet;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::arith::{Monty, Uint};

use cycles::Cycles;

pub use base::{FactorBase, Prime, multiplier};
pub use job::{QsCollector, QsJob, QsSieve, Start};
pub use linalg::dependencies;
pub use modp::{pow_mod, sqrt_mod};
pub use params::{BLOCK_SIZE, DOUBLE_LARGE_BITS, Params};
pub use sieve::{Relation, Sieve};

/// Relations beyond the factor-base size (plus one for the sign) to
/// collect. Each is one more dependency, and each dependency splits `N`
/// with probability at least ½.
const EXTRA_RELATIONS: usize = 32;

/// The quadratic sieve set up for one number: its factor base and
/// parameters.
///
/// Sieving and collecting are separate so that they can run on different
/// threads: each thread sieves with its own [`Sieve`], from
/// [`Siqs::sieve`] with a different seed, and one [`Relations`] gathers
/// what they find until [`Relations::is_complete`], when [`Siqs::finish`]
/// turns it into a factor. [`qs`] does all of it on one thread.
#[derive(Clone, Debug)]
pub struct Siqs<const LIMBS: usize> {
    n: Uint<LIMBS>,
    params: Params,
    fb: Arc<FactorBase<LIMBS>>,
}

impl<const LIMBS: usize> Siqs<LIMBS> {
    /// Sets up the sieve for `n` with `params`.
    ///
    /// Some numbers need no sieving: `Err(Some(f))` gives a proper factor
    /// found along the way (2 for even `n`, a factor-base prime dividing
    /// `n`, or a factor from `kN` being a square), and `Err(None)` means
    /// `n` turned out to be one of the factor-base primes.
    ///
    /// # Panics
    ///
    /// Panics if `n` is too small for the factor base.
    pub fn new(n: &Uint<LIMBS>, params: &Params) -> Result<Self, Option<Uint<LIMBS>>> {
        let two = Uint::from_u64(2);
        if !n.is_odd() {
            return Err((*n > two).then_some(two));
        }
        let proper = |g: Uint<LIMBS>| (g != Uint::ONE && g != *n).then_some(g);
        let fb = FactorBase::new(n, multiplier(n), params.fb_size).map_err(proper)?;
        let s = fb.kn.isqrt();
        if s.widening_square() == (fb.kn, Uint::ZERO) {
            return Err(proper(s.gcd(n)));
        }
        Ok(Self {
            n: *n,
            params: *params,
            fb: Arc::new(fb),
        })
    }

    pub fn factor_base(&self) -> &FactorBase<LIMBS> {
        &self.fb
    }

    /// A sieve for this number. Sieves with different seeds find different
    /// relations.
    pub fn sieve(&self, seed: u64) -> Sieve<LIMBS> {
        Sieve::new(Arc::clone(&self.fb), &self.params, seed)
    }

    /// An empty collection of relations for this number.
    pub fn relations(&self) -> Relations<LIMBS> {
        Relations {
            monty: Monty::new(self.n).expect("n is odd"),
            target: self.fb.primes.len() + 1 + EXTRA_RELATIONS,
            rows: Vec::new(),
            seen: BTreeSet::new(),
            partials: Vec::new(),
            cycles: Cycles::default(),
        }
    }

    /// Finds a factor from the collected relations.
    ///
    /// Returns a proper factor of `n`, or `None` if every dependency gave a
    /// trivial one, which is what happens when `n` is prime or a prime
    /// power.
    pub fn finish(&self, relations: &Relations<LIMBS>) -> Option<Uint<LIMBS>> {
        let rows = &relations.rows;
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
        let n = self.n;
        dependencies(&matrix, self.fb.primes.len() + 1)
            .iter()
            .map(|dep| square_root(&relations.monty, &self.fb, rows, dep))
            .find(|&g| g != Uint::ONE && g != n)
    }
}

/// A row of the matrix: `t² ≡ ±S²·∏ p (mod N)`, from a full relation
/// (`S = 1`) or a cycle of partial relations, whose large primes each
/// appear twice and multiply to `S²`.
#[derive(Clone, Debug)]
struct Row<const LIMBS: usize> {
    /// `t mod N`, in Montgomery form.
    t: Uint<LIMBS>,
    negative: bool,
    factors: Vec<u32>,
    /// `S mod N`, in Montgomery form.
    large: Uint<LIMBS>,
}

/// Relations collected for one number, as rows of the matrix.
#[derive(Clone, Debug)]
pub struct Relations<const LIMBS: usize> {
    monty: Monty<LIMBS>,
    target: usize,
    rows: Vec<Row<LIMBS>>,
    /// The same relation can come from more than one polynomial.
    seen: BTreeSet<Uint<LIMBS>>,
    partials: Vec<Relation<LIMBS>>,
    cycles: Cycles,
}

impl<const LIMBS: usize> Relations<LIMBS> {
    /// Adds a relation. A full one is a row by itself; a partial one makes
    /// a row whenever its large primes close a cycle with earlier ones.
    pub fn add(&mut self, r: Relation<LIMBS>) {
        if !self.seen.insert(r.t) {
            return;
        }
        let monty = &self.monty;
        if r.large == [1, 1] {
            self.rows.push(Row {
                t: monty.to_monty(&r.t),
                negative: r.negative,
                factors: r.factors,
                large: monty.one(),
            });
            return;
        }
        let [a, b] = r.large;
        self.partials.push(r);
        let Some(cycle) = self.cycles.add(a, b, self.partials.len() - 1) else {
            return;
        };
        let mut row = Row {
            t: monty.one(),
            negative: false,
            factors: Vec::new(),
            large: monty.one(),
        };
        let mut large = Vec::new();
        for r in cycle.iter().map(|&i| &self.partials[i]) {
            row.t = monty.mul(&row.t, &monty.to_monty(&r.t));
            row.negative ^= r.negative;
            row.factors.extend(&r.factors);
            large.extend(r.large.iter().filter(|&&l| l != 1));
        }
        // Each large prime appears twice: take one of each pair.
        large.sort_unstable();
        debug_assert!(large.chunks(2).all(|pair| pair[0] == pair[1]));
        for pair in large.chunks(2) {
            row.large = monty.mul(&row.large, &monty.to_monty(&Uint::from_u64(pair[0])));
        }
        self.rows.push(row);
    }

    /// Rows so far.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Rows needed: one more than the columns, plus some to spare.
    pub fn target(&self) -> usize {
        self.target
    }

    pub fn is_complete(&self) -> bool {
        self.rows.len() >= self.target
    }
}

/// Factors `n` with the quadratic sieve on one thread, using `params`.
///
/// Returns a proper factor of `n`, or `None` if there is none to find,
/// which is the case when `n` is prime or a prime power. Even `n` greater
/// than 2 returns 2.
///
/// # Panics
///
/// Panics if `n` is too small for the factor base.
pub fn qs<const LIMBS: usize>(n: &Uint<LIMBS>, params: &Params) -> Option<Uint<LIMBS>> {
    qs_with(n, params, |_, _| ControlFlow::Continue(())).unwrap_or(None)
}

/// The callback asked [`qs_with`] to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stopped;

/// Factors `n` like [`qs`], calling `progress(rows, target)` after each
/// polynomial, and stopping if it returns `Break`.
pub fn qs_with<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    params: &Params,
    mut progress: impl FnMut(usize, usize) -> ControlFlow<()>,
) -> Result<Option<Uint<LIMBS>>, Stopped> {
    let siqs = match Siqs::new(n, params) {
        Ok(siqs) => siqs,
        Err(found) => return Ok(found),
    };
    let mut sieve = siqs.sieve(0);
    let mut relations = siqs.relations();
    while !relations.is_complete() {
        for r in sieve.sieve_poly() {
            relations.add(r);
        }
        if progress(relations.len(), relations.target()).is_break() {
            return Err(Stopped);
        }
    }
    Ok(siqs.finish(&relations))
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
        y = monty.mul(&y, &r.large);
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
