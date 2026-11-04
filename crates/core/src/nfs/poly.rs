//! Polynomial selection: the base-m method, then Murphy's size and root
//! optimization.
//!
//! GNFS needs two polynomials with a common root `m` modulo `N`. Writing
//! `N` in base `m` gives `f` with `f(m) = N`, and the other is the linear
//! `g(x) = x − m`. A good `f` has small values over the sieve region (its
//! size) and many roots modulo small primes, which makes those values more
//! often smooth (its α).
//!
//! Two changes keep `f(m) ≡ 0 (mod N)` while changing both: translating
//! `x` to `x + k`, which moves the root to `m − k`, and rotating `f` by a
//! multiple of `g`. Size optimization searches both for the smallest
//! values; root optimization sieves over small rotations for the most
//! roots modulo small primes. Murphy's E, which weighs both sides' sizes
//! and α together, picks the winner.

use alloc::vec;
use alloc::vec::Vec;

use num_bigint::BigInt;

use super::kleinjung;
use super::modp::{Coeff, eval, inv, reduce};
use super::params::Params;
use super::score::{Murphy, lognorm};
use crate::arith::{Limb, Uint};
use crate::factor::root;
use crate::parallel::{Parallel, Serial, map_dynamic};
use crate::primes::Primes;
use crate::qs::log2;

/// Leading coefficients tried.
const LEADS: u64 = 2000;

/// Values of `m` tried below the largest for each leading coefficient.
const OFFSETS: u64 = 4;

/// Base-m candidates kept by size for optimizing.
const FINALISTS: usize = 32;

/// Kleinjung's candidates come from leading coefficients that are the
/// first multiples of `KJ_STEP`, as many as [`Effort`] says, with `u` up
/// to `KJ_SPREAD·P²`, at most `KJ_MAX` in all.
const KJ_STEP: u64 = 12;
const KJ_SPREAD: u64 = 4;
const KJ_MAX: usize = 20000;

/// Size-optimized candidates kept for root optimization.
const ROOT_FINALISTS: usize = 32;

/// How hard to search, by the size of `N`: the leading coefficients for
/// Kleinjung's method, and the rotations `(j₁·x + j₀)·g` root optimization
/// sieves, `|j₁| ≤ rot1` (for degree 4 and up) and `|j₀| < rot0`. The
/// search grows with the time sieving will take, from the small settings
/// below 210 bits to the full ones from 290; at 90 and 100 digits, the
/// full ones made sieving a third and three fifths faster.
#[derive(Clone, Copy, Debug)]
struct Effort {
    leads: u64,
    rot1: i64,
    rot0: i64,
}

impl Effort {
    fn for_bits(bits: u32) -> Self {
        // 0 up to 210 bits, 4 from 290.
        let step = (bits.saturating_sub(210) / 20).min(4);
        Self {
            leads: (125 << step).min(2000),
            rot1: 4 << step.saturating_sub(2),
            rot0: 1 << (12 + step),
        }
    }
}

/// Rotations with the best α kept for scoring by size.
const ROOT_CELLS: usize = 32;

/// Of those, the best by size plus α kept for Murphy's E.
const ROOT_KEEP: usize = 4;

/// Steps tried in size optimization, from `2^SIZE_STEPS` down to 1, each
/// for at most `SIZE_PASSES` passes over the moves.
const SIZE_STEPS: u32 = 48;
const SIZE_PASSES: usize = 8;

/// α counts the roots modulo primes up to this.
const ALPHA_PRIMES: u64 = 200;

/// Two polynomials for GNFS: `f` of degree `d` and `g(x) = l·x − m`, with
/// `F(m, l) = l^d·f(m/l) = N`, so that `m/l` is a common root modulo `N`.
///
/// The coefficients are of any size: those [`Poly::select`] chooses fit
/// in 128 bits, but past about 480 bits polynomials come from other
/// programs, through [`parse_poly`](super::parse_poly).
#[derive(Clone, Debug, PartialEq)]
pub struct Poly {
    /// The coefficients of `f`, lowest degree first.
    pub f: Vec<BigInt>,
    /// `g`'s constant term, negated: positive.
    pub m: BigInt,
    /// `g`'s leading coefficient: 1 for base-m, and positive.
    pub l: BigInt,
    /// How much wider the sieve region should be in `a` than in `b`: the
    /// `s` that makes `f`'s values smallest.
    pub skew: f64,
    /// `log₂` of the L² norm of `f`'s values over a region of area 1 at
    /// that skew.
    pub size: f64,
    /// Murphy's α in bits: how much smaller than its size a value of `f`
    /// behaves for smoothness, from its roots modulo small primes. Negative
    /// is better.
    pub alpha: f64,
    /// Murphy's E: how often values of both polynomials are smooth at once,
    /// for the parameters of this size of `N`. Larger is better.
    pub e: f64,
}

/// A candidate during selection, whose coefficients all fit in machine
/// integers, which keeps the search fast.
#[derive(Clone)]
struct Small {
    f: Vec<i128>,
    m: u128,
    l: u128,
    size: f64,
}

impl Small {
    fn new(f: Vec<i128>, m: u128, l: u128) -> Self {
        let size = lognorm(&f).1;
        Self { f, m, l, size }
    }
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
    /// Chooses polynomials for `n`: for each leading coefficient `a`, `m`
    /// just below `(n/a)^(1/d)` makes `f`'s leading coefficient `a`, and the
    /// others are `n`'s digits in base `m`, centered on 0. The smallest are
    /// size-optimized and root-optimized, and the best by Murphy's E wins.
    ///
    /// `None` if `m` doesn't fit in 96 bits, which happens past about 480
    /// bits: numbers that large need polynomials from elsewhere.
    pub fn select<const LIMBS: usize>(n: &Uint<LIMBS>) -> Option<Self> {
        Self::select_with(n, &Serial)
    }

    /// [`Poly::select`], with Kleinjung's search, size optimization and
    /// root optimization shared out by `par`.
    pub fn select_with<const LIMBS: usize>(n: &Uint<LIMBS>, par: &impl Parallel) -> Option<Self> {
        let d = degree(n.bits());
        let murphy = Murphy::new(&Params::for_bits(n.bits()));
        let mut candidates: Vec<Small> = Vec::new();
        for a in 1..=LEADS {
            let (q, _) = n.div_rem(&Uint::from_u64(a));
            let top = root(&q, d as u32);
            for k in 0..OFFSETS {
                let m = top.wrapping_sub(&Uint::from_u64(k));
                if let Some(poly) = base_m(n, &m, d) {
                    candidates.push(poly);
                }
            }
        }
        if candidates.is_empty() {
            return None;
        }
        candidates.sort_by(|x, y| x.size.total_cmp(&y.size));
        candidates.truncate(FINALISTS);
        // Kleinjung's candidates, whose a_{d−1} is small.
        let p0 = kleinjung_prime(&candidates[0]);
        let u_max = (KJ_SPREAD * p0 * p0) as i64;
        let effort = Effort::for_bits(n.bits());
        let found = map_dynamic(par, effort.leads as usize, |i| {
            kleinjung::candidates(n, d, KJ_STEP * (i as u64 + 1), p0, u_max)
        });
        candidates.extend(
            found
                .into_iter()
                .flatten()
                .take(KJ_MAX)
                .map(|(f, m, l)| Small::new(f, m, l)),
        );
        // Size-optimize them all; nearby m often optimize to the same one.
        let mut sized = map_dynamic(par, candidates.len(), |i| {
            let c = &candidates[i];
            let (f, m) = size_optimize(&c.f, c.m, c.l, true);
            Small::new(f, m, c.l)
        });
        sized.sort_by(|x, y| x.size.total_cmp(&y.size));
        sized.dedup_by(|x, y| x.f == y.f && x.m == y.m);
        sized.truncate(ROOT_FINALISTS);
        let rated = map_dynamic(par, sized.len(), |i| {
            let c = &sized[i];
            root_optimize(&c.f, c.m, c.l, effort)
                .into_iter()
                .map(|(f, m)| {
                    let mut poly = Self::from_small(&f, m, c.l);
                    poly.rate(&murphy);
                    poly
                })
                .collect::<Vec<_>>()
        });
        rated
            .into_iter()
            .flatten()
            .reduce(|best, poly| if poly.e > best.e { poly } else { best })
    }

    /// The pair `f`, `l·x − m`, with its skew and size.
    fn new(f: Vec<BigInt>, m: BigInt, l: BigInt) -> Self {
        let (skew, size) = lognorm(&f);
        Self {
            f,
            m,
            l,
            skew,
            size,
            alpha: 0.0,
            e: 0.0,
        }
    }

    /// The pair `f`, `l·x − m`, with its skew, size, α and Murphy's E for
    /// `params`.
    pub(crate) fn rated(f: Vec<BigInt>, m: BigInt, l: BigInt, params: &Params) -> Self {
        let mut poly = Self::new(f, m, l);
        poly.rate(&Murphy::new(params));
        poly
    }

    fn from_small(f: &[i128], m: u128, l: u128) -> Self {
        Self::new(f.iter().map(|&c| c.into()).collect(), m.into(), l.into())
    }

    /// Fills in α and Murphy's E.
    fn rate(&mut self, murphy: &Murphy) {
        self.alpha = alpha(&self.f);
        let alpha_g = alpha(&[-&self.m, self.l.clone()]);
        let g = (self.m.to_f64(), self.l.to_f64());
        self.e = murphy.e(&self.f, g, self.skew, self.alpha, alpha_g);
    }

    /// Murphy's E for these polynomials with `params`.
    pub fn murphy_e(&self, params: &Params) -> f64 {
        let mut p = self.clone();
        p.rate(&Murphy::new(params));
        p.e
    }

    /// `n`'s digits in base `m`, centered on 0, as a polynomial of degree
    /// `d`, or `None` if the leading digit is 0, or it or `m` is 2^96 or
    /// more.
    pub fn base_m<const LIMBS: usize>(n: &Uint<LIMBS>, m: &Uint<LIMBS>, d: usize) -> Option<Self> {
        base_m(n, m, d).map(|c| Self::from_small(&c.f, c.m, c.l))
    }

    /// The degree of `f`.
    pub fn degree(&self) -> usize {
        self.f.len() - 1
    }
}

/// [`Poly::base_m`], as a candidate for selection.
fn base_m<const LIMBS: usize>(n: &Uint<LIMBS>, m: &Uint<LIMBS>, d: usize) -> Option<Small> {
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
    Some(Small::new(digits, m128, 1))
}

/// The size of the primes `ℓ` in Kleinjung's `l = q·ℓ₁·ℓ₂`: about
/// `m^(1/6)`, for a base-m candidate's `m`.
fn kleinjung_prime(base: &Small) -> u64 {
    let bits = 128 - base.m.leading_zeros();
    1u64 << ((bits + 3) / 6 + 1).clamp(6, 15)
}

/// `f(x + k)` and `g(x + k) = l·x − (m − l·k)`: the new `f` and `m`, or
/// `None` on overflow.
fn translate(f: &[i128], m: u128, l: u128, k: i128) -> Option<(Vec<i128>, u128)> {
    let lk = (l as i128).checked_mul(k)?;
    let m = (m as i128).checked_sub(lk).filter(|&m| m > 1)?;
    // Horner's rule with x + k for x: g ← g·(x + k) + cᵢ.
    let mut g: Vec<i128> = Vec::with_capacity(f.len());
    for &c in f.iter().rev() {
        g.push(0);
        for i in (1..g.len()).rev() {
            g[i] = g[i - 1].checked_add(g[i].checked_mul(k)?)?;
        }
        g[0] = g[0].checked_mul(k)?.checked_add(c)?;
    }
    Some((g, m as u128))
}

/// `f + j·xⁱ·(l·x − m)`, or `None` on overflow.
fn rotate(f: &[i128], m: u128, l: u128, i: usize, j: i128) -> Option<Vec<i128>> {
    let mut g = f.to_vec();
    g[i + 1] = g[i + 1].checked_add(j.checked_mul(l as i128)?)?;
    g[i] = g[i].checked_sub(j.checked_mul(m as i128)?)?;
    Some(g)
}

/// Searches translations and, if `rotations`, rotations by `xⁱ·g` for
/// `i ≤ d − 3` for the smallest size, by steps from `2^SIZE_STEPS` down to
/// 1 in each in turn. Translation leaves the number of roots modulo each
/// prime, and so α, as it is.
fn size_optimize(f: &[i128], m: u128, l: u128, rotations: bool) -> (Vec<i128>, u128) {
    let d = f.len() - 1;
    let moves = if rotations { d.saturating_sub(2) } else { 0 };
    let (mut f, mut m) = (f.to_vec(), m);
    let mut size = lognorm(&f).1;
    for e in (0..=SIZE_STEPS).rev() {
        let step = 1i128 << e;
        for _ in 0..SIZE_PASSES {
            let mut improved = false;
            // Move 0 translates; move i + 1 rotates by xⁱ·g.
            for mv in 0..=moves {
                for j in [step, -step] {
                    let next = if mv == 0 {
                        translate(&f, m, l, j)
                    } else {
                        rotate(&f, m, l, mv - 1, j).map(|g| (g, m))
                    };
                    let Some((g, gm)) = next else { continue };
                    let s = lognorm(&g).1;
                    if s < size - 1e-6 {
                        (f, m, size) = (g, gm, s);
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }
    (f, m)
}

/// Sieves rotations `f + (j₁·x + j₀)·g` for the most roots modulo
/// small primes, and returns the best few by size plus α, each
/// size-optimized by translation.
///
/// For a prime `p` and an `x` with `g(x) ≢ 0`, the rotation has the root `x`
/// exactly when `j₀ + j₁·x ≡ −f(x)/g(x) (mod p)`: for each `j₁`, an
/// arithmetic progression of `j₀`. Each root lowers α by
/// `p/(p + 1)·log₂ p/(p − 1)`.
/// A small prime for the root sieve: its weight in α, and the pairs
/// `(x, t)` with `f + t·g ≡ 0` at `x`.
struct RootPrime {
    p: u64,
    w: f32,
    roots: Vec<(u64, u64)>,
}

fn root_optimize(f: &[i128], m: u128, l: u128, effort: Effort) -> Vec<(Vec<i128>, u128)> {
    let d = f.len() - 1;
    let rot1 = if d >= 4 { effort.rot1 } else { 0 };
    let rot0 = effort.rot0;
    let width = 2 * rot0 as usize;
    // For each small prime, its weight and the pairs (x, t) with
    // f + t·g ≡ 0 at x: rotations by j₀ ≡ t − j₁·x gain a root at x.
    let primes: Vec<RootPrime> = Primes::new(2, ALPHA_PRIMES)
        .map(|p| {
            let w = (p as f64 / (p + 1) as f64 * log2(p as f64) / (p - 1) as f64) as f32;
            let fp = reduce(f, p);
            let mp = (m % u128::from(p)) as u64;
            let lp = (l % u128::from(p)) as u64;
            let roots = (0..p)
                .filter_map(|x| {
                    let gx = (lp * x % p + p - mp) % p;
                    (gx != 0).then(|| (x, (p - eval(&fp, x, p)) % p * inv(gx, p) % p))
                })
                .collect();
            RootPrime { p, w, roots }
        })
        .collect();
    // One row of j₀ at a time, so that it stays in cache, keeping the best
    // cells of all rows seen as (gain, j₁, index of j₀).
    let mut best: Vec<(f32, i64, usize)> = Vec::new();
    let mut row = vec![0f32; width];
    let mut order: Vec<u32> = (0..width as u32).collect();
    for j1 in -rot1..=rot1 {
        row.fill(0.0);
        for &RootPrime { p, w, ref roots } in &primes {
            for &(x, t) in roots {
                // j₀ ≡ t − j₁·x, at index j₀ + rot0.
                let r = (t as i64 - j1 * x as i64).rem_euclid(p as i64) as u64;
                let mut i = ((r + rot0 as u64) % p) as usize;
                while i < width {
                    row[i] += w;
                    i += p as usize;
                }
            }
        }
        let by_gain = |a: &u32, b: &u32| row[*b as usize].total_cmp(&row[*a as usize]);
        order.select_nth_unstable_by(ROOT_CELLS - 1, by_gain);
        best.extend(
            order[..ROOT_CELLS]
                .iter()
                .map(|&i| (row[i as usize], j1, i as usize)),
        );
        if best.len() > ROOT_CELLS {
            best.select_nth_unstable_by(ROOT_CELLS - 1, |a, b| b.0.total_cmp(&a.0));
            best.truncate(ROOT_CELLS);
        }
    }
    let mut rotated: Vec<(f64, Vec<i128>, u128)> = best
        .into_iter()
        .filter_map(|(_, j1, i)| {
            let (j1, j0) = (i128::from(j1), i as i128 - i128::from(rot0));
            let g = rotate(&rotate(f, m, l, 1, j1)?, m, l, 0, j0)?;
            let (g, gm) = size_optimize(&g, m, l, false);
            Some((lognorm(&g).1 + alpha(&g), g, gm))
        })
        .collect();
    rotated.sort_by(|x, y| x.0.total_cmp(&y.0));
    rotated
        .into_iter()
        .take(ROOT_KEEP)
        .map(|(_, g, gm)| (g, gm))
        .collect()
}

/// Murphy's α in bits: the sum over small primes `p` of
/// `(1 − q_p·p/(p+1))·log₂ p/(p − 1)`, with `q_p` the roots of `f` modulo
/// `p`, counting a root at infinity when `p` divides the leading
/// coefficient. A random integer has `1/(p − 1)` factors of `p` on
/// average, and a value of `f` about `q_p·p/(p² − 1)`.
pub fn alpha<C: Coeff>(f: &[C]) -> f64 {
    let lead = &f[f.len() - 1];
    Primes::new(2, ALPHA_PRIMES)
        .map(|p| {
            let mut q = super::modp::roots(f, p).len() as f64;
            if lead.mod_p(p) == 0 {
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
