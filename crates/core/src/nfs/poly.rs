//! Polynomial selection: the base-m method and Kleinjung's, then size and
//! root optimization.
//!
//! GNFS needs two polynomials with a common root `m` modulo `N`. Writing
//! `N` in base `m` gives `f` with `f(m) = N`, and the other is the linear
//! `g(x) = x − m`. A good `f` has small values over the sieve region (its
//! size) and many roots modulo small primes, which makes those values more
//! often smooth (its α).
//!
//! Kleinjung's method, in `kleinjung.rs`, finds better starting points than
//! base-m: `g(x) = l·x − m` with `l` a product of chosen primes, which
//! makes `f`'s top coefficients small.
//!
//! Three changes keep `f(m/l) ≡ 0 (mod N)` while changing both
//! polynomials: translating `x` to `x + k`, which moves the root to
//! `m/l − k`; adding a multiple of `g` to `f`, rotation; and multiplying
//! `f` by a small integer first, which multiplies the resultant, `N`, by
//! it too. Size optimization searches these for the smallest values, with
//! LLL reduction as in CADO-NFS; root optimization sieves over small
//! rotations for the most roots modulo small primes. Murphy's E, which
//! weighs both sides' sizes and α together, picks the winner.

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
/// first `KJ_LEADS` multiples of [`Effort`]'s step, as in CADO-NFS, with
/// `u` up to `KJ_SPREAD·P²`, and as many special-q roots for each as
/// [`Effort`] says. The `SIZE_KEEP` smallest are size-optimized.
const KJ_LEADS: usize = 100;
const KJ_SPREAD: u64 = 4;
const SIZE_KEEP: usize = 2000;

/// Size-optimized candidates kept for root optimization.
const ROOT_FINALISTS: usize = 32;

/// How hard to search, by the size of `N`: the special-q roots for each of
/// Kleinjung's leading coefficients, and the rotations `(j₁·x + j₀)·g` root
/// optimization sieves, `|j₁| ≤ rot1` (for degree 4 and up) and
/// `|j₀| < rot0`. The search grows with the time sieving will take.
///
/// The special-q roots, `KJ_LEADS·nq` in all, double every 20 bits from
/// 100,000 at 266 bits to about 3 million at RSA-110's 364, then every 30,
/// as in CADO-NFS's parameters: 6 million at 120 digits and 14 million at
/// 130. Doubling every 20 bits all the way took RSA-130 31 minutes on 10
/// threads. At least 64 for each leading
/// coefficient. The rotations grow from the small settings below 210 bits
/// to the full ones from 290; at 90 and 100 digits, the full ones made
/// sieving a third and three fifths faster.
///
/// From [`KJ_WIDEN`] bits, about 125 digits, Kleinjung's primes `ℓ` and
/// leading coefficients grow faster than `m^(1/6)` and the fixed step of
/// 60 allow, as CADO-NFS's parameters do: the primes double every 25 bits,
/// to 124,000 at RSA-140 (CADO's 125,000), and the step doubles every 16
/// bits up to 480. With the old primes of 2^15 and leading coefficients
/// up to 6,000, RSA-140's polynomial had Murphy's E 2.10e-10 by CADO-NFS;
/// now it has 4.68e-10 (CADO-NFS's own, 4.58e-10) and sieves 1.3 to 2.2
/// times as fast. RSA-130's sieves 12% faster. On 10 threads the searches
/// take 38 and 6 minutes.
///
/// From there too, `j₀` reaches 2^20 rather than 2^16, in about 20 s on 10
/// threads. RSA-130's polynomial sieves 5 to 7% faster. RSA-140's Murphy's
/// E rose by a tenth, to 5.16e-10 (what CADO-NFS's root sieve gets from the
/// same polynomial), but it sieves no faster. At RSA-100 to 120 it found the
/// same polynomials, and 2^23 nothing more.
#[derive(Clone, Copy, Debug)]
struct Effort {
    nq: usize,
    rot1: i64,
    rot0: i64,
    /// The step between Kleinjung's leading coefficients.
    lead_step: u64,
    /// The least size of Kleinjung's primes, if more than `m^(1/6)` says.
    prime_floor: u64,
}

/// The size, in bits, from which Kleinjung's search widens past the
/// settings that suit 90 to 120 digits.
const KJ_WIDEN: u32 = 415;

impl Effort {
    fn for_bits(bits: u32) -> Self {
        // 0 up to 210 bits, 4 from 290.
        let step = (bits.saturating_sub(210) / 20).min(4);
        let b = f64::from(bits);
        let roots = 100_000.0 * exp2((b.min(364.0) - 266.0) / 20.0 + (b - 364.0).max(0.0) / 30.0);
        let wide = bits.saturating_sub(KJ_WIDEN);
        Self {
            nq: ((roots / KJ_LEADS as f64) as usize).max(64),
            rot1: 4 << step.saturating_sub(2),
            rot0: if wide == 0 { 1 << (12 + step) } else { 1 << 20 },
            lead_step: 60 << (wide / 16).min(3),
            prime_floor: if wide == 0 {
                0
            } else {
                exp2(15.0 + f64::from(wide) / 25.0) as u64
            },
        }
    }
}

/// Rotations with the best estimated size plus α kept for scoring exactly.
const ROOT_CELLS: usize = 32;

/// Cells of a row the root sieve works on at a time: 64 KB.
const ROOT_BLOCK: usize = 1 << 14;

/// The root sieve uses prime powers up to this, and estimates the size
/// along each row from this many points.
const ROOT_POWER: u64 = 4096;
const ROOT_GRID: usize = 16;

/// Of those, the best by size plus α kept for Murphy's E.
const ROOT_KEEP: usize = 4;

/// Steps tried in size optimization, from `2^SIZE_STEPS` down to 1, each
/// for at most `SIZE_PASSES` passes over the moves.
const SIZE_STEPS: u32 = 48;
const SIZE_PASSES: usize = 8;

/// Size optimization first scores translations `±2^(t/SCAN_DENSITY)` up to
/// `2^SCAN_BITS`, and searches from the best `SCAN_STARTS` of them.
const SCAN_BITS: u32 = 24;
const SCAN_DENSITY: u32 = 2;
const SCAN_STARTS: usize = 4;

/// Steps after which LLL gives up, far more than its few dimensions need.
const LLL_STEPS: usize = 1000;

/// α counts the factors of primes up to this, as in CADO-NFS.
const ALPHA_PRIMES: u64 = 2000;

/// Root optimization sieves with the powers of primes up to this.
const ROOT_PRIMES: u64 = 200;

/// Two polynomials for GNFS: `f` of degree `d` and `g(x) = l·x − m`, with
/// `F(m, l) = l^d·f(m/l)` equal to `N` or a small multiple of it, so that
/// `m/l` is a common root modulo `N`.
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
        let murphy = Murphy::new(&Params::for_bits(n.bits()));
        let effort = Effort::for_bits(n.bits());
        let sized = finalists(n, degree(n.bits()), effort, par)?;
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

/// Size-optimized candidates of degree `d` for root optimization: base-m
/// ones and Kleinjung's, the best `ROOT_FINALISTS` by size.
fn finalists<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    d: usize,
    effort: Effort,
    par: &impl Parallel,
) -> Option<Vec<Small>> {
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
    let p0 = kleinjung_prime(&candidates[0]).max(effort.prime_floor);
    let u_max = (KJ_SPREAD * p0 * p0) as i64;
    let found = map_dynamic(par, KJ_LEADS, |i| {
        kleinjung::candidates(
            n,
            d,
            effort.lead_step * (i as u64 + 1),
            p0,
            u_max,
            effort.nq,
        )
        .into_iter()
        .map(|(f, m, l)| Small::new(f, m, l))
        .collect::<Vec<_>>()
    });
    candidates.extend(found.into_iter().flatten());
    candidates.sort_by(|x, y| x.size.total_cmp(&y.size));
    candidates.truncate(SIZE_KEEP);
    // Size-optimize them; nearby m often optimize to the same one.
    let mut sized = map_dynamic(par, candidates.len(), |i| {
        let c = &candidates[i];
        let (f, m) = size_optimize(&c.f, c.m, c.l, true);
        Small::new(f, m, c.l)
    });
    sized.sort_by(|x, y| x.size.total_cmp(&y.size));
    sized.dedup_by(|x, y| x.f == y.f && x.m == y.m);
    sized.truncate(ROOT_FINALISTS);
    Some(sized)
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

/// Searches translations, and if `rotations` the polynomials
/// `c·f + r·g` with `deg r ≤ d − 2`, for the smallest size. Translation
/// leaves the number of roots modulo each prime, and so α, as it is.
///
/// A translation changes every coefficient, and with them the best
/// rotation, so stepping through one move at a time stalls far from the
/// best pair. Instead each translation tried is scored with the best
/// polynomial [`reduced`] finds for it: first a coarse scan, then steps
/// from `2^SIZE_STEPS` down to 1 from the best few, then [`descend`]
/// polishes the result. On 3,000 of CADO-NFS's raw candidates for
/// RSA-110, sizes averaged 52.0 bits, against 51.5 from CADO-NFS and 57.3
/// from moving one step at a time.
fn size_optimize(f: &[i128], m: u128, l: u128, rotations: bool) -> (Vec<i128>, u128) {
    if !rotations || f.len() < 4 {
        return descend(f, m, l, rotations);
    }
    // Score a coarse scan of translations, each with its best rotation,
    // then search from the best few.
    let mut scanned: Vec<(f64, Vec<i128>, u128)> = translation_scan()
        .filter_map(|k| {
            let (g, gm) = translate(f, m, l, k)?;
            let (g, s) = rotated(&g, gm, l)?;
            Some((s, g, gm))
        })
        .collect();
    scanned.sort_by(|x, y| x.0.total_cmp(&y.0));
    let best = scanned
        .into_iter()
        .take(SCAN_STARTS)
        .map(|(s, g, gm)| translation_search(g, gm, l, s))
        .min_by(|x, y| x.2.total_cmp(&y.2));
    match best {
        Some((g, gm, _)) => descend(&g, gm, l, true),
        None => descend(f, m, l, rotations),
    }
}

/// Translations scanned before searching: 0 and `±2^(t/SCAN_DENSITY)` up
/// to `2^SCAN_BITS`.
fn translation_scan() -> impl Iterator<Item = i128> {
    let steps = SCAN_BITS * SCAN_DENSITY;
    core::iter::once(0).chain((0..steps).flat_map(|t| {
        let k = (exp2(f64::from(t) / f64::from(SCAN_DENSITY)) + 0.5) as i128;
        [k, -k]
    }))
}

/// Steps from `2^SIZE_STEPS` down to 1 through the translations of `f`,
/// each scored with its best rotation.
fn translation_search(
    mut f: Vec<i128>,
    mut m: u128,
    l: u128,
    mut size: f64,
) -> (Vec<i128>, u128, f64) {
    for e in (0..=SIZE_STEPS).rev() {
        let step = 1i128 << e;
        for _ in 0..SIZE_PASSES {
            let mut improved = false;
            for k in [step, -step] {
                let Some((g, gm)) = translate(&f, m, l, k) else {
                    continue;
                };
                let Some((g, s)) = rotated(&g, gm, l) else {
                    continue;
                };
                if s < size - 1e-6 {
                    (f, m, size) = (g, gm, s);
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }
    (f, m, size)
}

/// `f` replaced by [`reduced`] at its skew, done twice since the skew moves
/// with it, and its size; or `None` on overflow.
fn rotated(f: &[i128], m: u128, l: u128) -> Option<(Vec<i128>, f64)> {
    let mut g = f.to_vec();
    for _ in 0..2 {
        g = reduced(&g, m, l, lognorm(&g).0)?;
    }
    let size = lognorm(&g).1;
    Some((g, size))
}

/// The smallest polynomial of degree `d` at skew `s` of the form
/// `c·f + r·(l·x − m)`, with `c ≠ 0` and `deg r ≤ d − 2`, or `None` on
/// overflow: the shortest vector of degree `d` after LLL reduction of the
/// lattice that `f` and the `xⁱ·g` span, with the coefficient of `xʲ`
/// weighted by `s^j`. Each still has the root `m/l` modulo `N`, and a
/// resultant with `g` of `c·N`. This is CADO-NFS's size optimization.
fn reduced(f: &[i128], m: u128, l: u128, s: f64) -> Option<Vec<i128>> {
    let d = f.len() - 1;
    let ls = log2(s);
    let scale: Vec<f64> = (0..=d).map(|j| exp2(ls * j as f64)).collect();
    let mut basis = vec![f.to_vec()];
    for i in 0..d - 1 {
        let mut v = vec![0i128; d + 1];
        v[i] = -i128::try_from(m).ok()?;
        v[i + 1] = i128::try_from(l).ok()?;
        basis.push(v);
    }
    lll(&mut basis, &scale)?;
    let norm = |v: &[i128]| -> f64 {
        v.iter()
            .zip(&scale)
            .map(|(&c, &x)| (c as f64 * x) * (c as f64 * x))
            .sum()
    };
    basis
        .into_iter()
        .filter(|v| v[d] != 0)
        .min_by(|x, y| norm(x).total_cmp(&norm(y)))
}

/// LLL-reduces the rows of `basis` for the norm with coordinate `j`
/// weighted by `scale[j]`, with δ = 0.99: exactly in integers, with the
/// Gram–Schmidt vectors in floating point, which is plenty for so few
/// dimensions. `None` on overflow.
fn lll(basis: &mut [Vec<i128>], scale: &[f64]) -> Option<()> {
    let n = basis.len();
    let dot = |x: &[f64], y: &[f64]| -> f64 { x.iter().zip(y).map(|(a, b)| a * b).sum() };
    // The Gram–Schmidt vectors' squared lengths, and the μ.
    let gram_schmidt = |basis: &[Vec<i128>]| -> (Vec<f64>, Vec<Vec<f64>>) {
        let b: Vec<Vec<f64>> = basis
            .iter()
            .map(|v| v.iter().zip(scale).map(|(&c, &x)| c as f64 * x).collect())
            .collect();
        let mut star: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut len = vec![0.0; n];
        let mut mu = vec![vec![0.0; n]; n];
        for i in 0..n {
            let mut v = b[i].clone();
            for j in 0..i {
                mu[i][j] = if len[j] > 0.0 {
                    dot(&b[i], &star[j]) / len[j]
                } else {
                    0.0
                };
                for (x, y) in v.iter_mut().zip(&star[j]) {
                    *x -= mu[i][j] * y;
                }
            }
            len[i] = dot(&v, &v);
            star.push(v);
        }
        (len, mu)
    };
    let mut k = 1;
    for _ in 0..LLL_STEPS {
        if k >= n {
            return Some(());
        }
        // Size-reduce row k by the rows before it.
        for j in (0..k).rev() {
            let (_, mu) = gram_schmidt(basis);
            let r = mu[k][j];
            if !r.is_finite() || r.abs() >= 1e30 {
                return None;
            }
            // Rounded to the nearest; the cast truncates toward 0.
            let r = (r + if r < 0.0 { -0.5 } else { 0.5 }) as i128;
            if r != 0 {
                for c in 0..basis[k].len() {
                    let t = basis[j][c].checked_mul(r)?;
                    basis[k][c] = basis[k][c].checked_sub(t)?;
                }
            }
        }
        let (len, mu) = gram_schmidt(basis);
        if len[k] >= (0.99 - mu[k][k - 1] * mu[k][k - 1]) * len[k - 1] {
            k += 1;
        } else {
            basis.swap(k, k - 1);
            k = (k - 1).max(1);
        }
    }
    Some(())
}

/// Searches translations and, if `rotations`, rotations by `xⁱ·g` for
/// `i ≤ d − 3` for the smallest size, one move at a time, by steps from
/// `2^SIZE_STEPS` down to 1 in each in turn.
fn descend(f: &[i128], m: u128, l: u128, rotations: bool) -> (Vec<i128>, u128) {
    let d = f.len() - 1;
    let moves = if rotations { d.saturating_sub(1) } else { 0 };
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

/// A progression of the root sieve: for a power `pk` of a small prime and
/// an `x` modulo it, `f + t·g ≡ 0` at `x`, which lowers α by `w`.
struct RootHit {
    pk: u32,
    w: f32,
    x: u64,
    t: u64,
}

/// Sieves rotations `f + (j₁·x + j₀)·g` for the smallest size plus α, and
/// returns the best few, each size-optimized by translation.
///
/// For a prime power `p^k` and an `x` with `g(x)` a unit, the rotation's
/// value at `x` is divisible by `p^k` exactly when
/// `j₀ + j₁·x ≡ −f(x)/g(x) (mod p^k)`: for each `j₁`, an arithmetic
/// progression of `j₀`. Each lowers α by `log₂ p/(p^(k−1)·(p + 1))`, and
/// summed over `k` these are α's average valuations. The
/// rotations also grow the size, so each cell is scored by the size after
/// translation, interpolated along its row from `ROOT_GRID` points, less
/// its gain. Rows are sieved `ROOT_BLOCK` cells at a time, so that wide
/// ones stay in cache.
fn root_optimize(f: &[i128], m: u128, l: u128, effort: Effort) -> Vec<(Vec<i128>, u128)> {
    let d = f.len() - 1;
    let rot1 = if d >= 4 { effort.rot1 } else { 0 };
    let rot0 = effort.rot0;
    let width = 2 * rot0 as usize;
    let mut hits: Vec<RootHit> = Vec::new();
    for p in Primes::new(2, ROOT_PRIMES) {
        let (mut pk, mut below) = (p, 1u64);
        while pk <= ROOT_POWER {
            let w = (log2(p as f64) / (below as f64 * (p + 1) as f64)) as f32;
            let fp = reduce(f, pk);
            let (mp, lp) = ((m % u128::from(pk)) as u64, (l % u128::from(pk)) as u64);
            hits.extend((0..pk).filter_map(|x| {
                let gx = (lp * x % pk + pk - mp) % pk;
                (gx % p != 0).then(|| RootHit {
                    pk: pk as u32,
                    w,
                    x,
                    t: (pk - eval(&fp, x, pk)) % pk * inv(gx, pk) % pk,
                })
            }));
            below = pk;
            pk *= p;
        }
    }
    // The best cells of all rows seen, as (score, j₁, index of j₀), and
    // where each progression next hits the row.
    let mut best: Vec<(f32, i64, usize)> = Vec::new();
    let mut next = vec![0u32; hits.len()];
    let mut block = vec![0f32; ROOT_BLOCK];
    let mut order: Vec<u32> = Vec::with_capacity(ROOT_BLOCK);
    let step = width / ROOT_GRID;
    for j1 in -rot1..=rot1 {
        let Some(rotated) = rotate(f, m, l, 1, i128::from(j1)) else {
            continue;
        };
        // The size after translation, which the best rotations need: each
        // grid point starts from the translation that suited the last.
        let mut shift = 0i128;
        let sizes: Vec<f32> = (0..=ROOT_GRID)
            .map(|k| {
                let j0 = (k * step) as i128 - i128::from(rot0);
                let Some((g, gm)) =
                    rotate(&rotated, m, l, 0, j0).and_then(|g| translate(&g, m, l, shift))
                else {
                    return f32::MAX;
                };
                let (g, gm2) = descend(&g, gm, l, false);
                shift += (gm as i128 - gm2 as i128) / l as i128;
                lognorm(&g).1 as f32
            })
            .collect();
        for (n, h) in next.iter_mut().zip(&hits) {
            // j₀ ≡ t − j₁·x, at index j₀ + rot0.
            let pk = i64::from(h.pk);
            let r = (h.t as i64 - j1 * h.x as i64).rem_euclid(pk);
            *n = ((r + rot0) % pk) as u32;
        }
        for start in (0..width).step_by(ROOT_BLOCK) {
            // The size, interpolated between grid points, less the gains.
            let cells = &mut block[..ROOT_BLOCK.min(width - start)];
            for (i, cell) in (start..).zip(cells.iter_mut()) {
                let (k, frac) = (i / step, (i % step) as f32 / step as f32);
                *cell = sizes[k] + (sizes[(k + 1).min(ROOT_GRID)] - sizes[k]) * frac;
            }
            let end = (start + cells.len()) as u32;
            for (n, h) in next.iter_mut().zip(&hits) {
                let mut i = *n;
                while i < end {
                    cells[i as usize - start] -= h.w;
                    i += h.pk;
                }
                *n = i;
            }
            order.clear();
            order.extend(0..cells.len() as u32);
            let keep = ROOT_CELLS.min(cells.len());
            order.select_nth_unstable_by(keep - 1, |a, b| {
                cells[*a as usize].total_cmp(&cells[*b as usize])
            });
            best.extend(
                order[..keep]
                    .iter()
                    .map(|&i| (cells[i as usize], j1, start + i as usize)),
            );
            if best.len() > ROOT_CELLS {
                best.select_nth_unstable_by(ROOT_CELLS - 1, |a, b| a.0.total_cmp(&b.0));
                best.truncate(ROOT_CELLS);
            }
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

/// Murphy's α in bits: the sum over primes `p` below `ALPHA_PRIMES` of
/// `(1/(p − 1) − v_p)·log₂ p`, where a random integer has `1/(p − 1)`
/// factors of `p` on average and a value `F(a, b)` of `f` at coprime
/// `a`, `b` has `v_p`, from `valuation`. Negative is better.
pub fn alpha<C: Coeff>(f: &[C]) -> f64 {
    Primes::new(2, ALPHA_PRIMES)
        .map(|p| (1.0 / (p - 1) as f64 - valuation(f, p)) * log2(p as f64))
        .sum()
}

/// The average number of factors of `p` in `F(a, b)` over coprime `a`,
/// `b`: `b` is a unit with chance `p/(p + 1)`, and then this is the average
/// over `x = a/b` in `ℤ_p` of `f(x)`'s; otherwise it is the average over
/// `y = b/a` in `pℤ_p` of the reversed polynomial's. Both are worked out
/// modulo the largest power of `p` that keeps products in a `u128`, which
/// leaves out only valuations past it.
pub fn valuation<C: Coeff>(f: &[C], p: u64) -> f64 {
    let mut pk = p;
    while pk.checked_mul(p).is_some_and(|x| x < 1 << 40) {
        pk *= p;
    }
    let fp: Vec<u64> = f.iter().map(|c| c.mod_p(pk)).collect();
    // The reversed polynomial at p·y: coefficient i of x^d·f(1/x), times p^i.
    let mut scale = 1u64;
    let rev: Vec<u64> = fp
        .iter()
        .rev()
        .map(|&c| {
            let v = mul_mod(c, scale, pk);
            scale = mul_mod(scale, p, pk);
            v
        })
        .collect();
    let pf = p as f64;
    (pf * affine_valuation(fp, p, pk) + affine_valuation(rev, p, pk)) / (pf + 1.0)
}

/// The average number of factors of `p` in `f(x)` over `x` in `ℤ_p`, for
/// `f` modulo `pk`: for each root `r` modulo `p`, a `1/p` chance of `x`
/// near it, and then `f(r + p·y) = p·g(y)` with `y` again uniform.
fn affine_valuation(mut f: Vec<u64>, p: u64, mut pk: u64) -> f64 {
    // Factors of p common to every coefficient count for every x.
    let mut v = 0.0;
    while pk > 1 && f.iter().all(|&c| c % p == 0) {
        if f.iter().all(|&c| c == 0) {
            return v + (pk.ilog(p)) as f64;
        }
        f.iter_mut().for_each(|c| *c /= p);
        pk /= p;
        v += 1.0;
    }
    if pk <= 1 {
        return v;
    }
    let reduced: Vec<u64> = f.iter().map(|&c| c % p).collect();
    for r in 0..p {
        if super::modp::eval(&reduced, r, p) != 0 {
            continue;
        }
        // g(y) = f(r + p·y)/p, modulo pk/p: shift by r, scale by p^i.
        let mut g = f.clone();
        for i in 0..g.len() {
            for j in (i..g.len() - 1).rev() {
                g[j] = (g[j] + mul_mod(r, g[j + 1], pk)) % pk;
            }
        }
        let mut scale = 1u64;
        for c in g.iter_mut() {
            *c = mul_mod(*c, scale, pk) / p;
            scale = mul_mod(scale, p, pk);
        }
        v += (1.0 + affine_valuation(g, p, pk / p)) / p as f64;
    }
    v
}

fn mul_mod(a: u64, b: u64, k: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(k)) as u64
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
