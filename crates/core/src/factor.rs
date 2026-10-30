//! The factoring pipeline: trial division, then Pollard's rho, p−1, p+1 and
//! ECM on whatever composite remains, with BPSW deciding when a piece is
//! prime.

use core::fmt;
use core::ops::ControlFlow;

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::arith::{Limb, ParseUintError, Uint};
use crate::ecm::ecm_curve;
use crate::pm1::pollard_pm1;
use crate::pp1::williams_pp1;
use crate::prime::is_prime;
use crate::rho::pollard_rho;

/// Trial division covers every divisor below this.
const TRIAL_LIMIT: u64 = 1 << 12;

/// Step budget for the first, quick rho attempt, which catches factors up to
/// about 8 digits before anything more expensive runs.
const QUICK_RHO_STEPS: u64 = 1 << 16;

/// Stage 1 and stage 2 bounds for p−1.
const PM1_B1: u64 = 50_000;
const PM1_B2: u64 = 5_000_000;

/// Bounds and seeds for p+1. The seeds' A² − 4 values (5 and 12 = 4·3) have
/// different square-free parts, so each prime gets two independent chances
/// of landing in its group of order p + 1.
const PP1_B1: u64 = 50_000;
const PP1_B2: u64 = 5_000_000;
const PP1_SEEDS: [u64; 2] = [3, 4];

/// ECM levels as (digits, B1, curves), with B2 = 100·B1. Each level runs
/// about as many curves as it takes on average to find a prime of that many
/// digits, so it finds one with probability around 1 − 1/e or better, and
/// smaller primes almost surely. The counts come from Dickman's rho
/// applied to the group order divided by 23.4, the average contribution of
/// Suyama's torsion; at 15 and 20 digits they were checked against measured
/// success rates. Each level costs about four times the one before.
const ECM_LEVELS: [(u32, u64, u64); 6] = [
    (15, 2_000, 30),
    (20, 11_000, 100),
    (25, 50_000, 325),
    (30, 250_000, 760),
    (35, 1_000_000, 1_900),
    (40, 3_000_000, 5_400),
];

/// The first ECM curve's sigma. Smaller values give degenerate curves.
const FIRST_SIGMA: u64 = 6;

/// Polynomials `x² + c` tried by rho before giving up on a small composite.
const RHO_ATTEMPTS: u64 = 3;

/// Step budget for each rho attempt. About 4 million steps reliably finds
/// factors up to roughly 12 digits, so composites up to twice that many
/// bits (about 80) are always split.
const RHO_STEPS: u64 = 1 << 22;
const RHO_MAX_BITS: u32 = 80;

/// Widths, in limbs, that [`factor_str`] can pick from.
const WIDTHS: [usize; 5] = [1, 2, 4, 8, 16];

/// One factor in a factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Factor<T> {
    pub value: T,
    /// `false` for a composite that the available algorithms couldn't split.
    pub is_prime: bool,
}

/// Prints the value, wrapped in parentheses if it is composite.
impl<T: fmt::Display> fmt::Display for Factor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_prime {
            write!(f, "{}", self.value)
        } else {
            write!(f, "({})", self.value)
        }
    }
}

/// How hard [`factor_with`] tries, and how it shares the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Run ECM up to the level for primes of this many digits (15, 20, …,
    /// 40). Each level takes about four times as long as the one before. 0
    /// skips ECM.
    pub ecm_digits: u32,
    /// ECM curves are independent, so `workers` runs of the pipeline, on
    /// separate threads or Web Workers, can split them: the run with
    /// `worker = i` takes curves `i`, `i + workers`, `i + 2·workers`, … of
    /// each level. Each run also repeats the cheap stages before ECM, and
    /// whichever finishes first with every factor prime has the answer.
    pub worker: u32,
    pub workers: u32,
}

impl Default for Config {
    /// ECM up to 20 digits, in a single run. On a composite with no factor
    /// that small, ECM gives up after about 3 seconds at 100 digits and 12
    /// seconds at 300 digits, natively. 25 digits takes about four times as
    /// long.
    fn default() -> Self {
        Self {
            ecm_digits: 20,
            worker: 0,
            workers: 1,
        }
    }
}

/// What the pipeline is working on, reported to [`factor_with`]'s
/// callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// Starting `method` ("rho", "p-1" or "p+1") on a composite of `bits`
    /// bits.
    Method { method: &'static str, bits: u32 },
    /// ECM on a composite of `bits` bits: `done` of this run's `total`
    /// curves at the level for `digits`-digit primes.
    Ecm {
        bits: u32,
        digits: u32,
        done: u64,
        total: u64,
    },
}

/// Factors `n` with the default [`Config`], returning its factors in
/// ascending order with multiplicity.
///
/// Returns an empty vector for `n < 2`. Every factor is prime unless the
/// pipeline ran out of methods, in which case the leftover composite is
/// returned with `is_prime: false`.
pub fn factor<const LIMBS: usize>(n: &Uint<LIMBS>) -> Vec<Factor<Uint<LIMBS>>> {
    factor_with(n, &Config::default(), |_| ControlFlow::Continue(()))
}

/// Factors `n` like [`factor`], with the given [`Config`], calling
/// `progress` before each method and after each ECM curve.
///
/// If `progress` returns `Break`, the pipeline stops within one method or
/// curve. Pieces not yet split are still tested for primality, and the
/// composite ones come back with `is_prime: false`.
pub fn factor_with<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    config: &Config,
    mut progress: impl FnMut(Progress) -> ControlFlow<()>,
) -> Vec<Factor<Uint<LIMBS>>> {
    let mut factors = Vec::new();
    let mut m = *n;
    if m < Uint::from_u64(2) {
        return factors;
    }
    let prime = |value| Factor {
        value,
        is_prime: true,
    };

    // Trial division by 2 and odd numbers below TRIAL_LIMIT. Dividing by
    // odd composites too is wasteful but harmless, and keeps this simple.
    let mut d = 2;
    while d < TRIAL_LIMIT {
        if m < Uint::from_u64(d * d) {
            // m has no factor below d, so it's 1 or prime.
            if m != Uint::ONE {
                factors.push(prime(m));
            }
            return factors;
        }
        loop {
            let (q, r) = m.div_rem_limb(d as Limb);
            if r != 0 {
                break;
            }
            factors.push(prime(Uint::from_u64(d)));
            m = q;
        }
        d += if d == 2 { 1 } else { 2 };
    }

    // Everything left is free of factors below TRIAL_LIMIT, so any piece
    // smaller than TRIAL_LIMIT² is prime.
    let trial_bound = Uint::from_u64(TRIAL_LIMIT * TRIAL_LIMIT);
    let mut pending = vec![m];
    let mut stopped = false;
    while let Some(m) = pending.pop() {
        if m == Uint::ONE {
            continue;
        }
        if m < trial_bound || is_prime(&m) {
            factors.push(prime(m));
            continue;
        }
        let found = if stopped {
            None
        } else {
            split(&m, config, &mut progress).unwrap_or_else(|()| {
                stopped = true;
                None
            })
        };
        match found {
            Some(g) => {
                pending.push(g);
                pending.push(m.div_rem(&g).0);
            }
            None => factors.push(Factor {
                value: m,
                is_prime: false,
            }),
        }
    }

    factors.sort_by_key(|f| f.value);
    factors
}

/// Looks for a factor of `m`, a composite with no factors below the
/// trial-division limit. `Err` means `progress` asked to stop.
fn split<const LIMBS: usize>(
    m: &Uint<LIMBS>,
    config: &Config,
    progress: &mut impl FnMut(Progress) -> ControlFlow<()>,
) -> Result<Option<Uint<LIMBS>>, ()> {
    let bits = m.bits();
    let mut report = |event| match progress(event) {
        ControlFlow::Continue(()) => Ok(()),
        ControlFlow::Break(()) => Err(()),
    };
    let method = |method| Progress::Method { method, bits };

    // Cheapest first: a short rho run for small factors, then p−1 and p+1
    // for factors of any size whose p − 1 or p + 1 is smooth, then ECM, and
    // finally the full rho budget, which is certain to split a small
    // composite.
    report(method("rho"))?;
    if let Some(g) = pollard_rho(m, 1, QUICK_RHO_STEPS) {
        return Ok(Some(g));
    }
    report(method("p-1"))?;
    if let Some(g) = pollard_pm1(m, PM1_B1, PM1_B2) {
        return Ok(Some(g));
    }
    report(method("p+1"))?;
    for seed in PP1_SEEDS {
        if let Some(g) = williams_pp1(m, seed, PP1_B1, PP1_B2) {
            return Ok(Some(g));
        }
    }

    // ECM, one level at a time. Levels for primes larger than √m are
    // skipped, since the smallest prime factor is at most that.
    let workers = u64::from(config.workers.max(1));
    let worker = u64::from(config.worker) % workers;
    let mut sigma = FIRST_SIGMA;
    let mut covered = 0;
    for (digits, b1, curves) in ECM_LEVELS {
        // A d-digit prime has more than 3.32·(d − 1) bits, and the smallest
        // prime factor has at most bits(m)/2.
        if digits > config.ecm_digits || covered * 332 / 100 >= bits.div_ceil(2) {
            break;
        }
        let total = curves.saturating_sub(worker).div_ceil(workers);
        let ecm = |done| Progress::Ecm {
            bits,
            digits,
            done,
            total,
        };
        report(ecm(0))?;
        let mine = (worker..curves).step_by(workers as usize);
        for (done, curve) in (1..).zip(mine) {
            if let Some(g) = ecm_curve(m, sigma + curve, b1, 100 * b1) {
                return Ok(Some(g));
            }
            report(ecm(done))?;
        }
        sigma += curves;
        covered = digits;
    }

    if bits <= RHO_MAX_BITS {
        report(method("rho"))?;
        for c in 1..=RHO_ATTEMPTS {
            if let Some(g) = pollard_rho(m, c, RHO_STEPS) {
                return Ok(Some(g));
            }
        }
    }
    Ok(None)
}

/// Factors a decimal string with the default [`Config`], picking the
/// narrowest integer width that fits.
///
/// Accepts numbers up to `16 * LIMB_BITS` bits: 1024 bits natively, 512 on
/// wasm32.
pub fn factor_str(s: &str) -> Result<Vec<Factor<String>>, ParseUintError> {
    factor_str_with(s, &Config::default(), |_| ControlFlow::Continue(()))
}

/// Factors a decimal string like [`factor_str`], with the given [`Config`]
/// and progress callback, as in [`factor_with`].
pub fn factor_str_with(
    s: &str,
    config: &Config,
    progress: impl FnMut(Progress) -> ControlFlow<()>,
) -> Result<Vec<Factor<String>>, ParseUintError> {
    fn run<const LIMBS: usize>(
        s: &str,
        config: &Config,
        progress: impl FnMut(Progress) -> ControlFlow<()>,
    ) -> Result<Vec<Factor<String>>, ParseUintError> {
        Ok(factor_with(&s.parse::<Uint<LIMBS>>()?, config, progress)
            .into_iter()
            .map(|f| Factor {
                value: f.value.to_string(),
                is_prime: f.is_prime,
            })
            .collect())
    }

    // Each width is a separate copy of the whole pipeline, so keep the list
    // short.
    let bits = s.parse::<Uint<16>>()?.bits();
    match WIDTHS
        .iter()
        .find(|&&w| bits <= w as u32 * crate::arith::LIMB_BITS)
    {
        Some(1) => run::<1>(s, config, progress),
        Some(2) => run::<2>(s, config, progress),
        Some(4) => run::<4>(s, config, progress),
        Some(8) => run::<8>(s, config, progress),
        _ => run::<16>(s, config, progress),
    }
}
