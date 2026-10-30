//! The factoring pipeline: trial division, then Pollard's rho, p−1 and p+1
//! on whatever composite remains, with BPSW deciding when a piece is prime.

use core::fmt;

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::arith::{Limb, ParseUintError, Uint};
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

/// Polynomials `x² + c` tried by rho before giving up on a composite.
const RHO_ATTEMPTS: u64 = 3;

/// Step budget for each rho attempt. About 4 million steps reliably finds
/// factors up to roughly 12 digits.
const RHO_STEPS: u64 = 1 << 22;

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

/// Factors `n`, returning its factors in ascending order with multiplicity.
///
/// Returns an empty vector for `n < 2`. Every factor is prime unless rho ran
/// out of steps, in which case the leftover composite is returned with
/// `is_prime: false`.
pub fn factor<const LIMBS: usize>(n: &Uint<LIMBS>) -> Vec<Factor<Uint<LIMBS>>> {
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
    while let Some(m) = pending.pop() {
        if m == Uint::ONE {
            continue;
        }
        if m < trial_bound || is_prime(&m) {
            factors.push(prime(m));
            continue;
        }
        // Cheapest first: a short rho run for small factors, then p−1 and
        // p+1 for factors of any size whose p − 1 or p + 1 is smooth, then
        // the full rho budget.
        let found = pollard_rho(&m, 1, QUICK_RHO_STEPS)
            .or_else(|| pollard_pm1(&m, PM1_B1, PM1_B2))
            .or_else(|| {
                PP1_SEEDS
                    .iter()
                    .find_map(|&seed| williams_pp1(&m, seed, PP1_B1, PP1_B2))
            })
            .or_else(|| (1..=RHO_ATTEMPTS).find_map(|c| pollard_rho(&m, c, RHO_STEPS)));
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

/// Factors a decimal string, picking the narrowest integer width that fits.
///
/// Accepts numbers up to `16 * LIMB_BITS` bits: 1024 bits natively, 512 on
/// wasm32.
pub fn factor_str(s: &str) -> Result<Vec<Factor<String>>, ParseUintError> {
    fn run<const LIMBS: usize>(s: &str) -> Result<Vec<Factor<String>>, ParseUintError> {
        Ok(factor(&s.parse::<Uint<LIMBS>>()?)
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
        Some(1) => run::<1>(s),
        Some(2) => run::<2>(s),
        Some(4) => run::<4>(s),
        Some(8) => run::<8>(s),
        _ => run::<16>(s),
    }
}
