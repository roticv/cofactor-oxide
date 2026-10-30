//! The multiplier and the factor base.

use alloc::vec::Vec;

use super::log2;
use super::modp::{pow_mod, sqrt_mod};
use crate::arith::{Limb, Uint};
use crate::primes::Primes;

/// Square-free multipliers to choose from.
const MULTIPLIERS: [u32; 44] = [
    1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38,
    39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70,
];

/// Primes up to this bound score each multiplier.
const SCORE_PRIMES: u64 = 2000;

/// `n mod d`, for `d < 2^32`.
// Limb is u32 or u64, so the cast is needed on one target and redundant on
// the other.
#[allow(clippy::unnecessary_cast)]
pub(super) fn rem<const LIMBS: usize>(n: &Uint<LIMBS>, d: u64) -> u64 {
    n.div_rem_limb(d as Limb).1 as u64
}

/// `kN`, or `None` if it overflows.
fn times<const LIMBS: usize>(n: &Uint<LIMBS>, k: u32) -> Option<Uint<LIMBS>> {
    match n.overflowing_mul_limb(k as Limb) {
        (kn, 0) => Some(kn),
        _ => None,
    }
}

/// Chooses the multiplier `k` for sieving `kN`, by the Knuth–Schroeppel
/// function.
///
/// Sieving `kN` instead of `N` changes which primes are in the factor base.
/// A good `k` makes many small primes quadratic residues, which makes
/// `Q(x)` more often smooth, at the cost of `Q(x)` being `√k` times larger.
/// Each `k` is scored by the expected contribution of each small prime `p`
/// to `log₂ Q(x)`:
///
/// - `p = 2`: 2 if `kN ≡ 1 (mod 8)`, 1 if `kN ≡ 5 (mod 8)`, ½ otherwise;
/// - `p` dividing `k`: `1/p`;
/// - `kN` a square mod `p`: `2/(p − 1)`, since `Q(x)` then has two roots
///   mod `p`;
///
/// each times `log₂ p`, minus `½·log₂ k`. Returns the `k` with the highest
/// score among those for which `kN` fits in the integer width.
///
/// `n` must be odd and have no prime factors below 2000.
pub fn multiplier<const LIMBS: usize>(n: &Uint<LIMBS>) -> u32 {
    // Residues of n mod each scoring prime, shared by all multipliers.
    let primes: Vec<(u64, u64)> = Primes::new(3, SCORE_PRIMES)
        .map(|p| (p, rem(n, p)))
        .collect();
    let n8 = rem(n, 8);
    let mut best = (f64::NEG_INFINITY, 1);
    for k in MULTIPLIERS {
        if times(n, k).is_none() {
            continue;
        }
        let k = u64::from(k);
        let mut score = -0.5 * log2(k as f64);
        score += match (k * n8) % 8 {
            1 => 2.0,
            5 => 1.0,
            _ => 0.5,
        };
        for &(p, r) in &primes {
            let weight = if k % p == 0 {
                1.0 / p as f64
            } else if pow_mod(k * r, (p - 1) / 2, p) == 1 {
                2.0 / (p - 1) as f64
            } else {
                continue;
            };
            score += weight * log2(p as f64);
        }
        if score > best.0 {
            best = (score, k as u32);
        }
    }
    best.1
}

/// A factor-base prime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prime {
    pub p: u32,
    /// The smaller square root of `kN` mod `p`.
    pub sqrt: u32,
    /// `log₂ p`, rounded, as the sieve adds it up.
    pub log: u8,
}

/// The primes the sieve works with: 2, the primes dividing `k`, and the odd
/// primes `p` for which `kN` is a square mod `p`.
#[derive(Clone, Debug)]
pub struct FactorBase<const LIMBS: usize> {
    pub multiplier: u32,
    /// `kN`, the number being sieved.
    pub kn: Uint<LIMBS>,
    /// In increasing order.
    pub primes: Vec<Prime>,
}

impl<const LIMBS: usize> FactorBase<LIMBS> {
    /// The factor base of `size` primes for `kN`.
    ///
    /// Returns `Err(p)` if a prime `p` found along the way divides `n`.
    ///
    /// # Panics
    ///
    /// Panics if `kN` overflows the integer width, or if `size` is too large
    /// for primes below 2^32.
    pub fn new(n: &Uint<LIMBS>, multiplier: u32, size: usize) -> Result<Self, Uint<LIMBS>> {
        let kn = times(n, multiplier).expect("kN overflows");
        let k = u64::from(multiplier);
        let mut primes = Vec::with_capacity(size);
        for p in Primes::new(2, u64::from(u32::MAX)) {
            if primes.len() == size {
                break;
            }
            let r = rem(&kn, p);
            if r == 0 && k % p != 0 {
                return Err(Uint::from_u64(p));
            }
            if let Some(sqrt) = sqrt_mod(r, p) {
                primes.push(Prime {
                    p: p as u32,
                    sqrt: sqrt as u32,
                    log: (log2(p as f64) + 0.5) as u8,
                });
            }
        }
        assert_eq!(primes.len(), size, "factor base too large");
        Ok(Self {
            multiplier,
            kn,
            primes,
        })
    }
}
