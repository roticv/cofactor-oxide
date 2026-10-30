//! Sieving for relations.

use alloc::vec;
use alloc::vec::Vec;

use super::base::{FactorBase, rem};
use super::params::BLOCK_SIZE;
use crate::arith::{Limb, Uint};

/// A congruence `t² ≡ ±∏ p (mod N)` with every `p` in the factor base.
#[derive(Clone, Debug)]
pub struct Relation<const LIMBS: usize> {
    pub t: Uint<LIMBS>,
    /// Whether `t² − kN` is negative.
    pub negative: bool,
    /// Indices into the factor base, one per prime factor of `|t² − kN|`
    /// counted with multiplicity.
    pub factors: Vec<u32>,
}

/// Marks a prime that isn't sieved, only tried in trial division: 2, and
/// the primes dividing `k`, which have a single root.
const UNSIEVED: u32 = u32::MAX;

/// Sieves `Q(x) = (x + m)² − kN` with `m = ⌈√kN⌉`, one block of `x` at a
/// time.
///
/// The sieve adds `log₂ p` at every `x` where `p` divides `Q(x)`, which for
/// a prime with roots `r₁`, `r₂` of `Q` mod `p` is `x ≡ r₁` or `r₂`. Where
/// the sum comes close to `log₂ |Q(x)|`, `Q(x)` is likely smooth, and trial
/// division decides.
pub struct Sieve<'a, const LIMBS: usize> {
    fb: &'a FactorBase<LIMBS>,
    m: Uint<LIMBS>,
    /// The roots of `Q` mod each factor-base prime, or [`UNSIEVED`].
    roots: Vec<(u32, u32)>,
    /// How far below `log₂ |Q(x)|` a sum may be and still be tried.
    slack: u32,
    block: Vec<u8>,
}

impl<'a, const LIMBS: usize> Sieve<'a, LIMBS> {
    /// `kN` must not be a square.
    pub fn new(fb: &'a FactorBase<LIMBS>) -> Self {
        let m = fb.kn.isqrt().wrapping_add(&Uint::ONE);
        let k = fb.multiplier;
        let roots = fb
            .primes
            .iter()
            .map(|prime| {
                let p = prime.p;
                if p == 2 || k.is_multiple_of(p) {
                    return (UNSIEVED, UNSIEVED);
                }
                // x + m ≡ ±sqrt(kN).
                let (s, p) = (u64::from(prime.sqrt), u64::from(p));
                let mm = rem(&m, p);
                (((s + p - mm) % p) as u32, ((2 * p - s - mm) % p) as u32)
            })
            .collect();
        let largest = fb.primes.last().map_or(1, |p| u32::from(p.log));
        Self {
            fb,
            m,
            roots,
            slack: 2 * largest,
            block: vec![0; BLOCK_SIZE],
        }
    }

    /// `|x + m|`.
    pub fn t(&self, x: i64) -> Uint<LIMBS> {
        let d = Uint::from_u64(x.unsigned_abs());
        if x >= 0 {
            self.m.wrapping_add(&d)
        } else if self.m >= d {
            self.m.wrapping_sub(&d)
        } else {
            d.wrapping_sub(&self.m)
        }
    }

    /// `Q(x)` as its sign and magnitude.
    ///
    /// # Panics
    ///
    /// Panics if `|Q(x)|` doesn't fit in the integer width.
    pub fn value(&self, x: i64) -> (bool, Uint<LIMBS>) {
        let (lo, hi) = self.t(x).widening_square();
        let kn = &self.fb.kn;
        if hi.is_zero() && lo < *kn {
            return (true, kn.wrapping_sub(&lo));
        }
        let (q, borrow) = lo.overflowing_sub(kn);
        assert!(
            hi.checked_sub(&Uint::from_u64(borrow.into()))
                .is_some_and(|hi| hi.is_zero()),
            "Q(x) too large"
        );
        (false, q)
    }

    /// Sieves `x` in `[x0, x0 + BLOCK_SIZE)` and returns the relations found.
    pub fn block(&mut self, x0: i64) -> Vec<Relation<LIMBS>> {
        self.block.fill(0);
        for (prime, &(r1, r2)) in self.fb.primes.iter().zip(&self.roots) {
            if r1 == UNSIEVED {
                continue;
            }
            let p = i64::from(prime.p);
            for r in [r1, r2] {
                let mut j = (i64::from(r) - x0).rem_euclid(p) as usize;
                while j < BLOCK_SIZE {
                    self.block[j] = self.block[j].saturating_add(prime.log);
                    j += p as usize;
                }
            }
        }

        // |Q| grows with |x|, so the end of the block farthest from 0 gives
        // the largest log₂ |Q(x)| in it.
        let far = if x0 >= 0 {
            x0 + BLOCK_SIZE as i64 - 1
        } else {
            x0
        };
        let threshold = self.value(far).1.bits().saturating_sub(self.slack);
        let threshold = threshold.min(u32::from(u8::MAX)) as u8;
        let mut relations = Vec::new();
        for j in 0..BLOCK_SIZE {
            if self.block[j] >= threshold {
                let x = x0 + j as i64;
                if let Some(relation) = self.trial_divide(x) {
                    relations.push(relation);
                }
            }
        }
        relations
    }

    /// Factors `Q(x)` over the factor base, if it is smooth.
    fn trial_divide(&self, x: i64) -> Option<Relation<LIMBS>> {
        let (negative, mut q) = self.value(x);
        if q.is_zero() {
            return None;
        }
        let mut factors = Vec::new();
        for (i, (prime, &(r1, r2))) in self.fb.primes.iter().zip(&self.roots).enumerate() {
            let p = prime.p;
            if r1 != UNSIEVED {
                // A sieved prime divides Q(x) exactly at its roots, which
                // is cheaper to check than a remainder.
                let xp = x.rem_euclid(i64::from(p)) as u32;
                if xp != r1 && xp != r2 {
                    continue;
                }
            }
            loop {
                let (quot, r) = q.div_rem_limb(p as Limb);
                if r != 0 {
                    break;
                }
                q = quot;
                factors.push(i as u32);
            }
        }
        (q == Uint::ONE).then(|| Relation {
            t: self.t(x),
            negative,
            factors,
        })
    }
}
