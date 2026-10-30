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
//! So far this module has the groundwork: square roots mod `p`
//! ([`sqrt_mod`]), the choice of multiplier `k` ([`multiplier`]), the factor
//! base ([`FactorBase`]) and the parameters for each size of `N`
//! ([`Params`]).

mod base;
mod modp;
mod params;

pub use base::{FactorBase, Prime, multiplier};
pub use modp::{pow_mod, sqrt_mod};
pub use params::{BLOCK_SIZE, Params};

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
