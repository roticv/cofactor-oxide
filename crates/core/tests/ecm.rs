//! Tests for ECM stage 1.
//!
//! The thresholds were computed in Python with affine arithmetic on
//! `B·y² = x³ + Ax² + x` (independent of the x-only formulas being tested):
//! count the curve's points, find the exact order of Suyama's starting
//! point, and take its largest prime power. Stage 1 multiplies by the
//! largest power of each prime up to B1, so that prime power is the smallest
//! B1 that finds p.

use cofox_core::Uint;
use cofox_core::ecm::{ecm, ecm_curve};

/// The largest test number is a 50-bit prime times 2^89 − 1, about 139
/// bits, so use 6 limbs: 192 bits even with 32-bit limbs.
type U = Uint<6>;

/// 2^89 − 1, a prime cofactor far too large for ECM to find by accident.
fn m89() -> U {
    (U::ONE << 89).wrapping_sub(&U::ONE)
}

fn times_m89(p: u64) -> U {
    let (lo, hi) = U::from_u64(p).widening_mul(&m89());
    assert!(hi.is_zero());
    lo
}

fn product(a: u64, b: u64) -> U {
    U::from_u64(a).widening_mul(&U::from_u64(b)).0
}

/// (p, sigma, smallest B1 that finds p, order of the starting point mod p)
const THRESHOLDS: [(u64, u64, u64, &str); 6] = [
    (100003, 110, 19, "2³·3·5·11·19"),
    (100019, 159, 7, "2²·3·7"),
    (100003, 87, 17, "2·3·5·7·17"),
    // The 2³ needs B1 ≥ 8, although the largest prime is 7.
    (131071, 15, 8, "2³·3·5·7"),
    // A prime power: needs B1 ≥ 3⁵ = 243.
    (100019, 158, 243, "3⁵·…"),
    // The group has order 2²·3⁵·103, but the point's order is only
    // 3⁴·103, so B1 = 103 is enough. Group order alone would say 243.
    (100003, 18, 103, "3⁴·103"),
];

#[test]
fn stage_1_thresholds() {
    for (p, sigma, threshold, order) in THRESHOLDS {
        let n = times_m89(p);
        assert_eq!(
            ecm_curve(&n, sigma, threshold),
            Some(U::from_u64(p)),
            "p = {p}, sigma = {sigma}, order {order}, B1 = {threshold}"
        );
        assert_eq!(
            ecm_curve(&n, sigma, threshold - 1),
            None,
            "p = {p}, sigma = {sigma}, order {order}, B1 = {}",
            threshold - 1
        );
    }
}

#[test]
fn replay_separates_factors() {
    // With sigma = 8, B1 = 31 finds 131071 and B1 = 67 finds 100019. At
    // B1 = 100 both happen inside the first chunk, so its GCD is n and the
    // replay finds 131071 first.
    let n = product(131071, 100019);
    assert_eq!(ecm_curve(&n, 8, 50), Some(U::from_u64(131071)));
    assert_eq!(ecm_curve(&n, 8, 100), Some(U::from_u64(131071)));
}

#[test]
fn failed_curve_setup_is_a_factor() {
    // sigma = p makes Suyama's denominator divisible by p, so building the
    // curve fails with gcd = p, which is a factor.
    let n = times_m89(100003);
    assert_eq!(ecm_curve(&n, 100003, 1000), Some(U::from_u64(100003)));
}

#[test]
fn edge_cases() {
    assert_eq!(ecm_curve(&U::from_u64(4), 6, 100), Some(U::from_u64(2)));
    assert_eq!(ecm_curve(&U::from_u64(2), 6, 100), None);
    assert_eq!(ecm_curve(&U::ONE, 6, 100), None);
    assert_eq!(ecm_curve(&U::ZERO, 6, 100), None);
    // A prime: stage 1 reaches infinity mod p only together with mod n.
    assert_eq!(ecm_curve(&m89(), 6, 1000), None);
}

#[test]
fn many_curves_find_what_one_misses() {
    // A 15-digit prime that rho and p±1 can't reach: p − 1 = 2·(15-digit
    // prime) and p + 1 has the prime factor 402294200629. Some curve among
    // a few hundred has a 2000-smooth order.
    let p = 941_368_429_471_859;
    let n = times_m89(p);
    assert_eq!(ecm_curve(&n, 6, 2000), None);
    assert_eq!(ecm(&n, 2000, 500, 6), Some(U::from_u64(p)));
}
