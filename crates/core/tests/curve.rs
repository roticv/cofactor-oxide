//! Tests for the Montgomery curve arithmetic.
//!
//! The group orders below were found by brute-force point counting in
//! Python: `#E = p + 1 + Σ_x (x³ + Ax² + x | p)`, or `2(p + 1) − #E` when
//! the starting point lies on the quadratic twist. All of them are divisible
//! by 12, as Suyama's parametrization promises.

use cofox_core::curve::{Curve, Point};
use cofox_core::{Monty, Uint};

/// (p, sigma, order of the Suyama curve's group mod p)
const ORDERS: [(u64, u64, u64); 40] = [
    (10007, 6, 9900),      // 2²·3²·5²·11
    (10007, 7, 10044),     // 2²·3⁴·31
    (10007, 8, 9816),      // 2³·3·409
    (10007, 9, 10152),     // 2³·3³·47
    (10007, 10, 10056),    // 2³·3·419
    (10007, 11, 10056),    // 2³·3·419
    (10007, 12, 9960),     // 2³·3·5·83
    (10007, 13, 10032),    // 2⁴·3·11·19
    (10007, 100, 10080),   // 2⁵·3²·5·7
    (10007, 12345, 9888),  // 2⁵·3·103
    (10009, 6, 10116),     // 2²·3²·281
    (10009, 7, 9936),      // 2⁴·3³·23
    (10009, 8, 9900),      // 2²·3²·5²·11
    (10009, 9, 10164),     // 2²·3·7·11²
    (10009, 10, 9852),     // 2²·3·821
    (10009, 11, 10068),    // 2²·3·839
    (10009, 12, 9984),     // 2⁸·3·13
    (10009, 13, 9900),     // 2²·3²·5²·11
    (10009, 100, 10116),   // 2²·3²·281
    (10009, 12345, 10140), // 2²·3·5·13²
    (30011, 6, 30048),     // 2⁵·3·313
    (30011, 7, 29832),     // 2³·3·11·113
    (30011, 8, 29892),     // 2²·3·47·53
    (30011, 9, 29952),     // 2⁸·3²·13
    (30011, 10, 29856),    // 2⁵·3·311
    (30011, 11, 29832),    // 2³·3·11·113
    (30011, 12, 30240),    // 2⁵·3³·5·7
    (30011, 13, 29784),    // 2³·3·17·73
    (30011, 100, 30168),   // 2³·3²·419
    (30011, 12345, 29760), // 2⁶·3·5·31
    (65537, 6, 65856),     // 2⁶·3·7³
    (65537, 7, 65208),     // 2³·3·11·13·19
    (65537, 8, 65556),     // 2²·3³·607
    (65537, 9, 65640),     // 2³·3·5·547
    (65537, 10, 65592),    // 2³·3²·911
    (65537, 11, 65088),    // 2⁶·3²·113
    (65537, 12, 65904),    // 2⁴·3·1373
    (65537, 13, 65460),    // 2²·3·5·1091
    (65537, 100, 65256),   // 2³·3·2719
    (65537, 12345, 65808), // 2⁴·3²·457
];

/// 2^61 − 1, a prime cofactor for the "ECM in miniature" tests.
const M61: u64 = (1 << 61) - 1;

/// `p · (2^61 − 1)`. About 77 bits, so it needs 4 limbs when limbs are
/// 32-bit.
fn times_m61(p: u64) -> Uint<4> {
    let (lo, hi) = Uint::<4>::from_u64(p).widening_mul(&Uint::from_u64(M61));
    assert!(hi.is_zero());
    lo
}

/// Whether two projective points have the same x-coordinate.
fn same_x<const L: usize>(m: &Monty<L>, a: &Point<L>, b: &Point<L>) -> bool {
    m.mul(&a.x, &b.z) == m.mul(&b.x, &a.z)
}

#[test]
fn multiplying_by_the_group_order_gives_infinity() {
    for (p, sigma, order) in ORDERS {
        assert_eq!(order % 12, 0, "Suyama order {order} for p = {p}");
        let monty = Monty::new(Uint::<1>::from_u64(p)).unwrap();
        let (curve, point) = Curve::suyama(&monty, sigma).unwrap();
        assert!(
            curve.mul(&point, order).z.is_zero(),
            "[{order}]P mod {p}, sigma {sigma}"
        );
        // One more step lands back on P, so P itself isn't the identity.
        let next = curve.mul(&point, order + 1);
        assert!(!next.z.is_zero());
        assert!(same_x(&monty, &next, &point), "[{order} + 1]P = P mod {p}");
    }
}

#[test]
fn ecm_in_miniature() {
    // Modulo n = p·q, multiplying by #E(F_p) gives infinity mod p but not
    // mod the large prime q, so gcd(Z, n) = p. That's ECM's stage 1 when the
    // multiplier happens to be a multiple of the group order.
    for (p, sigma, order) in ORDERS {
        let n = times_m61(p);
        let monty = Monty::new(n).unwrap();
        let (curve, point) = Curve::suyama(&monty, sigma).unwrap();
        let z = curve.mul(&point, order).z;
        assert_eq!(z.gcd(&n), Uint::from_u64(p), "p = {p}, sigma = {sigma}");
    }
}

#[test]
fn smooth_order_is_found_by_prime_powers() {
    // For sigma = 100 mod 10007 the order is 10080 = 2⁵·3²·5·7, so it
    // divides the product of the prime powers up to 32: 32·27·25·7·11·13·…
    // Multiplying by each in turn, as stage 1 does, reaches infinity mod p.
    let n = times_m61(10007);
    let monty = Monty::new(n).unwrap();
    let (curve, mut point) = Curve::suyama(&monty, 100).unwrap();
    for q in [32, 27, 25, 7, 11, 13, 17, 19, 23, 29, 31] {
        point = curve.mul(&point, q);
    }
    assert_eq!(point.z.gcd(&n), Uint::from_u64(10007));
    // Without the 7, the order doesn't divide the multiplier.
    let (curve, mut point) = Curve::suyama(&monty, 100).unwrap();
    for q in [32, 27, 25, 11, 13] {
        point = curve.mul(&point, q);
    }
    assert_eq!(point.z.gcd(&n), Uint::ONE);
}

/// A curve modulo the prime 2^127 − 1 for the consistency checks. It uses
/// 4 limbs so it also fits with 32-bit limbs.
fn big_curve() -> (Monty<4>, Curve<4>, Point<4>) {
    let p = (Uint::<4>::ONE << 127).wrapping_sub(&Uint::ONE);
    let monty = Monty::new(p).unwrap();
    let (curve, point) = Curve::suyama(&monty, 42).unwrap();
    (monty, curve, point)
}

#[test]
fn ladder_matches_repeated_addition() {
    let (monty, curve, p) = big_curve();
    // Walk P, 2P, 3P, ... using [k+1]P = [k]P + P with difference [k−1]P.
    let mut prev = p;
    let mut cur = curve.double(&p);
    for k in 3..300 {
        let next = curve.add(&cur, &p, &prev);
        assert!(same_x(&monty, &next, &curve.mul(&p, k)), "k = {k}");
        prev = cur;
        cur = next;
    }
}

#[test]
fn scalar_multiplication_composes() {
    let (monty, curve, p) = big_curve();
    let mut seed = 12345u64;
    for _ in 0..50 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let a = (seed >> 33) | 1;
        let b = (seed & 0xFFFF_FFFF) | 2;
        let ab = curve.mul(&curve.mul(&p, a), b);
        assert!(
            same_x(&monty, &ab, &curve.mul(&p, a * b)),
            "a = {a}, b = {b}"
        );
    }
}

#[test]
fn small_multiples() {
    let (monty, curve, p) = big_curve();
    assert!(curve.mul(&p, 0).z.is_zero());
    assert_eq!(curve.mul(&p, 1), p);
    assert!(same_x(&monty, &curve.mul(&p, 2), &curve.double(&p)));
}

#[test]
fn suyama_reports_common_factors() {
    // sigma = p makes v = 4σ ≡ 0 (mod p), so the denominator 16u³v shares
    // the factor p with n.
    let p = 10007u64;
    let n = times_m61(p);
    let monty = Monty::new(n).unwrap();
    assert_eq!(Curve::suyama(&monty, p).err(), Some(Uint::from_u64(p)));
    // Modulo p alone, the shared factor is the whole modulus.
    let monty = Monty::new(Uint::<1>::from_u64(p)).unwrap();
    assert_eq!(Curve::suyama(&monty, 3 * p).err(), Some(Uint::from_u64(p)));
}
