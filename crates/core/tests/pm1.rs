//! Tests for Pollard's p−1 method.
//!
//! The test primes were generated with known factorizations of p − 1:
//!
//! - p1 − 1 = 2·7·17·61·83·131·257·487·499·547·569
//! - p2 − 1 = 2·193·269·641·787·911·971·50021
//! - p4 − 1 = 2·3·31·59·149·311²·409·541·653·701
//! - p5 − 1 = 2·127·347·787·821·883·983·50023
//! - q and q2 are safe primes (q = 2r + 1 with r prime), so p − 1 can never
//!   find them with practical bounds.
//!
//! Products are at most 203 bits, so `Uint<8>` holds them with 32-bit limbs.

use cofox_core::pm1::pollard_pm1;
use cofox_core::{Factor, Uint, factor_str};

type U = Uint<8>;

const P1: &str = "3068445422297759556683";
const P2: &str = "2317727230097987350079";
const P4: &str = "16018571153544046776223";
const P5: &str = "2472663602922855357923";
const Q: &str = "734123452237455403272119";

fn u(s: &str) -> U {
    s.parse().unwrap()
}

fn product(a: &str, b: &str) -> U {
    let (lo, hi) = u(a).widening_mul(&u(b));
    assert!(hi.is_zero());
    lo
}

#[test]
fn stage_1() {
    let n = product(P1, Q);
    assert_eq!(pollard_pm1(&n, 1000, 1000), Some(u(P1)));
    assert_eq!(pollard_pm1(&n, 569, 569), Some(u(P1)));
    // 569 is the largest prime factor of p1 − 1.
    assert_eq!(pollard_pm1(&n, 568, 568), None);
}

#[test]
fn stage_2() {
    let n = product(P2, Q);
    // p2 − 1 is 1000-smooth apart from the prime 50021.
    assert_eq!(pollard_pm1(&n, 1000, 1000), None);
    assert_eq!(pollard_pm1(&n, 1000, 50_020), None);
    assert_eq!(pollard_pm1(&n, 1000, 50_021), Some(u(P2)));
    assert_eq!(pollard_pm1(&n, 1000, 100_000), Some(u(P2)));
    // Stage 2 also picks up the largest stage-1 prime when B1 is too low.
    assert_eq!(pollard_pm1(&product(P1, Q), 568, 1000), Some(u(P1)));
}

#[test]
fn stage_1_replay() {
    // With B1 = 100,000 both p1 − 1 and p4 − 1 (which needs 311²) divide
    // the exponent by the end of the first chunk, so that chunk's GCD is n.
    // Replaying it one prime power at a time finds p1 first, at 569.
    let n = product(P1, P4);
    assert_eq!(pollard_pm1(&n, 100_000, 100_000), Some(u(P1)));
}

#[test]
fn stage_2_replay() {
    // The stage-2 primes 50021 and 50023 are consecutive, so they land in
    // the same chunk and its GCD is n. The replay finds p2 at 50021.
    let n = product(P2, P5);
    assert_eq!(pollard_pm1(&n, 1000, 100_000), Some(u(P2)));
}

#[test]
fn safe_primes_resist() {
    let n = product(Q, "1192754665013696431541079322883");
    assert_eq!(pollard_pm1(&n, 50_000, 5_000_000), None);
}

#[test]
fn edge_cases() {
    assert_eq!(
        pollard_pm1(&U::from_u64(4), 100, 1000),
        Some(U::from_u64(2))
    );
    assert_eq!(pollard_pm1(&U::from_u64(2), 100, 1000), None);
    assert_eq!(pollard_pm1(&U::ONE, 100, 1000), None);
    assert_eq!(pollard_pm1(&U::ZERO, 100, 1000), None);
    // A prime: 3^E ≡ 1 (mod p) once p − 1 | E, so the GCD is p itself.
    assert_eq!(pollard_pm1(&u(P1), 1000, 1000), None);
    // B2 at or below B1 means no stage 2.
    assert_eq!(pollard_pm1(&product(P2, Q), 1000, 10), None);
}

#[test]
fn pipeline_uses_pm1() {
    // p3 − 1 is 50,000-smooth, but p3 has 30 digits, which is far beyond
    // rho's reach. Only p−1 can split this.
    let p3 = "861892838777146873591119616019";
    let q2 = "1192754665013696431541079322883";
    let n = product(p3, q2).to_string();
    let prime = |v: &str| Factor {
        value: v.to_string(),
        is_prime: true,
    };
    assert_eq!(factor_str(&n).unwrap(), [prime(p3), prime(q2)]);
}
