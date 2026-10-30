//! Tests for Williams' p+1 method.
//!
//! The test primes were generated with known factorizations of p + 1 and
//! chosen Jacobi symbols. Seed 3 has A² − 4 = 5 and seed 4 has A² − 4 = 12,
//! so seed 3 takes the p+1 path when (5/p) = −1 and seed 4 when (3/p) = −1.
//!
//! - pa + 1 = 2·7·101·109·211·449·593·769·937·983, (5/pa) = −1, (3/pa) = +1
//! - pb + 1 = 2·3·5·61·89·509·587·617·727·733·971, (5/pb) = +1, (3/pb) = −1
//! - pc + 1 = 2·3·31·409·647·661·811·863·50021, (5/pc) = −1
//! - pd + 1 = 2·59·127·641·823·827·881·947·977, (5/pd) = −1
//! - pe + 1 = 2·29·89·137·521·557·859·983·50023, (5/pe) = −1
//! - p1 − 1 = 2·7·17·61·83·131·257·487·499·547·569 (from the p−1 tests),
//!   with (5/p1) = −1 and (3/p1) = +1
//! - h1 and h2 are safe primes whose h + 1 also has a large prime factor.
//!
//! Products are at most 204 bits, so `Uint<8>` holds them with 32-bit limbs.

use cofox_core::pp1::williams_pp1;
use cofox_core::{Factor, Uint, factor_str};

type U = Uint<8>;

const PA: &str = "6133083049196562700597";
const PB: &str = "15536019561998628169769";
const PC: &str = "1139006000498336762573";
const PD: &str = "5329292455663758877093";
const PE: &str = "8668578799127078693557";
const P1: &str = "3068445422297759556683";
const H1: &str = "807637421151890919476963";
const H2: &str = "1051424183656750164329928625583";

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
    let n = product(PA, H1);
    assert_eq!(williams_pp1(&n, 3, 1000, 1000), Some(u(PA)));
    assert_eq!(williams_pp1(&n, 3, 983, 983), Some(u(PA)));
    // 983 is the largest prime factor of pa + 1.
    assert_eq!(williams_pp1(&n, 3, 982, 982), None);
}

#[test]
fn seed_selects_the_group() {
    // (5/pa) = −1 but (3/pa) = +1, so only seed 3 sees the group of order
    // pa + 1. For pb it's the other way round.
    let a = product(PA, H1);
    assert_eq!(williams_pp1(&a, 3, 1000, 1000), Some(u(PA)));
    assert_eq!(williams_pp1(&a, 4, 1000, 1000), None);
    let b = product(PB, H1);
    assert_eq!(williams_pp1(&b, 3, 1000, 1000), None);
    assert_eq!(williams_pp1(&b, 4, 1000, 1000), Some(u(PB)));
}

#[test]
fn finds_p_minus_1_smooth_primes_too() {
    // (3/p1) = +1, so with seed 4 the sequence lives in the group of order
    // p1 − 1, which is 1000-smooth. Seed 3 sees order p1 + 1 instead.
    let n = product(P1, H1);
    assert_eq!(williams_pp1(&n, 4, 1000, 1000), Some(u(P1)));
    assert_eq!(williams_pp1(&n, 3, 1000, 1000), None);
}

#[test]
fn stage_2() {
    let n = product(PC, H1);
    // pc + 1 is 1000-smooth apart from the prime 50021. With B1 = 1000, the
    // first stage-2 prime (1009) rounds to giant step k = 0, which needs
    // V_(−D) = V_D to start the giant-step recurrence.
    assert_eq!(williams_pp1(&n, 3, 1000, 1000), None);
    assert_eq!(williams_pp1(&n, 3, 1000, 50_020), None);
    assert_eq!(williams_pp1(&n, 3, 1000, 50_021), Some(u(PC)));
    assert_eq!(williams_pp1(&n, 3, 1000, 1_000_000), Some(u(PC)));
    // Stage 2 also covers the largest stage-1 prime when B1 is too low.
    assert_eq!(williams_pp1(&product(PA, H1), 3, 982, 1000), Some(u(PA)));
    // A B1 below D = 2310 and one well above it, so stage 2 starts at
    // different giant steps.
    assert_eq!(williams_pp1(&n, 3, 2000, 60_000), Some(u(PC)));
    assert_eq!(williams_pp1(&n, 3, 10_000, 60_000), Some(u(PC)));
}

#[test]
fn stage_1_replay() {
    // pa + 1 and pd + 1 are both done by 983, so the first chunk's GCD is n.
    // The replay finds pd first, at 977.
    let n = product(PA, PD);
    assert_eq!(williams_pp1(&n, 3, 1000, 1000), Some(u(PD)));
}

#[test]
fn stage_2_replay() {
    // The stage-2 primes 50021 and 50023 land in the same chunk.
    let n = product(PC, PE);
    assert_eq!(williams_pp1(&n, 3, 1000, 100_000), Some(u(PC)));
}

#[test]
fn hard_primes_resist() {
    let n = product(H1, H2);
    assert_eq!(williams_pp1(&n, 3, 50_000, 5_000_000), None);
    assert_eq!(williams_pp1(&n, 4, 50_000, 5_000_000), None);
}

#[test]
fn edge_cases() {
    assert_eq!(
        williams_pp1(&U::from_u64(4), 3, 100, 1000),
        Some(U::from_u64(2))
    );
    assert_eq!(williams_pp1(&U::from_u64(2), 3, 100, 1000), None);
    assert_eq!(williams_pp1(&U::ONE, 3, 100, 1000), None);
    assert_eq!(williams_pp1(&U::ZERO, 3, 100, 1000), None);
    assert_eq!(williams_pp1(&u(PA), 3, 1000, 1000), None);
    assert_eq!(williams_pp1(&product(PC, H1), 3, 1000, 10), None);
}

#[test]
fn pipeline_uses_pp1() {
    // pf + 1 is 50,000-smooth with (5/pf) = −1, while pf − 1 has a 29-digit
    // prime factor. pf has 32 digits, so neither rho nor p−1 can split this.
    let pf = "11792549486203707942476487425257";
    let n = product(pf, H2).to_string();
    let prime = |v: &str| Factor {
        value: v.to_string(),
        is_prime: true,
    };
    assert_eq!(factor_str(&n).unwrap(), [prime(H2), prime(pf)]);
}
