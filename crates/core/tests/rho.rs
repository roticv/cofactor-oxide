//! Tests for Pollard's rho.
//!
//! Sizes are chosen so the numbers fit with 32-bit limbs too.

use cofox_core::Uint;
use cofox_core::rho::pollard_rho;

const MAX_STEPS: u64 = 10_000_000;

/// Runs rho with successive values of `c` until it finds a factor.
fn find_factor<const L: usize>(n: &Uint<L>) -> Uint<L> {
    for c in 1..20 {
        if let Some(g) = pollard_rho(n, c, MAX_STEPS) {
            return g;
        }
    }
    panic!("no factor found for {n}");
}

/// Checks that rho splits `p * q` into `p` and `q`.
fn check_semiprime<const L: usize>(p: &str, q: &str) {
    let p: Uint<L> = p.parse().unwrap();
    let q: Uint<L> = q.parse().unwrap();
    let (n, hi) = p.widening_mul(&q);
    assert!(hi.is_zero(), "{p} * {q} overflows Uint<{L}>");
    let g = find_factor(&n);
    assert!(g == p || g == q, "rho returned {g} for {p} * {q}");
}

#[test]
fn small_semiprimes() {
    check_semiprime::<2>("3", "5");
    check_semiprime::<2>("101", "103");
    check_semiprime::<2>("65537", "4294967291");
    check_semiprime::<2>("1000000007", "1000000009");
}

#[test]
fn prime_squares() {
    check_semiprime::<2>("3", "3");
    check_semiprime::<2>("1000003", "1000003");
}

#[test]
fn small_factor_of_large_number() {
    // Rho's running time depends on the smallest factor, not on n.
    // 2^127 - 1 is prime.
    check_semiprime::<8>("4294967291", "170141183460469231731687303715884105727");
    // 2^89 - 1 is prime.
    check_semiprime::<6>("1000000007", "618970019642690137449562111");
}

#[test]
fn several_factors() {
    // 561 = 3 · 11 · 17, a Carmichael number.
    for n in [561u64, 3 * 5 * 7 * 11 * 13, 1_000_000_007 * 97 * 89] {
        let g = find_factor(&Uint::<2>::from_u64(n)).to_u64().unwrap();
        assert!(g > 1 && g < n && n % g == 0, "rho returned {g} for {n}");
    }
}

#[test]
fn single_polynomial_success_rate() {
    // For small n, one batch often spans the cycles modulo both factors, so
    // the batched GCD is n and the batch has to be replayed step by step.
    // With c = 1 this splits 921 of the 990 products of two primes below
    // 200. Skipping the replay drops that to 728.
    let primes: Vec<u64> = (3..200u64)
        .filter(|&p| (2..p).all(|d| p % d != 0))
        .collect();
    let mut found = 0;
    for (i, &p) in primes.iter().enumerate() {
        for &q in &primes[i + 1..] {
            if let Some(g) = pollard_rho(&Uint::<1>::from_u64(p * q), 1, 100_000) {
                let g = g.to_u64().unwrap();
                assert!(g == p || g == q, "rho returned {g} for {p} * {q}");
                found += 1;
            }
        }
    }
    assert!(found >= 900, "only {found} of 990 split with c = 1");
}

#[test]
fn even_numbers() {
    assert_eq!(pollard_rho(&Uint::<1>::from_u64(2), 1, 100), None);
    assert_eq!(
        pollard_rho(&Uint::<1>::from_u64(4), 1, 100),
        Some(Uint::from_u64(2))
    );
    assert_eq!(
        pollard_rho(&Uint::<1>::from_u64(1 << 31), 1, 100),
        Some(Uint::from_u64(2))
    );
}

#[test]
fn gives_up_on_primes() {
    for n in [0u64, 1, 3, 1_000_000_007, (1 << 61) - 1] {
        assert_eq!(
            pollard_rho(&Uint::<2>::from_u64(n), 1, 10_000),
            None,
            "n = {n}"
        );
    }
}
