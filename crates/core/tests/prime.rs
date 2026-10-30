//! Tests for primality testing.

use cofox_core::prime::{is_prime, miller_rabin, strong_lucas};
use cofox_core::{Monty, Uint};

const LIMIT: usize = 100_000;

/// Strong pseudoprimes to base 2 below 100,000 (OEIS A001262).
const SPSP2: [u64; 16] = [
    2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141, 52633, 65281, 74665, 80581, 85489,
    88357, 90751,
];

/// Strong Lucas pseudoprimes (Selfridge parameters) below 100,000
/// (OEIS A217255).
const SLPSP: [u64; 12] = [
    5459, 5777, 10877, 16109, 18971, 22499, 24569, 25199, 40309, 58519, 75077, 97439,
];

fn sieve(limit: usize) -> Vec<bool> {
    let mut is_prime = vec![true; limit];
    is_prime[0] = false;
    is_prime[1] = false;
    for i in 2..limit {
        if is_prime[i] {
            for j in (i * i..limit).step_by(i) {
                is_prime[j] = false;
            }
        }
    }
    is_prime
}

#[test]
fn matches_sieve() {
    let primes = sieve(LIMIT);
    for (n, &expected) in primes.iter().enumerate() {
        assert_eq!(
            is_prime(&Uint::<1>::from_u64(n as u64)),
            expected,
            "n = {n}"
        );
    }
}

#[test]
fn miller_rabin_pseudoprimes() {
    let primes = sieve(LIMIT);
    let mut liars = Vec::new();
    for n in (3..LIMIT as u64).step_by(2) {
        let monty = Monty::new(Uint::<1>::from_u64(n)).unwrap();
        let passes = miller_rabin(&monty, 2);
        if primes[n as usize] {
            assert!(passes, "prime {n} failed Miller-Rabin");
        } else if passes {
            liars.push(n);
        }
    }
    assert_eq!(liars, SPSP2);
}

#[test]
fn strong_lucas_pseudoprimes() {
    let primes = sieve(LIMIT);
    let mut liars = Vec::new();
    // Start above 13: for n ≤ 13, Selfridge's D can equal ±n.
    for n in (15..LIMIT as u64).step_by(2) {
        let monty = Monty::new(Uint::<1>::from_u64(n)).unwrap();
        let passes = strong_lucas(&monty);
        if primes[n as usize] {
            assert!(passes, "prime {n} failed strong Lucas");
        } else if passes {
            liars.push(n);
        }
    }
    assert_eq!(liars, SLPSP);
}

#[test]
fn large_primes() {
    // Mersenne primes 2^61 - 1, 2^89 - 1, 2^127 - 1 and 2^521 - 1. Seventeen
    // limbs fit 2^521 - 1 even with 32-bit limbs.
    for e in [61, 89, 127, 521] {
        let p = (Uint::<17>::ONE << e).wrapping_sub(&Uint::ONE);
        assert!(is_prime(&p), "2^{e} - 1");
    }
    // 2^67 - 1 = 193707721 · 761838257287 is composite.
    assert!(!is_prime(&(Uint::<4>::ONE << 67).wrapping_sub(&Uint::ONE)));
}

#[test]
fn large_composites() {
    let p = (Uint::<8>::ONE << 127).wrapping_sub(&Uint::ONE);
    let q = (Uint::<8>::ONE << 89).wrapping_sub(&Uint::ONE);
    // A product of two large primes, and a perfect square, which needs the
    // square check in the Lucas parameter search.
    assert!(!is_prime(&p.widening_mul(&q).0));
    assert!(!is_prime(&p.widening_mul(&p).0));
    assert!(!is_prime(&Uint::<2>::from_u64(1_000_003 * 1_000_003)));
}

#[test]
fn all_ones_modulus() {
    // n + 1 overflows here, which the Lucas test has to handle.
    // 2^64 - 1 = 3 · 5 · 17 · 257 · 641 · 65537 · 6700417.
    assert!(!is_prime(&Uint::<2>::from_u64(u64::MAX)));
    let monty = Monty::new(Uint::<1>::MAX).unwrap();
    let _ = strong_lucas(&monty);
}
