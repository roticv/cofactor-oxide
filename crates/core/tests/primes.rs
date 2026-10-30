//! Tests for the segmented prime sieve.

use cofox_core::primes::Primes;

fn reference(lo: u64, hi: u64) -> Vec<u64> {
    let mut is_prime = vec![true; hi as usize + 1];
    let mut primes = Vec::new();
    for i in 2..=hi as usize {
        if is_prime[i] {
            if i as u64 >= lo {
                primes.push(i as u64);
            }
            for j in (i * i..=hi as usize).step_by(i) {
                is_prime[j] = false;
            }
        }
    }
    primes
}

#[test]
fn matches_reference() {
    let ranges = [
        (0, 0),
        (0, 1),
        (0, 2),
        (2, 2),
        (3, 3),
        (4, 4),
        (0, 100),
        (90, 97),
        (98, 100),
        // A segment holds 2^15 odd numbers, so the first boundary is at
        // 3 + 2^16 = 65539.
        (65_500, 65_600),
        (0, 300_000),
        (1_000_000, 1_100_000),
        (999_983, 999_983),
    ];
    for (lo, hi) in ranges {
        assert_eq!(
            Primes::new(lo, hi).collect::<Vec<_>>(),
            reference(lo, hi),
            "[{lo}, {hi}]"
        );
    }
}

#[test]
fn empty_ranges() {
    assert_eq!(Primes::new(10, 5).next(), None);
    assert_eq!(Primes::new(24, 28).next(), None);
}

#[test]
fn prime_counting() {
    // π(10^7) = 664,579.
    assert_eq!(Primes::new(0, 10_000_000).count(), 664_579);
    // π(10^9 + 10^6) − π(10^9) = 48,155.
    assert_eq!(Primes::new(1_000_000_000, 1_001_000_000).count(), 48_155);
}
