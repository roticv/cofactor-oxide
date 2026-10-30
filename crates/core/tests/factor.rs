//! Tests for the factoring pipeline.

use cofox_core::prime::is_prime;
use cofox_core::{Factor, Uint, factor, factor_str};

fn factor_u64(n: u64) -> Vec<u64> {
    factor(&Uint::<2>::from_u64(n))
        .iter()
        .map(|f| {
            assert!(f.is_prime, "{} not split", f.value);
            f.value.to_u64().unwrap()
        })
        .collect()
}

fn primes(values: &[&str]) -> Vec<Factor<String>> {
    values
        .iter()
        .map(|v| Factor {
            value: v.to_string(),
            is_prime: true,
        })
        .collect()
}

#[test]
fn small_values() {
    assert_eq!(factor_u64(0), Vec::<u64>::new());
    assert_eq!(factor_u64(1), Vec::<u64>::new());
    assert_eq!(factor_u64(2), [2]);
    assert_eq!(factor_u64(360), [2, 2, 2, 3, 3, 5]);
    assert_eq!(factor_u64(1_000_000_007), [1_000_000_007]);
}

#[test]
fn needs_rho() {
    assert_eq!(
        factor_u64(1_000_000_016_000_000_063),
        [1_000_000_007, 1_000_000_009]
    );
    assert_eq!(factor_u64(u64::MAX), [3, 5, 17, 257, 641, 65537, 6700417]);
    // A prime power above the trial-division limit.
    assert_eq!(factor_u64(4099u64.pow(5)), [4099; 5]);
}

#[test]
fn random_u64s() {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..300 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        // Vary the size so every path gets exercised.
        let n = x >> (x % 60);
        let factors = factor_u64(n);
        assert_eq!(factors.iter().product::<u64>(), n.max(1));
        assert!(factors.is_sorted());
        for &p in &factors {
            assert!(is_prime(&Uint::<2>::from_u64(p)), "{p} in factors of {n}");
        }
    }
}

#[test]
fn from_strings() {
    assert_eq!(
        factor_str("360").unwrap(),
        primes(&["2", "2", "2", "3", "3", "5"])
    );
    // 4294967291 · (2^127 - 1), which needs more than 64 bits.
    assert_eq!(
        factor_str("730750817814745541799496257699704991244250775557").unwrap(),
        primes(&["4294967291", "170141183460469231731687303715884105727"])
    );
    // 2^521 - 1 is prime, and needs the widest integers.
    let m521 = (Uint::<16>::ONE << 521)
        .wrapping_sub(&Uint::ONE)
        .to_string();
    let m521 = &m521[..];
    if (Uint::<16>::BITS) > 521 {
        assert_eq!(factor_str(m521).unwrap(), primes(&[m521]));
    }
}

#[test]
fn unsplit_composite() {
    // Two safe primes (q = 2r + 1, r prime) of 24 and 31 digits: too big for
    // rho, and q − 1 has a huge prime factor, so p−1 can't find them either.
    let n = "875629172352184491588547325680801513040676851112599077";
    let factors = factor_str(n).unwrap();
    assert_eq!(
        factors,
        [Factor {
            value: n.to_string(),
            is_prime: false
        }]
    );
    assert_eq!(factors[0].to_string(), format!("({n})"));
}

#[test]
fn bad_input() {
    assert!(factor_str("").is_err());
    assert!(factor_str("12x").is_err());
    let too_big = "9".repeat(400);
    assert!(factor_str(&too_big).is_err());
}
