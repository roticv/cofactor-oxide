//! Tests for the quadratic sieve groundwork.
//!
//! Reference values for the multiplier and the factor base come from an
//! independent Python implementation of the same definitions, which uses its
//! own primes, logarithms and square roots (Cipolla's algorithm rather than
//! Tonelli–Shanks).

use cofox_core::Uint;
use cofox_core::primes::Primes;
use cofox_core::qs::{FactorBase, Params, Prime, multiplier, pow_mod, sqrt_mod};

/// RSA-100 is 330 bits, and kN a few bits more: 12 limbs is at least 384
/// bits with either limb size.
type U = Uint<12>;

const RSA100: &str = "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139";

fn u(s: &str) -> U {
    s.parse().unwrap()
}

#[test]
fn square_roots_of_every_residue() {
    for p in Primes::new(2, 2000) {
        let mut squares = vec![false; p as usize];
        for x in 0..p {
            squares[(x * x % p) as usize] = true;
        }
        for a in 0..p {
            match sqrt_mod(a, p) {
                Some(r) => {
                    assert_eq!(r * r % p, a, "sqrt({a}) mod {p} = {r}");
                    assert!(r <= p - r || r == 0, "{r} isn't the smaller root mod {p}");
                }
                None => assert!(!squares[a as usize], "{a} is a square mod {p}"),
            }
            if squares[a as usize] {
                assert!(sqrt_mod(a, p).is_some(), "{a} mod {p}");
            }
        }
    }
}

#[test]
fn square_roots_mod_large_primes() {
    // 3·2^30 + 1 makes Tonelli–Shanks take up to 30 rounds; 2^32 − 5 is
    // 3 mod 4, the shortcut case; 4294967161 is 1 mod 8, with p − 1 = 8·odd,
    // a typical case near the top of the range.
    for p in [3 * (1 << 30) + 1, (1 << 32) - 5, 4_294_967_161] {
        assert_eq!(pow_mod(3, p - 1, p), 1, "{p} isn't prime");
        let mut x = 0x9E37_79B9u64;
        for _ in 0..2000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (x >> 32) % p;
            let a = r * r % p;
            let root = sqrt_mod(a, p).unwrap();
            assert!(root == r || root == p - r, "sqrt({a}) mod {p}");
            // r·z is a non-residue whenever z is, so -a is one if p ≡ 3 (4).
            if p % 4 == 3 && a != 0 {
                assert_eq!(sqrt_mod(p - a, p), None);
            }
        }
    }
}

#[test]
fn multipliers() {
    // (N, Knuth–Schroeppel multiplier from the Python reference). Each
    // winner beats the runner-up by at least 0.2, far beyond rounding.
    for (n, k) in [
        (RSA100, 1),
        (
            "3990194206234976856228988863013364209693291429792598841419",
            11,
        ),
        (
            "22402621473499677702312787220798894771372529738736380992481737381422651598174113",
            1,
        ),
        ("875629172352184491588547325680801513040676851112599077", 13),
        // 7N ≡ 1 (mod 8), and the full weight of 2 for that case decides it.
        ("3313211328043926463409960759101908832987539554864689663", 7),
    ] {
        assert_eq!(multiplier(&u(n)), k, "N = {n}");
    }
}

#[test]
fn multiplier_fits_the_width() {
    // N just under 2^64 in a single 64-bit limb: kN overflows for every
    // k > 1, so only k = 1 remains.
    if cofox_core::arith::LIMB_BITS == 64 {
        let n = Uint::<1>::from_u64(18_446_744_073_709_551_557); // 2^64 − 59, prime
        assert_eq!(multiplier(&n), 1);
    }
}

/// Checks a factor base against the Python reference: its size, first
/// primes, last prime, and the sums of the primes, roots and logs.
fn check_base(
    base: &FactorBase<12>,
    first: &[(u32, u32, u8)],
    last: (u32, u32, u8),
    sums: (u64, u64, u64),
) {
    let prime = |(p, sqrt, log)| Prime { p, sqrt, log };
    let primes = &base.primes;
    let expected_first: Vec<_> = first.iter().copied().map(prime).collect();
    assert_eq!(primes[..first.len()], expected_first);
    assert_eq!(*primes.last().unwrap(), prime(last));
    let sum = |f: fn(&Prime) -> u64| primes.iter().map(f).sum::<u64>();
    assert_eq!(
        (
            sum(|p| p.p.into()),
            sum(|p| p.sqrt.into()),
            sum(|p| p.log.into())
        ),
        sums
    );
    let kn = &base.kn;
    for &Prime { p, sqrt, .. } in primes {
        let r = kn.div_rem(&Uint::from_u64(p.into())).1.to_u64().unwrap();
        assert_eq!(
            u64::from(sqrt) * u64::from(sqrt) % u64::from(p),
            r,
            "p = {p}"
        );
    }
    assert!(primes.windows(2).all(|w| w[0].p < w[1].p));
}

#[test]
fn rsa100_factor_base() {
    let n = u(RSA100);
    let k = multiplier(&n);
    let base = FactorBase::new(&n, k, Params::for_bits(330).fb_size).unwrap();
    assert_eq!(base.primes.len(), 97_647);
    check_base(
        &base,
        &[
            (2, 1, 1),
            (3, 1, 2),
            (5, 2, 2),
            (13, 5, 4),
            (19, 4, 4),
            (23, 8, 5),
        ],
        (2_671_577, 33_471, 21),
        (125_483_822_132, 31_298_864_278, 1_930_172),
    );
}

#[test]
fn factor_base_with_a_multiplier() {
    // k = 11 divides kN, so 11 is in the factor base with root 0. (N ≡ 3 is
    // a square mod 11 too, but with k = 11 that no longer matters.)
    let n = u("3990194206234976856228988863013364209693291429792598841419");
    let base = FactorBase::new(&n, 11, 2705).unwrap();
    assert_eq!(base.multiplier, 11);
    check_base(
        &base,
        &[
            (2, 1, 1),
            (3, 1, 2),
            (5, 2, 2),
            (7, 2, 3),
            (11, 0, 3),
            (13, 1, 4),
        ],
        (53_161, 24_603, 16),
        (67_830_066, 16_781_913, 38_053),
    );
}

#[test]
fn factor_base_finds_a_factor() {
    // 4099 is above the trial-division limit. A 1000-prime factor base goes
    // past it (only 564 primes are below 4100, and about half qualify), so
    // building it finds that 4099 divides N.
    let n = U::from_u64(4099)
        .widening_mul(&u("618970019642690137449562111"))
        .0;
    assert_eq!(FactorBase::new(&n, 1, 1000).unwrap_err(), U::from_u64(4099));
}

#[test]
fn params_interpolate() {
    let p = |bits| {
        let p = Params::for_bits(bits);
        (p.fb_size, p.large_mult, p.blocks)
    };
    // Table rows, between rows, and clamped at both ends.
    assert_eq!(p(64), (100, 40, 1));
    assert_eq!(p(200), (3000, 50, 1));
    assert_eq!(p(330), (97_647, 150, 9));
    assert_eq!(p(250), (28_352, 100, 3));
    // The large-prime multiplier goes down between 266 and 283 bits.
    assert_eq!(p(290), (57_333, 80, 5));
    assert_eq!(p(20), (100, 40, 1));
    assert_eq!(p(400), (210_000, 150, 13));
}
