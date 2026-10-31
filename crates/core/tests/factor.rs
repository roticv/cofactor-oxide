//! Tests for the factoring pipeline.

use cofox_core::prime::is_prime;
use std::ops::ControlFlow;

use cofox_core::{Config, Factor, Progress, QsMode, Uint, factor, factor_str, factor_str_with};

fn factor_u64(n: u64) -> Vec<u64> {
    factor(&Uint::<2>::from_u64(n))
        .iter()
        .map(|f| {
            assert!(f.is_prime, "{} not split", f.value);
            f.value.to_u64().unwrap()
        })
        .collect()
}

/// Factors with the given ECM level, no quadratic sieve and no progress
/// callback.
fn with_ecm(n: &str, ecm_digits: u32) -> Vec<Factor<String>> {
    let config = Config::default().ecm_digits(ecm_digits).qs(QsMode::Off);
    factor_str_with(n, &config, |_| ControlFlow::Continue(())).unwrap()
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

/// 2^89 − 1, a 27-digit prime.
const M89: &str = "618970019642690137449562111";

fn times_m89(p: &str) -> String {
    let (n, hi) = p
        .parse::<Uint<8>>()
        .unwrap()
        .widening_mul(&M89.parse().unwrap());
    assert!(hi.is_zero());
    n.to_string()
}

#[test]
fn needs_ecm() {
    // None of these primes is in reach of rho, and p − 1 and p + 1 all have
    // a prime factor above 10^9, so only ECM can find them.
    for p in [
        "941368429471859",
        "11888570237800257629",
        "60542640892254544061",
    ] {
        let n = times_m89(p);
        assert_eq!(factor_str(&n).unwrap(), primes(&[p, M89]), "p = {p}");
        assert!(!with_ecm(&n, 0)[0].is_prime, "p = {p} found without ECM");
    }
}

#[test]
fn ecm_levels_are_probabilistic() {
    // Each level runs about as many curves as it takes on average to find a
    // prime of its size, so it misses one now and then. These 20-digit
    // curves miss this 20-digit prime, and the next level finds it.
    let p = "98936812917358800691";
    let n = times_m89(p);
    assert!(!with_ecm(&n, 20)[0].is_prime);
    assert_eq!(with_ecm(&n, 25), primes(&[p, M89]));
}

#[test]
fn ecm_beyond_its_level() {
    // ECM capped at 20 digits still has a fair chance at a 24-digit prime,
    // and with these curves it finds 734123452237455403272119.
    let n = "875629172352184491588547325680801513040676851112599077";
    assert_eq!(
        with_ecm(n, 20),
        primes(&[
            "734123452237455403272119",
            "1192754665013696431541079322883"
        ])
    );
}

#[test]
fn reports_progress() {
    // The 15-digit level's 30 curves miss 941368429471859, and the 20-digit
    // level's fourth curve (sigma = 39) finds it.
    let n = times_m89("941368429471859");
    let bits = n.parse::<Uint<8>>().unwrap().bits();
    let mut events = Vec::new();
    let config = Config::default().qs(QsMode::Off);
    let factors = factor_str_with(&n, &config, |event| {
        events.push(event);
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(factors, primes(&["941368429471859", M89]));
    let method = |method| Progress::Method { method, bits };
    assert_eq!(events[..3], [method("rho"), method("p-1"), method("p+1")]);
    let ecm = |digits, done, total| Progress::Ecm {
        bits,
        digits,
        done,
        total,
    };
    // Every curve is reported, except the one that finds p. The two prime
    // pieces then need no work at all.
    let expected: Vec<_> = (0..=30)
        .map(|done| ecm(15, done, 30))
        .chain((0..=3).map(|done| ecm(20, done, 100)))
        .collect();
    assert_eq!(events[3..], expected);
}

#[test]
fn stops_when_asked() {
    // The quick rho run finds 1000003, then the 20-digit prime needs ECM,
    // which is stopped before its first curve. The prime piece is still
    // reported as prime.
    let rest = times_m89("11888570237800257629");
    let n = rest
        .parse::<Uint<8>>()
        .unwrap()
        .widening_mul(&Uint::from_u64(1000003))
        .0
        .to_string();
    let factors = factor_str_with(&n, &Config::default(), |event| match event {
        Progress::Ecm { .. } => ControlFlow::Break(()),
        _ => ControlFlow::Continue(()),
    })
    .unwrap();
    let unsplit = Factor {
        value: rest,
        is_prime: false,
    };
    assert_eq!(factors, [primes(&["1000003"])[0].clone(), unsplit]);
}

#[test]
fn workers_share_the_curves() {
    let n = times_m89("941368429471859");
    let mut found_by = [false; 4];
    let mut totals = [0; 2];
    for worker in 0..4 {
        let config = Config::default().worker(worker, 4).qs(QsMode::Off);
        let factors = factor_str_with(&n, &config, |event| {
            if let Progress::Ecm {
                digits,
                done: 0,
                total,
                ..
            } = event
            {
                totals[(digits / 5 - 3) as usize] += total;
            }
            ControlFlow::Continue(())
        })
        .unwrap();
        if factors.iter().all(|f| f.is_prime) {
            assert_eq!(factors, primes(&["941368429471859", M89]));
            found_by[worker as usize] = true;
        }
    }
    // The four shares add up to the 30 curves of the 15-digit level and the
    // 100 of the 20-digit level, which every worker reaches. Worker 3 has
    // the curve that finds p in a single run (sigma = 39); the others may
    // find it with curves of their own.
    assert_eq!(totals, [30, 100]);
    assert!(found_by[3]);
}

#[test]
fn unsplit_composite() {
    // Two 40-digit safe primes (q = 2r + 1, r prime): too big for rho, q − 1
    // has a huge prime factor so p−1 can't find them, and ECM at 20 digits
    // has no realistic chance.
    let n = "22402621473499677702312787220798894771372529738736380992481737381422651598174113";
    let factors = with_ecm(n, 20);
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

/// Two 25-digit primes, beyond what ECM tries at 49 digits.
const QS_SEMIPRIME: (&str, &str, &str) = (
    "9371289581805834728418039021175914244350477206171",
    "1424093917487413198662847",
    "6580527777507807889536293",
);

#[test]
fn needs_qs() {
    let (n, p, q) = QS_SEMIPRIME;
    let mut events = Vec::new();
    let factors = factor_str_with(n, &Config::default(), |event| {
        events.push(event);
        ControlFlow::Continue(())
    })
    .unwrap();
    assert_eq!(factors, primes(&[p, q]));
    // ECM stops at the 15-digit level (30% of 49 digits is below the
    // lowest level), and the sieve reports after every polynomial until it
    // has enough relations.
    let bits = n.parse::<Uint<8>>().unwrap().bits();
    assert!(events.contains(&Progress::Ecm {
        bits,
        digits: 15,
        done: 30,
        total: 30
    }));
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Progress::Ecm { digits: 20, .. }))
    );
    let Some(&Progress::Qs {
        relations, target, ..
    }) = events.last()
    else {
        panic!("last event {:?}", events.last());
    };
    assert!(
        relations >= target && target > 1000,
        "{relations} of {target}"
    );
    // Without the sieve, ECM gives up.
    assert!(!with_ecm(n, 20)[0].is_prime);
}

#[test]
fn stops_while_sieving() {
    let (n, ..) = QS_SEMIPRIME;
    let mut sieved = 0;
    let factors = factor_str_with(n, &Config::default(), |event| match event {
        Progress::Qs { .. } if sieved == 10 => ControlFlow::Break(()),
        Progress::Qs { .. } => {
            sieved += 1;
            ControlFlow::Continue(())
        }
        _ => ControlFlow::Continue(()),
    })
    .unwrap();
    assert_eq!(
        factors,
        [Factor {
            value: n.to_string(),
            is_prime: false
        }]
    );
}

#[test]
fn deferred_sieving() {
    // Defer leaves the composite for the caller, after the same short ECM.
    let (n, ..) = QS_SEMIPRIME;
    let config = Config::default().qs(QsMode::Defer);
    let mut ecm_levels = Vec::new();
    let factors = factor_str_with(n, &config, |event| {
        if let Progress::Ecm {
            digits, done: 0, ..
        } = event
        {
            ecm_levels.push(digits);
        }
        assert!(!matches!(event, Progress::Qs { .. }));
        ControlFlow::Continue(())
    })
    .unwrap();
    assert!(!factors[0].is_prime);
    assert_eq!(ecm_levels, [15]);
}
