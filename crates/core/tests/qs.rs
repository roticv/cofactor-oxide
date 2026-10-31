//! Tests for the quadratic sieve groundwork.
//!
//! Reference values for the multiplier and the factor base come from an
//! independent Python implementation of the same definitions, which uses its
//! own primes, logarithms and square roots (Cipolla's algorithm rather than
//! Tonelli–Shanks).

use cofox_core::Uint;
use cofox_core::primes::Primes;
use cofox_core::qs::{
    FactorBase, Params, Prime, QsJob, Sieve, Start, dependencies, multiplier, pow_mod, qs, sqrt_mod,
};

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
    assert_eq!(p(330), (97_647, 150, 4));
    assert_eq!(p(250), (28_352, 100, 1));
    // The large-prime multiplier goes down between 266 and 283 bits.
    assert_eq!(p(290), (57_333, 80, 3));
    assert_eq!(p(20), (100, 40, 1));
    assert_eq!(p(400), (210_000, 150, 4));
}

/// Semiprimes with two random primes of equal size, from Python.
const SEMIPRIMES: [(&str, &str, &str); 7] = [
    ("16676409402120693011", "2030509027", "8212920593"),
    ("493209351866533939609967", "535900657741", "920337276587"),
    (
        "728817225498169816759827770471",
        "848253368666219",
        "859197561035509",
    ),
    (
        "4813603616902910938815036056040719",
        "51891425250798533",
        "92762987211048643",
    ),
    (
        "396241995040211763643692640739477966561",
        "14864641071324530231",
        "26656680988053227431",
    ),
    (
        "9371289581805834728418039021175914244350477206171",
        "1424093917487413198662847",
        "6580527777507807889536293",
    ),
    (
        "95741717322359106137862766319375166032677068850856466452653",
        "266273800333898804119934270521",
        "359561162992010719067582390293",
    ),
];

#[test]
fn sieve_semiprimes() {
    for (n, p, q) in SEMIPRIMES {
        let n = u(n);
        let f = qs(&n, &Params::for_bits(n.bits())).expect("no factor");
        assert!(f == u(p) || f == u(q), "{n} gave {f}");
    }
}

#[test]
fn relations_hold() {
    // Every relation, full or partial, is an exact factorization of
    // t² − kN, across enough polynomials to need a second a.
    let n = u(SEMIPRIMES[4].0);
    let params = Params::for_bits(n.bits());
    let fb = FactorBase::new(&n, multiplier(&n), params.fb_size).unwrap();
    let mut sieve = Sieve::new(std::sync::Arc::new(fb.clone()), &params, 1);
    let (mut full, mut partial, mut a_values) = (0, 0, Vec::new());
    for _ in 0..40 {
        let (a, b_negative, b) = sieve.poly();
        // b² ≡ kN (mod a), so a divides every Q(x).
        let b2 = b.widening_square().0.div_rem(&a).1;
        assert_eq!(b2, fb.kn.div_rem(&a).1, "a = {a}, b = {b}");
        if !a_values.contains(&a) {
            a_values.push(a);
        }
        for r in sieve.sieve_poly() {
            let (sq, hi) = r.t.widening_square();
            assert!(hi.is_zero());
            let product = r.factors.iter().fold(U::from_u64(r.large), |acc, &i| {
                acc.widening_mul(&U::from_u64(fb.primes[i as usize].p.into()))
                    .0
            });
            let q = if r.negative {
                fb.kn.checked_sub(&sq)
            } else {
                sq.checked_sub(&fb.kn)
            };
            assert_eq!(q, Some(product), "t = {}, b negative: {b_negative}", r.t);
            assert!(r.factors.is_sorted());
            if r.large == 1 {
                full += 1;
            } else {
                assert!(r.large > u64::from(fb.primes.last().unwrap().p));
                partial += 1;
            }
        }
    }
    assert!(a_values.len() >= 2, "{a_values:?}");
    assert!(
        full > 20 && partial > full,
        "{full} full, {partial} partial"
    );
}

#[test]
fn dependencies_sum_to_zero() {
    // Random sparse rows, with repeated columns, which cancel.
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut next = |bound: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % bound as u64) as usize
    };
    for (rows, cols) in [(10usize, 4), (70, 60), (150, 130), (200, 200)] {
        let matrix: Vec<Vec<usize>> = (0..rows)
            .map(|_| (0..next(12)).map(|_| next(cols)).collect())
            .collect();
        let deps = dependencies(&matrix, cols);
        assert!(deps.len() >= rows.saturating_sub(cols), "{rows}×{cols}");
        for dep in &deps {
            assert!(!dep.is_empty() && dep.is_sorted());
            let mut sum = vec![false; cols];
            for &r in dep {
                for &c in &matrix[r] {
                    sum[c] ^= true;
                }
            }
            assert!(sum.iter().all(|&b| !b), "{rows}×{cols}: {dep:?}");
        }
    }
    // Large enough for block Lanczos, with small columns much denser than
    // large ones, as for primes in relations, and 32 more rows than
    // columns. Many of the sparse columns are singletons, which pruning
    // takes out with their rows.
    for (rows, cols) in [(1500usize, 1468), (6000, 5968)] {
        let matrix: Vec<Vec<usize>> = (0..rows)
            .map(|_| {
                (0..10 + next(20))
                    .map(|_| {
                        let c = next(cols);
                        c * c / cols
                    })
                    .collect()
            })
            .collect();
        let deps = dependencies(&matrix, cols);
        assert!(
            deps.len() >= 32,
            "{rows}×{cols}: {} dependencies",
            deps.len()
        );
        for dep in &deps {
            assert!(!dep.is_empty() && dep.is_sorted());
            let mut sum = vec![false; cols];
            for &r in dep {
                for &c in &matrix[r] {
                    sum[c] ^= true;
                }
            }
            assert!(sum.iter().all(|&b| !b), "{rows}×{cols}: {dep:?}");
        }
    }
    // An all-zero row is a dependency by itself, and a repeated row pairs
    // up with its copy.
    let deps = dependencies(&[vec![1, 2], vec![], vec![3, 3], vec![2, 1]], 4);
    assert_eq!(deps, [vec![1], vec![2], vec![0, 3]]);
}

#[test]
fn qs_edge_cases() {
    let params = Params::for_bits(64);
    // A prime has no proper factor to find.
    assert_eq!(qs(&u("1000000000000000003"), &params), None);
    // A small factor turns up while building the factor base.
    assert_eq!(
        qs(&u("401").widening_mul(&u(SEMIPRIMES[0].0)).0, &params),
        Some(u("401"))
    );
    assert_eq!(qs(&u("1000000014"), &params), Some(u("2")));
    // A square, with multiplier 1, so kN is a square too.
    let p = u("1000000007");
    let n = p.widening_square().0;
    assert_eq!(multiplier(&n), 1);
    assert_eq!(qs(&n, &params), Some(p));
}

/// About 14 minutes on one thread, so run with `--ignored` when changing
/// the sieve or block Lanczos.
#[test]
#[ignore]
fn eighty_digits() {
    let n = u("5783078982887739757788301949603111810137352482505406356698736447542686900836153");
    let f = qs(&n, &Params::for_bits(n.bits())).expect("no factor");
    let (p, q) = (
        u("1328777112197581296397318694224180349887"),
        u("4352181362699322384166653796609905670919"),
    );
    assert!(f == p || f == q, "{n} gave {f}");
}

#[test]
fn threads_share_the_sieving() {
    // Three sieves on their own threads, sending their relations as words
    // to one collector.
    let (n, p, q) = SEMIPRIMES[5];
    let Start::Sieve(job) = QsJob::start(n).unwrap() else {
        panic!("no sieving needed");
    };
    assert_eq!(job.bits(), u(n).bits());
    let mut collector = job.collector();
    let (send, receive) = std::sync::mpsc::channel();
    let done = std::sync::atomic::AtomicBool::new(false);
    let mut from = [0; 3];
    std::thread::scope(|scope| {
        for seed in 0..3 {
            let (send, job, done) = (send.clone(), &job, &done);
            scope.spawn(move || {
                let mut sieve = job.sieve(seed);
                while !done.load(std::sync::atomic::Ordering::Relaxed) {
                    if send.send((seed, sieve.sieve_poly())).is_err() {
                        break;
                    }
                }
            });
        }
        while !collector.is_complete() {
            let (seed, words) = receive.recv().unwrap();
            if !words.is_empty() {
                from[seed as usize] += 1;
            }
            collector.add(&words);
        }
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        drop(receive);
    });
    assert!(from.iter().all(|&batches| batches > 0), "{from:?}");
    let f = collector.finish().expect("no factor");
    assert!(f == p || f == q, "{n} gave {f}");
}

#[test]
fn jobs_that_need_no_sieving() {
    assert!(matches!(QsJob::start("1000000014").unwrap(), Start::Factor(f) if f == "2"));
    let n = u("401").widening_mul(&u(SEMIPRIMES[0].0)).0.to_string();
    assert!(matches!(QsJob::start(&n).unwrap(), Start::Factor(f) if f == "401"));
    assert!(QsJob::start("12x").is_err());
}

/// RSA-100 on all cores, sharing the sieving as the CLI does. About 2
/// hours on 10 cores.
#[test]
#[ignore]
fn rsa100() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let Start::Sieve(job) = QsJob::start(RSA100).unwrap() else {
        panic!("no sieving needed");
    };
    let mut collector = job.collector();
    let (send, receive) = std::sync::mpsc::channel();
    let done = AtomicBool::new(false);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    std::thread::scope(|scope| {
        for seed in 0..threads as u64 {
            let (send, job, done) = (send.clone(), &job, &done);
            scope.spawn(move || {
                let mut sieve = job.sieve(seed);
                while !done.load(Ordering::Relaxed) {
                    if send.send(sieve.sieve_poly()).is_err() {
                        break;
                    }
                }
            });
        }
        while !collector.is_complete() {
            collector.add(&receive.recv().unwrap());
        }
        done.store(true, Ordering::Relaxed);
    });
    let f = collector.finish().expect("no factor");
    let p = "37975227936943673922808872755445627854565536638199";
    let q = "40094690950920881030683735292761468389214899724061";
    assert!(f == p || f == q, "RSA-100 gave {f}");
}
