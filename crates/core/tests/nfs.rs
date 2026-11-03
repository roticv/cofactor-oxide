use cofox_core::Uint;
use cofox_core::nfs::{FactorBase, Params, Poly, Relation, Sieve, alpha, degree, modp, special_qs};
use cofox_core::primes::Primes;
use num_bigint::BigInt;

fn u(s: &str) -> Uint<16> {
    s.parse().unwrap()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn eval_mod(f: &[i128], r: u64, p: u64) -> u64 {
    modp::eval(&modp::reduce(f, p), r, p)
}

#[test]
fn roots_match_brute_force() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for _ in 0..200 {
        let d = 3 + (rng.next() % 3) as usize;
        let f: Vec<i128> = (0..=d)
            .map(|_| i128::from(rng.next() as i64) >> 10)
            .collect();
        for p in Primes::new(2, 3000) {
            if modp::reduce(&f, p).is_empty() {
                continue;
            }
            let want: Vec<u64> = (0..p).filter(|&r| eval_mod(&f, r, p) == 0).collect();
            assert_eq!(modp::roots(&f, p), want, "f = {f:?}, p = {p}");
        }
    }
}

#[test]
fn roots_of_products_of_linear_factors() {
    // f = (x − r₁)…(x − r₅) has exactly those roots modulo a large prime.
    let mut rng = Rng(0xfeed_beef_dead_cafe);
    for p in [1_000_003u64, 4_294_967_291] {
        for _ in 0..50 {
            let mut want: Vec<u64> = (0..5).map(|_| rng.next() % 1000).collect();
            let mut f = vec![1i128];
            for &r in &want {
                let mut g = vec![0i128; f.len() + 1];
                for (i, &c) in f.iter().enumerate() {
                    g[i + 1] += c;
                    g[i] -= c * r as i128;
                }
                f = g;
            }
            want.sort_unstable();
            want.dedup();
            assert_eq!(modp::roots(&f, p), want, "p = {p}");
        }
    }
}

#[test]
fn roots_of_irreducible_and_degenerate_polynomials() {
    // x² + 1 has no roots mod 3 (mod 4), two mod 5 (1 mod 4).
    assert!(modp::roots(&[1, 0, 1], 1_000_003).is_empty());
    assert_eq!(modp::roots(&[1, 0, 1], 1_000_033).len(), 2);
    // Constant or zero mod p: no roots reported.
    assert!(modp::roots(&[7], 1_000_003).is_empty());
    assert!(modp::roots(&[1_000_003, 2 * 1_000_003], 1_000_003).is_empty());
    // A double root counts once.
    assert_eq!(modp::roots(&[1, -2, 1], 1_000_003), vec![1]);
}

#[test]
fn base_m_writes_n_in_base_m() {
    let numbers = [
        "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139",
        "250782620106046385035853403635783676572083888259628917477091391671746381406504458761424731",
        "5783078982887739757788301949603111810137352482505406356698736447542686900836153",
        "35794234179725868774991807832568455403003778024228226193532908190484670252364677411513516111204504060317568667",
    ];
    for s in numbers {
        let n = u(s);
        let poly = Poly::select(&n);
        assert_eq!(poly.degree(), degree(n.bits()));
        let m = BigInt::from(poly.m);
        let value: BigInt = poly
            .f
            .iter()
            .rev()
            .fold(BigInt::from(0), |acc, &c| acc * &m + BigInt::from(c));
        assert_eq!(value, s.parse::<BigInt>().unwrap(), "f(m) ≠ N for {poly:?}");
        let half = (poly.m / 2 + 1) as i128;
        for &c in &poly.f[..poly.degree()] {
            assert!(c.abs() <= half, "{c} not centered in {poly:?}");
        }
        // At least as good as plain base-m, with leading coefficient 1.
        let plain = Poly::base_m(&n, &u(&m_plain(s, poly.degree())), poly.degree()).unwrap();
        let score = |p: &Poly| p.size + alpha(&p.f);
        assert!(
            score(&poly) <= score(&plain),
            "{poly:?} worse than {plain:?}"
        );
    }
}

/// `⌊n^(1/d)⌋`.
fn m_plain(n: &str, d: usize) -> String {
    n.parse::<BigInt>().unwrap().nth_root(d as u32).to_string()
}

#[test]
fn alpha_rewards_roots() {
    // x³ − x has every residue mod 2 and 3 as a root; x³ + x + 1 has none
    // mod 2 and 3.
    assert!(alpha(&[0, -1, 0, 1]) < alpha(&[1, 1, 0, 1]));
}

#[test]
fn factor_base_primes_divide() {
    let n = u(
        "250782620106046385035853403635783676572083888259628917477091391671746381406504458761424731",
    );
    let poly = Poly::select(&n);
    let fb = FactorBase::new(&poly, 100_000, 200_000, 40);
    assert_eq!(fb.rational.len(), Primes::new(2, 100_000).count());
    for q in &fb.rational {
        assert_eq!(u128::from(q.r), poly.m % u128::from(q.p));
    }
    let lead = poly.f[poly.degree()];
    for q in &fb.algebraic {
        let p = u64::from(q.p);
        assert!(p <= 200_000);
        if q.projective() {
            assert_eq!(lead.rem_euclid(i128::from(q.p)), 0);
        } else {
            assert_eq!(eval_mod(&poly.f, q.r.into(), p), 0, "{q:?}");
        }
    }
    // About one root per prime on average, for an irreducible f.
    let primes = Primes::new(2, 200_000).count() as f64;
    let ratio = fb.algebraic.len() as f64 / primes;
    assert!((0.8..1.2).contains(&ratio), "{ratio}");
    assert_eq!(fb.characters.len(), 40);
    for c in &fb.characters {
        assert!(c.p > 200_000);
        assert_eq!(eval_mod(&poly.f, c.r.into(), c.p.into()), 0);
    }
}

/// Checks that `rel` factors as it says on both sides, that `q` divides its
/// algebraic norm, and that each algebraic prime within the factor base is
/// one of its ideals: `(p, a/b mod p)` or, for `p | b`, the one at
/// infinity.
fn check_relation(poly: &Poly, fb: &FactorBase, q: u64, rel: &Relation) {
    let (a, b) = (BigInt::from(rel.a), BigInt::from(rel.b));
    let rational = &a - &b * BigInt::from(poly.m);
    assert_eq!(rational < BigInt::from(0), rel.negative, "{rel:?}");
    let product = |ps: &[u64]| {
        ps.iter()
            .fold(BigInt::from(1), |acc, &p| acc * BigInt::from(p))
    };
    assert_eq!(
        product(&rel.rational),
        rational.magnitude().clone().into(),
        "{rel:?}"
    );
    let d = poly.degree();
    let norm: BigInt = poly
        .f
        .iter()
        .enumerate()
        .map(|(i, &c)| BigInt::from(c) * a.pow(i as u32) * b.pow((d - i) as u32))
        .sum();
    assert_eq!(
        product(&rel.algebraic),
        norm.magnitude().clone().into(),
        "{rel:?}"
    );
    assert!(rel.algebraic.contains(&q), "{q} missing from {rel:?}");
    let largest = u64::from(fb.algebraic.last().unwrap().p);
    for &p in rel.algebraic.iter().filter(|&&p| p <= largest) {
        let ideal = if rel.b.is_multiple_of(p) {
            p
        } else {
            let binv = modp_inv(rel.b % p, p);
            (rel.a.rem_euclid(p as i64) as u64) * binv % p
        };
        assert!(
            fb.algebraic
                .iter()
                .any(|x| u64::from(x.p) == p && u64::from(x.r) == ideal),
            "({p}, {ideal}) not in the factor base, {rel:?}"
        );
    }
}

fn modp_inv(a: u64, p: u64) -> u64 {
    let (mut r, mut b, mut e) = (1u64, a % p, p - 2);
    while e > 0 {
        if e & 1 == 1 {
            r = (r as u128 * b as u128 % p as u128) as u64;
        }
        b = (b as u128 * b as u128 % p as u128) as u64;
        e >>= 1;
    }
    r
}

/// Sieves the first `count` special-q for `n` and checks every relation
/// found.
fn sieve_special_qs(n: &str, count: usize) -> usize {
    let n = u(n);
    let poly = Poly::select(&n);
    let params = Params::for_bits(n.bits());
    let fb = FactorBase::new(&poly, params.rational_bound, params.algebraic_bound, 0);
    let mut sieve = Sieve::new(&poly, &fb, &params);
    let start = u64::from(params.algebraic_bound) + 1;
    let qs: Vec<(u64, u64)> = special_qs(&poly, start, start + 10_000)
        .into_iter()
        .take(count)
        .collect();
    let t = std::time::Instant::now();
    let mut found = 0;
    for &(q, r) in &qs {
        for rel in sieve.special_q(q, r) {
            check_relation(&poly, &fb, q, &rel);
            found += 1;
        }
    }
    println!(
        "{} bits, {params:?}: {found} relations from {count} special-q in {:.2?}, factor bases {} + {}",
        n.bits(),
        t.elapsed(),
        fb.rational.len(),
        fb.algebraic.len()
    );
    found
}

#[test]
fn relations_hold() {
    // 40 and 60 digits.
    assert!(sieve_special_qs("1176217249472123917871099760550263221639", 20) > 0);
    assert!(
        sieve_special_qs(
            "336824692466468372742240828730960315277872725533762205050251",
            4
        ) > 0
    );
}

#[test]
fn gnfs_factors() {
    // 40 digits, two 20-digit primes.
    let (n, p, q) = (
        u("1176217249472123917871099760550263221639"),
        u("98936812917358800691"),
        u("11888570237800257629"),
    );
    let t = std::time::Instant::now();
    let f = cofox_core::nfs::gnfs(&n).expect("no factor");
    println!("{} bits: {f} in {:.2?}", n.bits(), t.elapsed());
    assert!(f == p || f == q, "{n} gave {f}");
}
