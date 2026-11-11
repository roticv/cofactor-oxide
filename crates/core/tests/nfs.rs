use cofox_core::Uint;
use cofox_core::nfs::{
    CHARACTER_PRIMES, FactorBase, Gnfs, Params, Poly, Relation, Sieve, alpha, degree, modp,
    parse_poly, special_qs,
};
use cofox_core::prime::is_prime;
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

fn eval_mod<C: modp::Coeff>(f: &[C], r: u64, p: u64) -> u64 {
    modp::eval(&modp::reduce(f, p), r, p)
}

fn derivative(f: &[BigInt]) -> Vec<BigInt> {
    f.iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| c * BigInt::from(i))
        .collect()
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
fn selected_polynomials_have_root_m() {
    let numbers = [
        "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139",
        "250782620106046385035853403635783676572083888259628917477091391671746381406504458761424731",
        "5783078982887739757788301949603111810137352482505406356698736447542686900836153",
    ];
    // At these sizes the search is long, so it runs on every core.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    for s in numbers {
        let n = u(s);
        let poly = Poly::select_with(&n, &Threads(threads)).unwrap();
        assert_eq!(poly.degree(), degree(n.bits()));
        // Translation and rotation keep F(m, l) = N, and the LLL reduction
        // in size optimization can multiply it by a small c ≠ 0.
        let (m, l) = (&poly.m, &poly.l);
        let value: BigInt = poly
            .f
            .iter()
            .enumerate()
            .map(|(i, c)| c * m.pow(i as u32) * l.pow((poly.degree() - i) as u32))
            .sum();
        let big_n = s.parse::<BigInt>().unwrap();
        let c = &value / &big_n;
        assert!(
            &c * &big_n == value && c != BigInt::from(0) && c.magnitude().bits() < 20,
            "F(m, l) isn't a small multiple of N for {poly:?}"
        );
        // Better by Murphy's E than plain base-m with leading coefficient 1.
        let params = Params::for_bits(n.bits());
        let plain = Poly::base_m(&n, &u(&m_plain(s, poly.degree())), poly.degree()).unwrap();
        assert!(
            poly.murphy_e(&params) > plain.murphy_e(&params),
            "{poly:?} worse than {plain:?}"
        );
    }
}

/// `⌊n^(1/d)⌋`.
fn m_plain(n: &str, d: usize) -> String {
    n.parse::<BigInt>().unwrap().nth_root(d as u32).to_string()
}

#[test]
fn alpha_matches_cado_nfs() {
    // CADO-NFS's polynomial for RSA-110, which it gives α −5.43 in natural
    // logarithms: −7.83 bits.
    let f: [i128; 6] = [
        -9739644201068194179961740,
        3798467775703585729918,
        172225589060678412,
        -11267352477454,
        -140707281,
        3420,
    ];
    let nat = alpha(&f) * std::f64::consts::LN_2;
    assert!((nat + 5.43).abs() < 0.005, "α {nat}");
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
    let poly = Poly::select(&n).unwrap();
    let fb = FactorBase::new(&poly, 100_000, 200_000, 40);
    assert_eq!(fb.rational.len(), Primes::new(2, 100_000).count());
    for q in &fb.rational {
        let p = BigInt::from(q.p);
        if q.projective() {
            assert_eq!(&poly.l % &p, BigInt::ZERO);
        } else {
            let g = (&poly.l * BigInt::from(q.r) - &poly.m) % &p;
            assert_eq!(g, BigInt::ZERO, "{q:?}");
        }
    }
    let lead = &poly.f[poly.degree()];
    for q in &fb.algebraic {
        let p = u64::from(q.p);
        assert!(p <= 200_000);
        if q.projective() {
            assert_eq!(lead % BigInt::from(q.p), BigInt::ZERO);
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
        assert!(c.q >= CHARACTER_PRIMES);
        assert_eq!(eval_mod(&poly.f, c.s, c.q), 0);
        // A simple root.
        assert_ne!(eval_mod(&derivative(&poly.f), c.s, c.q), 0);
    }
}

/// Checks that `rel` factors as it says on both sides, that `q` divides its
/// algebraic norm, and that each algebraic prime within the factor base is
/// one of its ideals: `(p, a/b mod p)` or, for `p | b`, the one at
/// infinity.
fn check_relation(poly: &Poly, fb: &FactorBase, q: u64, rel: &Relation) {
    let (a, b) = (BigInt::from(rel.a), BigInt::from(rel.b));
    let rational = &poly.l * &a - &b * &poly.m;
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
        .map(|(i, c)| c * a.pow(i as u32) * b.pow((d - i) as u32))
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
    let poly = Poly::select(&n).unwrap();
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

/// Factors `n` with GNFS on one thread using `poly`.
fn gnfs_with_poly(n: &Uint<16>, poly: Poly) -> Option<Uint<16>> {
    let gnfs = Gnfs::with_poly(n, &Params::for_bits(n.bits()), poly);
    let mut sieve = gnfs.sieve();
    let mut relations = gnfs.relations();
    'sieve: for k in 0.. {
        for (q, r) in gnfs.special_qs(k) {
            relations.add(sieve.special_q(q, r));
            if relations.is_ready() {
                break 'sieve;
            }
        }
    }
    relations
        .dependencies()
        .iter()
        .find_map(|dep| relations.try_dependency(&dep))
}

#[test]
fn gnfs_factors() {
    // 40 and 50 digits: with a base-m polynomial, so that g = x − m; with
    // the one selected, from Kleinjung's method, so that g = l·x − m with
    // l > 1; and with 3·f + g for that one, whose resultant with g is 3·N,
    // as size optimization's LLL reduction can give.
    let cases = [
        (
            "1176217249472123917871099760550263221639",
            "98936812917358800691",
            "11888570237800257629",
        ),
        (
            "59858698299029333151760743317766029830337454377407",
            "6005189888675219489059739",
            "9967827730462411250327213",
        ),
    ];
    for (s, p, q) in cases {
        let (n, p, q) = (u(s), u(p), u(q));
        let params = Params::for_bits(n.bits());
        let d = degree(n.bits());
        let selected = Poly::select(&n).unwrap();
        assert!(selected.l > BigInt::from(1));
        let mut tripled = selected.f.iter().map(|c| c * 3).collect::<Vec<_>>();
        tripled[0] -= &selected.m;
        tripled[1] += &selected.l;
        let text = format!(
            "n: {s}\n{}Y0: {}\nY1: {}\n",
            tripled
                .iter()
                .enumerate()
                .map(|(i, c)| format!("c{i}: {c}\n"))
                .collect::<String>(),
            -&selected.m,
            selected.l
        );
        let polys = [
            ("base-m", Poly::base_m(&n, &u(&m_plain(s, d)), d).unwrap()),
            ("selected", selected.clone()),
            ("3·f + g", parse_poly(&n, &text, &params).unwrap()),
        ];
        for (name, poly) in polys {
            let t = std::time::Instant::now();
            let f = gnfs_with_poly(&n, poly).expect("no factor");
            println!("{} bits, {name}: {f} in {:.2?}", n.bits(), t.elapsed());
            assert!(f == p || f == q, "{n} gave {f} with {name}");
        }
    }
}

/// Work on this many scoped threads.
struct Threads(usize);

impl cofox_core::parallel::Parallel for Threads {
    fn threads(&self) -> usize {
        self.0
    }

    fn map<T: Send>(&self, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        let f = &f;
        std::thread::scope(|scope| {
            let handles: Vec<_> = (1..self.0).map(|i| scope.spawn(move || f(i))).collect();
            let first = f(0);
            std::iter::once(first)
                .chain(handles.into_iter().map(|h| h.join().unwrap()))
                .collect()
        })
    }
}

/// Work split as for this many threads, run on this one.
struct Pieces(usize);

impl cofox_core::parallel::Parallel for Pieces {
    fn threads(&self) -> usize {
        self.0
    }

    fn map<T: Send>(&self, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        (0..self.0).map(f).collect()
    }
}

#[test]
fn gnfs_factors_in_pieces() {
    // The matrix and the square roots, split as for three threads.
    let (n, p, q) = (
        u("1176217249472123917871099760550263221639"),
        u("98936812917358800691"),
        u("11888570237800257629"),
    );
    let gnfs = cofox_core::nfs::Gnfs::new(&n, &Params::for_bits(n.bits())).unwrap();
    let mut sieve = gnfs.sieve();
    let mut relations = gnfs.relations();
    for k in 0.. {
        for (q, r) in gnfs.special_qs(k) {
            relations.add(sieve.special_q(q, r));
        }
        if relations.is_ready() {
            break;
        }
    }
    let f = relations
        .dependencies_with(&Pieces(3))
        .iter()
        .find_map(|dep| relations.try_dependency_with(&dep, &Pieces(3)))
        .expect("no factor");
    assert!(f == p || f == q, "{n} gave {f}");
}

/// Keeps the relations in a vector, as the command line keeps them in a
/// file.
#[derive(Default)]
struct VecSpill {
    words: Vec<Vec<u32>>,
    fail_after: Option<usize>,
}

impl cofox_core::nfs::Spill for VecSpill {
    fn write(&mut self, words: &[u32]) -> bool {
        if self.fail_after == Some(self.words.len()) {
            return false;
        }
        self.words.push(words.to_vec());
        true
    }

    fn read(&mut self, f: &mut dyn FnMut(&[u32])) {
        self.words.iter().for_each(|w| f(w));
    }
}

#[test]
fn gnfs_factors_with_relations_spilled() {
    // The relations leave memory while the matrix is solved, and come
    // back, in order, for the square roots; a spill that fails part way
    // leaves them in memory.
    let (n, p, q) = (
        u("1176217249472123917871099760550263221639"),
        u("98936812917358800691"),
        u("11888570237800257629"),
    );
    let gnfs = cofox_core::nfs::Gnfs::new(&n, &Params::for_bits(n.bits())).unwrap();
    let mut sieve = gnfs.sieve();
    let mut relations = gnfs.relations();
    for k in 0.. {
        for (q, r) in gnfs.special_qs(k) {
            relations.add(sieve.special_q(q, r));
        }
        if relations.is_ready() {
            break;
        }
    }
    let len = relations.len();
    for fail_after in [None, Some(len / 2)] {
        let mut spill = VecSpill {
            fail_after,
            ..VecSpill::default()
        };
        let deps = relations.dependencies_spilling(&Pieces(3), &mut spill);
        assert_eq!(relations.len(), len);
        assert_eq!(spill.words.len(), fail_after.unwrap_or(len));
        let f = deps
            .iter()
            .find_map(|dep| relations.try_dependency_with(&dep, &Pieces(3)))
            .expect("no factor");
        assert!(f == p || f == q, "{n} gave {f}");
    }
}

#[test]
fn relations_as_text() {
    use cofox_core::nfs::{parse_relation, relation_line};
    // 50 digits, so that l > 1.
    let n = u("59858698299029333151760743317766029830337454377407");
    let poly = Poly::select(&n).unwrap();
    let params = Params::for_bits(n.bits());
    let fb = FactorBase::new(&poly, params.rational_bound, params.algebraic_bound, 0);
    let mut sieve = Sieve::new(&poly, &fb, &params);
    let start = u64::from(params.algebraic_bound) + 1;
    let rels: Vec<Relation> = special_qs(&poly, start, start + 1000)
        .into_iter()
        .take(10)
        .flat_map(|(q, r)| sieve.special_q(q, r))
        .collect();
    assert!(rels.len() > 20, "{} relations", rels.len());
    let sorted = |mut v: Vec<u64>| {
        v.sort_unstable();
        v
    };
    for rel in &rels {
        let line = relation_line(rel);
        // Read back, and with the primes below 1000 left out.
        let small = |list: &str| {
            list.split(',')
                .filter(|p| u64::from_str_radix(p, 16).unwrap() >= 1000)
                .collect::<Vec<_>>()
                .join(",")
        };
        let parts: Vec<&str> = line.split(':').collect();
        let short = format!("{}:{}:{}", parts[0], small(parts[1]), small(parts[2]));
        for text in [&line, &short] {
            let back = parse_relation(&poly, text).unwrap_or_else(|| panic!("{text}"));
            assert_eq!(
                (back.a, back.b, back.negative),
                (rel.a, rel.b, rel.negative)
            );
            assert_eq!(
                sorted(back.rational),
                sorted(rel.rational.clone()),
                "{text}"
            );
            assert_eq!(
                sorted(back.algebraic),
                sorted(rel.algebraic.clone()),
                "{text}"
            );
        }
        // Another b, or a prime too many or missing, isn't a relation.
        let wrong_b = format!("{},{}:{}:{}", rel.a, rel.b + 1, parts[1], parts[2]);
        assert!(parse_relation(&poly, &wrong_b).is_none(), "{wrong_b}");
        let extra = format!("{line},3fd");
        assert!(parse_relation(&poly, &extra).is_none(), "{extra}");
        // The relation with its largest algebraic prime dropped.
        let mut algebraic = rel.algebraic.clone();
        algebraic.sort_unstable();
        if algebraic.pop().is_some_and(|p| p >= 1000) {
            let missing = relation_line(&Relation {
                algebraic,
                ..rel.clone()
            });
            assert!(parse_relation(&poly, &missing).is_none(), "{missing}");
        }
    }
    assert!(parse_relation(&poly, "garbage").is_none());
    assert!(parse_relation(&poly, "1,0::").is_none());
}

#[test]
fn polynomials_as_text() {
    use cofox_core::nfs::{parse_poly, poly_text};
    let n = u("59858698299029333151760743317766029830337454377407");
    let params = Params::for_bits(n.bits());
    let poly = Poly::select(&n).unwrap();
    let text = poly_text(&n, &poly);
    let back = parse_poly(&n, &text, &params).expect(&text);
    assert_eq!((&back.f, &back.m, &back.l), (&poly.f, &poly.m, &poly.l));
    assert!(
        (back.e - poly.e).abs() <= 1e-9 * poly.e,
        "{} vs {}",
        back.e,
        poly.e
    );
    // msieve's .fb keys.
    let mut fb = format!("N {n}\nSKEW {}\nR0 -{}\nR1 {}\n", poly.skew, poly.m, poly.l);
    for (i, c) in poly.f.iter().enumerate() {
        fb += &format!("A{i} {c}\n");
    }
    let back = parse_poly(&n, &fb, &params).expect(&fb);
    assert_eq!((&back.f, &back.m, &back.l), (&poly.f, &poly.m, &poly.l));
    // Another number, or a g without the common root.
    assert!(
        parse_poly(
            &u("1176217249472123917871099760550263221639"),
            &text,
            &params
        )
        .is_none()
    );
    let shifted = text.replace(
        &format!("Y0: -{}", poly.m),
        &format!("Y0: -{}", &poly.m + 1),
    );
    assert!(parse_poly(&n, &shifted, &params).is_none());
}

/// Polynomials with coefficients past 128 bits, as for numbers past
/// about 480 bits: `f = (K + 1) + K·x − K·x² − K·x³ + 3·x⁴ + x⁵` and
/// `g = (K + 3)·x − (K − 3)` for `K = 3^89`, about 2^141, with `N = F(m, l)`.
/// They make `F(1, 1) = 5`, `G(1, 1) = 6`, `F(−1, 1) = 3` and
/// `G(−1, 1) = −2·3^89`, so relations there are known.
#[test]
fn wide_polynomials() {
    use cofox_core::nfs::{parse_poly, parse_relation, poly_text};
    let k = BigInt::from(3).pow(89);
    let f: [BigInt; 6] = [&k + 1, k.clone(), -&k, -&k, 3.into(), 1.into()];
    let (m, l): (BigInt, BigInt) = (&k - 3, &k + 3);
    let n: BigInt = f
        .iter()
        .enumerate()
        .map(|(i, c)| c * m.pow(i as u32) * l.pow(5 - i as u32))
        .sum();
    // Past 512 bits, which Uint<16> holds with 32-bit limbs.
    let n: Uint<32> = n.magnitude().to_string().parse().unwrap();
    let mut text = format!("n: {n}\nY0: -{m}\nY1: {l}\n");
    for (i, c) in f.iter().enumerate() {
        text += &format!("c{i}: {c}\n");
    }
    let params = Params::for_bits(n.bits());
    let poly = parse_poly(&n, &text, &params).expect(&text);
    assert_eq!((&poly.f[..], &poly.m, &poly.l), (&f[..], &m, &l));
    let back = parse_poly(&n, &poly_text(&n, &poly), &params).unwrap();
    assert_eq!((&back.f, &back.m, &back.l), (&poly.f, &poly.m, &poly.l));

    // The primes below 1000 left out are found again.
    let rel = parse_relation(&poly, "1,1:2,3:5").unwrap();
    assert_eq!(
        (rel.negative, rel.rational, rel.algebraic),
        (false, vec![2, 3], vec![5])
    );
    let rel = parse_relation(&poly, "-1,1::").unwrap();
    let mut rational = vec![2];
    rational.extend([3; 89]);
    assert_eq!(
        (rel.negative, rel.rational, rel.algebraic),
        (true, rational, vec![3])
    );
    assert!(parse_relation(&poly, "1,1:2,3:7").is_none());

    // The factor bases' roots, reduced from the wide coefficients.
    let fb = FactorBase::new(&poly, 2_000, 2_000, 8);
    let value = |c: &[BigInt], r: u64| -> BigInt {
        c.iter()
            .enumerate()
            .map(|(i, c)| c * BigInt::from(r).pow(i as u32))
            .sum()
    };
    for q in &fb.rational {
        let p = BigInt::from(q.p);
        let g = if q.projective() {
            l.clone()
        } else {
            &l * q.r - &m
        };
        assert_eq!(g % &p, BigInt::ZERO, "{q:?}");
    }
    for q in &fb.algebraic {
        let p = BigInt::from(q.p);
        let v = if q.projective() {
            f[5].clone()
        } else {
            value(&f, q.r.into())
        };
        assert_eq!(v % &p, BigInt::ZERO, "{q:?}");
    }
    for c in &fb.characters {
        assert_eq!(value(&f, c.s) % BigInt::from(c.q), BigInt::ZERO, "{c:?}");
    }

    // Sieving with them runs, and whatever it finds holds.
    let mut sieve = Sieve::new(&poly, &fb, &params);
    for (q, r) in special_qs(&poly, 3000, 3100) {
        for rel in sieve.special_q(q, r) {
            check_relation(&poly, &fb, q, &rel);
        }
    }
}

/// With a small factor base and a cofactor bound of three large primes,
/// some relations have three primes past the factor base on the algebraic
/// side, besides the special-q, and they all hold.
#[test]
fn three_large_primes() {
    let n = u("336824692466468372742240828730960315277872725533762205050251");
    let poly = Poly::select(&n).unwrap();
    let params = Params {
        rational_bound: 20_000,
        algebraic_bound: 20_000,
        rational_lpb: 24,
        algebraic_lpb: 24,
        rational_mfb: 48,
        algebraic_mfb: 72,
        log_i: 9,
        ..Params::for_bits(n.bits())
    };
    let fb = FactorBase::new(&poly, params.rational_bound, params.algebraic_bound, 0);
    let mut sieve = Sieve::new(&poly, &fb, &params);
    let (mut found, mut three) = (0, 0);
    let t = std::time::Instant::now();
    for (q, r) in special_qs(&poly, 30_000, 30_100).into_iter().take(3) {
        for rel in sieve.special_q(q, r) {
            check_relation(&poly, &fb, q, &rel);
            found += 1;
            let large = rel.algebraic.iter().filter(|&&p| p > 20_000).count();
            if large > 3 {
                three += 1;
            }
        }
    }
    println!(
        "{found} relations, {three} with three large primes, in {:.2?}",
        t.elapsed()
    );
    assert!(three > 0);
}

/// Factoring with up to three large primes on the algebraic side.
#[test]
fn gnfs_factors_with_three_large_primes() {
    use cofox_core::nfs::gnfs_with;
    let n = u("1176217249472123917871099760550263221639");
    let params = Params {
        rational_bound: 3_000,
        algebraic_bound: 3_000,
        rational_lpb: 15,
        algebraic_lpb: 15,
        rational_mfb: 15,
        algebraic_mfb: 45,
        ..Params::for_bits(n.bits())
    };
    let t = std::time::Instant::now();
    let f = gnfs_with(&n, &params, |_, _| std::ops::ControlFlow::Continue(()))
        .unwrap()
        .expect("no factor");
    println!("{f} in {:.2?}", t.elapsed());
    assert!(f == u("98936812917358800691") || f == u("11888570237800257629"));
}

/// With large primes past 2^32, and special-q past 2^32 too, relations
/// hold and their primes past the factor base are prime and below the
/// bound.
#[test]
fn large_primes_past_32_bits() {
    let n = u("336824692466468372742240828730960315277872725533762205050251");
    let poly = Poly::select(&n).unwrap();
    let params = Params {
        rational_bound: 100_000,
        algebraic_bound: 100_000,
        rational_lpb: 33,
        algebraic_lpb: 33,
        rational_mfb: 66,
        algebraic_mfb: 99,
        log_i: 9,
        ..Params::for_bits(n.bits())
    };
    let fb = FactorBase::new(&poly, params.rational_bound, params.algebraic_bound, 0);
    let mut sieve = Sieve::new(&poly, &fb, &params);
    let (mut found, mut wide) = (0, 0);
    let t = std::time::Instant::now();
    for start in [1 << 20, 1 << 32] {
        for (q, r) in special_qs(&poly, start, start + 1000).into_iter().take(1) {
            for rel in sieve.special_q(q, r) {
                check_relation(&poly, &fb, q, &rel);
                found += 1;
                let others = rel.rational.iter().chain(&rel.algebraic);
                for &p in others.filter(|&&p| p > 100_000 && p != q) {
                    assert!(p < 1 << 33, "{p} in {rel:?}");
                    assert!(is_prime(&Uint::<2>::from_u64(p)), "{p} in {rel:?}");
                    wide += usize::from(p >> 32 != 0);
                }
            }
        }
    }
    println!(
        "{found} relations, {wide} primes past 2^32, in {:.2?}",
        t.elapsed()
    );
    assert!(wide > 0);
}

/// Factoring from special-q past 2^32, so that every relation has a
/// prime that wide, to be stored, put in the matrix and taken the square
/// root of.
#[test]
fn gnfs_factors_with_special_q_past_32_bits() {
    use cofox_core::nfs::gnfs_with;
    let n = u("1176217249472123917871099760550263221639");
    let params = Params {
        rational_bound: 3_000,
        algebraic_bound: 3_000,
        rational_lpb: 16,
        algebraic_lpb: 16,
        rational_mfb: 32,
        algebraic_mfb: 32,
        qmin: 1 << 32,
        ..Params::for_bits(n.bits())
    };
    let t = std::time::Instant::now();
    let f = gnfs_with(&n, &params, |_, _| std::ops::ControlFlow::Continue(()))
        .unwrap()
        .expect("no factor");
    println!("{f} in {:.2?}", t.elapsed());
    assert!(f == u("98936812917358800691") || f == u("11888570237800257629"));
}
