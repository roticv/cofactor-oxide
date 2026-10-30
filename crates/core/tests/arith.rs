//! Checks `Uint` and `Monty` against `num-bigint` on random inputs.
//!
//! Run with `RUSTFLAGS="--cfg cofox_limb32"` as well to cover 32-bit limbs.

use cofox_core::arith::{LIMB_BITS, Limb, Monty, ParseUintError, Uint};
use num_bigint::BigUint;

const LIMB_BYTES: usize = (LIMB_BITS / 8) as usize;
const ROUNDS: usize = 2000;

/// Deterministic xorshift64* generator, so failures are reproducible.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn limb(&mut self) -> Limb {
        self.next_u64() as Limb
    }

    /// A random value with a random bit length, plus the occasional limb
    /// forced to 0 or all ones, to reach carry and borrow edge cases.
    fn uint<const L: usize>(&mut self) -> Uint<L> {
        let mut limbs = [0; L];
        for limb in &mut limbs {
            *limb = match self.next_u64() % 8 {
                0 => 0,
                1 => Limb::MAX,
                _ => self.limb(),
            };
        }
        let bits = (self.next_u64() % (Uint::<L>::BITS as u64 + 1)) as u32;
        let x = Uint::from_limbs(limbs);
        if bits == 0 {
            Uint::ZERO
        } else {
            x >> (Uint::<L>::BITS - bits)
        }
    }
}

fn to_big<const L: usize>(x: &Uint<L>) -> BigUint {
    let bytes: Vec<u8> = x.as_limbs().iter().flat_map(|l| l.to_le_bytes()).collect();
    BigUint::from_bytes_le(&bytes)
}

fn from_big<const L: usize>(x: &BigUint) -> Uint<L> {
    let mut bytes = x.to_bytes_le();
    assert!(bytes.len() <= L * LIMB_BYTES, "{x} does not fit");
    bytes.resize(L * LIMB_BYTES, 0);
    let mut limbs = [0; L];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks(LIMB_BYTES)) {
        *limb = Limb::from_le_bytes(chunk.try_into().unwrap());
    }
    Uint::from_limbs(limbs)
}

fn modulus_big<const L: usize>() -> BigUint {
    BigUint::from(1u8) << Uint::<L>::BITS
}

fn check_uint<const L: usize>(seed: u64) {
    let mut rng = Rng(seed);
    let r = modulus_big::<L>();
    for _ in 0..ROUNDS {
        let (a, b) = (rng.uint::<L>(), rng.uint::<L>());
        let (ab, bb) = (to_big(&a), to_big(&b));

        let (sum, carry) = a.overflowing_add(&b);
        assert_eq!(to_big(&sum), (&ab + &bb) % &r, "{a:?} + {b:?}");
        assert_eq!(carry, &ab + &bb >= r);

        let (diff, borrow) = a.overflowing_sub(&b);
        assert_eq!(to_big(&diff), (&r + &ab - &bb) % &r, "{a:?} - {b:?}");
        assert_eq!(borrow, ab < bb);

        assert_eq!(a.widening_square(), a.widening_mul(&a), "{a:?}²");

        let (lo, hi) = a.widening_mul(&b);
        assert_eq!(
            to_big(&lo) + (to_big(&hi) << Uint::<L>::BITS),
            &ab * &bb,
            "{a:?} * {b:?}"
        );

        assert_eq!(a.cmp(&b), ab.cmp(&bb));
        assert_eq!(a.bits() as u64, ab.bits());
        assert_eq!(to_big(&a.gcd(&b)), num_gcd(&ab, &bb), "gcd({a:?}, {b:?})");

        let shift = (rng.next_u64() % (Uint::<L>::BITS as u64 + 8)) as u32;
        assert_eq!(
            to_big(&(a << shift)),
            (&ab << shift) % &r,
            "{a:?} << {shift}"
        );
        assert_eq!(to_big(&(a >> shift)), &ab >> shift, "{a:?} >> {shift}");

        let d = rng.limb().max(1);
        let (q, rem) = a.div_rem_limb(d);
        assert_eq!(to_big(&q), &ab / d);
        assert_eq!(BigUint::from(rem), &ab % d);

        if !b.is_zero() {
            let (q, rem) = a.div_rem(&b);
            assert_eq!(to_big(&q), &ab / &bb, "{a:?} / {b:?}");
            assert_eq!(to_big(&rem), &ab % &bb, "{a:?} % {b:?}");
        }
        assert_eq!(to_big(&a.isqrt()), ab.sqrt(), "isqrt({a:?})");

        assert_eq!(a.to_string(), ab.to_string());
        assert_eq!(a.to_string().parse::<Uint<L>>(), Ok(a));
    }
}

fn num_gcd(a: &BigUint, b: &BigUint) -> BigUint {
    let (mut a, mut b) = (a.clone(), b.clone());
    while b != BigUint::ZERO {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a
}

fn check_monty<const L: usize>(seed: u64) {
    let mut rng = Rng(seed);
    for round in 0..ROUNDS / 10 {
        // Every few rounds use a full-width modulus, which exercises the
        // extra top limb in Montgomery multiplication.
        let mut limbs = *rng.uint::<L>().as_limbs();
        limbs[0] |= 1;
        if round % 3 == 0 {
            limbs[L - 1] |= 1 << (LIMB_BITS - 1);
        }
        let n = Uint::from_limbs(limbs);
        if n == Uint::ONE {
            continue;
        }
        let monty = Monty::new(n).unwrap();
        let nb = to_big(&n);

        for _ in 0..10 {
            let (a, b) = (rng.uint::<L>(), rng.uint::<L>());
            let (ab, bb) = (to_big(&a) % &nb, to_big(&b) % &nb);
            let (am, bm) = (monty.to_monty(&a), monty.to_monty(&b));

            assert_eq!(
                to_big(&monty.from_monty(&am)),
                ab,
                "round trip of {a:?} mod {n:?}"
            );
            assert_eq!(
                to_big(&monty.from_monty(&monty.mul(&am, &bm))),
                (&ab * &bb) % &nb
            );
            assert_eq!(
                to_big(&monty.from_monty(&monty.add(&am, &bm))),
                (&ab + &bb) % &nb
            );
            assert_eq!(
                to_big(&monty.from_monty(&monty.sub(&am, &bm))),
                (&nb + &ab - &bb) % &nb
            );

            let (lo, hi) = am.widening_mul(&bm);
            assert_eq!(monty.reduce(&lo, &hi), monty.mul(&am, &bm), "reduce");
            let (lo, hi) = am.widening_square();
            assert_eq!(monty.reduce(&lo, &hi), monty.square(&am), "reduce square");

            let g = num_gcd(&ab, &nb);
            match monty.inv(&am) {
                Ok(inv) => {
                    assert_eq!(g, BigUint::from(1u8), "inverted {a:?} mod {n:?}");
                    assert_eq!(monty.mul(&inv, &am), monty.one(), "inverse of {a:?}");
                }
                Err(e) => assert_eq!(to_big(&e), if ab == BigUint::ZERO { nb.clone() } else { g }),
            }

            let two = BigUint::from(2u8);
            assert_eq!((to_big(&monty.half(&am)) * &two) % &nb, to_big(&am), "half");
            assert_eq!(
                (to_big(&monty.neg(&am)) + to_big(&am)) % &nb,
                BigUint::ZERO,
                "neg"
            );

            let e = rng.uint::<2>();
            let expected = ab.modpow(&to_big(&e), &nb);
            assert_eq!(
                to_big(&monty.from_monty(&monty.pow(&am, &e))),
                expected,
                "{a:?}^{e:?} mod {n:?}"
            );
        }
    }
}

#[test]
fn uint_matches_bigint() {
    check_uint::<1>(1);
    check_uint::<2>(2);
    check_uint::<3>(3);
    check_uint::<5>(5);
    check_uint::<8>(8);
}

#[test]
fn monty_matches_bigint() {
    check_monty::<1>(11);
    check_monty::<2>(12);
    check_monty::<3>(13);
    check_monty::<5>(15);
    check_monty::<8>(18);
}

#[test]
fn inverse_reports_common_factors() {
    // n = 1000003 · 1000033; any multiple of either prime has no inverse,
    // and inv returns the shared factor.
    let p = 1_000_003u64;
    let q = 1_000_033u64;
    let monty = Monty::new(Uint::<2>::from_u64(p * q)).unwrap();
    for (a, gcd) in [(p, p), (3 * q, q), (p * 12345, p), (0, p * q)] {
        let am = monty.to_monty(&Uint::from_u64(a));
        assert_eq!(monty.inv(&am), Err(Uint::from_u64(gcd)), "a = {a}");
    }
    let am = monty.to_monty(&Uint::from_u64(2));
    let inv = monty.inv(&am).unwrap();
    assert_eq!(monty.from_monty(&inv), Uint::from_u64((p * q).div_ceil(2)));
}

#[test]
fn monty_rejects_bad_moduli() {
    assert!(Monty::new(Uint::<2>::ZERO).is_none());
    assert!(Monty::new(Uint::<2>::ONE).is_none());
    assert!(Monty::new(Uint::<2>::from_u64(10)).is_none());
    assert!(Monty::new(Uint::<2>::from_u64(3)).is_some());
}

#[test]
fn fermat_test_on_mersenne_prime() {
    // 2^127 - 1 is prime, so 3^(p-1) ≡ 1 (mod p).
    let p: Uint<4> = "170141183460469231731687303715884105727".parse().unwrap();
    let monty = Monty::new(p).unwrap();
    let three = monty.to_monty(&Uint::from_u64(3));
    let exp = p.wrapping_sub(&Uint::ONE);
    assert_eq!(monty.pow(&three, &exp), monty.one());
}

#[test]
fn parse_errors() {
    assert_eq!("".parse::<Uint<2>>(), Err(ParseUintError::Empty));
    assert_eq!("12a".parse::<Uint<2>>(), Err(ParseUintError::InvalidDigit));
    assert_eq!("-1".parse::<Uint<2>>(), Err(ParseUintError::InvalidDigit));
    let too_big = modulus_big::<2>().to_string();
    assert_eq!(too_big.parse::<Uint<2>>(), Err(ParseUintError::Overflow));
    let max = (modulus_big::<2>() - 1u8).to_string();
    assert_eq!(max.parse::<Uint<2>>(), Ok(Uint::MAX));
}

#[test]
fn u64_conversions() {
    for x in [0, 1, 0xFFFF_FFFF, 0x1_0000_0000, u64::MAX] {
        assert_eq!(Uint::<2>::from_u64(x).to_u64(), Some(x));
        assert_eq!(Uint::<2>::from_u64(x).to_string(), x.to_string());
    }
    assert_eq!((Uint::<4>::ONE << 64).to_u64(), None);
}

#[test]
fn display_padding() {
    let x: Uint<4> = "100000000000000000000000000000000000001".parse().unwrap();
    assert_eq!(x.to_string(), "100000000000000000000000000000000000001");
    assert_eq!(format!("{:>6}", Uint::<1>::from_u64(42)), "    42");
}

#[test]
fn bigint_helpers_round_trip() {
    let mut rng = Rng(99);
    for _ in 0..100 {
        let x = rng.uint::<3>();
        assert_eq!(from_big::<3>(&to_big(&x)), x);
    }
}
