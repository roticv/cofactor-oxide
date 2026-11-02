//! The square roots of a dependency on both sides, and the factor they
//! give. This is the only module that needs integers larger than `Uint`:
//! the algebraic square root has coefficients of hundreds of thousands of
//! bits. It uses `num-bigint`, which stays inside this module.
//!
//! With `c` the leading coefficient of `f`, `θ = c·α` is a root of the
//! monic `F(x) = c^(d−1)·f(x/c)`, so `ℤ[θ]` is in the ring of integers.
//! For a dependency of `k` relations (`k` is even),
//! `δ = ∏ (c·a − b·θ) · F′(θ)²` is a square in `ℤ[θ]`, and `S = √δ` is
//! found modulo a prime `p` that stays prime in `ℤ[θ]`, then lifted by
//! Newton's method modulo `p^(2^i)` until its coefficients are certain.
//! Mapping `θ` to `c·m` gives `X = S(c·m)`, and on the rational side
//! `Y = c^(k/2)·F′(c·m)·√∏ (a − b·m)`, with `X² ≡ Y² (mod N)`.

use alloc::collections::BTreeMap;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use super::modp::{irreducible, mul_mod};
use super::poly::Poly;
use super::sieve::Relation;
use crate::arith::Uint;
use crate::primes::Primes;

/// The inert prime is searched for in `[2^19, 2^20)`, small enough that
/// `p^d` fits in a `u128` up to degree 6.
const INERT_PRIMES: (u64, u64) = (1 << 19, 1 << 20);

/// Bits of margin on the coefficient bound before the first check.
const MARGIN: u64 = 64;

/// Times the bound may double after a failed check before giving up.
const RETRIES: u32 = 3;

/// `x mod m` in `[0, m)`.
fn modulo(x: &BigInt, m: &BigInt) -> BigInt {
    let r = x % m;
    if r.is_negative() { r + m } else { r }
}

/// `a·b mod F` for the monic `F`, with coefficients lowest first.
fn mul(a: &[BigInt], b: &[BigInt], f: &[BigInt]) -> Vec<BigInt> {
    let d = f.len() - 1;
    let mut c = vec![BigInt::zero(); 2 * d - 1];
    for (i, x) in a.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            c[i + j] += x * y;
        }
    }
    for k in (d..2 * d - 1).rev() {
        let t = core::mem::take(&mut c[k]);
        if !t.is_zero() {
            for (j, fj) in f[..d].iter().enumerate() {
                c[k - d + j] -= &t * fj;
            }
        }
    }
    c.truncate(d);
    c
}

/// `a·b mod (F, m)`.
fn mul_mod_m(a: &[BigInt], b: &[BigInt], f: &[BigInt], m: &BigInt) -> Vec<BigInt> {
    mul(a, b, f).iter().map(|c| modulo(c, m)).collect()
}

/// The product of `elems` mod `F`, multiplying in pairs so that the
/// operands stay balanced.
fn product(mut elems: Vec<Vec<BigInt>>, f: &[BigInt]) -> Vec<BigInt> {
    while elems.len() > 1 {
        elems = elems
            .chunks(2)
            .map(|pair| match pair {
                [x, y] => mul(x, y, f),
                [x] => x.clone(),
                _ => unreachable!(),
            })
            .collect();
    }
    elems.pop().unwrap()
}

/// `x^e` in `𝔽_p[x]/(f)`.
fn field_pow(x: &[u64], mut e: u128, f: &[u64], p: u64) -> Vec<u64> {
    let mut b = x.to_vec();
    let mut r = vec![1];
    while e > 0 {
        if e & 1 == 1 {
            r = mul_mod(&r, &b, f, p);
        }
        b = mul_mod(&b, &b, f, p);
        e >>= 1;
    }
    r
}

/// A square root of `a` in the field `𝔽_p[x]/(f)` of `q` elements, by
/// Tonelli–Shanks, or `None` if `a` isn't a square.
fn field_sqrt(a: &[u64], f: &[u64], p: u64, q: u128) -> Option<Vec<u64>> {
    let one = vec![1u64];
    if field_pow(a, (q - 1) / 2, f, p) != one {
        return None;
    }
    let s = (q - 1).trailing_zeros();
    let t = (q - 1) >> s;
    let minus_one = vec![p - 1];
    let z = (1..)
        .map(|c| vec![c, 1])
        .find(|z| field_pow(z, (q - 1) / 2, f, p) == minus_one)?;
    let mut m = s;
    let mut c = field_pow(&z, t, f, p);
    let mut tt = field_pow(a, t, f, p);
    let mut r = field_pow(a, t.div_ceil(2), f, p);
    while tt != one {
        let mut i = 0;
        let mut t2 = tt.clone();
        while t2 != one {
            t2 = mul_mod(&t2, &t2, f, p);
            i += 1;
        }
        if i >= m {
            return None;
        }
        let mut b = c.clone();
        for _ in 0..m - i - 1 {
            b = mul_mod(&b, &b, f, p);
        }
        m = i;
        c = mul_mod(&b, &b, f, p);
        tt = mul_mod(&tt, &c, f, p);
        r = mul_mod(&r, &b, f, p);
    }
    Some(r)
}

/// `Σ cᵢ·xⁱ mod n`.
fn eval_mod(c: &[BigInt], x: &BigInt, n: &BigInt) -> BigInt {
    c.iter()
        .rev()
        .fold(BigInt::zero(), |acc, ci| modulo(&(acc * x + ci), n))
}

/// The bits of the largest coefficient.
fn max_bits(c: &[BigInt]) -> u64 {
    c.iter().map(|x| x.bits()).max().unwrap_or(0)
}

/// The square root of `δ` in `ℤ[θ]/(F)`, if it is a square there.
fn algebraic_sqrt(delta: &[BigInt], f: &[BigInt], lead: &BigInt) -> Option<Vec<BigInt>> {
    let d = f.len() - 1;
    // A prime that stays prime: F irreducible mod p.
    let (p, fp) = Primes::new(INERT_PRIMES.0, INERT_PRIMES.1).find_map(|p| {
        let bp = BigInt::from(p);
        if modulo(lead, &bp).is_zero() {
            return None;
        }
        let fp: Vec<u64> = f.iter().map(|c| to_u64(&modulo(c, &bp))).collect();
        irreducible(&fp, p).then_some((p, fp))
    })?;
    let q = (0..d).fold(1u128, |acc, _| acc * u128::from(p));
    let bp = BigInt::from(p);
    let mut dp: Vec<u64> = delta.iter().map(|c| to_u64(&modulo(c, &bp))).collect();
    while dp.last() == Some(&0) {
        dp.pop();
    }
    let s0 = field_sqrt(&dp, &fp, p, q)?;
    // 1/√δ mod p, to lift: r ← r + r·(1 − δ·r²)/2 doubles its precision
    // with no division but by 2.
    let r0 = field_pow(&s0, q - 2, &fp, p);
    let mut r: Vec<BigInt> = (0..d)
        .map(|i| BigInt::from(r0.get(i).copied().unwrap_or(0)))
        .collect();
    let mut modulus = bp;
    let mut bound = max_bits(delta) / 2 + MARGIN;
    for _ in 0..=RETRIES {
        while modulus.bits() < bound + 1 {
            modulus = &modulus * &modulus;
            let dm: Vec<BigInt> = delta.iter().map(|c| modulo(c, &modulus)).collect();
            let r2 = mul_mod_m(&r, &r, f, &modulus);
            let mut e = mul_mod_m(&dm, &r2, f, &modulus);
            for c in &mut e {
                *c = -&*c;
            }
            e[0] += 1;
            let half = (&modulus + 1u32) >> 1u32;
            let step = mul_mod_m(&r, &e, f, &modulus);
            r = r
                .iter()
                .zip(&step)
                .map(|(x, y)| modulo(&(x + y * &half), &modulus))
                .collect();
        }
        // S = δ/√δ, with coefficients centered on 0.
        let dm: Vec<BigInt> = delta.iter().map(|c| modulo(c, &modulus)).collect();
        let half = &modulus >> 1u32;
        let s: Vec<BigInt> = mul_mod_m(&dm, &r, f, &modulus)
            .into_iter()
            .map(|c| if c > half { c - &modulus } else { c })
            .collect();
        if mul(&s, &s, f) == delta {
            return Some(s);
        }
        bound *= 2;
    }
    None
}

fn to_u64(x: &BigInt) -> u64 {
    x.iter_u64_digits().next().unwrap_or(0)
}

/// Tries the dependency `deps` for a factor of `n`: returns
/// `gcd(X − Y, n)` if it is proper.
pub fn factor<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    poly: &Poly,
    deps: &[&Relation],
) -> Option<Uint<LIMBS>> {
    let nb: BigInt = n.to_string().parse().ok()?;
    let d = poly.degree();
    let lead = BigInt::from(poly.f[d]);
    // F(x) = c^(d−1)·f(x/c), monic.
    let mut f: Vec<BigInt> = (0..d)
        .map(|i| BigInt::from(poly.f[i]) * num_traits::pow(lead.clone(), d - 1 - i))
        .collect();
    f.push(BigInt::one());
    let df: Vec<BigInt> = (1..=d).map(|i| &f[i] * BigInt::from(i)).collect();

    let elems = deps
        .iter()
        .map(|r| {
            let mut e = vec![BigInt::zero(); d];
            e[0] = &lead * BigInt::from(r.a);
            e[1] = -BigInt::from(r.b);
            e
        })
        .collect();
    let gamma = product(elems, &f);
    let delta = mul(&mul(&gamma, &df, &f), &df, &f);
    let s = algebraic_sqrt(&delta, &f, &lead)?;

    let cm = modulo(&(&lead * BigInt::from(poly.m)), &nb);
    let x = eval_mod(&s, &cm, &nb);
    // Y = c^(k/2)·F′(c·m)·∏ p^(e/2).
    let mut exponents: BTreeMap<u64, u32> = BTreeMap::new();
    for r in deps {
        for &p in &r.rational {
            *exponents.entry(p.into()).or_default() += 1;
        }
        if r.large[0] > 1 {
            *exponents.entry(r.large[0]).or_default() += 1;
        }
    }
    let mut y = modulo(&lead, &nb).modpow(&BigInt::from(deps.len() / 2), &nb);
    y = modulo(&(y * eval_mod(&df, &cm, &nb)), &nb);
    for (p, e) in exponents {
        debug_assert!(e % 2 == 0, "odd exponent of {p}");
        y = modulo(
            &(y * BigInt::from(p).modpow(&BigInt::from(e / 2), &nb)),
            &nb,
        );
    }
    let g = num_integer_gcd(modulo(&(x - y), &nb), nb.clone());
    (g > BigInt::one() && g < nb).then(|| g.to_string().parse().ok())?
}

fn num_integer_gcd(mut a: BigInt, mut b: BigInt) -> BigInt {
    while !b.is_zero() {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a.abs()
}
