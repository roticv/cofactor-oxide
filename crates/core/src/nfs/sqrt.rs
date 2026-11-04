//! The square roots of a dependency on both sides, and the factor they
//! give. The algebraic square root has coefficients of hundreds of
//! thousands of bits, in `num-bigint`'s integers, as are a [`Poly`]'s.
//! This module also checks that a polynomial read from a file has a root
//! modulo `N`.
//!
//! With `c` the leading coefficient of `f`, `θ = c·α` is a root of the
//! monic `F(x) = c^(d−1)·f(x/c)`, so `ℤ[θ]` is in the ring of integers.
//! For a dependency of `k` relations (`k` is even),
//! `δ = ∏ (c·a − b·θ) · F′(θ)²` is a square in `ℤ[θ]`, and `S = √δ` is
//! found modulo a prime `p` that stays prime in `ℤ[θ]`, then lifted by
//! Newton's method modulo `p^(2^i)` until its coefficients are certain.
//! Mapping `θ` to `c·m/l`, for the common root `m/l` of `f` and
//! `g = l·x − m` modulo `N`, gives `X = S(c·m/l)`, and on the rational side
//! `Y = c^(k/2)·F′(c·m/l)·√∏ (l·a − m·b) / l^(k/2)`, with `X² ≡ Y² (mod N)`.

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
use crate::parallel::{Parallel, Serial, map_dynamic};
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

/// `a·b mod F` for the monic `F`, with coefficients lowest first, the
/// products of coefficients shared out by `par`.
fn mul(a: &[BigInt], b: &[BigInt], f: &[BigInt], par: &impl Parallel) -> Vec<BigInt> {
    let d = f.len() - 1;
    let pairs: Vec<(usize, usize)> = (0..a.len())
        .flat_map(|i| (0..b.len()).map(move |j| (i, j)))
        .filter(|&(i, j)| !a[i].is_zero() && !b[j].is_zero())
        .collect();
    let products = map_dynamic(par, pairs.len(), |k| &a[pairs[k].0] * &b[pairs[k].1]);
    let mut c = vec![BigInt::zero(); 2 * d - 1];
    for (&(i, j), p) in pairs.iter().zip(products) {
        c[i + j] += p;
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

/// `c mod m` for each coefficient, shared out by `par`.
fn modulo_all(c: &[BigInt], m: &BigInt, par: &impl Parallel) -> Vec<BigInt> {
    map_dynamic(par, c.len(), |i| modulo(&c[i], m))
}

/// `a·b mod (F, m)`.
fn mul_mod_m(
    a: &[BigInt],
    b: &[BigInt],
    f: &[BigInt],
    m: &BigInt,
    par: &impl Parallel,
) -> Vec<BigInt> {
    modulo_all(&mul(a, b, f, par), m, par)
}

/// The product of `elems` mod `F`, multiplying in pairs so that the
/// operands stay balanced. While there are more pairs than threads, the
/// pairs are shared out; after that, the products within each.
fn product(mut elems: Vec<Vec<BigInt>>, f: &[BigInt], par: &impl Parallel) -> Vec<BigInt> {
    while elems.len() > 1 {
        let pairs = elems.len() / 2;
        let mut next: Vec<Vec<BigInt>> = if pairs >= par.threads() {
            map_dynamic(par, pairs, |k| {
                mul(&elems[2 * k], &elems[2 * k + 1], f, &Serial)
            })
        } else {
            (0..pairs)
                .map(|k| mul(&elems[2 * k], &elems[2 * k + 1], f, par))
                .collect()
        };
        if elems.len() % 2 == 1 {
            next.push(elems.pop().unwrap());
        }
        elems = next;
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

/// `Σ cᵢ·xⁱ·y^(d−1−i) mod n`, for `d` the number of coefficients `c`.
fn eval_mod(c: &[BigInt], x: &BigInt, y: &BigInt, n: &BigInt) -> BigInt {
    let mut yk = BigInt::one();
    let mut acc = BigInt::zero();
    for ci in c.iter().rev() {
        acc = modulo(&(acc * x + ci * &yk), n);
        yk = modulo(&(yk * y), n);
    }
    acc
}

/// The bits of the largest coefficient.
fn max_bits(c: &[BigInt]) -> u64 {
    c.iter().map(|x| x.bits()).max().unwrap_or(0)
}

/// The square root of `δ` in `ℤ[θ]/(F)`, if it is a square there.
fn algebraic_sqrt(
    delta: &[BigInt],
    f: &[BigInt],
    lead: &BigInt,
    par: &impl Parallel,
) -> Option<Vec<BigInt>> {
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
            let dm = modulo_all(delta, &modulus, par);
            let r2 = mul_mod_m(&r, &r, f, &modulus, par);
            let mut e = mul_mod_m(&dm, &r2, f, &modulus, par);
            for c in &mut e {
                *c = -&*c;
            }
            e[0] += 1;
            let half = (&modulus + 1u32) >> 1u32;
            let step = mul_mod_m(&r, &e, f, &modulus, par);
            r = r
                .iter()
                .zip(&step)
                .map(|(x, y)| modulo(&(x + y * &half), &modulus))
                .collect();
        }
        // S = δ/√δ, with coefficients centered on 0.
        let dm = modulo_all(delta, &modulus, par);
        let half = &modulus >> 1u32;
        let s: Vec<BigInt> = mul_mod_m(&dm, &r, f, &modulus, par)
            .into_iter()
            .map(|c| if c > half { c - &modulus } else { c })
            .collect();
        if mul(&s, &s, f, par) == delta {
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
/// `gcd(X − Y, n)` if it is proper. The large products are shared out by
/// `par`.
pub fn factor<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    poly: &Poly,
    deps: &[&Relation],
    par: &impl Parallel,
) -> Option<Uint<LIMBS>> {
    let nb: BigInt = n.to_string().parse().ok()?;
    let d = poly.degree();
    let lead = poly.f[d].clone();
    // F(x) = c^(d−1)·f(x/c), monic.
    let mut f: Vec<BigInt> = (0..d)
        .map(|i| &poly.f[i] * num_traits::pow(lead.clone(), d - 1 - i))
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
    let gamma = product(elems, &f, par);
    let delta = mul(&mul(&gamma, &df, &f, par), &df, &f, par);
    let s = algebraic_sqrt(&delta, &f, &lead, par)?;

    // θ maps to c·m/l. Rather than invert l, both sides are multiplied by
    // l^(d−1+k/2): X = S(c·m, l)·l^(k/2) and Y = c^(k/2)·F′(c·m, l)·∏ p^(e/2),
    // with S and F′ homogenized to degree d − 1.
    let cm = modulo(&(&lead * &poly.m), &nb);
    let l = modulo(&poly.l, &nb);
    let half = BigInt::from(deps.len() / 2);
    let x = modulo(&(eval_mod(&s, &cm, &l, &nb) * l.modpow(&half, &nb)), &nb);
    let mut exponents: BTreeMap<u64, u32> = BTreeMap::new();
    for r in deps {
        for &p in &r.rational {
            *exponents.entry(p).or_default() += 1;
        }
    }
    let mut y = modulo(&lead, &nb).modpow(&half, &nb);
    y = modulo(&(y * eval_mod(&df, &cm, &l, &nb)), &nb);
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

/// Whether `m/l` is a root of `f` modulo `n`: `F(m, l) ≡ 0 (mod n)`.
pub(crate) fn has_common_root<const LIMBS: usize>(n: &Uint<LIMBS>, poly: &Poly) -> bool {
    let Ok(nb) = n.to_string().parse::<BigInt>() else {
        return false;
    };
    eval_mod(&poly.f, &poly.m, &poly.l, &nb).is_zero()
}

fn num_integer_gcd(mut a: BigInt, mut b: BigInt) -> BigInt {
    while !b.is_zero() {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a.abs()
}
