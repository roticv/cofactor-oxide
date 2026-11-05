//! Polynomials over 𝔽ₚ for a word-sized prime `p`, and their roots.
//!
//! Coefficients are `u64` below `p`, lowest degree first. Below 2^32,
//! products fit in a word before reduction; larger `p`, such as the
//! quadratic characters', take 128-bit products.

use alloc::vec;
use alloc::vec::Vec;

use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;

/// Primes up to this have their roots found by trying every residue.
const BRUTE_FORCE: u64 = 64;

/// An integer coefficient of a polynomial: an `i128` while selecting
/// polynomials, which is fast, and a [`BigInt`] in a [`Poly`](super::Poly),
/// which holds any size.
pub trait Coeff {
    /// The coefficient modulo `p`, in `[0, p)`.
    fn mod_p(&self, p: u64) -> u64;
    /// The nearest `f64`.
    fn to_f64(&self) -> f64;
}

impl Coeff for i128 {
    fn mod_p(&self, p: u64) -> u64 {
        self.rem_euclid(i128::from(p)) as u64
    }

    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

impl Coeff for BigInt {
    fn mod_p(&self, p: u64) -> u64 {
        // The magnitude's digits from the top, as with long division.
        let r = self.iter_u64_digits().rev().fold(0, |r: u64, d| {
            (((u128::from(r) << 64) | u128::from(d)) % u128::from(p)) as u64
        });
        if self.sign() == Sign::Minus && r != 0 {
            p - r
        } else {
            r
        }
    }

    fn to_f64(&self) -> f64 {
        ToPrimitive::to_f64(self).unwrap_or(f64::INFINITY)
    }
}

/// `f(x) mod p` for integer coefficients, trimmed of leading zeros.
pub fn reduce<C: Coeff>(f: &[C], p: u64) -> Vec<u64> {
    let mut g: Vec<u64> = f.iter().map(|c| c.mod_p(p)).collect();
    trim(&mut g);
    g
}

/// `f(r) mod p`, by Horner's rule.
pub fn eval(f: &[u64], r: u64, p: u64) -> u64 {
    f.iter().rev().fold(0, |acc, &c| mul_add(c, acc, r, p))
}

fn trim(f: &mut Vec<u64>) {
    while f.last() == Some(&0) {
        f.pop();
    }
}

/// `acc + x·y mod p`, for `acc`, `x` and `y` below `p`.
pub(crate) fn mul_add(acc: u64, x: u64, y: u64, p: u64) -> u64 {
    if p <= 1 << 32 {
        (acc + x * y) % p
    } else {
        ((u128::from(acc) + u128::from(x) * u128::from(y)) % u128::from(p)) as u64
    }
}

/// `a⁻¹ mod p`, by Euclid's algorithm, for `a` coprime to `p`.
pub(crate) fn inv(a: u64, p: u64) -> u64 {
    let (mut r0, mut r1) = (i128::from(p), i128::from(a % p));
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let k = r0 / r1;
        (r0, r1) = (r1, r0 - k * r1);
        (t0, t1) = (t1, t0 - k * t1);
    }
    t0.rem_euclid(i128::from(p)) as u64
}

/// The Jacobi symbol `(a/n)` for odd `n`: the Legendre symbol, `−1`, `0`
/// or `1`, when `n` is prime. By reciprocity, with no products, so that
/// it is quick for any `n`.
pub(crate) fn jacobi(a: u64, n: u64) -> i32 {
    let (mut a, mut n) = (a % n, n);
    let mut sign = 1;
    while a != 0 {
        let twos = a.trailing_zeros();
        a >>= twos;
        // (2/n) = −1 for n ≡ ±3 (mod 8).
        if twos % 2 == 1 && matches!(n % 8, 3 | 5) {
            sign = -sign;
        }
        if a % 4 == 3 && n % 4 == 3 {
            sign = -sign;
        }
        (a, n) = (n % a, a);
    }
    if n == 1 { sign } else { 0 }
}

/// `f` divided by its leading coefficient.
fn monic(f: &mut [u64], p: u64) {
    if let Some(&lead) = f.last() {
        let i = inv(lead, p);
        for c in f.iter_mut() {
            *c = mul_add(0, *c, i, p);
        }
    }
}

/// The remainder of `a` divided by the monic `b`, in place.
fn rem(a: &mut Vec<u64>, b: &[u64], p: u64) {
    let d = b.len() - 1;
    while a.len() > d {
        let lead = a.pop().unwrap();
        if lead != 0 {
            let shift = a.len() - d;
            for (j, &c) in b[..d].iter().enumerate() {
                a[shift + j] = mul_add(a[shift + j], p - lead, c, p);
            }
        }
    }
    trim(a);
}

/// `a·b mod m`, for monic `m`.
pub(crate) fn mul_mod(a: &[u64], b: &[u64], m: &[u64], p: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = vec![0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            r[i + j] = mul_add(r[i + j], x, y, p);
        }
    }
    rem(&mut r, m, p);
    r
}

/// `base^e mod m`, for monic `m`.
pub(crate) fn pow_mod(base: &[u64], mut e: u64, m: &[u64], p: u64) -> Vec<u64> {
    let mut b = base.to_vec();
    rem(&mut b, m, p);
    let mut r = vec![1];
    rem(&mut r, m, p);
    while e > 0 {
        if e & 1 == 1 {
            r = mul_mod(&r, &b, m, p);
        }
        b = mul_mod(&b, &b, m, p);
        e >>= 1;
    }
    r
}

/// The monic greatest common divisor.
pub(crate) fn gcd(mut a: Vec<u64>, mut b: Vec<u64>, p: u64) -> Vec<u64> {
    trim(&mut a);
    trim(&mut b);
    while !b.is_empty() {
        monic(&mut b, p);
        rem(&mut a, &b, p);
        core::mem::swap(&mut a, &mut b);
    }
    monic(&mut a, p);
    a
}

/// `a / b` for monic `b` dividing `a`.
fn div_exact(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let d = b.len() - 1;
    let mut r = a.to_vec();
    let mut q = vec![0; a.len() - d];
    for i in (0..q.len()).rev() {
        let lead = r[i + d];
        q[i] = lead;
        for (j, &c) in b.iter().enumerate() {
            r[i + j] = mul_add(r[i + j], p - lead, c, p);
        }
    }
    q
}

/// The distinct roots in `[0, p)` of `f`, an integer polynomial, modulo the
/// prime `p`, in increasing order. Empty if `f ≡ 0`.
pub fn roots<C: Coeff>(f: &[C], p: u64) -> Vec<u64> {
    let mut g = reduce(f, p);
    if g.len() <= 1 {
        return Vec::new();
    }
    let mut roots = if p <= BRUTE_FORCE {
        (0..p).filter(|&r| eval(&g, r, p) == 0).collect()
    } else {
        monic(&mut g, p);
        // The product of the distinct linear factors: gcd(x^p − x, g).
        let mut h = pow_mod(&[0, 1], p, &g, p);
        h.resize(h.len().max(2), 0);
        h[1] = (h[1] + p - 1) % p;
        let linear = gcd(g.clone(), h, p);
        let mut out = Vec::new();
        split(linear, p, 1, &mut out);
        out
    };
    roots.sort_unstable();
    roots
}

/// Splits a monic product of distinct linear factors by Cantor–Zassenhaus:
/// `gcd((x + δ)^((p−1)/2) − 1, g)` collects the roots `r` with `r + δ` a
/// nonzero square, about half of them for each `δ`.
fn split(g: Vec<u64>, p: u64, mut delta: u64, out: &mut Vec<u64>) {
    match g.len() {
        0 | 1 => {}
        2 => out.push((p - g[0]) % p),
        _ => loop {
            let mut t = pow_mod(&[delta, 1], (p - 1) / 2, &g, p);
            delta += 1;
            if t.is_empty() {
                continue;
            }
            t[0] = (t[0] + p - 1) % p;
            let u = gcd(g.clone(), t, p);
            if u.len() > 1 && u.len() < g.len() {
                let v = div_exact(&g, &u, p);
                split(u, p, delta, out);
                split(v, p, delta, out);
                return;
            }
        },
    }
}

/// Whether the monic `f` of degree `d` is irreducible mod `p`: `x^(p^d)
/// ≡ x (mod f)`, and `gcd(x^(p^(d/q)) − x, f) = 1` for each prime `q`
/// dividing `d`, so that no factor has degree below `d`.
pub fn irreducible(f: &[u64], p: u64) -> bool {
    let d = f.len() - 1;
    // x^(p^i) for i = 0..=d.
    let mut powers = vec![vec![0, 1]];
    for i in 0..d {
        let next = pow_mod(&powers[i], p, f, p);
        powers.push(next);
    }
    let minus_x = |mut g: Vec<u64>| {
        g.resize(g.len().max(2), 0);
        g[1] = (g[1] + p - 1) % p;
        g
    };
    let mut top = minus_x(powers[d].clone());
    trim(&mut top);
    if !top.is_empty() {
        return false;
    }
    (2..=d)
        .filter(|q| d.is_multiple_of(*q) && (2..*q).all(|k| !q.is_multiple_of(k)))
        .all(|q| gcd(f.to_vec(), minus_x(powers[d / q].clone()), p).len() == 1)
}
