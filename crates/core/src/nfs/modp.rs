//! Polynomials over 𝔽ₚ for a word-sized prime `p`, and their roots.
//!
//! Coefficients are `u64` below `p < 2^32`, lowest degree first, so
//! products fit in a word before reduction.

use alloc::vec;
use alloc::vec::Vec;

/// Primes up to this have their roots found by trying every residue.
const BRUTE_FORCE: u64 = 64;

/// `f(x) mod p` for integer coefficients, trimmed of leading zeros.
pub fn reduce(f: &[i128], p: u64) -> Vec<u64> {
    let mut g: Vec<u64> = f
        .iter()
        .map(|&c| c.rem_euclid(i128::from(p)) as u64)
        .collect();
    trim(&mut g);
    g
}

/// `f(r) mod p`, by Horner's rule.
pub fn eval(f: &[u64], r: u64, p: u64) -> u64 {
    f.iter().rev().fold(0, |acc, &c| (acc * r + c) % p)
}

fn trim(f: &mut Vec<u64>) {
    while f.last() == Some(&0) {
        f.pop();
    }
}

fn inv(a: u64, p: u64) -> u64 {
    crate::qs::modp::pow_mod(a, p - 2, p)
}

/// `f` divided by its leading coefficient.
fn monic(f: &mut [u64], p: u64) {
    if let Some(&lead) = f.last() {
        let i = inv(lead, p);
        for c in f.iter_mut() {
            *c = *c * i % p;
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
                a[shift + j] = (a[shift + j] + (p - lead) * c) % p;
            }
        }
    }
    trim(a);
}

/// `a·b mod m`, for monic `m`.
fn mul_mod(a: &[u64], b: &[u64], m: &[u64], p: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = vec![0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            r[i + j] = (r[i + j] + x * y) % p;
        }
    }
    rem(&mut r, m, p);
    r
}

/// `base^e mod m`, for monic `m`.
fn pow_mod(base: &[u64], mut e: u64, m: &[u64], p: u64) -> Vec<u64> {
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
fn gcd(mut a: Vec<u64>, mut b: Vec<u64>, p: u64) -> Vec<u64> {
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
            r[i + j] = (r[i + j] + (p - lead) * c) % p;
        }
    }
    q
}

/// The distinct roots in `[0, p)` of `f`, an integer polynomial, modulo the
/// prime `p < 2^32`, in increasing order. Empty if `f ≡ 0`.
pub fn roots(f: &[i128], p: u64) -> Vec<u64> {
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
