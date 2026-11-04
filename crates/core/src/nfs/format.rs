//! Relations and polynomials as text, in the formats that msieve, GGNFS
//! and CADO-NFS share, so that relations can be saved, loaded, and
//! exchanged with them.
//!
//! A relation is a line `a,b:r₁,r₂,…:a₁,a₂,…`, with `a` and `b` in decimal
//! and the primes of `|l·a − m·b|` and then of `|F(a, b)|` in hex, each as
//! often as it divides. Some sievers leave out primes below 1000, which
//! reading puts back.
//!
//! A polynomial pair is GGNFS's and CADO-NFS's `.poly` text: lines `n:`,
//! `skew:`, `c0:` up to `c6:` for `f`, and `Y0:`, `Y1:` for `g`. msieve's
//! `.fb` keys, `N`, `SKEW`, `A0` up to `A6`, `R0` and `R1`, are read too.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use super::params::Params;
use super::poly::Poly;
use super::sieve::{Relation, Wide, norm};
use super::sqrt::has_common_root;
use crate::arith::{Limb, Uint};
use crate::primes::Primes;

/// Primes below this may be missing from a relation read, and are found
/// by trial division.
const LEFT_OUT: u64 = 1000;

/// `rel` as a line, without a newline.
pub fn relation_line(rel: &Relation) -> String {
    let mut line = format!("{},{}:", rel.a, rel.b);
    hex_list(&mut line, &rel.rational);
    line.push(':');
    hex_list(&mut line, &rel.algebraic);
    line
}

fn hex_list(out: &mut String, primes: &[u64]) {
    for (i, p) in primes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{p:x}");
    }
}

/// The relation on `line` for `poly`, if it is one: `a` and `b` coprime,
/// and the primes listed, with any below 1000 left out, the factorizations
/// of `|l·a − m·b|` and `|F(a, b)|`.
pub fn parse_relation(poly: &Poly, line: &str) -> Option<Relation> {
    let mut parts = line.trim().split(':');
    let (ab, rat, alg) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let (a, b) = ab.split_once(',')?;
    let (a, b): (i64, u64) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
    if b == 0 || gcd(a.unsigned_abs(), b) != 1 {
        return None;
    }
    // l·a − m·b.
    let la = i128::try_from(poly.l).ok()?.checked_mul(a.into())?;
    let mb = i128::try_from(poly.m).ok()?.checked_mul(b.into())?;
    let g = la.checked_sub(mb)?;
    let mut rational = parse_primes(rat)?;
    let mut algebraic = parse_primes(alg)?;
    let g_wide = Wide::from_u64(g.unsigned_abs() as u64)
        .wrapping_add(&(Wide::from_u64((g.unsigned_abs() >> 64) as u64) << 64));
    if !factors(g_wide, &mut rational) || !factors(norm(&poly.f, a, b), &mut algebraic) {
        return None;
    }
    Some(Relation {
        a,
        b,
        negative: g < 0,
        rational,
        algebraic,
    })
}

fn parse_primes(list: &str) -> Option<Vec<u64>> {
    list.split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| u64::from_str_radix(p, 16).ok().filter(|&p| p >= 2))
        .collect()
}

/// Whether `primes`, with the primes below [`LEFT_OUT`] it lacks added,
/// multiply to `v`, which must not be 0.
fn factors(mut v: Wide, primes: &mut Vec<u64>) -> bool {
    if v.is_zero() {
        return false;
    }
    for &p in primes.iter() {
        match divide(&v, p) {
            Some(q) => v = q,
            None => return false,
        }
    }
    if v == Wide::ONE {
        return true;
    }
    for p in Primes::new(2, LEFT_OUT) {
        while let Some(q) = divide(&v, p) {
            v = q;
            primes.push(p);
        }
    }
    v == Wide::ONE
}

/// `v/p`, if `p` divides `v`.
// Limb is u32 or u64, so the casts are needed on one target and redundant
// on the other.
#[allow(clippy::unnecessary_cast)]
fn divide(v: &Wide, p: u64) -> Option<Wide> {
    let (q, r) = match Limb::try_from(p) {
        Ok(limb) => {
            let (q, r) = v.div_rem_limb(limb);
            (q, r as u64)
        }
        Err(_) => {
            let (q, r) = v.div_rem(&Wide::from_u64(p));
            (q, u64::from(!r.is_zero()))
        }
    };
    (r == 0).then_some(q)
}

/// `poly` for `n` as `.poly` text.
pub fn poly_text<const LIMBS: usize>(n: &Uint<LIMBS>, poly: &Poly) -> String {
    let mut text = format!("n: {n}\nskew: {:.3}\n", poly.skew);
    for (i, c) in poly.f.iter().enumerate() {
        let _ = writeln!(text, "c{i}: {c}");
    }
    let _ = writeln!(text, "Y0: -{}\nY1: {}", poly.m, poly.l);
    text
}

/// The polynomial pair in `.poly` or `.fb` `text`, if it is one for `n`:
/// `f` and `g` with a common root modulo `n`, `g` of degree one. `params`
/// rates it.
pub fn parse_poly<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    text: &str,
    params: &Params,
) -> Option<Poly> {
    let mut f: Vec<i128> = Vec::new();
    let (mut y0, mut y1): (Option<i128>, Option<i128>) = (None, None);
    let mut skew = None;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some((key, value)) = line
            .split_once(':')
            .or_else(|| line.split_once(char::is_whitespace))
        else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "n" | "N" => {
                if value.parse::<Uint<LIMBS>>().ok()? != *n {
                    return None;
                }
            }
            "skew" | "SKEW" => skew = Some(value.parse::<f64>().ok()?),
            "Y0" | "R0" => y0 = Some(value.parse().ok()?),
            "Y1" | "R1" => y1 = Some(value.parse().ok()?),
            key => {
                let Some(i) = key
                    .strip_prefix('c')
                    .or_else(|| key.strip_prefix('A'))
                    .and_then(|i| i.parse::<usize>().ok())
                    .filter(|&i| i <= 8)
                else {
                    continue;
                };
                if f.len() <= i {
                    f.resize(i + 1, 0);
                }
                f[i] = value.parse().ok()?;
            }
        }
    }
    while f.last() == Some(&0) {
        f.pop();
    }
    // g = Y1·x + Y0 = l·x − m, with l > 0.
    let (mut y0, mut y1) = (y0?, y1?);
    if y1 < 0 {
        (y0, y1) = (-y0, -y1);
    }
    let (m, l) = (u128::try_from(-y0).ok()?, u128::try_from(y1).ok()?);
    if f.len() < 3 || m == 0 || l == 0 {
        return None;
    }
    let mut poly = Poly::rated(f, m, l, params);
    if let Some(s) = skew {
        poly.skew = s;
    }
    has_common_root(n, &poly).then_some(poly)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
