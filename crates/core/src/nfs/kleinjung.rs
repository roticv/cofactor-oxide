//! Kleinjung's polynomial selection, stage 1: candidates with a small
//! leading coefficient and small next ones.
//!
//! With `g = l·x − m`, `N` has the expansion `N = Σ aᵢ·mⁱ·l^(d−i)`. Kleinjung
//! works with `Ñ = d^d·a_d^(d−1)·N` and `m̃ = d·a_d·m + a_{d−1}·l`: if
//! `m̃^d ≡ Ñ (mod l²)`, then `m̃` gives `m` and an `a_{d−1}` of at most
//! `d·a_d/2`, and the expansion exists. For `m̃ = m̃₀ + t` near
//! `m̃₀ = Ñ^(1/d)`, `a_{d−2}` is about `t·m̃^(d−1)/l²` up to constants: small,
//! if `l` is large and `t` small.
//!
//! For `l = q·ℓ₁·ℓ₂`, the congruence holds when `m̃ ≡ ρ (mod q²)` and
//! `m̃ ≡ rᵢ (mod ℓᵢ²)` for roots of `x^d ≡ Ñ`. Here `q`, the special-q,
//! is a product of small primes, whose roots combine into many `ρ`; and
//! `ℓ₁`, `ℓ₂` are primes around `P`. For each `ρ`, listing, for every such
//! prime `ℓ` and root `r`, the `u` in `[−U, U]` with
//! `m̃₀ + t_ρ + q²·u ≡ r (mod ℓ²)`, two entries with the same `u` from
//! different primes give a candidate.
//!
//! [Kleinjung, *On polynomial selection for the general number field
//! sieve*, Math. Comp. 2006; *Polynomial selection*, CADO workshop, 2008;
//! and CADO-NFS's `polyselect`.]

use alloc::vec;
use alloc::vec::Vec;

use super::modp::{inv, roots};
use super::poly::to_u128;
use crate::arith::Uint;
use crate::factor::root;
use crate::primes::Primes;

/// Special-q primes are taken from below this, as in CADO-NFS.
const Q_PRIMES: u64 = 272;

/// The most primes in a special-q, which keeps it in a `u64`.
const Q_FACTORS: usize = 8;

/// A candidate: `f`, `m` and `l`, with `F(m, l) = N`.
pub type Candidate = (Vec<i128>, u128, u128);

/// Candidates for `n` of degree `d` with leading coefficient `ad`, from
/// primes `ℓ` in `[p0, 2·p0)`, `|u| ≤ u_max`, and special-q roots `ρ`, at
/// most `nq` of them.
///
/// This is Kleinjung's method on `Ñ = d^d·a_d^(d−1)·N`: `m̃` near
/// `Ñ^(1/d)` with `m̃^d ≡ Ñ (mod l²)` gives `m̃ = d·a_d·m + a_{d−1}·l` with
/// `|a_{d−1}| ≤ d·a_d/2`, so `a_{d−1}` is tiny and the search's freedom
/// goes into `a_{d−2}`. As in CADO-NFS, each special-q is a product of `k` small primes with
/// roots, `k` as large as keeps `q·(2·p0)⁴` below `m₀` and so `a_{d−1}`
/// small; and the special-q are the products of `k` of the primes with the
/// most roots, taking as many primes as `nq` roots need.
pub fn candidates<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    d: usize,
    ad: u64,
    p0: u64,
    u_max: i64,
    nq: usize,
) -> Vec<Candidate> {
    // Ñ = d^d·a_d^(d−1)·N, and m̃₀ its d-th root.
    let scale = (0..d - 1).try_fold(u128::pow(d as u128, d as u32), |x, _| {
        x.checked_mul(u128::from(ad))
    });
    let Some(nt) = scale.and_then(|x| {
        let (lo, hi) = n.widening_mul(&big(x));
        hi.is_zero().then_some(lo)
    }) else {
        return Vec::new();
    };
    let Some(m0) = to_u128(&root(&nt, d as u32)).filter(|&m| m < 1 << 120) else {
        return Vec::new();
    };
    let da = d as u64 * ad;
    // The small primes and their roots modulo p², most roots first.
    let mut small: Vec<(u64, Vec<u64>)> = Primes::new(3, Q_PRIMES)
        .filter(|&p| !da.is_multiple_of(p) && rem(n, p) != 0)
        .map(|p| (p, roots_mod_square(&nt, 1, p, d)))
        .filter(|(_, rs)| !rs.is_empty())
        .collect();
    small.sort_by_key(|(_, rs)| core::cmp::Reverse(rs.len()));
    // k primes in each special-q, and the first `lq` to choose them from.
    let reach = (2 * p0) as f64;
    let reach = reach * reach * reach * reach;
    let (mut k, mut prod, mut sq) = (0, 1usize, 1.0f64);
    for &(p, _) in &small {
        if prod >= nq || k == Q_FACTORS || reach * sq * p as f64 >= m0 as f64 {
            break;
        }
        prod = prod.saturating_mul(d);
        sq *= p as f64;
        k += 1;
    }
    let k = k.max(1).min(small.len());
    if k == 0 {
        return Vec::new();
    }
    let mut lq = k;
    while lq < small.len() && combinations(&small[..lq], k) < nq {
        lq += 1;
    }
    // The large primes, each root's m₀-relative residue.
    let mut large: Vec<(u64, u64)> = Vec::new();
    for ell in Primes::new(p0, 2 * p0) {
        if da.is_multiple_of(ell) {
            continue;
        }
        let ell2 = ell * ell;
        let m0_2 = (m0 % u128::from(ell2)) as u64;
        for r in roots_mod_square(&nt, 1, ell, d) {
            large.push((ell, (r + ell2 - m0_2) % ell2));
        }
    }
    let mut found = Vec::new();
    let mut entries: Vec<(i64, u64)> = Vec::new();
    let mut inv_q2: Vec<u64> = Vec::new();
    let mut done = 0;
    let mut pick: Vec<usize> = (0..k).collect();
    'q: loop {
        // q, and its roots modulo q² by the CRT.
        let (mut q, mut rho) = (1u64, vec![0u64]);
        for &i in &pick {
            let (p, ref rs) = small[i];
            let (q2, p2) = (u128::from(q * q), u128::from(p * p));
            let q2_inv = u128::from(inverse((q2 % p2) as u64, p2 as u64).unwrap_or(0));
            rho = rho
                .iter()
                .flat_map(|&a| {
                    rs.iter().map(move |&r| {
                        let t = (u128::from(r) + p2 - u128::from(a) % p2) % p2 * q2_inv % p2;
                        (u128::from(a) + q2 * t) as u64
                    })
                })
                .collect();
            q *= p;
        }
        let q2 = u128::from(q) * u128::from(q);
        inv_q2.clear();
        inv_q2.extend(large.iter().map(|&(ell, _)| {
            let ell2 = ell * ell;
            inverse((q2 % u128::from(ell2)) as u64, ell2).unwrap_or(0)
        }));
        for &rq in &rho {
            if done >= nq {
                break 'q;
            }
            done += 1;
            // t ≡ ρ − m₀ (mod q²), the smallest such t_ρ.
            let t_rho = ((u128::from(rq) + q2 - m0 % q2) % q2) as i128;
            let t_rho = if t_rho > (q2 / 2) as i128 {
                t_rho - q2 as i128
            } else {
                t_rho
            };
            entries.clear();
            for (&(ell, e), &inv) in large.iter().zip(&inv_q2) {
                let ell2 = ell * ell;
                // u ≡ (e − t_ρ)/q² (mod ℓ²), from −u_max up.
                let base = (i128::from(e) - t_rho).rem_euclid(i128::from(ell2)) as u64;
                let res = (u128::from(base) * u128::from(inv) % u128::from(ell2)) as i64;
                let ell2 = ell2 as i64;
                let mut u = res - ell2 * ((res + u_max) / ell2);
                while u <= u_max {
                    entries.push((u, ell));
                    u += ell2;
                }
            }
            entries.sort_unstable();
            for (i, &(u, ell1)) in entries.iter().enumerate() {
                for &(_, ell2) in entries[i + 1..].iter().take_while(|e| e.0 == u) {
                    if ell2 == ell1 {
                        continue;
                    }
                    let t = t_rho + q2 as i128 * i128::from(u);
                    let mt = (m0 as i128 + t) as u128;
                    let l = u128::from(q) * u128::from(ell1) * u128::from(ell2);
                    // m̃ = d·a_d·m + a_{d−1}·l with |a_{d−1}| ≤ d·a_d/2.
                    let Some(inv_l) = inverse((l % u128::from(da)) as u64, da) else {
                        continue;
                    };
                    let mut a1 = (mt % u128::from(da) * u128::from(inv_l) % u128::from(da)) as i128;
                    if 2 * a1 >= i128::from(da) {
                        a1 -= i128::from(da);
                    }
                    let Some(rest) = (mt as i128).checked_sub(a1 * l as i128) else {
                        continue;
                    };
                    if rest <= 0 || rest % i128::from(da) != 0 {
                        continue;
                    }
                    let m = (rest / i128::from(da)) as u128;
                    if let Some(f) = expand(n, ad, m, l, d) {
                        found.push((f, m, l));
                    }
                }
            }
        }
        if !next_pick(&mut pick, lq) {
            break;
        }
    }
    found
}

/// The number of roots modulo the products of `k` of `primes`: the sum,
/// over each choice, of the product of their numbers of roots.
fn combinations(primes: &[(u64, Vec<u64>)], k: usize) -> usize {
    // The coefficient of z^k in the product of (1 + r·z).
    let mut c = vec![0usize; k + 1];
    c[0] = 1;
    for (_, rs) in primes {
        for j in (1..=k).rev() {
            c[j] = c[j].saturating_add(c[j - 1].saturating_mul(rs.len()));
        }
    }
    c[k]
}

/// The next `k`-element subset of `0..n` after `pick` in lexicographic
/// order, or `false` after the last.
fn next_pick(pick: &mut [usize], n: usize) -> bool {
    let k = pick.len();
    for i in (0..k).rev() {
        if pick[i] < n - k + i {
            pick[i] += 1;
            for j in i + 1..k {
                pick[j] = pick[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// The roots of `a_d·x^d ≡ n (mod p²)` lifted from the simple ones mod `p`.
fn roots_mod_square<const LIMBS: usize>(n: &Uint<LIMBS>, ad: u64, p: u64, d: usize) -> Vec<u64> {
    let n2 = rem(n, p * p);
    let mut f = vec![0i128; d + 1];
    f[0] = -i128::from(n2 % p);
    f[d] = i128::from(ad);
    roots(&f, p)
        .into_iter()
        .filter_map(|r| lift(r, p, ad, n2, d))
        .collect()
}

/// `n mod k`.
fn rem<const LIMBS: usize>(n: &Uint<LIMBS>, k: u64) -> u64 {
    n.div_rem(&Uint::from_u64(k)).1.to_u64().unwrap_or(0)
}

/// Lifts the root `r` of `a_d·x^d ≡ N (mod ℓ)` to one modulo `ℓ²` by
/// Newton's method, given `N mod ℓ²`, or `None` if it is a multiple root.
fn lift(r: u64, ell: u64, ad: u64, n2: u64, d: usize) -> Option<u64> {
    let ell2 = ell * ell;
    let mul = |a: u64, b: u64, k: u64| (u128::from(a) * u128::from(b) % u128::from(k)) as u64;
    let pow = |x: u64, e: usize, k: u64| (0..e).fold(1 % k, |acc, _| mul(acc, x, k));
    let fr = (mul(ad % ell2, pow(r, d, ell2), ell2) + ell2 - n2) % ell2;
    let df = mul(mul(d as u64 % ell, ad % ell, ell), pow(r, d - 1, ell), ell);
    if df == 0 {
        return None;
    }
    // f(r) = ℓ·s, and r − ℓ·s/f′(r) is a root mod ℓ².
    let s = fr / ell;
    let step = mul(s, inv(df, ell), ell);
    Some((r + ell2 - step * ell % ell2) % ell2)
}

/// A signed big integer, as a magnitude and whether it is negative.
type Signed<const LIMBS: usize> = (Uint<LIMBS>, bool);

/// `x − y`.
fn sub<const LIMBS: usize>((x, xn): Signed<LIMBS>, (y, yn): Signed<LIMBS>) -> Signed<LIMBS> {
    if xn != yn {
        (x.wrapping_add(&y), xn)
    } else if x >= y {
        (x.wrapping_sub(&y), xn)
    } else {
        (y.wrapping_sub(&x), !xn)
    }
}

fn big<const LIMBS: usize>(x: u128) -> Uint<LIMBS> {
    Uint::from_u64(x as u64).wrapping_add(&(Uint::from_u64((x >> 64) as u64) << 64))
}

/// `x` as an `i128`, if it fits.
fn small<const LIMBS: usize>((x, neg): Signed<LIMBS>) -> Option<i128> {
    let v = i128::try_from(to_u128(&x)?).ok()?;
    Some(if neg { -v } else { v })
}

/// The expansion of `n` as `Σ aᵢ·mⁱ·l^(d−i)` with `a_d = ad`, each other
/// `aᵢ` the nearest to `rᵢ/mⁱ` allowed modulo `l`, or `None` if `l` doesn't
/// divide where it must or a coefficient doesn't fit.
fn expand<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    ad: u64,
    m: u128,
    l: u128,
    d: usize,
) -> Option<Vec<i128>> {
    let (bm, bl) = (big::<LIMBS>(m), big::<LIMBS>(l));
    let mut pow = vec![Uint::<LIMBS>::from_u64(1); d + 1];
    for i in 1..=d {
        let (lo, hi) = pow[i - 1].widening_mul(&bm);
        if !hi.is_zero() {
            return None;
        }
        pow[i] = lo;
    }
    let l64 = u64::try_from(l).ok()?;
    let mut a = vec![0i128; d + 1];
    a[d] = i128::from(ad);
    let mut r: Signed<LIMBS> = (*n, false);
    for i in (1..=d).rev() {
        // r ← (r − aᵢ·mⁱ)/l.
        let (lo, hi) = big::<LIMBS>(a[i].unsigned_abs()).widening_mul(&pow[i]);
        if !hi.is_zero() {
            return None;
        }
        r = sub(r, (lo, a[i] < 0));
        let (q, rest) = r.0.div_rem(&bl);
        if !rest.is_zero() {
            return None;
        }
        r.0 = q;
        let k = i - 1;
        if k == 0 {
            a[0] = small(r)?;
            break;
        }
        // The nearest value to r/m^k that is ≡ r·m^(−k) (mod l).
        let (q, _) = r.0.wrapping_add(&(pow[k] >> 1)).div_rem(&pow[k]);
        let near = small((q, r.1))?;
        let rl = rem(&r.0, l64);
        let rl = if r.1 && rl != 0 { l64 - rl } else { rl };
        let mk = rem(&pow[k], l64);
        let want = (u128::from(rl) * u128::from(inverse(mk, l64)?) % u128::from(l64)) as i128;
        let l = i128::from(l64);
        let mut delta = (want - near).rem_euclid(l);
        if delta > l / 2 {
            delta -= l;
        }
        a[k] = near + delta;
    }
    Some(a)
}

/// `x⁻¹ mod k` for any `k`, if `x` is invertible.
fn inverse(x: u64, k: u64) -> Option<u64> {
    let (mut r0, mut r1) = (i128::from(k), i128::from(x));
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    (r0 == 1).then(|| t0.rem_euclid(i128::from(k)) as u64)
}
