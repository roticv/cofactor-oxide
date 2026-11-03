//! Kleinjung's polynomial selection, stage 1: candidates with a small
//! leading coefficient and a small next one.
//!
//! With `g = l·x − m`, `N` has the expansion `N = Σ aᵢ·mⁱ·l^(d−i)`: each
//! `aᵢ` for `i < d` is fixed modulo `l` and otherwise free, and comes out
//! about `rᵢ/mⁱ` for the remainder `rᵢ` at that step. The first remainder is
//! `(N − a_d·m^d)/l`, so `a_{d−1}` is about `d·a_d·t/l` for `m = m₀ + t`
//! near `m₀ = (N/a_d)^(1/d)`: small, if `l` is large and `t` small, where
//! base-m gives `a_{d−1}` as large as `m`. It also needs
//! `l² | N − a_d·m^d`.
//!
//! For `l = q·ℓ₁·ℓ₂`, this holds when `m ≡ ρ (mod q²)` and
//! `m ≡ rᵢ (mod ℓᵢ²)` for roots of `a_d·x^d ≡ N`. Here `q`, the special-q,
//! is a product of small primes, whose roots combine into many `ρ`; and
//! `ℓ₁`, `ℓ₂` are primes around `P`. For each `ρ`, listing, for every such
//! prime `ℓ` and root `r`, the `u` in `[−U, U]` with
//! `m₀ + t_ρ + q²·u ≡ r (mod ℓ²)`, two entries with the same `u` from
//! different primes give a candidate. Then `a_{d−2}`, about
//! `d·a_d·m·t/l²`, is smaller by `q²` than without `q` for the same chance
//! of a collision.
//!
//! [Kleinjung, *Polynomial selection*, CADO workshop, 2008.]

use alloc::vec;
use alloc::vec::Vec;

use super::modp::{inv, roots};
use super::poly::to_u128;
use crate::arith::Uint;
use crate::factor::root;
use crate::primes::Primes;

/// Special-q primes are taken from below this.
const Q_PRIMES: u64 = 200;

/// Roots of the special-q wanted, and the most bits of `q`.
const Q_ROOTS: usize = 64;
const Q_BITS: u32 = 20;

/// A candidate: `f`, `m` and `l`, with `F(m, l) = N`.
pub type Candidate = (Vec<i128>, u128, u128);

/// Candidates for `n` of degree `d` with leading coefficient `ad`, from
/// primes `ℓ` in `[p0, 2·p0)` and `|u| ≤ u_max`.
pub fn candidates<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    d: usize,
    ad: u64,
    p0: u64,
    u_max: i64,
) -> Vec<Candidate> {
    let (quot, _) = n.div_rem(&Uint::from_u64(ad));
    let Some(m0) = to_u128(&root(&quot, d as u32)).filter(|&m| m < 1 << 96) else {
        return Vec::new();
    };
    // The special-q and its roots modulo q².
    let (mut q, mut rho) = (1u64, vec![0u64]);
    for p in Primes::new(3, Q_PRIMES) {
        if rho.len() >= Q_ROOTS {
            break;
        }
        let rs = roots_mod_square(n, ad, p, d);
        if rs.len() < 2 || ad.is_multiple_of(p) || (q * p).ilog2() >= Q_BITS {
            continue;
        }
        // CRT: x ≡ a (mod q²), x ≡ r (mod p²).
        let (q2, p2) = (u128::from(q * q), u128::from(p * p));
        let q2_inv = u128::from(inverse((q2 % p2) as u64, p2 as u64).unwrap_or(0));
        rho = rho
            .iter()
            .flat_map(|&a| {
                rs.iter().map(move |&r| {
                    let k = (u128::from(r) + p2 - u128::from(a) % p2) % p2 * q2_inv % p2;
                    (u128::from(a) + q2 * k) as u64
                })
            })
            .collect();
        q *= p;
    }
    let q2 = u128::from(q) * u128::from(q);
    // The large primes, each root's m₀-relative residue, and 1/q² mod ℓ².
    let mut large: Vec<(u64, u64, u64)> = Vec::new();
    for ell in Primes::new(p0, 2 * p0) {
        if ad.is_multiple_of(ell) || q.is_multiple_of(ell) {
            continue;
        }
        let ell2 = ell * ell;
        let Some(inv_q2) = inverse((q2 % u128::from(ell2)) as u64, ell2) else {
            continue;
        };
        let m0_2 = (m0 % u128::from(ell2)) as u64;
        for r in roots_mod_square(n, ad, ell, d) {
            large.push((ell, (r + ell2 - m0_2) % ell2, inv_q2));
        }
    }
    let mut found = Vec::new();
    let mut entries: Vec<(i64, u64)> = Vec::new();
    for &rq in &rho {
        // t ≡ ρ − m₀ (mod q²), the smallest such t_ρ.
        let t_rho = ((u128::from(rq) + q2 - m0 % q2) % q2) as i128;
        let t_rho = if t_rho > (q2 / 2) as i128 {
            t_rho - q2 as i128
        } else {
            t_rho
        };
        entries.clear();
        for &(ell, e, inv_q2) in &large {
            let ell2 = ell * ell;
            // u ≡ (e − t_ρ)/q² (mod ℓ²), from −u_max up.
            let base = (i128::from(e) - t_rho).rem_euclid(i128::from(ell2)) as u64;
            let res = (u128::from(base) * u128::from(inv_q2) % u128::from(ell2)) as i64;
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
                let m = (m0 as i128 + t) as u128;
                let l = u128::from(q) * u128::from(ell1) * u128::from(ell2);
                if let Some(f) = expand(n, ad, m, l, d) {
                    found.push((f, m, l));
                }
            }
        }
    }
    found
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
