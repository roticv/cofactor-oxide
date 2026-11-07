//! Multiplication and reduction of integers of millions of bits, for the
//! algebraic square root. `num-bigint` multiplies by Toom-3 at best, about
//! `n^1.5`: two numbers of 64 million bits take 6 s, and a remainder
//! 15 s. Here large products go through number-theoretic transforms, in
//! `O(n log n)`, and remainders by a fixed modulus through Barrett's
//! method, a reciprocal and two products.
//!
//! The digits are 32 bits, and each transform is taken modulo two primes
//! `c·2^32 + 1` below 2^62, whose product, past 2^123, bounds the
//! convolution's coefficients for up to 2^32 digits: each is a sum of at
//! most 2^32 products of two digits. A product of polynomials sums a few
//! such convolutions with signs, still well inside half of it.
//!
//! Most of the square root's time is in transforms, and they are limited
//! by memory more than by arithmetic, so the gains came from doing fewer:
//! a product of polynomials transforms each coefficient once
//! ([`poly_mul`]), and a reduction of several numbers transforms the
//! reciprocal and modulus once ([`Barrett::reduce_all`]). Each level of a
//! transform reads its twiddle factors in order, and the levels within 64
//! KB blocks run a block at a time, in cache. On one thread, as in wasm,
//! the two primes are taken one after the other, which halves the memory.
//! On RSA-110 these took the first dependency's square root from 63 s to
//! 18 s on 6 threads, and from 364 s to 195 s in wasm, where it still fits
//! in the memory the matrix step leaves.

use alloc::vec;
use alloc::vec::Vec;

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

use crate::parallel::{Parallel, map_dynamic};

/// The primes, with a primitive root of each.
const PRIMES: [(u64, u64); 2] = [(0x3fff_ffee_0000_0001, 3), (0x3fff_ffb4_0000_0001, 19)];

/// Below this many digits in the smaller operand, `num-bigint` multiplies.
const THRESHOLD: usize = 1 << 11;

/// `a·b`.
pub fn mul(a: &BigInt, b: &BigInt) -> BigInt {
    let sign = match (a.sign(), b.sign()) {
        (Sign::NoSign, _) | (_, Sign::NoSign) => return BigInt::zero(),
        (x, y) if x == y => Sign::Plus,
        _ => Sign::Minus,
    };
    BigInt::from_biguint(sign, mul_unsigned(a.magnitude(), b.magnitude()))
}

/// `a·b`, for magnitudes.
pub fn mul_unsigned(a: &BigUint, b: &BigUint) -> BigUint {
    let (x, y) = (a.to_u32_digits(), b.to_u32_digits());
    if x.len().min(y.len()) < THRESHOLD {
        return a * b;
    }
    let len = x.len() + y.len();
    let n = len.next_power_of_two();
    let [f0, f1] = PRIMES.map(|(p, g)| Field::new(p, g));
    let (r0, r1) = (f0.convolve(&x, &y, n), f1.convolve(&x, &y, n));
    // The coefficient is r0 + p0·t for t = (r1 − r0)/p0 mod p1.
    let (p0, p1) = (f0.p, f1.p);
    let inv = f1.mont(pow_mod(p0 % p1, p1 - 2, p1));
    let mut digits = Vec::with_capacity(len);
    let mut carry = 0u128;
    for (&a, &b) in r0.iter().zip(&r1).take(len) {
        let a1 = if a >= p1 { a - p1 } else { a };
        let t = f1.mul(f1.sub(b, a1), inv);
        let v = u128::from(a) + u128::from(p0) * u128::from(t) + carry;
        digits.push(v as u32);
        carry = v >> 32;
    }
    debug_assert_eq!(carry, 0);
    BigUint::new(digits)
}

/// The product of the polynomials `a` and `b`, with coefficients lowest
/// first: `c_k = Σ a_i·b_(k−i)`. Each coefficient is transformed once, at a
/// length that fits every product, rather than once for each product it
/// takes part in; each `c_k` is summed in the transform and transformed
/// back once; and squares transform one operand. The transforms are shared
/// out by `par`. `None` if the coefficients are too small for transforms
/// to pay.
pub fn poly_mul(a: &[BigInt], b: &[BigInt], par: &impl Parallel) -> Option<Vec<BigInt>> {
    let square = core::ptr::eq(a, b);
    let digits = |x: &[BigInt]| -> Vec<(bool, Vec<u32>)> {
        x.iter()
            .map(|c| (c.sign() == Sign::Minus, c.magnitude().to_u32_digits()))
            .collect()
    };
    let da = digits(a);
    let db = if square { Vec::new() } else { digits(b) };
    let db = if square { &da } else { &db };
    let longest = |d: &[(bool, Vec<u32>)]| d.iter().map(|(_, x)| x.len()).max().unwrap_or(0);
    let (la, lb) = (longest(&da), longest(db));
    if la.min(lb) < THRESHOLD {
        return None;
    }
    let len = la + lb + 1;
    let n = len.next_power_of_two();
    let out = a.len() + b.len() - 1;
    let [f0, f1] = PRIMES.map(|(p, g)| Field::new(p, g));
    // The terms of each c_k: (i, j, negative).
    let terms: Vec<Vec<(usize, usize, bool)>> = (0..out)
        .map(|k| {
            (k.saturating_sub(b.len() - 1)..=k.min(a.len() - 1))
                .map(|i| (i, k - i))
                .filter(|&(i, j)| !da[i].1.is_empty() && !db[j].1.is_empty())
                .map(|(i, j)| (i, j, da[i].0 != db[j].0))
                .collect()
        })
        .collect();
    let job = Convolution {
        da: &da,
        db,
        square,
        terms: &terms,
        n,
        len,
    };
    // One prime at a time, which halves the memory for little time: the
    // first's residues are kept, and each of the second's is combined with
    // its counterpart by the CRT as soon as it is ready.
    let r0 = job.residues(&f0, par, |_, r| r);
    let (p0, p1) = (f0.p, f1.p);
    let inv = f1.mont(pow_mod(p0 % p1, p1 - 2, p1));
    let modulus = i128::from(p0) * i128::from(p1);
    Some(job.residues(&f1, par, |k, r1| {
        // Each position's sum is below p0·p1/2 in size, so the CRT's value
        // past half of p0·p1 stands for a negative one.
        let mut digits = Vec::with_capacity(len);
        let mut carry = 0i128;
        for (&x, &y) in r0[k].iter().zip(&r1) {
            let x1 = if x >= p1 { x - p1 } else { x };
            let t = f1.mul(f1.sub(y, x1), inv);
            let mut v = i128::from(x) + i128::from(p0) * i128::from(t);
            if v > modulus / 2 {
                v -= modulus;
            }
            let v = v + carry;
            digits.push(v as u32);
            carry = v >> 32;
        }
        let low = BigInt::from_biguint(Sign::Plus, BigUint::new(digits));
        low + (BigInt::from(carry) << (32 * len))
    }))
}

/// A product of polynomials for [`poly_mul`]: the operands' coefficients
/// as signs and digits, and the terms `(i, j, negative)` of each `c_k`.
struct Convolution<'a> {
    da: &'a [(bool, Vec<u32>)],
    db: &'a [(bool, Vec<u32>)],
    square: bool,
    terms: &'a [Vec<(usize, usize, bool)>],
    n: usize,
    len: usize,
}

impl Convolution<'_> {
    /// `then(k, c_k mod p)` for each `k`: each coefficient transformed once,
    /// each `c_k` summed in the transform and transformed back.
    fn residues<T: Send>(
        &self,
        f: &Field,
        par: &impl Parallel,
        then: impl Fn(usize, Vec<u64>) -> T + Sync,
    ) -> Vec<T> {
        let n = self.n;
        let forward = f.twiddles(n, false);
        let transform = |d: &[(bool, Vec<u32>)]| {
            map_dynamic(par, d.len(), |i| {
                if d[i].1.is_empty() {
                    return Vec::new();
                }
                let mut v: Vec<u64> = d[i].1.iter().map(|&x| u64::from(x)).collect();
                v.resize(n, 0);
                f.dif(&mut v, &forward);
                v
            })
        };
        let ta = transform(self.da);
        let tb = if self.square {
            Vec::new()
        } else {
            transform(self.db)
        };
        let tb = if self.square { &ta } else { &tb };
        drop(forward);
        let inverse = f.twiddles(n, true);
        let scale = f.scale(n);
        map_dynamic(par, self.terms.len(), |k| {
            let mut acc = vec![0u64; n];
            for &(i, j, negative) in &self.terms[k] {
                for ((c, &x), &y) in acc.iter_mut().zip(&ta[i]).zip(&tb[j]) {
                    let t = f.mul(x, y);
                    *c = if negative { f.sub(*c, t) } else { f.add(*c, t) };
                }
            }
            for c in acc.iter_mut() {
                *c = f.mul(*c, scale);
            }
            f.dit(&mut acc, &inverse);
            acc.truncate(self.len);
            then(k, acc)
        })
    }
}

/// `x·c` for each `x` in `xs`, transforming `c` once for them all, with the
/// products shared out by `par`.
fn mul_each(xs: &[BigUint], c: &BigUint, par: &impl Parallel) -> Vec<BigUint> {
    let cd = c.to_u32_digits();
    let xd: Vec<Vec<u32>> = xs.iter().map(BigUint::to_u32_digits).collect();
    let longest = xd.iter().map(Vec::len).max().unwrap_or(0);
    if longest.min(cd.len()) < THRESHOLD {
        return map_dynamic(par, xs.len(), |i| &xs[i] * c);
    }
    let len = longest + cd.len();
    let n = len.next_power_of_two();
    let fields = PRIMES.map(|(p, g)| Field::new(p, g));
    let forward = map_dynamic(par, 2, |f| fields[f].twiddles(n, false));
    let inverse = map_dynamic(par, 2, |f| fields[f].twiddles(n, true));
    let load = |d: &[u32]| {
        let mut v: Vec<u64> = d.iter().map(|&x| u64::from(x)).collect();
        v.resize(n, 0);
        v
    };
    // c transformed modulo prime f, and x_i·c modulo prime f given that.
    let transform_c = |f: usize| {
        let mut v = load(&cd);
        fields[f].dif(&mut v, &forward[f]);
        v
    };
    let residue = |f: usize, ct: &[u64], i: usize| {
        if xd[i].is_empty() {
            return Vec::new();
        }
        let field = &fields[f];
        let mut v = load(&xd[i]);
        field.dif(&mut v, &forward[f]);
        let scale = field.scale(n);
        for (a, &b) in v.iter_mut().zip(ct) {
            *a = field.mul(field.mul(*a, b), scale);
        }
        field.dit(&mut v, &inverse[f]);
        v.truncate(len);
        v
    };
    let [f0, f1] = &fields;
    let (p0, p1) = (f0.p, f1.p);
    let inv = f1.mont(pow_mod(p0 % p1, p1 - 2, p1));
    let crt = |r0: &[u64], r1: &[u64]| {
        let mut digits = Vec::with_capacity(len);
        let mut carry = 0u128;
        for (&a, &b) in r0.iter().zip(r1) {
            let a1 = if a >= p1 { a - p1 } else { a };
            let t = f1.mul(f1.sub(b, a1), inv);
            let v = u128::from(a) + u128::from(p0) * u128::from(t) + carry;
            digits.push(v as u32);
            carry = v >> 32;
        }
        BigUint::new(digits)
    };
    if par.threads() > 1 {
        // Both primes' jobs at once, job f·len + i, for the threads to share.
        let ct = map_dynamic(par, 2, transform_c);
        let r = map_dynamic(par, 2 * xs.len(), |job| {
            let (f, i) = (job / xs.len(), job % xs.len());
            residue(f, &ct[f], i)
        });
        map_dynamic(par, xs.len(), |i| crt(&r[i], &r[xs.len() + i]))
    } else {
        // One prime at a time, each of the second's residues combined as
        // soon as it is ready, which halves the memory.
        let ct = transform_c(0);
        let r0: Vec<Vec<u64>> = (0..xs.len()).map(|i| residue(0, &ct, i)).collect();
        let ct = transform_c(1);
        (0..xs.len())
            .map(|i| crt(&r0[i], &residue(1, &ct, i)))
            .collect()
    }
}

/// `x^e mod p`, for the constants.
fn pow_mod(mut x: u64, mut e: u64, p: u64) -> u64 {
    let mul = |a: u64, b: u64| (u128::from(a) * u128::from(b) % u128::from(p)) as u64;
    let mut r = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = mul(r, x);
        }
        x = mul(x, x);
        e >>= 1;
    }
    r
}

/// Arithmetic modulo `p < 2^62`, multiplying in Montgomery's form: `mul`
/// gives `a·b/2^64`, so a factor in Montgomery's form, `b·2^64`, leaves
/// the other as it was.
struct Field {
    p: u64,
    /// `−p⁻¹ mod 2^64`.
    neg_inv: u64,
    g: u64,
}

impl Field {
    fn new(p: u64, g: u64) -> Self {
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
        }
        Self {
            p,
            neg_inv: inv.wrapping_neg(),
            g,
        }
    }

    /// `x` in Montgomery's form.
    fn mont(&self, x: u64) -> u64 {
        ((u128::from(x) << 64) % u128::from(self.p)) as u64
    }

    fn mul(&self, a: u64, b: u64) -> u64 {
        let t = u128::from(a) * u128::from(b);
        let m = (t as u64).wrapping_mul(self.neg_inv);
        let u = ((t + u128::from(m) * u128::from(self.p)) >> 64) as u64;
        if u >= self.p { u - self.p } else { u }
    }

    fn add(&self, a: u64, b: u64) -> u64 {
        let s = a + b;
        if s >= self.p { s - self.p } else { s }
    }

    fn sub(&self, a: u64, b: u64) -> u64 {
        if a >= b { a - b } else { a + self.p - b }
    }

    /// The twiddle factors of each level, in Montgomery's form: for blocks
    /// of `m = 2h`, the powers `w_m^j`, `j < h`, of a primitive `m`-th root
    /// of unity (or of its inverse) at `[h, 2h)`, so that each level reads
    /// its own in order rather than striding through the largest's.
    fn twiddles(&self, n: usize, inverse: bool) -> Vec<u64> {
        let mut w = pow_mod(self.g, (self.p - 1) / n as u64, self.p);
        if inverse {
            w = pow_mod(w, self.p - 2, self.p);
        }
        let w = self.mont(w);
        let mut t = vec![0u64; n.max(2)];
        // The largest level's, at [n/2, n); each smaller level takes every
        // other one of the next.
        let mut x = self.mont(1);
        for j in 0..n / 2 {
            t[n / 2 + j] = x;
            x = self.mul(x, w);
        }
        let mut h = n / 4;
        while h >= 1 {
            for j in 0..h {
                t[h + j] = t[2 * h + 2 * j];
            }
            h /= 2;
        }
        t
    }

    /// `x * y` over the integers modulo `p`, of length `n`, a power of 2.
    fn convolve(&self, x: &[u32], y: &[u32], n: usize) -> Vec<u64> {
        let load = |d: &[u32]| {
            let mut v: Vec<u64> = d.iter().map(|&d| u64::from(d)).collect();
            v.resize(n, 0);
            v
        };
        let (mut a, mut b) = (load(x), load(y));
        let forward = self.twiddles(n, false);
        self.dif(&mut a, &forward);
        self.dif(&mut b, &forward);
        drop(forward);
        // a·b/2^64 here, then 2^64/n before the inverse transform's 1/n
        // is due: the scale is 2^128/n in Montgomery's form.
        let scale = self.scale(n);
        for (a, &b) in a.iter_mut().zip(&b) {
            *a = self.mul(self.mul(*a, b), scale);
        }
        drop(b);
        self.dit(&mut a, &self.twiddles(n, true));
        a
    }

    /// What a pointwise product `a·b/2^64` is multiplied by before the
    /// inverse transform: `2^128/n` in Montgomery's form, which leaves
    /// `a·b/n`, the inverse transform's `1/n` included.
    fn scale(&self, n: usize) -> u64 {
        let r = self.mont(1);
        let n_inv = pow_mod(n as u64 % self.p, self.p - 2, self.p);
        self.mul(self.mul(r, self.mont(r)), self.mont(n_inv))
    }

    /// One level of Gentleman–Sande butterflies, over blocks of `m` in `a`.
    fn dif_level(&self, a: &mut [u64], w: &[u64], m: usize) {
        let h = m / 2;
        let w = &w[h..m];
        for block in a.chunks_exact_mut(m) {
            let (lo, hi) = block.split_at_mut(h);
            for ((u, v), &w) in lo.iter_mut().zip(hi.iter_mut()).zip(w) {
                let (x, y) = (*u, *v);
                *u = self.add(x, y);
                *v = self.mul(self.sub(x, y), w);
            }
        }
    }

    /// One level of Cooley–Tukey butterflies, as [`Field::dif_level`].
    fn dit_level(&self, a: &mut [u64], w: &[u64], m: usize) {
        let h = m / 2;
        let w = &w[h..m];
        for block in a.chunks_exact_mut(m) {
            let (lo, hi) = block.split_at_mut(h);
            for ((u, v), &w) in lo.iter_mut().zip(hi.iter_mut()).zip(w) {
                let (x, y) = (*u, self.mul(*v, w));
                *u = self.add(x, y);
                *v = self.sub(x, y);
            }
        }
    }

    /// The transform, from natural order to bit-reversed, by
    /// Gentleman–Sande butterflies: the levels whose blocks are larger than
    /// `CACHE_BLOCK` across the whole array, then the rest one block at a
    /// time, while it is in cache.
    fn dif(&self, a: &mut [u64], w: &[u64]) {
        let n = a.len();
        let mut m = n;
        while m > CACHE_BLOCK {
            self.dif_level(a, w, m);
            m /= 2;
        }
        for block in a.chunks_exact_mut(m) {
            let mut k = m;
            while k >= 2 {
                self.dif_level(block, w, k);
                k /= 2;
            }
        }
    }

    /// The inverse transform without its factor `1/n`, from bit-reversed
    /// order to natural, by Cooley–Tukey butterflies: the small levels a
    /// block at a time, then the rest, as [`Field::dif`] in reverse.
    fn dit(&self, a: &mut [u64], w: &[u64]) {
        let n = a.len();
        let b = n.min(CACHE_BLOCK);
        for block in a.chunks_exact_mut(b) {
            let mut k = 2;
            while k <= b {
                self.dit_level(block, w, k);
                k *= 2;
            }
        }
        let mut m = 2 * b;
        while m <= n {
            self.dit_level(a, w, m);
            m *= 2;
        }
    }
}

/// Transforms finish their levels within blocks of this many elements,
/// 64 KB, which stay in the L1 cache.
const CACHE_BLOCK: usize = 1 << 13;

/// Bits of headroom: [`Barrett::reduce`] is quick for `|x|` below
/// `2^(2k + SLACK)`, for a `k`-bit modulus, and divides past that.
const SLACK: u64 = 1 << 12;

/// Reduction modulo a fixed `m` of `k` bits by Barrett's method: with
/// `μ ≈ 2^(2k + SLACK)/m`, the quotient of `x` is within a few of
/// `⌊x/2^(k−1)⌋·μ/2^(k + 1 + SLACK)`.
pub struct Barrett {
    m: BigUint,
    k: u64,
    mu: BigUint,
}

impl Barrett {
    pub fn new(m: &BigInt) -> Self {
        let m = m.magnitude().clone();
        let k = m.bits();
        let mu = reciprocal(&m, 2 * k + SLACK);
        Self { m, k, mu }
    }

    /// `x mod m` for each `x` in `xs`, in `[0, m)`, with the products by the
    /// reciprocal and by `m` each transforming them once for all of `xs`,
    /// and shared out by `par`.
    pub fn reduce_all(&self, xs: &[BigInt], par: &impl Parallel) -> Vec<BigInt> {
        if xs.iter().any(|x| x.magnitude().bits() > 2 * self.k + SLACK) {
            return map_dynamic(par, xs.len(), |i| self.reduce(&xs[i]));
        }
        let shifted: Vec<BigUint> = xs.iter().map(|x| x.magnitude() >> (self.k - 1)).collect();
        let q: Vec<BigUint> = mul_each(&shifted, &self.mu, par)
            .into_iter()
            .map(|t| t >> (self.k + 1 + SLACK))
            .collect();
        let qm = mul_each(&q, &self.m, par);
        let m = BigInt::from(self.m.clone());
        xs.iter()
            .zip(qm)
            .map(|(x, qm)| {
                let mut r = BigInt::from(x.magnitude().clone()) - BigInt::from(qm);
                while r.sign() == Sign::Minus {
                    r += &m;
                }
                while r >= m {
                    r -= &m;
                }
                if x.sign() == Sign::Minus && !r.is_zero() {
                    &m - r
                } else {
                    r
                }
            })
            .collect()
    }

    /// `x mod m`, in `[0, m)`.
    pub fn reduce(&self, x: &BigInt) -> BigInt {
        let mag = x.magnitude();
        if mag.bits() > 2 * self.k + SLACK {
            let m = BigInt::from(self.m.clone());
            let r = x % &m;
            return if r.sign() == Sign::Minus { r + m } else { r };
        }
        let q = mul_unsigned(&(mag >> (self.k - 1)), &self.mu) >> (self.k + 1 + SLACK);
        let mut r = BigInt::from(mag.clone()) - BigInt::from(mul_unsigned(&q, &self.m));
        let m = BigInt::from(self.m.clone());
        while r.sign() == Sign::Minus {
            r += &m;
        }
        while r >= m {
            r -= &m;
        }
        if x.sign() == Sign::Minus && !r.is_zero() {
            m - r
        } else {
            r
        }
    }
}

/// About `2^e/m`, within a few, for `e` at least `m`'s `k` bits: its
/// `e − k + 1` bits from the leading half by Newton's iteration, which
/// doubles the bits that are right.
fn reciprocal(m: &BigUint, e: u64) -> BigUint {
    let k = m.bits();
    let bits = e - k + 1;
    // Only m's leading bits matter, past those of the result.
    let t = k.min(bits + 64);
    let (m, e) = (m >> (k - t), e - (k - t));
    if bits <= 1 << 12 {
        return (BigUint::one() << e) / m;
    }
    let half = bits / 2 + 64;
    // y ≈ 2^e2/m with `half` bits, so x0 ≈ 2^e/m to about 2^−half.
    let e2 = t + half - 1;
    let x0 = BigInt::from(reciprocal(&m, e2) << (e - e2));
    let m = BigInt::from(m);
    let err = (BigInt::one() << e) - mul(&m, &x0);
    let x1 = &x0 + (mul(&x0, &err) >> e);
    x1.magnitude().clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn number(digits: usize, seed: u64) -> BigUint {
        let mut x = seed | 1;
        BigUint::new(
            (0..digits)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u32
                })
                .collect(),
        )
    }

    #[test]
    fn products_match() {
        for (la, lb) in [
            (1 << 11, 1 << 11),
            (5000, 3001),
            (1 << 14, 1 << 12),
            (40_000, 39_999),
        ] {
            let (a, b) = (number(la, 1), number(lb, 2));
            assert_eq!(mul_unsigned(&a, &b), &a * &b, "{la} × {lb}");
        }
        let all_ones = BigUint::new(vec![u32::MAX; 1 << 12]);
        assert_eq!(mul_unsigned(&all_ones, &all_ones), &all_ones * &all_ones);
    }

    #[test]
    fn polynomial_products_match() {
        let coeff = |digits: usize, seed: u64, negative: bool| {
            let x = BigInt::from(number(digits, seed));
            if negative { -x } else { x }
        };
        let a: Vec<BigInt> = (0..4)
            .map(|i| coeff(3000 + 500 * i, i as u64 + 1, i % 2 == 1))
            .collect();
        let b: Vec<BigInt> = (0..3)
            .map(|i| coeff(2500 + 700 * i, i as u64 + 9, i == 0))
            .chain([BigInt::zero()])
            .collect();
        let naive = |x: &[BigInt], y: &[BigInt]| {
            let mut c = vec![BigInt::zero(); x.len() + y.len() - 1];
            for (i, xi) in x.iter().enumerate() {
                for (j, yj) in y.iter().enumerate() {
                    c[i + j] += xi * yj;
                }
            }
            c
        };
        let serial = crate::parallel::Serial;
        assert_eq!(poly_mul(&a, &b, &serial).unwrap(), naive(&a, &b));
        assert_eq!(poly_mul(&a, &a, &serial).unwrap(), naive(&a, &a));
    }

    #[test]
    fn reductions_match() {
        for digits in [100, 1 << 13, 30_000] {
            let m = BigInt::from(number(digits, 3));
            let barrett = Barrett::new(&m);
            for x in [
                number(2 * digits, 4),
                number(2 * digits + 100, 5),
                number(digits / 2, 6),
            ] {
                let x = BigInt::from(x);
                let r = &x % &m;
                assert_eq!(barrett.reduce(&x), r);
                assert_eq!(
                    barrett.reduce(&-&x),
                    if r.is_zero() { r.clone() } else { &m - &r }
                );
                let all = barrett.reduce_all(&[x.clone(), -&x], &crate::parallel::Serial);
                assert_eq!(all, [barrett.reduce(&x), barrett.reduce(&-&x)]);
            }
        }
    }
}
