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
//! most 2^32 products of two digits.

use alloc::vec::Vec;

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

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

    /// The powers `w^j`, `j < n/2`, of a primitive `n`-th root of unity,
    /// or of its inverse, in Montgomery's form.
    fn twiddles(&self, n: usize, inverse: bool) -> Vec<u64> {
        let mut w = pow_mod(self.g, (self.p - 1) / n as u64, self.p);
        if inverse {
            w = pow_mod(w, self.p - 2, self.p);
        }
        let w = self.mont(w);
        let mut t = Vec::with_capacity(n / 2);
        let mut x = self.mont(1);
        for _ in 0..n / 2 {
            t.push(x);
            x = self.mul(x, w);
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
        let r = self.mont(1);
        let n_inv = pow_mod(n as u64 % self.p, self.p - 2, self.p);
        let scale = self.mul(self.mul(r, self.mont(r)), self.mont(n_inv));
        for (a, &b) in a.iter_mut().zip(&b) {
            *a = self.mul(self.mul(*a, b), scale);
        }
        drop(b);
        self.dit(&mut a, &self.twiddles(n, true));
        a
    }

    /// The transform, from natural order to bit-reversed, by
    /// Gentleman–Sande butterflies.
    fn dif(&self, a: &mut [u64], w: &[u64]) {
        let n = a.len();
        let mut m = n;
        while m >= 2 {
            let h = m / 2;
            let stride = n / m;
            for block in a.chunks_exact_mut(m) {
                let (lo, hi) = block.split_at_mut(h);
                for (j, (u, v)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let (x, y) = (*u, *v);
                    *u = self.add(x, y);
                    *v = self.mul(self.sub(x, y), w[j * stride]);
                }
            }
            m = h;
        }
    }

    /// The inverse transform without its factor `1/n`, from bit-reversed
    /// order to natural, by Cooley–Tukey butterflies.
    fn dit(&self, a: &mut [u64], w: &[u64]) {
        let n = a.len();
        let mut m = 2;
        while m <= n {
            let h = m / 2;
            let stride = n / m;
            for block in a.chunks_exact_mut(m) {
                let (lo, hi) = block.split_at_mut(h);
                for (j, (u, v)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let (x, y) = (*u, self.mul(*v, w[j * stride]));
                    *u = self.add(x, y);
                    *v = self.sub(x, y);
                }
            }
            m *= 2;
        }
    }
}

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
                assert_eq!(barrett.reduce(&-&x), if r.is_zero() { r } else { &m - &r });
            }
        }
    }
}
