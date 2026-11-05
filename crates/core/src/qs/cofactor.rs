//! Splitting cofactors below 2^62 into two large primes, which has to be
//! fast: with double large primes the sieve hands over several per
//! polynomial.
//!
//! ECM with small bounds fixed at compile time goes first: for two primes
//! of 31 bits, 32 curves split all but about 1 in 2000, in an eighth of the
//! time Pollard's rho takes, and are faster at every size down to primes
//! of 20 bits. Rho finishes the few that ECM misses.

/// Montgomery arithmetic modulo an odd `n < 2^62`, with `R = 2^64`. The
/// bound keeps `t + m·n` in [`Mod64::mul`] below 2^128.
struct Mod64 {
    n: u64,
    /// `−n⁻¹ mod 2^64`.
    neg_inv: u64,
    /// `R mod n`, which is 1 in Montgomery form.
    one: u64,
}

impl Mod64 {
    fn new(n: u64) -> Self {
        debug_assert!(!n.is_multiple_of(2) && n < 1 << 62);
        // Newton's iteration, doubling the correct low bits each time.
        let mut inv: u64 = 1;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(inv)));
        }
        Self {
            n,
            neg_inv: inv.wrapping_neg(),
            // 2^64 mod n, as (2^64 − 1) mod n + 1, without 128-bit division.
            one: (u64::MAX % n + 1) % n,
        }
    }

    /// `a·b·R⁻¹ mod n`.
    fn mul(&self, a: u64, b: u64) -> u64 {
        let t = u128::from(a) * u128::from(b);
        let m = (t as u64).wrapping_mul(self.neg_inv);
        let u = ((t + u128::from(m) * u128::from(self.n)) >> 64) as u64;
        if u >= self.n { u - self.n } else { u }
    }

    fn add(&self, a: u64, b: u64) -> u64 {
        let s = a + b;
        if s >= self.n { s - self.n } else { s }
    }

    fn sub(&self, a: u64, b: u64) -> u64 {
        if a >= b { a - b } else { a + self.n - b }
    }

    fn out_of_monty(&self, x: u64) -> u64 {
        self.mul(x, 1)
    }

    fn pow(&self, base: u64, mut e: u64) -> u64 {
        let (mut b, mut r) = (base, self.one);
        while e > 0 {
            if e & 1 == 1 {
                r = self.mul(r, b);
            }
            b = self.mul(b, b);
            e >>= 1;
        }
        r
    }
}

/// Whether odd `n` passes the strong probable-prime test to base 2.
///
/// A composite that passes only costs a failed split, so one base is
/// enough here.
fn probably_prime(m: &Mod64) -> bool {
    let n = m.n;
    let s = (n - 1).trailing_zeros();
    let minus_one = m.sub(0, m.one);
    let mut x = m.pow(m.add(m.one, m.one), (n - 1) >> s);
    if x == m.one || x == minus_one {
        return true;
    }
    for _ in 1..s {
        x = m.mul(x, x);
        if x == minus_one {
            return true;
        }
    }
    false
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Steps of rho per polynomial `x² + c` before giving up: about 2^16
/// finds primes up to 2^32 or so, all the large primes there are.
const RHO_STEPS: u32 = 1 << 16;

/// Products of this many differences go into each gcd.
const BATCH: u32 = 64;

/// A proper factor of the odd composite `n`, by Brent's variant of
/// Pollard's rho, or `None` if it takes too long.
fn rho(m: &Mod64, c: u64) -> Option<u64> {
    let n = m.n;
    let f = |x: u64| m.add(m.mul(x, x), c);
    let (mut y, mut product) = (m.one, m.one);
    let mut steps = 0;
    let mut r = 1;
    loop {
        let x = y;
        for _ in 0..r {
            y = f(y);
        }
        let mut k = 0;
        while k < r {
            let saved = y;
            for _ in 0..BATCH.min(r - k) {
                y = f(y);
                product = m.mul(product, m.sub(x, y));
            }
            let g = gcd(product, n);
            if g != 1 {
                if g != n {
                    return Some(g);
                }
                // The batch overshot: replay it one step at a time.
                let mut y = saved;
                loop {
                    y = f(y);
                    let g = gcd(m.sub(x, y), n);
                    if g != 1 {
                        return (g != n).then_some(g);
                    }
                }
            }
            k += BATCH;
            steps += BATCH;
            if steps > RHO_STEPS {
                return None;
            }
        }
        r *= 2;
    }
}

/// Splits `c < 2^62`, whose prime factors are all larger than its cube
/// root so that it has at most two, into primes `[p, q]` with `p ≤ q`, if
/// it isn't prime itself.
pub fn split(c: u64) -> Option<[u64; 2]> {
    if c.is_multiple_of(2) || c >= 1 << 62 {
        return None;
    }
    let m = Mod64::new(c);
    if probably_prime(&m) {
        return None;
    }
    let curves = if c < 1 << RHO_BELOW_BITS { 0 } else { CURVES };
    let p = (FIRST_SIGMA..FIRST_SIGMA + curves)
        .find_map(|sigma| ecm(&m, sigma))
        .or_else(|| (1..=3).find_map(|seed| rho(&m, seed)))?;
    let (p, q) = (p.min(c / p), p.max(c / p));
    Some([p, q])
}

/// ECM's stage-1 and stage-2 bounds, and the most curves tried. Bounds of
/// 125 and 2500 were faster for primes below 2^26 and slower above.
const B1: u64 = 200;
const B2: u64 = 5_000;
const CURVES: u64 = 32;
/// Below 2^44 a cofactor's smaller prime is below 2^22, which rho finds
/// faster than ECM's first curve.
const RHO_BELOW_BITS: u32 = 44;

/// The first curve's sigma: smaller ones give degenerate curves.
const FIRST_SIGMA: u64 = 6;

/// Stage 2 pairs each prime `k·D ± j` up to B2 with `j` coprime to `D`
/// and below `D/2`, of which there are `BABIES`.
const D: u64 = 210;
const BABIES: usize = 24;

const fn is_prime(n: u64) -> bool {
    let mut d = 2;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    n >= 2
}

const fn gcd_const(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The largest power of each prime up to B1, multiplied together as far
/// as 64 bits hold, and how many words that takes.
const STAGE1: ([u64; 16], usize) = {
    let mut words = [1u64; 16];
    let mut len = 0;
    let mut p = 2;
    while p <= B1 {
        if is_prime(p) {
            let mut q = p;
            while q <= B1 / p {
                q *= p;
            }
            match words[len].checked_mul(q) {
                Some(w) => words[len] = w,
                None => {
                    len += 1;
                    words[len] = q;
                }
            }
        }
        p += 1;
    }
    (words, len + 1)
};

/// The odd `j < D/2` coprime to `D`.
const BABY: [u64; BABIES] = {
    let mut out = [0; BABIES];
    let (mut j, mut i) = (1, 0);
    while j < D / 2 {
        if gcd_const(j, D) == 1 {
            out[i] = j;
            i += 1;
        }
        j += 2;
    }
    out
};

/// The first `k` of stage 2, and for each from it, a mask of the babies
/// `j` to pair with `k·D`. Every prime past B1 then has `k ≥ 1`.
const _: () = assert!(B1 >= D / 2);
const FIRST_K: u64 = (B1 + D / 2) / D;
const GIANTS: usize = ((B2 + D / 2) / D - FIRST_K + 1) as usize;
const PAIRS: [u32; GIANTS] = {
    let mut masks = [0u32; GIANTS];
    let mut p = B1 + 1;
    while p <= B2 {
        if is_prime(p) {
            let k = (p + D / 2) / D;
            let j = p.abs_diff(k * D);
            let mut i = 0;
            while i < BABIES {
                if BABY[i] == j {
                    masks[(k - FIRST_K) as usize] |= 1 << i;
                }
                i += 1;
            }
        }
        p += 1;
    }
    masks
};

/// A point `(X : Z)` of a Montgomery curve, in Montgomery form.
#[derive(Clone, Copy)]
struct Point {
    x: u64,
    z: u64,
}

impl Mod64 {
    /// `x mod n` in Montgomery form, for `x < n`.
    fn to_monty(&self, x: u64) -> u64 {
        ((u128::from(x) << 64) % u128::from(self.n)) as u64
    }

    /// `[2]P` on the curve with `a24 = (A + 2)/4`.
    fn double(&self, p: Point, a24: u64) -> Point {
        let s = self.add(p.x, p.z);
        let d = self.sub(p.x, p.z);
        let (s2, d2) = (self.mul(s, s), self.mul(d, d));
        let t = self.sub(s2, d2);
        Point {
            x: self.mul(s2, d2),
            z: self.mul(t, self.add(d2, self.mul(a24, t))),
        }
    }

    /// `P + Q`, given `P − Q`.
    fn add_points(&self, p: Point, q: Point, diff: Point) -> Point {
        let u = self.mul(self.sub(p.x, p.z), self.add(q.x, q.z));
        let v = self.mul(self.add(p.x, p.z), self.sub(q.x, q.z));
        let (s, d) = (self.add(u, v), self.sub(u, v));
        Point {
            x: self.mul(diff.z, self.mul(s, s)),
            z: self.mul(diff.x, self.mul(d, d)),
        }
    }

    /// `[k]P` for `k ≥ 1`, by the Montgomery ladder.
    fn ladder(&self, p: Point, k: u64, a24: u64) -> Point {
        let (mut r0, mut r1) = (p, self.double(p, a24));
        for bit in (0..63 - k.leading_zeros()).rev() {
            if (k >> bit) & 1 == 1 {
                r0 = self.add_points(r1, r0, p);
                r1 = self.double(r1, a24);
            } else {
                r1 = self.add_points(r1, r0, p);
                r0 = self.double(r0, a24);
            }
        }
        r0
    }
}

/// `x⁻¹ mod n`, or `Err(gcd(x, n))` if there is none.
fn inverse(x: u64, n: u64) -> Result<u64, u64> {
    let (mut r0, mut r1) = (i128::from(n), i128::from(x));
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let k = r0 / r1;
        (r0, r1) = (r1, r0 - k * r1);
        (t0, t1) = (t1, t0 - k * t1);
    }
    if r0 == 1 {
        Ok(t0.rem_euclid(i128::from(n)) as u64)
    } else {
        Err(r0 as u64)
    }
}

/// A proper factor of `n`, from one ECM curve with Suyama's
/// parametrization by `sigma`, if it finds one.
fn ecm(m: &Mod64, sigma: u64) -> Option<u64> {
    let n = m.n;
    let reduce = |x: u128| (x % u128::from(n)) as u64;
    // u = σ² − 5, v = 4σ; the curve has a24 = (v − u)³·(3u + v)/(16u³v)
    // and the point (u³ : v³).
    let u = reduce(u128::from(sigma) * u128::from(sigma) + u128::from(n) - 5);
    let v = reduce(4 * u128::from(sigma));
    let mul = |a: u64, b: u64| reduce(u128::from(a) * u128::from(b));
    let (u3, v3) = (mul(mul(u, u), u), mul(mul(v, v), v));
    let vu = (v + n - u) % n;
    let num = mul(mul(mul(vu, vu), vu), (mul(3, u) + v) % n);
    let den = mul(mul(16, u3), v);
    let a24 = match inverse(den, n) {
        Ok(inv) => mul(num, inv),
        Err(g) => return (g != 1 && g != n).then_some(g),
    };
    let a24 = m.to_monty(a24);
    let mut q = Point {
        x: m.to_monty(u3),
        z: m.to_monty(v3),
    };
    for &e in &STAGE1.0[..STAGE1.1] {
        q = m.ladder(q, e, a24);
    }
    let g = gcd(m.out_of_monty(q.z), n);
    if g != 1 {
        return (g != n).then_some(g);
    }
    // Baby steps [j]Q for odd j below D/2, from [2]Q.
    let two = m.double(q, a24);
    let mut odd = [q; (D / 4) as usize];
    odd[1] = m.add_points(two, q, q);
    for i in 2..odd.len() {
        odd[i] = m.add_points(odd[i - 1], two, odd[i - 2]);
    }
    let baby = BABY.map(|j| odd[(j / 2) as usize]);
    // Giant steps [k·D]Q, pairing each with its babies.
    let step = m.ladder(q, D, a24);
    let mut giant = m.ladder(q, FIRST_K * D, a24);
    let mut next = m.ladder(q, (FIRST_K + 1) * D, a24);
    let mut acc = m.one;
    for &mask in &PAIRS {
        let mut bits = mask;
        while bits != 0 {
            let b = baby[bits.trailing_zeros() as usize];
            bits &= bits - 1;
            acc = m.mul(acc, m.sub(m.mul(giant.x, b.z), m.mul(b.x, giant.z)));
        }
        (giant, next) = (next, m.add_points(next, step, giant));
    }
    let g = gcd(m.out_of_monty(acc), n);
    (g != 1 && g != n).then_some(g)
}
