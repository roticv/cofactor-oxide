//! Splitting cofactors below 2^62 into two large primes, which has to be
//! fast: with double large primes the sieve hands over several per
//! polynomial.

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
            one: ((1u128 << 64) % u128::from(n)) as u64,
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
    let p = (1..=3).find_map(|seed| rho(&m, seed))?;
    let (p, q) = (p.min(c / p), p.max(c / p));
    Some([p, q])
}
