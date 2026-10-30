//! Pollard's p−1 method.
//!
//! By Fermat's little theorem, `a^(p-1) ≡ 1 (mod p)` for a prime `p`, so
//! `a^E ≡ 1 (mod p)` for any multiple `E` of `p − 1`. If `E` is the product
//! of all prime powers up to a bound B1, and `p − 1` has no prime factor
//! above B1, then `gcd(a^E − 1, n)` reveals `p`. This finds `p` no matter how
//! large it is, but only when `p − 1` is smooth.
//!
//! Stage 2 extends this to `p − 1` that is B1-smooth apart from one prime `q`
//! with `B1 < q ≤ B2`, by also trying `(a^E)^q` for each such `q`.

use alloc::vec::Vec;

use crate::arith::{Monty, Uint};
use crate::primes::Primes;

/// Number of primes processed between GCD checks.
const CHUNK: usize = 1024;

/// What a batched GCD check found.
enum Check<const LIMBS: usize> {
    /// No factor yet; carry on.
    Continue,
    Found(Uint<LIMBS>),
    /// Every factor of `n` turned up at the same step, so this base and
    /// these bounds can't separate them.
    Fail,
}

/// Searches for a factor of `n` whose `p − 1` is B1-smooth, apart from at
/// most one prime factor up to B2.
///
/// Returns `None` if no factor was found. That's the usual outcome when no
/// factor has a smooth enough `p − 1`, so this is a cheap extra chance rather
/// than a complete method. Even `n` greater than 2 returns 2 without
/// searching.
///
/// # Panics
///
/// Panics if `b1` is 2^32 or more.
pub fn pollard_pm1<const LIMBS: usize>(n: &Uint<LIMBS>, b1: u64, b2: u64) -> Option<Uint<LIMBS>> {
    assert!(b1 < 1 << 32, "B1 too large");
    let two = Uint::from_u64(2);
    if !n.is_odd() {
        return (*n > two).then_some(two);
    }
    let monty = Monty::new(*n)?;
    let one = monty.one();
    let pow_u64 = |x: &Uint<LIMBS>, e: u64| monty.pow(x, &Uint::<2>::from_u64(e));

    // Stage 1: x = 3^E, where E is the product of the largest power of each
    // prime that is at most B1. GCDs are taken once per CHUNK primes, and a
    // checkpoint is kept so a chunk can be replayed one prime at a time if
    // its GCD is n.
    let mut x = monty.to_monty(&Uint::from_u64(3));
    let mut checkpoint = x;
    let mut chunk = Vec::with_capacity(CHUNK);
    let mut primes = Primes::new(2, b1).peekable();
    while let Some(p) = primes.next() {
        let mut q = p;
        while q <= b1 / p {
            q *= p;
        }
        x = pow_u64(&x, q);
        chunk.push(q);
        if chunk.len() == CHUNK || primes.peek().is_none() {
            match check(&monty, &monty.sub(&x, &one), || {
                // Replay from the checkpoint, testing after every prime power.
                let mut y = checkpoint;
                chunk.iter().map(move |&q| {
                    y = pow_u64(&y, q);
                    monty.sub(&y, &one)
                })
            }) {
                Check::Continue => {}
                Check::Found(g) => return Some(g),
                Check::Fail => return None,
            }
            checkpoint = x;
            chunk.clear();
        }
    }

    // Stage 2: for each prime q in (B1, B2], test x^q. Consecutive primes
    // differ by a small even gap d, so x^q comes from the previous one by a
    // multiplication with x^d, taken from a table that grows as needed.
    // The differences (x^q − 1) are multiplied together and the product's
    // GCD is taken once per chunk.
    let mut primes = Primes::new(b1.max(2) + 1, b2);
    let first = primes.next()?;
    let mut y = pow_u64(&x, first);
    let g = monty.sub(&y, &one).gcd(n);
    if g != Uint::ONE {
        return (g != *n).then_some(g);
    }
    let x2 = monty.square(&x);
    // gap_powers[i] = x^(2i + 2)
    let mut gap_powers = Vec::from([x2]);
    let mut prev = first;
    let mut checkpoint = (y, first);
    let mut acc = one;
    let mut primes = primes.peekable();
    while let Some(q) = primes.next() {
        let step = ((q - prev) / 2) as usize;
        while gap_powers.len() < step {
            let next = monty.mul(gap_powers.last().unwrap(), &x2);
            gap_powers.push(next);
        }
        y = monty.mul(&y, &gap_powers[step - 1]);
        acc = monty.mul(&acc, &monty.sub(&y, &one));
        prev = q;
        chunk.push(q);
        if chunk.len() == CHUNK || primes.peek().is_none() {
            match check(&monty, &acc, || {
                let (mut z, mut at) = checkpoint;
                let gap_powers = &gap_powers;
                chunk.iter().map(move |&q| {
                    z = monty.mul(&z, &gap_powers[((q - at) / 2) as usize - 1]);
                    at = q;
                    monty.sub(&z, &one)
                })
            }) {
                Check::Continue => {}
                Check::Found(g) => return Some(g),
                Check::Fail => return None,
            }
            checkpoint = (y, q);
            chunk.clear();
            acc = one;
        }
    }
    None
}

/// Checks `gcd(acc, n)`, where `acc` is a product of values that each might
/// share a factor with `n`. If the GCD is `n` itself, `replay` regenerates
/// those values one at a time so they can be checked individually.
fn check<const LIMBS: usize, I: Iterator<Item = Uint<LIMBS>>>(
    monty: &Monty<LIMBS>,
    acc: &Uint<LIMBS>,
    replay: impl FnOnce() -> I,
) -> Check<LIMBS> {
    let n = monty.modulus();
    let g = acc.gcd(n);
    if g == Uint::ONE {
        return Check::Continue;
    }
    if g != *n {
        return Check::Found(g);
    }
    for value in replay() {
        let g = value.gcd(n);
        if g == *n {
            break;
        }
        if g != Uint::ONE {
            return Check::Found(g);
        }
    }
    Check::Fail
}
