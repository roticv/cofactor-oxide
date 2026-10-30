//! Williams' p+1 method.
//!
//! This is p−1's counterpart for primes `p` where `p + 1` is smooth. It works
//! with the Lucas sequence `V_0 = 2`, `V_1 = A`, `V_k = A·V_(k-1) − V_(k-2)`.
//! Writing `V_k = α^k + α^(-k)` for a root `α` of `x² − Ax + 1`, `α` lives
//! in a group of order `p + 1` when `A² − 4` is a non-square mod `p`, or of
//! order `p − 1` when it is a square. So `V_E(A) ≡ 2 (mod p)` once that
//! order divides `E`, and `gcd(V_E(A) − 2, n)` reveals `p`.
//!
//! Since whether `A² − 4` is a square mod `p` can't be known in advance,
//! callers should try a couple of seeds whose `A² − 4` have different
//! square-free parts, such as 3 (giving 5) and 4 (giving 12 = 4·3).

use alloc::vec::Vec;

use crate::arith::{Monty, Uint};
use crate::batch::{CHUNK, Check, check};
use crate::primes::Primes;
use crate::stage2::{Stage2Group, stage2};

/// Searches for a factor of `n` whose `p + 1` is B1-smooth apart from at
/// most one prime factor up to B2, using the Lucas sequence with
/// parameter `seed`.
///
/// Like p−1, this usually returns `None`, and it can also find `p` when
/// `p − 1` is smooth instead, depending on the seed. Even `n` greater than 2
/// returns 2 without searching.
///
/// # Panics
///
/// Panics if `b1` is 2^32 or more, or if `seed` is less than 3 (seeds 0, 1
/// and 2 give degenerate sequences).
pub fn williams_pp1<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    seed: u64,
    b1: u64,
    b2: u64,
) -> Option<Uint<LIMBS>> {
    assert!(b1 < 1 << 32, "B1 too large");
    assert!(seed >= 3, "seed must be at least 3");
    let two = Uint::from_u64(2);
    if !n.is_odd() {
        return (*n > two).then_some(two);
    }
    let monty = Monty::new(*n)?;
    let lucas = Lucas {
        monty: &monty,
        two: monty.to_monty(&two),
    };
    let minus_two = |v: &Uint<LIMBS>| monty.sub(v, &lucas.two);

    // Stage 1: x = V_E(seed), applying V_q for each maximal prime power
    // q ≤ B1, since V_ab(A) = V_a(V_b(A)).
    let mut x = monty.to_monty(&Uint::from_u64(seed));
    let mut checkpoint = x;
    let mut chunk = Vec::with_capacity(CHUNK);
    let mut primes = Primes::new(2, b1).peekable();
    while let Some(p) = primes.next() {
        let mut q = p;
        while q <= b1 / p {
            q *= p;
        }
        x = lucas.v(&x, q);
        chunk.push(q);
        if chunk.len() == CHUNK || primes.peek().is_none() {
            match check(&monty, &minus_two(&x), || {
                let mut y = checkpoint;
                chunk.iter().map(move |&q| {
                    y = lucas.v(&y, q);
                    minus_two(&y)
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

    // Stage 2: V_kD − V_j = α^(−kD)·(α^kD − α^j)·(α^kD − α^(−j)), so it
    // vanishes mod p exactly when α^(kD+j) or α^(kD−j) is 1.
    stage2(&lucas, &x, b1, b2)
}

/// Lucas sequence arithmetic `V_k(x) mod n`, in Montgomery form.
#[derive(Clone, Copy)]
struct Lucas<'a, const LIMBS: usize> {
    monty: &'a Monty<LIMBS>,
    /// 2 in Montgomery form.
    two: Uint<LIMBS>,
}

impl<const LIMBS: usize> Lucas<'_, LIMBS> {
    /// Computes `V_k(x)` with a Montgomery ladder, keeping the pair
    /// `(V_m, V_(m+1))` and using
    ///   `V_2m = V_m² − 2` and `V_(2m+1) = V_m·V_(m+1) − x`.
    fn v(&self, x: &Uint<LIMBS>, k: u64) -> Uint<LIMBS> {
        let m = self.monty;
        if k == 0 {
            return self.two;
        }
        let mut a = *x;
        let mut b = m.sub(&m.square(x), &self.two);
        for bit in (0..63 - k.leading_zeros()).rev() {
            let ab = m.sub(&m.mul(&a, &b), x);
            if (k >> bit) & 1 == 1 {
                a = ab;
                b = m.sub(&m.square(&b), &self.two);
            } else {
                b = ab;
                a = m.sub(&m.square(&a), &self.two);
            }
        }
        a
    }
}

/// Lucas sequences as a stage-2 group: `V_m` stands for `α^m` up to
/// inversion, and `V_(a+b) = V_a·V_b − V_(a−b)` plays the part of addition.
impl<const LIMBS: usize> Stage2Group<LIMBS> for Lucas<'_, LIMBS> {
    type Elem = Uint<LIMBS>;

    fn monty(&self) -> &Monty<LIMBS> {
        self.monty
    }

    fn double(&self, a: &Uint<LIMBS>) -> Uint<LIMBS> {
        self.monty.sub(&self.monty.square(a), &self.two)
    }

    fn add(&self, a: &Uint<LIMBS>, b: &Uint<LIMBS>, diff: &Uint<LIMBS>) -> Uint<LIMBS> {
        self.monty.sub(&self.monty.mul(a, b), diff)
    }

    fn mul(&self, a: &Uint<LIMBS>, k: u64) -> Uint<LIMBS> {
        self.v(a, k)
    }

    fn compare(&self, a: &Uint<LIMBS>, b: &Uint<LIMBS>) -> Uint<LIMBS> {
        self.monty.sub(a, b)
    }
}
