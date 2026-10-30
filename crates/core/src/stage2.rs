//! Baby-step giant-step stage 2, shared by p+1 and ECM.
//!
//! After stage 1 there is an element `x` whose order mod `p` (in the hidden
//! group mod `p`) may be B1-smooth apart from a single prime `q ≤ B2`.
//! Writing `q = kD ± j` with `j ≤ D/2`, `[q]x` is the identity exactly when
//! `[kD]x = ∓[j]x`. Both groups here only see elements up to sign (the
//! x-coordinate of a curve point, or `V_m = α^m + α^(−m)` for Lucas
//! sequences), so that's a test of whether `[kD]x` and `[j]x` are equal,
//! which costs one comparison per prime.
//!
//! The baby steps `[j]x` are computed once. The giant steps `[kD]x` advance
//! by one addition each. Neither group can add two arbitrary elements, only
//! two whose difference is known, which this arrangement always provides.

use alloc::vec::Vec;

use crate::arith::{Monty, Uint};
use crate::batch::{CHUNK, Check, check};
use crate::primes::Primes;

/// Giant-step size: 2·3·5·7·11. Every prime above 11 is `kD ± j` with `j`
/// odd, coprime to D, and at most D/2.
const D: u64 = 2310;

/// A group whose elements are known only up to sign, supporting the
/// operations stage 2 needs.
pub(crate) trait Stage2Group<const LIMBS: usize> {
    type Elem: Copy;

    fn monty(&self) -> &Monty<LIMBS>;

    /// Computes `[2]a`.
    fn double(&self, a: &Self::Elem) -> Self::Elem;

    /// Computes `a + b`, given `a − b`.
    fn add(&self, a: &Self::Elem, b: &Self::Elem, diff: &Self::Elem) -> Self::Elem;

    /// Computes `[k]a`.
    fn mul(&self, a: &Self::Elem, k: u64) -> Self::Elem;

    /// A value that is `≡ 0 (mod p)` exactly when `a = ±b` mod `p`.
    fn compare(&self, a: &Self::Elem, b: &Self::Elem) -> Uint<LIMBS>;
}

/// Runs stage 2 on `x` for the primes in `(b1, b2]`, returning a factor of
/// `n` if one turns up.
pub(crate) fn stage2<const LIMBS: usize, G: Stage2Group<LIMBS>>(
    group: &G,
    x: &G::Elem,
    b1: u64,
    b2: u64,
) -> Option<Uint<LIMBS>> {
    let monty = group.monty();
    let mut primes = Primes::new(b1.max(2) + 1, b2).peekable();
    let &first = primes.peek()?;

    // Baby steps: baby[i] = [2i + 1]x for odd j = 2i + 1 ≤ D/2, using
    // [j + 2]x = [j]x + [2]x, whose difference is [j − 2]x. For j = 1 that
    // difference is [−1]x, which is x up to sign.
    let x2 = group.double(x);
    let mut baby = Vec::with_capacity((D / 4 + 1) as usize);
    baby.push(*x);
    baby.push(group.add(&x2, x, x));
    while (baby.len() as u64) * 2 < D / 2 {
        let i = baby.len();
        baby.push(group.add(&baby[i - 1], &x2, &baby[i - 2]));
    }

    // Giant steps: [kD]x, with [(k + 1)D]x = [kD]x + [D]x, whose difference
    // is [(k − 1)D]x. That difference is the identity when k = 1, and the
    // x-only curve formulas can't add with an identity difference (they'd
    // need to double), so the steps to k = 1 and k = 2 are taken directly.
    let round = |q: u64| (q + D / 2) / D;
    let step = group.mul(x, D);
    let mut k = round(first);
    let mut giant = group.mul(x, k * D);
    let mut giant_prev = group.mul(x, k.saturating_sub(1) * D);

    let mut acc = monty.one();
    let mut values = Vec::with_capacity(CHUNK);
    while let Some(q) = primes.next() {
        while k < round(q) {
            let next = match k {
                0 => step,
                1 => group.double(&step),
                _ => group.add(&giant, &step, &giant_prev),
            };
            giant_prev = giant;
            giant = next;
            k += 1;
        }
        let j = q.abs_diff(k * D);
        let value = group.compare(&giant, &baby[(j / 2) as usize]);
        acc = monty.mul(&acc, &value);
        values.push(value);
        if values.len() == CHUNK || primes.peek().is_none() {
            match check(monty, &acc, || values.iter().copied()) {
                Check::Continue => {}
                Check::Found(g) => return Some(g),
                Check::Fail => return None,
            }
            values.clear();
            acc = monty.one();
        }
    }
    None
}
