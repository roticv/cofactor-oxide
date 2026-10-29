//! Pollard's rho algorithm.
//!
//! Iterating `x ↦ x² + c (mod n)` gives a pseudo-random sequence. Reduced
//! modulo an unknown prime factor `p` of `n`, the sequence must repeat within
//! about `√p` steps. Once it has, `gcd(x_i - x_j, n)` exposes `p`. The running
//! time depends on the size of the smallest factor, not of `n`, which makes
//! rho a good way to strip factors of up to roughly 20 digits.
//!
//! This version uses Brent's refinements (R. P. Brent, "An improved Monte
//! Carlo factorization algorithm", BIT 20, 1980): cycle detection that needs
//! one function evaluation per step instead of Floyd's three, and batching
//! that multiplies many differences together so a GCD is only taken once
//! per batch.

use crate::arith::{Monty, Uint};

/// Number of differences multiplied together between GCDs.
const BATCH: u64 = 128;

/// Searches for a nontrivial factor of `n`.
///
/// `c` picks the polynomial `x² + c`. It should not be 0 or `n - 2`, which
/// give sequences with no useful randomness. `max_steps` caps the number of
/// polynomial evaluations.
///
/// Returns `None` if the step budget runs out, which always happens when `n`
/// is prime, so test primality first. It also returns `None` if this `c`
/// finds every factor at once and yields only `n`. Retrying with a different
/// `c` usually succeeds.
///
/// Even `n` greater than 2 returns 2 without searching.
pub fn pollard_rho<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    c: u64,
    max_steps: u64,
) -> Option<Uint<LIMBS>> {
    let two = Uint::from_u64(2);
    if !n.is_odd() {
        return (*n > two).then_some(two);
    }
    let monty = Monty::new(*n)?;
    let c = monty.to_monty(&Uint::from_u64(c));
    let f = |x: &Uint<LIMBS>| monty.add(&monty.square(x), &c);

    let mut steps = 0;
    let mut y = monty.to_monty(&two);
    let mut x;
    let mut ys;
    let mut g;
    // Brent's cycle detection: hold `x` fixed at the start of a run of
    // length `r`, compare it against each `y` in the run, then double `r`.
    let mut r = 1;
    loop {
        x = y;
        for _ in 0..r {
            y = f(&y);
        }
        steps += r;

        let mut k = 0;
        loop {
            // Remember where this batch started, so it can be replayed one
            // step at a time if the batched GCD overshoots.
            ys = y;
            let batch = BATCH.min(r - k);
            let mut q = monty.one();
            for _ in 0..batch {
                y = f(&y);
                q = monty.mul(&q, &monty.sub(&x, &y));
            }
            steps += batch;
            k += batch;
            // The product is in Montgomery form, `Q·R mod n`. Since `R` is a
            // power of two and `n` is odd, its GCD with `n` is that of `Q`.
            g = q.gcd(n);
            if g != Uint::ONE || k >= r {
                break;
            }
        }
        if g != Uint::ONE {
            break;
        }
        if steps >= max_steps {
            return None;
        }
        r *= 2;
    }

    // A GCD of `n` means the batch contained a difference divisible by every
    // factor of `n`, or the product picked up different factors from
    // different differences. Replay the batch one step at a time to find the
    // first difference that shares a factor with `n`.
    if g == *n {
        loop {
            ys = f(&ys);
            g = monty.sub(&x, &ys).gcd(n);
            if g != Uint::ONE {
                break;
            }
        }
    }

    (g != *n).then_some(g)
}
