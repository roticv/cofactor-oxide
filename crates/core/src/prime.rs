//! Primality testing.
//!
//! [`is_prime`] is the Baillie–PSW test: a strong Fermat test to base 2
//! followed by a strong Lucas test. No composite is known to pass both, and
//! none exists below 2^64, so for numbers that size the answer is exact.

use crate::arith::{Limb, Monty, Uint};

/// Small primes checked by division before the expensive tests.
const SMALL_PRIMES: [u64; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

/// Returns whether `n` is prime, using the Baillie–PSW test.
pub fn is_prime<const LIMBS: usize>(n: &Uint<LIMBS>) -> bool {
    if *n < Uint::from_u64(2) {
        return false;
    }
    for p in SMALL_PRIMES {
        if n.div_rem_limb(p as Limb).1 == 0 {
            return *n == Uint::from_u64(p);
        }
    }
    // No factor below 53 and n < 53² means n is prime.
    if *n < Uint::from_u64(53 * 53) {
        return true;
    }
    let monty = Monty::new(*n).expect("n is odd and greater than 1");
    miller_rabin(&monty, 2) && strong_lucas(&monty)
}

/// Strong probable-prime (Miller–Rabin) test of `n = monty.modulus()` to the
/// given base.
///
/// Returns `true` if `n` is prime or a strong pseudoprime to `base`. `n` must
/// not divide `base`.
pub fn miller_rabin<const LIMBS: usize>(monty: &Monty<LIMBS>, base: u64) -> bool {
    let n = monty.modulus();
    // n - 1 = d · 2^s with d odd.
    let n_minus_1 = n.wrapping_sub(&Uint::ONE);
    let s = n_minus_1.trailing_zeros();
    let d = n_minus_1 >> s;

    let one = monty.one();
    let minus_one = monty.neg(&one);
    let mut x = monty.pow(&monty.to_monty(&Uint::from_u64(base)), &d);
    if x == one || x == minus_one {
        return true;
    }
    for _ in 1..s {
        x = monty.square(&x);
        if x == minus_one {
            return true;
        }
    }
    false
}

/// Strong Lucas probable-prime test of `n = monty.modulus()`, with
/// parameters chosen by Selfridge's method A.
///
/// Returns `true` if `n` is prime or a strong Lucas pseudoprime. `n` must be
/// odd and greater than 1.
pub fn strong_lucas<const LIMBS: usize>(monty: &Monty<LIMBS>) -> bool {
    let n = monty.modulus();

    // Selfridge: the first D in 5, -7, 9, -11, 13, ... with Jacobi (D/n) = -1,
    // then P = 1 and Q = (1 - D) / 4. A perfect square has no such D, so
    // check for one if the search runs long.
    let mut d: i64 = 5;
    loop {
        match jacobi(d, n) {
            -1 => break,
            0 if Uint::from_u64(d.unsigned_abs()) != *n => return false,
            _ => {}
        }
        if d == 13 && is_square(n) {
            return false;
        }
        d = if d > 0 { -(d + 2) } else { -d + 2 };
    }
    let q = (1 - d) / 4;
    let signed_to_monty = |x: i64| {
        let m = monty.to_monty(&Uint::from_u64(x.unsigned_abs()));
        if x < 0 { monty.neg(&m) } else { m }
    };
    let d_m = signed_to_monty(d);
    let q_m = signed_to_monty(q);

    // n + 1 = k · 2^s with k odd. n + 1 can overflow when n = 2^BITS - 1.
    let (k, s) = match n.overflowing_add(&Uint::ONE) {
        (n_plus_1, false) => {
            let s = n_plus_1.trailing_zeros();
            (n_plus_1 >> s, s)
        }
        (_, true) => (Uint::ONE, Uint::<LIMBS>::BITS),
    };

    // Compute U_k, V_k and Q^k by a left-to-right binary ladder, using
    //   U_2j = U_j V_j            V_2j = V_j² - 2Q^j
    //   U_j+1 = (P U_j + V_j)/2    V_j+1 = (D U_j + P V_j)/2
    // with P = 1, starting from U_1 = 1, V_1 = P.
    let mut u = monty.one();
    let mut v = monty.one();
    let mut qj = q_m;
    for i in (0..k.bits() - 1).rev() {
        u = monty.mul(&u, &v);
        v = monty.sub(&monty.square(&v), &monty.add(&qj, &qj));
        qj = monty.square(&qj);
        if k.bit(i) {
            let (u0, v0) = (u, v);
            u = monty.half(&monty.add(&u0, &v0));
            v = monty.half(&monty.add(&monty.mul(&d_m, &u0), &v0));
            qj = monty.mul(&qj, &q_m);
        }
    }

    // Strong test: U_k ≡ 0, or V_(k·2^r) ≡ 0 for some 0 ≤ r < s.
    if u.is_zero() || v.is_zero() {
        return true;
    }
    for _ in 1..s {
        v = monty.sub(&monty.square(&v), &monty.add(&qj, &qj));
        qj = monty.square(&qj);
        if v.is_zero() {
            return true;
        }
    }
    false
}

fn is_square<const LIMBS: usize>(n: &Uint<LIMBS>) -> bool {
    let root = n.isqrt();
    let (sq, hi) = root.widening_mul(&root);
    hi.is_zero() && sq == *n
}

/// Jacobi symbol `(a/n)` for small `a` and odd `n`.
fn jacobi<const LIMBS: usize>(a: i64, n: &Uint<LIMBS>) -> i32 {
    // (-1/n) = 1 if n ≡ 1 (mod 4), else -1.
    let n_mod_4 = n.as_limbs()[0] & 3;
    let sign = if a < 0 && n_mod_4 == 3 { -1 } else { 1 };
    let m = a.unsigned_abs();
    if m == 1 {
        return sign;
    }
    // Callers only pass odd |a|, so quadratic reciprocity applies directly:
    // (m/n) = (n/m), negated when both are 3 (mod 4).
    debug_assert!(m % 2 == 1);
    let reciprocity = if m % 4 == 3 && n_mod_4 == 3 { -1 } else { 1 };
    let m = m as Limb;
    sign * reciprocity * jacobi_limb(n.div_rem_limb(m).1, m)
}

/// Jacobi symbol `(a/n)` for odd `n`.
fn jacobi_limb(mut a: Limb, mut n: Limb) -> i32 {
    let mut result = 1;
    a %= n;
    while a != 0 {
        while a.is_multiple_of(2) {
            a /= 2;
            // (2/n) = -1 exactly when n ≡ 3 or 5 (mod 8).
            if n % 8 == 3 || n % 8 == 5 {
                result = -result;
            }
        }
        core::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a %= n;
    }
    if n == 1 { result } else { 0 }
}
