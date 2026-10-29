//! Montgomery modular arithmetic.

use super::limb::{LIMB_BITS, Limb, adc, mac};
use super::uint::Uint;

/// Arithmetic modulo a fixed odd modulus `n`, using Montgomery form.
///
/// A residue `x` is stored as `x·R mod n`, where `R = 2^Uint::<LIMBS>::BITS`.
/// In that form a modular multiplication needs only multiplies, adds and
/// shifts, with no division by `n`. That's what makes it the workhorse of
/// rho, p±1 and ECM.
///
/// Unless a method says otherwise, its inputs and outputs are in Montgomery
/// form and fully reduced (less than `n`). Use [`Monty::to_monty`] and
/// [`Monty::from_monty`] to convert.
#[derive(Clone, Copy, Debug)]
pub struct Monty<const LIMBS: usize> {
    modulus: Uint<LIMBS>,
    /// `-n⁻¹ mod 2^LIMB_BITS`.
    neg_inv: Limb,
    /// `R mod n`, which is 1 in Montgomery form.
    one: Uint<LIMBS>,
    /// `R² mod n`, used to convert into Montgomery form.
    r2: Uint<LIMBS>,
}

impl<const LIMBS: usize> Monty<LIMBS> {
    /// Sets up arithmetic modulo `modulus`.
    ///
    /// Returns `None` unless `modulus` is odd and greater than 1.
    pub fn new(modulus: Uint<LIMBS>) -> Option<Self> {
        if !modulus.is_odd() || modulus == Uint::ONE {
            return None;
        }

        // Newton's iteration for the inverse mod 2^LIMB_BITS. Starting from
        // 1, which is correct to one bit, each step doubles the number of
        // correct bits, so log2(LIMB_BITS) steps are enough.
        let n0 = modulus.as_limbs()[0];
        let mut inv: Limb = 1;
        for _ in 0..LIMB_BITS.trailing_zeros() {
            inv = inv.wrapping_mul((2 as Limb).wrapping_sub(n0.wrapping_mul(inv)));
        }

        // R mod n and R² mod n by repeated modular doubling. That's slow,
        // but it only happens once per modulus and needs no division.
        let double = |x: Uint<LIMBS>| {
            let (y, carry) = x.overflowing_add(&x);
            if carry || y >= modulus {
                y.wrapping_sub(&modulus)
            } else {
                y
            }
        };
        let mut one = Uint::ONE;
        for _ in 0..Uint::<LIMBS>::BITS {
            one = double(one);
        }
        let mut r2 = one;
        for _ in 0..Uint::<LIMBS>::BITS {
            r2 = double(r2);
        }

        Some(Self {
            modulus,
            neg_inv: inv.wrapping_neg(),
            one,
            r2,
        })
    }

    pub fn modulus(&self) -> &Uint<LIMBS> {
        &self.modulus
    }

    /// 1 in Montgomery form.
    pub fn one(&self) -> Uint<LIMBS> {
        self.one
    }

    /// Converts an ordinary integer into Montgomery form, reducing it mod
    /// `n`. Any value of the type is accepted.
    pub fn to_monty(&self, x: &Uint<LIMBS>) -> Uint<LIMBS> {
        // x < R and r2 < n, so their product is below n·R as `mul` requires.
        self.mul(x, &self.r2)
    }

    /// Converts out of Montgomery form.
    pub fn from_monty(&self, x: &Uint<LIMBS>) -> Uint<LIMBS> {
        self.mul(x, &Uint::ONE)
    }

    /// Computes `a·b·R⁻¹ mod n`, which is the product in Montgomery form.
    ///
    /// This is the CIOS (coarsely integrated operand scanning) method. It
    /// requires `a·b < n·R`, which holds when both inputs are reduced.
    pub fn mul(&self, a: &Uint<LIMBS>, b: &Uint<LIMBS>) -> Uint<LIMBS> {
        let a = a.as_limbs();
        let b = b.as_limbs();
        let n = self.modulus.as_limbs();

        // The accumulator is LIMBS + 2 limbs wide: `t` plus `t_hi`, and a
        // transient top limb inside the loop. Stable Rust can't size an
        // array `LIMBS + 2`, so the top two limbs are separate variables.
        let mut t = [0 as Limb; LIMBS];
        let mut t_hi: Limb = 0;
        for i in 0..LIMBS {
            // t += a · b[i]
            let mut carry = 0;
            for j in 0..LIMBS {
                (t[j], carry) = mac(t[j], a[j], b[i], carry);
            }
            let t_top;
            (t_hi, t_top) = adc(t_hi, carry, 0);

            // Add the multiple of n that zeroes the bottom limb, then shift
            // down by one limb: t = (t + m·n) / 2^LIMB_BITS.
            let m = t[0].wrapping_mul(self.neg_inv);
            let (_, mut carry) = mac(t[0], m, n[0], 0);
            for j in 1..LIMBS {
                (t[j - 1], carry) = mac(t[j], m, n[j], carry);
            }
            let top_carry;
            (t[LIMBS - 1], top_carry) = adc(t_hi, carry, 0);
            t_hi = t_top + top_carry;
        }

        // Now t < 2n, so one conditional subtraction fully reduces it.
        let t = Uint::from_limbs(t);
        if t_hi != 0 || t >= self.modulus {
            t.wrapping_sub(&self.modulus)
        } else {
            t
        }
    }

    pub fn square(&self, a: &Uint<LIMBS>) -> Uint<LIMBS> {
        self.mul(a, a)
    }

    /// Modular addition. Works the same in or out of Montgomery form.
    pub fn add(&self, a: &Uint<LIMBS>, b: &Uint<LIMBS>) -> Uint<LIMBS> {
        let (sum, carry) = a.overflowing_add(b);
        if carry || sum >= self.modulus {
            sum.wrapping_sub(&self.modulus)
        } else {
            sum
        }
    }

    /// Modular subtraction. Works the same in or out of Montgomery form.
    pub fn sub(&self, a: &Uint<LIMBS>, b: &Uint<LIMBS>) -> Uint<LIMBS> {
        let (diff, borrow) = a.overflowing_sub(b);
        if borrow {
            diff.wrapping_add(&self.modulus)
        } else {
            diff
        }
    }

    /// Computes `base^exp`. `base` and the result are in Montgomery form;
    /// `exp` is an ordinary integer of any width.
    pub fn pow<const E: usize>(&self, base: &Uint<LIMBS>, exp: &Uint<E>) -> Uint<LIMBS> {
        // Left-to-right binary exponentiation.
        let mut acc = self.one;
        for i in (0..exp.bits()).rev() {
            acc = self.square(&acc);
            if exp.bit(i) {
                acc = self.mul(&acc, base);
            }
        }
        acc
    }
}
