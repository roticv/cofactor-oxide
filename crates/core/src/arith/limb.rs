//! The machine word ("limb") that fixed-width integers are built from.
//!
//! Native targets use 64-bit limbs with 128-bit products. WebAssembly has no
//! instruction for the high half of a 64×64-bit multiply, so a `u128` product
//! becomes a slow library call there. 32-bit limbs with `u64` products compile
//! to a single `i64.mul` instead.
//!
//! Build with `RUSTFLAGS="--cfg cofox_limb32"` to use 32-bit limbs on a native
//! target, which is how the wasm32 code path gets tested.

#[cfg(not(any(target_arch = "wasm32", cofox_limb32)))]
mod imp {
    pub type Limb = u64;
    pub type WideLimb = u128;
}

#[cfg(any(target_arch = "wasm32", cofox_limb32))]
mod imp {
    pub type Limb = u32;
    pub type WideLimb = u64;
}

pub use imp::{Limb, WideLimb};

/// Number of bits in a [`Limb`].
pub const LIMB_BITS: u32 = Limb::BITS;

/// Computes `a + b + carry`, returning `(sum, carry_out)`.
#[inline(always)]
pub const fn adc(a: Limb, b: Limb, carry: Limb) -> (Limb, Limb) {
    let t = a as WideLimb + b as WideLimb + carry as WideLimb;
    (t as Limb, (t >> LIMB_BITS) as Limb)
}

/// Computes `a - b - borrow`, returning `(difference, borrow_out)`.
///
/// `borrow` must be 0 or 1, and `borrow_out` is always 0 or 1.
#[inline(always)]
pub const fn sbb(a: Limb, b: Limb, borrow: Limb) -> (Limb, Limb) {
    let t = (a as WideLimb).wrapping_sub(b as WideLimb + borrow as WideLimb);
    (t as Limb, ((t >> LIMB_BITS) as Limb) & 1)
}

/// Computes `acc + a * b + carry`, returning `(low, high)`.
///
/// This cannot overflow: the largest possible result is
/// `(W-1) + (W-1)² + (W-1) = W² - 1` where `W = 2^LIMB_BITS`.
#[inline(always)]
pub const fn mac(acc: Limb, a: Limb, b: Limb, carry: Limb) -> (Limb, Limb) {
    let t = acc as WideLimb + (a as WideLimb) * (b as WideLimb) + carry as WideLimb;
    (t as Limb, (t >> LIMB_BITS) as Limb)
}
