//! Fixed-width unsigned integers.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Shl, Shr};
use core::str::FromStr;

use alloc::string::String;
use alloc::vec::Vec;

use super::limb::{LIMB_BITS, Limb, WideLimb, adc, mac, sbb};

/// An unsigned integer of `LIMBS * LIMB_BITS` bits, stored on the stack.
///
/// Limbs are little-endian: `limbs[0]` is the least significant. Because the
/// width is known at compile time, loops over the limbs have a fixed length
/// and nothing is ever allocated.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Uint<const LIMBS: usize> {
    limbs: [Limb; LIMBS],
}

impl<const LIMBS: usize> Uint<LIMBS> {
    /// Width of the integer in bits.
    pub const BITS: u32 = LIMBS as u32 * LIMB_BITS;
    pub const ZERO: Self = Self { limbs: [0; LIMBS] };
    pub const ONE: Self = Self::from_u64(1);
    pub const MAX: Self = Self {
        limbs: [Limb::MAX; LIMBS],
    };

    pub const fn from_limbs(limbs: [Limb; LIMBS]) -> Self {
        Self { limbs }
    }

    pub const fn as_limbs(&self) -> &[Limb; LIMBS] {
        &self.limbs
    }

    /// Converts a `u64`.
    ///
    /// # Panics
    ///
    /// Panics if `x` does not fit, which can only happen when the type is
    /// narrower than 64 bits.
    pub const fn from_u64(x: u64) -> Self {
        let mut limbs = [0; LIMBS];
        let mut x = x;
        let mut i = 0;
        while x != 0 {
            assert!(i < LIMBS, "value does not fit in Uint");
            limbs[i] = x as Limb;
            // A plain `>>` would overflow when limbs are 64 bits wide.
            x = match x.checked_shr(LIMB_BITS) {
                Some(rest) => rest,
                None => 0,
            };
            i += 1;
        }
        Self { limbs }
    }

    /// Returns the value as a `u64` if it fits.
    // The cast is a no-op with 64-bit limbs but needed with 32-bit ones.
    #[allow(clippy::unnecessary_cast)]
    pub fn to_u64(&self) -> Option<u64> {
        if self.bits() > 64 {
            return None;
        }
        let mut x = 0u64;
        for &limb in self.limbs.iter().rev() {
            x = x.checked_shl(LIMB_BITS).unwrap_or(0) | limb as u64;
        }
        Some(x)
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    pub fn is_odd(&self) -> bool {
        LIMBS > 0 && self.limbs[0] & 1 == 1
    }

    /// Number of significant bits, so `0` for zero and `1` for one.
    pub fn bits(&self) -> u32 {
        for i in (0..LIMBS).rev() {
            if self.limbs[i] != 0 {
                return i as u32 * LIMB_BITS + (LIMB_BITS - self.limbs[i].leading_zeros());
            }
        }
        0
    }

    /// Returns bit `i`, counting from the least significant. Bits past the
    /// width are zero.
    pub fn bit(&self, i: u32) -> bool {
        let limb = (i / LIMB_BITS) as usize;
        limb < LIMBS && (self.limbs[limb] >> (i % LIMB_BITS)) & 1 == 1
    }

    /// Number of trailing zero bits. Returns [`Self::BITS`] for zero.
    pub fn trailing_zeros(&self) -> u32 {
        for i in 0..LIMBS {
            if self.limbs[i] != 0 {
                return i as u32 * LIMB_BITS + self.limbs[i].trailing_zeros();
            }
        }
        Self::BITS
    }

    pub fn overflowing_add(&self, rhs: &Self) -> (Self, bool) {
        let mut limbs = [0; LIMBS];
        let mut carry = 0;
        for i in 0..LIMBS {
            (limbs[i], carry) = adc(self.limbs[i], rhs.limbs[i], carry);
        }
        (Self { limbs }, carry != 0)
    }

    pub fn overflowing_sub(&self, rhs: &Self) -> (Self, bool) {
        let mut limbs = [0; LIMBS];
        let mut borrow = 0;
        for i in 0..LIMBS {
            (limbs[i], borrow) = sbb(self.limbs[i], rhs.limbs[i], borrow);
        }
        (Self { limbs }, borrow != 0)
    }

    pub fn wrapping_add(&self, rhs: &Self) -> Self {
        self.overflowing_add(rhs).0
    }

    pub fn wrapping_sub(&self, rhs: &Self) -> Self {
        self.overflowing_sub(rhs).0
    }

    pub fn checked_add(&self, rhs: &Self) -> Option<Self> {
        match self.overflowing_add(rhs) {
            (sum, false) => Some(sum),
            (_, true) => None,
        }
    }

    pub fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        match self.overflowing_sub(rhs) {
            (diff, false) => Some(diff),
            (_, true) => None,
        }
    }

    /// Full product, returned as `(low, high)` halves.
    pub fn widening_mul(&self, rhs: &Self) -> (Self, Self) {
        let mut lo = [0; LIMBS];
        let mut hi = [0; LIMBS];
        // Schoolbook multiplication: add `self * rhs[i]` into the product at
        // limb offset `i`, one row at a time.
        for i in 0..LIMBS {
            let mut carry = 0;
            for j in 0..LIMBS {
                let k = i + j;
                let slot = if k < LIMBS {
                    &mut lo[k]
                } else {
                    &mut hi[k - LIMBS]
                };
                (*slot, carry) = mac(*slot, self.limbs[j], rhs.limbs[i], carry);
            }
            // Offset `i + LIMBS` has not been written by any earlier row.
            hi[i] = carry;
        }
        (Self { limbs: lo }, Self { limbs: hi })
    }

    /// Full square, returned as `(low, high)` halves.
    ///
    /// Same result as `widening_mul(self, self)`, but each cross product
    /// `a_i·a_j` with `i ≠ j` is computed once and doubled, rather than twice.
    pub fn widening_square(&self) -> (Self, Self) {
        let a = &self.limbs;
        let mut lo = [0; LIMBS];
        let mut hi = [0; LIMBS];
        // Sum of a_i·a_j over i < j, at limb offset i + j.
        for i in 0..LIMBS {
            let mut carry = 0;
            for j in i + 1..LIMBS {
                let k = i + j;
                let slot = if k < LIMBS {
                    &mut lo[k]
                } else {
                    &mut hi[k - LIMBS]
                };
                (*slot, carry) = mac(*slot, a[i], a[j], carry);
            }
            // Offset i + LIMBS hasn't been written by any earlier row.
            hi[i] = carry;
        }

        // Double it. The sum is below 2^(2·BITS − 1), so nothing is lost.
        let top_of_lo = lo[LIMBS - 1] >> (LIMB_BITS - 1);
        let mut hi = *(Self { limbs: hi } << 1).as_limbs();
        let mut lo = *(Self { limbs: lo } << 1).as_limbs();
        hi[0] |= top_of_lo;

        // Add the squares a_i² at offset 2i.
        let mut carry = 0;
        for i in 0..LIMBS {
            let (sq_lo, sq_hi) = mac(0, a[i], a[i], 0);
            for (k, limb) in [(2 * i, sq_lo), (2 * i + 1, sq_hi)] {
                let slot = if k < LIMBS {
                    &mut lo[k]
                } else {
                    &mut hi[k - LIMBS]
                };
                (*slot, carry) = adc(*slot, limb, carry);
            }
        }
        debug_assert_eq!(carry, 0);
        (Self { limbs: lo }, Self { limbs: hi })
    }

    /// Computes `self * rhs`, returning the result and the limb carried out
    /// of the top.
    pub fn overflowing_mul_limb(&self, rhs: Limb) -> (Self, Limb) {
        let mut limbs = [0; LIMBS];
        let mut carry = 0;
        for i in 0..LIMBS {
            (limbs[i], carry) = mac(0, self.limbs[i], rhs, carry);
        }
        (Self { limbs }, carry)
    }

    /// Divides by a single limb, returning `(quotient, remainder)`.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero.
    pub fn div_rem_limb(&self, divisor: Limb) -> (Self, Limb) {
        assert!(divisor != 0, "division by zero");
        let mut limbs = [0; LIMBS];
        let mut rem: Limb = 0;
        for i in (0..LIMBS).rev() {
            let t = ((rem as WideLimb) << LIMB_BITS) | self.limbs[i] as WideLimb;
            limbs[i] = (t / divisor as WideLimb) as Limb;
            rem = (t % divisor as WideLimb) as Limb;
        }
        (Self { limbs }, rem)
    }

    /// Divides by `divisor`, returning `(quotient, remainder)`.
    ///
    /// This is binary long division, one bit per step. It's simple rather
    /// than fast, so keep it out of inner loops.
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is zero.
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "division by zero");
        let mut q = Self::ZERO;
        let mut r = Self::ZERO;
        for i in (0..self.bits()).rev() {
            // r < divisor, so 2r + 1 fits in one more bit than the type has.
            // The bit shifted out is kept in `carry`.
            let carry = r.bit(Self::BITS - 1);
            r = r << 1;
            r.limbs[0] |= self.bit(i) as Limb;
            if carry || r >= *divisor {
                r = r.wrapping_sub(divisor);
                q.limbs[(i / LIMB_BITS) as usize] |= 1 << (i % LIMB_BITS);
            }
        }
        (q, r)
    }

    /// Integer square root: the largest `s` with `s² ≤ self`.
    pub fn isqrt(&self) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }
        // Digit-by-digit method in base 4, which needs only shifts, adds
        // and compares.
        let mut x = *self;
        let mut root = Self::ZERO;
        let mut bit = Self::ONE << ((self.bits() - 1) & !1);
        while !bit.is_zero() {
            let trial = root.wrapping_add(&bit);
            root = root >> 1;
            if x >= trial {
                x = x.wrapping_sub(&trial);
                root = root.wrapping_add(&bit);
            }
            bit = bit >> 2;
        }
        root
    }

    /// Greatest common divisor, using the binary GCD algorithm.
    ///
    /// `gcd(0, x)` is `x`.
    pub fn gcd(&self, other: &Self) -> Self {
        let (mut a, mut b) = (*self, *other);
        if a.is_zero() {
            return b;
        }
        if b.is_zero() {
            return a;
        }
        let shift = a.trailing_zeros().min(b.trailing_zeros());
        a = a >> a.trailing_zeros();
        loop {
            // `a` is odd here, so the common power of two is already out.
            b = b >> b.trailing_zeros();
            if a > b {
                core::mem::swap(&mut a, &mut b);
            }
            b = b.wrapping_sub(&a);
            if b.is_zero() {
                return a << shift;
            }
        }
    }
}

impl<const LIMBS: usize> Default for Uint<LIMBS> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const LIMBS: usize> From<u64> for Uint<LIMBS> {
    fn from(x: u64) -> Self {
        Self::from_u64(x)
    }
}

impl<const LIMBS: usize> Ord for Uint<LIMBS> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.limbs.iter().rev().cmp(other.limbs.iter().rev())
    }
}

impl<const LIMBS: usize> PartialOrd for Uint<LIMBS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Shifts left, discarding bits shifted out of the top. Shifting by
/// [`Uint::BITS`] or more gives zero.
impl<const LIMBS: usize> Shl<u32> for Uint<LIMBS> {
    type Output = Self;

    fn shl(self, shift: u32) -> Self {
        let limb_shift = (shift / LIMB_BITS) as usize;
        let bit_shift = shift % LIMB_BITS;
        let mut limbs = [0; LIMBS];
        for i in limb_shift..LIMBS {
            limbs[i] = self.limbs[i - limb_shift] << bit_shift;
            if bit_shift != 0 && i > limb_shift {
                limbs[i] |= self.limbs[i - limb_shift - 1] >> (LIMB_BITS - bit_shift);
            }
        }
        Self { limbs }
    }
}

/// Shifts right. Shifting by [`Uint::BITS`] or more gives zero.
impl<const LIMBS: usize> Shr<u32> for Uint<LIMBS> {
    type Output = Self;

    fn shr(self, shift: u32) -> Self {
        let limb_shift = (shift / LIMB_BITS) as usize;
        let bit_shift = shift % LIMB_BITS;
        let mut limbs = [0; LIMBS];
        for i in 0..LIMBS.saturating_sub(limb_shift) {
            limbs[i] = self.limbs[i + limb_shift] >> bit_shift;
            if bit_shift != 0 && i + limb_shift + 1 < LIMBS {
                limbs[i] |= self.limbs[i + limb_shift + 1] << (LIMB_BITS - bit_shift);
            }
        }
        Self { limbs }
    }
}

/// Number of decimal digits handled per limb when parsing and printing: the
/// largest `d` with `10^d` fitting in a limb.
const DECIMAL_CHUNK_DIGITS: usize = Limb::MAX.ilog10() as usize;
const DECIMAL_CHUNK: Limb = (10 as Limb).pow(DECIMAL_CHUNK_DIGITS as u32);

impl<const LIMBS: usize> fmt::Display for Uint<LIMBS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Peel off chunks of decimal digits from the bottom, then print them
        // from the top down.
        let mut chunks = Vec::new();
        let mut x = *self;
        while !x.is_zero() {
            let (q, r) = x.div_rem_limb(DECIMAL_CHUNK);
            chunks.push(r);
            x = q;
        }
        let mut s = String::new();
        match chunks.split_last() {
            None => s.push('0'),
            Some((top, rest)) => {
                fmt::write(&mut s, format_args!("{top}"))?;
                for chunk in rest.iter().rev() {
                    fmt::write(&mut s, format_args!("{chunk:0DECIMAL_CHUNK_DIGITS$}"))?;
                }
            }
        }
        f.pad_integral(true, "", &s)
    }
}

impl<const LIMBS: usize> fmt::Debug for Uint<LIMBS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Uint<{LIMBS}>({self})")
    }
}

/// Error returned when parsing a [`Uint`] from a string fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseUintError {
    Empty,
    InvalidDigit,
    Overflow,
}

impl fmt::Display for ParseUintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "cannot parse integer from empty string",
            Self::InvalidDigit => "invalid digit found in string",
            Self::Overflow => "number too large to fit in target type",
        })
    }
}

impl core::error::Error for ParseUintError {}

/// Parses a decimal string. Underscores and signs are not accepted.
impl<const LIMBS: usize> FromStr for Uint<LIMBS> {
    type Err = ParseUintError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(ParseUintError::Empty);
        }
        if !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ParseUintError::InvalidDigit);
        }
        // Consume up to DECIMAL_CHUNK_DIGITS digits at a time with one
        // multiply-add each, rather than one per digit.
        let mut x = Self::ZERO;
        for chunk in s.as_bytes().chunks(DECIMAL_CHUNK_DIGITS) {
            let mut scale: Limb = 1;
            let mut value: Limb = 0;
            for &b in chunk {
                scale *= 10;
                value = value * 10 + (b - b'0') as Limb;
            }
            let (shifted, carry) = x.overflowing_mul_limb(scale);
            if carry != 0 {
                return Err(ParseUintError::Overflow);
            }
            x = shifted
                .checked_add(&Self::from_u64(value as u64))
                .ok_or(ParseUintError::Overflow)?;
        }
        Ok(x)
    }
}
