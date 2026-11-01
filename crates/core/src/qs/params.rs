//! Sieve parameters by the size of `kN`.

/// Bits of `kN`, factor-base size, large-prime multiplier and sieve blocks
/// on each side. These start from msieve's table, with one block a side up
/// to 249 bits, which measured faster than two or three from 199 to 232
/// bits, and four from 298 bits: at RSA-100's 330 bits, four gave 8 times
/// the relations per second of msieve's nine. The rest is still to be
/// tuned; between rows, values are interpolated.
const TABLE: [(u32, u32, u32, u32); 14] = [
    (64, 100, 40, 1),
    (128, 450, 40, 1),
    (183, 2_000, 40, 1),
    (200, 3_000, 50, 1),
    (212, 5_400, 50, 1),
    (233, 10_000, 100, 1),
    (249, 27_000, 100, 1),
    (266, 50_000, 100, 3),
    (283, 55_000, 80, 3),
    (298, 60_000, 80, 4),
    (315, 80_000, 150, 4),
    (332, 115_000, 150, 4),
    (348, 140_000, 150, 4),
    (363, 210_000, 150, 4),
];

/// Bytes in one sieve block, sized to stay in L1 cache.
pub const BLOCK_SIZE: usize = 1 << 16;

/// How the sieve is set up for a given size of `kN`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    /// Primes in the factor base.
    pub fb_size: usize,
    /// A relation may have one prime outside the factor base, if it is
    /// below this multiple of the largest factor-base prime.
    pub large_mult: u32,
    /// Sieve blocks of [`BLOCK_SIZE`] bytes on each side of `x = 0`.
    pub blocks: usize,
}

impl Params {
    /// The parameters for a `kN` of `bits` bits.
    pub fn for_bits(bits: u32) -> Self {
        let row = |(_, fb, mult, blocks): (u32, u32, u32, u32)| Self {
            fb_size: fb as usize,
            large_mult: mult,
            blocks: blocks as usize,
        };
        let first = TABLE[0];
        let last = TABLE[TABLE.len() - 1];
        if bits <= first.0 {
            return row(first);
        }
        if bits >= last.0 {
            return row(last);
        }
        let i = TABLE.iter().position(|r| r.0 >= bits).unwrap();
        let (lo, hi) = (TABLE[i - 1], TABLE[i]);
        // Linear interpolation, rounding toward the lower row's value.
        let lerp = |a: u32, b: u32| {
            let (a, b) = (i64::from(a), i64::from(b));
            let t = i64::from(bits - lo.0);
            (a + (b - a) * t / i64::from(hi.0 - lo.0)) as u32
        };
        Self {
            fb_size: lerp(lo.1, hi.1) as usize,
            large_mult: lerp(lo.2, hi.2),
            blocks: lerp(lo.3, hi.3) as usize,
        }
    }
}
