//! Sieve parameters by the size of `N`.

/// Bits of `N`, then for each side, rational then algebraic: the
/// factor-base bound, `log₂` of the large-prime bound, and `log₂` of the
/// largest cofactor the sieve keeps, which past the large-prime bound it
/// splits into two large primes; then `log₂` of the width `I` of the
/// region sieved for each special-q, and the first special-q. Between
/// rows, values are interpolated.
///
/// These are large factor bases with one large prime a side, of about 10
/// to 30 times the bound, rounded up to a power of 2. CADO-NFS
/// instead uses factor bases 5 to 10 times smaller with two large primes a
/// side; at 90 digits its parameters took 532 s to collect enough
/// relations here, two large primes on the algebraic side 491 s, and these
/// 474 s, since this siever spends more on each candidate. At 60 digits,
/// with the line siever this replaced, bounds of 300,000 to 800,000 with
/// large primes 10 or 30 times larger all took 93 to 107 s; 60,000 never
/// finished. The rows past 330 bits are guesses.
#[allow(clippy::type_complexity)]
const TABLE: [(u32, [u32; 2], [u32; 2], [u32; 2], u32, u64); 7] = [
    (120, [5_000, 5_000], [18, 18], [18, 18], 9, 1_250),
    (160, [60_000, 60_000], [21, 21], [21, 21], 10, 15_000),
    (200, [500_000, 500_000], [23, 23], [23, 23], 11, 125_000),
    (260, [2_000_000, 2_000_000], [25, 25], [25, 25], 12, 500_000),
    (
        330,
        [6_000_000, 6_000_000],
        [26, 26],
        [26, 26],
        12,
        1_500_000,
    ),
    (
        400,
        [15_000_000, 15_000_000],
        [28, 28],
        [28, 28],
        13,
        3_750_000,
    ),
    (
        465,
        [40_000_000, 40_000_000],
        [29, 29],
        [29, 29],
        13,
        10_000_000,
    ),
];

/// Quadratic characters: each makes a dependency that is not a square in
/// the number field half as likely to survive.
pub const CHARACTERS: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    /// Primes up to this are in the rational factor base.
    pub rational_bound: u32,
    /// Primes up to this, with each root of `f`, are in the algebraic one.
    pub algebraic_bound: u32,
    /// `log₂` of the bound on primes outside the rational and algebraic
    /// factor bases that a relation may have.
    pub rational_lpb: u32,
    pub algebraic_lpb: u32,
    /// `log₂` of the largest cofactor, what the factor base leaves of a
    /// value, kept on each side: one large prime below the large-prime
    /// bound, or two, found by splitting it.
    pub rational_mfb: u32,
    pub algebraic_mfb: u32,
    /// Each special-q sieves `i` in `[−I/2, I/2)` and `j` in `[1, I/2)`,
    /// for `I = 2^log_i`.
    pub log_i: u32,
    /// The first special-q.
    pub qmin: u64,
}

impl Params {
    /// The parameters for an `N` of `bits` bits.
    pub fn for_bits(bits: u32) -> Self {
        let row =
            |(_, fb, lpb, mfb, log_i, qmin): (u32, [u32; 2], [u32; 2], [u32; 2], u32, u64)| Self {
                rational_bound: fb[0],
                algebraic_bound: fb[1],
                rational_lpb: lpb[0],
                algebraic_lpb: lpb[1],
                rational_mfb: mfb[0],
                algebraic_mfb: mfb[1],
                log_i,
                qmin,
            };
        let (first, last) = (TABLE[0], TABLE[TABLE.len() - 1]);
        if bits <= first.0 {
            return row(first);
        }
        if bits >= last.0 {
            return row(last);
        }
        let i = TABLE.iter().position(|r| r.0 >= bits).unwrap();
        let (lo, hi) = (TABLE[i - 1], TABLE[i]);
        // Rounded to the nearest, which matters for the bit counts.
        let lerp = |a: u64, b: u64| {
            let t = u64::from(bits - lo.0);
            let span = u64::from(hi.0 - lo.0);
            if b >= a {
                a + ((b - a) * t + span / 2) / span
            } else {
                a - ((a - b) * t + span / 2) / span
            }
        };
        let side = |a: [u32; 2], b: [u32; 2], s: usize| lerp(a[s].into(), b[s].into()) as u32;
        Self {
            rational_bound: side(lo.1, hi.1, 0),
            algebraic_bound: side(lo.1, hi.1, 1),
            rational_lpb: side(lo.2, hi.2, 0),
            algebraic_lpb: side(lo.2, hi.2, 1),
            rational_mfb: side(lo.3, hi.3, 0),
            algebraic_mfb: side(lo.3, hi.3, 1),
            log_i: lerp(lo.4.into(), hi.4.into()) as u32,
            qmin: lerp(lo.5, hi.5),
        }
    }
}
