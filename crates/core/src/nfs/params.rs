//! Sieve parameters by the size of `N`.

/// Bits of `N`, then for each side, rational then algebraic: the
/// factor-base bound, `log₂` of the large-prime bound, and `log₂` of the
/// largest cofactor the sieve keeps, which past the large-prime bound it
/// splits into two large primes; then `log₂` of the width `I` of the
/// region sieved for each special-q, the first special-q, and the extra
/// algebraic slack. Between rows, values are interpolated.
///
/// Up to 330 bits these are large factor bases with one large prime a
/// side, of about 10 to 30 times the bound, rounded up to a power of 2.
/// CADO-NFS instead uses factor bases 5 to 10 times smaller with two large
/// primes a side; at 90 digits its parameters took 532 s to collect enough
/// relations here, two large primes on the algebraic side 491 s, and these
/// 474 s, since this siever spends more on each candidate. End to end at
/// 90 digits, two large primes a side took 2.5 times as long. At 60
/// digits, with the line siever this replaced, bounds of 300,000 to
/// 800,000 with large primes 10 or 30 times larger all took 93 to 107 s;
/// 60,000 never finished.
///
/// From 364 bits the rows switch to the CADO-NFS pattern. RSA-110 with
/// the large factor bases, 10.4 million and one 27-bit large prime a side,
/// took 4 h 56 min on 6 threads; with factor bases of 5 million and two
/// 27-bit large primes a side, 3 h 19 min. It needed 10.1 million
/// relations rather than 6.3 million, but each special-q gave over twice
/// as many, and the larger matrix cost minutes against hours of sieving.
/// Between 330 and 364 bits the cofactor bound is interpolated, so sizes
/// there keep a large prime and a half on average.
///
/// The rows for RSA-120 (397 bits) and RSA-130 (430) come from sieving
/// 2000-wide slices of special-q at points from 250,000 to 32 million with
/// each candidate, and adding up the time to reach the relations needed,
/// `0.66·(π(2^lpb₀) + π(2^lpb₁))` as RSA-110 needed. A relation is found
/// again under each of its algebraic primes in the range sieved, so each
/// counted `1/m` for its `m` such primes: from 2.0 to 2.1 million that
/// predicted 1.6% duplicates, against 1.2% measured. At 120 digits, `I` of
/// 2^13 cost 20% more than 2^12, 27- and 28-bit large primes beat 28 and
/// 28, factor bases of 2.5 to 5 million were within 4% of each other, and
/// starting special-q lower than about 4 million saved less sieving than
/// the duplicates cost: about 10 core-hours of sieving in all. At 130
/// digits, `I` of 2^13 and factor bases of 10 million won by 6 to 10%
/// over CADO-NFS's c130 parameters and our 120-digit row: about 37
/// core-hours. For RSA-140 (463 bits), with the relations needed
/// recalibrated to `0.56·(…)` by a real RSA-120 run, factor bases of 20
/// million, 29/29 bits and special-q from 32 million came to about 174
/// core-hours: 30 million tied, 10 million cost 31% more, CADO-NFS's c140
/// parameters 40% more and three large primes 52% more, and starting at
/// 16 million cost more in duplicates than it saved. The rows past 465
/// bits are guesses.
///
/// The algebraic side's candidates are allowed 8 more bits from 430 bits,
/// for values with more than their share of the primes below 30, which
/// aren't sieved. At RSA-140 CADO-NFS's `las`, with the same polynomial
/// and parameters, found 17% more relations than this siever, nearly all
/// with two large primes on the algebraic side and a median of 22.7 bits
/// from the unsieved primes against 14.4 in the ones found. One thread,
/// 15,000 special-q: at RSA-140, 8 more bits gave 14% more relations and
/// 6.6% more a second (4 bits: 9% and 5.7%; 12 bits: 17% and 2.8%); at
/// RSA-130, 16% and 4.6% (4 bits: 10% and 5.2%); at RSA-110, 6.4% more
/// relations but 17.5% fewer a second, and 4 bits 7% fewer.
///
/// `I` stops at 2^13, where the buckets of the largest primes' hits take
/// a few hundred megabytes for each thread. Past 600 bits the last row
/// holds, though numbers that large need larger large primes, two or three
/// of them a side, to be practical. The siever takes large-prime bounds up
/// to 2^40, below the quadratic characters' primes.
#[allow(clippy::type_complexity)]
const TABLE: [(u32, [u32; 2], [u32; 2], [u32; 2], u32, u64, u32); 11] = [
    (120, [5_000, 5_000], [18, 18], [18, 18], 9, 1_250, 0),
    (160, [60_000, 60_000], [21, 21], [21, 21], 10, 15_000, 0),
    (200, [500_000, 500_000], [23, 23], [23, 23], 11, 125_000, 0),
    (
        260,
        [2_000_000, 2_000_000],
        [25, 25],
        [25, 25],
        12,
        500_000,
        0,
    ),
    (
        330,
        [6_000_000, 6_000_000],
        [26, 26],
        [26, 26],
        12,
        1_500_000,
        0,
    ),
    (
        364,
        [5_000_000, 5_000_000],
        [27, 27],
        [54, 54],
        12,
        1_250_000,
        0,
    ),
    (
        397,
        [5_000_000, 5_000_000],
        [27, 28],
        [52, 54],
        12,
        4_000_000,
        0,
    ),
    (
        430,
        [10_000_000, 10_000_000],
        [28, 29],
        [54, 57],
        13,
        8_000_000,
        8,
    ),
    (
        465,
        [20_000_000, 20_000_000],
        [29, 29],
        [58, 58],
        13,
        32_000_000,
        8,
    ),
    (
        530,
        [40_000_000, 40_000_000],
        [30, 30],
        [60, 60],
        13,
        10_000_000,
        8,
    ),
    (
        600,
        [75_000_000, 75_000_000],
        [31, 31],
        [62, 62],
        13,
        20_000_000,
        8,
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
    /// bound, or two or three, found by splitting it. Three need a
    /// cofactor bound past twice the large-prime bound's bits, which is at
    /// most 40; the cofactor bound is at most 127.
    pub rational_mfb: u32,
    pub algebraic_mfb: u32,
    /// Each special-q sieves `i` in `[−I/2, I/2)` and `j` in `[1, I/2)`,
    /// for `I = 2^log_i`.
    pub log_i: u32,
    /// The first special-q.
    pub qmin: u64,
    /// Bits beyond the cofactor bound that the algebraic side's candidate
    /// thresholds allow, for values richer than usual in the primes the
    /// sieve skips.
    pub algebraic_slack: u32,
}

impl Params {
    /// The parameters for an `N` of `bits` bits.
    pub fn for_bits(bits: u32) -> Self {
        let row = |(_, fb, lpb, mfb, log_i, qmin, slack): (
            u32,
            [u32; 2],
            [u32; 2],
            [u32; 2],
            u32,
            u64,
            u32,
        )| Self {
            rational_bound: fb[0],
            algebraic_bound: fb[1],
            rational_lpb: lpb[0],
            algebraic_lpb: lpb[1],
            rational_mfb: mfb[0],
            algebraic_mfb: mfb[1],
            log_i,
            qmin,
            algebraic_slack: slack,
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
            algebraic_slack: lerp(lo.6.into(), hi.6.into()) as u32,
        }
    }
}
