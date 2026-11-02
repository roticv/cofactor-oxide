//! Sieve parameters by the size of `N`.

/// Bits of `N`, the rational and algebraic factor-base bounds, the
/// large-prime multiplier and `log₂` of the sieve line's half width
/// before scaling by the skew. Between rows, values are interpolated.
///
/// Partial relations with a large prime on each side rarely combine until
/// there are about as many relations as possible large primes, so nearly
/// all the relations that count are full ones, and the factor bases have
/// to be large. At 60 digits, bounds of 300,000 to 800,000 with
/// multipliers of 10 or 30 all took 93 to 107 s; 60,000 never finished.
/// The other rows are guesses, the larger ones not yet measured.
const TABLE: [(u32, u32, u32, u32, u32); 6] = [
    (120, 5_000, 5_000, 30, 14),
    (160, 60_000, 60_000, 20, 15),
    (200, 500_000, 500_000, 10, 15),
    (260, 2_000_000, 2_000_000, 10, 16),
    (330, 6_000_000, 6_000_000, 10, 17),
    (400, 15_000_000, 15_000_000, 10, 18),
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
    /// Each side may have one prime outside its factor base, below this
    /// multiple of the side's bound.
    pub large_mult: u32,
    /// Each line `b` sieves `a` in `[−w, w)`, for `w` this times the
    /// square root of the polynomial's skew.
    pub half_width: u32,
}

impl Params {
    /// The parameters for an `N` of `bits` bits.
    pub fn for_bits(bits: u32) -> Self {
        let row = |(_, r, a, mult, w): (u32, u32, u32, u32, u32)| Self {
            rational_bound: r,
            algebraic_bound: a,
            large_mult: mult,
            half_width: 1 << w,
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
        let lerp = |a: u32, b: u32| {
            let (a, b) = (u64::from(a), u64::from(b));
            let t = u64::from(bits - lo.0);
            let span = u64::from(hi.0 - lo.0);
            if b >= a {
                (a + (b - a) * t / span) as u32
            } else {
                (a - (a - b) * t / span) as u32
            }
        };
        Self {
            rational_bound: lerp(lo.1, hi.1),
            algebraic_bound: lerp(lo.2, hi.2),
            large_mult: lerp(lo.3, hi.3),
            half_width: 1 << lerp(lo.4, hi.4),
        }
    }
}
