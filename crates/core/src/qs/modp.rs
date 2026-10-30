//! Arithmetic modulo a word-sized prime.

/// `b^e mod p`, for `p < 2^32`.
pub fn pow_mod(b: u64, mut e: u64, p: u64) -> u64 {
    debug_assert!(p < 1 << 32);
    let mut b = b % p;
    let mut r = 1 % p;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % p;
        }
        b = b * b % p;
        e >>= 1;
    }
    r
}

/// A square root of `a` modulo the prime `p < 2^32`, or `None` if `a` is
/// not a square mod `p`. Of the two roots `r` and `p − r`, returns the
/// smaller.
///
/// Uses Tonelli–Shanks: write `p − 1 = q·2^s` with `q` odd. The answer is
/// `a^((q+1)/2)` times a correction in the subgroup of order `2^s`, found
/// one bit at a time with powers of a non-residue.
pub fn sqrt_mod(a: u64, p: u64) -> Option<u64> {
    debug_assert!(p < 1 << 32);
    let a = a % p;
    if p == 2 || a == 0 {
        return Some(a);
    }
    // Euler's criterion.
    if pow_mod(a, (p - 1) / 2, p) != 1 {
        return None;
    }
    let root = if p % 4 == 3 {
        pow_mod(a, (p + 1) / 4, p)
    } else {
        let s = (p - 1).trailing_zeros();
        let q = (p - 1) >> s;
        let z = (2..)
            .find(|&z| pow_mod(z, (p - 1) / 2, p) == p - 1)
            .unwrap();
        let mut m = s;
        let mut c = pow_mod(z, q, p);
        let mut t = pow_mod(a, q, p);
        let mut r = pow_mod(a, q.div_ceil(2), p);
        // Invariant: r² = a·t, with t of order dividing 2^(m−1).
        while t != 1 {
            // The least i with t^(2^i) = 1.
            let mut i = 0;
            let mut t2 = t;
            while t2 != 1 {
                t2 = t2 * t2 % p;
                i += 1;
            }
            // The order of t halves at least once per round, which bounds
            // the loop by s rounds.
            assert!(i < m, "Tonelli–Shanks made no progress mod {p}");
            let b = pow_mod(c, 1 << (m - i - 1), p);
            m = i;
            c = b * b % p;
            t = t * c % p;
            r = r * b % p;
        }
        r
    };
    Some(root.min(p - root))
}
