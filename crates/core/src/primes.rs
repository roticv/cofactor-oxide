//! Prime generation with a segmented sieve of Eratosthenes.

use alloc::vec;
use alloc::vec::Vec;

/// Odd numbers covered by one sieve segment. 32 KiB of flags fits in L1.
const SEGMENT: usize = 1 << 15;

/// Iterator over the primes in `[lo, hi]`, in increasing order.
///
/// Memory use is one segment plus the primes up to `√hi`, however wide the
/// range. Sieving primes up to `√hi` uses a flag per number, so `hi` should
/// stay below about 2^44.
pub struct Primes {
    hi: u64,
    /// Odd primes up to `√hi`, used to cross off composites.
    base: Vec<u64>,
    /// `flags[i]` says whether `start + 2i` is prime.
    flags: Vec<bool>,
    /// The odd number that `flags[0]` stands for.
    start: u64,
    next_index: usize,
    /// Whether 2 is still due to be yielded.
    two_pending: bool,
}

impl Primes {
    pub fn new(lo: u64, hi: u64) -> Self {
        let root = hi.isqrt();
        let base = small_odd_primes(root);
        let mut primes = Self {
            hi,
            base,
            flags: vec![false; SEGMENT],
            start: (lo.max(3)) | 1,
            next_index: 0,
            two_pending: lo <= 2 && hi >= 2,
        };
        primes.sieve_segment();
        primes
    }

    /// Fills `flags` for the segment beginning at `start`.
    fn sieve_segment(&mut self) {
        self.flags.fill(true);
        self.next_index = 0;
        let end = self.start + 2 * SEGMENT as u64;
        for &p in &self.base {
            if p * p >= end {
                break;
            }
            // First odd multiple of p that's in the segment and at least p²,
            // since smaller multiples have a smaller prime factor.
            let mut m = (self.start.div_ceil(p) * p).max(p * p);
            if m % 2 == 0 {
                m += p;
            }
            let mut i = ((m - self.start) / 2) as usize;
            while i < SEGMENT {
                self.flags[i] = false;
                i += p as usize;
            }
        }
    }
}

impl Iterator for Primes {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.two_pending {
            self.two_pending = false;
            return Some(2);
        }
        loop {
            while self.next_index < SEGMENT {
                let i = self.next_index;
                self.next_index += 1;
                let n = self.start + 2 * i as u64;
                if n > self.hi {
                    return None;
                }
                if self.flags[i] {
                    return Some(n);
                }
            }
            self.start += 2 * SEGMENT as u64;
            if self.start > self.hi {
                return None;
            }
            self.sieve_segment();
        }
    }
}

/// Odd primes up to `limit`, by a plain sieve.
fn small_odd_primes(limit: u64) -> Vec<u64> {
    let limit = limit as usize;
    let mut is_prime = vec![true; limit + 1];
    let mut primes = Vec::new();
    for i in (3..=limit).step_by(2) {
        if is_prime[i] {
            primes.push(i as u64);
            for j in (i * i..=limit).step_by(2 * i) {
                is_prime[j] = false;
            }
        }
    }
    primes
}
