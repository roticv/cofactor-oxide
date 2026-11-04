//! Compact storage for millions of relations: each packed into a few
//! words of one vector rather than two vectors of its own, and the pairs
//! `(a, b)` seen so far as 64-bit fingerprints in an open-addressing
//! table. Together they take a fraction of the memory of `Vec<Relation>`
//! and a `BTreeSet`, which matters most in the browser, where memory is
//! capped at 4 GB.

use alloc::vec;
use alloc::vec::Vec;

use super::sieve::Relation;

/// Relations, each as words: `a` and `b`, low word first; a word with the
/// rational and algebraic prime counts and the sign; then the primes.
#[derive(Clone, Debug, Default)]
pub struct Store {
    words: Vec<u32>,
    /// Where each relation starts, and where the last ends.
    starts: Vec<usize>,
}

/// The words before a relation's primes.
const HEADER: usize = 5;

/// The sign bit of the counts word.
const NEGATIVE: u32 = 1 << 31;

impl Store {
    pub fn new() -> Self {
        Self {
            words: Vec::new(),
            starts: vec![0],
        }
    }

    pub fn len(&self) -> usize {
        self.starts.len() - 1
    }

    /// Adds `rel`, unless one of its primes or prime counts doesn't fit:
    /// primes must be below 2^32, as every large-prime bound here is, and
    /// counts below 2^15.
    pub fn push(&mut self, rel: &Relation) -> bool {
        let (r, a) = (rel.rational.len(), rel.algebraic.len());
        let fits = |p: &u64| u32::try_from(*p).is_ok();
        if r >= 1 << 15
            || a >= 1 << 15
            || !rel.rational.iter().all(fits)
            || !rel.algebraic.iter().all(fits)
        {
            return false;
        }
        let (ua, b) = (rel.a as u64, rel.b);
        let counts = r as u32 | (a as u32) << 15 | if rel.negative { NEGATIVE } else { 0 };
        self.words.extend([
            ua as u32,
            (ua >> 32) as u32,
            b as u32,
            (b >> 32) as u32,
            counts,
        ]);
        self.words
            .extend(rel.rational.iter().chain(&rel.algebraic).map(|&p| p as u32));
        self.starts.push(self.words.len());
        true
    }

    fn words(&self, i: usize) -> &[u32] {
        &self.words[self.starts[i]..self.starts[i + 1]]
    }

    /// Relation `i`'s `(a, b)`.
    pub fn ab(&self, i: usize) -> (i64, u64) {
        let w = self.words(i);
        let a = u64::from(w[0]) | u64::from(w[1]) << 32;
        (a as i64, u64::from(w[2]) | u64::from(w[3]) << 32)
    }

    /// Relation `i`, unpacked.
    pub fn get(&self, i: usize) -> Relation {
        let w = self.words(i);
        let (a, b) = self.ab(i);
        let r = (w[4] & 0x7fff) as usize;
        let primes = &w[HEADER..];
        Relation {
            a,
            b,
            negative: w[4] & NEGATIVE != 0,
            rational: primes[..r].iter().map(|&p| p.into()).collect(),
            algebraic: primes[r..].iter().map(|&p| p.into()).collect(),
        }
    }
}

/// The pairs `(a, b)` seen, as fingerprints. Two pairs with the same
/// 64-bit fingerprint would be taken for one, which with millions of pairs
/// happens with probability around 10⁻⁷, and costs one relation.
#[derive(Clone, Debug, Default)]
pub struct Seen {
    /// Fingerprints, 0 for an empty slot; the length is a power of 2.
    slots: Vec<u64>,
    len: usize,
}

impl Seen {
    /// Records `(a, b)`, returning whether it is new.
    pub fn insert(&mut self, a: i64, b: u64) -> bool {
        if 2 * (self.len + 1) > self.slots.len() {
            self.grow();
        }
        let key = fingerprint(a, b);
        if place(&mut self.slots, key) {
            self.len += 1;
            true
        } else {
            false
        }
    }

    fn grow(&mut self) {
        let old = core::mem::take(&mut self.slots);
        self.slots = vec![0; (2 * old.len()).max(1 << 10)];
        for key in old.into_iter().filter(|&k| k != 0) {
            place(&mut self.slots, key);
        }
    }
}

/// Puts `key` in the first free slot from its own, by linear probing,
/// unless it is already there. Returns whether it was put.
fn place(slots: &mut [u64], key: u64) -> bool {
    let mask = slots.len() - 1;
    let mut i = key as usize & mask;
    loop {
        match slots[i] {
            0 => {
                slots[i] = key;
                return true;
            }
            k if k == key => return false,
            _ => i = (i + 1) & mask,
        }
    }
}

/// A hash of `(a, b)` that is never 0.
fn fingerprint(a: i64, b: u64) -> u64 {
    let mix = |mut x: u64| {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    };
    mix(mix(a as u64) ^ b.rotate_left(32)).max(1)
}
