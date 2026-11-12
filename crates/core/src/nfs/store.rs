//! Compact storage for millions of relations: each packed into a few
//! words of a chunked [`Records`] rather than two vectors of its own, and
//! the pairs `(a, b)` seen so far as 64-bit fingerprints in an
//! open-addressing table. Together they take a fraction of the memory of `Vec<Relation>`
//! and a `BTreeSet`, which matters most in the browser, where memory is
//! capped at 4 GB.

use alloc::vec;
use alloc::vec::Vec;

use super::sieve::Relation;

/// Relations, each as words: `a` and `b`, low word first; a word with the
/// rational prime count, the sign, and whether the primes are wide; then
/// the rational primes, a word each, or two, low first, if any of the
/// relation's is 2^32 or more.
///
/// The algebraic primes aren't kept: once the relation's row is in the
/// matrix, only the square roots need the relation again, and they need
/// only `(a, b)` and the rational primes. At RSA-140 the algebraic primes
/// would have taken 2.5 GB, two thirds of the primes.
#[derive(Clone, Debug, Default)]
pub struct Store {
    records: Records,
    /// The words of the relation being added.
    words: Vec<u32>,
}

/// The words before a relation's primes.
const HEADER: usize = 5;

/// The sign bit of the counts word, and the one for wide primes.
const NEGATIVE: u32 = 1 << 31;
const WIDE: u32 = 1 << 30;

impl Store {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Adds `rel`, unless its rational prime count doesn't fit, below
    /// 2^15.
    pub fn push(&mut self, rel: &Relation) -> bool {
        let r = rel.rational.len();
        if r >= 1 << 15 {
            return false;
        }
        let primes = rel.rational.iter();
        let wide = primes.clone().any(|&p| p >> 32 != 0);
        let (ua, b) = (rel.a as u64, rel.b);
        let counts =
            r as u32 | if wide { WIDE } else { 0 } | if rel.negative { NEGATIVE } else { 0 };
        self.words.clear();
        self.words.extend([
            ua as u32,
            (ua >> 32) as u32,
            b as u32,
            (b >> 32) as u32,
            counts,
        ]);
        if wide {
            self.words
                .extend(primes.flat_map(|&p| [p as u32, (p >> 32) as u32]));
        } else {
            self.words.extend(primes.map(|&p| p as u32));
        }
        self.records.push(&self.words);
        true
    }

    /// Relation `i`'s words, as [`Store::push_words`] takes them back.
    pub(super) fn words(&self, i: usize) -> &[u32] {
        self.records.get(i)
    }

    /// Adds a relation's words, as [`Store::words`] gave them.
    pub(super) fn push_words(&mut self, words: &[u32]) {
        self.records.push(words);
    }

    /// Relation `i`'s `(a, b)`.
    pub fn ab(&self, i: usize) -> (i64, u64) {
        let w = self.words(i);
        let a = u64::from(w[0]) | u64::from(w[1]) << 32;
        (a as i64, u64::from(w[2]) | u64::from(w[3]) << 32)
    }

    /// Relation `i`'s rational primes, without unpacking the rest.
    pub fn rational(&self, i: usize) -> impl Iterator<Item = u64> + '_ {
        let w = self.words(i);
        let r = (w[4] & 0x7fff) as usize;
        let wide = w[4] & WIDE != 0;
        let step = if wide { 2 } else { 1 };
        w[HEADER..HEADER + r * step]
            .chunks(step)
            .map(|p| u64::from(p[0]) | p.get(1).map_or(0, |&h| u64::from(h) << 32))
    }
}

/// Records of words, kept in chunks of a fixed size rather than one
/// vector, so that adding to millions of them never copies them all to a
/// vector twice the size: in the browser's 4 GB, the copy and the space
/// the old vector leaves can be what runs out.
#[derive(Clone, Debug, Default)]
pub struct Records {
    chunks: Vec<Vec<u32>>,
    /// Where each record starts in its chunk, each record within one.
    starts: Vec<Vec<u32>>,
    /// The index of each chunk's first record, which with the starts
    /// takes 4 bytes a record rather than 8 for a chunk and an offset each.
    firsts: Vec<usize>,
    len: usize,
}

/// Words in a chunk, unless a record needs more.
const CHUNK: usize = 1 << 22;

/// Starts in each vector of them.
const STARTS: usize = 1 << 20;

impl Records {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push(&mut self, record: &[u32]) {
        match self.chunks.last() {
            Some(c) if c.len() + record.len() <= c.capacity() => {}
            _ => {
                self.chunks
                    .push(Vec::with_capacity(CHUNK.max(record.len())));
                self.firsts.push(self.len);
            }
        }
        let words = self.chunks.last_mut().expect("just pushed");
        if self.len.is_multiple_of(STARTS) {
            self.starts.push(Vec::with_capacity(STARTS));
        }
        let starts = self.starts.last_mut().expect("just pushed");
        starts.push(words.len() as u32);
        words.extend_from_slice(record);
        self.len += 1;
    }

    fn offset(&self, i: usize) -> usize {
        self.starts[i / STARTS][i % STARTS] as usize
    }

    /// Record `i`.
    pub fn get(&self, i: usize) -> &[u32] {
        let c = self.firsts.partition_point(|&f| f <= i) - 1;
        let end = if i + 1 < self.len && self.firsts.get(c + 1) != Some(&(i + 1)) {
            self.offset(i + 1)
        } else {
            self.chunks[c].len()
        };
        &self.chunks[c][self.offset(i)..end]
    }

    /// Every record, in order.
    pub fn iter(&self) -> impl Iterator<Item = &[u32]> {
        self.chunks.iter().enumerate().flat_map(move |(c, chunk)| {
            let next = self.firsts.get(c + 1).copied().unwrap_or(self.len);
            (self.firsts[c]..next).map(move |i| {
                let end = if i + 1 < next {
                    self.offset(i + 1)
                } else {
                    chunk.len()
                };
                &chunk[self.offset(i)..end]
            })
        })
    }

    /// Every record's words, in order.
    pub fn words(&self) -> impl Iterator<Item = &u32> {
        self.chunks.iter().flatten()
    }

    /// Calls `f` with each record and its index, in order, letting go of
    /// each chunk once its records are passed, so that copying them out
    /// never holds both copies whole.
    pub fn drain(self, mut f: impl FnMut(usize, &[u32])) {
        let Self {
            chunks,
            starts,
            firsts,
            len,
        } = self;
        let offset = |i: usize| starts[i / STARTS][i % STARTS] as usize;
        for (c, chunk) in chunks.into_iter().enumerate() {
            let next = firsts.get(c + 1).copied().unwrap_or(len);
            for i in firsts[c]..next {
                let end = if i + 1 < next {
                    offset(i + 1)
                } else {
                    chunk.len()
                };
                f(i, &chunk[offset(i)..end]);
            }
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
    /// Records `(a, b)`, returning whether it is new. The table grows
    /// when three quarters full, where probes stay short: at RSA-140's 39.2
    /// million pairs that is 64 million slots, 0.5 GB, rather than the 128
    /// million of growing at half.
    pub fn insert(&mut self, a: i64, b: u64) -> bool {
        if 4 * (self.len + 1) > 3 * self.slots.len() {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Records of varied lengths over several chunks, with one longer
    /// than a chunk, read back by index, in order, and drained.
    #[test]
    fn records_across_chunks() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut lens: Vec<usize> = (0..400_000).map(|_| (next() % 60) as usize).collect();
        lens[200_000] = CHUNK + 5;
        let record =
            |i: usize, len: usize| -> Vec<u32> { (0..len).map(|k| (i * 7 + k) as u32).collect() };
        let mut records = Records::default();
        for (i, &len) in lens.iter().enumerate() {
            records.push(&record(i, len));
        }
        assert_eq!(records.len(), lens.len());
        assert!(records.chunks.len() > 3);
        for i in (0..lens.len())
            .step_by(997)
            .chain([199_999, 200_000, 200_001, lens.len() - 1])
        {
            assert_eq!(records.get(i), record(i, lens[i]), "record {i}");
        }
        assert!(
            records
                .iter()
                .enumerate()
                .all(|(i, r)| r == record(i, lens[i]))
        );
        let mut seen = 0;
        records.drain(|i, r| {
            assert_eq!((i, r), (seen, &record(i, lens[i])[..]));
            seen += 1;
        });
        assert_eq!(seen, lens.len());
    }
}
