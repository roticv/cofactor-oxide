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
/// rational and algebraic prime counts, the sign, and whether the primes
/// are wide; then the primes, a word each, or two, low first, if any of
/// the relation's is 2^32 or more.
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

    /// Adds `rel`, unless its prime counts don't fit, below 2^15.
    pub fn push(&mut self, rel: &Relation) -> bool {
        let (r, a) = (rel.rational.len(), rel.algebraic.len());
        if r >= 1 << 15 || a >= 1 << 15 {
            return false;
        }
        let primes = rel.rational.iter().chain(&rel.algebraic);
        let wide = primes.clone().any(|&p| p >> 32 != 0);
        let (ua, b) = (rel.a as u64, rel.b);
        let counts = r as u32
            | (a as u32) << 15
            | if wide { WIDE } else { 0 }
            | if rel.negative { NEGATIVE } else { 0 };
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

    fn words(&self, i: usize) -> &[u32] {
        self.records.get(i)
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

    /// Relation `i`, unpacked.
    pub fn get(&self, i: usize) -> Relation {
        let w = self.words(i);
        let (a, b) = self.ab(i);
        let r = (w[4] & 0x7fff) as usize;
        let primes: Vec<u64> = if w[4] & WIDE != 0 {
            w[HEADER..]
                .chunks(2)
                .map(|p| u64::from(p[0]) | u64::from(p[1]) << 32)
                .collect()
        } else {
            w[HEADER..].iter().map(|&p| p.into()).collect()
        };
        Relation {
            a,
            b,
            negative: w[4] & NEGATIVE != 0,
            rational: primes[..r].to_vec(),
            algebraic: primes[r..].to_vec(),
        }
    }
}

/// Records of words, kept in chunks of a fixed size rather than one
/// vector, so that adding to millions of them never copies them all to a
/// vector twice the size: in the browser's 4 GB, the copy and the space
/// the old vector leaves can be what runs out.
#[derive(Clone, Debug, Default)]
pub struct Records {
    chunks: Vec<Vec<u32>>,
    /// Where each record starts: its chunk, and its offset in the chunk,
    /// each record within one chunk.
    starts: Vec<Vec<(u32, u32)>>,
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
            _ => self
                .chunks
                .push(Vec::with_capacity(CHUNK.max(record.len()))),
        }
        let chunk = self.chunks.len() - 1;
        let words = &mut self.chunks[chunk];
        if self.len.is_multiple_of(STARTS) {
            self.starts.push(Vec::with_capacity(STARTS));
        }
        let starts = self.starts.last_mut().expect("just pushed");
        starts.push((chunk as u32, words.len() as u32));
        words.extend_from_slice(record);
        self.len += 1;
    }

    fn start(&self, i: usize) -> (usize, usize) {
        let (c, o) = self.starts[i / STARTS][i % STARTS];
        (c as usize, o as usize)
    }

    /// Record `i`.
    pub fn get(&self, i: usize) -> &[u32] {
        let (c, o) = self.start(i);
        let end = match (i + 1 < self.len).then(|| self.start(i + 1)) {
            Some((next, e)) if next == c => e,
            _ => self.chunks[c].len(),
        };
        &self.chunks[c][o..end]
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
            len,
        } = self;
        let start = |i: usize| {
            let (c, o) = starts[i / STARTS][i % STARTS];
            (c as usize, o as usize)
        };
        let mut i = 0;
        for (c, chunk) in chunks.into_iter().enumerate() {
            while i < len && start(i).0 == c {
                let o = start(i).1;
                let end = match (i + 1 < len).then(|| start(i + 1)) {
                    Some((next, e)) if next == c => e,
                    _ => chunk.len(),
                };
                f(i, &chunk[o..end]);
                i += 1;
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
