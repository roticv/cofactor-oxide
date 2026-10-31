//! The quadratic sieve on a decimal string, for front ends that share the
//! sieving between threads or Web Workers.
//!
//! [`QsJob`] picks an integer width with room for `kN`, and hides it:
//! relations travel between sieves and the collector as `u32` words, which
//! can be sent through a channel or posted to another Web Worker as they
//! are.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::{Params, Relation, Relations, Sieve, Siqs};
use crate::arith::{LIMB_BITS, Limb, ParseUintError, Uint};

/// Widths, in limbs, to pick from. `kN` needs up to 7 bits more than `n`.
const WIDTHS: [usize; 3] = [4, 8, 16];

/// The outcome of setting up a [`QsJob`].
pub enum Start {
    /// The number needs sieving.
    Sieve(QsJob),
    /// A factor turned up without sieving.
    Factor(String),
    /// The number has no factor for the sieve to find: it is a prime from
    /// the factor base.
    Fail,
}

/// The quadratic sieve set up for one number, which can hand out sieves
/// and a collector for their relations.
pub struct QsJob(Box<dyn Job>);

/// One thread's or Web Worker's sieve.
pub struct QsSieve(Box<dyn SieveWords>);

/// Gathers relations from all the sieves, and finds the factor at the end.
pub struct QsCollector(Box<dyn Collect>);

trait Job: Send + Sync {
    fn bits(&self) -> u32;
    fn sieve(&self, seed: u64) -> Box<dyn SieveWords>;
    fn collector(&self) -> Box<dyn Collect>;
}

trait SieveWords: Send {
    fn sieve_poly(&mut self) -> Vec<u32>;
}

trait Collect: Send {
    fn add(&mut self, words: &[u32]);
    fn len(&self) -> usize;
    fn target(&self) -> usize;
    fn finish(&self) -> Option<String>;
}

impl QsJob {
    /// Sets up the sieve for the decimal number `n`, with the parameters
    /// for its size.
    ///
    /// # Panics
    ///
    /// Panics if `n` is too small for the factor base.
    pub fn start(n: &str) -> Result<Start, ParseUintError> {
        let bits = n.parse::<Uint<16>>()?.bits();
        let params = Params::for_bits(bits);
        match WIDTHS.iter().find(|&&w| bits + 8 <= w as u32 * LIMB_BITS) {
            Some(4) => start::<4>(n, &params),
            Some(8) => start::<8>(n, &params),
            _ => start::<16>(n, &params),
        }
    }

    /// Bits of the number.
    pub fn bits(&self) -> u32 {
        self.0.bits()
    }

    /// A sieve for this number. Each sieve needs its own seed.
    pub fn sieve(&self, seed: u64) -> QsSieve {
        QsSieve(self.0.sieve(seed))
    }

    pub fn collector(&self) -> QsCollector {
        QsCollector(self.0.collector())
    }
}

impl QsSieve {
    /// Sieves one polynomial, returning its relations encoded for
    /// [`QsCollector::add`].
    pub fn sieve_poly(&mut self) -> Vec<u32> {
        self.0.sieve_poly()
    }
}

impl QsCollector {
    /// Adds the relations from one or more calls to
    /// [`QsSieve::sieve_poly`], concatenated.
    ///
    /// # Panics
    ///
    /// Panics if `words` isn't a sequence of encoded relations.
    pub fn add(&mut self, words: &[u32]) {
        self.0.add(words);
    }

    /// Rows of the matrix so far.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    /// Rows needed.
    pub fn target(&self) -> usize {
        self.0.target()
    }

    pub fn is_complete(&self) -> bool {
        self.0.len() >= self.0.target()
    }

    /// Finds a factor from the relations, once complete. `None` means
    /// every dependency gave a trivial factor, which is what happens when
    /// the number is prime or a prime power.
    pub fn finish(&self) -> Option<String> {
        self.0.finish()
    }
}

fn start<const LIMBS: usize>(n: &str, params: &Params) -> Result<Start, ParseUintError> {
    let n = n.parse::<Uint<LIMBS>>()?;
    Ok(match Siqs::new(&n, params) {
        Ok(siqs) => Start::Sieve(QsJob(Box::new(siqs))),
        Err(Some(f)) => Start::Factor(f.to_string()),
        Err(None) => Start::Fail,
    })
}

impl<const LIMBS: usize> Job for Siqs<LIMBS> {
    fn bits(&self) -> u32 {
        self.n.bits()
    }

    fn sieve(&self, seed: u64) -> Box<dyn SieveWords> {
        Box::new(Siqs::sieve(self, seed))
    }

    fn collector(&self) -> Box<dyn Collect> {
        Box::new((self.clone(), self.relations()))
    }
}

impl<const LIMBS: usize> SieveWords for Sieve<LIMBS> {
    fn sieve_poly(&mut self) -> Vec<u32> {
        let mut words = Vec::new();
        for r in Sieve::sieve_poly(self) {
            encode(&r, &mut words);
        }
        words
    }
}

impl<const LIMBS: usize> Collect for (Siqs<LIMBS>, Relations<LIMBS>) {
    fn add(&mut self, mut words: &[u32]) {
        while !words.is_empty() {
            self.1.add(decode(&mut words));
        }
    }

    fn len(&self) -> usize {
        self.1.len()
    }

    fn target(&self) -> usize {
        self.1.target()
    }

    fn finish(&self) -> Option<String> {
        self.0.finish(&self.1).map(|f| f.to_string())
    }
}

/// Appends a relation as: the number of words of `t`, the number of
/// factors, the sign, the large prime's low and high halves, the words of
/// `t` from least significant, and the factors.
// Limb is u32 or u64, so the casts are needed on one target and redundant
// on the other.
#[allow(clippy::unnecessary_cast)]
fn encode<const LIMBS: usize>(r: &Relation<LIMBS>, out: &mut Vec<u32>) {
    let per_limb = LIMB_BITS as usize / 32;
    let mut t: Vec<u32> =
        r.t.as_limbs()
            .iter()
            .flat_map(|&limb| {
                let limb = limb as u64;
                [limb as u32, (limb >> 32) as u32]
                    .into_iter()
                    .take(per_limb)
            })
            .collect();
    while t.last() == Some(&0) {
        t.pop();
    }
    out.extend([
        t.len() as u32,
        r.factors.len() as u32,
        u32::from(r.negative),
        r.large as u32,
        (r.large >> 32) as u32,
    ]);
    out.extend(t);
    out.extend(&r.factors);
}

/// Reads one relation written by [`encode`] from the front of `words`.
// Limb is u32 or u64, so the casts are needed on one target and redundant
// on the other.
#[allow(clippy::unnecessary_cast)]
fn decode<const LIMBS: usize>(words: &mut &[u32]) -> Relation<LIMBS> {
    let (head, rest) = words.split_at(5);
    let (t_len, factors) = (head[0] as usize, head[1] as usize);
    let (t, rest) = rest.split_at(t_len);
    let (factors, rest) = rest.split_at(factors);
    *words = rest;
    let per_limb = LIMB_BITS as usize / 32;
    let mut limbs = [0 as Limb; LIMBS];
    for (i, &w) in t.iter().enumerate() {
        limbs[i / per_limb] |= (w as Limb) << (32 * (i % per_limb));
    }
    Relation {
        t: Uint::from_limbs(limbs),
        negative: head[2] != 0,
        factors: factors.to_vec(),
        large: u64::from(head[3]) | u64::from(head[4]) << 32,
    }
}
