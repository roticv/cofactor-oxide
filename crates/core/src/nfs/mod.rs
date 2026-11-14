//! The general number field sieve, under construction.
//!
//! GNFS finds pairs `(a, b)` for which both `a − b·m` and
//! `F(a, b) = b^d·f(a/b)` are smooth, for polynomials `f` and `x − m` with
//! a common root `m` modulo `N`. A set of pairs whose products are squares
//! on both sides, the rational integers and the number field `ℚ(α)` with
//! `f(α) = 0`, gives a congruence of squares modulo `N`, as in the
//! quadratic sieve. Its values grow more slowly than the quadratic sieve's
//! with the size of `N`, which makes it faster past about 100 digits.
//!
//! [`gnfs`] chooses polynomials by the base-m method with Murphy's size
//! and root optimization ([`Poly`]), sieves
//! special-q lattices of `(a, b)` ([`Sieve`]), makes the matrix smaller by
//! clique removal and merging, solves it with the quadratic sieve's block
//! Lanczos, and takes the square roots in `sqrt`. It is a
//! first, unoptimized version: correct, but slower than the quadratic
//! sieve at every size so far.

mod base;
mod bigmul;
mod cofactor;
mod filter;
mod format;
mod kleinjung;
mod matrix;
pub mod modp;
mod params;
mod poly;
mod score;
mod sieve;
mod sqrt;
mod store;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

pub use base::{CHARACTER_PRIMES, Character, FactorBase, Prime};
pub use format::{parse_poly, parse_relation, poly_text, relation_line};
pub use params::{CHARACTERS, Params};
pub use poly::{Poly, alpha, degree};
pub use sieve::{Relation, Sieve, special_qs};

use crate::arith::Uint;
use crate::parallel::{Parallel, Serial, split};
use crate::qs::{Pending, dependency_masks, split_masks};
use matrix::{Matrix, characters};
use store::{Seen, Store};

/// Rows beyond the columns, after pruning, before solving.
const EXTRA: usize = 64;

/// The densest columns, the quadratic characters among them, are left out
/// of block Lanczos's iterations and satisfied at the end, which takes
/// this many more rows than columns besides [`EXTRA`].
const POST_ROWS: usize = 48;

/// Special-q primes are handed out in blocks of this many integers.
const Q_BLOCK: u64 = 64;

/// The callback asked [`gnfs_with`] to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stopped;

/// GNFS set up for one number: its polynomial, factor bases and
/// parameters.
///
/// Sieving and collecting are separate so that they can run on different
/// threads, as with the quadratic sieve: each thread sieves the special-q
/// of different [`Gnfs::special_qs`] blocks with its own [`Sieve`] from
/// [`Gnfs::sieve`], and one [`Relations`] gathers what they find until
/// [`Relations::is_ready`].
/// Each of [`Relations::dependencies`] then gives a factor with
/// probability about ½, through [`Relations::try_dependency`]. [`gnfs`]
/// does all of it on one thread.
#[derive(Clone, Debug)]
pub struct Gnfs<const LIMBS: usize> {
    n: Uint<LIMBS>,
    params: Params,
    poly: Poly,
    fb: FactorBase,
}

impl<const LIMBS: usize> Gnfs<LIMBS> {
    /// Chooses the polynomial and builds the factor bases for `n`, an odd
    /// composite that isn't a prime power. `None` past about 480 bits,
    /// where [`Poly::select`] can't: there, [`Gnfs::with_poly`] takes
    /// polynomials from elsewhere.
    pub fn new(n: &Uint<LIMBS>, params: &Params) -> Option<Self> {
        Self::new_with(n, params, &Serial)
    }

    /// [`Gnfs::new`], with the polynomial search shared out by `par`.
    pub fn new_with(n: &Uint<LIMBS>, params: &Params, par: &impl Parallel) -> Option<Self> {
        Some(Self::with_poly(n, params, Poly::select_with(n, par)?))
    }

    /// GNFS for `n` with the polynomials `poly`, which must have a common
    /// root modulo `n`, as from [`parse_poly`].
    pub fn with_poly(n: &Uint<LIMBS>, params: &Params, poly: Poly) -> Self {
        let fb = FactorBase::new(
            &poly,
            params.rational_bound,
            params.algebraic_bound,
            CHARACTERS,
        );
        Self {
            n: *n,
            params: *params,
            poly,
            fb,
        }
    }

    pub fn poly(&self) -> &Poly {
        &self.poly
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    pub fn factor_base(&self) -> &FactorBase {
        &self.fb
    }

    /// A sieve for this number.
    pub fn sieve(&self) -> Sieve {
        Sieve::new(&self.poly, &self.fb, &self.params)
    }

    /// The special-q of block `k`: the ideals `(q, r)` with `q` in the
    /// `k`th run of `Q_BLOCK` integers from [`Params::qmin`]. Smaller `q`
    /// give smaller norms, and a `q` within the factor base is divided out
    /// like any other.
    pub fn special_qs(&self, k: u64) -> Vec<(u64, u64)> {
        let start = self.params.qmin + k * Q_BLOCK;
        special_qs(&self.poly, start, start + Q_BLOCK)
    }

    /// The special-q ideals `(q, r)` with `q` in `[lo, hi)`, for sieving a
    /// range of them, as one of several machines can.
    pub fn special_qs_between(&self, lo: u64, hi: u64) -> Vec<(u64, u64)> {
        special_qs(&self.poly, lo, hi)
    }

    /// An empty collection of relations for this number, with its own copy
    /// of the polynomials and factor bases.
    pub fn relations(&self) -> Relations<LIMBS> {
        Relations {
            gnfs: self.clone(),
            store: Store::new(),
            seen: Seen::default(),
            matrix: Matrix::new(&self.fb),
            // Partial relations only start to combine once there are about
            // as many relations as ideals in the factor bases.
            next_check: self.fb.rational.len() + self.fb.algebraic.len(),
            last_check: None,
            ready: false,
            solved: false,
            split: None,
        }
    }
}

/// The relations found so far for one number, and their matrix.
#[derive(Clone, Debug)]
pub struct Relations<const LIMBS: usize> {
    gnfs: Gnfs<LIMBS>,
    store: Store,
    /// The pairs `(a, b)` so far, since special-q lattices overlap.
    seen: Seen,
    matrix: Matrix,
    next_check: usize,
    /// The relations and surplus at the last check that left any rows.
    last_check: Option<(usize, isize)>,
    ready: bool,
    /// Whether `seen` and `matrix` were let go of to solve the matrix.
    /// The store has no algebraic primes to rebuild them from, so no
    /// relations can be added after.
    solved: bool,
    /// The matrix while [`MatrixPart`](crate::qs::MatrixPart)s solve it,
    /// from [`Relations::split_matrix`]: the relations kept, and each
    /// filtered row's.
    split: Option<(Vec<u32>, Vec<Vec<u32>>, Pending)>,
}

const SOLVED: &str = "the matrix was solved: relations can't be added or checked after";

impl<const LIMBS: usize> Relations<LIMBS> {
    /// The polynomials the relations are for.
    pub fn poly(&self) -> &Poly {
        &self.gnfs.poly
    }

    /// Adds `relations`, skipping any already found.
    ///
    /// # Panics
    ///
    /// If the matrix was solved.
    pub fn add(&mut self, relations: impl IntoIterator<Item = Relation>) {
        assert!(!self.solved, "{SOLVED}");
        for rel in relations {
            if self.seen.insert(rel.a, rel.b) && self.store.push(&rel) {
                self.matrix.push(&self.gnfs.fb, &rel);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.len() == 0
    }

    /// Whether the matrix has more rows than columns after pruning.
    /// Checked every 10% more relations until pruning leaves any rows, then
    /// when the last two checks' surpluses say it will be, but never more
    /// often than every 1%.
    ///
    /// RSA-140 was ready at 36.0 million relations, but checks every 10%
    /// missed it at 35.6 million and went on sieving to 39.2 million,
    /// about 4 h more. The surplus climbs steadily once rows survive, from
    /// −2.0 million at 28.5 million relations; on that curve this stops
    /// within 1% of the point. A check took 17 s there, while the sieving
    /// threads went on.
    pub fn is_ready(&mut self) -> bool {
        if !self.ready && self.len() >= self.next_check {
            let (len, surplus) = (self.len(), self.surplus());
            self.ready = surplus >= 0;
            self.next_check = len + self.check_step(len, surplus);
        }
        self.ready
    }

    /// How many more relations to wait for before checking again, after a
    /// check at `len` relations found `surplus`.
    fn check_step(&mut self, len: usize, surplus: isize) -> usize {
        let (most, least) = (len / 10, len / 100);
        // While pruning takes every row, the surplus says nothing.
        if surplus == -((CHARACTERS + POST_ROWS + EXTRA) as isize) {
            self.last_check = None;
            return most;
        }
        let step = match self.last_check.replace((len, surplus)) {
            // Where the line through the two checks reaches 0.
            Some((then, before)) if surplus > before => {
                let short = surplus.unsigned_abs() as u64;
                let gained = (surplus - before) as u64;
                (short * (len - then) as u64 / gained) as usize + 1
            }
            _ => most,
        };
        step.clamp(least, most)
    }

    /// How many more relations than needed are left after pruning:
    /// negative while too few.
    ///
    /// # Panics
    ///
    /// If the matrix was solved.
    pub fn surplus(&mut self) -> isize {
        assert!(!self.solved, "{SOLVED}");
        self.matrix.surplus(CHARACTERS + POST_ROWS, EXTRA)
    }

    /// Lets go of the pairs seen and hands over the matrix, which with
    /// millions of relations take as much memory as filtering needs, for
    /// the browser's 4 GB.
    fn release(&mut self) -> Matrix {
        assert!(!self.solved, "the matrix can only be solved once");
        self.seen = Seen::default();
        self.solved = true;
        core::mem::replace(&mut self.matrix, Matrix::new(&self.gnfs.fb))
    }

    /// The dependencies among the relations.
    ///
    /// # Panics
    ///
    /// If called a second time: solving takes the matrix.
    pub fn dependencies(&mut self) -> Dependencies {
        self.dependencies_with(&Serial)
    }

    /// [`Relations::dependencies_with`], keeping the relations in `spill`
    /// rather than in memory while the matrix is filtered and solved. At
    /// RSA-140 they take about 2 GB.
    pub fn dependencies_spilling(
        &mut self,
        par: &impl Parallel,
        spill: &mut impl Spill,
    ) -> Dependencies {
        self.solve(par, Some(spill))
    }

    /// [`Relations::dependencies`], with block Lanczos's products split by
    /// `par`.
    ///
    /// Each stage takes the last one's data rather than copying it, since
    /// at RSA-120 a copy of the matrix is about a gigabyte: the pairs seen
    /// go first, the matrix's rows are moved out as they are pruned, the
    /// characters are added to the filtered rows in place, and block
    /// Lanczos lets go of the rows once it has packed them.
    pub fn dependencies_with(&mut self, par: &impl Parallel) -> Dependencies {
        self.solve(par, None::<&mut NoSpill>)
    }

    fn solve(&mut self, par: &impl Parallel, spill: Option<&mut impl Spill>) -> Dependencies {
        let (kept, rows, sets, cols, spill) = self.filtered(par, spill);
        let masks = dependency_masks(rows, cols, POST_ROWS, par);
        let deps = Self::combine(&kept, &sets, &masks);
        drop(sets);
        if let Some(spill) = spill {
            drop(kept);
            spill.read(&mut |words| self.store.push_words(words));
        }
        deps
    }

    /// The matrix split into `parts` for [`MatrixPart`]s to solve
    /// together, which [`Relations::finish_split`] then takes their answer
    /// from: each part's [`MatrixPart::dependencies`], joined in order. No
    /// parts if the matrix is small enough to solve at once, and then
    /// [`Relations::finish_split`] takes an empty slice.
    ///
    /// [`MatrixPart`]: crate::qs::MatrixPart
    /// [`MatrixPart::dependencies`]: crate::qs::MatrixPart::dependencies
    ///
    /// # Panics
    ///
    /// If the matrix was solved already.
    pub fn split_matrix(&mut self, parts: usize) -> Vec<Vec<u32>> {
        let (kept, rows, sets, cols, _) = self.filtered(&Serial, None::<&mut NoSpill>);
        let (pending, parts) = split_masks(rows, cols, POST_ROWS, parts);
        self.split = Some((kept, sets, pending));
        parts
    }

    /// The dependencies from the parts of [`Relations::split_matrix`]'s
    /// answer, joined.
    ///
    /// # Panics
    ///
    /// If the matrix wasn't split.
    pub fn finish_split(&mut self, found: &[u64]) -> Dependencies {
        let (kept, sets, pending) = self.split.take().expect("the matrix wasn't split");
        Self::combine(&kept, &sets, &pending.finish(found))
    }

    /// The dependencies among the relations `kept`, from the masks of the
    /// filtered rows, which are the sums of `sets` of them.
    fn combine(kept: &[u32], sets: &[Vec<u32>], masks: &[u64]) -> Dependencies {
        // Each relation is in a dependency if it is in an odd number of
        // the dependency's rows.
        let mut of_kept = vec![0u64; kept.len()];
        for (set, &m) in sets.iter().zip(masks) {
            if m != 0 {
                for &i in set {
                    of_kept[i as usize] ^= m;
                }
            }
        }
        Dependencies::new(kept, &of_kept)
    }

    /// The relations kept after pruning, the filtered rows with their
    /// characters, the relations each is the sum of, and the columns; and
    /// `spill` if the relations were moved there.
    #[allow(clippy::type_complexity)]
    fn filtered<'s, S: Spill>(
        &mut self,
        par: &impl Parallel,
        spill: Option<&'s mut S>,
    ) -> (
        Vec<u32>,
        Vec<Vec<u32>>,
        Vec<Vec<u32>>,
        usize,
        Option<&'s mut S>,
    ) {
        let (kept, rows, cols) = self.release().into_pruned();
        let fb = &self.gnfs.fb;
        let pieces = split(kept.iter().map(|_| 1), par.threads());
        let chars: Vec<u64> = par
            .map(|i| {
                kept[pieces[i]..pieces[i + 1]]
                    .iter()
                    .map(|&r| characters(fb, self.store.ab(r as usize)))
                    .collect::<Vec<_>>()
            })
            .concat();
        // The pairs are only needed again for the square roots, so they
        // can leave memory before filtering, the peak.
        let spill = spill.and_then(|spill| {
            let store = &self.store;
            (0..store.len())
                .all(|i| spill.write(store.words(i)))
                .then_some(spill)
        });
        if spill.is_some() {
            self.store = Store::new();
        }
        let filter::Filtered { mut rows, sets } =
            filter::filter(rows, cols, fb.characters.len() + POST_ROWS + EXTRA);
        for (row, set) in rows.iter_mut().zip(&sets) {
            let mask = set.iter().fold(0, |m, &i| m ^ chars[i as usize]);
            row.reserve_exact(mask.count_ones() as usize);
            row.extend(
                (0..fb.characters.len())
                    .filter(|i| mask >> i & 1 == 1)
                    .map(|i| (cols + i) as u32),
            );
        }
        drop(chars);
        (kept, rows, sets, cols + fb.characters.len(), spill)
    }

    /// Takes the square roots for the dependency `dep`, returning a proper
    /// factor of `n` if it gives one.
    pub fn try_dependency(&self, dep: &[usize]) -> Option<Uint<LIMBS>> {
        self.try_dependency_with(dep, &Serial)
    }

    /// [`Relations::try_dependency`], with the large products split by
    /// `par`.
    pub fn try_dependency_with(&self, dep: &[usize], par: &impl Parallel) -> Option<Uint<LIMBS>> {
        // Only the pairs and the rational primes are needed, read straight
        // from the store rather than unpacking each relation.
        let pairs: Vec<(i64, u64)> = dep.iter().map(|&i| self.store.ab(i)).collect();
        let mut rational = BTreeMap::new();
        for &i in dep {
            for p in self.store.rational(i) {
                *rational.entry(p).or_insert(0u32) += 1;
            }
        }
        sqrt::factor(&self.gnfs.n, &self.gnfs.poly, &pairs, &rational, par)
    }
}

/// Somewhere other than memory to keep the relations, such as a file,
/// while [`Relations::dependencies_spilling`] solves the matrix.
pub trait Spill {
    /// Keeps the next relation's words, returning false if it can't, in
    /// which case the relations stay in memory.
    fn write(&mut self, words: &[u32]) -> bool;

    /// Calls `f` with each relation's words, in the order written. The
    /// relations are lost if this can't, so it should panic rather than
    /// return early.
    fn read(&mut self, f: &mut dyn FnMut(&[u32]));
}

/// The relations stay in memory.
struct NoSpill;

impl Spill for NoSpill {
    fn write(&mut self, _: &[u32]) -> bool {
        false
    }

    fn read(&mut self, _: &mut dyn FnMut(&[u32])) {}
}

/// The dependencies among a [`Relations`]' relations, up to 64, as a
/// word for each relation in any: its bits say which. Lists of millions of
/// relations each would take far more memory.
#[derive(Clone, Debug, Default)]
pub struct Dependencies {
    /// The relations in some dependency, by index, and the word for each.
    relations: Vec<u32>,
    masks: Vec<u64>,
    /// The bits of the distinct dependencies.
    bits: Vec<u32>,
}

impl Dependencies {
    /// From each relation `kept[i]`'s word `masks[i]`.
    fn new(kept: &[u32], masks: &[u64]) -> Self {
        let (relations, masks): (Vec<u32>, Vec<u64>) = kept
            .iter()
            .zip(masks)
            .filter(|&(_, &m)| m != 0)
            .map(|(&r, &m)| (r, m))
            .unzip();
        // A fingerprint of each dependency, to drop repeats.
        let mut prints = [0u64; 64];
        for (&r, &m) in relations.iter().zip(&masks) {
            let h = u64::from(r)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left(29)
                | 1;
            for (q, print) in prints.iter_mut().enumerate() {
                if m >> q & 1 == 1 {
                    *print = print.wrapping_add(h);
                }
            }
        }
        let mut bits: Vec<u32> = Vec::new();
        for q in 0..64 {
            let p = prints[q as usize];
            if p != 0 && bits.iter().all(|&b| prints[b as usize] != p) {
                bits.push(q);
            }
        }
        Self {
            relations,
            masks,
            bits,
        }
    }

    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Each dependency's relations in turn, as from [`Dependencies::get`].
    pub fn iter(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }

    /// Dependency `i`'s relations, as indices for
    /// [`Relations::try_dependency`].
    pub fn get(&self, i: usize) -> Vec<usize> {
        let q = self.bits[i];
        self.relations
            .iter()
            .zip(&self.masks)
            .filter(|&(_, &m)| m >> q & 1 == 1)
            .map(|(&r, _)| r as usize)
            .collect()
    }
}

/// Factors `n`, an odd composite that isn't a prime power, with GNFS on
/// one thread. Returns a proper factor, or `None` if none was found or
/// `n` is too large to choose polynomials for.
pub fn gnfs<const LIMBS: usize>(n: &Uint<LIMBS>) -> Option<Uint<LIMBS>> {
    let params = Params::for_bits(n.bits());
    gnfs_with(n, &params, |_, _| ControlFlow::Continue(())).unwrap_or(None)
}

/// Factors `n` like [`gnfs`] with `params`, calling `progress(q,
/// relations)` after each special-q, and stopping if it returns `Break`.
pub fn gnfs_with<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    params: &Params,
    mut progress: impl FnMut(u64, usize) -> ControlFlow<()>,
) -> Result<Option<Uint<LIMBS>>, Stopped> {
    let Some(gnfs) = Gnfs::new(n, params) else {
        return Ok(None);
    };
    let mut sieve = gnfs.sieve();
    let mut relations = gnfs.relations();
    'sieve: for k in 0.. {
        for (q, r) in gnfs.special_qs(k) {
            relations.add(sieve.special_q(q, r));
            if progress(q, relations.len()).is_break() {
                return Err(Stopped);
            }
            if relations.is_ready() {
                break 'sieve;
            }
        }
    }
    Ok(relations
        .dependencies()
        .iter()
        .find_map(|dep| relations.try_dependency(&dep)))
}
