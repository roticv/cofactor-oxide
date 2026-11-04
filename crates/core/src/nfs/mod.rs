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

use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

pub use base::{FactorBase, Prime};
pub use format::{parse_poly, parse_relation, poly_text, relation_line};
pub use params::{CHARACTERS, Params};
pub use poly::{Poly, alpha, degree};
pub use sieve::{Relation, Sieve, special_qs};

use crate::arith::Uint;
use crate::parallel::{Parallel, Serial, split};
use crate::qs::dependencies_post;
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
            ready: false,
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
    ready: bool,
}

impl<const LIMBS: usize> Relations<LIMBS> {
    /// The polynomials the relations are for.
    pub fn poly(&self) -> &Poly {
        &self.gnfs.poly
    }

    /// Adds `relations`, skipping any already found.
    pub fn add(&mut self, relations: impl IntoIterator<Item = Relation>) {
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

    /// Whether the matrix has more rows than columns after pruning. Only
    /// checked every 10% more relations.
    pub fn is_ready(&mut self) -> bool {
        if !self.ready && self.len() >= self.next_check {
            self.next_check = self.len() + self.len() / 10;
            self.ready = self.matrix.ready(CHARACTERS + POST_ROWS, EXTRA);
        }
        self.ready
    }

    /// The dependencies among the relations, as indices into them.
    pub fn dependencies(&self) -> Vec<Vec<usize>> {
        self.dependencies_with(&Serial)
    }

    /// [`Relations::dependencies`], with block Lanczos's products split by
    /// `par`.
    pub fn dependencies_with(&self, par: &impl Parallel) -> Vec<Vec<usize>> {
        let fb = &self.gnfs.fb;
        let (kept, rows, cols) = self.matrix.pruned();
        let filtered = filter::filter(rows, cols, fb.characters.len() + POST_ROWS + EXTRA);
        let pieces = split(kept.iter().map(|_| 1), par.threads());
        let chars: Vec<u64> = par
            .map(|i| {
                kept[pieces[i]..pieces[i + 1]]
                    .iter()
                    .map(|&r| characters(fb, self.store.ab(r)))
                    .collect::<Vec<_>>()
            })
            .concat();
        let rows: Vec<Vec<usize>> = filtered
            .rows
            .iter()
            .zip(&filtered.sets)
            .map(|(row, set)| {
                let mask = set.iter().fold(0, |m, &i| m ^ chars[i as usize]);
                let dense = (0..fb.characters.len()).filter(|i| mask >> i & 1 == 1);
                row.iter()
                    .map(|&c| c as usize)
                    .chain(dense.map(|i| cols + i))
                    .collect()
            })
            .collect();
        let deps = dependencies_post(&rows, cols + fb.characters.len(), POST_ROWS, par);
        // Each dependency's relations: those in an odd number of its rows.
        let pieces = split(deps.iter().map(|_| 1), par.threads());
        par.map(|i| {
            deps[pieces[i]..pieces[i + 1]]
                .iter()
                .map(|dep| {
                    let mut odd = vec![false; kept.len()];
                    for &i in dep.iter().flat_map(|&row| &filtered.sets[row]) {
                        odd[i as usize] ^= true;
                    }
                    (0..kept.len())
                        .filter(|&i| odd[i])
                        .map(|i| kept[i])
                        .collect()
                })
                .collect::<Vec<Vec<usize>>>()
        })
        .concat()
    }

    /// Takes the square roots for the dependency `dep`, returning a proper
    /// factor of `n` if it gives one.
    pub fn try_dependency(&self, dep: &[usize]) -> Option<Uint<LIMBS>> {
        self.try_dependency_with(dep, &Serial)
    }

    /// [`Relations::try_dependency`], with the large products split by
    /// `par`.
    pub fn try_dependency_with(&self, dep: &[usize], par: &impl Parallel) -> Option<Uint<LIMBS>> {
        let dep: Vec<Relation> = dep.iter().map(|&i| self.store.get(i)).collect();
        let dep: Vec<&Relation> = dep.iter().collect();
        sqrt::factor(&self.gnfs.n, &self.gnfs.poly, &dep, par)
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
        .find_map(|dep| relations.try_dependency(dep)))
}
