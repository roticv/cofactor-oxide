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
//! special-q lattices of `(a, b)` ([`Sieve`]), solves the matrix with the quadratic
//! sieve's block Lanczos, and takes the square roots in `sqrt`. It is a
//! first, unoptimized version: correct, but slower than the quadratic
//! sieve at every size so far.

mod base;
mod matrix;
pub mod modp;
mod params;
mod poly;
mod score;
mod sieve;
mod sqrt;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::ops::ControlFlow;

pub use base::{FactorBase, Prime};
pub use params::{CHARACTERS, Params};
pub use poly::{Poly, alpha, degree};
pub use sieve::{Relation, Sieve, special_qs};

use crate::arith::Uint;
use crate::qs::dependencies;
use matrix::Matrix;

/// Rows beyond the columns, after pruning, before solving.
const EXTRA: usize = 64;

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
    /// composite that isn't a prime power.
    pub fn new(n: &Uint<LIMBS>, params: &Params) -> Self {
        let poly = Poly::select(n);
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

    pub fn factor_base(&self) -> &FactorBase {
        &self.fb
    }

    /// A sieve for this number.
    pub fn sieve(&self) -> Sieve<'_> {
        Sieve::new(&self.poly, &self.fb, &self.params)
    }

    /// The special-q of block `k`: the ideals `(q, r)` with `q` in the
    /// `k`th run of [`Q_BLOCK`] integers from a quarter of the algebraic
    /// bound. Smaller `q` give smaller norms, and a `q` within the factor
    /// base is divided out like any other.
    pub fn special_qs(&self, k: u64) -> Vec<(u64, u64)> {
        let start = u64::from(self.params.algebraic_bound) / 4 + k * Q_BLOCK;
        special_qs(&self.poly, start, start + Q_BLOCK)
    }

    /// An empty collection of relations for this number.
    pub fn relations(&self) -> Relations<'_, LIMBS> {
        Relations {
            gnfs: self,
            relations: Vec::new(),
            seen: BTreeSet::new(),
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
pub struct Relations<'a, const LIMBS: usize> {
    gnfs: &'a Gnfs<LIMBS>,
    relations: Vec<Relation>,
    /// The pairs `(a, b)` so far, since special-q lattices overlap.
    seen: BTreeSet<(i64, u64)>,
    matrix: Matrix,
    next_check: usize,
    ready: bool,
}

impl<const LIMBS: usize> Relations<'_, LIMBS> {
    /// Adds `relations`, skipping any already found.
    pub fn add(&mut self, relations: impl IntoIterator<Item = Relation>) {
        for rel in relations {
            if !self.seen.insert((rel.a, rel.b)) {
                continue;
            }
            self.matrix.push(&self.gnfs.fb, &rel);
            self.relations.push(rel);
        }
    }

    pub fn len(&self) -> usize {
        self.relations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    /// Whether the matrix has more rows than columns after pruning. Only
    /// checked every 10% more relations.
    pub fn is_ready(&mut self) -> bool {
        if !self.ready && self.relations.len() >= self.next_check {
            self.next_check = self.relations.len() + self.relations.len() / 10;
            self.ready = self.matrix.ready(CHARACTERS, EXTRA);
        }
        self.ready
    }

    /// The dependencies among the relations, as indices into them.
    pub fn dependencies(&self) -> Vec<Vec<usize>> {
        let (kept, rows, cols) = self.matrix.solvable(&self.gnfs.fb, &self.relations);
        dependencies(&rows, cols)
            .into_iter()
            .map(|dep| dep.into_iter().map(|i| kept[i]).collect())
            .collect()
    }

    /// Takes the square roots for the dependency `dep`, returning a proper
    /// factor of `n` if it gives one.
    pub fn try_dependency(&self, dep: &[usize]) -> Option<Uint<LIMBS>> {
        let dep: Vec<&Relation> = dep.iter().map(|&i| &self.relations[i]).collect();
        sqrt::factor(&self.gnfs.n, &self.gnfs.poly, &dep)
    }
}

/// Factors `n`, an odd composite that isn't a prime power, with GNFS on
/// one thread. Returns a proper factor, or `None` if none was found.
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
    let gnfs = Gnfs::new(n, params);
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
