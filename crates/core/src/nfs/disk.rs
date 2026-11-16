//! Relations kept outside memory while they are collected and filtered,
//! as CADO-NFS keeps them in files between `las`, `dup` and `purge`.
//!
//! Each relation becomes a row on arrival, written to a [`Rows`] with
//! the words a [`Store`](super::store::Store) would keep for the square
//! roots: `(a, b)`, the sign and the rational primes, then its columns.
//! In memory there are only the pairs seen, for duplicates, and the
//! column of each ideal past the factor bases, about 1 GB at RSA-140
//! against 7.5 GB for [`Relations`](super::Relations). Checking for
//! enough relations prunes singletons in passes over the rows, with a
//! count for each column; once there are enough, only the rows that
//! survive pruning are read back to be filtered and solved, and each
//! dependency's relations are read back for its square roots.
//!
//! The rows, the columns and so the matrix are the ones
//! [`Relations`](super::Relations) builds from the same relations in the
//! same order, and so are the dependencies.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use super::matrix::{base_columns, characters, columns};
use super::store::{Ideals, Seen, encode, record_ab, record_len, record_rational};
use super::{
    CHARACTERS, Dependencies, EXTRA, Gnfs, POST_ROWS, Relation, Schedule, combine, filter_rows,
    sqrt,
};
use crate::arith::Uint;
use crate::parallel::{Parallel, split};
use crate::qs::{Checkpoint, dependency_masks};

/// Somewhere outside memory, such as a file, for [`DiskRelations`]'
/// rows.
pub trait Rows {
    /// Appends a row's words. The relations can't go on without it, so
    /// this should panic rather than drop a row.
    fn append(&mut self, words: &[u32]);

    /// Calls `f` with each row's words, in the order appended, and as for
    /// [`Rows::append`], panics rather than stop early.
    fn read(&mut self, f: &mut dyn FnMut(&[u32]));
}

/// The relations found so far for one number, their rows kept in a
/// [`Rows`], which each method that needs them takes.
#[derive(Clone, Debug)]
pub struct DiskRelations<const LIMBS: usize> {
    gnfs: Gnfs<LIMBS>,
    /// The pairs `(a, b)` so far, since special-q lattices overlap.
    seen: Seen,
    /// The column of each prime or ideal past the factor bases.
    ideals: Ideals,
    cols: usize,
    len: usize,
    schedule: Schedule,
    /// Whether the matrix was solved, after which `seen` and `ideals` are
    /// gone and nothing can be added.
    solved: bool,
    /// The words of the row being added.
    words: Vec<u32>,
}

const SOLVED: &str = "the matrix was solved: relations can't be added or checked after";

impl<const LIMBS: usize> Gnfs<LIMBS> {
    /// An empty collection of relations for this number that keeps their
    /// rows in a [`Rows`] rather than in memory.
    pub fn disk_relations(&self) -> DiskRelations<LIMBS> {
        DiskRelations {
            gnfs: self.clone(),
            seen: Seen::default(),
            ideals: Ideals::default(),
            cols: base_columns(&self.fb),
            len: 0,
            schedule: Schedule::new(&self.fb),
            solved: false,
            words: Vec::new(),
        }
    }
}

impl<const LIMBS: usize> DiskRelations<LIMBS> {
    /// Adds `relations` to `rows`, skipping any already found.
    ///
    /// # Panics
    ///
    /// If the matrix was solved.
    pub fn add(&mut self, relations: impl IntoIterator<Item = Relation>, rows: &mut impl Rows) {
        assert!(!self.solved, "{SOLVED}");
        for rel in relations {
            if !self.seen.insert(rel.a, rel.b) {
                continue;
            }
            self.words.clear();
            if !encode(&rel, &mut self.words) {
                continue;
            }
            let (ideals, cols) = (&mut self.ideals, &mut self.cols);
            let row = columns(&self.gnfs.fb, &rel, |key| {
                ideals.column(key, || {
                    *cols += 1;
                    (*cols - 1) as u32
                })
            });
            self.words.extend(row);
            rows.append(&self.words);
            self.len += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the matrix has more rows than columns after pruning,
    /// checked as [`Relations::is_ready`](super::Relations::is_ready) is,
    /// each time by passes over `rows`.
    pub fn is_ready(&mut self, rows: &mut impl Rows) -> bool {
        if self.schedule.due(self.len) {
            let surplus = self.surplus(rows);
            self.schedule.record(self.len, surplus);
        }
        self.schedule.ready
    }

    /// How many more relations than needed are left after pruning:
    /// negative while too few.
    ///
    /// # Panics
    ///
    /// If the matrix was solved.
    pub fn surplus(&mut self, rows: &mut impl Rows) -> isize {
        assert!(!self.solved, "{SOLVED}");
        let (_, live, weight) = prune(rows, self.len, self.cols);
        let cols = weight.iter().filter(|&&w| w > 0).count();
        live as isize - (cols + CHARACTERS + POST_ROWS + EXTRA) as isize
    }

    /// The dependencies among the relations, from the rows that survive
    /// pruning, read back from `rows`, with block Lanczos's products
    /// split by `par` and its state kept in `checkpoint`, as for
    /// [`Relations::dependencies_resuming`](super::Relations::dependencies_resuming).
    ///
    /// # Panics
    ///
    /// If called a second time.
    pub fn dependencies(
        &mut self,
        rows: &mut impl Rows,
        par: &impl Parallel,
        checkpoint: &mut impl Checkpoint,
    ) -> Dependencies {
        assert!(!self.solved, "the matrix can only be solved once");
        self.solved = true;
        self.seen = Seen::default();
        self.ideals = Ideals::default();
        let (alive, count, weight) = prune(rows, self.len, self.cols);
        let mut renumber = weight;
        let mut used = 0;
        for w in &mut renumber {
            *w = if *w > 0 {
                used += 1;
                used - 1
            } else {
                u32::MAX
            };
        }
        let mut kept = Vec::with_capacity(count);
        let mut pruned: Vec<Vec<u32>> = Vec::with_capacity(count);
        let mut pairs = Vec::with_capacity(count);
        let mut i = 0;
        rows.read(&mut |w| {
            if alive[i] {
                kept.push(i as u32);
                pairs.push(record_ab(w));
                pruned.push(
                    w[record_len(w)..]
                        .iter()
                        .map(|&c| renumber[c as usize])
                        .collect(),
                );
            }
            i += 1;
        });
        drop((alive, renumber));
        let fb = &self.gnfs.fb;
        let pieces = split(pairs.iter().map(|_| 1), par.threads());
        let chars: Vec<u64> = par
            .map(|t| {
                pairs[pieces[t]..pieces[t + 1]]
                    .iter()
                    .map(|&ab| characters(fb, ab))
                    .collect::<Vec<_>>()
            })
            .concat();
        drop(pairs);
        let (filtered, sets, cols) = filter_rows(fb, pruned, used as usize, &chars);
        drop(chars);
        let masks = dependency_masks(filtered, cols, POST_ROWS, par, checkpoint);
        combine(&kept, &sets, &masks)
    }

    /// Takes the square roots for the dependency `dep`, reading its
    /// relations back from `rows`, with the large products split by
    /// `par`. Returns a proper factor of `n` if it gives one.
    pub fn try_dependency_with(
        &self,
        rows: &mut impl Rows,
        dep: &[usize],
        par: &impl Parallel,
    ) -> Option<Uint<LIMBS>> {
        let mut wanted = dep.to_vec();
        wanted.sort_unstable();
        let mut pairs: Vec<(i64, u64)> = Vec::with_capacity(wanted.len());
        let mut rational = BTreeMap::new();
        let (mut i, mut next) = (0, 0);
        rows.read(&mut |w| {
            if wanted.get(next) == Some(&i) {
                next += 1;
                pairs.push(record_ab(w));
                for p in record_rational(w) {
                    *rational.entry(p).or_insert(0u32) += 1;
                }
            }
            i += 1;
        });
        sqrt::factor(&self.gnfs.n, &self.gnfs.poly, &pairs, &rational, par)
    }
}

/// Removes rows with a column no other row has, in passes over the `len`
/// rows, with columns below `cols`, until none are left, as
/// [`Matrix`](super::matrix::Matrix) does in memory. Returns which rows
/// survive, how many there are, and each column's weight among them.
fn prune(rows: &mut impl Rows, len: usize, cols: usize) -> (Vec<bool>, usize, Vec<u32>) {
    let mut weight = vec![0u32; cols];
    rows.read(&mut |w| {
        for &c in &w[record_len(w)..] {
            weight[c as usize] += 1;
        }
    });
    let mut alive = vec![true; len];
    let mut live = len;
    loop {
        let before = live;
        let mut i = 0;
        rows.read(&mut |w| {
            let row = &w[record_len(w)..];
            if alive[i] && row.iter().any(|&c| weight[c as usize] == 1) {
                alive[i] = false;
                live -= 1;
                for &c in row {
                    weight[c as usize] -= 1;
                }
            }
            i += 1;
        });
        if live == before {
            break;
        }
    }
    (alive, live, weight)
}
