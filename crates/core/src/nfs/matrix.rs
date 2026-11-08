//! The matrix over GF(2): which ideals each relation has an odd number of.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use super::base::{FactorBase, Prime};
use super::modp::{inv, jacobi, mul_add};
use super::sieve::Relation;
use super::store::Records;

/// Columns before the factor bases': one set in every row, so that each
/// dependency has an even number of relations, and the rational sign.
const FIXED: usize = 2;

/// The ideal `(p, a/b mod p)` that `p` stands for in `rel`'s algebraic
/// factorization, as its root, or `p` for the one at infinity.
pub fn root_of(rel: &Relation, p: u64) -> u64 {
    if rel.b.is_multiple_of(p) {
        p
    } else {
        mul_add(
            0,
            rel.a.rem_euclid(p as i64) as u64 % p,
            inv(rel.b % p, p),
            p,
        )
    }
}

/// The rows of the matrix, built as relations arrive: the fixed columns,
/// the rational primes, the algebraic ideals, then a column for each prime
/// or ideal past the factor bases, the special-q included, numbered as
/// they appear. The quadratic
/// characters, a fixed number of dense columns, are only added at the
/// end, by [`characters`], to the rows that survive filtering.
///
/// A large prime in only one relation makes a singleton column, which
/// pruning removes, so partial relations combine through the matrix
/// itself, with no cycle finding.
#[derive(Clone, Debug)]
pub struct Matrix {
    /// Each row's columns.
    rows: Records,
    cols: usize,
    large: BTreeMap<(u64, u64), u32>,
}

impl Matrix {
    pub fn new(fb: &FactorBase) -> Self {
        Self {
            rows: Records::default(),
            cols: FIXED + fb.rational.len() + fb.algebraic.len(),
            large: BTreeMap::new(),
        }
    }

    /// Adds the row for `rel`, with each column once, sorted.
    pub fn push(&mut self, fb: &FactorBase, rel: &Relation) {
        let a_base = FIXED + fb.rational.len();
        let r_max = fb.rational.last().map_or(0, |q| u64::from(q.p));
        let a_max = fb.algebraic.last().map_or(0, |q| u64::from(q.p));
        let mut row = Vec::with_capacity(rel.rational.len() + rel.algebraic.len() + 2);
        row.push(0);
        if rel.negative {
            row.push(1);
        }
        let mut column = |key| {
            *self.large.entry(key).or_insert_with(|| {
                self.cols += 1;
                (self.cols - 1) as u32
            })
        };
        for &p in &rel.rational {
            row.push(if p <= r_max {
                let i = fb.rational.partition_point(|q| u64::from(q.p) < p);
                (FIXED + i) as u32
            } else {
                column((p, 0))
            });
        }
        for &p in &rel.algebraic {
            let r = root_of(rel, p);
            row.push(if p <= a_max {
                let key = |q: &Prime| (u64::from(q.p), u64::from(q.r));
                let i = fb.algebraic.partition_point(|q| key(q) < (p, r));
                debug_assert_eq!(key(&fb.algebraic[i]), (p, r));
                (a_base + i) as u32
            } else {
                column((p, 1 + r))
            });
        }
        row.sort_unstable();
        // Columns an even number of times cancel.
        let mut odd: Vec<u32> = Vec::with_capacity(row.len());
        for c in row {
            if odd.last() == Some(&c) {
                odd.pop();
            } else {
                odd.push(c);
            }
        }
        self.rows.push(&odd);
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    fn row(&self, r: usize) -> &[u32] {
        self.rows.get(r)
    }

    /// Removes rows with a column no other row has, until none are left.
    /// Returns which rows survive, how many there are, and each column's
    /// weight among them.
    ///
    /// In passes over the rows, each removing the rows with a column of
    /// weight 1 among those still alive, rather than through an index of
    /// each column's rows, which would take as much memory again as the
    /// matrix. What is left doesn't depend on the order.
    fn prune(&self) -> (Vec<bool>, usize, Vec<u32>) {
        let mut weight = vec![0u32; self.cols];
        for &c in self.rows.words() {
            weight[c as usize] += 1;
        }
        let mut alive = vec![true; self.len()];
        let mut rows = self.len();
        loop {
            let before = rows;
            for (r, alive) in alive.iter_mut().enumerate() {
                let row = self.row(r);
                if *alive && row.iter().any(|&c| weight[c as usize] == 1) {
                    *alive = false;
                    rows -= 1;
                    for &c in row {
                        weight[c as usize] -= 1;
                    }
                }
            }
            if rows == before {
                break;
            }
        }
        (alive, rows, weight)
    }

    /// Whether, after pruning, at least `extra` more rows than columns are
    /// left, counting `dense` columns that pruning never removes.
    pub fn ready(&self, dense: usize, extra: usize) -> bool {
        let (_, rows, weight) = self.prune();
        let cols = weight.iter().filter(|&&w| w > 0).count();
        rows >= cols + dense + extra
    }

    /// The rows that survive pruning: their indices, the rows, with the
    /// columns they use renumbered from 0 in order, and the number of
    /// those columns. Takes the matrix, whose rows are let go of as they
    /// are copied.
    pub fn into_pruned(self) -> (Vec<u32>, Vec<Vec<u32>>, usize) {
        let (alive, count, weight) = self.prune();
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
        let Self {
            rows: records,
            large,
            ..
        } = self;
        drop(large);
        let mut kept = Vec::with_capacity(count);
        let mut rows = Vec::with_capacity(count);
        records.drain(|r, row| {
            if alive[r] {
                kept.push(r as u32);
                rows.push(row.iter().map(|&c| renumber[c as usize]).collect());
            }
        });
        (kept, rows, used as usize)
    }
}

/// The quadratic characters of the pair `(a, b)`, a bit for each of
/// `fb`'s that is −1: the Legendre symbol of `a − b·s` mod `q`.
pub fn characters(fb: &FactorBase, (a, b): (i64, u64)) -> u64 {
    fb.characters.iter().enumerate().fold(0, |mask, (i, c)| {
        let (q, s) = (c.q, c.s);
        let bs = i128::from(b % q) * i128::from(s);
        let v = (i128::from(a) - bs).rem_euclid(q.into()) as u64;
        mask | u64::from(jacobi(v, q) == -1) << i
    })
}
