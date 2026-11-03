//! The matrix over GF(2): which ideals each relation has an odd number of.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use super::base::{FactorBase, Prime};
use super::sieve::Relation;
use crate::qs::modp::pow_mod;

/// Columns before the factor bases': one set in every row, so that each
/// dependency has an even number of relations, and the rational sign.
const FIXED: usize = 2;

/// The ideal `(p, a/b mod p)` that `p` stands for in `rel`'s algebraic
/// factorization, as its root, or `p` for the one at infinity.
pub fn root_of(rel: &Relation, p: u64) -> u64 {
    if rel.b.is_multiple_of(p) {
        p
    } else {
        let b_inv = pow_mod(rel.b % p, p - 2, p);
        (rel.a.rem_euclid(p as i64) as u64 % p) * b_inv % p
    }
}

/// The rows of the matrix, built as relations arrive: the fixed columns,
/// the rational primes, the algebraic ideals, then a column for each prime
/// or ideal past the factor bases, the special-q included, numbered as
/// they appear. The quadratic
/// characters, a fixed number of dense columns, are only added at the
/// end, by [`Matrix::solvable`], to the rows that survive pruning.
///
/// A large prime in only one relation makes a singleton column, which
/// pruning removes, so partial relations combine through the matrix
/// itself, with no cycle finding.
#[derive(Clone, Debug)]
pub struct Matrix {
    rows: Vec<Vec<u32>>,
    cols: usize,
    large: BTreeMap<(u64, u64), u32>,
}

impl Matrix {
    pub fn new(fb: &FactorBase) -> Self {
        Self {
            rows: Vec::new(),
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
        let mut odd = Vec::with_capacity(row.len());
        for c in row {
            if odd.last() == Some(&c) {
                odd.pop();
            } else {
                odd.push(c);
            }
        }
        self.rows.push(odd);
    }

    /// Removes rows with a column no other row has, until none are left.
    /// Returns which rows survive, how many there are, and how many
    /// columns they use.
    fn prune(&self) -> (Vec<bool>, usize, usize) {
        // Each column's rows, and its weight among the rows still alive.
        let mut weight = vec![0u32; self.cols];
        for &c in self.rows.iter().flatten() {
            weight[c as usize] += 1;
        }
        let mut start = vec![0usize; self.cols + 1];
        for (c, &w) in weight.iter().enumerate() {
            start[c + 1] = start[c] + w as usize;
        }
        let mut fill = start.clone();
        let mut by_col = vec![0u32; start[self.cols]];
        for (r, row) in self.rows.iter().enumerate() {
            for &c in row {
                by_col[fill[c as usize]] = r as u32;
                fill[c as usize] += 1;
            }
        }
        let mut alive = vec![true; self.rows.len()];
        let mut singles: Vec<usize> = (0..self.cols).filter(|&c| weight[c] == 1).collect();
        let mut rows = self.rows.len();
        let mut cols = weight.iter().filter(|&&w| w > 0).count();
        while let Some(c) = singles.pop() {
            let Some(&r) = by_col[start[c]..start[c + 1]]
                .iter()
                .find(|&&r| alive[r as usize])
            else {
                continue;
            };
            alive[r as usize] = false;
            rows -= 1;
            for &c in &self.rows[r as usize] {
                let c = c as usize;
                weight[c] -= 1;
                match weight[c] {
                    0 => cols -= 1,
                    1 => singles.push(c),
                    _ => {}
                }
            }
        }
        (alive, rows, cols)
    }

    /// Whether, after pruning, at least `extra` more rows than columns are
    /// left, counting `dense` columns that pruning never removes.
    pub fn ready(&self, dense: usize, extra: usize) -> bool {
        let (_, rows, cols) = self.prune();
        rows >= cols + dense + extra
    }

    /// The rows that survive pruning, with the quadratic characters of
    /// `relations`, the relations the rows were built from, in the last
    /// columns. Returns the indices of the rows kept, the rows, and the
    /// number of columns.
    pub fn solvable(
        &self,
        fb: &FactorBase,
        relations: &[Relation],
    ) -> (Vec<usize>, Vec<Vec<usize>>, usize) {
        let (alive, _, _) = self.prune();
        let kept: Vec<usize> = (0..self.rows.len()).filter(|&r| alive[r]).collect();
        let c_base = self.cols;
        let rows = kept
            .iter()
            .map(|&r| {
                let rel = &relations[r];
                let mut row: Vec<usize> = self.rows[r].iter().map(|&c| c as usize).collect();
                for (i, c) in fb.characters.iter().enumerate() {
                    // The Legendre symbol of a − b·s mod q.
                    let q = u64::from(c.p);
                    let bs = i128::from(rel.b % q) * i128::from(c.r);
                    let v = (i128::from(rel.a) - bs).rem_euclid(q.into()) as u64;
                    if pow_mod(v, (q - 1) / 2, q) == q - 1 {
                        row.push(c_base + i);
                    }
                }
                row
            })
            .collect();
        (kept, rows, c_base + fb.characters.len())
    }
}
