//! The matrix over GF(2): which ideals each relation has an odd number of.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::base::FactorBase;
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

/// Builds the rows for `relations`: the fixed columns, the rational
/// primes, the algebraic ideals, the quadratic characters, then a column
/// for each large prime or large ideal that appears. Returns the rows and
/// the number of columns.
///
/// A large prime in only one relation makes a singleton column, which
/// pruning removes, so partial relations combine through the matrix
/// itself, with no cycle finding.
pub fn rows(fb: &FactorBase, relations: &[Relation]) -> (Vec<Vec<usize>>, usize) {
    let r_base = FIXED;
    let a_base = r_base + fb.rational.len();
    let c_base = a_base + fb.algebraic.len();
    let mut cols = c_base + fb.characters.len();
    let mut large: BTreeMap<(u64, u64), usize> = BTreeMap::new();
    let mut column = |key: (u64, u64)| {
        *large.entry(key).or_insert_with(|| {
            cols += 1;
            cols - 1
        })
    };
    let rows = relations
        .iter()
        .map(|rel| {
            let mut row = Vec::with_capacity(rel.rational.len() + rel.algebraic.len() + 8);
            row.push(0);
            if rel.negative {
                row.push(1);
            }
            for &p in &rel.rational {
                let i = fb.rational.partition_point(|q| q.p < p);
                row.push(r_base + i);
            }
            for &p in &rel.algebraic {
                let r = root_of(rel, p.into()) as u32;
                let i = fb.algebraic.partition_point(|q| (q.p, q.r) < (p, r));
                debug_assert_eq!((fb.algebraic[i].p, fb.algebraic[i].r), (p, r));
                row.push(a_base + i);
            }
            for (i, c) in fb.characters.iter().enumerate() {
                // The Legendre symbol of a − b·s mod q.
                let q = u64::from(c.p);
                let bs = i128::from(rel.b % q) * i128::from(c.r);
                let v = (i128::from(rel.a) - bs).rem_euclid(q.into()) as u64;
                if pow_mod(v, (q - 1) / 2, q) == q - 1 {
                    row.push(c_base + i);
                }
            }
            if rel.large[0] > 1 {
                row.push(column((rel.large[0], 0)));
            }
            if rel.large[1] > 1 {
                let q = rel.large[1];
                row.push(column((q, 1 + root_of(rel, q))));
            }
            row
        })
        .collect();
    (rows, cols)
}
