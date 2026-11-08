//! Linear algebra over GF(2).

use alloc::vec;
use alloc::vec::Vec;

use super::lanczos::null_space;
use crate::parallel::{Parallel, Serial};

/// Matrices with more rows than this go to block Lanczos.
const DENSE_ROWS: usize = 1000;

/// Seeds for block Lanczos to try before giving up.
const LANCZOS_TRIES: u64 = 4;

/// Finds subsets of `rows` that sum to zero over GF(2), each as a list of
/// row indices. Each row is given by its set columns, all below `cols`; a
/// column listed twice cancels.
///
/// Small matrices use dense Gaussian elimination, which finds every
/// dependency: at least `rows.len() − cols` of them. Larger ones are pruned
/// and passed to block Lanczos, which finds up to 64.
pub fn dependencies(rows: &[Vec<usize>], cols: usize) -> Vec<Vec<usize>> {
    dependencies_with(rows, cols, &Serial)
}

/// [`dependencies`], with block Lanczos's products split by `par`.
pub fn dependencies_with(rows: &[Vec<usize>], cols: usize, par: &impl Parallel) -> Vec<Vec<usize>> {
    if rows.len() <= DENSE_ROWS {
        return dense(rows, cols);
    }
    let rows = rows
        .iter()
        .map(|row| row.iter().map(|&c| c as u32).collect())
        .collect();
    lists(&dependency_masks(rows, cols, 0, par))
}

/// [`dependencies_with`] as words: for each row, a word whose bit `q` says
/// whether the row is in the `q`th dependency, of up to 64, which takes
/// far less memory than lists when the dependencies have millions of rows
/// each. Block Lanczos leaves the `post` densest columns out of its
/// iterations, to be satisfied at the end, so `rows` should have at least
/// `post + 64` more rows than columns after pruning. The rows are taken so
/// that each copy can be let go of as soon as the next is made.
pub(crate) fn dependency_masks(
    rows: Vec<Vec<u32>>,
    cols: usize,
    post: usize,
    par: &impl Parallel,
) -> Vec<u64> {
    let n = rows.len();
    if n <= DENSE_ROWS {
        let rows: Vec<Vec<usize>> = rows
            .iter()
            .map(|row| row.iter().map(|&c| c as usize).collect())
            .collect();
        let mut masks = vec![0u64; n];
        for (q, dep) in dense(&rows, cols).iter().take(64).enumerate() {
            for &r in dep {
                masks[r] |= 1 << q;
            }
        }
        return masks;
    }
    let (kept, pruned, cols) = prune(rows, cols);
    let found = null_space(pruned, cols, post, LANCZOS_TRIES, par);
    let mut masks = vec![0u64; n];
    for (&r, &m) in kept.iter().zip(&found) {
        masks[r as usize] = m;
    }
    masks
}

/// The dependencies in `masks`, from [`dependency_masks`], as lists of
/// rows: the distinct, nonempty ones.
fn lists(masks: &[u64]) -> Vec<Vec<usize>> {
    let mut deps: Vec<Vec<usize>> = (0..64)
        .map(|q| {
            (0..masks.len())
                .filter(|&r| masks[r] >> q & 1 == 1)
                .collect::<Vec<_>>()
        })
        .filter(|dep| !dep.is_empty())
        .collect();
    deps.sort_unstable();
    deps.dedup();
    deps
}

/// Cancels repeated columns, then drops rows with a column that no other
/// row has, which can't be in any dependency, until there are none, and
/// renumbers the columns that are left, all in place.
///
/// Returns the indices of the rows kept, those rows, and the number of
/// columns.
fn prune(mut rows: Vec<Vec<u32>>, cols: usize) -> (Vec<u32>, Vec<Vec<u32>>, usize) {
    for row in &mut rows {
        row.sort_unstable();
        let (mut i, mut kept) = (0, 0);
        while i < row.len() {
            if i + 1 < row.len() && row[i] == row[i + 1] {
                i += 2;
            } else {
                row[kept] = row[i];
                kept += 1;
                i += 1;
            }
        }
        row.truncate(kept);
    }
    let mut weight = vec![0u32; cols];
    for &c in rows.iter().flatten() {
        weight[c as usize] += 1;
    }
    let mut alive = vec![true; rows.len()];
    loop {
        let mut changed = false;
        for (r, row) in rows.iter().enumerate() {
            if alive[r] && row.iter().any(|&c| weight[c as usize] == 1) {
                alive[r] = false;
                changed = true;
                for &c in row {
                    weight[c as usize] -= 1;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut renumber = vec![u32::MAX; cols];
    let mut used = 0;
    for (c, &w) in weight.iter().enumerate() {
        if w > 0 {
            renumber[c] = used;
            used += 1;
        }
    }
    let mut kept = Vec::new();
    let mut pruned = Vec::new();
    for (r, mut row) in rows.into_iter().enumerate() {
        if alive[r] {
            for c in &mut row {
                *c = renumber[*c as usize];
            }
            kept.push(r as u32);
            pruned.push(row);
        }
    }
    (kept, pruned, used as usize)
}

/// Dense Gaussian elimination, with each row carrying the set of original
/// rows it is the sum of. Rows that end up zero without being used as a
/// pivot are the dependencies.
fn dense(rows: &[Vec<usize>], cols: usize) -> Vec<Vec<usize>> {
    let n = rows.len();
    // Each row: `cols` bits of matrix, then `n` bits of history.
    let words = (cols + n).div_ceil(64);
    let mut m = vec![0u64; n * words];
    let flip = |m: &mut [u64], r: usize, c: usize| m[r * words + c / 64] ^= 1 << (c % 64);
    let bit = |m: &[u64], r: usize, c: usize| m[r * words + c / 64] >> (c % 64) & 1 == 1;
    for (r, row) in rows.iter().enumerate() {
        for &c in row {
            debug_assert!(c < cols);
            flip(&mut m, r, c);
        }
        flip(&mut m, r, cols + r);
    }

    let mut pivot = vec![false; n];
    for c in 0..cols {
        let Some(p) = (0..n).find(|&r| !pivot[r] && bit(&m, r, c)) else {
            continue;
        };
        pivot[p] = true;
        // Earlier columns are already zero in every row but their pivots.
        let from = c / 64;
        let (lo, hi) = (p * words + from, (p + 1) * words);
        let src: Vec<u64> = m[lo..hi].to_vec();
        for r in 0..n {
            if r != p && bit(&m, r, c) {
                let dst = &mut m[r * words + from..(r + 1) * words];
                for (d, s) in dst.iter_mut().zip(&src) {
                    *d ^= s;
                }
            }
        }
    }

    (0..n)
        .filter(|&r| !pivot[r])
        .map(|r| {
            debug_assert!((0..cols).all(|c| !bit(&m, r, c)));
            (0..n).filter(|&i| bit(&m, r, cols + i)).collect()
        })
        .collect()
}
