//! Linear algebra over GF(2).

use alloc::vec;
use alloc::vec::Vec;

use super::lanczos::null_space;

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
    if rows.len() <= DENSE_ROWS {
        return dense(rows, cols);
    }
    let (kept, pruned, cols) = prune(rows, cols);
    (0..LANCZOS_TRIES)
        .map(|seed| null_space(&pruned, cols, seed))
        .find(|deps| !deps.is_empty())
        .unwrap_or_default()
        .into_iter()
        .map(|dep| dep.into_iter().map(|i| kept[i]).collect())
        .collect()
}

/// Cancels repeated columns, then drops rows with a column that no other
/// row has, which can't be in any dependency, until there are none, and
/// renumbers the columns that are left.
///
/// Returns the indices of the rows kept, those rows, and the number of
/// columns.
fn prune(rows: &[Vec<usize>], cols: usize) -> (Vec<usize>, Vec<Vec<usize>>, usize) {
    let mut rows: Vec<Vec<usize>> = rows
        .iter()
        .map(|row| {
            let mut row = row.clone();
            row.sort_unstable();
            let mut odd: Vec<usize> = Vec::with_capacity(row.len());
            for c in row {
                if odd.last() == Some(&c) {
                    odd.pop();
                } else {
                    odd.push(c);
                }
            }
            odd
        })
        .collect();
    let mut weight = vec![0u32; cols];
    for &c in rows.iter().flatten() {
        weight[c] += 1;
    }
    let mut alive = vec![true; rows.len()];
    loop {
        let mut changed = false;
        for (r, row) in rows.iter().enumerate() {
            if alive[r] && row.iter().any(|&c| weight[c] == 1) {
                alive[r] = false;
                changed = true;
                for &c in row {
                    weight[c] -= 1;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut renumber = vec![usize::MAX; cols];
    let mut used = 0;
    for (c, &w) in weight.iter().enumerate() {
        if w > 0 {
            renumber[c] = used;
            used += 1;
        }
    }
    let kept: Vec<usize> = (0..rows.len()).filter(|&r| alive[r]).collect();
    let pruned = kept
        .iter()
        .map(|&r| {
            core::mem::take(&mut rows[r])
                .iter()
                .map(|&c| renumber[c])
                .collect()
        })
        .collect();
    (kept, pruned, used)
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
