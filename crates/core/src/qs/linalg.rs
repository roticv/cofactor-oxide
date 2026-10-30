//! Linear algebra over GF(2).

use alloc::vec;
use alloc::vec::Vec;

/// Finds subsets of `rows` that sum to zero over GF(2), each as a list of
/// row indices. Each row is given by its set columns, all below `cols`; a
/// column listed twice cancels.
///
/// Uses Gaussian elimination on a dense matrix, with each row carrying the
/// set of original rows it is the sum of. Rows that end up zero without
/// being used as a pivot are the dependencies, of which there are at least
/// `rows.len() − cols`.
pub fn dependencies(rows: &[Vec<usize>], cols: usize) -> Vec<Vec<usize>> {
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
