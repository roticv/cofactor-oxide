//! Filtering: making the matrix smaller before block Lanczos, whose time
//! grows with the number of rows times their weight.
//!
//! Pruning singletons leaves many more rows than columns, and many columns
//! in only a few rows. Clique removal drops the surplus rows, taking whole
//! groups linked by columns in two rows each, which take those columns
//! with them. Merging then eliminates light columns: each time, one of the
//! column's rows is added to the others and dropped, which takes one row
//! and one column away, until the rows get too heavy. Each row left stands
//! for the sum of a set of the original rows, a dependency among the new
//! rows gives one among the original rows by the sum of their sets, and
//! the quadratic characters, being linear, can be added up the same way.

use alloc::vec;
use alloc::vec::Vec;

/// Rounds of clique removal, each followed by pruning.
const CLIQUE_PASSES: usize = 20;

/// Merging stops once the rows average this many columns.
const DENSITY: usize = 70;

/// Columns in more rows than this are never merged.
const MAX_MERGE: u32 = 30;

/// The filtered matrix: each row's columns, sorted, and the original rows
/// whose sum it is.
pub struct Filtered {
    pub rows: Vec<Vec<u32>>,
    pub sets: Vec<Vec<u32>>,
}

/// Filters `rows`, with columns below `cols`, down to `excess` more rows
/// than columns, then merges.
pub fn filter(rows: Vec<Vec<u32>>, cols: usize, excess: usize) -> Filtered {
    let mut m = State::new(rows, cols);
    m.prune();
    m.remove_cliques(excess);
    m.merge();
    m.prune();
    let (rows, sets) = m
        .rows
        .into_iter()
        .zip(m.sets)
        .zip(&m.alive)
        .filter(|(_, alive)| **alive)
        .map(|(row_set, _)| row_set)
        .unzip();
    Filtered { rows, sets }
}

struct State {
    rows: Vec<Vec<u32>>,
    sets: Vec<Vec<u32>>,
    alive: Vec<bool>,
    /// Each column's number of live rows.
    weight: Vec<u32>,
    /// For each column, rows that have it, and possibly others that once
    /// did: [`State::rows_of`] checks.
    by_col: Vec<Vec<u32>>,
    live_rows: usize,
    live_cols: usize,
    /// The total weight of the live rows.
    total: usize,
}

impl State {
    fn new(rows: Vec<Vec<u32>>, cols: usize) -> Self {
        let mut weight = vec![0u32; cols];
        let mut by_col = vec![Vec::new(); cols];
        for (r, row) in rows.iter().enumerate() {
            for &c in row {
                weight[c as usize] += 1;
                by_col[c as usize].push(r as u32);
            }
        }
        Self {
            sets: (0..rows.len() as u32).map(|r| vec![r]).collect(),
            alive: vec![true; rows.len()],
            live_rows: rows.len(),
            live_cols: weight.iter().filter(|&&w| w > 0).count(),
            total: rows.iter().map(Vec::len).sum(),
            rows,
            weight,
            by_col,
        }
    }

    /// The live rows with column `c`, which also become its list.
    fn rows_of(&mut self, c: u32) -> Vec<u32> {
        let (rows, alive) = (&self.rows, &self.alive);
        let list = &mut self.by_col[c as usize];
        list.retain(|&r| alive[r as usize] && rows[r as usize].binary_search(&c).is_ok());
        list.sort_unstable();
        list.dedup();
        list.clone()
    }

    /// Drops row `r`, returning the columns left with one row.
    fn kill(&mut self, r: u32, singles: &mut Vec<u32>) {
        self.alive[r as usize] = false;
        self.live_rows -= 1;
        self.total -= self.rows[r as usize].len();
        for &c in &self.rows[r as usize] {
            let w = &mut self.weight[c as usize];
            *w -= 1;
            match *w {
                0 => self.live_cols -= 1,
                1 => singles.push(c),
                _ => {}
            }
        }
    }

    /// Drops rows with a column no other row has, until there are none.
    fn prune(&mut self) {
        let mut singles: Vec<u32> = (0..self.weight.len() as u32)
            .filter(|&c| self.weight[c as usize] == 1)
            .collect();
        while let Some(c) = singles.pop() {
            if self.weight[c as usize] != 1 {
                continue;
            }
            let r = self.rows_of(c)[0];
            self.kill(r, &mut singles);
        }
    }

    fn excess(&self) -> usize {
        self.live_rows.saturating_sub(self.live_cols)
    }

    /// Drops the largest cliques, the groups of rows linked by columns in
    /// two rows each, until only `target` more rows than columns are left.
    /// Dropping a clique of `k` rows drops at least its `k − 1` linking
    /// columns, so each costs at most one row of excess.
    fn remove_cliques(&mut self, target: usize) {
        for _ in 0..CLIQUE_PASSES {
            let excess = self.excess();
            if excess <= target {
                return;
            }
            let n = self.rows.len();
            let mut parent: Vec<u32> = (0..n as u32).collect();
            let find = |parent: &mut [u32], mut x: u32| {
                while parent[x as usize] != x {
                    let up = parent[parent[x as usize] as usize];
                    parent[x as usize] = up;
                    x = up;
                }
                x
            };
            for c in 0..self.weight.len() as u32 {
                if self.weight[c as usize] == 2 {
                    let rs = self.rows_of(c);
                    let (a, b) = (find(&mut parent, rs[0]), find(&mut parent, rs[1]));
                    parent[a as usize] = b;
                }
            }
            let mut size = vec![0u32; n];
            for r in 0..n as u32 {
                if self.alive[r as usize] {
                    let root = find(&mut parent, r);
                    size[root as usize] += 1;
                }
            }
            let mut roots: Vec<u32> = (0..n as u32).filter(|&r| size[r as usize] > 0).collect();
            roots.sort_unstable_by_key(|&r| core::cmp::Reverse(size[r as usize]));
            let mut doomed = vec![false; n];
            for &r in roots.iter().take(excess - target) {
                doomed[r as usize] = true;
            }
            let mut singles = Vec::new();
            for r in 0..n as u32 {
                if self.alive[r as usize] && doomed[find(&mut parent, r) as usize] {
                    self.kill(r, &mut singles);
                }
            }
            self.prune();
        }
    }

    /// Eliminates columns in up to [`MAX_MERGE`] rows, lightest first,
    /// until the rows average [`DENSITY`] columns.
    fn merge(&mut self) {
        for w in 2..=MAX_MERGE {
            let light: Vec<u32> = (0..self.weight.len() as u32)
                .filter(|&c| (2..=w).contains(&self.weight[c as usize]))
                .collect();
            for c in light {
                if self.total > DENSITY * self.live_rows {
                    return;
                }
                if (2..=w).contains(&self.weight[c as usize]) {
                    self.merge_col(c);
                }
            }
        }
    }

    /// Adds the lightest row with column `c` to its other rows, and drops
    /// it.
    fn merge_col(&mut self, c: u32) {
        let rs = self.rows_of(c);
        let pivot = *rs
            .iter()
            .min_by_key(|&&r| self.rows[r as usize].len())
            .expect("a column being merged has rows");
        let prow = core::mem::take(&mut self.rows[pivot as usize]);
        let pset = core::mem::take(&mut self.sets[pivot as usize]);
        for &r in rs.iter().filter(|&&r| r != pivot) {
            let row = core::mem::take(&mut self.rows[r as usize]);
            self.total -= row.len();
            let sum = sym_diff(&row, &prow);
            // Columns of the pivot row that r lacked are new to it; the
            // others cancel.
            for &k in &prow {
                if row.binary_search(&k).is_ok() {
                    self.weight[k as usize] -= 1;
                    if self.weight[k as usize] == 0 {
                        self.live_cols -= 1;
                    }
                } else {
                    self.weight[k as usize] += 1;
                    self.by_col[k as usize].push(r);
                }
            }
            self.total += sum.len();
            self.rows[r as usize] = sum;
            let set = core::mem::take(&mut self.sets[r as usize]);
            self.sets[r as usize] = sym_diff(&set, &pset);
        }
        self.rows[pivot as usize] = prow;
        self.sets[pivot as usize] = pset;
        let mut singles = Vec::new();
        self.kill(pivot, &mut singles);
    }
}

/// The entries in exactly one of the sorted lists `a` and `b`, sorted.
fn sym_diff(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            core::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}
