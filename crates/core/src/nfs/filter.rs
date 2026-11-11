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

/// Columns in up to this many rows are merged along a minimum spanning
/// tree; past that, the k² sums it weighs cost too much.
const MST_ROWS: usize = 10;

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
    m.index();
    m.merge();
    m.prune();
    let (rows, sets) = m
        .rows
        .into_iter()
        .zip(m.sets)
        .enumerate()
        .filter(|&(r, _)| m.alive[r])
        .map(|(r, (row, set))| {
            let set = if set.is_empty() { vec![r as u32] } else { set };
            (row, set)
        })
        .unzip();
    Filtered { rows, sets }
}

struct State {
    rows: Vec<Vec<u32>>,
    /// The original rows each row is the sum of, sorted; empty for just
    /// itself, as most rows stay, which saves an allocation a row.
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
        for &c in rows.iter().flatten() {
            weight[c as usize] += 1;
        }
        let mut m = Self {
            sets: vec![Vec::new(); rows.len()],
            alive: vec![true; rows.len()],
            live_rows: rows.len(),
            live_cols: weight.iter().filter(|&&w| w > 0).count(),
            total: rows.iter().map(Vec::len).sum(),
            rows,
            weight,
            by_col: vec![Vec::new(); cols],
        };
        m.index();
        m
    }

    /// Rebuilds the lists of each column's rows from the live rows alone,
    /// each list its exact size: merging only ever adds to them, and at
    /// RSA-140 they had grown from 2.6 to 4.2 GB.
    fn index(&mut self) {
        for (list, &w) in self.by_col.iter_mut().zip(&self.weight) {
            list.clear();
            list.shrink_to(w as usize);
            list.reserve_exact(w as usize);
        }
        for (r, row) in self.rows.iter().enumerate() {
            if self.alive[r] {
                for &c in row {
                    self.by_col[c as usize].push(r as u32);
                }
            }
        }
    }

    /// The sum of the original rows of rows `r` and `p`.
    fn sum_sets(&self, r: usize, p: usize) -> Vec<u32> {
        let (own_r, own_p) = ([r as u32], [p as u32]);
        let a: &[u32] = if self.sets[r].is_empty() {
            &own_r
        } else {
            &self.sets[r]
        };
        let b: &[u32] = if self.sets[p].is_empty() {
            &own_p
        } else {
            &self.sets[p]
        };
        sym_diff(a, b)
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
        // Most rows die: their memory goes with them.
        let row = core::mem::take(&mut self.rows[r as usize]);
        self.sets[r as usize] = Vec::new();
        self.total -= row.len();
        for &c in &row {
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
            // The lists only grow as rows are summed: rebuilt once they
            // are twice what the live rows need.
            let listed: usize = self.by_col.iter().map(Vec::len).sum();
            if listed > 2 * self.total {
                self.index();
            }
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

    /// Eliminates column `c`: of its rows, all but one are replaced by
    /// their sums with a neighbour in a tree joining them, and the last is
    /// dropped. Up to [`MST_ROWS`] rows, the tree is the one whose sums
    /// are lightest, as in msieve; beyond, every row is added to the
    /// lightest.
    fn merge_col(&mut self, c: u32) {
        let rs = self.rows_of(c);
        let k = rs.len();
        let root = (0..k)
            .min_by_key(|&i| self.rows[rs[i] as usize].len())
            .expect("a column being merged has rows");
        // parent[i]: the row that rs[i] is added to; root has none.
        let mut parent = vec![root; k];
        if k <= MST_ROWS {
            // Prim's algorithm from the root, on the weights of the sums.
            let weight = |i: usize, j: usize| {
                sym_diff_len(&self.rows[rs[i] as usize], &self.rows[rs[j] as usize])
            };
            let mut in_tree = vec![false; k];
            in_tree[root] = true;
            let mut best: Vec<usize> = (0..k).map(|i| weight(i, root)).collect();
            for _ in 1..k {
                let i = (0..k)
                    .filter(|&i| !in_tree[i])
                    .min_by_key(|&i| best[i])
                    .unwrap();
                in_tree[i] = true;
                for j in (0..k).filter(|&j| !in_tree[j]) {
                    let w = weight(i, j);
                    if w < best[j] {
                        (best[j], parent[j]) = (w, i);
                    }
                }
            }
        }
        let sums: Vec<(Vec<u32>, Vec<u32>)> = (0..k)
            .map(|i| {
                if i == root {
                    return (Vec::new(), Vec::new());
                }
                let (r, p) = (rs[i] as usize, rs[parent[i]] as usize);
                (sym_diff(&self.rows[r], &self.rows[p]), self.sum_sets(r, p))
            })
            .collect();
        let mut pending = Vec::with_capacity(k);
        for (i, (row, set)) in sums.into_iter().enumerate() {
            if i == root {
                continue;
            }
            let (r, p) = (rs[i], rs[parent[i]] as usize);
            // The parent's columns that r had cancel; the others are new
            // to it.
            for &col in &self.rows[p] {
                let w = &mut self.weight[col as usize];
                if self.rows[r as usize].binary_search(&col).is_ok() {
                    *w -= 1;
                    if *w == 0 {
                        self.live_cols -= 1;
                    }
                } else {
                    *w += 1;
                    if *w == 1 {
                        self.live_cols += 1;
                    }
                    self.by_col[col as usize].push(r);
                }
            }
            self.total = self.total - self.rows[r as usize].len() + row.len();
            pending.push((r, row, set));
        }
        for (r, row, set) in pending {
            self.rows[r as usize] = row;
            self.sets[r as usize] = set;
        }
        let mut singles = Vec::new();
        self.kill(rs[root], &mut singles);
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
    // Room for both was set aside, and rows are kept for the rest of the
    // run: at RSA-120 the slack took 0.3 GB.
    out.shrink_to_fit();
    out
}

/// The number of entries in exactly one of the sorted lists `a` and `b`.
fn sym_diff_len(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut common) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                common += 1;
                i += 1;
                j += 1;
            }
        }
    }
    a.len() + b.len() - 2 * common
}
