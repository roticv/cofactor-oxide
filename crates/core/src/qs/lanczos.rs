//! Montgomery's block Lanczos algorithm over GF(2).
//!
//! The relations are the columns of a sparse matrix `B`, one row per prime,
//! and a dependency is a vector `x` with `Bx = 0`. Block Lanczos works with
//! the symmetric `A = BᵀB` and 64 vectors at a time, packed one bit each
//! into the `u64` words of a *block vector* (one word per relation). From
//! `V₀ = AY` for a random `Y`, it builds blocks `V₀, V₁, …` that are
//! `A`-orthogonal to each other, each step needing only the last three, and
//! accumulates the `X` with `AX = AY`. Then `A(X − Y) = 0`, which, with the
//! last block, gives vectors in the null space of `B` after a small
//! elimination.
//!
//! See P. L. Montgomery, "A Block Lanczos Algorithm for Finding
//! Dependencies over GF(2)", EUROCRYPT '95.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use crate::parallel::{Parallel, map_dynamic, split};

/// The sparse part of `B` is stored in blocks this many rows and columns
/// on a side, so that a block's slices of the vectors read and written
/// stay in the L1 cache. Measured at 90 digits, 2048 beat 1024 and 4096.
const BLOCK: usize = 2048;

/// Work is split into this many pieces per thread, which take the next
/// piece left as they finish, so that faster cores do more of them.
const PIECES_PER_THREAD: usize = 8;

/// A 64×64 matrix over GF(2), one row per word.
type Small = [u64; 64];

const IDENTITY: Small = {
    let mut m = [0; 64];
    let mut i = 0;
    while i < 64 {
        m[i] = 1 << i;
        i += 1;
    }
    m
};

/// `xy`.
fn mul(x: &Small, y: &Small) -> Small {
    let mut out = [0; 64];
    for (o, &row) in out.iter_mut().zip(x) {
        let mut bits = row;
        while bits != 0 {
            *o ^= y[bits.trailing_zeros() as usize];
            bits &= bits - 1;
        }
    }
    out
}

fn add(x: &Small, y: &Small) -> Small {
    core::array::from_fn(|i| x[i] ^ y[i])
}

/// `x` with only the columns in `mask` kept: `x·SSᵀ`.
fn keep_cols(x: &Small, mask: u64) -> Small {
    x.map(|row| row & mask)
}

/// Tables for multiplying many words by `x`: entry `[b][v]` is the sum of
/// the rows of `x` picked by the bits of byte value `v` at byte `b`.
fn tables(x: &Small) -> Vec<[u64; 256]> {
    (0..8)
        .map(|b| {
            let mut t = [0u64; 256];
            for v in 1..256 {
                let low = v & (v - 1);
                t[v] = t[low] ^ x[8 * b + (v ^ low).trailing_zeros() as usize];
            }
            t
        })
        .collect()
}

/// The word `w` times the matrix of `tables`.
fn mul_word(t: &[[u64; 256]], w: u64) -> u64 {
    t.iter()
        .enumerate()
        .fold(0, |acc, (b, t)| acc ^ t[(w >> (8 * b)) as usize & 0xff])
}

/// `vᵀw` for block vectors `v` and `w`.
fn inner(v: &[u64], w: &[u64]) -> Small {
    // Sum w into a bucket per byte value at each byte position of v, then
    // read off each bit's row as the sum of the buckets with that bit.
    let mut buckets = vec![[0u64; 256]; 8];
    for (&a, &b) in v.iter().zip(w) {
        for (i, bucket) in buckets.iter_mut().enumerate() {
            bucket[(a >> (8 * i)) as usize & 0xff] ^= b;
        }
    }
    let mut out = [0; 64];
    for (i, bucket) in buckets.iter().enumerate() {
        for bit in 0..8 {
            out[8 * i + bit] = (0..256)
                .filter(|v| v >> bit & 1 == 1)
                .fold(0, |acc, v| acc ^ bucket[v]);
        }
    }
    out
}

/// The matrix `B`: its heaviest rows packed, a bit per column, and the
/// rest sparse, by columns (relations) and by rows, so that both `Bv` and
/// `Bᵀw` gather. Each is split into pieces of about equal weight.
struct Sparse<'a, P> {
    rows: usize,
    cols: usize,
    /// The sparse entries in blocks of [`BLOCK`] rows by [`BLOCK`]
    /// columns, block row by block row, each entry its row and column
    /// within its block as `row << 16 | column`. Each product reads and
    /// writes within one block's range at a time, which stays in cache.
    entries: Vec<u32>,
    /// Where each block's entries start, and where the last ends.
    starts: Vec<usize>,
    /// The dense rows, by bit.
    dense: Vec<usize>,
    /// Each column's bits in the dense rows.
    masks: Vec<u64>,
    /// The rows left out of `A` until the end, by bit, and each column's
    /// bits in them.
    post: Vec<usize>,
    post_masks: Vec<u64>,
    /// Equal pieces of a block vector, one word per column.
    vec_pieces: Vec<usize>,
    par: &'a P,
}

impl<'a, P: Parallel> Sparse<'a, P> {
    fn new(cols: &[Vec<usize>], rows: usize, post: usize, par: &'a P) -> Self {
        let mut weight = vec![0usize; rows];
        for &r in cols.iter().flatten() {
            weight[r] += 1;
        }
        let mut order: Vec<usize> = (0..rows).collect();
        order.sort_unstable_by_key(|&r| core::cmp::Reverse(weight[r]));
        order.truncate(post + 64);
        let dense = order.split_off(post.min(order.len()));
        let post = order;
        // Bits 0–63 for the dense rows, and 64 on for the post rows.
        let mut bit = vec![u8::MAX; rows];
        for (i, &r) in dense.iter().chain(&post).enumerate() {
            bit[r] = i as u8;
        }
        let n = cols.len();
        let mut masks = vec![0u64; n];
        let mut post_masks = vec![0u64; n];
        let (row_blocks, col_blocks) = (rows.div_ceil(BLOCK), n.div_ceil(BLOCK));
        let block = |r: usize, c: usize| r / BLOCK * col_blocks + c / BLOCK;
        let mut starts = vec![0usize; row_blocks * col_blocks + 1];
        for (c, col) in cols.iter().enumerate() {
            for &r in col {
                match bit[r] {
                    u8::MAX => starts[block(r, c) + 1] += 1,
                    b @ 0..64 => masks[c] |= 1 << b,
                    b => post_masks[c] |= 1 << (b - 64),
                }
            }
        }
        for i in 1..starts.len() {
            starts[i] += starts[i - 1];
        }
        let mut fill = starts.clone();
        let mut entries = vec![0u32; starts[starts.len() - 1]];
        for (c, col) in cols.iter().enumerate() {
            for &r in col.iter().filter(|&&r| bit[r] == u8::MAX) {
                let k = block(r, c);
                entries[fill[k]] = (((r % BLOCK) << 16) | (c % BLOCK)) as u32;
                fill[k] += 1;
            }
        }
        let pieces = PIECES_PER_THREAD * par.threads();
        Self {
            rows,
            cols: n,
            entries,
            starts,
            vec_pieces: split(cols.iter().map(|_| 1), pieces),
            dense,
            masks,
            post,
            post_masks,
            par,
        }
    }

    /// `f` over the pieces `bounds` of `0..n`, shared out among the
    /// threads as they become free, giving each piece's results.
    fn pieces<T: Send>(&self, bounds: &[usize], f: impl Fn(Range<usize>) -> T + Sync) -> Vec<T> {
        map_dynamic(self.par, bounds.len() - 1, |i| f(bounds[i]..bounds[i + 1]))
    }

    /// `f` over the pieces `bounds` of `0..n`, each giving the words of
    /// its range, joined.
    fn gather(&self, bounds: &[usize], f: impl Fn(Range<usize>) -> Vec<u64> + Sync) -> Vec<u64> {
        self.pieces(bounds, f).concat()
    }

    /// `f` at each word of a block vector, split across the threads.
    fn each(&self, f: impl Fn(usize) -> u64 + Sync) -> Vec<u64> {
        self.gather(&self.vec_pieces, |range| range.map(&f).collect())
    }

    /// `vᵀw`, split across the threads.
    fn inner(&self, v: &[u64], w: &[u64]) -> Small {
        self.pieces(&self.vec_pieces, |r| inner(&v[r.clone()], &w[r]))
            .iter()
            .fold([0; 64], |acc, x| add(&acc, x))
    }

    /// The entries of the block at block row `i`, block column `j`.
    fn block(&self, i: usize, j: usize) -> &[u32] {
        let k = i * self.cols.div_ceil(BLOCK) + j;
        &self.entries[self.starts[k]..self.starts[k + 1]]
    }

    /// `Bv`, with one word per row of `B`: each block row's words from its
    /// blocks in turn.
    fn mul(&self, v: &[u64]) -> Vec<u64> {
        let col_blocks = self.cols.div_ceil(BLOCK);
        let mut out = map_dynamic(self.par, self.rows.div_ceil(BLOCK), |i| {
            let mut out = vec![0u64; BLOCK.min(self.rows - i * BLOCK)];
            for j in 0..col_blocks {
                let v = &v[j * BLOCK..];
                for &e in self.block(i, j) {
                    out[(e >> 16) as usize] ^= v[(e & 0xffff) as usize];
                }
            }
            out
        })
        .concat();
        // Dense row i's word is the sum of v over the columns with bit i.
        for (&r, w) in self.dense.iter().zip(self.inner(&self.masks, v)) {
            out[r] = w;
        }
        out
    }

    /// `Bv` with the post rows too: the whole of `B`.
    fn mul_full(&self, v: &[u64]) -> Vec<u64> {
        let mut out = self.mul(v);
        for (&r, w) in self.post.iter().zip(self.inner(&self.post_masks, v)) {
            out[r] = w;
        }
        out
    }

    /// `Bᵀw`, with one word per column of `B`.
    fn mul_t(&self, w: &[u64]) -> Vec<u64> {
        let mut rows = [0u64; 64];
        for (i, &r) in self.dense.iter().enumerate() {
            rows[i] = w[r];
        }
        let t = tables(&rows);
        let row_blocks = self.rows.div_ceil(BLOCK);
        map_dynamic(self.par, self.cols.div_ceil(BLOCK), |j| {
            let range = j * BLOCK..self.cols.min((j + 1) * BLOCK);
            let mut out: Vec<u64> = self.masks[range].iter().map(|&m| mul_word(&t, m)).collect();
            for i in 0..row_blocks {
                let w = &w[i * BLOCK..];
                for &e in self.block(i, j) {
                    out[(e & 0xffff) as usize] ^= w[(e >> 16) as usize];
                }
            }
            out
        })
        .concat()
    }

    /// `Av = BᵀBv`.
    fn mul_a(&self, v: &[u64]) -> Vec<u64> {
        self.mul_t(&self.mul(v))
    }
}

/// Montgomery's choice of the columns `S` of `V` to use this step and of
/// `W⁻¹ = S(SᵀTS)⁻¹Sᵀ`, given `T = VᵀAV` and the previous step's `S`.
///
/// Elimination on `[T | I]` picks a set of columns on which `T` is
/// invertible, as large as possible, preferring columns left out last
/// time: every column has to be used within two steps for the blocks to
/// stay orthogonal.
fn choose(t: &Small, prev: u64) -> (Small, u64) {
    let mut m: [u128; 64] = core::array::from_fn(|i| u128::from(t[i]) | 1 << (64 + i));
    let mut order: Vec<usize> = (0..64).filter(|&i| prev >> i & 1 == 0).collect();
    order.extend((0..64).filter(|&i| prev >> i & 1 == 1));
    let mut s = 0u64;
    for j in 0..64 {
        let c = order[j];
        if let Some(k) = (j..64).find(|&k| m[order[k]] >> c & 1 == 1) {
            m.swap(order[k], c);
        }
        if m[c] >> c & 1 == 1 {
            s |= 1 << c;
            for r in 0..64 {
                if r != c && m[r] >> c & 1 == 1 {
                    m[r] ^= m[c];
                }
            }
        } else {
            let k = (j..64)
                .find(|&k| m[order[k]] >> (64 + c) & 1 == 1)
                .expect("the right half stays invertible");
            m.swap(order[k], c);
            for r in 0..64 {
                if r != c && m[r] >> (64 + c) & 1 == 1 {
                    m[r] ^= m[c];
                }
            }
            m[c] = 0;
        }
    }
    let winv: Small = core::array::from_fn(|r| {
        if s >> r & 1 == 1 {
            (m[r] >> 64) as u64 & s
        } else {
            0
        }
    });
    debug_assert!(
        {
            let wt = mul(&winv, t);
            (0..64)
                .filter(|&r| s >> r & 1 == 1)
                .all(|r| wt[r] & s == 1 << r)
        },
        "W⁻¹ isn't the inverse of T on S"
    );
    (winv, s)
}

/// Finds up to 64 vectors `x ≠ 0` with `Bx = 0`, as lists of columns, for
/// `B` with `rows` rows and the given columns, using `par` for the
/// matrix and vector products. Returns nothing if this `seed` fails, which
/// another seed is unlikely to repeat.
///
/// The `post` densest rows of `B` are left out of the iterations, which
/// makes each cheaper, and the vectors found are combined at the end to
/// satisfy them as well, as in msieve. That leaves fewer dependencies, so
/// `B` should have at least `post + 64` more columns than rows.
pub fn null_space(
    cols: &[Vec<usize>],
    rows: usize,
    seed: u64,
    post: usize,
    par: &impl Parallel,
) -> Vec<Vec<usize>> {
    let n = cols.len();
    let b = Sparse::new(cols, rows, post, par);
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let y: Vec<u64> = (0..n)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        })
        .collect();
    let v0 = b.mul_a(&y);

    // V for this step and the two before, and the previous steps' W⁻¹,
    // VᵀAV, VᵀA²V and S.
    let mut v = v0.clone();
    let (mut v1, mut v2) = (vec![0u64; n], vec![0u64; n]);
    let (mut winv1, mut winv2) = ([0; 64], [0; 64]);
    let (mut vav1, mut va2v1) = ([0; 64], [0; 64]);
    let mut s1 = u64::MAX;
    let mut x = vec![0u64; n];
    // Each step uses about 63 of the 64 dimensions on average.
    let max_steps = n / 60 + 20;
    let mut steps = 0;
    loop {
        let av = b.mul_a(&v);
        let vav = b.inner(&v, &av);
        if vav == [0; 64] {
            break;
        }
        steps += 1;
        if steps > max_steps {
            return Vec::new();
        }
        let va2v = b.inner(&av, &av);
        let (winv, s) = choose(&vav, s1);
        if s == 0 {
            return Vec::new();
        }
        // X += V·W⁻¹·VᵀV₀.
        let t = tables(&mul(&winv, &b.inner(&v, &v0)));
        x = b.each(|i| x[i] ^ mul_word(&t, v[i]));

        // V' = AV·SSᵀ + V·D + V₁·E + V₂·F, which makes V' A-orthogonal to
        // V, V₁ and V₂, and so to every earlier block.
        let d = add(&IDENTITY, &mul(&winv, &add(&keep_cols(&va2v, s), &vav)));
        let e = mul(&winv1, &keep_cols(&vav, s));
        let f = mul(
            &mul(&winv2, &add(&IDENTITY, &mul(&vav1, &winv1))),
            &keep_cols(&add(&keep_cols(&va2v1, s1), &vav1), s),
        );
        let (d, e, f) = (tables(&d), tables(&e), tables(&f));
        let next =
            b.each(|i| av[i] & s ^ mul_word(&d, v[i]) ^ mul_word(&e, v1[i]) ^ mul_word(&f, v2[i]));

        v2 = core::mem::replace(&mut v1, core::mem::replace(&mut v, next));
        winv2 = core::mem::replace(&mut winv1, winv);
        vav1 = vav;
        va2v1 = va2v;
        s1 = s;
    }

    // A(X − Y) = 0 now, give or take the last V, and B maps the 128
    // vectors of [X − Y | V] into a space of small dimension. Find the
    // combinations that B maps to 0.
    for (xi, yi) in x.iter_mut().zip(&y) {
        *xi ^= yi;
    }
    let (bx, bv) = (b.mul_full(&x), b.mul_full(&v));
    let words = rows.div_ceil(64);
    // Each of the 128 images as a bit string over the rows of B, reduced
    // against the earlier ones, with the combination it stands for.
    let mut pivots: Vec<(usize, Vec<u64>, u128)> = Vec::new();
    let mut combos: Vec<u128> = Vec::new();
    for lane in 0..128 {
        let (src, bit) = if lane < 64 {
            (&bx, lane)
        } else {
            (&bv, lane - 64)
        };
        let mut image = vec![0u64; words];
        for (r, &w) in src.iter().enumerate() {
            image[r / 64] |= (w >> bit & 1) << (r % 64);
        }
        let mut combo = 1u128 << lane;
        for (col, pivot, pcombo) in &pivots {
            if image[col / 64] >> (col % 64) & 1 == 1 {
                for (a, b) in image.iter_mut().zip(pivot) {
                    *a ^= b;
                }
                combo ^= pcombo;
            }
        }
        match image.iter().position(|&w| w != 0) {
            Some(i) => {
                let col = 64 * i + image[i].trailing_zeros() as usize;
                pivots.push((col, image, combo));
            }
            None => combos.push(combo),
        }
    }

    // Apply them, up to 64 at once, and keep the nonzero results.
    combos.truncate(64);
    let deps: Vec<u64> = x
        .iter()
        .zip(&v)
        .map(|(&xw, &vw)| {
            let row = u128::from(xw) | u128::from(vw) << 64;
            combos.iter().enumerate().fold(0, |acc, (q, &c)| {
                acc | u64::from((row & c).count_ones() & 1) << q
            })
        })
        .collect();
    debug_assert!(b.mul_full(&deps).iter().all(|&w| w == 0), "Bx ≠ 0");
    let mut found: Vec<Vec<usize>> = (0..combos.len())
        .map(|q| {
            (0..n)
                .filter(|&i| deps[i] >> q & 1 == 1)
                .collect::<Vec<_>>()
        })
        .filter(|dep| !dep.is_empty())
        .collect();
    found.sort_unstable();
    found.dedup();
    found
}
