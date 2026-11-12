//! Montgomery's block Lanczos algorithm over GF(2).
//!
//! The relations are the columns of a sparse matrix `B`, one row per prime,
//! and a dependency is a vector `x` with `Bx = 0`. Block Lanczos works with
//! the symmetric `A = BᵀB` and 64 or 128 vectors at a time, packed one bit
//! each into the [`Word`]s of a *block vector* (one word per relation). From
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

/// A block of vectors' bits at one index: one bit per vector.
pub(crate) trait Word:
    Copy + Send + Sync + Eq + core::ops::BitXor<Output = Self> + core::ops::BitAnd<Output = Self>
{
    const BITS: usize;
    const ZERO: Self;
    const ONES: Self;
    /// Bit `i` alone.
    fn unit(i: usize) -> Self;
    fn bit(self, i: usize) -> bool;
    /// Byte `b`, counting from the low end.
    fn byte(self, b: usize) -> usize;
    fn count_ones(self) -> u32;
    fn trailing_zeros(self) -> u32;
    /// A word from 64-bit pieces drawn from `next`.
    fn draw(next: impl FnMut() -> u64) -> Self;
}

impl Word for u64 {
    const BITS: usize = 64;
    const ZERO: Self = 0;
    const ONES: Self = u64::MAX;
    fn unit(i: usize) -> Self {
        1 << i
    }
    fn bit(self, i: usize) -> bool {
        self >> i & 1 == 1
    }
    fn byte(self, b: usize) -> usize {
        (self >> (8 * b)) as usize & 0xff
    }
    fn count_ones(self) -> u32 {
        u64::count_ones(self)
    }
    fn trailing_zeros(self) -> u32 {
        u64::trailing_zeros(self)
    }
    fn draw(mut next: impl FnMut() -> u64) -> Self {
        next()
    }
}

/// 128 bits in a SIMD register, where a xor or an and is one instruction
/// rather than the two it is on a `u128`. In the sparse products, an entry
/// took 0.45 ns with these on an M4, as with `u64`, but 0.62 ns with `u128`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct Wide(simd::V);

#[cfg(target_arch = "aarch64")]
mod simd {
    pub(super) use core::arch::aarch64::{uint64x2_t as V, vandq_u64 as and, veorq_u64 as xor};
}

#[cfg(target_arch = "x86_64")]
mod simd {
    pub(super) use core::arch::x86_64::{__m128i as V, _mm_and_si128 as and, _mm_xor_si128 as xor};
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl Wide {
    const fn new(x: u128) -> Self {
        // SAFETY: both are 16 bytes of plain bits.
        Self(unsafe { core::mem::transmute::<u128, simd::V>(x) })
    }

    fn get(self) -> u128 {
        // SAFETY: as in `new`.
        unsafe { core::mem::transmute::<simd::V, u128>(self.0) }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl core::ops::BitXor for Wide {
    type Output = Self;
    fn bitxor(self, other: Self) -> Self {
        // SAFETY: the instruction is in the target's baseline, NEON or SSE2.
        Self(unsafe { simd::xor(self.0, other.0) })
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl core::ops::BitAnd for Wide {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        // SAFETY: the instruction is in the target's baseline, NEON or SSE2.
        Self(unsafe { simd::and(self.0, other.0) })
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl PartialEq for Wide {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl Eq for Wide {}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl Word for Wide {
    const BITS: usize = 128;
    const ZERO: Self = Self::new(0);
    const ONES: Self = Self::new(u128::MAX);
    fn unit(i: usize) -> Self {
        Self::new(1 << i)
    }
    fn bit(self, i: usize) -> bool {
        self.get() >> i & 1 == 1
    }
    fn byte(self, b: usize) -> usize {
        (self.get() >> (8 * b)) as usize & 0xff
    }
    fn count_ones(self) -> u32 {
        self.get().count_ones()
    }
    fn trailing_zeros(self) -> u32 {
        self.get().trailing_zeros()
    }
    fn draw(mut next: impl FnMut() -> u64) -> Self {
        Self::new(u128::from(next()) | u128::from(next()) << 64)
    }
}

/// A `W::BITS`-square matrix over GF(2), one row per word.
type Small<W> = Vec<W>;

fn identity<W: Word>() -> Small<W> {
    (0..W::BITS).map(W::unit).collect()
}

/// `xy`.
fn mul<W: Word>(x: &[W], y: &[W]) -> Small<W> {
    x.iter()
        .map(|&row| {
            let (mut bits, mut o) = (row, W::ZERO);
            while bits != W::ZERO {
                let i = bits.trailing_zeros() as usize;
                o = o ^ y[i];
                bits = bits ^ W::unit(i);
            }
            o
        })
        .collect()
}

fn add<W: Word>(x: &[W], y: &[W]) -> Small<W> {
    x.iter().zip(y).map(|(&a, &b)| a ^ b).collect()
}

/// `x` with only the columns in `mask` kept: `x·SSᵀ`.
fn keep_cols<W: Word>(x: &[W], mask: W) -> Small<W> {
    x.iter().map(|&row| row & mask).collect()
}

/// Tables for multiplying many words by `x`, of `x.len()` rows: entry
/// `[b][v]` is the sum of the rows of `x` picked by the bits of byte value
/// `v` at byte `b`.
fn tables<W: Word>(x: &[W]) -> Vec<[W; 256]> {
    x.chunks(8)
        .map(|rows| {
            let mut t = [W::ZERO; 256];
            for v in 1..256 {
                let low = v & (v - 1);
                t[v] = t[low] ^ rows[(v ^ low).trailing_zeros() as usize];
            }
            t
        })
        .collect()
}

/// The word `w` times the matrix of `tables`.
fn mul_word<A: Word, W: Word>(t: &[[W; 256]], w: A) -> W {
    t.iter()
        .enumerate()
        .fold(W::ZERO, |acc, (b, t)| acc ^ t[w.byte(b)])
}

/// `vᵀw` for block vectors `v` and `w`, a row of `w`'s words for each bit
/// of `v`'s.
fn inner<A: Word, W: Word>(v: &[A], w: &[W]) -> Small<W> {
    // Sum w into a bucket per byte value at each byte position of v, then
    // read off each bit's row as the sum of the buckets with that bit.
    let mut buckets = vec![[W::ZERO; 256]; A::BITS / 8];
    for (&a, &b) in v.iter().zip(w) {
        for (i, bucket) in buckets.iter_mut().enumerate() {
            let k = a.byte(i);
            bucket[k] = bucket[k] ^ b;
        }
    }
    let mut out = vec![W::ZERO; A::BITS];
    for (i, bucket) in buckets.iter().enumerate() {
        for bit in 0..8 {
            out[8 * i + bit] = (0..256)
                .filter(|v| v >> bit & 1 == 1)
                .fold(W::ZERO, |acc, v| acc ^ bucket[v]);
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
    ///
    /// Within a block the entries go round-robin over its columns: each
    /// column's first, then each one's second, and so on. The first block
    /// rows, the small primes', have several entries in many columns, and
    /// in column order `Bᵀw` would add them into the same word one after
    /// another, each waiting on the last; at RSA-110 that made it twice as
    /// slow as `Bv`. Round-robin takes a quarter to a third of that off.
    entries: Vec<u32>,
    /// Where each block's entries start, and where the last ends.
    starts: Vec<usize>,
    /// For `Bᵀw`, if there is memory to spare: the entries again, block
    /// column by block column, each block's sorted by row, so that it reads
    /// `w` in order and adds to different words in turn, as fast as `Bv`.
    /// At RSA-110 it takes 240 MB, too much beside the rest in the
    /// browser's 4 GB.
    by_column: Option<(Vec<u32>, Vec<usize>)>,
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
    fn new(cols: &[Vec<u32>], rows: usize, post: usize, by_column: bool, par: &'a P) -> Self {
        let mut weight = vec![0usize; rows];
        for &r in cols.iter().flatten() {
            weight[r as usize] += 1;
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
            for r in col.iter().map(|&r| r as usize) {
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
            for r in col
                .iter()
                .map(|&r| r as usize)
                .filter(|&r| bit[r] == u8::MAX)
            {
                let k = block(r, c);
                entries[fill[k]] = (((r % BLOCK) << 16) | (c % BLOCK)) as u32;
                fill[k] += 1;
            }
        }
        // Each block's entries are in column order; the rank of each in its
        // column's run, then the column, puts them round-robin.
        let mut keyed = Vec::new();
        for k in 0..starts.len() - 1 {
            let block = &mut entries[starts[k]..starts[k + 1]];
            keyed.clear();
            let mut rank = 0u32;
            for (n, &e) in block.iter().enumerate() {
                rank = if n > 0 && block[n - 1] & 0xffff == e & 0xffff {
                    rank + 1
                } else {
                    0
                };
                keyed.push((rank, e));
            }
            keyed.sort_unstable_by_key(|&(rank, e)| (rank, e & 0xffff));
            for (e, &(_, k)) in block.iter_mut().zip(&keyed) {
                *e = k;
            }
        }
        let by_column = by_column.then(|| {
            let mut t = Vec::with_capacity(entries.len());
            let mut t_starts = Vec::with_capacity(starts.len());
            t_starts.push(0);
            for j in 0..col_blocks {
                for i in 0..row_blocks {
                    let k = i * col_blocks + j;
                    let from = t.len();
                    t.extend_from_slice(&entries[starts[k]..starts[k + 1]]);
                    t[from..].sort_unstable();
                    t_starts.push(t.len());
                }
            }
            (t, t_starts)
        });
        let pieces = PIECES_PER_THREAD * par.threads();
        Self {
            rows,
            cols: n,
            entries,
            starts,
            by_column,
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
    fn gather<W: Word>(
        &self,
        bounds: &[usize],
        f: impl Fn(Range<usize>) -> Vec<W> + Sync,
    ) -> Vec<W> {
        self.pieces(bounds, f).concat()
    }

    /// `f` at each word of a block vector, split across the threads.
    fn each<W: Word>(&self, f: impl Fn(usize) -> W + Sync) -> Vec<W> {
        self.gather(&self.vec_pieces, |range| range.map(&f).collect())
    }

    /// `vᵀw`, split across the threads.
    fn inner<A: Word, W: Word>(&self, v: &[A], w: &[W]) -> Small<W> {
        self.pieces(&self.vec_pieces, |r| inner(&v[r.clone()], &w[r]))
            .iter()
            .fold(vec![W::ZERO; A::BITS], |acc, x| add(&acc, x))
    }

    /// The entries of the block at block row `i`, block column `j`.
    fn block(&self, i: usize, j: usize) -> &[u32] {
        let k = i * self.cols.div_ceil(BLOCK) + j;
        &self.entries[self.starts[k]..self.starts[k + 1]]
    }

    /// `Bv`, with one word per row of `B`: each block row's words from its
    /// blocks in turn.
    fn mul<W: Word>(&self, v: &[W]) -> Vec<W> {
        let col_blocks = self.cols.div_ceil(BLOCK);
        let mut out = map_dynamic(self.par, self.rows.div_ceil(BLOCK), |i| {
            let mut out = vec![W::ZERO; BLOCK.min(self.rows - i * BLOCK)];
            for j in 0..col_blocks {
                let v = &v[j * BLOCK..];
                for &e in self.block(i, j) {
                    let o = &mut out[(e >> 16) as usize];
                    *o = *o ^ v[(e & 0xffff) as usize];
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
    fn mul_full<W: Word>(&self, v: &[W]) -> Vec<W> {
        let mut out = self.mul(v);
        for (&r, w) in self.post.iter().zip(self.inner(&self.post_masks, v)) {
            out[r] = w;
        }
        out
    }

    /// `Bᵀw`, with one word per column of `B`.
    fn mul_t<W: Word>(&self, w: &[W]) -> Vec<W> {
        let mut rows = [W::ZERO; 64];
        for (i, &r) in self.dense.iter().enumerate() {
            rows[i] = w[r];
        }
        let t = tables(&rows);
        let row_blocks = self.rows.div_ceil(BLOCK);
        map_dynamic(self.par, self.cols.div_ceil(BLOCK), |j| {
            let range = j * BLOCK..self.cols.min((j + 1) * BLOCK);
            let mut out: Vec<W> = self.masks[range].iter().map(|&m| mul_word(&t, m)).collect();
            for i in 0..row_blocks {
                let w = &w[i * BLOCK..];
                let block = match &self.by_column {
                    Some((t, starts)) => {
                        &t[starts[j * row_blocks + i]..starts[j * row_blocks + i + 1]]
                    }
                    None => self.block(i, j),
                };
                for &e in block {
                    let o = &mut out[(e & 0xffff) as usize];
                    *o = *o ^ w[(e >> 16) as usize];
                }
            }
            out
        })
        .concat()
    }

    /// `Av = BᵀBv`.
    fn mul_a<W: Word>(&self, v: &[W]) -> Vec<W> {
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
fn choose<W: Word>(t: &[W], prev: W) -> (Small<W>, W) {
    let n = W::BITS;
    let mut m: Vec<(W, W)> = (0..n).map(|i| (t[i], W::unit(i))).collect();
    let xor = |a: (W, W), b: (W, W)| (a.0 ^ b.0, a.1 ^ b.1);
    let mut order: Vec<usize> = (0..n).filter(|&i| !prev.bit(i)).collect();
    order.extend((0..n).filter(|&i| prev.bit(i)));
    let mut s = W::ZERO;
    for j in 0..n {
        let c = order[j];
        if let Some(k) = (j..n).find(|&k| m[order[k]].0.bit(c)) {
            m.swap(order[k], c);
        }
        if m[c].0.bit(c) {
            s = s ^ W::unit(c);
            for r in 0..n {
                if r != c && m[r].0.bit(c) {
                    m[r] = xor(m[r], m[c]);
                }
            }
        } else {
            let k = (j..n)
                .find(|&k| m[order[k]].1.bit(c))
                .expect("the right half stays invertible");
            m.swap(order[k], c);
            for r in 0..n {
                if r != c && m[r].1.bit(c) {
                    m[r] = xor(m[r], m[c]);
                }
            }
            m[c] = (W::ZERO, W::ZERO);
        }
    }
    let winv: Small<W> = (0..n)
        .map(|r| if s.bit(r) { m[r].1 & s } else { W::ZERO })
        .collect();
    debug_assert!(
        {
            let wt = mul(&winv, t);
            (0..n)
                .filter(|&r| s.bit(r))
                .all(|r| wt[r] & s == W::unit(r))
        },
        "W⁻¹ isn't the inverse of T on S"
    );
    (winv, s)
}

/// Matrices with at least this many columns are solved 128 vectors at a
/// time, where [`Wide`] is available, rather than 64. That halves the
/// steps, and a step's sparse products cost about the same, but its
/// vector work twice as much. Run on one thread, a 100,000-column
/// matrix's solve took 5.6 s rather than 6.8, and 150,000 columns' 13.6 s
/// rather than 18.1; at 10,000 columns 128 was 10% slower.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const WIDE_COLS: usize = 20_000;

/// Finds up to 64 vectors `x ≠ 0` with `Bx = 0`, for `B` with `rows` rows
/// and the given columns, using `par` for the matrix and vector products,
/// and trying up to `tries` random starts. Returns a word for each column
/// whose bit `q` says whether the column is in the `q`th vector: all zero
/// if every start failed. The columns are taken rather than borrowed so
/// that they can be let go of once packed.
///
/// The `post` densest rows of `B` are left out of the iterations, which
/// makes each cheaper, and the vectors found are combined at the end to
/// satisfy them as well, as in msieve. That leaves fewer dependencies, so
/// `B` should have at least `post + 64` more columns than rows.
pub fn null_space(
    cols: Vec<Vec<u32>>,
    rows: usize,
    post: usize,
    tries: u64,
    par: &impl Parallel,
) -> Vec<u64> {
    let n = cols.len();
    // The second copy of the entries where memory isn't as tight as the
    // browser's. The columns themselves aren't needed once the entries
    // are packed.
    let b = Sparse::new(&cols, rows, post, !cfg!(target_arch = "wasm32"), par);
    drop(cols);
    (0..tries)
        .find_map(|seed| {
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            if n >= WIDE_COLS {
                return attempt::<Wide, _>(&b, rows, seed);
            }
            attempt::<u64, _>(&b, rows, seed)
        })
        .unwrap_or_else(|| vec![0; n])
}

/// One run of block Lanczos on `b`, `W::BITS` vectors at a time, from the
/// random start `seed`: for each column, a word whose bit `q` says whether
/// it is in the `q`th vector found, or `None` if this seed fails or finds
/// nothing.
fn attempt<W: Word, P: Parallel>(b: &Sparse<'_, P>, rows: usize, seed: u64) -> Option<Vec<u64>> {
    let n = b.cols;
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    // xorshift*: the multiply makes successive draws nonlinear in each
    // other over GF(2), else a wide word's halves would be one linear map
    // of each other, and Y, so V, would have only 64 independent vectors.
    let y: Vec<W> = (0..n)
        .map(|_| {
            W::draw(|| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng.wrapping_mul(0x2545_F491_4F6C_DD1D)
            })
        })
        .collect();
    let v0 = b.mul_a(&y);

    // V for this step and the two before, and the previous steps' W⁻¹,
    // VᵀAV, VᵀA²V and S.
    let zero = || vec![W::ZERO; W::BITS];
    let mut v = v0.clone();
    let (mut v1, mut v2) = (vec![W::ZERO; n], vec![W::ZERO; n]);
    let (mut winv1, mut winv2) = (zero(), zero());
    let (mut vav1, mut va2v1) = (zero(), zero());
    let mut s1 = W::ONES;
    let mut x = vec![W::ZERO; n];
    // Each step uses all but about one of the dimensions on average.
    let max_steps = n / (W::BITS - 4) + 20;
    let mut steps = 0;
    loop {
        let av = b.mul_a(&v);
        let vav = b.inner(&v, &av);
        if vav.iter().all(|&w| w == W::ZERO) {
            break;
        }
        steps += 1;
        if steps > max_steps {
            return None;
        }
        let va2v = b.inner(&av, &av);
        let (winv, s) = choose(&vav, s1);
        if s == W::ZERO {
            return None;
        }
        // X += V·W⁻¹·VᵀV₀.
        let t = tables(&mul(&winv, &b.inner(&v, &v0)));
        x = b.each(|i| x[i] ^ mul_word(&t, v[i]));

        // V' = AV·SSᵀ + V·D + V₁·E + V₂·F, which makes V' A-orthogonal to
        // V, V₁ and V₂, and so to every earlier block.
        let d = add(&identity(), &mul(&winv, &add(&keep_cols(&va2v, s), &vav)));
        let e = mul(&winv1, &keep_cols(&vav, s));
        let f = mul(
            &mul(&winv2, &add(&identity(), &mul(&vav1, &winv1))),
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

    // A(X − Y) = 0 now, give or take the last V, and B maps the vectors of
    // [X − Y | V] into a space of small dimension. Find the combinations
    // that B maps to 0.
    for (xi, &yi) in x.iter_mut().zip(&y) {
        *xi = *xi ^ yi;
    }
    let (bx, bv) = (b.mul_full(&x), b.mul_full(&v));
    let words = rows.div_ceil(64);
    // Each of the images as a bit string over the rows of B, reduced
    // against the earlier ones, with the combination it stands for.
    let mut pivots: Vec<(usize, Vec<u64>, (W, W))> = Vec::new();
    let mut combos: Vec<(W, W)> = Vec::new();
    for lane in 0..2 * W::BITS {
        let (src, bit, mut combo) = if lane < W::BITS {
            (&bx, lane, (W::unit(lane), W::ZERO))
        } else {
            (&bv, lane - W::BITS, (W::ZERO, W::unit(lane - W::BITS)))
        };
        let mut image = vec![0u64; words];
        for (r, &w) in src.iter().enumerate() {
            image[r / 64] |= u64::from(w.bit(bit)) << (r % 64);
        }
        for (col, pivot, pcombo) in &pivots {
            if image[col / 64] >> (col % 64) & 1 == 1 {
                for (a, b) in image.iter_mut().zip(pivot) {
                    *a ^= b;
                }
                combo = (combo.0 ^ pcombo.0, combo.1 ^ pcombo.1);
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
            combos.iter().enumerate().fold(0, |acc, (q, &(cx, cv))| {
                let parity = ((xw & cx).count_ones() + (vw & cv).count_ones()) & 1;
                acc | u64::from(parity) << q
            })
        })
        .collect();
    debug_assert!(b.mul_full(&deps).iter().all(|&w| w == 0), "Bx ≠ 0");
    deps.iter().any(|&w| w != 0).then_some(deps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::Serial;

    /// `Bv` and `Bᵀw` with and without the second copy, against the sums
    /// taken entry by entry, on a matrix of several blocks a side whose
    /// first rows are heavy, as the small primes' are.
    #[test]
    fn products_match() {
        let (rows, n) = (5000, 4500);
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let cols: Vec<Vec<u32>> = (0..n)
            .map(|_| {
                let mut col: Vec<u32> = (0..20).map(|_| (next() % 300) as u32).collect();
                col.extend((0..10).map(|_| (next() % rows as u64) as u32));
                col.sort_unstable();
                col.dedup();
                col
            })
            .collect();
        let v: Vec<u64> = (0..n).map(|_| next()).collect();
        let w: Vec<u64> = (0..rows).map(|_| next()).collect();
        let mut bv = vec![0u64; rows];
        let mut btw = vec![0u64; n];
        for (c, col) in cols.iter().enumerate() {
            for r in col.iter().map(|&r| r as usize) {
                bv[r] ^= v[c];
                btw[c] ^= w[r];
            }
        }
        for by_column in [false, true] {
            let b = Sparse::new(&cols, rows, 0, by_column, &Serial);
            assert_eq!(b.mul_full(&v), bv, "Bv, copy {by_column}");
            assert_eq!(b.mul_t(&w), btw, "Bᵀw, copy {by_column}");
            // The same with 128-bit words, whose high halves are the low.
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            {
                let wide = |x: &[u64]| -> Vec<Wide> {
                    x.iter()
                        .map(|&w| Wide::new(u128::from(w) * (1 | 1 << 64)))
                        .collect()
                };
                let bits = |x: Vec<Wide>| -> Vec<u128> { x.iter().map(|w| w.get()).collect() };
                assert_eq!(
                    bits(b.mul_full(&wide(&v))),
                    bits(wide(&bv)),
                    "wide Bv, copy {by_column}"
                );
                assert_eq!(
                    bits(b.mul_t(&wide(&w))),
                    bits(wide(&btw)),
                    "wide Bᵀw, copy {by_column}"
                );
            }
        }
    }
}
