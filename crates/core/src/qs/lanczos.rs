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

/// `acc += v·x` for a block vector `v`.
fn add_mul(acc: &mut [u64], v: &[u64], x: &Small) {
    let t = tables(x);
    for (a, &w) in acc.iter_mut().zip(v) {
        for (b, t) in t.iter().enumerate() {
            *a ^= t[(w >> (8 * b)) as usize & 0xff];
        }
    }
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

/// The sparse matrix `B`: for each column (relation), its set rows.
struct Sparse<'a> {
    cols: &'a [Vec<usize>],
    rows: usize,
}

impl Sparse<'_> {
    /// `Bv`, with one word per row of `B`.
    fn mul(&self, v: &[u64]) -> Vec<u64> {
        let mut out = vec![0; self.rows];
        for (col, &w) in self.cols.iter().zip(v) {
            for &r in col {
                out[r] ^= w;
            }
        }
        out
    }

    /// `Bᵀw`, with one word per column of `B`.
    fn mul_t(&self, w: &[u64]) -> Vec<u64> {
        self.cols
            .iter()
            .map(|col| col.iter().fold(0, |acc, &r| acc ^ w[r]))
            .collect()
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
/// `B` with `rows` rows and the given columns. Returns nothing if this
/// `seed` fails, which another seed is unlikely to repeat.
pub fn null_space(cols: &[Vec<usize>], rows: usize, seed: u64) -> Vec<Vec<usize>> {
    let n = cols.len();
    let b = Sparse { cols, rows };
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
        let vav = inner(&v, &av);
        if vav == [0; 64] {
            break;
        }
        steps += 1;
        if steps > max_steps {
            return Vec::new();
        }
        let va2v = inner(&av, &av);
        let (winv, s) = choose(&vav, s1);
        if s == 0 {
            return Vec::new();
        }
        // X += V·W⁻¹·VᵀV₀.
        add_mul(&mut x, &v, &mul(&winv, &inner(&v, &v0)));

        // V' = AV·SSᵀ + V·D + V₁·E + V₂·F, which makes V' A-orthogonal to
        // V, V₁ and V₂, and so to every earlier block.
        let d = add(&IDENTITY, &mul(&winv, &add(&keep_cols(&va2v, s), &vav)));
        let e = mul(&winv1, &keep_cols(&vav, s));
        let f = mul(
            &mul(&winv2, &add(&IDENTITY, &mul(&vav1, &winv1))),
            &keep_cols(&add(&keep_cols(&va2v1, s1), &vav1), s),
        );
        let mut next: Vec<u64> = av.iter().map(|&w| w & s).collect();
        add_mul(&mut next, &v, &d);
        add_mul(&mut next, &v1, &e);
        add_mul(&mut next, &v2, &f);

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
    let (bx, bv) = (b.mul(&x), b.mul(&v));
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
    debug_assert!(b.mul(&deps).iter().all(|&w| w == 0), "Bx ≠ 0");
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
