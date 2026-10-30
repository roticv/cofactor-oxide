//! Benchmarks for the modular arithmetic at each integer width.
//!
//! Run with `cargo bench -p cofox-core`, or with
//! `RUSTFLAGS="--cfg cofox_limb32"` for the 32-bit limbs used on wasm32.

use std::hint::black_box;

use cofox_core::arith::Limb;
use cofox_core::{Monty, Uint};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// Deterministic pseudo-random limbs (xorshift64*).
fn random<const L: usize>(seed: &mut u64) -> Uint<L> {
    let mut limbs = [0; L];
    for limb in &mut limbs {
        *seed ^= *seed >> 12;
        *seed ^= *seed << 25;
        *seed ^= *seed >> 27;
        *limb = seed.wrapping_mul(0x2545_F491_4F6C_DD1D) as Limb;
    }
    Uint::from_limbs(limbs)
}

/// A random odd modulus using the full width, and two reduced operands.
fn setup<const L: usize>() -> (Monty<L>, Uint<L>, Uint<L>) {
    let mut seed = 0x9E37_79B9_7F4A_7C15;
    let mut limbs = *random::<L>(&mut seed).as_limbs();
    limbs[0] |= 1;
    limbs[L - 1] |= 1 << (Limb::BITS - 1);
    let monty = Monty::new(Uint::from_limbs(limbs)).unwrap();
    let a = monty.to_monty(&random(&mut seed));
    let b = monty.to_monty(&random(&mut seed));
    (monty, a, b)
}

fn bench_width<const L: usize>(c: &mut Criterion) {
    let (monty, a, b) = setup::<L>();
    let bits = Uint::<L>::BITS;
    let id = |name: &str| BenchmarkId::new(name, format!("{bits} bits"));

    let mut group = c.benchmark_group("monty");
    group.bench_function(id("mul"), |bench| {
        bench.iter(|| monty.mul(black_box(&a), black_box(&b)))
    });
    group.bench_function(id("square"), |bench| {
        bench.iter(|| monty.square(black_box(&a)))
    });
    group.bench_function(id("square via mul"), |bench| {
        bench.iter(|| monty.mul(black_box(&a), black_box(&a)))
    });
    group.bench_function(id("square via widening_square + reduce"), |bench| {
        bench.iter(|| {
            let (lo, hi) = black_box(&a).widening_square();
            monty.reduce(&lo, &hi)
        })
    });
    group.bench_function(id("add"), |bench| {
        bench.iter(|| monty.add(black_box(&a), black_box(&b)))
    });
    group.bench_function(id("inv"), |bench| bench.iter(|| monty.inv(black_box(&a))));
    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_width::<1>(c);
    bench_width::<2>(c);
    bench_width::<4>(c);
    bench_width::<8>(c);
    bench_width::<16>(c);
}

criterion_group!(arith, benches);
criterion_main!(arith);
