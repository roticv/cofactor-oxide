//! ECM stage 1 on Metal against cofox-core on every CPU core: curves per
//! second at B1 = 50,000 (the 25-digit level), for 256- and 384-bit n, and
//! how the GPU's rate depends on how many curves run at once.
//!
//! The GPU uses the same Montgomery-curve formulas and ladder as
//! cofox-core's `Curve`, and the same R = 2^bits, so every coordinate must
//! match the CPU's exactly.

use cofox_core::curve::{Curve, Point};
use cofox_core::primes::Primes;
use cofox_core::{Monty, Uint};
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
#define N %N%

struct Num { uint w[N]; };

inline Num mont_mul(thread const Num& a, thread const Num& b, constant uint* n, uint ninv) {
    uint t[N + 2];
    for (int i = 0; i < N + 2; i++) t[i] = 0;
    for (int i = 0; i < N; i++) {
        ulong c = 0;
        for (int j = 0; j < N; j++) {
            c += (ulong)t[j] + (ulong)a.w[j] * b.w[i];
            t[j] = (uint)c;
            c >>= 32;
        }
        c += t[N];
        t[N] = (uint)c;
        t[N + 1] = (uint)(c >> 32);
        uint m = t[0] * ninv;
        c = ((ulong)t[0] + (ulong)m * n[0]) >> 32;
        for (int j = 1; j < N; j++) {
            c += (ulong)t[j] + (ulong)m * n[j];
            t[j - 1] = (uint)c;
            c >>= 32;
        }
        c += t[N];
        t[N - 1] = (uint)c;
        t[N] = t[N + 1] + (uint)(c >> 32);
    }
    Num r, s;
    long borrow = 0;
    for (int j = 0; j < N; j++) {
        long d = (long)t[j] - (long)n[j] + borrow;
        s.w[j] = (uint)d;
        borrow = d >> 32;
    }
    bool ge = t[N] != 0 || borrow == 0;
    for (int j = 0; j < N; j++) r.w[j] = ge ? s.w[j] : t[j];
    return r;
}

inline Num add_mod(thread const Num& a, thread const Num& b, constant uint* n) {
    Num s, d;
    ulong c = 0;
    for (int j = 0; j < N; j++) {
        c += (ulong)a.w[j] + b.w[j];
        s.w[j] = (uint)c;
        c >>= 32;
    }
    long borrow = 0;
    for (int j = 0; j < N; j++) {
        long t = (long)s.w[j] - (long)n[j] + borrow;
        d.w[j] = (uint)t;
        borrow = t >> 32;
    }
    bool ge = c != 0 || borrow == 0;
    Num r;
    for (int j = 0; j < N; j++) r.w[j] = ge ? d.w[j] : s.w[j];
    return r;
}

inline Num sub_mod(thread const Num& a, thread const Num& b, constant uint* n) {
    Num d, s;
    long borrow = 0;
    for (int j = 0; j < N; j++) {
        long t = (long)a.w[j] - (long)b.w[j] + borrow;
        d.w[j] = (uint)t;
        borrow = t >> 32;
    }
    ulong c = 0;
    for (int j = 0; j < N; j++) {
        c += (ulong)d.w[j] + n[j];
        s.w[j] = (uint)c;
        c >>= 32;
    }
    Num r;
    for (int j = 0; j < N; j++) r.w[j] = borrow != 0 ? s.w[j] : d.w[j];
    return r;
}

struct Pt { Num x; Num z; };

inline Pt dbl(thread const Pt& p, thread const Num& a24, constant uint* n, uint ninv) {
    Num s = add_mod(p.x, p.z, n);
    Num sum2 = mont_mul(s, s, n, ninv);
    Num d = sub_mod(p.x, p.z, n);
    Num diff2 = mont_mul(d, d, n, ninv);
    Num four_xz = sub_mod(sum2, diff2, n);
    Pt r;
    r.x = mont_mul(sum2, diff2, n, ninv);
    Num t = add_mod(diff2, mont_mul(a24, four_xz, n, ninv), n);
    r.z = mont_mul(four_xz, t, n, ninv);
    return r;
}

inline Pt add(thread const Pt& p, thread const Pt& q, thread const Pt& diff, constant uint* n, uint ninv) {
    Num s = mont_mul(sub_mod(p.x, p.z, n), add_mod(q.x, q.z, n), n, ninv);
    Num t = mont_mul(add_mod(p.x, p.z, n), sub_mod(q.x, q.z, n), n, ninv);
    Num u = add_mod(s, t, n);
    Num v = sub_mod(s, t, n);
    Pt r;
    r.x = mont_mul(diff.z, mont_mul(u, u, n, ninv), n, ninv);
    r.z = mont_mul(diff.x, mont_mul(v, v, n, ninv), n, ninv);
    return r;
}

// Each thread: one curve. Multiplies its point by ks[0..count] in turn.
kernel void stage1(device uint* points [[buffer(0)]], device const uint* a24s [[buffer(1)]],
                   constant uint* n [[buffer(2)]], constant uint& ninv [[buffer(3)]],
                   device const uint* ks [[buffer(4)]], constant uint& count [[buffer(5)]],
                   uint gid [[thread_position_in_grid]]) {
    Pt p;
    Num a24;
    for (int i = 0; i < N; i++) {
        p.x.w[i] = points[gid * 2 * N + i];
        p.z.w[i] = points[gid * 2 * N + N + i];
        a24.w[i] = a24s[gid * N + i];
    }
    for (uint c = 0; c < count; c++) {
        uint k = ks[c];
        Pt r0 = p;
        Pt r1 = dbl(p, a24, n, ninv);
        for (int bit = 30 - clz(k); bit >= 0; bit--) {
            if ((k >> bit) & 1) {
                r0 = add(r1, r0, p, n, ninv);
                r1 = dbl(r1, a24, n, ninv);
            } else {
                r1 = add(r1, r0, p, n, ninv);
                r0 = dbl(r0, a24, n, ninv);
            }
        }
        p = r0;
    }
    for (int i = 0; i < N; i++) {
        points[gid * 2 * N + i] = p.x.w[i];
        points[gid * 2 * N + N + i] = p.z.w[i];
    }
}
"#;

const B1: u64 = 50_000;
/// Prime powers per GPU dispatch, to keep each one short.
const CHUNK: usize = 400;

fn to_u32s<const L: usize>(x: &Uint<L>) -> Vec<u32> {
    x.as_limbs()
        .iter()
        .flat_map(|&l| [l as u32, (l >> 32) as u32])
        .collect()
}

fn from_u32s<const L: usize>(w: &[u32]) -> Uint<L> {
    let mut limbs = [0u64; L];
    for (i, l) in limbs.iter_mut().enumerate() {
        *l = u64::from(w[2 * i]) | u64::from(w[2 * i + 1]) << 32;
    }
    Uint::from_limbs(limbs.map(|l| l as _))
}

/// The largest power of each prime up to B1, as stage 1 multiplies by.
fn prime_powers() -> Vec<u32> {
    Primes::new(2, B1)
        .map(|p| {
            let mut q = p;
            while q <= B1 / p {
                q *= p;
            }
            q as u32
        })
        .collect()
}

/// A Suyama curve for `sigma`: (a24, starting point), as cofox-core builds
/// it. The a24 field is private there, so it's recomputed the same way.
fn suyama<const L: usize>(m: &Monty<L>, sigma: u64) -> (Uint<L>, Point<L>) {
    let s = m.to_monty(&Uint::from_u64(sigma));
    let u = m.sub(&m.square(&s), &m.to_monty(&Uint::from_u64(5)));
    let v = m.add(&m.add(&s, &s), &m.add(&s, &s));
    let cube = |t: &Uint<L>| m.mul(&m.square(t), t);
    let (u3, v3) = (cube(&u), cube(&v));
    let numerator = m.mul(
        &cube(&m.sub(&v, &u)),
        &m.add(&m.add(&u, &u), &m.add(&u, &v)),
    );
    let denominator = m.mul(&m.mul(&m.to_monty(&Uint::from_u64(16)), &u3), &v);
    let a24 = m.mul(
        &numerator,
        &m.inv(&denominator).expect("no factor by accident"),
    );
    (a24, Point { x: u3, z: v3 })
}

fn run<const L: usize>(device: &Device, ks: &[u32]) {
    let n32 = 2 * L;
    let bits = 64 * L;
    // A fixed odd modulus with the top bit set.
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut limbs = [0u64; L];
    for l in &mut limbs {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *l = x;
    }
    limbs[0] |= 1;
    limbs[L - 1] |= 1 << 63;
    // The next prime, so that curve setup never stumbles on a factor.
    let mut n = Uint::<L>::from_limbs(limbs.map(|l| l as _));
    while !cofox_core::prime::is_prime(&n) {
        n = n.wrapping_add(&Uint::from_u64(2));
    }
    let monty = Monty::new(n).unwrap();
    let n_words = to_u32s(&n);
    let mut inv: u32 = 1;
    for _ in 0..5 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(n_words[0].wrapping_mul(inv)));
    }
    let ninv = inv.wrapping_neg();

    let src = KERNEL.replace("%N%", &n32.to_string());
    let lib = device
        .new_library_with_source(&src, &CompileOptions::new())
        .unwrap();
    let pso = device
        .new_compute_pipeline_state_with_function(&lib.get_function("stage1", None).unwrap())
        .unwrap();
    let queue = device.new_command_queue();
    let opts = MTLResourceOptions::StorageModeShared;
    let nb = device.new_buffer_with_data(n_words.as_ptr().cast(), (n32 * 4) as u64, opts);
    let kb = device.new_buffer_with_data(ks.as_ptr().cast(), (ks.len() * 4) as u64, opts);

    // CPU: curves on every core, the same per-prime-power ladder.
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let cpu_curves = 4 * cores;
    let t = Instant::now();
    let cpu_points: Vec<Point<L>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..cores)
            .map(|c| {
                let monty = &monty;
                s.spawn(move || {
                    (0..cpu_curves / cores)
                        .map(|i| {
                            let (a24, mut p) = suyama(monty, (6 + c + cores * i) as u64);
                            let curve = Curve::new(*monty, a24);
                            for &k in ks {
                                p = curve.mul(&p, k.into());
                            }
                            p
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    let cpu_rate = cpu_curves as f64 / t.elapsed().as_secs_f64();
    println!("{bits} bits, B1 = {B1}: CPU {cores} threads: {cpu_rate:.1} curves/s");

    for curves in [512usize, 2048, 8192] {
        // Same sigmas as the CPU for the first ones, so they can be checked.
        let setup: Vec<(Uint<L>, Point<L>)> = (0..curves)
            .map(|i| suyama(&monty, (6 + (i % cores) + cores * (i / cores)) as u64))
            .collect();
        let points: Vec<u32> = setup
            .iter()
            .flat_map(|(_, p)| to_u32s(&p.x).into_iter().chain(to_u32s(&p.z)))
            .collect();
        let a24s: Vec<u32> = setup.iter().flat_map(|(a, _)| to_u32s(a)).collect();
        let pb =
            device.new_buffer_with_data(points.as_ptr().cast(), (points.len() * 4) as u64, opts);
        let ab = device.new_buffer_with_data(a24s.as_ptr().cast(), (a24s.len() * 4) as u64, opts);
        let t = Instant::now();
        for (c, chunk) in ks.chunks(CHUNK).enumerate() {
            let count = chunk.len() as u32;
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&pb), 0);
            enc.set_buffer(1, Some(&ab), 0);
            enc.set_buffer(2, Some(&nb), 0);
            enc.set_bytes(3, 4, (&ninv as *const u32).cast());
            enc.set_buffer(4, Some(&kb), (c * CHUNK * 4) as u64);
            enc.set_bytes(5, 4, (&count as *const u32).cast());
            let tg = pso.max_total_threads_per_threadgroup().min(64);
            enc.dispatch_threads(MTLSize::new(curves as u64, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let secs = t.elapsed().as_secs_f64();
        // The CPU computed curves in thread order; map GPU index i to it.
        let out = unsafe { std::slice::from_raw_parts(pb.contents() as *const u32, points.len()) };
        let mut checked = 0;
        for i in 0..curves.min(cpu_curves) {
            let (c, k) = (i % cores, i / cores);
            let cpu = cpu_points[c * (cpu_curves / cores) + k];
            let gx: Uint<L> = from_u32s(&out[i * 2 * n32..i * 2 * n32 + n32]);
            let gz: Uint<L> = from_u32s(&out[i * 2 * n32 + n32..(i + 1) * 2 * n32]);
            assert_eq!(
                (gx, gz),
                (cpu.x, cpu.z),
                "{bits} bits, curve {i}: GPU and CPU disagree"
            );
            checked += 1;
        }
        let rate = curves as f64 / secs;
        println!(
            "  GPU {curves:5} curves at once: {rate:7.1} curves/s ({secs:.2} s), {:.1}x the CPU; {checked} curves checked",
            rate / cpu_rate
        );
    }
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    let ks = prime_powers();
    println!(
        "GPU: {}; {} prime powers up to {B1}",
        device.name(),
        ks.len()
    );
    run::<4>(&device, &ks);
    run::<6>(&device, &ks);
}
