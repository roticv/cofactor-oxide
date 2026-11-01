//! Montgomery multiplications per second: Metal GPU (32-bit limbs) against
//! cofox-core's Monty on every CPU core (64-bit limbs), at 256, 384 and 512
//! bits. Both use R = 2^bits, so repeated squaring must agree exactly.

use cofox_core::{Monty, Uint};
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
#define N %N%

// r = a·b·2^(−32N) mod n, CIOS with 32-bit limbs.
inline void mont_mul(thread uint* r, thread const uint* a, thread const uint* b,
                     constant uint* n, uint ninv) {
    uint t[N + 2];
    for (int i = 0; i < N + 2; i++) t[i] = 0;
    for (int i = 0; i < N; i++) {
        ulong c = 0;
        for (int j = 0; j < N; j++) {
            c += (ulong)t[j] + (ulong)a[j] * b[i];
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
    // Subtract n once if t ≥ n.
    uint s[N];
    long borrow = 0;
    for (int j = 0; j < N; j++) {
        long d = (long)t[j] - (long)n[j] + borrow;
        s[j] = (uint)d;
        borrow = d >> 32;
    }
    bool ge = t[N] != 0 || borrow == 0;
    for (int j = 0; j < N; j++) r[j] = ge ? s[j] : t[j];
}

kernel void bench(device uint* xs [[buffer(0)]], constant uint* n [[buffer(1)]],
                  constant uint& ninv [[buffer(2)]], constant uint& iters [[buffer(3)]],
                  uint gid [[thread_position_in_grid]]) {
    uint x[N], y[N];
    for (int i = 0; i < N; i++) x[i] = xs[gid * N + i];
    for (uint k = 0; k < iters; k++) {
        mont_mul(y, x, x, n, ninv);
        for (int i = 0; i < N; i++) x[i] = y[i];
    }
    for (int i = 0; i < N; i++) xs[gid * N + i] = x[i];
}
"#;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

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

fn run<const L: usize>(device: &Device, rng: &mut Rng) {
    let n32 = 2 * L; // 32-bit limbs
    let bits = 64 * L;
    // An odd modulus with the top bit set.
    let mut limbs = [0u64; L];
    for l in &mut limbs {
        *l = rng.next();
    }
    limbs[0] |= 1;
    limbs[L - 1] |= 1 << 63;
    let n = Uint::<L>::from_limbs(limbs.map(|l| l as _));
    let monty = Monty::new(n).unwrap();
    let n_words = to_u32s(&n);
    let mut inv: u32 = 1;
    for _ in 0..5 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(n_words[0].wrapping_mul(inv)));
    }
    let ninv = inv.wrapping_neg();

    // Starting values below n.
    let threads: u64 = 1 << 16;
    let iters: u32 = 16384;
    let start: Vec<Uint<L>> = (0..threads)
        .map(|_| {
            let mut l = [0u64; L];
            for x in &mut l {
                *x = rng.next();
            }
            l[L - 1] >>= 1;
            Uint::from_limbs(l.map(|x| x as _))
        })
        .collect();
    let words: Vec<u32> = start.iter().flat_map(to_u32s).collect();

    // GPU.
    let src = KERNEL.replace("%N%", &n32.to_string());
    let lib = device
        .new_library_with_source(&src, &CompileOptions::new())
        .unwrap();
    let pso = device
        .new_compute_pipeline_state_with_function(&lib.get_function("bench", None).unwrap())
        .unwrap();
    let opts = MTLResourceOptions::StorageModeShared;
    let xs = device.new_buffer_with_data(words.as_ptr().cast(), (words.len() * 4) as u64, opts);
    let nb = device.new_buffer_with_data(n_words.as_ptr().cast(), (n_words.len() * 4) as u64, opts);
    let queue = device.new_command_queue();
    let dispatch = |iters: u32| {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&xs), 0);
        enc.set_buffer(1, Some(&nb), 0);
        enc.set_bytes(2, 4, (&ninv as *const u32).cast());
        enc.set_bytes(3, 4, (&iters as *const u32).cast());
        let tg = pso.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    };
    let t = Instant::now();
    dispatch(iters);
    let gpu_secs = t.elapsed().as_secs_f64();
    let gpu_rate = threads as f64 * f64::from(iters) / gpu_secs;

    // Check the first few threads against the CPU.
    let out = unsafe { std::slice::from_raw_parts(xs.contents() as *const u32, words.len()) };
    for i in 0..8 {
        let mut x = start[i];
        for _ in 0..iters {
            x = monty.mul(&x, &x);
        }
        let g: Uint<L> = from_u32s(&out[i * n32..(i + 1) * n32]);
        assert_eq!(g, x, "{bits} bits, thread {i}: GPU and CPU disagree");
    }

    // CPU on every core.
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let per_core: u64 = 8_000_000;
    let t = Instant::now();
    std::thread::scope(|s| {
        for &x0 in start.iter().take(cores) {
            let monty = &monty;
            s.spawn(move || {
                let mut x = x0;
                for _ in 0..per_core {
                    x = monty.mul(&x, &x);
                }
                std::hint::black_box(x);
            });
        }
    });
    let cpu_rate = (cores as u64 * per_core) as f64 / t.elapsed().as_secs_f64();
    println!(
        "{bits} bits: GPU {:.0} M mulmod/s ({gpu_secs:.2} s), CPU {cores} threads {:.0} M mulmod/s, GPU/CPU {:.1}x (results verified)",
        gpu_rate / 1e6,
        cpu_rate / 1e6,
        gpu_rate / cpu_rate
    );
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    println!("GPU: {}", device.name());
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    run::<4>(&device, &mut rng);
    run::<6>(&device, &mut rng);
    run::<8>(&device, &mut rng);
}
