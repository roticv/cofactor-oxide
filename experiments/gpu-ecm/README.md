# GPU ECM benchmark

Does ECM run faster on an Apple GPU than on the CPU? Two benchmarks compare a
Metal kernel against `cofox-core` on every CPU core:

- `mulmod`: Montgomery multiplication alone, repeated squaring of 65,536
  values at 256, 384 and 512 bits.
- `stage1`: ECM stage 1 with B1 = 50,000 (the 25-digit level), one Suyama
  curve per GPU thread, at 256 and 384 bits, with 512, 2,048 and 8,192
  curves at once.

The GPU uses 32-bit limbs and the same R = 2^bits as `cofox-core`, and
`stage1` uses the same curve formulas and ladder, so the results must
match the CPU's exactly. Both benchmarks check that they do.

It needs macOS, and stays out of the main workspace for that reason:

```sh
cd experiments/gpu-ecm
cargo run --release --bin mulmod
cargo run --release --bin stage1
```

## Results

Apple M4 (4 performance and 6 efficiency cores, 10-core GPU), with other
applications keeping the load average near 10, so the CPU figures are
somewhat pessimistic.

Montgomery multiplications per second:

| Size | GPU | CPU, 10 threads | GPU ÷ CPU |
| ---- | --- | --------------- | --------- |
| 256 bits | 1,370 M | 269–339 M | 4–5× |
| 384 bits | 643 M | 153–198 M | 3–4× |
| 512 bits | 367 M | 110–113 M | 3.3× |

ECM stage 1, B1 = 50,000, curves per second:

| Size | CPU, 10 threads | GPU, 512 curves | GPU, 2,048 curves | GPU, 8,192 curves |
| ---- | --------------- | --------------- | ----------------- | ----------------- |
| 256 bits | 401 | 331 (0.8×) | 970 (2.4×) | 1,457 (3.6×) |
| 384 bits | 236 | 200 (0.8×) | 481 (2.0×) | 675 (2.9×) |

## Conclusions

- Keeping a whole curve per thread costs little: at 8,192 curves, 256-bit
  stage 1 reaches about 85% of the raw multiplication rate.
- The GPU needs thousands of curves in flight. With 512 it loses to the CPU.
- The pipeline's ECM levels run 30 to 760 curves up to the 30-digit level,
  which would gain nothing. Only the 35- and 40-digit levels (1,900 and
  5,400 curves) are large enough, for roughly 2–3.5× on stage 1, with
  stage 2 still on the CPU.
- So it isn't worth a Metal-only second implementation of ECM for now. It
  would be worth revisiting for deep ECM on numbers beyond the quadratic
  sieve, ideally with more, shorter curves (lower B1) to suit the GPU.
