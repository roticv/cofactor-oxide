# cofactor-oxide

Integer factorization in Rust, compiled both to a native binary and to
WebAssembly so you can factor numbers right in your browser. **cofox** for
short, which is also the name of the command-line tool.

> **Status:** early development. Fixed-width arithmetic is in place, but the only
> algorithm so far is trial division on 64-bit numbers. Most of this README
> describes the plan.

## Goals

- **One codebase, two targets.** The same Rust core runs as a native CLI and as
  a WebAssembly module for the browser.
- **Pick the right algorithm for the job.** Small factors are removed cheaply
  first, and the expensive general-purpose methods only run on what remains.
- **Readable implementations.** The code should be a useful reference for how
  these algorithms work, not only a fast black box.

## Factoring pipeline

A number moves through these stages in order. Each stage peels off the factors
it can find, and whatever composite is left goes to the next stage.

| Stage | Algorithm | Good at |
| ----- | --------- | ------- |
| 1 | **Trial division** | Tiny primes (up to roughly 10⁶) |
| 2 | **Pollard p−1** / **Williams p+1** | Primes *p* where *p−1* or *p+1* is smooth |
| 3 | **ECM** (Lenstra elliptic curve method) | Primes up to about 30–40 digits, whatever the size of *n* |
| 4 | **Quadratic Sieve** (SIQS) | Composites up to about 100 digits |
| 5 | **General Number Field Sieve** (GNFS) | Composites beyond about 100 digits |

A primality test (such as BPSW) runs between stages so that nothing is spent on
cofactors that are already prime.

The digit ranges are rough. The real crossover points will be measured once
the implementations exist, and they will differ between native and Wasm.

## Native vs. browser

The browser build is mostly about convenience: open a page and factor a
number, nothing to install. Expect it to be slower than the native binary:

- **No native 64×64→128-bit multiply.** Wasm has `i64.mul` but nothing that
  returns the high half, so multi-precision arithmetic, the hot loop of ECM
  and the sieves, has to be emulated.
- **Limited SIMD.** Wasm SIMD is fixed at 128 bits, while native code can use
  AVX2 or AVX-512.
- **Threads need extra setup.** Wasm threads rely on `SharedArrayBuffer`, which
  only works when the page is served with cross-origin isolation headers. Work
  can also be split across Web Workers.
- **Memory limits.** 32-bit Wasm can address at most 4 GiB. That's plenty for
  ECM and QS, but the linear algebra step of a large GNFS job may not fit.

The likely outcome is that the browser handles trial division, p±1, ECM and
QS comfortably, while large GNFS jobs are a job for the native binary.

## Layout

```
cofactor-oxide/
├── crates/
│   ├── core/    # cofox-core: arithmetic, primality testing, factoring (no_std)
│   ├── cli/     # cofox: native command-line binary
│   └── wasm/    # cofox-wasm: wasm-bindgen bindings for the browser
└── web/         # browser front end
```

## Building

Native:

```sh
cargo build --release
./target/release/cofox 1000000016000000063
# 1000000016000000063: 1000000007 1000000009
```

WebAssembly (needs the `wasm32-unknown-unknown` target and
[wasm-pack](https://rustwasm.github.io/wasm-pack/)):

```sh
rustup target add wasm32-unknown-unknown
wasm-pack build crates/wasm --target web --out-dir ../../web/pkg
python3 -m http.server -d web 8000   # then open http://localhost:8000
```

## Roadmap

- [x] Fixed-width integers and Montgomery multiplication
- [ ] Trial division
- [ ] Primality testing (Miller–Rabin, BPSW)
- [x] Pollard rho (Brent)
- [ ] Pollard p−1
- [ ] Williams p+1
- [ ] ECM (stage 1 and stage 2)
- [ ] Self-initializing quadratic sieve (SIQS)
- [ ] GNFS
- [ ] Native CLI
- [ ] Wasm bindings and browser demo
- [ ] Web Worker parallelism in the browser

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
