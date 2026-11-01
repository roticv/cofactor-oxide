# cofactor-oxide

[![CI](https://github.com/roticv/cofactor-oxide/actions/workflows/ci.yml/badge.svg)](https://github.com/roticv/cofactor-oxide/actions/workflows/ci.yml)
[![Demo](https://img.shields.io/badge/demo-try_in_browser-b4532a)](https://roticv.github.io/cofactor-oxide/)

Integer factorization in Rust, compiled both to a native binary and to
WebAssembly so you can factor numbers right in your browser. **cofox** for
short, which is also the name of the command-line tool.

> **Status:** early development. The CLI factors numbers of up to 1024 bits
> using trial division, Pollard rho, Pollard p−1, Williams p+1, ECM, the
> self-initializing quadratic sieve and the BPSW primality test. ECM finds
> factors up to about 25 digits by default (more with `--ecm`), and the
> quadratic sieve splits what is left up to 110 digits. Both run on every
> core, natively and in the browser. GNFS is planned.

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
| 1 | **Trial division** | Tiny primes (currently below 4096) |
| 2 | **Pollard rho** (Brent's variant) | Primes up to about 8 digits, in a quick first run |
| 3 | **Pollard p−1** and **Williams p+1** (B1 = 5·10⁴, B2 = 5·10⁶) | Primes *p* of any size where *p−1* or *p+1* is smooth |
| 4 | **ECM** (Lenstra elliptic curve method, levels of 15 to 40 digits; 25 by default) | Primes up to about 30–40 digits, whatever the size of *n* |
| 5 | **Quadratic sieve** (SIQS, double large primes, block Lanczos) | Composites up to about 100 digits |
| 6 | **General Number Field Sieve** (GNFS) | Composites beyond about 100 digits |

The Baillie–PSW primality test runs between stages so that nothing is spent on
cofactors that are already prime. No composite is known to pass it, and none
exists below 2⁶⁴.

A longer rho run follows ECM for composites up to 80 bits, where it is
certain to succeed.

Each ECM level runs about as many curves as it takes on average to find a
prime of that size: 30 curves with B1 = 2000 for 15 digits, 100 with
B1 = 11000 for 20, and so on, with B2 = 100·B1. So a level finds a prime of
its own size most of the time, and smaller primes almost always. Each level
takes about four times as long as the one before.

The curves are independent, so the CLI splits them across one thread per
core and the browser across up to eight Web Workers. On a 10-core machine,
giving up on a 100-digit composite with no factor below 25 digits takes about
7 seconds natively.

When the quadratic sieve follows, ECM stops sooner, at the level for primes
of about 30% of the composite's digits (at least 15): past that, sieving is
the quicker way to an answer. The sieve then runs on composites of 81 to 363
bits (110 digits). Each thread or Web Worker sieves its own polynomials and
sends the relations to one collector, which runs block Lanczos once it has
enough. Primes too large to hit a 64 KB sieve block more than a few times
are sieved through buckets: each polynomial drops their hits into the
block they land in, and trial division reads a candidate's large primes
back from its bucket. On a 10-core Apple M4 (4 performance and 6
efficiency cores), two 30-digit primes take under a second, two 33-digit
primes about a second, two 40-digit primes about 17 seconds, and RSA-100
(two 50-digit primes) about 28 minutes. From 280 bits (about 85 digits)
a relation may also have two primes outside the factor base: the sieve
splits such cofactors with a 64-bit rho, and the collector combines
relations whose large primes form cycles. That saves about 30% at 90 digits.

The crossover to GNFS is a rough guess for now, to be measured once it
exists.

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
./target/release/cofox 1000000016000000063 1427247692705959880439315947500961989719490561 7358668553614726823037645838094560989037094819 95741717322359106137862766319375166032677068850856466452653
# 1000000016000000063: 1000000007 1000000009
# 1427247692705959880439315947500961989719490561: 2305843009213693951 618970019642690137449562111
# 7358668553614726823037645838094560989037094819: 11888570237800257629 618970019642690137449562111
# 95741717322359106137862766319375166032677068850856466452653: 266273800333898804119934270521 359561162992010719067582390293
```

The second number is (2⁶¹−1)(2⁸⁹−1). Both factors are too large for rho, but
p−1 finds 2⁶¹−1 because 2⁶¹−2 has only small prime factors. The third has a
20-digit prime factor whose p−1 and p+1 aren't smooth, so it takes ECM. The
fourth is a product of two 30-digit primes, too large for ECM, so the
quadratic sieve splits it.

`--ecm 30` (or 35, 40) searches for larger factors, at about four times the
cost per level, `--threads N` sets the number of threads, and `--no-qs`
skips the quadratic sieve. A factor in parentheses is composite, but the
algorithms that ran couldn't split it. While ECM or the sieve runs, a status
line on stderr shows how far it has got.

WebAssembly needs the `wasm32-unknown-unknown` target, which requires a
rustup-managed toolchain (Homebrew's `rust` formula can't add targets), and a
`wasm-bindgen-cli` whose version matches `wasm-bindgen` in `Cargo.lock`:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked

cargo build --release --target wasm32-unknown-unknown -p cofox-wasm
wasm-bindgen --target web --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/cofox_wasm.wasm
python3 -m http.server -d web 8000   # then open http://localhost:8000
```

The page has to be served over HTTP, because browsers won't load Wasm or JS
modules from `file://`.

Every push to `main` also builds the demo and deploys it to GitHub Pages at
<https://roticv.github.io/cofactor-oxide/>, using
[`.github/workflows/pages.yml`](.github/workflows/pages.yml).

## Roadmap

- [x] Fixed-width integers and Montgomery multiplication
- [x] Trial division
- [x] Primality testing (Miller–Rabin, BPSW)
- [x] Pollard rho (Brent)
- [x] Pollard p−1 (two stages)
- [x] Williams p+1 (two stages, baby-step giant-step stage 2)
- [x] ECM (Montgomery curves, Suyama parametrization, two stages)
- [x] Self-initializing quadratic sieve (SIQS), with block Lanczos
- [ ] GNFS
- [x] Native CLI
- [x] Wasm bindings and browser demo
- [x] Parallel ECM and sieving: threads in the CLI, Web Workers in the browser
- [x] Progress reporting and cancellation

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
