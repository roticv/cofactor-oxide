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
> quadratic sieve splits what is left up to 120 digits. Both run on every
> core, natively and in the browser. The number field sieve works end to
> end in the CLI with `--gnfs`, where it factored RSA-100 in 29 minutes on
> six threads (the quadratic sieve takes about 20 on ten), RSA-120 in 2
> hours and RSA-140 in 59.5 hours on a ten-core M4 MacBook Air, and in the
> browser, which uses it from about 105 digits.

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
[docs/optimizations.md](docs/optimizations.md) records what made each stage
fast, with the measurements, and what was tried and dropped;
[docs/ideas.md](docs/ideas.md) lists what to try next, and what not to.

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

A composite that is a perfect power is replaced by its root first, since none
of the methods can split a prime power. A longer rho run follows ECM for composites up to 80 bits, where it is
certain to succeed. For composites of up to 200 bits (about 60 digits) that the
quadratic sieve will take, p−1 and p+1 are skipped: they take about 50 ms
whatever the size, longer than the sieve needs.

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
the quicker way to an answer. The sieve then runs on composites of 81 to 400
bits (120 digits). Each thread or Web Worker sieves its own polynomials and
sends the relations to one collector, which runs block Lanczos once it has
enough. Primes too large to hit a 64 KB sieve block more than a few times
are sieved through buckets: each polynomial drops their hits into the
block they land in, and trial division reads a candidate's large primes
back from its bucket. From 280 bits (about 85 digits) a relation may also
have two primes outside the factor base: the sieve splits such cofactors
with a 64-bit rho, and the collector combines relations whose large primes
form cycles. That saves about 30% at 90 digits.

On a 10-core Apple M4 (4 performance and 6 efficiency cores), two 30-digit
primes take under a second, two 33-digit primes about a second, two
40-digit primes about 15 seconds, two 45-digit primes about 2 minutes, and
RSA-100 (two 50-digit primes) about 20 minutes.

The crossover to GNFS is a rough guess for now, to be measured once it
exists.

## Native vs. browser

The browser build is mostly about convenience: open a page and factor a
number, nothing to install. Expect it to be slower than the native binary:

- **No native 64×64→128-bit multiply.** Wasm has `i64.mul` but nothing that
  returns the high half, so multi-precision arithmetic, the hot loop of ECM
  and the sieves, has to be emulated, or done with 32-bit limbs. The
  [wide-arithmetic proposal](https://github.com/WebAssembly/wide-arithmetic)
  adds `i64.mul_wide_u`, and the page loads a second build that uses it, with
  64-bit limbs, in browsers that support it (Firefox 155 at the time of
  writing, and Chrome behind `--js-flags=--wasm-staging`). In headless Chrome
  it made ECM on RSA-100 2.8 times as fast and GNFS sieving 1.3 times as
  fast, while the quadratic sieve, which spends its time adding bytes, gained
  2%.
- **Limited SIMD.** Wasm SIMD is fixed at 128 bits, while native code can use
  AVX2 or AVX-512.
- **Threads need extra setup.** Wasm threads rely on `SharedArrayBuffer`, which
  only works when the page is served with cross-origin isolation headers,
  which GitHub Pages can't send. Work is split across Web Workers instead,
  which pass messages: the GNFS matrix too, a part on each worker, with the
  workers summing their vectors directly with each other at every step.
- **Memory limits.** 32-bit Wasm can address at most 4 GiB. That's plenty for
  ECM and QS. GNFS keeps its relations packed, in chunks rather than one
  growing vector, and lets go of its duplicate check and unfiltered matrix
  before filtering: from RSA-110's 10 million relations the collecting
  worker reached 2.6 GB, and 3.6 GB through the square roots. The matrix's
  entries and vectors are spread over the workers. Much past that it won't
  fit.

In practice the browser handles trial division, p±1, ECM and QS
comfortably. It runs GNFS too, from about 105 digits (`?gnfs=BITS` in the
URL lowers that, to try it on smaller numbers), but with the square roots
on one worker and everything about 3× slower, jobs of that size take hours:
they're a job for the native binary.

GPUs were measured too: [`experiments/gpu-ecm`](experiments/gpu-ecm)
benchmarks ECM stage 1 on an Apple M4's GPU through Metal. It runs about 3×
faster than all ten CPU cores, but only with thousands of curves at once,
more than the pipeline's ECM levels use below 35 digits.

## Layout

```
cofactor-oxide/
├── crates/
│   ├── core/    # cofox-core: arithmetic, primality testing, factoring (no_std)
│   ├── cli/     # cofox: native command-line binary
│   └── wasm/    # cofox-wasm: wasm-bindgen bindings for the browser
├── experiments/
│   └── gpu-ecm/ # ECM on Apple GPUs (Metal), outside the workspace
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

The second number is (2⁶¹−1)(2⁸⁹−1). Both factors are too large for rho, and
p−1 would find 2⁶¹−1 because 2⁶¹−2 has only small prime factors. The third
has a 20-digit prime factor whose p−1 and p+1 aren't smooth, which ECM can
find. At 46 digits, though, the quadratic sieve on all cores is quicker than
either, so it splits both; with `--no-qs`, p−1 and ECM do. The fourth is a
product of two 30-digit primes, too large for ECM, so the quadratic sieve
splits it.

`--ecm 30` (or 35, 40) searches for larger factors, at about four times the
cost per level, `--threads N` sets the number of threads, `--no-qs` skips
the quadratic sieve, and `--gnfs` uses the number field sieve in its place,
for composites of 40 digits or more. A factor in parentheses is composite,
but the algorithms that ran couldn't split it. While ECM or the sieve runs,
a status line on stderr shows how far it has got, and GNFS prints a line
with the time at each stage.

`--save-relations FILE` appends GNFS's relations to `FILE`, in the format
msieve, GGNFS and CADO-NFS share, and writes the polynomials to
`FILE.poly`. `--load-relations FILE` starts from the relations in `FILE`,
with the polynomials in `FILE.poly`, and skips the special-q it records as
done, so that giving both with the same file resumes an interrupted run:

```sh
cofox --gnfs --save-relations rsa100.rels 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
# interrupted; later:
cofox --gnfs --load-relations rsa100.rels --save-relations rsa100.rels 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
```

Relations are flushed to the file every 10 seconds, so an interruption
loses at most the last few seconds' worth.

Sieving can also be split across machines. `--poly FILE` makes every
machine use the same polynomials: the first run to find `FILE` missing
chooses them and writes them there, and the others read them, so copy
the file around or keep it on a shared disk. `--special-q START:END` sieves
only that range of special-q and saves the relations. Then one run that
loads every machine's file solves the matrix, sieving more first if it
needs to:

```sh
cofox --gnfs --poly n.poly --special-q 1500000:1700000 --save-relations a.rels N   # machine A
cofox --gnfs --poly n.poly --special-q 1700000:1900000 --save-relations b.rels N   # machine B
cofox --gnfs --poly n.poly --load-relations a.rels --load-relations b.rels N
```

The first line GNFS prints on stderr says where its special-q start, for
this size of number; ranges from there up are the most productive, as
smaller special-q give smaller values to factor.

cofox chooses polynomials for numbers up to about 160 digits (530 bits),
searching as CADO-NFS does: Kleinjung's method over many special-q, size
optimization by LLL reduction, then a root sieve over prime powers that
weighs each rotation's size against its roots. For RSA-110 its
polynomial's Murphy E, by CADO-NFS's measure, is 5.74·10⁻⁶, against
3.34·10⁻⁶ for the one it chose before and 5.89·10⁻⁶ for CADO-NFS's own;
sieving with the earlier of these improvements alone took 1.4 ms a
relation instead of 2.5 to 2.9. For RSA-110 the search takes about 2
minutes on 10 threads; at 150 digits its full budget would take about 8
hours, and a twentieth of it 25 minutes. Past 160 digits, give it
ones chosen by CADO-NFS or msieve with `--poly`, in either's format, of
degree up to 6 with coefficients of any size below 2^256. The sieve's
parameters run to 600 bits (180 digits), measured up to 130 digits and
guessed beyond. The siever can keep up to three large primes a side, each below 2^40, and
special-q up to 2^40 too, as numbers such as RSA-270 need. Cofactors
below 2^62 are split by ECM and rho, and larger ones, of up to 127 bits,
by a quick P−1 alone: at 90 digits with 34-bit large primes, ECM curves
there cost far more than the relations they added, and dropping them
gave 3 to 4 times as many relations a second. Up to 330 bits the parameter table keeps
one large prime a side, as at 90 digits two a side took 2.5 times as
long; from 364 bits it uses smaller factor bases with two a side, which
factored RSA-110 in 3 h 19 min on 6 threads against 4 h 56 min. The rows
for 120 and 130 digits come from timing slices of special-q with each
candidate and counting the duplicates a range of special-q gives:
RSA-120 should take about 10 core-hours of sieving and RSA-130 about 37.
Real runs took 2 h 03 min for RSA-120 and 59.5 h for RSA-140, which
needed more relations than the model predicted and 14.8 GB of memory;
[docs/optimizations.md](docs/optimizations.md) has the details.

WebAssembly needs the `wasm32-unknown-unknown` target, which requires a
rustup-managed toolchain (Homebrew's `rust` formula can't add targets), and a
`wasm-bindgen-cli` whose version matches `wasm-bindgen` in `Cargo.lock`:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked

cargo build --release --target wasm32-unknown-unknown -p cofox-wasm
wasm-bindgen --target web --out-dir web/pkg \
  target/wasm32-unknown-unknown/release/cofox_wasm.wasm

# Optional: the build for browsers with wide arithmetic, which also uses
# SIMD, as all of them have it. Rust still warns that wide arithmetic is
# unstable.
RUSTFLAGS="-C target-feature=+wide-arithmetic,+simd128 --cfg cofox_wide_arithmetic" \
  cargo build --release --target wasm32-unknown-unknown -p cofox-wasm \
  --target-dir target/wide
wasm-bindgen --target web --out-dir web/pkg-wide \
  target/wide/wasm32-unknown-unknown/release/cofox_wasm.wasm

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
- [x] GNFS: Kleinjung polynomials, lattice sieve, filtering, block Lanczos,
      Newton square root; in the CLI and the browser
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
