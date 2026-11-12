# Optimizations

This is a record of what made cofox fast, step by step through the
factoring pipeline: what each change was, what it measured, and what was
tried and dropped. The commit messages have the details; this gathers
them in one place, along with experiments that never became commits.

Unless noted, times are on a 10-core Apple M4 (4 performance and 6
efficiency cores), natively, with release builds. "RSA-100" and "RSA-110"
are the RSA challenge numbers of 330 and 364 bits. Numbers are from the
commit that made the change, so later ones build on earlier ones.

## Contents

- [Arithmetic](#arithmetic)
- [Small factors: trial division, rho, P−1, P+1](#small-factors-trial-division-rho-p1-p1)
- [ECM](#ecm)
- [Quadratic sieve](#quadratic-sieve)
- [Number field sieve](#number-field-sieve)
  - [Polynomial selection](#polynomial-selection)
  - [Sieving](#sieving)
  - [Large primes and cofactors](#large-primes-and-cofactors)
  - [Filtering](#filtering)
  - [Linear algebra](#linear-algebra)
  - [Square root](#square-root)
  - [Memory](#memory)
- [The browser](#the-browser)
- [Ways of measuring that paid off](#ways-of-measuring-that-paid-off)
- [Not yet tried](#not-yet-tried)

## Arithmetic

Numbers are fixed-width `Uint<LIMBS>`, on the stack, with Montgomery
multiplication (CIOS) for everything done modulo `N`.

| Change | Effect |
|---|---|
| 64-bit limbs natively, 32-bit on wasm32 | wasm has no 64×64→128-bit multiply, so a `u128` product is a library call; `u32` limbs with `u64` products are one `i64.mul` |
| Dedicated squaring | Wins only at one limb in the benchmarks, so `Monty::square` uses it there and multiplication elsewhere |
| Inverses by binary extended GCD | Returns the common factor when there is no inverse, which ECM uses as a found factor |
| Wide-arithmetic wasm build | See [the browser](#the-browser) |

`--cfg cofox_limb32` forces 32-bit limbs natively, which is how CI tests
the wasm code path.

## Small factors: trial division, rho, P−1, P+1

| Change | Effect |
|---|---|
| Brent's rho with batched GCDs | 128 differences per GCD, replaying the batch step by step when it collapses to `n`; all in Montgomery form |
| Batched GCDs with checkpoint replay, shared by P−1, P+1 and ECM | One GCD per 1024 primes; a batch whose GCD is `n` is replayed from its checkpoint to separate the factors |
| Stage 2 by baby-step giant-step, shared by P+1 and ECM | `D = 2310`, multiplying differences over primes `kD ± j` |
| Skip P−1 and P+1 below 200 bits | They cost about 50 ms whatever the size, in every worker, longer than the quadratic sieve needs below about 60 digits. A 39-digit semiprime: 0.11 s → 0.02 s; 58 digits: 0.27 s → 0.19 s |
| Roots of perfect powers first | No method splits a prime power, and the quadratic sieve never can, so `r^k` is factored as `r` |

## ECM

Montgomery curves with Suyama's parametrization, x-only arithmetic, the
Montgomery ladder for stage 1, and the shared baby-step giant-step stage 2
with `B2 = 100·B1`. It runs in levels, for 15-, 20-, 25-digit primes and
so on, each with about the expected number of curves, and stops at about
30% of the composite's digits when the quadratic sieve follows. The
curves are independent, so threads and Web Workers share each level's.

**Tried and dropped: ECM on the GPU.** `experiments/gpu-ecm` runs stage 1
in a Metal kernel on the M4's GPU. It does 3 to 5 times the Montgomery
multiplications of all ten CPU cores, and stage 1 reaches about 3.6 times
with 8,192 curves at once, but loses to the CPU with 512. That's more
curves than the pipeline's levels use below 35 digits, so ECM stays on
the CPU.

## Quadratic sieve

Self-initializing (SIQS), with the Knuth–Schroeppel multiplier,
polynomials in Gray-code order, a log sieve over 64 KB blocks, and block
Lanczos for the matrix.

| Change | Effect |
|---|---|
| SIQS with one large prime | 59 digits in about 2 s, 66 in about 10 s, on one thread |
| Values from `(ax + 2b)x + c`, Lemire's remainder by multiplication for trial division, primes below 100 left out of the sieve | Part of the above |
| Block Lanczos instead of dense elimination | 6,500 columns: 0.6 s → 0.1 s; 100k columns about 40 s |
| Sieving on every thread, one collector | Two 33-digit primes in 2.6 s on 10 cores |
| Four sieve blocks a side from 298 bits | At RSA-100, eight times the relations a second of msieve's nine blocks |
| Large primes through buckets | Each polynomial drops their hits into the block they land in; trial division reads them back. RSA-100: 2 h 05 min → 39.7 min; 80 digits: 42.5 s → 17 s; 66 digits: 2.6 s → 1 s |
| Two large primes, from 280 bits | Cofactors split with a 64-bit rho, relations combined through cycles. 90 digits: 224 s → 151 s; RSA-100: 39.7 → 28 min |
| Branch-free root updates and bucket fills | Roots in flat arrays, updated with NEON `umin`: 200 µs → 25 µs a polynomial at RSA-100; unconditional bucket writes: 326 µs → 180 µs. RSA-100: 28 → 19.7 min |

RSA-100 now takes about 20 minutes on 10 threads.

**Compared with msiqs.** [msiqs](https://github.com/danaj/msiqs), a C
SIQS, factors RSA-100 in 4 min 43 s on 8 M1 Pro threads, so this sieve
has several times of headroom. Its notes point at sieving large primes
directly with one-hit and two-hit loops, which it found faster than
buckets; smaller factor bases with more polynomials; and an unrolled
two-root loop. None is tried here yet.

## Number field sieve

### Polynomial selection

Polynomial quality turned out to matter more than anything in the
siever. The table follows RSA-110's polynomial through the changes; the
last three columns compare them with CADO-NFS's polynomial for the same
number.

| Change | Effect |
|---|---|
| Base-m, size- and root-optimized, ranked by Murphy's E | RSA-100: α from +0.81 to −4.39, about twice the relations per special-q and per second |
| Kleinjung's method, `g = l·x − m` | RSA-100: size 64.6 → 55.6, Murphy's E 26 times higher, 3.1 times the relations a second |
| A harder search, on all threads | 32% more relations a second at 90 digits, 61% more at RSA-100; 90 digits factored in 365 s instead of 505 s |
| CADO-NFS's stage 1 and size optimization | RSA-110 sieving: 2.5–2.9 → about 1.4 ms a relation |
| Root optimization by size and p-adic α, Murphy's E with the large-prime bounds | RSA-110 sieving: about 1.4 → 1.14–1.18 ms a relation |

The last two came from measuring against CADO-NFS (see [ways of
measuring](#ways-of-measuring-that-paid-off)). On RSA-110:

| Polynomial | Murphy's E, by CADO-NFS | Sieving, ms a relation at special-q 1.5M / 3.5M / 6.5M |
|---|---|---|
| Before (degree 4) | 3.34·10⁻⁶ | 2.80 / 2.86 / 2.51 |
| After stage 1 and size optimization | 4.92·10⁻⁶ | 1.39 / 1.39 / 1.40 |
| After root optimization | 5.74·10⁻⁶ | 1.14 / 1.14 / 1.18 |
| CADO-NFS's (degree 5) | 5.89·10⁻⁶ | 1.07 / 1.27 / 1.44 |

What made the difference, in order of discovery:

1. **Many special-q per leading coefficient.** Stage 1 built one
   special-q from primes with at least two roots, giving at most 64
   roots, and found 954 raw candidates for RSA-110. CADO-NFS walks
   through products of a few small primes, including those with one
   root, up to tens of thousands of roots per coefficient, with
   `q·(2P)⁴ < m₀`.
2. **Kleinjung's transformed problem.** Searching `m̃` with
   `m̃^d ≡ d^d·a_d^(d−1)·N (mod l²)` makes `a_{d−1}` at most `d·a_d/2`,
   so the search's freedom goes into `a_{d−2}`. With both changes, raw
   candidates matched CADO-NFS's: best 49.0 against 50.0, average 58.4
   against 58.3 (log₂ of the L² norm).
3. **Size optimization by LLL.** At each translation, LLL reduction of
   the lattice spanned by `f` and `xⁱ·g`, `i ≤ d − 2`, scaled by the
   skew, finds the smallest `c·f + r·g`. Stepping one coefficient at a
   time gained 1 bit on CADO-NFS's raw candidates, where CADO-NFS gained
   6.8. LLL gets within 0.5 bits of it. The resultant becomes `c·N`,
   which the rest of the pipeline handles; a test factors with `3·f + g`.
4. **A coarse scan of translations first.** The landscape has several
   basins. Without the scan, sizes averaged 54.0; with it, 52.0, against
   CADO-NFS's 51.5.
5. **p-adic α.** Counting prime powers and multiple roots, over primes
   below 2000, rather than simple roots below 200. It matches CADO-NFS's
   α for its polynomial to two decimals.
6. **Root sieve scored by size, not just roots.** Rotations far from 0
   gain roots but grow the size. Each cell is now scored by its size
   after translation, interpolated along the row, less its root gain.
   The root sieve also counts prime powers up to 4096.
7. **Murphy's E with the large-prime bounds.** With the factor bases'
   bounds, E sat so far in the tail of Dickman's ρ that it ranked
   RSA-110's candidates in nearly random order, discarding the root
   sieve's best. CADO-NFS changed to the large-prime bounds in 2018 for
   the same reason.

Tried and dropped:

- **Least-squares rotation.** Solving for the best real rotation at a
  fixed skew and rounding improved size optimization by 2.9 bits, but
  stayed 3 bits behind CADO-NFS, because it can't multiply `f`.
- **Translations from the roots of `f`'s derivatives** as starting
  points: no better than 0.
- **Degree 5 below 370 bits.** CADO-NFS switches at about 100 digits.
  With the old search our degree-5 polynomials were worse; with the new
  one, RSA-110's degree 4 and 5 tie (5.74 and 5.65·10⁻⁶; 1.15 and 1.14
  ms a relation on average), so the switch stays.
- **Nonlinear pairs**, two cubics or two quadratics, built by Coxon's
  and Montgomery's methods in a Python experiment and scored with the
  same Murphy's E: 12 times worse than our linear pair at RSA-100, 29 at
  RSA-120 and 33 at RSA-130. Even a pair at the resultant bound only
  ties at 130 digits; [ideas.md](ideas.md#nonlinear-polynomial-pairs)
  has the numbers.

Selection now takes about 3 s at 262 bits, 26 s at 330 and about 2
minutes at RSA-110, on 10 threads.

### Sieving

A lattice siever: for each special-q ideal, a basis reduced for the
polynomial's skew, and a `2^I` by `2^(I−1)` region, with primes below `I`
sieved row by row and larger ones walked from hit to hit by Franke and
Kleinjung's method.

| Change | Effect |
|---|---|
| Lattice siever instead of a line siever | Needed to scale; with the line siever's parameters it wasn't faster at first |
| One bucket per slab for large primes | Each large prime is walked once per special-q, not once per slab |
| Franke–Kleinjung step by multiplying, not branching | Its case is close to random |
| 32-bit Euclid for root inverses and steps, four primes at once | Their division chains overlap |
| Small primes carry their first hit from row to row | Saves two divisions a row |
| Candidate scan 32 positions at a time | Together with the four rows above, 90 digits: 18.1 s → 10.4 s for the same 138 special-q on one thread; a full matrix in 265 s instead of 559 s on ten |
| Resieving candidates one by one, with a bitset of marked offsets | 64 KB instead of 512 KB, keeping the bucket scans in L1. RSA-110: 2–3.5% faster |
| Only odd `x` on even rows | Where `i` and `j` are both even, so are `a` and `b`. RSA-110: 3.3% faster |

Tried and dropped:

- **Leaving even-even positions out of the buckets too:** slower, as the
  branch mispredicts.
- **4-byte bucket entries:** 2–3% slower than 8-byte ones.

**Compared with CADO-NFS's `las`.** On the same RSA-110 polynomial and
settings, one thread, special-q 6.5M to 6.503M: cofox found 7,654
relations in 18.9 s (2.48 ms each), `las` 5,587 in 17.1 s (3.06 ms).
`las` sieves only about 700 of the 2,048 rows per special-q. So the inner
loops aren't where cofox was behind; the polynomial was.

### Large primes and cofactors

Each side has a factor-base bound, a large-prime bound and a cofactor
bound, as in CADO-NFS, with up to three large primes a side and special-q
and large primes up to 2⁴⁰.

| Change | Effect |
|---|---|
| Two-prime cofactors past 2⁴⁴ split by 64-bit ECM | 32 curves, `B1 = 200`, `B2 = 5000`, bounds worked out at compile time. Two 31-bit primes: 297 µs → 38 µs; sieving with two 28-bit large primes a side 2.1 times as fast. Rho stays below 2⁴⁴, where it's faster |
| Cofactors past 2⁶² get a quick P−1 alone | Nearly all have a prime too large; the ECM curves found relations at a twentieth of the sieve's rate. 3.3 times the relations a second with two large primes, 3.8 with three |
| Two large primes a side with smaller factor bases from 364 bits | RSA-110: 4 h 56 min → 3 h 19 min on 6 threads. It needed 10.1 million relations rather than 6.3 million, but each special-q gave over twice as many |
| Small primes divided out in 32-bit pieces | No 128-bit division in software |
| Splits wait until both sides pass the cheap tests | A cofactor is first checked for size, `B² ≤ c < L²` and a base-2 probable-prime test; only then are the 2LP splits run, rational side first. RSA-130 slice at 8M with two large primes: 1.03 million rational splits (98,000 successes) → 5,800, split time 2.5 s → 0.27 s, sieving 38.4 s → 36.5 s (5%), the same relations |

Tried and dropped:

- **Two large primes at 90 digits:** more relations, but a full matrix
  later (738 s against 559 s), before the 64-bit ECM splitter existed.
- **CADO-NFS's parameters at 90 digits:** 366 s to enough relations
  against 270 s with this table's larger factor bases. Below 364 bits
  the table still uses large factor bases with one large prime a side.
- **Three large primes:** supported, but below 600 bits they cost more
  per relation than they gain. At RSA-130 (factor bases 10M, 28/29 bits)
  an algebraic cofactor bound of 82 bits gave 14% more relations than 2LP
  with the quick P−1, and 49% more with 200 ECM curves per cofactor, the
  most any splitter could find. But the sieve and resieve alone took
  about 64 s a slice against 2LP's 36 s, so even free splitting loses.
  At RSA-140 (factor bases 20M, 29/29 bits) an algebraic bound of 87 bits
  took 1.6 times as long a slice for about the same relations: 264
  core-hours to enough relations against 174.
- **Batch cofactorization** (Bernstein's) with two large primes: on that
  slice the 2LP splits that fail cost 0.9 s of 38 s, all batch could save,
  and the successes still need splitting, since a gcd says a cofactor is
  smooth without giving its primes. Deferring the splits saved more.

**Parameters for 120 to 140 digits.** No trial factorization is quick
enough to compare settings at these sizes, so each candidate sieved
2000-wide slices of special-q from 250,000 to 32 million (128 million at
140 digits) on one thread, and the time to reach the relations needed
was added up, a relation counting `1/m` for the `m` algebraic primes it
has in the range sieved, since it is found again under each. That
predicted 1.6% duplicates from 2.0 to 2.1 million, against 1.2%
measured, and makes starting low cost more than it saves:

| | Best | Runner-up | Worst tried |
|---|---|---|---|
| RSA-120 | factor bases 5M, 27/28-bit large primes, `I` = 2^12, from 4M: about 10 core-hours | CADO-NFS's c120 parameters, within 4% | 8M, 28/28 bits, `I` = 2^13 (the old guess): about 20% more per relation |
| RSA-130 | factor bases 10M, 28/29 bits, `I` = 2^13, from 8M: about 37 core-hours | CADO-NFS's c130 parameters: 39 | 10M with 29/29 bits: 47 |
| RSA-140 | factor bases 20M, 29/29 bits, `I` = 2^13, from 32M: about 174 core-hours (30M: 172) | 20M with 28/29 bits: 178 | CADO-NFS's c140 parameters (30/30 bits): 244 |

The relations needed are `0.56·(π(2^lpb₀) + π(2^lpb₁))`, from the real
RSA-120 run, which needed 12.2 million; the RSA-120 and RSA-130 rows
were chosen with 0.66.

Real runs with these rows, on an M4 MacBook Air with 10 threads (4
performance and 6 efficiency cores):

| | RSA-120 | RSA-140 |
|---|---|---|
| Total | 2 h 03 min, with 3 min of polynomial search | 59.5 h, with the polynomial from the tuning (its search took 62 min while sharing the cores) |
| Sieving | 1 h 51 min, special-q 4M to 8.5M | 54.7 h, special-q 30.6M to 75.7M |
| Relations | 12.19M distinct of 13.48M | 39.2M distinct of 46.1M |
| Relations needed | 0.56 of the large primes | 0.63 to 0.70: 35.7M was too few |
| Filtering and matrix | 6.8 min | 4.7 h |
| Square roots | 15 s a dependency; the first 8 failed, by chance | 90 s a dependency; the second split |
| CPU time | 18.3 core-hours | 538 core-hours |
| Peak resident memory | 8.0 GB (5.8 GB after 8b4c806) | 14.8 GB |

RSA-140 took twice the model's 30 hours: it needed 13 to 24% more
relations than the RSA-120 ratio predicted, its 10 threads sieved at
about half the rate the one-thread slices implied, and the matrix took
hours rather than minutes. A profile of the running sieve put 40% of
the time in each special-q's prime setup and bucket fill, 37% in
sieving and 23% in resieving, with cofactorization about 1%. The
readiness check, after every 10% more relations, can also oversieve by
up to a tenth near the end.

The polynomial search's budget doubled every 20 bits, which took
RSA-130's search 31 minutes on 10 threads; past RSA-110 it now doubles every 30,
as CADO-NFS's does.

### Filtering

Duplicates and singletons are removed, then clique removal drops surplus
rows in groups linked by weight-two columns, and merging eliminates light
columns along a minimum spanning tree of their rows' sums, until rows
average 70 entries. At 90 digits the matrix shrinks from 776k × 732k to
186k × 186k.

The merge target was swept on RSA-110's relations:

| Target density | Matrix step | Peak memory |
|---|---|---|
| 50 | 655 s | 5.6 GB |
| **70 (kept)** | 558 s | 7.0 GB |
| 90 | 559 s | 7.3 GB |
| 120 | 557 s | 7.3 GB |
| 160 | 609 s | 8.1 GB |

### Linear algebra

Montgomery's block Lanczos, 64 vectors at a time, or 128 natively for
matrices of 20,000 columns or more, shared by both sieves.

| Change | Effect |
|---|---|
| Threads take pieces of work as they become free | Faster cores do more. 90 digits: 1690 s on one thread → 24 s on ten, with filtering |
| The 48 densest rows, quadratic characters among them, left out of the iterations | Satisfied in the final elimination instead |
| `B` stored once in 2048 × 2048 blocks of 16-bit offset pairs | Both products run block by block, in L1. 90 digits: 24 s → 17 s on ten threads, half the memory |
| Entries round-robin over each block's columns, and a by-column copy for `Bᵀw` | The small primes' rows put several entries in one column, and `Bᵀw` added them into one word in a chain, each waiting on the last. RSA-110: 27.5 → 18.4 ms a step; matrix step 931–963 s → 574 s. The copy is left out in wasm for memory |
| 128 vectors at a time in a NEON or SSE2 register, from 20,000 columns | Half the steps. A step's products cost about the same, 8.4 rather than 7.9 ms at 150,000 columns, but its vector work twice as much. One thread: 100,000 columns 6.8 → 5.6 s, 150,000 18.1 → 13.6 s; 10,000 columns 10% slower. RSA-120 on ten threads, two runs each on a busy machine: 372 and 420 s → 328 and 383 s, about 12%; its block vectors no longer fit in L2 |

Tried and dropped:

- **4096-row blocks:** 28% slower; 8192 was slower still, and stopped
  after 2 h 18 min.
- **Summing each column in a register in `Bᵀw`,** and **stripes** of
  columns: no gain.
- **128-bit words as `u128`:** the compiler splits each xor in two, and an
  entry of the products took 0.62 ns rather than 0.45.
- **4-bit tables for the 128-bit vector updates,** 8 KB rather than 64:
  faster alone, but a whole solve took 18 s rather than 14.

### Square root

The algebraic square root is found modulo an inert prime by
Tonelli–Shanks in `F_(p^d)`, lifted by Newton's method and checked
exactly.

| Change | Effect |
|---|---|
| One dependency at a time on all threads | Each gives a factor with probability about 1/2, so few are needed. 90 digits: 63 s → 25.6 s a square root on six threads |
| NTT multiplication and Barrett reduction | num-bigint multiplies two 64-million-bit numbers in 6 s and takes a remainder in 15. Products now go through number-theoretic transforms modulo two primes below 2⁶², remainders through Barrett's method with a Newton reciprocal, and the lifting goes up by powers of `p` ending just past the bound. RSA-110: 339 s → 84 s natively, 40 → 6 minutes in wasm |
| Products of polynomials transforming each coefficient once | Rather than both operands of every product of two coefficients: about 3 times fewer transforms. RSA-110's first dependency on 6 threads: 63 s → about 40 s |
| Batched Barrett reductions | The reciprocal and modulus transformed once for all of a polynomial's coefficients, and each coefficient and prime its own job: → about 33 s |
| Transforms in cache-sized blocks, each level's twiddle factors in order | Inside a block, each level read its twiddles at a stride across a table of `n/2`: → about 22.5 s |
| The rational side as one product tree | A modular power for each of hundreds of thousands of primes, on one thread, became words multiplied up on all threads and reduced once: → about 18 s |

On one thread the two primes go one after the other, which halves the
memory. In wasm, RSA-110's square root takes 195 s instead of 364 s, at
1.8 GB on its own instead of 1.4 GB, which still fits in the memory the
matrix step leaves: the whole run peaks at 3.6 GB, as before.

Tried and dropped:

- **Shoup's multiplication** by the twiddle factors, and **radix-4
  passes** over the large levels: no faster, as the transforms are
  limited by memory rather than arithmetic.
- **Both primes at once in products of polynomials:** about 2% faster
  natively, but 2.3 GB in wasm.

### Memory

| Change | Effect |
|---|---|
| Relations packed in one vector of words, pairs seen as 64-bit fingerprints | 90 digits: relations in 526 MB instead of 977 MB |
| Relations and matrix rows in fixed 16 MB chunks, pruning in passes, letting go of the duplicate check before filtering | RSA-110 fits the browser's 4 GB: 2.6 GB peak loading, 3.6 GB through the matrix and square roots |
| Each linear-algebra stage takes the last one's data instead of copying it; dependencies kept as one word per relation; square roots read only `(a, b)` and the rational primes from the store | RSA-120 from its saved relations: live heap peak 4.49 → 2.99 GB, macOS peak footprint 6.48 → 4.43 GB, RSS 8.41 → 5.76 GB. Block Lanczos went from 4.10 to 2.13 GB and the dependencies from 4.49 to 1.46 GB; sieving takes about 50 MB a thread |
| The command-line tool spills the relations to a temporary file once the quadratic characters are taken, before filtering, and reads them back for the square roots. Filtering frees dead rows as they go, rebuilds its column index after clique removal and whenever it grows to twice what it needs, and gives one-row sets no allocation | RSA-140 from its saved relations, live heap: filtering's peak 8.08 → 6.57 GB, with 4.9 GB of relations on disk. Loading, at 11.0 GB with the relations and matrix rows both held, is now the peak |

## The browser

The page runs the same library as wasm, with Web Workers in place of
threads. Natively the arithmetic is about 3 times as fast.

| Change | Effect |
|---|---|
| 32-bit limbs | See [arithmetic](#arithmetic) |
| A second build for the wide-arithmetic proposal | `i64.mul_wide_u` lets wasm use 64-bit limbs. In headless Chrome: ECM on RSA-100 7.75 s → 2.78 s; GNFS sieving 6.99 s → 5.40 s; the quadratic sieve 2%. The worker checks support with a 34-byte module and falls back |
| No by-column copy of the matrix | Saves memory; the round-robin order alone took `Bᵀw` from 27.5 to 21.4 ms a step |

## Ways of measuring that paid off

- **A fixed slice of special-q.** Sieving 3,000 special-q at a few points
  of RSA-110's range on one thread, with the relations' checksum,
  separates siever speed from yield, and catches changes that alter the
  relations found.
- **Comparing with CADO-NFS's tools.** Building `las`, `polyselect`,
  `polyselect_ropt` and `polyselect3` showed which stage was behind: the
  siever matched `las`; CADO-NFS's polynomial sieved 2.0–2.6 times as
  fast in our own siever; CADO-NFS's root optimizer on our finalists
  reached Murphy's E 5.74·10⁻⁶, showing the loss was in root
  optimization; and `polyselect3` scores polynomials on one scale,
  independent of the machine.
- **Feeding one stage's input to both programs.** Size-optimizing
  CADO-NFS's raw candidates with our code isolated size optimization
  from stage 1. (Run CADO-NFS's `polyselect` on one thread for this:
  with several, its output interleaves.)
- **Checking the machine.** Two things made timings wrong. A backup
  running alongside pushed the load average to 59. And a resumed Claude
  Code session ran at macOS's background priority (`ps -o pri` showed 4
  instead of 31), confined to the efficiency cores and about 4 times
  slower. Re-running a known slice first catches both.

## Not yet tried

- **The quadratic sieve against msiqs:** direct one-hit and two-hit loops
  for large primes, smaller factor bases, and an unrolled two-root loop.
- **Plain byte adds** in the GNFS small sieve, where the log sums can't
  overflow, instead of saturating ones.
- **The default thread count:** the efficiency cores add little to
  sieving, and 4 threads may do as well as 10.
- **Batch cofactorization with three large primes** past 130 digits, if
  three large primes start to pay there.
- **Further square-root work:** less padding to powers of 2, and keeping
  idle threads busy where a step has fewer jobs than threads.
