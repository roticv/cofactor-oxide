# Ideas

What to try next, and what not to, with the reasons. Where
[optimizations.md](optimizations.md) records what was done and measured,
this records what hasn't been. Each idea worth trying comes with the
first experiment that would tell whether it pays, so that none needs a
large build before it can be judged.

The yardstick throughout is RSA-896, the 270-digit challenge number
factored on September 19, 2026 by Stephen Weis with a GPU port of
CADO-NFS, two weeks after Eric Lu's RSA-260. Neither claimed an
algorithmic advance. Its parameters show what a run at that size needs,
and where this code's limits sit against them.

## Contents

- [The record, and where the code stops](#the-record-and-where-the-code-stops)
- [Worth trying: algorithmic questions](#worth-trying-algorithmic-questions)
- [Worth trying: engineering](#worth-trying-engineering)
- [Where an individual can set records](#where-an-individual-can-set-records)
- [Not worth trying](#not-worth-trying)
- [On an algorithmic breakthrough](#on-an-algorithmic-breakthrough)

## The record, and where the code stops

From [Weis's write-up](https://saweis.net/posts/rsa-896.html):

| Stage | RSA-896 | This code's limit |
|---|---|---|
| Polynomial | Degree 6, α −11.12, Murphy's E 5.3·10⁻¹⁰, 14.9 h on 128 nodes | Selection stops at about 480 bits (`m` in 96 bits, coefficients in `i128`); `--poly` reads degree 6 with 256-bit coefficients |
| Factor bases | 2³² − 1 both sides | `Prime.p` is a `u32`: exactly the ceiling |
| Large primes | 37 bits rational, 40 algebraic | 40 is the siever's cap, where the character primes start |
| Cofactors | 74 bits (two primes) and 120 (three) | 128-bit cofactor type fits; the splitter past 2⁶² runs one P−1 and no ECM curves |
| Cofactor splitting | ECM on the GPU, B1 = 500, B2 = 25,000, 35 to 100 curves | 64-bit ECM to 2⁴⁴, rho below, P−1 above |
| Sieve region | `A = 33`: 2¹⁷ by 2¹⁶ per special-q | `I` stops at 2¹³, a 2²⁵ region, where one level of buckets takes hundreds of MB a thread: 256 times smaller |
| Special-q | 6.0·10⁹ to 4.9·10¹⁰, 1.77 billion of them, 21 relations each | `u64`, fine |
| Relations | 37.8 G found, 29.1 G unique, 23% duplicates removed in the siever | Everything in one process's memory: a packed store, a 64-bit fingerprint table, a `BTreeMap` of large ideals |
| Matrix columns | 20.1 G before purge | Column indices are `u32`: five times over |
| Matrix | 1.48 G rows and columns, 296.5 G nonzeros after merging to 200 a row, the 32 heaviest columns held out dense | Single-node block Lanczos in memory, no checkpoint; the densest-rows trick is already here |
| Linear algebra | Block Wiedemann, 56 sequences of width 64, 827,234 iterations, 99 h | — |
| Compute | 10 days wall, about 178,000 H100-hours | Weis's CPU reference: 163 to 334 core-seconds per special-q with stock `las`, so 10,000 to 19,000 core-years of sieving |

So a 270-digit number is a GPU fleet or a very large cluster, whatever
the code's quality. On one machine the ceiling is around 150 to 170
digits, and the items below are ordered with that in mind: what makes
the code able to run at the record's parameters, what makes it
distributable, and what makes it faster per core.

## Worth trying: algorithmic questions

Open questions where this codebase is a usable test bed, each with a
cheap first experiment.

### Nonlinear polynomial pairs

Two polynomials of degrees 3 and 3, or 2 and 4, have smaller combined
norms than degree 6 against linear once numbers pass about 200 digits,
on paper. Montgomery's two-quadratics construction works but gives poor
polynomials, Prest and Zimmermann's 2010 search did better without
winning, and no selection method has beaten linear pairs in practice.
Using them would change more than the rational norm: the rational side
would need ideals and roots in the factor base, quadratic characters,
and a second algebraic square root.

*Measured, and not worth it below about 200 digits.* A Python
experiment built pairs with Coxon's algorithms ([arXiv
1109.6398](https://arxiv.org/abs/1109.6398)): his Algorithm 4.5 for
cubics from a progression of length 4, his Algorithm 5.2 (Koo, Jo and
Kwon's length 5, with Kleinjung-style collisions for `q² | m³ − kN`),
and Montgomery's two quadratics. It reproduced Coxon's c91 example
exactly and scored every pair with a copy of `score.rs`'s Murphy's E,
α computed the same way for both kinds:

| | Our linear pair | Best (3,3) | Best (2,2) | Linear ahead by |
|---|---|---|---|---|
| RSA-100 (degree 4) | 5.6·10⁻⁶ | 2.0·10⁻⁷ | 4.5·10⁻⁷ | 12× |
| RSA-120 (degree 5) | 1.3·10⁻⁶ | 4.4·10⁻⁸ | 3.6·10⁻⁸ | 29× |
| RSA-130 (degree 5) | 3.9·10⁻⁷ | 7.4·10⁻⁹ | 1.2·10⁻⁸ | 33× |

Algorithm 5.2 reaches the optimal `N^(1/6)` cubics only with `p` near
`N^(2/9)` and `|m − ∛(kN)| ≤ p^1.5`; collisions found `p` of 10⁴ to 10⁶.
And the gap isn't only the construction: the pair `x³ ∓ S` that meets
the resultant bound exactly, which no real pair can beat, scores 0.22,
0.26 and 0.32 times our linear E at α 0, and only ties at RSA-130 with
α −4 on both sides, as good as the root sieve gets on one. The ratio
grows with size, so the crossover on paper past 200 digits stands, but
only for pairs nobody can construct.

### Predicting yield per special-q

In the record, unique relations per special-q fell from 32 at
`q = 6·10⁹` to 11 at `5.5·10¹⁰`, and 23% of relations were duplicates
from overlapping lattices. If yield per `q` is predictable from cheap
features of its reduced basis, skipping poor ones or reordering saves
sieve time directly.

*First experiment:* log relations per special-q at RSA-120 with the
basis skewness, the basis lengths and `q`, and fit the variance.
Anything over a third predictable is worth a scheduler.

### Composite and rational-side special-q

CADO-NFS sieves special-q that are products of two primes, and on the
rational side, which widens the pool of good `q` at the low end where
yield is highest and delays the climb to large, poor `q`.

*First experiment:* let `special_qs` return `q = q₁·q₂` with roots
combined by the Chinese remainder theorem, and compare relations per
sieve-second against prime `q` in the same range at RSA-120.

### Three large primes and batch cofactorization past 130 digits

The docs show three large primes losing at RSA-130 because the sieve
dominates, and batch cofactorization unable to pay with two. The record
used three on the algebraic side at 120-bit cofactors with 35 to 100 ECM
curves. Where the crossover sits for this siever, and whether
Bernstein's product trees move it, is unmeasured.

*First experiment:* the slice method from the parameter tuning at 150
digits, with `mfb` up to 100 on the algebraic side and 30 ECM curves,
against the best two-large-prime row.

### Automated parameter search

Every tuning pass so far found 5 to 20% by hand, sieving fixed slices.
A model fitting time-to-matrix as a function of factor-base bound,
large-prime bits, `I` and `qmin`, searched automatically, would find
those gains at every size and keep the table honest past 430 bits,
where its rows are guesses.

*First experiment:* script the slice sieve and the duplicate-aware
relation count, fit a response surface at RSA-120 where the answer is
known, and check that it recovers the chosen row.

## Worth trying: engineering

Known to pay; the question is only the order.

### A GPU siever

What both September records did, and neither released. A Rust port on
Metal or wgpu, with the CPU siever as the oracle for the relations found,
is the single most valuable thing this project could produce. The place
to start is the cofactorization kernel: it is embarrassingly parallel,
the GPU ECM experiment in `experiments/gpu-ecm` already does stage 1 on
the M4, and it lost to the CPU only below about 8,000 curves at once,
which batching cofactors across many special-q supplies. The lattice
walk and bucket fill come second.

### Wider types

`Prime.p` and the matrix's column indices to 64 bits, and `I` past 2¹³.
The record's factor base is `2³² − 1` and its column count 20 billion.
None of the rest can be tested at scale until these move.

### Disk-based filtering

Duplicate removal by hashing `(a, b)` into files by range, singleton
removal by bit arrays indexed by prime, clique removal and merging in
passes over files. The in-memory `Seen`, `Store` and `Matrix` cap the
run by RAM before anything else does. CADO-NFS's `dup1`, `dup2`, `purge`
and `merge` are the reference design.

### Two-level bucket sieving

With a hundred million factor-base primes a side, one bucket per slab no
longer fits in cache. CADO-NFS fills coarse buckets over the whole
region first and fine ones from them, which is what makes `I = 2¹⁶`
practical.

### Block Wiedemann with checkpoints

Needed past about 150 digits, where the solve takes days: it lets the
Krylov sequences run on separate machines and survive a crash. The
sparse product is the one block Lanczos already has. Lingen is the hard
part, and the CADO-NFS papers on it are thorough. Checkpointing the
current block Lanczos is the smaller first step and is worth doing on
its own.

### Distributed relation collection

`--special-q` ranges and `--load-relations` work for a handful of
machines. Past that: a server handing out ranges and retiring them,
binary or compressed relation files, on-the-fly duplicate removal in the
siever as the record did, and streaming import so merging shares never
loads everything at once.

## Where an individual can set records

Not RSA challenge numbers, at this point. But:

- **Special number field sieve targets.** Mersenne, Fibonacci and
  Cunningham-table numbers come with polynomials for free, and the
  community factors them with ggnfs and CADO-NFS on volunteer hardware.
  A GPU siever two or three times faster than `las` would set real
  records there, at sizes where the matrix still fits one machine.
- **ECM records.** Factors of 80 to 85 digits found by ECM are record
  territory, and the bottleneck is stage-1 throughput: millions of
  curves at `B1` around 10⁹. That is what the GPU ECM experiment is good
  at once the curve count is large.

## Not worth trying

- **Schnorr's lattice factoring.** The 1991–2021 method, finding smooth
  relations as close vectors in a prime-number lattice, rests on a
  heuristic about how often close vectors give relations. Ducas's Sage
  implementation of the 2021 paper measured that rate far below the
  claim and falling with dimension; honest analyses put the method at
  `L(1/2)` at best if its heuristics held. The 2022 "372 qubits break
  RSA-2048" paper inherited the same flaw. Reimplementing it would
  reproduce a plot that exists. Running Ducas's code is an afternoon, if
  the failure needs to be seen first-hand.
- **Quantum-inspired, annealing or optimizer-based factoring.** Every
  published instance factors numbers the quadratic sieve does in
  microseconds, and none has a scaling argument.
- **Degree 7 and up.** Theory puts the switch past about 300 digits,
  and the sieve regions involved are out of reach anyway.
- **ECM on the GPU for the pipeline's own levels** below 35 digits:
  tried, and the curve counts are too small to win. It is for record
  attempts and for cofactoring, not for the default pipeline.
- **The factorization factory.** Coppersmith's shared-side NFS reaches
  `L(1/3, 1.639)` amortized over many numbers of one size. A genuine
  asymptotic win, and irrelevant unless there is a batch of moduli to
  factor.

## On an algorithmic breakthrough

GNFS has been the best general algorithm since 1993, at
`L(1/3, 1.923)`; Coppersmith's multiple-polynomial variant lowers the
constant to 1.902 and has never been worth running. The exponent has not
moved since 1990, through continuous effort: nonlinear and tower
variants, Bernstein's batch methods, Schnorr's lattices, and the
quasi-polynomial function-field results of 2013 that broke small
characteristic discrete logarithms but have no integer analogue.

There is no proof it is hard. Factoring is in NP ∩ co-NP, almost
certainly not NP-hard, with no known lower bound; Shor's algorithm shows
the problem has structure a machine can exploit. So the honest position
is "no known idea", not "impossible". Every index-calculus method pays
for the size of the numbers it tests for smoothness, and `L(1/3)` is
what degree-`d` number fields buy; beating it needs a ring with two maps
to `ℤ/Nℤ` and smaller elements, and nobody knows how to build one.

For factoring a given number, none of this is a plan. The levers that
moved the record were polynomial quality, three large primes with real
ECM, a 2³³ region, duplicate removal in the siever, and a GPU fleet.
Nonlinear polynomial pairs, the one algorithmic direction where this
code was a useful instrument, were measured above and lose by more than
an order of magnitude at the sizes it can reach.
