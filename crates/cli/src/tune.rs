//! `--tune`: a search for the GNFS sieve parameters that reach a matrix
//! soonest for one number, with its polynomials, on this machine.
//!
//! Each parameter set sieves short slices of special-q on all threads at
//! a few points around the table's first special-q. From each slice come
//! the time and the relations per unit of special-q, and a model adds
//! them up from a starting special-q until there are enough relations.
//! A relation is found again under each of its algebraic primes in the
//! range sieved, so it counts `1/m` for its `m` such primes, and the range
//! itself depends on the count, so the two are solved together. The
//! search starts from the table's row and moves one parameter at a time to
//! whichever neighbor the model rates best, until none is better by
//! [`MARGIN`]. Each round sieves the incumbent again with its neighbors.
//!
//! The large-prime and cofactor bounds stay the table's: the relations a
//! matrix needs are only roughly a fixed share of the large-prime ideals
//! (0.67 at RSA-120 with 27 and 27 bits, 0.55 with 27 and 28), too
//! roughly to compare different bounds, while with the same bounds the
//! error is common to every candidate.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::thread;
use std::time::Instant;

use cofox_core::nfs::{Gnfs, Params, Relation};

use crate::Wide;

/// Each slice is sized to take about this many seconds with the table's
/// parameters.
const SLICE_SECONDS: f64 = 10.0;

/// The points sampled, as multiples of the table's first special-q.
const POINTS: [f64; 4] = [0.5, 1.0, 2.0, 4.0];

/// The relations a matrix needs, as a fraction of the large-prime ideals
/// on both sides: 0.55 to 0.67 in real runs at 110 to 140 digits, with two
/// large primes a side. The candidates share their large-prime bounds,
/// so an error here moves the range they are compared over, not the bounds.
const NEEDED: f64 = 0.6;

/// A neighbor replaces the incumbent only if it's predicted to be this
/// much faster.
const MARGIN: f64 = 0.02;

/// Starting special-q tried between the first and last points.
const STARTS: usize = 24;

/// Steps of the integration over special-q, as a fraction of `q`.
const STEP: f64 = 0.005;

/// One slice of special-q, `[q, q + width)`: how long it took and, for
/// each relation found, its distinct algebraic primes from the first
/// point up, ascending.
struct Slice {
    q: u64,
    width: u64,
    seconds: f64,
    primes: Vec<Vec<u64>>,
}

/// What the model predicts for one parameter set.
#[derive(Clone, Copy, Debug)]
struct Prediction {
    seconds: f64,
    start: u64,
    end: u64,
    /// Relations a second over the slices, for the record.
    rate: f64,
}

/// Searches parameters for `gnfs`'s number and polynomials, starting from
/// its own, printing each round, and the best at the end.
///
/// The rounds sample at the same special-q, around the table's first
/// special-q at first, with slices sized by a trial slice at each point;
/// they move only when the incumbent's run would start at the lowest point
/// or end well past the highest, to half its start up to its end, sized
/// from its times. The same parameters measured in different rounds
/// varied by 5 to 15%, so the last step sieves the table's row and the
/// winner side by side, twice, in turn each way.
pub fn tune(gnfs: &Gnfs<16>, n: &Wide, threads: usize) {
    let row = *gnfs.params();
    let poly = gnfs.poly().clone();
    let build = |p: &Params| Gnfs::with_poly(n, p, poly.clone());
    let mut points: Vec<u64> = POINTS
        .iter()
        .map(|&k| ((row.qmin as f64 * k) as u64).max(1000))
        .collect();
    let mut widths: Vec<u64> = points
        .iter()
        .map(|&q| {
            let trial = least_width(q, threads);
            let s = slice(gnfs, q, trial, threads, points[0]);
            sized(q, s.seconds / trial as f64, threads)
        })
        .collect();
    let mut best = row;
    let mut tried: Vec<Params> = Vec::new();
    for round in 1.. {
        let mut candidates = vec![best];
        candidates.extend(neighbors(&best).filter(|p| !tried.contains(p)));
        let built: Vec<Gnfs<16>> = candidates.iter().map(build).collect();
        let slices = measure(&built, &points, &widths, threads);
        let predictions: Vec<Option<Prediction>> = candidates
            .iter()
            .zip(&slices)
            .map(|(p, s)| predict(s, needed(p)))
            .collect();
        println!(
            "Round {round}, on {threads} threads, special-q {}:",
            points
                .iter()
                .zip(&widths)
                .map(|(q, w)| format!("{q} (+{w})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for (p, prediction) in candidates.iter().zip(&predictions) {
            println!("  {}  {}", describe(p), show(prediction));
        }
        tried.extend(candidates.iter().copied());
        let incumbent = predictions[0].map_or(f64::INFINITY, |x| x.seconds);
        let winner = predictions
            .iter()
            .enumerate()
            .filter_map(|(i, x)| Some((i, (*x)?.seconds)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let next = match winner {
            Some((i, seconds)) if i != 0 && seconds < incumbent * (1.0 - MARGIN) => i,
            _ => 0,
        };
        let Some(prediction) = predictions[next] else {
            println!("No parameters tried get enough relations within the special-q sampled");
            return;
        };
        best = candidates[next];
        let last = points[points.len() - 1];
        let replace = (prediction.start <= points[0] + points[0] / 20 && points[0] > 1000)
            || prediction.end > last + last / 2;
        if next == 0 && !replace {
            break;
        }
        if next == 0 {
            // Its neighbors were tried; sample again for it alone.
            tried.retain(|p| *p != best);
        }
        if replace {
            let (s, e) = (prediction.start as f64 / 2.0, prediction.end as f64);
            let old = &slices[next];
            let ln: Vec<f64> = old.iter().map(|x| (x.q as f64).ln()).collect();
            let time: Vec<f64> = old.iter().map(|x| x.seconds / x.width as f64).collect();
            points = (0..POINTS.len())
                .map(|k| {
                    let q = s * (e / s).powf(k as f64 / (POINTS.len() - 1) as f64);
                    (q as u64).max(1000)
                })
                .collect();
            points.dedup();
            widths = points
                .iter()
                .map(|&q| sized(q, interpolate(&ln, &time, (q as f64).ln()), threads))
                .collect();
        }
    }
    // The table's row against the winner, side by side.
    let pair = [row, best];
    let built = [build(&row), build(&best)];
    let mut seconds = [0.0; 2];
    let mut chosen = None;
    for order in [[0, 1], [1, 0]] {
        let mut slices: [Vec<Slice>; 2] = [Vec::new(), Vec::new()];
        for (&q, &width) in points.iter().zip(&widths) {
            for i in order {
                slices[i].push(slice(&built[i], q, width, threads, points[0]));
            }
        }
        for i in 0..2 {
            let prediction = predict(&slices[i], needed(&pair[i]));
            seconds[i] += prediction.map_or(f64::INFINITY, |x| x.seconds);
            if i == 1 {
                chosen = chosen.or(prediction);
            }
        }
    }
    let Some(prediction) = chosen else {
        println!("The best parameters didn't get enough relations side by side");
        return;
    };
    println!(
        "Side by side, twice: the table's row {}, the best {} ({:+.1}%)",
        duration(seconds[0] / 2.0),
        duration(seconds[1] / 2.0),
        (seconds[1] / seconds[0] - 1.0) * 100.0
    );
    best.qmin = prediction.start;
    println!("Best: {}  {}", describe(&best), show(&Some(prediction)));
    println!(
        "As a table row: ({}, [{}, {}], [{}, {}], [{}, {}], {}, {}, {}),",
        n.bits(),
        best.rational_bound,
        best.algebraic_bound,
        best.rational_lpb,
        best.algebraic_lpb,
        best.rational_mfb,
        best.algebraic_mfb,
        best.log_i,
        round_q(prediction.start),
        best.algebraic_slack
    );
}

/// Each of `built` sieves a slice at each point, slice by slice, each in
/// turn, so that the machine warming up doesn't favor whichever ran first.
fn measure(built: &[Gnfs<16>], points: &[u64], widths: &[u64], threads: usize) -> Vec<Vec<Slice>> {
    let mut slices: Vec<Vec<Slice>> = built.iter().map(|_| Vec::new()).collect();
    for (&q, &width) in points.iter().zip(widths) {
        for (g, out) in built.iter().zip(&mut slices) {
            out.push(slice(g, q, width, threads, points[0]));
        }
    }
    slices
}

/// Enough special-q near `q` for every thread several times over.
fn least_width(q: u64, threads: usize) -> u64 {
    ((8 * threads) as f64 * (q as f64).ln()) as u64
}

/// A slice width near `q` taking about [`SLICE_SECONDS`] at `per_q`
/// seconds a unit of special-q.
fn sized(q: u64, per_q: f64, threads: usize) -> u64 {
    let least = least_width(q, threads);
    ((SLICE_SECONDS / per_q.max(1e-12)) as u64).clamp(least, q.max(least))
}

/// `ys` at `x`, linear between the points `xs` and along the nearest
/// segment beyond them.
fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.len() < 2 {
        return ys[0];
    }
    let i = xs.partition_point(|&l| l < x).clamp(1, xs.len() - 1);
    let t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
    ys[i - 1] + (ys[i] - ys[i - 1]) * t
}

/// The parameter sets one step from `p`: factor bases half as large again
/// or two thirds the size, each large-prime bound a bit either way with its
/// cofactor bound following, each cofactor bound two bits either way, `I`
/// either way, and the algebraic slack 4 bits either way.
fn neighbors(p: &Params) -> impl Iterator<Item = Params> {
    let p = *p;
    let mut out = Vec::new();
    for scale in [1.5, 1.0 / 1.5] {
        let mut x = p;
        x.rational_bound = (f64::from(p.rational_bound) * scale) as u32;
        x.algebraic_bound = (f64::from(p.algebraic_bound) * scale) as u32;
        out.push(x);
    }
    for d in [1i32, -1] {
        let mut x = p;
        x.log_i = p.log_i.saturating_add_signed(d);
        out.push(x);
        let mut x = p;
        x.algebraic_slack = p.algebraic_slack.saturating_add_signed(4 * d);
        out.push(x);
    }
    out.into_iter().filter(move |x| *x != p && valid(x))
}

/// Whether the siever takes `p`, within the ranges worth trying.
fn valid(p: &Params) -> bool {
    let side = |bound: u32, lpb: u32, mfb: u32| {
        bound >= 1000
            && (16..=40).contains(&lpb)
            && u64::from(bound) < 1 << lpb
            && (lpb..=(3 * lpb).min(127)).contains(&mfb)
    };
    side(p.rational_bound, p.rational_lpb, p.rational_mfb)
        && side(p.algebraic_bound, p.algebraic_lpb, p.algebraic_mfb)
        && (9..=13).contains(&p.log_i)
        && p.algebraic_slack <= 16
}

/// Sieves the special-q in `[q, q + width)` on `threads` threads, timing
/// only the sieving, not setting up the sieves.
fn slice(gnfs: &Gnfs<16>, q: u64, width: u64, threads: usize, floor: u64) -> Slice {
    let qs = gnfs.special_qs_between(q, q + width);
    let next = AtomicUsize::new(0);
    let barrier = Barrier::new(threads + 1);
    let found: Mutex<Vec<Relation>> = Mutex::new(Vec::new());
    let seconds = thread::scope(|scope| {
        for _ in 0..threads {
            let (qs, next, barrier, found) = (&qs, &next, &barrier, &found);
            scope.spawn(move || {
                let mut sieve = gnfs.sieve();
                let mut mine = Vec::new();
                barrier.wait();
                while let Some(&(q, r)) = qs.get(next.fetch_add(1, Ordering::Relaxed)) {
                    mine.extend(sieve.special_q(q, r));
                }
                found.lock().unwrap().extend(mine);
            });
        }
        barrier.wait();
        let start = Instant::now();
        // The scope joins the threads before returning.
        start
    })
    .elapsed()
    .as_secs_f64();
    let primes = found
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|rel| {
            let mut p: Vec<u64> = rel.algebraic.into_iter().filter(|&p| p >= floor).collect();
            p.sort_unstable();
            p.dedup();
            p
        })
        .collect();
    Slice {
        q,
        width,
        seconds,
        primes,
    }
}

/// The relations a matrix needs with `p`'s large-prime bounds.
fn needed(p: &Params) -> f64 {
    NEEDED * (pi(exp2(p.rational_lpb)) + pi(exp2(p.algebraic_lpb)))
}

fn exp2(bits: u32) -> f64 {
    2f64.powi(bits as i32)
}

/// The number of primes up to `x`, near enough.
fn pi(x: f64) -> f64 {
    let l = x.ln();
    x / l * (1.0 + 1.0 / l + 2.0 / (l * l))
}

/// The fastest way to `needed` relations from the slices, over starting
/// special-q between the first and last points; `None` if none gets there
/// within 64 times the last point.
fn predict(slices: &[Slice], needed: f64) -> Option<Prediction> {
    let (first, last) = (slices[0].q as f64, slices[slices.len() - 1].q as f64);
    let rate = slices.iter().map(|s| s.primes.len() as f64).sum::<f64>()
        / slices.iter().map(|s| s.seconds).sum::<f64>();
    let ln: Vec<f64> = slices.iter().map(|s| (s.q as f64).ln()).collect();
    let time: Vec<f64> = slices.iter().map(|s| s.seconds / s.width as f64).collect();
    (0..STARTS)
        .filter_map(|k| {
            let start = first * (last / first).powf(k as f64 / (STARTS - 1) as f64);
            // The end depends on the duplicates, which depend on the end.
            let mut end = last;
            let mut seconds = f64::INFINITY;
            for _ in 0..4 {
                let distinct: Vec<f64> = slices.iter().map(|s| density(s, start, end)).collect();
                (seconds, end) = integrate(&ln, &distinct, &time, start, needed, last * 64.0)?;
            }
            Some(Prediction {
                seconds,
                start: start as u64,
                end: end as u64,
                rate,
            })
        })
        .min_by(|a, b| a.seconds.total_cmp(&b.seconds))
}

/// Distinct relations per unit of special-q in `s`, for a run sieving
/// from `start` to `end`: each counts `1/m` for its `m` algebraic primes
/// in that range, as it's found under each.
fn density(s: &Slice, start: f64, end: f64) -> f64 {
    s.primes
        .iter()
        .map(|p| {
            let m = p
                .iter()
                .filter(|&&p| (start..end).contains(&(p as f64)))
                .count();
            1.0 / m.max(1) as f64
        })
        .sum::<f64>()
        / s.width as f64
}

/// Adds up relations and time over special-q from `start` until there are
/// `needed`, with each density interpolated linearly in `ln q` between the
/// points and extended along the nearest segment beyond them.
fn integrate(
    ln: &[f64],
    relations: &[f64],
    time: &[f64],
    start: f64,
    needed: f64,
    limit: f64,
) -> Option<(f64, f64)> {
    let at = |ys: &[f64], x: f64| interpolate(ln, ys, x);
    let (mut q, mut found, mut seconds) = (start, 0.0, 0.0);
    while found < needed {
        if q > limit {
            return None;
        }
        let (x, dq) = (q.ln(), q * STEP);
        found += at(relations, x).max(0.0) * dq;
        seconds += at(time, x).max(time.iter().copied().fold(f64::INFINITY, f64::min) / 4.0) * dq;
        q += dq;
    }
    Some((seconds, q))
}

/// A special-q rounded to two significant figures.
fn round_q(q: u64) -> u64 {
    let unit = 10u64.pow((q.max(1).ilog10()).saturating_sub(1));
    (q + unit / 2) / unit * unit
}

fn describe(p: &Params) -> String {
    format!(
        "fb {}/{} lpb {}/{} mfb {}/{} I 2^{} slack {}",
        p.rational_bound,
        p.algebraic_bound,
        p.rational_lpb,
        p.algebraic_lpb,
        p.rational_mfb,
        p.algebraic_mfb,
        p.log_i,
        p.algebraic_slack
    )
}

fn show(prediction: &Option<Prediction>) -> String {
    match prediction {
        Some(x) => format!(
            "{} sieving, special-q {} to {}, {:.0} relations a second",
            duration(x.seconds),
            round_q(x.start),
            round_q(x.end),
            x.rate
        ),
        None => "doesn't get there".to_string(),
    }
}

fn duration(seconds: f64) -> String {
    if seconds < 3600.0 {
        format!("{:.1} min", seconds / 60.0)
    } else {
        format!("{:.2} h", seconds / 3600.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_primes() {
        // π(10⁶) = 78,498 and π(2²⁷) = 7,603,553.
        assert!((pi(1e6) / 78_498.0 - 1.0).abs() < 0.01);
        assert!((pi(exp2(27)) / 7_603_553.0 - 1.0).abs() < 0.01);
    }

    #[test]
    fn integrates_flat_densities() {
        // 2 relations and 0.5 s per unit of special-q: 10⁶ relations take
        // 500,000 units and 250,000 s, but for the last step, 0.5% of q.
        let ln = [1e6f64.ln(), 2e6f64.ln()];
        let (seconds, end) = integrate(&ln, &[2.0, 2.0], &[0.5, 0.5], 1e6, 1e6, 1e9).unwrap();
        assert!((seconds / 250_000.0 - 1.0).abs() < 0.02, "{seconds}");
        assert!((end / 1_500_000.0 - 1.0).abs() < 0.02, "{end}");
        assert!(integrate(&ln, &[0.0, 0.0], &[0.5, 0.5], 1e6, 1000.0, 1e7).is_none());
    }

    #[test]
    fn counts_duplicates_once() {
        // Found under both its primes in range, a relation counts a half;
        // with one of them below the start, it counts once.
        let s = Slice {
            q: 1_000_000,
            width: 100,
            seconds: 1.0,
            primes: vec![vec![1_000_003, 1_500_007]; 200],
        };
        assert_eq!(density(&s, 1e6, 2e6), 1.0);
        assert_eq!(density(&s, 1.2e6, 2e6), 2.0);
        assert_eq!(density(&s, 1e6, 1.2e6), 2.0);
    }

    #[test]
    fn neighbors_are_valid() {
        for bits in [300, 364, 397, 430, 465] {
            let p = Params::for_bits(bits);
            let near: Vec<Params> = neighbors(&p).collect();
            assert!(near.len() >= 4, "{bits}: {}", near.len());
            assert!(near.iter().all(|x| valid(x) && *x != p));
        }
    }
}
