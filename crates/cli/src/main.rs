use std::io::{IsTerminal, Write};
use std::ops::ControlFlow;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use cofox_core::arith::ParseUintError;
use cofox_core::{Config, Factor, Progress, factor_str_with};

const USAGE: &str = "usage: cofox [--ecm DIGITS] [--threads N] <n>...

  --ecm DIGITS  run ECM up to the level for primes of this many digits
                (15 to 40, default 25); each level takes about four times
                as long as the one before
  --threads N   split ECM's curves across N threads (default: one per core)";

/// How often the status line is redrawn.
const REDRAW: Duration = Duration::from_millis(100);

fn main() -> ExitCode {
    let mut ecm_digits = 25;
    let mut threads = thread::available_parallelism().map_or(1, |n| n.get());
    let mut numbers = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name| {
            let value = args.next().and_then(|v| v.parse().ok());
            if value.is_none() {
                eprintln!("cofox: {name} needs a number\n\n{USAGE}");
            }
            value
        };
        match arg.as_str() {
            "--ecm" => match value("--ecm") {
                Some(digits) => ecm_digits = digits,
                None => return ExitCode::FAILURE,
            },
            "--threads" => match value("--threads") {
                Some(0) => {
                    eprintln!("cofox: --threads needs at least 1");
                    return ExitCode::FAILURE;
                }
                Some(n) => threads = n as usize,
                None => return ExitCode::FAILURE,
            },
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            _ => numbers.push(arg),
        }
    }
    if numbers.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    }

    let show_status = std::io::stderr().is_terminal();
    let mut status = ExitCode::SUCCESS;
    let mut unsplit = false;
    for arg in &numbers {
        match factor_parallel(arg, ecm_digits, threads, show_status) {
            Ok(factors) => {
                unsplit |= factors.iter().any(|f| !f.is_prime);
                let factors: Vec<String> = factors.iter().map(ToString::to_string).collect();
                println!("{arg}: {}", factors.join(" "));
            }
            Err(e) => {
                eprintln!("cofox: invalid number {arg:?}: {e}");
                status = ExitCode::FAILURE;
            }
        }
    }
    if unsplit {
        eprintln!(
            "cofox: factors in parentheses are composite but could not be split \
             (--ecm {} would search for larger factors)",
            ecm_digits + 5
        );
    }
    status
}

/// Runs the pipeline on `threads` threads, each with its share of the ECM
/// curves. The first run to end with every factor prime stops the others.
/// If none does, the run that split the most wins.
fn factor_parallel(
    n: &str,
    ecm_digits: u32,
    threads: usize,
    show_status: bool,
) -> Result<Vec<Factor<String>>, ParseUintError> {
    let done = AtomicBool::new(false);
    let status = Mutex::new(Status::new(threads));
    let results: Vec<_> = thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|worker| {
                let (done, status) = (&done, &status);
                scope.spawn(move || {
                    let config = Config {
                        ecm_digits,
                        worker: worker as u32,
                        workers: threads as u32,
                    };
                    let result = factor_str_with(n, &config, |event| {
                        if done.load(Ordering::Relaxed) {
                            return ControlFlow::Break(());
                        }
                        if show_status {
                            status.lock().unwrap().update(worker, event);
                        }
                        ControlFlow::Continue(())
                    });
                    if result.as_ref().is_ok_and(|f| f.iter().all(|f| f.is_prime)) {
                        done.store(true, Ordering::Relaxed);
                    }
                    result
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    if show_status {
        status.lock().unwrap().clear();
    }
    let mut results = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    let best = (0..results.len())
        .max_by_key(|&i| {
            let f = &results[i];
            (f.iter().all(|f| f.is_prime), f.len())
        })
        .unwrap();
    Ok(results.swap_remove(best))
}

/// The latest event from each thread, drawn as one line on stderr.
struct Status {
    events: Vec<Option<Progress>>,
    start: Instant,
    drawn: Option<Instant>,
}

impl Status {
    fn new(threads: usize) -> Self {
        Self {
            events: vec![None; threads],
            start: Instant::now(),
            drawn: None,
        }
    }

    fn update(&mut self, worker: usize, event: Progress) {
        self.events[worker] = Some(event);
        // Stay quiet for quick factorizations.
        if self.start.elapsed() < Duration::from_millis(300)
            || self.drawn.is_some_and(|t| t.elapsed() < REDRAW)
        {
            return;
        }
        self.drawn = Some(Instant::now());
        let line = describe(&self.events);
        let secs = self.start.elapsed().as_secs_f64();
        eprint!("\r\x1b[K{line} ({secs:.1} s)");
        let _ = std::io::stderr().flush();
    }

    fn clear(&self) {
        if self.drawn.is_some() {
            eprint!("\r\x1b[K");
        }
    }
}

/// Summarizes the threads' progress: during ECM, the curves done at the
/// highest level any thread has reached, out of the curves of the threads
/// at that level; otherwise the first thread's method.
fn describe(events: &[Option<Progress>]) -> String {
    let ecm = events.iter().flatten().filter_map(|e| match *e {
        Progress::Ecm {
            bits,
            digits,
            done,
            total,
        } => Some((digits, bits, done, total)),
        Progress::Method { .. } => None,
    });
    if let Some((digits, bits, _, _)) = ecm.clone().max() {
        let (done, total) = ecm
            .filter(|e| (e.0, e.1) == (digits, bits))
            .fold((0, 0), |(d, t), e| (d + e.2, t + e.3));
        return format!(
            "ECM, {digits}-digit level: {done}/{total} curves on a {bits}-bit composite"
        );
    }
    match events.iter().flatten().next() {
        Some(Progress::Method { method, bits }) => format!("{method} on a {bits}-bit composite"),
        _ => String::new(),
    }
}
