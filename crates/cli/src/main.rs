use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, IsTerminal, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use cofox_core::arith::ParseUintError;
use cofox_core::nfs::{Gnfs, Params, parse_poly, parse_relation, poly_text, relation_line};
use cofox_core::parallel::Parallel;
use cofox_core::qs::{QsJob, Start};
use cofox_core::{Config, Factor, Progress, QS_MAX_BITS, QsMode, Uint, factor_str_with};

const USAGE: &str = "usage: cofox [--ecm DIGITS] [--threads N] [--no-qs] [--gnfs]
             [--save-relations FILE] [--load-relations FILE] <n>...

  --ecm DIGITS  run ECM up to the level for primes of this many digits
                (15 to 40, default 25); each level takes about four times
                as long as the one before. Before the quadratic sieve, ECM
                stops sooner, at about 30% of the composite's digits
  --threads N   split ECM's curves and the sieving across N threads
                (default: one per core)
  --no-qs       leave composites that ECM can't split, instead of running
                the quadratic sieve on those of up to 120 digits
  --gnfs        use the number field sieve instead of the quadratic sieve,
                on composites of 40 digits or more (about as fast as the
                quadratic sieve at 100 digits, and faster beyond)
  --save-relations FILE
                with --gnfs, append the relations found to FILE, in the
                format msieve, GGNFS and CADO-NFS use, and write the
                polynomials to FILE.poly
  --load-relations FILE
                with --gnfs, start from the relations in FILE, with the
                polynomials in FILE.poly if there is one, skipping the
                special-q it records as done. With --save-relations to the
                same FILE, this resumes a run";

/// The widest numbers the command line takes.
type Wide = Uint<16>;

/// The smallest composite `--gnfs` takes; the quadratic sieve takes the
/// rest.
const GNFS_MIN_BITS: u32 = 130;

/// How often relations being saved are flushed to the file.
const SAVE_FLUSH: Duration = Duration::from_secs(10);

/// How often the status line is redrawn.
const REDRAW: Duration = Duration::from_millis(100);

fn main() -> ExitCode {
    let mut ecm_digits = 25;
    let mut qs = true;
    let mut gnfs = false;
    let mut threads = thread::available_parallelism().map_or(1, |n| n.get());
    let mut files = Files::default();
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
            "--no-qs" => qs = false,
            "--gnfs" => gnfs = true,
            "--save-relations" | "--load-relations" => {
                let Some(path) = args.next().map(PathBuf::from) else {
                    eprintln!("cofox: {arg} needs a file\n\n{USAGE}");
                    return ExitCode::FAILURE;
                };
                if arg == "--save-relations" {
                    files.save = Some(path);
                } else {
                    files.load = Some(path);
                }
            }
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
    if let Some(path) = &files.load
        && let Err(e) = File::open(path)
    {
        eprintln!("cofox: can't read {}: {e}", path.display());
        return ExitCode::FAILURE;
    }
    if let Some(path) = &files.save
        && let Err(e) = OpenOptions::new().create(true).append(true).open(path)
    {
        eprintln!("cofox: can't write {}: {e}", path.display());
        return ExitCode::FAILURE;
    }

    let show_status = std::io::stderr().is_terminal();
    let mut status = ExitCode::SUCCESS;
    let mut unsplit = false;
    for arg in &numbers {
        let run = Run {
            ecm_digits,
            threads,
            qs,
            gnfs,
            files: files.clone(),
            show_status,
        };
        match run.factor(arg) {
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

/// Where GNFS saves its relations, and loads them from.
#[derive(Clone, Debug, Default)]
struct Files {
    save: Option<PathBuf>,
    load: Option<PathBuf>,
}

/// How to factor each number on the command line.
struct Run {
    ecm_digits: u32,
    threads: usize,
    qs: bool,
    gnfs: bool,
    files: Files,
    show_status: bool,
}

impl Run {
    /// Factors `n`: the pipeline up to ECM on all threads, then the
    /// quadratic sieve on all threads for each composite left, and the same
    /// again for the pieces it splits into.
    fn factor(&self, n: &str) -> Result<Vec<Factor<String>>, ParseUintError> {
        let mode = if self.qs { QsMode::Defer } else { QsMode::Off };
        let config = Config::default().ecm_digits(self.ecm_digits).qs(mode);
        let mut factors = Vec::new();
        for f in factor_parallel(n, &config, self.threads, self.show_status)? {
            let bits = f.value.parse::<Wide>()?.bits();
            let found = if f.is_prime || !self.qs {
                None
            } else if self.gnfs && bits >= GNFS_MIN_BITS {
                gnfs_parallel(&f.value, self.threads, &self.files, self.show_status)?
            } else if bits > QS_MAX_BITS {
                None
            } else {
                qs_parallel(&f.value, self.threads, self.show_status)?
            };
            match found {
                Some(d) => {
                    let (m, d) = (f.value.parse::<Wide>()?, d.parse::<Wide>()?);
                    factors.extend(self.factor(&d.to_string())?);
                    factors.extend(self.factor(&m.div_rem(&d).0.to_string())?);
                }
                None => factors.push(f),
            }
        }
        factors.sort_by_cached_key(|f| f.value.parse::<Wide>().unwrap());
        Ok(factors)
    }
}

/// Runs the quadratic sieve on `threads` threads, each with its own sieve,
/// sending relations to this thread until there are enough. Returns a
/// proper factor of `n`, or `None` if it found none.
fn qs_parallel(
    n: &str,
    threads: usize,
    show_status: bool,
) -> Result<Option<String>, ParseUintError> {
    let job = match QsJob::start(n)? {
        Start::Sieve(job) => job,
        Start::Factor(f) => return Ok(Some(f)),
        Start::Fail => return Ok(None),
    };
    let bits = job.bits();
    let mut collector = job.collector();
    let mut status = Status::new(1);
    let (send, receive) = mpsc::channel();
    let done = AtomicBool::new(false);
    thread::scope(|scope| {
        for seed in 0..threads {
            let (send, job, done) = (send.clone(), &job, &done);
            scope.spawn(move || {
                let mut sieve = job.sieve(seed as u64);
                while !done.load(Ordering::Relaxed) {
                    if send.send(sieve.sieve_poly()).is_err() {
                        break;
                    }
                }
            });
        }
        while !collector.is_complete() {
            collector.add(&receive.recv().expect("sieving threads stopped"));
            if show_status {
                let event = Progress::Qs {
                    bits,
                    relations: collector.len() as u64,
                    target: collector.target() as u64,
                };
                status.update(0, event);
            }
        }
        done.store(true, Ordering::Relaxed);
    });
    if show_status {
        status.show(&format!(
            "Quadratic sieve: solving for a {bits}-bit composite"
        ));
    }
    let found = collector.finish();
    if show_status {
        status.clear();
    }
    Ok(found)
}

/// Runs the number field sieve on `threads` threads: each sieves the next
/// block of special-q no thread has taken and sends its relations to this
/// thread, until there are enough. Then each thread tries the next dependency until one
/// gives a factor. Returns a proper factor of `n`, or `None` if it found
/// none, which is what happens when `n` is a prime power.
fn gnfs_parallel(
    n: &str,
    threads: usize,
    files: &Files,
    show_status: bool,
) -> Result<Option<String>, ParseUintError> {
    let n = n.parse::<Wide>()?;
    let bits = n.bits();
    let params = Params::for_bits(bits);
    let mut status = Status::new(1);
    let loaded_poly = files.load.as_deref().and_then(|path| {
        let path = poly_path(path);
        let text = std::fs::read_to_string(&path).ok()?;
        let poly = parse_poly(&n, &text, &params);
        if poly.is_none() {
            status.stage(&format!(
                "GNFS: {} isn't a polynomial pair for this number; choosing one",
                path.display()
            ));
        }
        poly
    });
    let from_file = loaded_poly.is_some();
    let gnfs = match loaded_poly {
        Some(poly) => Gnfs::with_poly(&n, &params, poly),
        None => Gnfs::new_with(&n, &params, &Threads(threads)),
    };
    let fb = gnfs.factor_base();
    let ideals = fb.rational.len() + fb.algebraic.len();
    status.stage(&format!(
        "GNFS on a {bits}-bit composite: polynomial of degree {} {}, {ideals} ideals in the \
         factor bases",
        gnfs.poly().f.len() - 1,
        if from_file { "read" } else { "chosen" }
    ));
    let mut relations = gnfs.relations();
    let mut done_qs: HashSet<(u64, u64)> = HashSet::new();
    if let Some(path) = &files.load {
        let (mut read, mut rejected) = (0, 0);
        let mut batch = Vec::new();
        for line in BufReader::new(File::open(path).expect("checked readable")).lines() {
            let Ok(line) = line else {
                break;
            };
            let line = line.trim();
            if let Some(q) = line.strip_prefix("# special-q ") {
                if let Some((q, r)) = q.split_once(' ')
                    && let (Ok(q), Ok(r)) = (q.parse(), r.parse())
                {
                    done_qs.insert((q, r));
                }
            } else if !line.is_empty() && !line.starts_with('#') {
                match parse_relation(gnfs.poly(), line) {
                    Some(rel) => {
                        batch.push(rel);
                        read += 1;
                    }
                    None => rejected += 1,
                }
                if batch.len() >= 100_000 {
                    relations.add(batch.drain(..));
                }
            }
        }
        relations.add(batch);
        status.stage(&format!(
            "GNFS: loaded {read} relations from {}, {} distinct, {rejected} lines rejected, \
             {} special-q done",
            path.display(),
            relations.len(),
            done_qs.len()
        ));
    }
    let mut save = files.save.as_deref().and_then(|path| {
        let opened = std::fs::write(poly_path(path), poly_text(&n, gnfs.poly()))
            .and_then(|()| OpenOptions::new().create(true).append(true).open(path));
        match opened {
            Ok(file) => Some(BufWriter::new(file)),
            Err(e) => {
                status.stage(&format!("GNFS: can't save to {}: {e}", path.display()));
                None
            }
        }
    });
    let (send, receive) = mpsc::channel();
    let (next_block, last_q) = (AtomicU64::new(0), AtomicU64::new(0));
    let done = AtomicBool::new(false);
    thread::scope(|scope| {
        if relations.is_ready() {
            return;
        }
        for _ in 0..threads {
            let (send, gnfs, next_block, last_q, done, done_qs) =
                (send.clone(), &gnfs, &next_block, &last_q, &done, &done_qs);
            scope.spawn(move || {
                let mut sieve = gnfs.sieve();
                while !done.load(Ordering::Relaxed) {
                    let k = next_block.fetch_add(1, Ordering::Relaxed);
                    for (q, r) in gnfs.special_qs(k) {
                        if done_qs.contains(&(q, r)) {
                            continue;
                        }
                        if done.load(Ordering::Relaxed)
                            || send.send((q, r, sieve.special_q(q, r))).is_err()
                        {
                            return;
                        }
                        last_q.fetch_max(q, Ordering::Relaxed);
                    }
                }
            });
        }
        let mut flushed = Instant::now();
        while !relations.is_ready() {
            let (q, r, found) = receive.recv().expect("sieving threads stopped");
            if let Some(out) = &mut save {
                let mut text = String::new();
                for rel in &found {
                    text += &relation_line(rel);
                    text.push('\n');
                }
                text += &format!("# special-q {q} {r}\n");
                let written = out.write_all(text.as_bytes()).and_then(|()| {
                    if flushed.elapsed() >= SAVE_FLUSH {
                        flushed = Instant::now();
                        out.flush()
                    } else {
                        Ok(())
                    }
                });
                if let Err(e) = written {
                    status.stage(&format!("GNFS: stopped saving relations: {e}"));
                    save = None;
                }
            }
            relations.add(found);
            if show_status {
                let q = last_q.load(Ordering::Relaxed);
                status.update_line(&format!(
                    "Number field sieve: {} relations for {ideals} ideals, special-q {q}, \
                     on a {bits}-bit composite",
                    relations.len()
                ));
            }
        }
        done.store(true, Ordering::Relaxed);
    });
    if let Some(mut out) = save
        && let Err(e) = out.flush()
    {
        status.stage(&format!("GNFS: couldn't finish saving relations: {e}"));
    }
    match last_q.load(Ordering::Relaxed) {
        0 => status.stage(&format!(
            "GNFS: {} relations, enough without sieving",
            relations.len()
        )),
        q => status.stage(&format!(
            "GNFS: sieved {} relations, special-q up to {q}",
            relations.len()
        )),
    }
    if show_status {
        status.show("Number field sieve: filtering and solving the matrix");
    }
    let deps = relations.dependencies_with(&Threads(threads));
    status.stage(&format!("GNFS: {} dependencies", deps.len()));
    if show_status {
        status.show("Number field sieve: square roots");
    }
    // One dependency at a time on all threads: each gives a factor with
    // probability about ½, so few are needed.
    let mut found = None;
    for (i, dep) in deps.iter().enumerate() {
        found = relations.try_dependency_with(dep, &Threads(threads));
        let outcome = if found.is_some() {
            "a factor"
        } else {
            "no factor"
        };
        status.stage(&format!(
            "GNFS: dependency {i}, of {} relations, gave {outcome}",
            dep.len()
        ));
        if found.is_some() {
            break;
        }
    }
    status.clear();
    Ok(found.map(|f| f.to_string()))
}

/// `FILE.poly`, where the polynomials for the relations in `FILE` go.
fn poly_path(relations: &Path) -> PathBuf {
    let mut name = relations.as_os_str().to_owned();
    name.push(".poly");
    PathBuf::from(name)
}

/// Splits work across this many scoped threads.
struct Threads(usize);

impl Parallel for Threads {
    fn threads(&self) -> usize {
        self.0
    }

    fn map<T: Send>(&self, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        let f = &f;
        thread::scope(|scope| {
            let handles: Vec<_> = (1..self.0).map(|i| scope.spawn(move || f(i))).collect();
            let first = f(0);
            std::iter::once(first)
                .chain(
                    handles
                        .into_iter()
                        .map(|h| h.join().expect("worker panicked")),
                )
                .collect()
        })
    }
}

/// Runs the pipeline on `threads` threads, each with its share of the ECM
/// curves. The first run to end with every factor prime stops the others.
/// If none does, the run that split the most wins.
fn factor_parallel(
    n: &str,
    config: &Config,
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
                    let config = config.worker(worker as u32, threads as u32);
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
        if self.due() {
            self.show(&describe(&self.events));
        }
    }

    /// Draws `line` if it is time to redraw.
    fn update_line(&mut self, line: &str) {
        if self.due() {
            self.show(line);
        }
    }

    /// Whether to redraw: not for quick factorizations, and not more often
    /// than every [`REDRAW`].
    fn due(&self) -> bool {
        self.start.elapsed() >= Duration::from_millis(300)
            && self.drawn.is_none_or(|t| t.elapsed() >= REDRAW)
    }

    fn show(&mut self, line: &str) {
        self.drawn = Some(Instant::now());
        let secs = self.start.elapsed().as_secs_f64();
        eprint!("\r\x1b[K{line} ({secs:.1} s)");
        let _ = std::io::stderr().flush();
    }

    fn clear(&self) {
        if self.drawn.is_some() {
            eprint!("\r\x1b[K");
        }
    }

    /// Prints `line` on a line of its own, with the time so far, whether or
    /// not stderr is a terminal: the stages of long runs. A status line
    /// being drawn is cleared first, to be redrawn later.
    fn stage(&mut self, line: &str) {
        self.clear();
        self.drawn = None;
        let secs = self.start.elapsed().as_secs_f64();
        eprintln!("[{secs:8.1} s] {line}");
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
        _ => None,
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
        Some(Progress::Qs {
            bits,
            relations,
            target,
        }) => format!("Quadratic sieve: {relations}/{target} relations on a {bits}-bit composite"),
        _ => String::new(),
    }
}
