use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, IsTerminal, Read, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use cofox_core::arith::ParseUintError;
use cofox_core::nfs::{
    Gnfs, Params, Poly, Relations, Spill, parse_poly, parse_relation, poly_text, relation_line,
};
use cofox_core::parallel::Parallel;
use cofox_core::qs::{QsJob, Start};
use cofox_core::{Config, Factor, Progress, QS_MAX_BITS, QsMode, Uint, factor_str_with};

const USAGE: &str = "usage: cofox [--ecm DIGITS] [--threads N] [--no-qs] [--gnfs]
             [--poly FILE] [--save-relations FILE] [--load-relations FILE]...
             [--special-q START:END] <n>...

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
                same FILE, this resumes a run. Give it more than once to
                merge relations sieved on several machines
  --poly FILE   with --gnfs, use the polynomials in FILE, choosing them and
                writing them there first if it doesn't exist, so that
                machines sieving one number share them; past about 480
                bits, FILE must hold ones from CADO-NFS or msieve
  --special-q START:END
                with --gnfs and --save-relations, only sieve the special-q
                from START to END and save the relations, as one machine's
                share; a run with --load-relations on every share then
                solves the matrix";

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
    let mut q_range = None;
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
            "--save-relations" | "--load-relations" | "--poly" => {
                let Some(path) = args.next().map(PathBuf::from) else {
                    eprintln!("cofox: {arg} needs a file\n\n{USAGE}");
                    return ExitCode::FAILURE;
                };
                match arg.as_str() {
                    "--save-relations" => files.save = Some(path),
                    "--load-relations" => files.load.push(path),
                    _ => files.poly = Some(path),
                }
            }
            "--special-q" => {
                let range = args.next().and_then(|v| {
                    let (lo, hi) = v.split_once(':')?;
                    Some((lo.parse().ok()?, hi.parse().ok()?))
                });
                match range {
                    Some((lo, hi)) if lo < hi => q_range = Some((lo, hi)),
                    _ => {
                        eprintln!("cofox: --special-q needs START:END with START < END\n\n{USAGE}");
                        return ExitCode::FAILURE;
                    }
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
    for path in &files.load {
        if let Err(e) = File::open(path) {
            eprintln!("cofox: can't read {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    }
    if let Some(path) = &files.save
        && let Err(e) = OpenOptions::new().create(true).append(true).open(path)
    {
        eprintln!("cofox: can't write {}: {e}", path.display());
        return ExitCode::FAILURE;
    }

    let show_status = std::io::stderr().is_terminal();
    if let Some(range) = q_range {
        if !gnfs || files.save.is_none() || numbers.len() != 1 {
            eprintln!(
                "cofox: --special-q sieves for one number with --gnfs, and needs \
                 --save-relations to keep what it finds"
            );
            return ExitCode::FAILURE;
        }
        return match gnfs_range(&numbers[0], threads, &files, range, show_status) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("cofox: invalid number {:?}: {e}", numbers[0]);
                ExitCode::FAILURE
            }
        };
    }
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
    load: Vec<PathBuf>,
    poly: Option<PathBuf>,
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
/// thread, until there are enough. Then the matrix is solved, and each
/// dependency tried in turn on all threads until one gives a factor.
/// Returns a proper factor of `n`, or `None` if it found none, which is
/// what happens when `n` is a prime power.
fn gnfs_parallel(
    n: &str,
    threads: usize,
    files: &Files,
    show_status: bool,
) -> Result<Option<String>, ParseUintError> {
    let n = n.parse::<Wide>()?;
    let mut status = Status::new(1);
    let (gnfs, relations, done_qs) = gnfs_start(&n, threads, files, &mut status);
    let save = files
        .save
        .as_deref()
        .and_then(|path| open_save(path, &n, &gnfs, &mut status));
    let mut collected = Collected {
        relations,
        done_qs,
        save,
    };
    let last_q = gnfs_sieve(
        &gnfs,
        &mut collected,
        threads,
        Range::UntilReady,
        (&mut status, show_status),
    );
    finish_save(collected.save.take(), &mut status);
    let mut relations = collected.relations;
    match last_q {
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
    let deps = match FileSpill::new() {
        Some(mut spill) => relations.dependencies_spilling(&Threads(threads), &mut spill),
        None => relations.dependencies_with(&Threads(threads)),
    };
    status.stage(&format!("GNFS: {} dependencies", deps.len()));
    if show_status {
        status.show("Number field sieve: square roots");
    }
    // One dependency at a time on all threads: each gives a factor with
    // probability about ½, so few are needed.
    let mut found = None;
    for (i, dep) in deps.iter().enumerate() {
        found = relations.try_dependency_with(&dep, &Threads(threads));
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

/// A temporary file the relations wait in while the matrix is solved,
/// freeing their memory for filtering and block Lanczos: about 4 GB of
/// RSA-140's 14.8 GB peak, which with other programs running pushed the
/// matrix into compressed memory and stretched it from about an hour to
/// almost five.
struct FileSpill {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
}

impl FileSpill {
    fn new() -> Option<Self> {
        let path = std::env::temp_dir().join(format!("cofox-{}.relations", std::process::id()));
        let file = File::create(&path).ok()?;
        Some(Self {
            path,
            writer: Some(BufWriter::with_capacity(1 << 20, file)),
        })
    }
}

impl Spill for FileSpill {
    fn write(&mut self, words: &[u32]) -> bool {
        let Some(writer) = &mut self.writer else {
            return false;
        };
        let mut bytes = Vec::with_capacity(4 * (words.len() + 1));
        bytes.extend_from_slice(&(words.len() as u32).to_le_bytes());
        bytes.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        if writer.write_all(&bytes).is_ok() {
            return true;
        }
        self.writer = None;
        false
    }

    fn read(&mut self, f: &mut dyn FnMut(&[u32])) {
        let fail = |e: std::io::Error| -> ! {
            panic!(
                "cofox: can't read back the relations from {}: {e}",
                self.path.display()
            )
        };
        if let Some(mut writer) = self.writer.take() {
            writer.flush().unwrap_or_else(|e| fail(e));
        }
        let mut reader =
            BufReader::with_capacity(1 << 20, File::open(&self.path).unwrap_or_else(|e| fail(e)));
        let mut words = Vec::new();
        let mut word = [0u8; 4];
        loop {
            match reader.read_exact(&mut word) {
                Ok(()) => {}
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
                Err(e) => fail(e),
            }
            let len = u32::from_le_bytes(word) as usize;
            let mut bytes = vec![0u8; 4 * len];
            reader.read_exact(&mut bytes).unwrap_or_else(|e| fail(e));
            words.clear();
            words.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| u32::from_le_bytes(b)),
            );
            f(&words);
        }
    }
}

impl Drop for FileSpill {
    fn drop(&mut self) {
        self.writer = None;
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Sieves the special-q in `[lo, hi)` for `n`, saving the relations to
/// `--save-relations`, for one machine's share of a distributed run.
fn gnfs_range(
    n: &str,
    threads: usize,
    files: &Files,
    (lo, hi): (u64, u64),
    show_status: bool,
) -> Result<(), ParseUintError> {
    let n = n.parse::<Wide>()?;
    let mut status = Status::new(1);
    let (gnfs, relations, done_qs) = gnfs_start(&n, threads, files, &mut status);
    let path = files.save.as_deref().expect("checked given");
    let save = open_save(path, &n, &gnfs, &mut status);
    let before = relations.len();
    let mut collected = Collected {
        relations,
        done_qs,
        save,
    };
    gnfs_sieve(
        &gnfs,
        &mut collected,
        threads,
        Range::Between(lo, hi),
        (&mut status, show_status),
    );
    finish_save(collected.save.take(), &mut status);
    status.stage(&format!(
        "GNFS: sieved special-q from {lo} to {hi}: {} new distinct relations saved to {}",
        collected.relations.len() - before,
        path.display()
    ));
    status.clear();
    Ok(())
}

/// The relations of a GNFS run, the special-q done, and where new
/// relations are saved.
struct Collected {
    relations: Relations<16>,
    done_qs: HashSet<(u64, u64)>,
    save: Option<BufWriter<File>>,
}

/// Which special-q to sieve: blocks from the first one the parameters
/// give, until there are enough relations, or all of a range.
#[derive(Clone, Copy, Debug)]
enum Range {
    UntilReady,
    Between(u64, u64),
}

/// Special-q are handed out to the threads in blocks of this many
/// integers, within a range.
const Q_RANGE_BLOCK: u64 = 64;

/// Chooses polynomials for `n`, or exits if it's too large to.
fn select(n: &Wide, threads: usize) -> Poly {
    Poly::select_with(n, &Threads(threads)).unwrap_or_else(|| {
        eprintln!(
            "cofox: GNFS can't choose polynomials past about 480 bits; give it ones from \
             CADO-NFS or msieve with --poly"
        );
        std::process::exit(1);
    })
}

/// GNFS for `n`, with the polynomials from `--poly`, which are chosen and
/// written there first if the file doesn't exist; or else from the first
/// file loaded with a `.poly` beside it; or else chosen. Then the
/// relations in each file loaded, and the special-q they record as done.
/// Exits if `--poly` can't be used.
fn gnfs_start(
    n: &Wide,
    threads: usize,
    files: &Files,
    status: &mut Status,
) -> (Gnfs<16>, Relations<16>, HashSet<(u64, u64)>) {
    let bits = n.bits();
    let params = Params::for_bits(bits);
    let (poly, how) = if let Some(path) = &files.poly {
        match std::fs::read_to_string(path) {
            Ok(text) => match parse_poly(n, &text, &params) {
                Some(poly) => (poly, "read"),
                None => {
                    eprintln!("cofox: {} isn't a polynomial pair for {n}", path.display());
                    std::process::exit(1);
                }
            },
            Err(e) if e.kind() == ErrorKind::NotFound => {
                let poly = select(n, threads);
                if let Err(e) = std::fs::write(path, poly_text(n, &poly)) {
                    eprintln!("cofox: can't write {}: {e}", path.display());
                    std::process::exit(1);
                }
                (poly, "chosen and written")
            }
            Err(e) => {
                eprintln!("cofox: can't read {}: {e}", path.display());
                std::process::exit(1);
            }
        }
    } else if let Some(poly) = files.load.iter().find_map(|path| {
        let text = std::fs::read_to_string(poly_path(path)).ok()?;
        parse_poly(n, &text, &params)
    }) {
        (poly, "read")
    } else {
        (select(n, threads), "chosen")
    };
    let gnfs = Gnfs::with_poly(n, &params, poly);
    let fb = gnfs.factor_base();
    status.stage(&format!(
        "GNFS on a {bits}-bit composite: polynomial of degree {} {how}, {} ideals in the \
         factor bases, special-q from {}",
        gnfs.poly().degree(),
        fb.rational.len() + fb.algebraic.len(),
        gnfs.params().qmin
    ));
    let mut relations = gnfs.relations();
    let mut done_qs = HashSet::new();
    for path in &files.load {
        let (mut read, mut rejected, before) = (0, 0, relations.len());
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
            "GNFS: loaded {read} relations from {}, {} new and distinct, {rejected} lines \
             rejected; {} special-q done in all",
            path.display(),
            relations.len() - before,
            done_qs.len()
        ));
    }
    (gnfs, relations, done_qs)
}

/// The file at `path` to append relations to, after writing the
/// polynomials to `path.poly`, or `None` if that fails.
fn open_save(
    path: &Path,
    n: &Wide,
    gnfs: &Gnfs<16>,
    status: &mut Status,
) -> Option<BufWriter<File>> {
    let opened = std::fs::write(poly_path(path), poly_text(n, gnfs.poly()))
        .and_then(|()| OpenOptions::new().create(true).append(true).open(path));
    match opened {
        Ok(file) => Some(BufWriter::new(file)),
        Err(e) => {
            status.stage(&format!("GNFS: can't save to {}: {e}", path.display()));
            None
        }
    }
}

fn finish_save(save: Option<BufWriter<File>>, status: &mut Status) {
    if let Some(mut out) = save
        && let Err(e) = out.flush()
    {
        status.stage(&format!("GNFS: couldn't finish saving relations: {e}"));
    }
}

/// Sieves the special-q of `range` that aren't done on `threads` threads,
/// adding their relations to those collected and saving them, each
/// special-q's followed by a comment line naming it. Returns the largest
/// special-q sieved, or 0 if none was.
fn gnfs_sieve(
    gnfs: &Gnfs<16>,
    collected: &mut Collected,
    threads: usize,
    range: Range,
    (status, show_status): (&mut Status, bool),
) -> u64 {
    let Collected {
        relations,
        done_qs,
        save,
    } = collected;
    let done_qs = &*done_qs;
    let fb = gnfs.factor_base();
    let ideals = fb.rational.len() + fb.algebraic.len();
    let block = |k: u64| match range {
        Range::UntilReady => Some(gnfs.special_qs(k)),
        Range::Between(lo, hi) => {
            let start = lo.checked_add(k.checked_mul(Q_RANGE_BLOCK)?)?;
            (start < hi).then(|| gnfs.special_qs_between(start, (start + Q_RANGE_BLOCK).min(hi)))
        }
    };
    let until_ready = matches!(range, Range::UntilReady);
    let (send, receive) = mpsc::channel();
    let (next_block, last_q) = (AtomicU64::new(0), AtomicU64::new(0));
    let done = AtomicBool::new(false);
    thread::scope(|scope| {
        if until_ready && relations.is_ready() {
            return;
        }
        for _ in 0..threads {
            let (send, next_block, last_q, done, block) =
                (send.clone(), &next_block, &last_q, &done, &block);
            scope.spawn(move || {
                let mut sieve = gnfs.sieve();
                while !done.load(Ordering::Relaxed) {
                    let Some(qs) = block(next_block.fetch_add(1, Ordering::Relaxed)) else {
                        return;
                    };
                    for (q, r) in qs {
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
        // The threads hold the only senders left, so receiving ends when
        // they all have.
        drop(send);
        let mut flushed = Instant::now();
        while let Ok((q, r, found)) = receive.recv() {
            if let Some(out) = save.as_mut() {
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
                    *save = None;
                }
            }
            relations.add(found);
            if show_status {
                let q = last_q.load(Ordering::Relaxed);
                status.update_line(&format!(
                    "Number field sieve: {} relations for {ideals} ideals, special-q {q}",
                    relations.len()
                ));
            }
            if until_ready && relations.is_ready() {
                break;
            }
        }
        done.store(true, Ordering::Relaxed);
    });
    last_q.load(Ordering::Relaxed)
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
