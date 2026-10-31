//! WebAssembly bindings for `cofox-core`.

use std::ops::ControlFlow;

use cofox_core::qs::{QsCollector, QsJob, QsSieve, Start};
use cofox_core::{Config, Progress, QsMode};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = "export function call(f, ...args) { f(...args); }")]
extern "C" {
    /// Calls the JS function `f` with a progress event.
    fn call(f: &JsValue, method: &str, bits: u32, digits: u32, done: f64, total: f64);
}

/// Factors `n`, given as a decimal string, returning its factors in
/// ascending order as decimal strings.
///
/// Strings are used both ways because JS numbers can't hold large integers.
/// A composite that couldn't be split is wrapped in parentheses.
///
/// ECM runs up to the level for `ecm_digits`-digit primes, and takes curves
/// `worker`, `worker + workers`, … of each level, so that several Web
/// Workers can share them. `progress(method, bits, digits, done, total)` is
/// called before each method ("rho", "p-1", "p+1") and after each ECM curve
/// (method "ecm", with `done` of this worker's `total` curves at the level
/// for `digits`-digit primes). The page cancels by terminating the worker.
///
/// Composites that ECM leaves are returned unsplit, after ECM stops at
/// about 30% of their digits, for the page to run the quadratic sieve on
/// them across its workers with [`Sieve`] and [`Collector`].
#[wasm_bindgen]
pub fn factor(
    n: &str,
    ecm_digits: u32,
    worker: u32,
    workers: u32,
    progress: &JsValue,
) -> Result<Vec<String>, JsError> {
    let config = Config::default()
        .ecm_digits(ecm_digits)
        .worker(worker, workers)
        .qs(QsMode::Defer);
    let factors = cofox_core::factor_str_with(n.trim(), &config, |event| {
        match event {
            Progress::Method { method, bits } => call(progress, method, bits, 0, 0.0, 0.0),
            Progress::Ecm {
                bits,
                digits,
                done,
                total,
            } => call(progress, "ecm", bits, digits, done as f64, total as f64),
            _ => {}
        }
        ControlFlow::Continue(())
    })?;
    Ok(factors.iter().map(ToString::to_string).collect())
}

/// One Web Worker's share of the quadratic sieve.
#[wasm_bindgen]
pub struct Sieve(QsSieve);

#[wasm_bindgen]
impl Sieve {
    /// A sieve for `n` with its own `seed`. Fails if `n` needs no sieving,
    /// which the [`Collector`] reports first.
    #[wasm_bindgen(constructor)]
    pub fn new(n: &str, seed: u32) -> Result<Sieve, JsError> {
        match QsJob::start(n.trim())? {
            Start::Sieve(job) => Ok(Sieve(job.sieve(seed.into()))),
            _ => Err(JsError::new("no sieving needed")),
        }
    }

    /// Sieves `polys` polynomials, returning their relations encoded for
    /// [`Collector::add`].
    pub fn sieve(&mut self, polys: u32) -> Vec<u32> {
        let mut words = Vec::new();
        for _ in 0..polys {
            words.extend(self.0.sieve_poly());
        }
        words
    }
}

/// Gathers the relations from every [`Sieve`] on one number and finds the
/// factor once there are enough.
#[wasm_bindgen]
pub struct Collector {
    collector: Option<QsCollector>,
    /// The answer when no sieving is needed: a factor, or none.
    immediate: Option<String>,
    bits: u32,
}

#[wasm_bindgen]
impl Collector {
    #[wasm_bindgen(constructor)]
    pub fn new(n: &str) -> Result<Collector, JsError> {
        let n = n.trim();
        let bits = n.parse::<cofox_core::Uint<16>>()?.bits();
        let (collector, immediate) = match QsJob::start(n)? {
            Start::Sieve(job) => (Some(job.collector()), None),
            Start::Factor(f) => (None, Some(f)),
            Start::Fail => (None, None),
        };
        Ok(Collector {
            collector,
            immediate,
            bits,
        })
    }

    /// Whether `n` needs sieving. If not, [`Collector::immediate`] has the
    /// answer.
    pub fn needs_sieving(&self) -> bool {
        self.collector.is_some()
    }

    /// A factor found without sieving, if any.
    pub fn immediate(&self) -> Option<String> {
        self.immediate.clone()
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Adds relations from [`Sieve::sieve`].
    pub fn add(&mut self, words: &[u32]) {
        if let Some(c) = &mut self.collector {
            c.add(words);
        }
    }

    /// Rows of the matrix so far.
    pub fn relations(&self) -> u32 {
        self.collector.as_ref().map_or(0, |c| c.len() as u32)
    }

    /// Rows needed.
    pub fn target(&self) -> u32 {
        self.collector.as_ref().map_or(0, |c| c.target() as u32)
    }

    pub fn is_complete(&self) -> bool {
        self.collector
            .as_ref()
            .is_some_and(QsCollector::is_complete)
    }

    /// Finds a factor from the relations, once complete, or `undefined` if
    /// every dependency gave a trivial one.
    pub fn finish(&self) -> Option<String> {
        self.collector.as_ref().and_then(QsCollector::finish)
    }
}
