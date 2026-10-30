//! WebAssembly bindings for `cofox-core`.

use std::ops::ControlFlow;

use cofox_core::{Config, Progress};
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
#[wasm_bindgen]
pub fn factor(
    n: &str,
    ecm_digits: u32,
    worker: u32,
    workers: u32,
    progress: &JsValue,
) -> Result<Vec<String>, JsError> {
    let config = Config {
        ecm_digits,
        worker,
        workers,
    };
    let factors = cofox_core::factor_str_with(n.trim(), &config, |event| {
        match event {
            Progress::Method { method, bits } => call(progress, method, bits, 0, 0.0, 0.0),
            Progress::Ecm {
                bits,
                digits,
                done,
                total,
            } => call(progress, "ecm", bits, digits, done as f64, total as f64),
        }
        ControlFlow::Continue(())
    })?;
    Ok(factors.iter().map(ToString::to_string).collect())
}
