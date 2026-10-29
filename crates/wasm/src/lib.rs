//! WebAssembly bindings for `cofox-core`.

use wasm_bindgen::prelude::*;

/// Factors `n`, returning its prime factors in ascending order.
///
/// `n` crosses the JS boundary as a `BigInt`, since a JS `number` cannot
/// represent every `u64`.
#[wasm_bindgen]
pub fn factor(n: u64) -> Vec<u64> {
    cofox_core::factor(n)
}
