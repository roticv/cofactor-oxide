//! WebAssembly bindings for `cofox-core`.

use wasm_bindgen::prelude::*;

/// Factors `n`, given as a decimal string, returning its factors in
/// ascending order as decimal strings.
///
/// Strings are used both ways because JS numbers can't hold large integers.
/// A composite that couldn't be split is wrapped in parentheses.
#[wasm_bindgen]
pub fn factor(n: &str) -> Result<Vec<String>, JsError> {
    let factors = cofox_core::factor_str(n.trim())?;
    Ok(factors.iter().map(ToString::to_string).collect())
}
