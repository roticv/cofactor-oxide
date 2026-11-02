//! The general number field sieve, under construction.
//!
//! GNFS finds pairs `(a, b)` for which both `a − b·m` and
//! `F(a, b) = b^d·f(a/b)` are smooth, for polynomials `f` and `x − m` with
//! a common root `m` modulo `N`. A set of pairs whose products are squares
//! on both sides, the rational integers and the number field `ℚ(α)` with
//! `f(α) = 0`, gives a congruence of squares modulo `N`, as in the
//! quadratic sieve. Its values grow more slowly than the quadratic sieve's
//! with the size of `N`, which makes it faster past about 100 digits.
//!
//! [`gnfs`] chooses polynomials by the base-m method ([`Poly`]), sieves
//! lines of `(a, b)` ([`Sieve`]), solves the matrix with the quadratic
//! sieve's block Lanczos, and takes the square roots in `sqrt`. It is a
//! first, unoptimized version: correct, but slower than the quadratic
//! sieve at every size so far.

mod base;
mod matrix;
pub mod modp;
mod params;
mod poly;
mod sieve;
mod sqrt;

use alloc::vec::Vec;
use core::ops::ControlFlow;

pub use base::{FactorBase, Prime};
pub use matrix::{root_of, rows};
pub use params::{CHARACTERS, Params};
pub use poly::{Poly, alpha, degree};
pub use sieve::{Relation, Sieve};

use crate::arith::Uint;
use crate::qs::dependencies;
use crate::qs::linalg::prune;

/// Rows beyond the columns, after pruning, before solving.
const EXTRA: usize = 64;

/// The callback asked [`gnfs_with`] to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stopped;

/// Factors `n`, an odd composite that isn't a prime power, with GNFS on
/// one thread. Returns a proper factor, or `None` if none was found.
pub fn gnfs<const LIMBS: usize>(n: &Uint<LIMBS>) -> Option<Uint<LIMBS>> {
    let params = Params::for_bits(n.bits());
    gnfs_with(n, &params, |_, _| ControlFlow::Continue(())).unwrap_or(None)
}

/// Factors `n` like [`gnfs`] with `params`, calling `progress(lines,
/// relations)` after each line, and stopping if it returns `Break`.
pub fn gnfs_with<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    params: &Params,
    mut progress: impl FnMut(u64, usize) -> ControlFlow<()>,
) -> Result<Option<Uint<LIMBS>>, Stopped> {
    let poly = Poly::select(n);
    let fb = FactorBase::new(
        &poly,
        params.rational_bound,
        params.algebraic_bound,
        CHARACTERS,
    );
    let mut sieve = Sieve::new(&poly, &fb, params);
    let mut relations: Vec<Relation> = Vec::new();
    // Partial relations only start to combine once there are about as
    // many relations as ideals in the factor bases.
    let mut next_check = fb.rational.len() + fb.algebraic.len();
    for b in 1.. {
        relations.extend(sieve.line(b));
        if progress(b, relations.len()).is_break() {
            return Err(Stopped);
        }
        if relations.len() < next_check {
            continue;
        }
        next_check = relations.len() + relations.len() / 10;
        let (rows, cols) = rows(&fb, &relations);
        let (_, pruned, pruned_cols) = prune(&rows, cols);
        if pruned.len() < pruned_cols + EXTRA {
            continue;
        }
        for dep in dependencies(&rows, cols) {
            let dep: Vec<&Relation> = dep.iter().map(|&i| &relations[i]).collect();
            if let Some(f) = sqrt::factor(n, &poly, &dep) {
                return Ok(Some(f));
            }
        }
    }
    Ok(None)
}
