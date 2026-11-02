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
//! So far: polynomial selection by the base-m method ([`Poly`]), the
//! roots of `f` modulo primes, and the factor bases ([`FactorBase`]).

mod base;
pub mod modp;
mod poly;

pub use base::{FactorBase, Prime};
pub use poly::{Poly, alpha, degree};
