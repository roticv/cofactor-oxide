//! The rational and algebraic factor bases, and the quadratic characters.

use alloc::vec::Vec;

use super::modp::{eval, reduce, roots};
use super::poly::Poly;
use crate::primes::Primes;
use crate::qs::log2;

/// A prime of a factor base with a root: `p` divides `a − b·r` (rational
/// side) or `F(a, b)` (algebraic side) exactly when `a ≡ b·r (mod p)`. On
/// the algebraic side, `r = p` stands for the root at infinity, where `p`
/// divides the leading coefficient and `F(a, b)` whenever `p | b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prime {
    pub p: u32,
    pub r: u32,
    /// `log₂ p`, rounded.
    pub log: u8,
}

impl Prime {
    fn new(p: u64, r: u64) -> Self {
        Self {
            p: p as u32,
            r: r as u32,
            log: (log2(p as f64) + 0.5) as u8,
        }
    }

    /// Whether this is the root at infinity.
    pub fn projective(&self) -> bool {
        self.r == self.p
    }
}

/// The primes both sides of a relation must factor over, and the
/// quadratic characters that make a product of `a − bα` a square in the
/// number field, not just of square norm.
#[derive(Clone, Debug)]
pub struct FactorBase {
    /// Primes up to the rational bound, each with `m mod p`.
    pub rational: Vec<Prime>,
    /// Every root of `f` modulo each prime up to the algebraic bound.
    pub algebraic: Vec<Prime>,
    /// Pairs `(q, s)` past the algebraic bound with `s` a simple root of
    /// `f` mod `q`, `q` not dividing the leading coefficient.
    pub characters: Vec<Prime>,
}

impl FactorBase {
    pub fn new(poly: &Poly, rational_bound: u32, algebraic_bound: u32, characters: usize) -> Self {
        let f = &poly.f;
        let lead = f[f.len() - 1];
        let m = poly.m;
        let rational = Primes::new(2, rational_bound.into())
            .map(|p| Prime::new(p, (m % u128::from(p)) as u64))
            .collect();
        let mut algebraic = Vec::new();
        for p in Primes::new(2, algebraic_bound.into()) {
            algebraic.extend(roots(f, p).into_iter().map(|r| Prime::new(p, r)));
            if lead.rem_euclid(i128::from(p)) == 0 {
                algebraic.push(Prime::new(p, p));
            }
        }
        let derivative: Vec<i128> = f
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| c * i as i128)
            .collect();
        let mut chars = Vec::with_capacity(characters);
        for q in Primes::new(u64::from(algebraic_bound) + 1, u64::MAX >> 32) {
            if chars.len() >= characters {
                break;
            }
            if lead.rem_euclid(i128::from(q)) == 0 {
                continue;
            }
            let df = reduce(&derivative, q);
            for s in roots(f, q) {
                if eval(&df, s, q) != 0 && chars.len() < characters {
                    chars.push(Prime::new(q, s));
                }
            }
        }
        Self {
            rational,
            algebraic,
            characters: chars,
        }
    }
}
