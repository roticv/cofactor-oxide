//! The rational and algebraic factor bases, and the quadratic characters.

use alloc::vec::Vec;

use num_bigint::BigInt;

use super::modp::{Coeff, eval, inv, reduce, roots};
use super::poly::Poly;
use crate::primes::Primes;
use crate::qs::log2;

/// The quadratic characters' primes start here, past every prime a
/// relation has: a character `(q, s)` says nothing about a relation with
/// `q | a − b·s`, as one from the special-q `(q, s)` would be.
pub const CHARACTER_PRIMES: u64 = 1 << 31;

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
    /// Primes up to the rational bound, each with the root of `g` mod `p`.
    pub rational: Vec<Prime>,
    /// Every root of `f` modulo each prime up to the algebraic bound.
    pub algebraic: Vec<Prime>,
    /// Pairs `(q, s)` from `CHARACTER_PRIMES` up with `s` a simple root
    /// of `f` mod `q`, `q` not dividing the leading coefficient.
    pub characters: Vec<Prime>,
}

impl FactorBase {
    pub fn new(poly: &Poly, rational_bound: u32, algebraic_bound: u32, characters: usize) -> Self {
        let f = &poly.f;
        let lead = &f[f.len() - 1];
        // g(x) = l·x − m has the root m/l mod p, or one at infinity if p | l.
        let (m, l) = (&poly.m, &poly.l);
        let rational = Primes::new(2, rational_bound.into())
            .map(|p| {
                let lp = l.mod_p(p);
                let r = if lp == 0 {
                    p
                } else {
                    m.mod_p(p) * inv(lp, p) % p
                };
                Prime::new(p, r)
            })
            .collect();
        let mut algebraic = Vec::new();
        for p in Primes::new(2, algebraic_bound.into()) {
            algebraic.extend(roots(f, p).into_iter().map(|r| Prime::new(p, r)));
            if lead.mod_p(p) == 0 {
                algebraic.push(Prime::new(p, p));
            }
        }
        let derivative: Vec<BigInt> = f
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| c * BigInt::from(i))
            .collect();
        let mut chars = Vec::with_capacity(characters);
        for q in Primes::new(CHARACTER_PRIMES, 1 << 32) {
            if chars.len() >= characters {
                break;
            }
            if lead.mod_p(q) == 0 {
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
