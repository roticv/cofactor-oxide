//! Elliptic curve arithmetic for ECM.
//!
//! ECM uses Montgomery curves `B·y² = x³ + A·x² + x` and works with the
//! x-coordinate alone, in projective form `(X : Z)` with `x = X/Z`. That
//! avoids modular inverses: no step divides, and the point at infinity is
//! simply `Z = 0`. The price is that two points can only be added when their
//! difference is known, which the Montgomery ladder arranges.
//!
//! All coordinates are in Montgomery form.

use crate::arith::{Monty, Uint};

/// A point `(X : Z)` on a Montgomery curve, identified by its x-coordinate
/// up to sign. `Z ≡ 0` is the point at infinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point<const LIMBS: usize> {
    pub x: Uint<LIMBS>,
    pub z: Uint<LIMBS>,
}

/// A Montgomery curve modulo `n`, described by `a24 = (A + 2) / 4`, which is
/// the only curve constant the x-only formulas need. `B` only selects the
/// quadratic twist, and never appears.
#[derive(Clone, Copy, Debug)]
pub struct Curve<const LIMBS: usize> {
    monty: Monty<LIMBS>,
    a24: Uint<LIMBS>,
}

impl<const LIMBS: usize> Curve<LIMBS> {
    /// The curve with `a24 = (A + 2) / 4`, given in Montgomery form.
    pub fn new(monty: Monty<LIMBS>, a24: Uint<LIMBS>) -> Self {
        Self { monty, a24 }
    }

    /// Suyama's parametrization: a curve and a point on it, from `sigma`.
    ///
    /// With `u = σ² − 5` and `v = 4σ`, the point is `(u³ : v³)` and
    /// `(A + 2)/4 = (v − u)³·(3u + v) / (16·u³·v)`. The group of every such
    /// curve over a prime field has order divisible by 12, which gives ECM a
    /// head start on smoothness.
    ///
    /// Computing `a24` needs one inverse. If `16·u³·v` shares a factor with
    /// `n`, that factor is returned as `Err`. It may be `n` itself, in which
    /// case this `sigma` is unusable and another should be tried. `sigma`
    /// should be at least 6, since 0 to 5 give degenerate curves.
    pub fn suyama(monty: &Monty<LIMBS>, sigma: u64) -> Result<(Self, Point<LIMBS>), Uint<LIMBS>> {
        let m = monty;
        let s = m.to_monty(&Uint::from_u64(sigma));
        let u = m.sub(&m.square(&s), &m.to_monty(&Uint::from_u64(5)));
        let v = m.add(&m.add(&s, &s), &m.add(&s, &s));
        let cube = |t: &Uint<LIMBS>| m.mul(&m.square(t), t);
        let u3 = cube(&u);
        let v3 = cube(&v);

        let v_minus_u = m.sub(&v, &u);
        let three_u_plus_v = m.add(&m.add(&u, &u), &m.add(&u, &v));
        let numerator = m.mul(&cube(&v_minus_u), &three_u_plus_v);
        let sixteen = m.to_monty(&Uint::from_u64(16));
        let denominator = m.mul(&m.mul(&sixteen, &u3), &v);
        let a24 = m.mul(&numerator, &m.inv(&denominator)?);

        Ok((Self::new(*monty, a24), Point { x: u3, z: v3 }))
    }

    pub fn monty(&self) -> &Monty<LIMBS> {
        &self.monty
    }

    /// Computes `2P`, with 2 squarings and 3 multiplications:
    ///
    /// ```text
    /// X₂ = (X + Z)²·(X − Z)²
    /// Z₂ = 4XZ·((X − Z)² + a24·4XZ),  where 4XZ = (X + Z)² − (X − Z)²
    /// ```
    pub fn double(&self, p: &Point<LIMBS>) -> Point<LIMBS> {
        let m = &self.monty;
        let sum2 = m.square(&m.add(&p.x, &p.z));
        let diff2 = m.square(&m.sub(&p.x, &p.z));
        let four_xz = m.sub(&sum2, &diff2);
        Point {
            x: m.mul(&sum2, &diff2),
            z: m.mul(&four_xz, &m.add(&diff2, &m.mul(&self.a24, &four_xz))),
        }
    }

    /// Computes `P + Q` given `P − Q`, with 2 squarings and 4 multiplications:
    ///
    /// ```text
    /// s = (Xp − Zp)(Xq + Zq),  t = (Xp + Zp)(Xq − Zq)
    /// X = Z₋·(s + t)²,  Z = X₋·(s − t)²
    /// ```
    ///
    /// `diff` must not be the point at infinity, so `P ≠ Q`. Use
    /// [`Curve::double`] for that.
    pub fn add(&self, p: &Point<LIMBS>, q: &Point<LIMBS>, diff: &Point<LIMBS>) -> Point<LIMBS> {
        let m = &self.monty;
        let s = m.mul(&m.sub(&p.x, &p.z), &m.add(&q.x, &q.z));
        let t = m.mul(&m.add(&p.x, &p.z), &m.sub(&q.x, &q.z));
        Point {
            x: m.mul(&diff.z, &m.square(&m.add(&s, &t))),
            z: m.mul(&diff.x, &m.square(&m.sub(&s, &t))),
        }
    }

    /// Computes `[k]P` with the Montgomery ladder.
    ///
    /// The ladder keeps a pair `(R0, R1) = ([j]P, [j+1]P)`, whose difference
    /// is always `P`, so the difference-based addition always applies. Each
    /// bit of `k` costs one doubling and one addition. `[0]P` is the point
    /// at infinity `(1 : 0)`.
    pub fn mul(&self, p: &Point<LIMBS>, k: u64) -> Point<LIMBS> {
        if k == 0 {
            return Point {
                x: self.monty.one(),
                z: Uint::ZERO,
            };
        }
        let mut r0 = *p;
        let mut r1 = self.double(p);
        for bit in (0..63 - k.leading_zeros()).rev() {
            if (k >> bit) & 1 == 1 {
                r0 = self.add(&r1, &r0, p);
                r1 = self.double(&r1);
            } else {
                r1 = self.add(&r1, &r0, p);
                r0 = self.double(&r0);
            }
        }
        r0
    }
}
