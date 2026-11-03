//! How good a polynomial pair is: the size of `f`'s values over a skewed
//! region, and Murphy's E, an estimate of how often values are smooth on
//! both sides at once.

use alloc::vec::Vec;

use super::params::Params;
use super::poly::exp2;
use crate::qs::log2;

/// Steps of `log₂` of the skew in the search for the best one.
const SKEW_STEPS: u32 = 60;

/// Points around the sieve region at which Murphy's E is sampled.
const E_POINTS: usize = 500;

/// Dickman's ρ is tabulated at steps of `1/RHO_STEPS`, up to `RHO_MAX`.
const RHO_STEPS: usize = 64;
const RHO_MAX: usize = 40;

/// `log₂` of the L² norm of `F(a, b)` over `a = √s·x`, `b = y/√s` for
/// `x` in `[−1, 1]` and `y` in `[0, 1]`, for the skew `s = 2^ls`.
///
/// With `eᵢ = cᵢ·s^(i − d/2)`, `F² = Σ eᵢ·eⱼ·x^(i+j)·y^(2d−i−j)`, and the
/// odd powers of `x` integrate to 0.
fn log_l2(f: &[f64], ls: f64) -> f64 {
    let d = f.len() - 1;
    let e: Vec<f64> = f
        .iter()
        .enumerate()
        .map(|(i, &c)| c * exp2(ls * (i as f64 - d as f64 / 2.0)))
        .collect();
    let mut sum = 0.0;
    for (i, &ei) in e.iter().enumerate() {
        for (j, &ej) in e.iter().enumerate().skip(i % 2).step_by(2) {
            sum += ei * ej * 2.0 / ((i + j + 1) * (2 * d - i - j + 1)) as f64;
        }
    }
    log2(sum.max(f64::MIN_POSITIVE)) / 2.0
}

/// The skew that minimizes the L² norm of `f`'s values, and `log₂` of that
/// norm.
pub fn lognorm(f: &[i128]) -> (f64, f64) {
    let f: Vec<f64> = f.iter().map(|&c| c as f64).collect();
    // Golden-section search on log₂ s.
    let phi = 0.618_033_988_749_895;
    let (mut lo, mut hi) = (-8.0f64, 80.0f64);
    let mut x1 = hi - phi * (hi - lo);
    let mut x2 = lo + phi * (hi - lo);
    let (mut y1, mut y2) = (log_l2(&f, x1), log_l2(&f, x2));
    for _ in 0..SKEW_STEPS {
        if y1 < y2 {
            hi = x2;
            (x2, y2) = (x1, y1);
            x1 = hi - phi * (hi - lo);
            y1 = log_l2(&f, x1);
        } else {
            lo = x1;
            (x1, y1) = (x2, y2);
            x2 = lo + phi * (hi - lo);
            y2 = log_l2(&f, x2);
        }
    }
    let ls = (lo + hi) / 2.0;
    (exp2(ls), log_l2(&f, ls))
}

/// Murphy's E for one size of number: the average over points around the
/// sieve region of `ρ(u_f)·ρ(u_g)`, the chance that values of both
/// polynomials there are smooth, with `u` the value's size over the
/// smoothness bound's, the values made smaller by their α.
pub struct Murphy {
    rho: Vec<f64>,
    /// `log₂` of the algebraic and rational smoothness bounds.
    log_bf: f64,
    log_bg: f64,
    /// `log₂` of the area of the region of `(a, b)` sieved per special-q.
    log_area: f64,
}

impl Murphy {
    pub fn new(params: &Params) -> Self {
        // ρ(u) = 1 up to 1, 1 − ln u up to 2, then ρ′(u) = −ρ(u − 1)/u,
        // integrated by the trapezoid rule.
        let h = 1.0 / RHO_STEPS as f64;
        let mut rho = Vec::with_capacity(RHO_MAX * RHO_STEPS + 1);
        for i in 0..=RHO_MAX * RHO_STEPS {
            let u = i as f64 * h;
            let r = if i <= RHO_STEPS {
                1.0
            } else if i <= 2 * RHO_STEPS {
                1.0 - log2(u) * core::f64::consts::LN_2
            } else {
                let back = |k: usize| rho[k - RHO_STEPS] / (k as f64 * h);
                rho[i - 1] - h / 2.0 * (back(i - 1) + back(i))
            };
            rho.push(r.max(0.0));
        }
        let q = f64::from(params.algebraic_bound) / 2.0;
        Self {
            rho,
            log_bf: log2(f64::from(params.algebraic_bound)),
            log_bg: log2(f64::from(params.rational_bound)),
            log_area: f64::from(2 * params.log_i - 1) + log2(q),
        }
    }

    fn rho(&self, u: f64) -> f64 {
        let x = u * RHO_STEPS as f64;
        if x <= 0.0 {
            return 1.0;
        }
        let i = x as usize;
        if i + 1 >= self.rho.len() {
            return 0.0;
        }
        let t = x - i as f64;
        self.rho[i] * (1.0 - t) + self.rho[i + 1] * t
    }

    /// Murphy's E for `f` and `g = l·x − m`, given as `(m, l)`, at skew
    /// `skew`, with each one's α.
    pub fn e(&self, f: &[i128], (m, l): (f64, f64), skew: f64, alpha_f: f64, alpha_g: f64) -> f64 {
        let f: Vec<f64> = f.iter().map(|&c| c as f64).collect();
        let ls = log2(skew);
        let (rx, ry) = (
            exp2((self.log_area + ls) / 2.0),
            exp2((self.log_area - ls) / 2.0),
        );
        let mut sum = 0.0;
        for k in 0..E_POINTS {
            let (c, s) = cos_sin(core::f64::consts::PI * (k as f64 + 0.5) / E_POINTS as f64);
            let (x, y) = (rx * c, ry * s);
            // F(x, y) = Σ cᵢ·xⁱ·y^(d−i), by Horner's rule in x.
            let (mut v, mut yk) = (0.0, 1.0);
            for &ci in f.iter().rev() {
                v = v * x + ci * yk;
                yk *= y;
            }
            let lf = log2(v.abs().max(1.0)) + alpha_f;
            let lg = log2((l * x - m * y).abs().max(1.0)) + alpha_g;
            sum += self.rho(lf / self.log_bf) * self.rho(lg / self.log_bg);
        }
        sum / E_POINTS as f64
    }
}

/// `(cos t, sin t)` for `t` in `[0, π]`, by their Taylor series about π/2.
fn cos_sin(t: f64) -> (f64, f64) {
    let u = t - core::f64::consts::FRAC_PI_2;
    // cos t = −sin u, sin t = cos u.
    let (mut sin_u, mut cos_u) = (0.0, 0.0);
    let mut term = 1.0;
    for k in 0..24 {
        if k % 2 == 0 {
            cos_u += if k % 4 == 0 { term } else { -term };
        } else {
            sin_u += if k % 4 == 1 { term } else { -term };
        }
        term *= u / f64::from(k + 1);
    }
    (-sin_u, cos_u)
}
