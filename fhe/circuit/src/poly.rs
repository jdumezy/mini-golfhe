//! The polynomials the circuit evaluates, as Chebyshev series on an interval.

use std::sync::OnceLock;

use anyhow::Context;
use poulpy_ckks::approximation::sign_composite_coeffs;

/// `Σ c_k T_k(t)` with `t` the image of `x` from `[lo, hi]` onto `[-1, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Poly {
    pub cheb: Vec<f64>,
    pub interval: (f64, f64),
    /// Whether the result tolerates coefficients to [`RELAXED_COEFF_BITS`]
    /// and one more level: an engine may evaluate it with fewer products.
    pub relaxed: bool,
}

/// Bits of the coefficients of a [`Poly::relaxed`] polynomial.
pub const RELAXED_COEFF_BITS: usize = 16;

impl Poly {
    pub fn degree(&self) -> usize {
        self.cheb.len().saturating_sub(1)
    }

    /// Levels of a minimum-depth evaluation: `⌈log2(degree + 1)⌉`.
    pub fn depth(&self) -> usize {
        (usize::BITS - self.degree().leading_zeros()) as usize
    }

    /// Clenshaw's recurrence.
    pub fn eval(&self, x: f64) -> f64 {
        let (lo, hi) = self.interval;
        let t = (2.0 * x - lo - hi) / (hi - lo);
        let (mut b1, mut b2) = (0.0, 0.0);
        for &c in self.cheb.iter().skip(1).rev() {
            (b1, b2) = (2.0 * t * b1 - b2 + c, b1);
        }
        t * b1 - b2 + self.cheb[0]
    }

    /// The polynomial of degree `degree` through `f` at the Chebyshev nodes
    /// of `[lo, hi]`; exact when `f` is a polynomial of at most that degree.
    pub fn interpolate(degree: usize, lo: f64, hi: f64, f: impl Fn(f64) -> f64) -> Poly {
        let n = degree + 1;
        let nodes: Vec<f64> = (0..n).map(|j| (std::f64::consts::PI * (j as f64 + 0.5) / n as f64).cos()).collect();
        let values: Vec<f64> = nodes.iter().map(|&t| f(0.5 * (lo + hi) + 0.5 * (hi - lo) * t)).collect();
        let cheb = (0..n)
            .map(|k| {
                let s: f64 = nodes.iter().zip(&values).map(|(&t, &v)| v * (k as f64 * t.acos()).cos()).sum();
                s * if k == 0 { 1.0 } else { 2.0 } / n as f64
            })
            .collect();
        Poly { cheb, interval: (lo, hi), relaxed: false }
    }

    /// `a + b · p(x)`.
    pub fn affine(&self, a: f64, b: f64) -> Poly {
        let mut cheb: Vec<f64> = self.cheb.iter().map(|c| b * c).collect();
        cheb[0] += a;
        Poly { cheb, interval: self.interval, relaxed: self.relaxed }
    }
}

/// Threshold of the comparison: `|x| ≥ TAU` has its sign within `2^-SIGN_BITS`.
pub const SIGN_LOG_TAU: f64 = 11.0;
pub const SIGN_BITS: f64 = 12.0;

/// A composite minimax approximation of `sign` on `[-1, 1]`, as the factors to
/// apply in turn: the one of least total depth among a few degree schedules.
pub fn sign() -> &'static [Poly] {
    static SIGN: OnceLock<Vec<Poly>> = OnceLock::new();
    SIGN.get_or_init(|| {
        let tau = 2f64.powf(-SIGN_LOG_TAU);
        let schedules: [&[usize]; 3] = [&[31, 31, 31], &[63, 15, 15], &[15, 15, 31]];
        schedules
            .iter()
            .filter_map(|degrees| sign_composite_coeffs::<f64>(tau, SIGN_BITS, degrees, 8).ok())
            // Relaxed: the smoothstep after the last factor flattens their error near 0 and 1.
            .map(|rows| {
                rows.into_iter().map(|cheb| Poly { cheb, interval: (-1.0, 1.0), relaxed: true }).collect::<Vec<_>>()
            })
            .min_by_key(|factors| {
                (factors.iter().map(Poly::depth).sum::<usize>(), factors.iter().map(Poly::degree).sum::<usize>())
            })
            .context("no composite sign reaches the target")
            .unwrap()
    })
}

/// `0` at 0, `1` at `1, …, degree`: is a count positive (exact up to `degree`).
pub fn is_positive(degree: usize) -> Poly {
    Poly::interpolate(degree, 0.0, (degree + 1) as f64, |x| {
        1.0 - (1..=degree).map(|m| 1.0 - x / m as f64).product::<f64>()
    })
}

pub fn compose(factors: &[Poly], x: f64) -> f64 {
    factors.iter().fold(x, |x, p| p.eval(x))
}

/// The worst error of `factors` on `[τ, 1]`, against 1.
pub fn sign_error(factors: &[Poly]) -> f64 {
    let tau = 2f64.powf(-SIGN_LOG_TAU);
    (0..=4000)
        .map(|i| tau + (1.0 - tau) * i as f64 / 4000.0)
        .map(|x| (compose(factors, x) - 1.0).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_is_exact_on_polynomials() {
        let p = Poly::interpolate(3, 0.0, 4.0, |x| 2.0 - x + 0.5 * x * x * x);
        for x in [0.0, 0.3, 1.0, 2.5, 4.0] {
            assert!((p.eval(x) - (2.0 - x + 0.5 * x * x * x)).abs() < 1e-12);
        }
    }

    #[test]
    fn counts_are_tested_exactly() {
        let q = is_positive(7);
        assert!(q.eval(0.0).abs() < 1e-12);
        for m in 1..=7 {
            assert!((q.eval(m as f64) - 1.0).abs() < 1e-9, "is_positive({m}) = {}", q.eval(m as f64));
        }
    }

    #[test]
    fn sign_resolves_its_threshold() {
        let s = sign();
        let depth: usize = s.iter().map(Poly::depth).sum();
        assert!(sign_error(s) <= 2f64.powf(-SIGN_BITS) * 1.01, "error {}", sign_error(s));
        assert!(depth <= 15, "sign depth {depth}");
        // Odd: negative inputs map to −1.
        assert!((compose(s, -0.5) + 1.0).abs() < 1e-3);
    }
}
