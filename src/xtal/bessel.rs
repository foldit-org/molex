//! Modified Bessel function approximations for ML crystallographic targets.
//!
//! Implements the Abramowitz & Stegun piecewise polynomial approximations for
//! the modified Bessel functions I₀ and I₁, along with derived quantities
//! (I₁/I₀ ratio and log I₀) used in maximum-likelihood crystallographic
//! refinement.

use std::f64::consts::TAU;

/// Modified Bessel function of the first kind, order zero.
///
/// Uses the Abramowitz & Stegun piecewise polynomial approximation
/// (formulas 9.8.1 and 9.8.2). Relative error is bounded by ~1.6e-7.
#[must_use]
#[allow(clippy::excessive_precision)]
#[allow(clippy::unreadable_literal)]
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 3.75 {
        let t = (x / 3.75) * (x / 3.75);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492
                        + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        let poly = 0.39894228
            + t * (0.01328592
                + t * (0.00225319
                    + t * (-0.00157565
                        + t * (0.00916281
                            + t * (-0.02057706
                                + t * (0.02635537
                                    + t * (-0.01647633 + t * 0.00392377)))))));
        ax.exp() / ax.sqrt() * poly
    }
}

/// Ratio I₁(x) / I₀(x), the `sim` figure-of-merit function in ML
/// crystallographic refinement.
///
/// For small `x` the Taylor expansion I₁/I₀ ≈ x/2 is used to avoid
/// cancellation; for very large `x` the asymptotic form 1 − 1/(2x) is
/// returned to avoid overflow.
///
/// # Panics
///
/// This function does not panic. Negative inputs are accepted but the
/// primary use-case is `x >= 0`.
#[must_use]
#[allow(clippy::excessive_precision)]
#[allow(clippy::unreadable_literal)]
pub fn bessel_i1_over_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1e-6 {
        return x / 2.0;
    }
    if ax > 700.0 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        return sign * (1.0 - 0.5 / ax);
    }
    let i0 = bessel_i0(x);
    let i1 = bessel_i1_raw(ax);
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * i1 / i0
}

/// Natural logarithm of I₀(x), the `sim_integ` function used in ML
/// crystallographic log-likelihood targets.
///
/// Uses a Taylor approximation for small arguments and an asymptotic
/// expansion for large arguments to avoid overflow.
#[must_use]
pub fn log_bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1e-6 {
        // ln(I0(x)) ≈ x²/4 for small x
        ax * ax / 4.0
    } else if ax > 12.0 {
        // Asymptotic: ln(I0(x)) ≈ x - 0.5 * ln(2πx)
        0.5_f64.mul_add(-(TAU * ax).ln(), ax)
    } else {
        bessel_i0(x).ln()
    }
}

/// Raw I₁(|x|) for positive argument (no sign handling).
///
/// Abramowitz & Stegun piecewise polynomial approximation for the modified
/// Bessel function of the first kind, order one.
#[allow(clippy::excessive_precision)]
#[allow(clippy::unreadable_literal)]
fn bessel_i1_raw(ax: f64) -> f64 {
    debug_assert!(ax >= 0.0, "bessel_i1_raw requires non-negative argument");
    if ax <= 3.75 {
        let t = (ax / 3.75) * (ax / 3.75);
        ax * (0.5
            + t * (0.87890594
                + t * (0.51498869
                    + t * (0.15084934
                        + t * (0.02658733
                            + t * (0.00301532 + t * 0.00032411))))))
    } else {
        let t = 3.75 / ax;
        let poly = 0.39894228
            + t * (-0.03988024
                + t * (-0.00362018
                    + t * (0.00163801
                        + t * (-0.01031555
                            + t * (0.02282967
                                + t * (-0.02895312
                                    + t * (0.01787654
                                        + t * (-0.00420059))))))));
        ax.exp() / ax.sqrt() * poly
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use super::*;

    /// Helper: check that `actual` is within `tol` of `expected`.
    fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "{label}: expected {expected}, got {actual}, diff {diff} > tol \
             {tol}"
        );
    }

    /// Helper: check relative error.
    fn assert_rel(actual: f64, expected: f64, rel_tol: f64, label: &str) {
        let rel = ((actual - expected) / expected).abs();
        assert!(
            rel <= rel_tol,
            "{label}: expected {expected}, got {actual}, relative error {rel} \
             > {rel_tol}"
        );
    }

    #[test]
    fn i0_at_zero() {
        assert_close(bessel_i0(0.0), 1.0, 1e-10, "I0(0)");
    }

    #[test]
    fn i0_at_one() {
        assert_close(bessel_i0(1.0), 1.2660658, 1e-5, "I0(1)");
    }

    #[test]
    fn i0_at_boundary() {
        // I0(3.75) — exactly at the branch point; test from the small-x side
        let val = bessel_i0(3.75);
        // Known value: I0(3.75) ≈ 9.11895
        assert_rel(val, 9.11895, 1e-4, "I0(3.75)");
    }

    #[test]
    fn i0_large_branch() {
        // I0(5.0) ≈ 27.2399
        let val = bessel_i0(5.0);
        assert_rel(val, 27.2399, 1e-5, "I0(5)");
    }

    #[test]
    fn i0_negative_argument() {
        // I0 is even: I0(-x) = I0(x)
        assert_close(bessel_i0(-2.0), bessel_i0(2.0), 1e-15, "I0(-2) == I0(2)");
    }

    #[test]
    fn i0_relative_error_several_points() {
        // Reference values from high-precision tables
        let cases: &[(f64, f64)] = &[
            (0.5, 1.0634834),
            (1.0, 1.2660658),
            (2.0, 2.2795853),
            (3.0, 4.8807926),
            (5.0, 27.239872),
            (10.0, 2815.7167),
        ];
        for &(x, expected) in cases {
            assert_rel(bessel_i0(x), expected, 1e-5, &format!("I0({x})"));
        }
    }

    #[test]
    fn i1_over_i0_at_zero() {
        assert_close(bessel_i1_over_i0(0.0), 0.0, 1e-10, "I1/I0(0)");
    }

    #[test]
    fn i1_over_i0_at_one() {
        // I1(1)/I0(1) = 0.5651591/1.2660658 ≈ 0.446390
        assert_close(bessel_i1_over_i0(1.0), 0.446390, 1e-4, "I1/I0(1)");
    }

    #[test]
    fn i1_over_i0_large_x() {
        let val = bessel_i1_over_i0(1000.0);
        assert_close(val, 1.0, 1e-3, "I1/I0(1000)");
    }

    #[test]
    fn i1_over_i0_very_small() {
        // Taylor branch: I1/I0 ≈ x/2
        let x = 1e-8;
        assert_close(bessel_i1_over_i0(x), x / 2.0, 1e-15, "I1/I0(tiny)");
    }

    #[test]
    fn log_i0_at_zero() {
        assert_close(log_bessel_i0(0.0), 0.0, 1e-10, "ln(I0(0))");
    }

    #[test]
    fn log_i0_large_x() {
        // For large x, ln(I0(x)) ≈ x. At x=1000 the relative error is well
        // under 1%.
        let x = 1000.0;
        let val = log_bessel_i0(x);
        let rel = ((val - x) / x).abs();
        assert!(
            rel < 0.01,
            "ln(I0(1000)) should be close to 1000, got {val}, relative error \
             {rel}"
        );
    }

    #[test]
    fn log_i0_medium() {
        // ln(I0(3.0)) = ln(4.8807926) ≈ 1.5853...
        let val = log_bessel_i0(3.0);
        assert_rel(val, 4.8807926_f64.ln(), 1e-5, "ln(I0(3))");
    }
}
