//! Maximum-likelihood target functions for crystallographic refinement.
//!
//! Provides the maximum-likelihood negative log-likelihood and R-factor
//! metrics.

use super::bessel::log_bessel_i0;
use super::scaling::{model_amplitude, ScalingResult};
use super::sigma_a::SigmaAResult;
use super::types::{Reflection, UnitCell};

// ── Maximum-likelihood target ───────────────────────────────────────

/// Compute the total negative log-likelihood over all working-set reflections.
///
/// For each reflection with `free_flag == false`, accumulates:
///
/// ```text
/// nll += (fo² + D² * fc²) / d - log_bessel_i0(x)
/// ```
///
/// where `d = epsilon * sigma_sq`, `x = 2 * fo * D * fc / d`, and epsilon is
/// fixed at 1.0.
///
/// The constant `log(d)` term is omitted because it does not affect gradients.
#[must_use]
pub fn maximum_likelihood_target(
    reflections: &[Reflection],
    fc_amplitudes: &[f64],
    sigma_a: &SigmaAResult,
    unit_cell: &UnitCell,
) -> f64 {
    let mut nll = 0.0_f64;

    for (i, refl) in reflections.iter().enumerate() {
        if refl.free_flag {
            continue;
        }

        let fo = f64::from(refl.f_obs);
        let fc = fc_amplitudes[i];
        let s2 = unit_cell.d_star_sq(refl.h, refl.k, refl.l);
        let (d_val, sigma_sq) = sigma_a.interpolate(s2);

        let denom = sigma_sq;
        let x = 2.0 * fo * d_val * fc / denom;

        nll +=
            fo.mul_add(fo, d_val * d_val * fc * fc) / denom - log_bessel_i0(x);
    }

    nll
}

// ── R-factors ───────────────────────────────────────────────────────

/// Compute R-work over the working set (`free_flag == false`).
///
/// `R-work = Σ|Fo − F_model| / Σ|Fo|`, where `F_model` is the full scaled
/// model amplitude (overall scale, anisotropic scaling, and bulk-solvent
/// contribution) shared with the scaling fit. Scoring the same model the fit
/// optimizes keeps R consistent with a nonzero fitted anisotropic B.
///
/// Returns 0.0 if the denominator is zero (no working-set reflections or all
/// Fo values are zero).
#[must_use]
pub fn r_work(
    reflections: &[Reflection],
    fc: &[[f32; 2]],
    f_mask: &[[f32; 2]],
    scaling: &ScalingResult,
    unit_cell: &UnitCell,
) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    for (i, refl) in reflections.iter().enumerate() {
        if refl.free_flag {
            continue;
        }
        let fo = f64::from(refl.f_obs);
        let y = model_amplitude(
            scaling.k_overall,
            scaling.k_sol,
            scaling.b_sol,
            scaling.b_iso,
            f64::from(fc[i][0]),
            f64::from(fc[i][1]),
            f64::from(f_mask[i][0]),
            f64::from(f_mask[i][1]),
            unit_cell.d_star_sq(refl.h, refl.k, refl.l) * 0.25,
        );
        num += (fo - y).abs();
        den += fo.abs();
    }

    if den < f64::EPSILON {
        0.0
    } else {
        num / den
    }
}

/// Compute R-free over the free set (`free_flag == true`).
///
/// `R-free = Σ|Fo − F_model| / Σ|Fo|`, using the same full scaled model as
/// [`r_work`].
///
/// Returns 0.0 if there are no free-set reflections.
#[must_use]
pub fn r_free_value(
    reflections: &[Reflection],
    fc: &[[f32; 2]],
    f_mask: &[[f32; 2]],
    scaling: &ScalingResult,
    unit_cell: &UnitCell,
) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    for (i, refl) in reflections.iter().enumerate() {
        if !refl.free_flag {
            continue;
        }
        let fo = f64::from(refl.f_obs);
        let y = model_amplitude(
            scaling.k_overall,
            scaling.k_sol,
            scaling.b_sol,
            scaling.b_iso,
            f64::from(fc[i][0]),
            f64::from(fc[i][1]),
            f64::from(f_mask[i][0]),
            f64::from(f_mask[i][1]),
            unit_cell.d_star_sq(refl.h, refl.k, refl.l) * 0.25,
        );
        num += (fo - y).abs();
        den += fo.abs();
    }

    if den < f64::EPSILON {
        0.0
    } else {
        num / den
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::super::sigma_a::SigmaAResult;
    use super::super::types::{Reflection, UnitCell};
    use super::*;

    /// Build a simple orthorhombic unit cell for testing.
    fn test_cell() -> UnitCell {
        UnitCell::new(50.0, 60.0, 70.0, 90.0, 90.0, 90.0)
    }

    /// Build a trivial `SigmaAResult` with uniform D and sigma_sq across all
    /// bins.
    fn uniform_sigma_a(d: f64, sigma_sq: f64) -> SigmaAResult {
        SigmaAResult {
            d_bins: [d; 20],
            sigma_sq_bins: [sigma_sq; 20],
            s2_min: 0.001,
            bin_width: 0.01,
        }
    }

    /// Build a small set of working-set reflections with known Fo values.
    fn make_working_reflections() -> Vec<Reflection> {
        let mut refls = Vec::new();
        for h in 1..=3_i32 {
            for k in 1..=3_i32 {
                for l in 1..=3_i32 {
                    refls.push(Reflection {
                        h,
                        k,
                        l,
                        f_obs: (10 * h + k + l) as f32,
                        sigma_f: 1.0,
                        free_flag: false,
                    });
                }
            }
        }
        refls
    }

    #[test]
    fn maximum_likelihood_target_decreases_when_fc_closer_to_fo() {
        let cell = test_cell();
        let sa = uniform_sigma_a(0.9, 100.0);
        let refls = make_working_reflections();

        // Fc = 0.5 * Fo (poor model)
        let fc_poor: Vec<f64> =
            refls.iter().map(|r| f64::from(r.f_obs) * 0.5).collect();
        // Fc = 0.9 * Fo (better model)
        let fc_good: Vec<f64> =
            refls.iter().map(|r| f64::from(r.f_obs) * 0.9).collect();

        let nll_poor = maximum_likelihood_target(&refls, &fc_poor, &sa, &cell);
        let nll_good = maximum_likelihood_target(&refls, &fc_good, &sa, &cell);

        assert!(
            nll_good < nll_poor,
            "better model should have lower NLL: good={nll_good}, \
             poor={nll_poor}"
        );
    }

    /// Identity scaling with no bulk solvent: `k_overall = 1`, zero
    /// anisotropic B, `k_sol = 0`.
    fn identity_scaling() -> ScalingResult {
        ScalingResult {
            k_overall: 1.0,
            b_iso: 0.0,
            k_sol: 0.0,
            b_sol: 46.0,
        }
    }

    #[test]
    fn r_work_zero_when_fc_equals_fo() {
        let cell = test_cell();
        let refls = make_working_reflections();
        // Complex Fc whose amplitude equals Fo, zero mask, identity scaling:
        // the full model reduces to |Fc| = Fo, so R-work is zero.
        let fc: Vec<[f32; 2]> = refls.iter().map(|r| [r.f_obs, 0.0]).collect();
        let f_mask = vec![[0.0_f32; 2]; refls.len()];
        let scaling = identity_scaling();

        let rw = r_work(&refls, &fc, &f_mask, &scaling, &cell);
        assert!(
            rw.abs() < 1e-6,
            "R-work should be zero for perfect model, got {rw}"
        );
    }

    #[test]
    fn r_free_value_returns_zero_when_no_free_set() {
        let cell = test_cell();
        let refls = make_working_reflections(); // all free_flag == false
        let fc: Vec<[f32; 2]> = refls.iter().map(|r| [r.f_obs, 0.0]).collect();
        let f_mask = vec![[0.0_f32; 2]; refls.len()];
        let scaling = identity_scaling();

        let rf = r_free_value(&refls, &fc, &f_mask, &scaling, &cell);
        assert!(
            rf.abs() < f64::EPSILON,
            "R-free should be 0 with no free reflections, got {rf}"
        );
    }
}
