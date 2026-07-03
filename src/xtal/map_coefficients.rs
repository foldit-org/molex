//! Maximum-likelihood map coefficient computation.
//!
//! Provides 2mFo-DFc and mFo-DFc difference map coefficients using sigma-A
//! weighting, along with inverse-FFT placement onto a 3D grid.

use super::bessel::bessel_i1_over_i0;
use super::fft_cpu::{fft_3d_inverse, FftError};
use super::sigma_a::SigmaAResult;
use super::types::{Reflection, UnitCell};

/// Threshold above which the Bessel ratio I₁/I₀ is effectively 1.0.
const FOM_X_CLAMP: f64 = 700.0;

/// Tolerance for denominators considered effectively zero.
const NEAR_ZERO: f64 = 1e-30;

/// Compute the figure of merit *m* for an acentric reflection.
///
/// The argument to the Bessel ratio is:
///
/// ```text
/// x = 2 · Fo · D · |Fc| / (ε · σ²)
/// ```
///
/// where σ² is the model error variance and ε is the multiplicity factor
/// (set to 1.0 in v1). If σ² or ε is near zero the perfect-model limit
/// *m* = 1 is returned. The argument is clamped at 700 to avoid overflow.
#[must_use]
pub fn figure_of_merit(
    fo: f64,
    d: f64,
    fc_amp: f64,
    sigma_sq: f64,
    epsilon: f64,
) -> f64 {
    let denom = epsilon * sigma_sq;
    if denom.abs() < NEAR_ZERO {
        return 1.0;
    }

    let x = 2.0 * fo * d * fc_amp / denom;

    if x > FOM_X_CLAMP {
        return 1.0;
    }

    bessel_i1_over_i0(x)
}

/// Weighted map coefficients for electron-density synthesis.
///
/// Contains both the primary `2mFo − DFc` map and the difference `mFo − DFc`
/// map, each stored as complex `[re, im]` pairs indexed to match the input
/// reflection list.
#[derive(Debug, Clone)]
pub struct MapCoefficients {
    /// 2mFo-DFc map coefficients `[re, im]` per reflection.
    pub two_fo_fc: Vec<[f32; 2]>,
    /// mFo-DFc difference map coefficients `[re, im]` per reflection.
    pub fo_fc: Vec<[f32; 2]>,
}

/// Compute ML-weighted map coefficients for every reflection.
///
/// For each reflection the sigma-A parameters D and σ² are interpolated at the
/// reflection's resolution, the figure of merit *m* is computed, and the
/// `2mFo − DFc` and `mFo − DFc` amplitudes are combined with the calculated
/// phase.
///
/// When `f_obs` is zero (missing data), the best estimate `D · Fc` is used for
/// the primary map and the difference coefficient is set to zero.
#[must_use]
pub fn compute_map_coefficients(
    reflections: &[Reflection],
    fc: &[[f32; 2]],
    sigma_a: &SigmaAResult,
    unit_cell: &UnitCell,
) -> MapCoefficients {
    let n = reflections.len();
    let mut two_fo_fc = Vec::with_capacity(n);
    let mut fo_fc = Vec::with_capacity(n);

    for (i, refl) in reflections.iter().enumerate() {
        let s2 = unit_cell.d_star_sq(refl.h, refl.k, refl.l);
        let (d_val, sigma_sq) = sigma_a.interpolate(s2);

        let fc_re = f64::from(fc[i][0]);
        let fc_im = f64::from(fc[i][1]);
        let fc_amp = fc_re.hypot(fc_im);
        let fc_phase = fc_im.atan2(fc_re);

        let f_obs = f64::from(refl.f_obs);

        if f_obs.abs() < f64::EPSILON {
            // Missing observation: use best estimate D·Fc.
            let amp = d_val * fc_amp;
            #[allow(clippy::cast_possible_truncation)]
            {
                two_fo_fc.push([
                    (amp * fc_phase.cos()) as f32,
                    (amp * fc_phase.sin()) as f32,
                ]);
                fo_fc.push([0.0_f32, 0.0_f32]);
            }
        } else {
            let m = figure_of_merit(f_obs, d_val, fc_amp, sigma_sq, 1.0);

            let amp_2fo = 2.0_f64.mul_add(m * f_obs, -(d_val * fc_amp));
            let amp_diff = m.mul_add(f_obs, -(d_val * fc_amp));

            #[allow(clippy::cast_possible_truncation)]
            {
                two_fo_fc.push([
                    (amp_2fo * fc_phase.cos()) as f32,
                    (amp_2fo * fc_phase.sin()) as f32,
                ]);
                fo_fc.push([
                    (amp_diff * fc_phase.cos()) as f32,
                    (amp_diff * fc_phase.sin()) as f32,
                ]);
            }
        }
    }

    MapCoefficients { two_fo_fc, fo_fc }
}

/// Place map coefficients onto a 3D reciprocal-space grid and inverse-FFT to
/// obtain a real-space electron-density map.
///
/// Each reflection `(h, k, l)` is mapped to the corresponding grid position
/// (with negative-index wrapping). The Friedel mate `(−h, −k, −l)` is set to
/// the complex conjugate so the resulting map is real-valued.
///
/// # Errors
///
/// Returns [`FftError`] if the inverse FFT fails (e.g. dimension mismatch).
pub fn map_from_coefficients(
    coefficients: &[[f32; 2]],
    reflections: &[Reflection],
    nu: usize,
    nv: usize,
    nw: usize,
) -> Result<Vec<f32>, FftError> {
    let n_total = nu * nv * nw;
    let mut grid = vec![[0.0_f32; 2]; n_total];

    for (i, refl) in reflections.iter().enumerate() {
        let coeff = coefficients[i];

        // Map Miller index to grid position with negative wrapping.
        let u = wrap_index(refl.h, nu);
        let v = wrap_index(refl.k, nv);
        let w = wrap_index(refl.l, nw);

        let idx = u * nv * nw + v * nw + w;
        grid[idx] = coeff;

        // Friedel mate: (-h, -k, -l) gets the complex conjugate.
        let fu = wrap_index(-refl.h, nu);
        let fv = wrap_index(-refl.k, nv);
        let fw = wrap_index(-refl.l, nw);

        let fidx = fu * nv * nw + fv * nw + fw;
        grid[fidx] = [coeff[0], -coeff[1]];
    }

    fft_3d_inverse(&grid, nu, nv, nw)
}

/// Wrap a signed Miller index into the range `[0, size)`.
#[must_use]
fn wrap_index(idx: i32, size: usize) -> usize {
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let s = size as i32;
    let wrapped = ((idx % s) + s) % s;
    #[allow(clippy::cast_sign_loss)]
    {
        wrapped as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FOM at x = 0 should be 0 (I₁(0)/I₀(0) = 0).
    #[test]
    fn fom_at_zero() {
        let m = figure_of_merit(0.0, 1.0, 1.0, 1.0, 1.0);
        assert!(m.abs() < 1e-10, "FOM with fo=0 should be ~0, got {m}");
    }

    /// FOM at very large x should approach 1.
    #[test]
    fn fom_large_x() {
        let m = figure_of_merit(1000.0, 1.0, 1000.0, 0.001, 1.0);
        assert!(
            (m - 1.0).abs() < 1e-3,
            "FOM at large x should be ~1.0, got {m}"
        );
    }

    /// FOM with sigma_sq near zero returns 1.0 (perfect model limit).
    #[test]
    fn fom_zero_sigma() {
        let m = figure_of_merit(10.0, 0.9, 10.0, 0.0, 1.0);
        assert!(
            (m - 1.0).abs() < f64::EPSILON,
            "FOM with sigma_sq=0 should be exactly 1.0, got {m}"
        );
    }

    /// FOM with epsilon near zero returns 1.0.
    #[test]
    fn fom_zero_epsilon() {
        let m = figure_of_merit(10.0, 0.9, 10.0, 1.0, 0.0);
        assert!(
            (m - 1.0).abs() < f64::EPSILON,
            "FOM with epsilon=0 should be exactly 1.0, got {m}"
        );
    }

    /// With D ≈ 1 and very small sigma_sq, the 2mFo-DFc coefficient should
    /// approximate the classical 2Fo - Fc.
    #[test]
    fn classical_limit_2fo_fc() {
        let refl = Reflection {
            h: 1,
            k: 0,
            l: 0,
            f_obs: 10.0,
            sigma_f: 1.0,
            free_flag: false,
        };
        let fc_val = [8.0_f32, 0.0_f32];
        let sigma_a_result = SigmaAResult {
            d_bins: [1.0; 20],
            sigma_sq_bins: [1e-6; 20],
            s2_min: 0.0,
            bin_width: 1.0,
        };
        let cell = UnitCell::new(50.0, 60.0, 70.0, 90.0, 90.0, 90.0);

        let result = compute_map_coefficients(
            &[refl],
            &[fc_val],
            &sigma_a_result,
            &cell,
        );

        // Classical: 2*Fo - Fc = 2*10 - 8 = 12.
        // With D=1 and very small sigma_sq, m ≈ 1, so 2*m*Fo - D*Fc ≈ 12.
        let amp = result.two_fo_fc[0][0].hypot(result.two_fo_fc[0][1]);
        assert!(
            (f64::from(amp) - 12.0).abs() < 0.1,
            "2mFo-DFc amplitude should be ~12, got {amp}"
        );
    }

    /// Missing Fobs (f_obs = 0): coefficient should equal D * Fc.
    #[test]
    fn missing_fobs_uses_d_fc() {
        let refl = Reflection {
            h: 1,
            k: 0,
            l: 0,
            f_obs: 0.0,
            sigma_f: 1.0,
            free_flag: false,
        };
        let fc_val = [6.0_f32, 3.0_f32];
        let d_val = 0.8;
        let sigma_a_result = SigmaAResult {
            d_bins: [d_val; 20],
            sigma_sq_bins: [0.01; 20],
            s2_min: 0.0,
            bin_width: 1.0,
        };
        let cell = UnitCell::new(50.0, 60.0, 70.0, 90.0, 90.0, 90.0);

        let result = compute_map_coefficients(
            &[refl],
            &[fc_val],
            &sigma_a_result,
            &cell,
        );

        let fc_amp = f64::from(fc_val[0]).hypot(f64::from(fc_val[1]));
        let expected_amp = d_val * fc_amp;

        let got_amp = f64::from(result.two_fo_fc[0][0])
            .hypot(f64::from(result.two_fo_fc[0][1]));
        assert!(
            (got_amp - expected_amp).abs() < 0.01,
            "missing Fobs: expected amp {expected_amp}, got {got_amp}"
        );

        // Difference map should be zero.
        let diff_amp =
            f64::from(result.fo_fc[0][0]).hypot(f64::from(result.fo_fc[0][1]));
        assert!(
            diff_amp < 1e-6,
            "missing Fobs difference map should be zero, got {diff_amp}"
        );
    }

    /// Single DC coefficient (h=k=l=0) should produce a constant grid after
    /// inverse FFT.
    #[test]
    fn map_from_coefficients_dc_constant() {
        let nu = 4;
        let nv = 4;
        let nw = 4;
        let dc_value = 2.5_f32;

        // One reflection at (0,0,0) with a real-valued coefficient.
        let refl = Reflection {
            h: 0,
            k: 0,
            l: 0,
            f_obs: 10.0,
            sigma_f: 1.0,
            free_flag: false,
        };
        let coeffs = [[dc_value, 0.0_f32]];

        let grid = map_from_coefficients(&coeffs, &[refl], nu, nv, nw);
        assert!(grid.is_ok(), "map_from_coefficients should succeed");

        // The inverse FFT divides by N, so a DC-only input gives dc_value / N
        // at every grid point.
        let grid = grid.unwrap_or_default();
        #[allow(clippy::cast_precision_loss)]
        let n = (nu * nv * nw) as f32;
        let expected = dc_value / n;

        for (i, &val) in grid.iter().enumerate() {
            assert!(
                (val - expected).abs() < 1e-5,
                "grid[{i}] = {val}, expected {expected}"
            );
        }
    }

    /// Wrap index maps negative Miller indices correctly.
    #[test]
    fn wrap_index_negative() {
        assert_eq!(wrap_index(-1, 10), 9);
        assert_eq!(wrap_index(-3, 8), 5);
        assert_eq!(wrap_index(0, 10), 0);
        assert_eq!(wrap_index(3, 10), 3);
    }
}
