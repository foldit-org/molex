//! Maximum-likelihood target functions for crystallographic refinement.
//!
//! Provides the ML negative log-likelihood, R-factor metrics, B-factor gradient
//! maps via FFT, trilinear interpolation on periodic grids, and simple
//! projected gradient descent for B-factor refinement.

use super::bessel::{bessel_i1_over_i0, log_bessel_i0};
use super::fft_cpu::{fft_3d_inverse, FftError};
use super::sigma_a::SigmaAResult;
use super::types::{Reflection, UnitCell};

/// Wrap a signed Miller index into `[0, size)` using modular arithmetic.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn wrap_idx(idx: i32, size: usize) -> usize {
    let s = size as i32;
    let wrapped = ((idx % s) + s) % s;
    wrapped as usize
}

// ── ML target ───────────────────────────────────────────────────────

/// Compute the total negative log-likelihood over all working-set reflections.
///
/// For each reflection with `free_flag == false`, accumulates:
///
/// ```text
/// nll += (fo² + D² * fc²) / d - log_bessel_i0(x)
/// ```
///
/// where `d = epsilon * sigma_sq`, `x = 2 * fo * D * fc / d`, and epsilon is
/// simplified to 1.0 for this initial version.
///
/// The constant `log(d)` term is omitted because it does not affect gradients.
#[must_use]
pub fn ml_target_value(
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

        // epsilon = 1.0 (simplified for v1)
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
/// R-work = Σ|Fo − k·Fc| / Σ|Fo|.
///
/// Returns 0.0 if the denominator is zero (no working-set reflections or all
/// Fo values are zero).
#[must_use]
pub fn r_work(
    reflections: &[Reflection],
    fc_amplitudes: &[f64],
    k_overall: f64,
) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    for (i, refl) in reflections.iter().enumerate() {
        if refl.free_flag {
            continue;
        }
        let fo = f64::from(refl.f_obs);
        let fc = fc_amplitudes[i];
        num += k_overall.mul_add(-fc, fo).abs();
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
/// R-free = Σ|Fo − k·Fc| / Σ|Fo|.
///
/// Returns 0.0 if there are no free-set reflections.
#[must_use]
pub fn r_free_value(
    reflections: &[Reflection],
    fc_amplitudes: &[f64],
    k_overall: f64,
) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    for (i, refl) in reflections.iter().enumerate() {
        if !refl.free_flag {
            continue;
        }
        let fo = f64::from(refl.f_obs);
        let fc = fc_amplitudes[i];
        num += k_overall.mul_add(-fc, fo).abs();
        den += fo.abs();
    }

    if den < f64::EPSILON {
        0.0
    } else {
        num / den
    }
}

// ── B-factor gradient map via FFT ───────────────────────────────────

/// Compute a B-factor gradient map via inverse FFT.
///
/// For each reflection the gradient coefficient is:
///
/// ```text
/// m = I₁(x) / I₀(x)   where x = 2·fo·D·fc_amp / (ε·σ²)
/// F_grad = m·fo·exp(iφ) − D·Fc
/// F_Bgrad = F_grad · (−s²/4)
/// ```
///
/// These coefficients are placed onto a complex 3D grid at the Miller index
/// positions (with Friedel mates set to conjugates), then inverse-FFT'd to
/// produce the real-space B-factor gradient map.
///
/// # Errors
///
/// Returns [`FftError`] if the inverse FFT fails (dimension mismatch).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]
pub fn b_factor_gradient_map(
    reflections: &[Reflection],
    fc: &[[f32; 2]],
    sigma_a: &SigmaAResult,
    unit_cell: &UnitCell,
    nu: usize,
    nv: usize,
    nw: usize,
) -> Result<Vec<f32>, FftError> {
    let n_total = nu * nv * nw;
    let mut grid = vec![[0.0_f32; 2]; n_total];

    for (i, refl) in reflections.iter().enumerate() {
        if refl.free_flag {
            continue;
        }

        let fo = f64::from(refl.f_obs);
        let fc_re = f64::from(fc[i][0]);
        let fc_im = f64::from(fc[i][1]);
        let fc_amp = fc_re.hypot(fc_im);

        if fc_amp < 1e-30 {
            continue;
        }

        let s2 = unit_cell.d_star_sq(refl.h, refl.k, refl.l);
        let (d_val, sigma_sq) = sigma_a.interpolate(s2);

        // epsilon = 1.0 (simplified)
        let x = 2.0 * fo * d_val * fc_amp / sigma_sq;
        let m = bessel_i1_over_i0(x);

        // Phase of Fc: exp(i*phase) = Fc / |Fc|
        let cos_phase = fc_re / fc_amp;
        let sin_phase = fc_im / fc_amp;

        // F_grad = m * fo * exp(i*phase) - D * Fc
        let m_fo = m * fo;
        let grad_re = m_fo.mul_add(cos_phase, -(d_val * fc_re));
        let grad_im = m_fo.mul_add(sin_phase, -(d_val * fc_im));

        // F_Bgrad = F_grad * (-s²/4)
        let stol2 = s2 / 4.0;
        let b_grad_re = grad_re * (-stol2);
        let b_grad_im = grad_im * (-stol2);

        // Place on grid at Miller index (h, k, l) with periodic wrapping.
        // Use modular arithmetic so indices with |h| > nu are safe.
        let iu = wrap_idx(refl.h, nu);
        let iv = wrap_idx(refl.k, nv);
        let iw = wrap_idx(refl.l, nw);

        if iu < nu && iv < nv && iw < nw {
            let idx = (iu * nv + iv) * nw + iw;
            grid[idx][0] += b_grad_re as f32;
            grid[idx][1] += b_grad_im as f32;

            // Friedel mate: F(-h,-k,-l) = conj(F(h,k,l))
            let fu = wrap_idx(-refl.h, nu);
            let fv = wrap_idx(-refl.k, nv);
            let fw = wrap_idx(-refl.l, nw);

            if fu < nu && fv < nv && fw < nw {
                let fidx = (fu * nv + fv) * nw + fw;
                grid[fidx][0] += b_grad_re as f32;
                grid[fidx][1] -= b_grad_im as f32;
            }
        }
    }

    fft_3d_inverse(&grid, nu, nv, nw)
}

// ── Trilinear interpolation ─────────────────────────────────────────

/// Interpolate a grid value at a fractional coordinate using trilinear
/// interpolation with periodic wrapping.
///
/// The fractional coordinates are mapped into grid space and the value is
/// computed as a weighted average of the 8 surrounding grid points.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
pub fn trilinear_interpolate(
    grid: &[f32],
    nu: usize,
    nv: usize,
    nw: usize,
    frac: [f64; 3],
) -> f32 {
    // Map fractional coords to grid space with wrapping into [0, n)
    let u_f = (frac[0] * nu as f64).rem_euclid(nu as f64);
    let v_f = (frac[1] * nv as f64).rem_euclid(nv as f64);
    let w_f = (frac[2] * nw as f64).rem_euclid(nw as f64);

    let u0 = u_f.floor() as usize;
    let v0 = v_f.floor() as usize;
    let w0 = w_f.floor() as usize;

    let u1 = (u0 + 1) % nu;
    let v1 = (v0 + 1) % nv;
    let w1 = (w0 + 1) % nw;

    let fu = (u_f - u0 as f64) as f32;
    let fv = (v_f - v0 as f64) as f32;
    let fw = (w_f - w0 as f64) as f32;

    let c000 = grid[(u0 * nv + v0) * nw + w0];
    let c001 = grid[(u0 * nv + v0) * nw + w1];
    let c010 = grid[(u0 * nv + v1) * nw + w0];
    let c011 = grid[(u0 * nv + v1) * nw + w1];
    let c100 = grid[(u1 * nv + v0) * nw + w0];
    let c101 = grid[(u1 * nv + v0) * nw + w1];
    let c110 = grid[(u1 * nv + v1) * nw + w0];
    let c111 = grid[(u1 * nv + v1) * nw + w1];

    let c00 = c000.mul_add(1.0 - fw, c001 * fw);
    let c01 = c010.mul_add(1.0 - fw, c011 * fw);
    let c10 = c100.mul_add(1.0 - fw, c101 * fw);
    let c11 = c110.mul_add(1.0 - fw, c111 * fw);

    let c0 = c00.mul_add(1.0 - fv, c01 * fv);
    let c1 = c10.mul_add(1.0 - fv, c11 * fv);

    c0.mul_add(1.0 - fu, c1 * fu)
}

// ── Per-atom B gradient ─────────────────────────────────────────────

/// Compute the per-atom B-factor gradient by interpolating the gradient map.
///
/// For each atom: `dL/dB = occupancy * interpolated_value / volume`.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
pub fn per_atom_b_gradient(
    gradient_map: &[f32],
    positions: &[[f64; 3]],
    occupancies: &[f64],
    nu: usize,
    nv: usize,
    nw: usize,
    volume: f64,
) -> Vec<f64> {
    positions
        .iter()
        .zip(occupancies.iter())
        .map(|(pos, &occ)| {
            let val = f64::from(trilinear_interpolate(
                gradient_map,
                nu,
                nv,
                nw,
                *pos,
            ));
            occ * val / volume
        })
        .collect()
}

// ── B-factor refinement step ────────────────────────────────────────

/// Minimum allowed B-factor (Ų).
const B_MIN: f64 = 2.0;

/// Maximum allowed B-factor (Ų).
const B_MAX: f64 = 300.0;

/// Perform a simple projected gradient descent step on B-factors.
///
/// Updates each B-factor as `B_new = clamp(B - lr * grad, 2.0, 300.0)`.
pub fn refine_b_factors_step(
    b_factors: &mut [f64],
    gradients: &[f64],
    learning_rate: f64,
) {
    for (b, &g) in b_factors.iter_mut().zip(gradients.iter()) {
        *b = learning_rate.mul_add(-g, *b).clamp(B_MIN, B_MAX);
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
    fn ml_target_decreases_when_fc_closer_to_fo() {
        let cell = test_cell();
        let sa = uniform_sigma_a(0.9, 100.0);
        let refls = make_working_reflections();

        // Fc = 0.5 * Fo (poor model)
        let fc_poor: Vec<f64> =
            refls.iter().map(|r| f64::from(r.f_obs) * 0.5).collect();
        // Fc = 0.9 * Fo (better model)
        let fc_good: Vec<f64> =
            refls.iter().map(|r| f64::from(r.f_obs) * 0.9).collect();

        let nll_poor = ml_target_value(&refls, &fc_poor, &sa, &cell);
        let nll_good = ml_target_value(&refls, &fc_good, &sa, &cell);

        assert!(
            nll_good < nll_poor,
            "better model should have lower NLL: good={nll_good}, \
             poor={nll_poor}"
        );
    }

    #[test]
    fn r_work_zero_when_fc_equals_fo() {
        let refls = make_working_reflections();
        let fc: Vec<f64> = refls.iter().map(|r| f64::from(r.f_obs)).collect();

        let rw = r_work(&refls, &fc, 1.0);
        assert!(
            rw.abs() < 1e-10,
            "R-work should be zero for perfect model, got {rw}"
        );
    }

    #[test]
    fn r_free_value_returns_zero_when_no_free_set() {
        let refls = make_working_reflections(); // all free_flag == false
        let fc: Vec<f64> = refls.iter().map(|r| f64::from(r.f_obs)).collect();

        let rf = r_free_value(&refls, &fc, 1.0);
        assert!(
            rf.abs() < f64::EPSILON,
            "R-free should be 0 with no free reflections, got {rf}"
        );
    }

    #[test]
    fn trilinear_constant_grid_returns_constant() {
        let nu = 4;
        let nv = 4;
        let nw = 4;
        let val = 7.5_f32;
        let grid = vec![val; nu * nv * nw];

        // Sample at various fractional positions
        let positions = [
            [0.0, 0.0, 0.0],
            [0.25, 0.5, 0.75],
            [0.1, 0.9, 0.3],
            [0.999, 0.001, 0.5],
        ];

        for pos in &positions {
            let result = trilinear_interpolate(&grid, nu, nv, nw, *pos);
            assert!(
                (result - val).abs() < 1e-5,
                "constant grid should give {val} at {pos:?}, got {result}"
            );
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn trilinear_linear_gradient_field() {
        // Grid values increase linearly along u: value = u_index
        let nu = 8;
        let nv = 4;
        let nw = 4;
        let mut grid = vec![0.0_f32; nu * nv * nw];

        for u in 0..nu {
            for v in 0..nv {
                for w in 0..nw {
                    grid[(u * nv + v) * nw + w] = u as f32;
                }
            }
        }

        // Interpolate at u = 2.5 (frac = 2.5/8 = 0.3125)
        let result =
            trilinear_interpolate(&grid, nu, nv, nw, [0.3125, 0.0, 0.0]);
        assert!(
            (result - 2.5).abs() < 1e-4,
            "expected 2.5 at u=2.5, got {result}"
        );

        // Interpolate at u = 5.25 (frac = 5.25/8 = 0.65625)
        let result2 =
            trilinear_interpolate(&grid, nu, nv, nw, [0.65625, 0.0, 0.0]);
        assert!(
            (result2 - 5.25).abs() < 1e-4,
            "expected 5.25 at u=5.25, got {result2}"
        );
    }

    #[test]
    fn per_atom_b_gradient_smoke_test() {
        let nu = 4;
        let nv = 4;
        let nw = 4;
        let n = nu * nv * nw;

        // Uniform gradient map with value 2.0
        let gradient_map = vec![2.0_f32; n];
        let positions = [[0.25, 0.25, 0.25]];
        let occupancies = [1.0];
        let volume = 1000.0;

        let grads = per_atom_b_gradient(
            &gradient_map,
            &positions,
            &occupancies,
            nu,
            nv,
            nw,
            volume,
        );

        assert_eq!(grads.len(), 1);
        // Expected: occ * 2.0 / 1000.0 = 0.002
        assert!(
            (grads[0] - 0.002).abs() < 1e-6,
            "expected ~0.002, got {}",
            grads[0]
        );
    }

    #[test]
    fn refine_b_factors_step_moves_in_gradient_direction() {
        let mut b_factors = vec![20.0, 50.0, 100.0];
        // Positive gradient means NLL increases with B, so descent should
        // decrease B
        let gradients = vec![1.0, -1.0, 0.5];
        let lr = 2.0;

        refine_b_factors_step(&mut b_factors, &gradients, lr);

        // B[0] = clamp(20 - 2*1, 2, 300) = 18.0
        assert!(
            (b_factors[0] - 18.0).abs() < 1e-10,
            "expected 18.0, got {}",
            b_factors[0]
        );
        // B[1] = clamp(50 - 2*(-1), 2, 300) = 52.0
        assert!(
            (b_factors[1] - 52.0).abs() < 1e-10,
            "expected 52.0, got {}",
            b_factors[1]
        );
        // B[2] = clamp(100 - 2*0.5, 2, 300) = 99.0
        assert!(
            (b_factors[2] - 99.0).abs() < 1e-10,
            "expected 99.0, got {}",
            b_factors[2]
        );
    }

    #[test]
    fn refine_b_factors_step_clamps_to_bounds() {
        let mut b_factors = vec![3.0, 299.0];
        // Large positive gradient drives B[0] below min
        // Large negative gradient drives B[1] above max
        let gradients = vec![100.0, -100.0];
        let lr = 1.0;

        refine_b_factors_step(&mut b_factors, &gradients, lr);

        assert!(
            (b_factors[0] - B_MIN).abs() < 1e-10,
            "should clamp to B_MIN, got {}",
            b_factors[0]
        );
        assert!(
            (b_factors[1] - B_MAX).abs() < 1e-10,
            "should clamp to B_MAX, got {}",
            b_factors[1]
        );
    }
}
