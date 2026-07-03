//! Levenberg-Marquardt scaling of calculated structure factors against
//! observed.
//!
//! Implements anisotropic scaling with bulk-solvent correction:
//!
//! ```text
//! |F_total(h)| = k_overall * kaniso(h) * |Fc(h) + k_sol * exp(-B_sol * s²) * Fmask(h)|
//! ```
//!
//! where `kaniso(h) = exp(-¼ hᵀ B* h)` and `s² = d*²/4`.

use super::types::{CrystalSystem, Reflection, UnitCell};

// ── Result type ──────────────────────────────────────────────────────

/// Result of the bulk-solvent + anisotropic scaling fit.
#[derive(Debug, Clone)]
pub struct ScalingResult {
    /// Overall isotropic scale factor.
    pub k_overall: f64,
    /// Anisotropic B-tensor components `[B11, B22, B33, B12, B13, B23]`.
    pub b_aniso: [f64; 6],
    /// Bulk-solvent scale factor.
    pub k_sol: f64,
    /// Bulk-solvent B factor (Å²).
    pub b_sol: f64,
}

// ── Anisotropic constraint basis ─────────────────────────────────────

/// Returns the independent anisotropic B-tensor basis vectors for the given
/// crystal system.
///
/// Each basis vector is a 6-component array `[B11, B22, B33, B12, B13, B23]`.
/// The number of vectors equals the number of free anisotropic parameters.
#[must_use]
#[allow(clippy::trivially_copy_pass_by_ref)]
pub fn aniso_constraint_basis(system: &CrystalSystem) -> Vec<[f64; 6]> {
    match system {
        CrystalSystem::Triclinic => vec![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        CrystalSystem::Monoclinic => vec![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], // B13
        ],
        CrystalSystem::Orthorhombic => vec![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        CrystalSystem::Tetragonal
        | CrystalSystem::Trigonal
        | CrystalSystem::Hexagonal => vec![
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0], // B11 = B22
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], // B33
        ],
        CrystalSystem::Cubic => vec![
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], // B11 = B22 = B33
        ],
    }
}

// ── Wilson plot ───────────────────────────────────────────────────────

/// Estimate overall scale and isotropic B-factor from a Wilson plot.
///
/// Returns `(k_initial, b_iso_initial)` by linear regression of
/// `ln(|Fobs| / |Fc|)` against `sin²θ/λ² = d*²/4`.
///
/// Falls back to `(1.0, 20.0)` when there are too few usable reflections or
/// the regression is degenerate.
#[must_use]
#[allow(clippy::suboptimal_flops)]
pub fn wilson_estimate(
    reflections: &[Reflection],
    fc_amplitudes: &[f64],
    unit_cell: &UnitCell,
) -> (f64, f64) {
    let min_points = 10_u64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut n = 0_u64;

    for (refl, &fc_amp) in reflections.iter().zip(fc_amplitudes.iter()) {
        let fo = f64::from(refl.f_obs);
        let sig = f64::from(refl.sigma_f);
        if fo < 1.0 || fo < sig || fc_amp <= 0.0 {
            continue;
        }
        let y = (fo / fc_amp).ln();
        let x = unit_cell.d_star_sq(refl.h, refl.k, refl.l) * 0.25;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        n += 1;
    }

    if n < min_points {
        return (1.0, 20.0);
    }

    #[allow(clippy::cast_precision_loss)]
    let nf = n as f64;
    #[allow(clippy::suspicious_operation_groupings)]
    let denom = nf * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return (1.0, 20.0);
    }

    let slope = (nf * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / nf;

    let k = intercept.exp();
    let b_iso = -slope;

    // Clamp to physically reasonable values.
    let k_clamped = k.clamp(0.01, 100.0);
    let b_clamped = b_iso.clamp(1.0, 200.0);

    (k_clamped, b_clamped)
}

// ── Dense linear solver ──────────────────────────────────────────────

/// Solve a symmetric positive-definite system `A x = b` in place using
/// Cholesky decomposition.
///
/// `a` is stored as a flat row-major `n×n` array; `b` has length `n`.
/// On success, returns `Some(x)`. Returns `None` if the matrix is not
/// positive-definite (zero or negative diagonal encountered).
#[allow(clippy::many_single_char_names)]
fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    // Cholesky factorization: A = L L^T
    let mut l = vec![0.0_f64; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for p in 0..j {
                sum += l[i * n + p] * l[j * n + p];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 {
                    return None;
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = 0.0_f64;
        for j in 0..i {
            sum += l[i * n + j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i * n + i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = 0.0_f64;
        for j in (i + 1)..n {
            sum += l[j * n + i] * x[j];
        }
        x[i] = (y[i] - sum) / l[i * n + i];
    }

    Some(x)
}

// ── Levenberg-Marquardt scaling ──────────────────────────────────────

/// Maximum number of LM iterations.
const MAX_ITER: usize = 100;
/// Convergence threshold on relative change in weighted sum of squared
/// residuals.
const REL_TOL: f64 = 1e-5;

/// Fit the anisotropic scale, overall scale, and bulk-solvent parameters to
/// the observed data using Levenberg-Marquardt optimization.
///
/// # Arguments
///
/// * `reflections` — observed reflection data
/// * `fc` — complex calculated structure factors `[re, im]` per reflection
/// * `f_mask` — complex bulk-solvent mask structure factors `[re, im]` per
///   reflection
/// * `unit_cell` — crystallographic unit cell
/// * `system` — crystal system for anisotropic constraint
///
/// Only working-set reflections (where `free_flag == false`) contribute to the
/// fit.
#[allow(
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref,
    clippy::suboptimal_flops
)]
pub fn fit_scaling(
    reflections: &[Reflection],
    fc: &[[f32; 2]],
    f_mask: &[[f32; 2]],
    unit_cell: &UnitCell,
    system: &CrystalSystem,
) -> ScalingResult {
    let basis = aniso_constraint_basis(system);
    let n_basis = basis.len();
    // Parameter vector: [k_overall, k_sol, B_sol, d_0 .. d_{n_basis-1}]
    let n_params = 3 + n_basis;

    // Collect working-set indices and precompute per-reflection quantities.
    let mut work_idx: Vec<usize> = Vec::new();
    for (i, refl) in reflections.iter().enumerate() {
        if !refl.free_flag && refl.sigma_f > 0.0 {
            work_idx.push(i);
        }
    }

    // Precompute stol2 and hkl as f64 for each reflection.
    let stol2: Vec<f64> = reflections
        .iter()
        .map(|r| unit_cell.d_star_sq(r.h, r.k, r.l) * 0.25)
        .collect();

    // Wilson estimate for initial values.
    let fc_amps: Vec<f64> = fc
        .iter()
        .map(|c| f64::from(c[0]).hypot(f64::from(c[1])))
        .collect();
    let (k_init, b_iso_init) =
        wilson_estimate(reflections, &fc_amps, unit_cell);

    // Build initial parameter vector.
    let mut params = vec![0.0_f64; n_params];
    params[0] = k_init;
    params[1] = 0.35; // k_sol
    params[2] = 46.0; // B_sol

    // Project isotropic B onto constraint basis.
    // Isotropic: B11=B22=B33=B_iso/3, off-diag=0.
    let b_iso_diag = b_iso_init / 3.0;
    let b_iso_vec = [b_iso_diag, b_iso_diag, b_iso_diag, 0.0, 0.0, 0.0];
    for (j, bv) in basis.iter().enumerate() {
        // d_j = dot(basis[j], b_iso_vec) / dot(basis[j], basis[j])
        let num = dot6(bv, &b_iso_vec);
        let den = dot6(bv, bv);
        params[3 + j] = if den > 0.0 { num / den } else { 0.0 };
    }

    // LM loop.
    let mut lambda = 1e-3_f64;
    let mut prev_wssr = compute_wssr(
        &params,
        reflections,
        fc,
        f_mask,
        &stol2,
        &work_idx,
        &basis,
    );
    let mut converge_count = 0u32;

    for _iter in 0..MAX_ITER {
        // Build J^T W J and J^T W r.
        let (jtj, jtr, wssr) = build_normal_equations(
            &params,
            reflections,
            fc,
            f_mask,
            &stol2,
            &work_idx,
            &basis,
            n_params,
        );

        // Apply damping: (J^T W J + lambda * diag) delta = J^T W r.
        let mut jtj_damped = jtj.clone();
        for i in 0..n_params {
            let diag = jtj[i * n_params + i];
            jtj_damped[i * n_params + i] = diag + lambda * diag.max(1e-12);
        }

        let Some(delta) = solve_spd(&jtj_damped, &jtr, n_params) else {
            // If Cholesky fails, increase damping and retry.
            lambda *= 10.0;
            continue;
        };

        // Trial parameters.
        let mut trial = params.clone();
        for i in 0..n_params {
            trial[i] += delta[i];
        }
        // Clamp k_overall > 0 and B_sol > 0.
        if trial[0] < 1e-6 {
            trial[0] = 1e-6;
        }
        if trial[2] < 0.1 {
            trial[2] = 0.1;
        }

        let trial_wssr = compute_wssr(
            &trial,
            reflections,
            fc,
            f_mask,
            &stol2,
            &work_idx,
            &basis,
        );

        if trial_wssr < wssr {
            // Accept step.
            let rel_change = if wssr > 0.0 {
                (wssr - trial_wssr) / wssr
            } else {
                0.0
            };
            params = trial;
            lambda *= 0.1;
            if lambda < 1e-15 {
                lambda = 1e-15;
            }

            if rel_change < REL_TOL {
                converge_count += 1;
                if converge_count >= 2 {
                    break;
                }
            } else {
                converge_count = 0;
            }
            prev_wssr = trial_wssr;
        } else {
            // Reject step.
            lambda *= 10.0;
            if lambda > 1e15 {
                break;
            }
            converge_count = 0;
        }
    }

    let _ = prev_wssr; // suppress unused warning

    // Reconstruct full B-tensor from basis coefficients.
    let mut b_aniso = [0.0_f64; 6];
    for (j, bv) in basis.iter().enumerate() {
        let d_j = params[3 + j];
        for c in 0..6 {
            b_aniso[c] += d_j * bv[c];
        }
    }

    ScalingResult {
        k_overall: params[0],
        b_aniso,
        k_sol: params[1],
        b_sol: params[2],
    }
}

// ── Internal helpers ─────────────────────────────────────────────────

/// Dot product of two 6-element vectors.
#[allow(clippy::suboptimal_flops)]
fn dot6(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    a[0] * b[0]
        + a[1] * b[1]
        + a[2] * b[2]
        + a[3] * b[3]
        + a[4] * b[4]
        + a[5] * b[5]
}

/// Compute `kaniso(h)` given B-tensor components and Miller indices.
#[allow(clippy::suboptimal_flops)]
fn kaniso(b: &[f64; 6], h: f64, k: f64, l: f64) -> f64 {
    let exponent = -0.25
        * (h * h * b[0]
            + k * k * b[1]
            + l * l * b[2]
            + 2.0 * h * k * b[3]
            + 2.0 * h * l * b[4]
            + 2.0 * k * l * b[5]);
    exponent.exp()
}

/// Reconstruct the full 6-component B-tensor from basis coefficients.
fn reconstruct_b(params: &[f64], basis: &[[f64; 6]]) -> [f64; 6] {
    let mut b = [0.0_f64; 6];
    for (j, bv) in basis.iter().enumerate() {
        let d_j = params[3 + j];
        for c in 0..6 {
            b[c] += d_j * bv[c];
        }
    }
    b
}

/// Compute model amplitude for one reflection given parameters.
#[allow(
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::suboptimal_flops
)]
fn model_amplitude(
    k_overall: f64,
    k_sol: f64,
    b_sol: f64,
    b_tensor: &[f64; 6],
    fc_re: f64,
    fc_im: f64,
    fm_re: f64,
    fm_im: f64,
    stol2: f64,
    hf: f64,
    kf: f64,
    lf: f64,
) -> f64 {
    let ka = kaniso(b_tensor, hf, kf, lf);
    let solv_b = (-b_sol * stol2).exp();
    let total_re = fc_re + k_sol * solv_b * fm_re;
    let total_im = fc_im + k_sol * solv_b * fm_im;
    let fc_total_amp = total_re.hypot(total_im);
    k_overall * ka * fc_total_amp
}

/// Weighted sum of squared residuals.
#[allow(clippy::too_many_arguments)]
fn compute_wssr(
    params: &[f64],
    reflections: &[Reflection],
    fc: &[[f32; 2]],
    f_mask: &[[f32; 2]],
    stol2: &[f64],
    work_idx: &[usize],
    basis: &[[f64; 6]],
) -> f64 {
    let k_overall = params[0];
    let k_sol = params[1];
    let b_sol = params[2];
    let b_tensor = reconstruct_b(params, basis);

    let mut wssr = 0.0_f64;
    for &i in work_idx {
        let refl = &reflections[i];
        let fo = f64::from(refl.f_obs);
        let w = 1.0 / f64::from(refl.sigma_f);
        let hf = f64::from(refl.h);
        let kf = f64::from(refl.k);
        let lf = f64::from(refl.l);

        let y_calc = model_amplitude(
            k_overall,
            k_sol,
            b_sol,
            &b_tensor,
            f64::from(fc[i][0]),
            f64::from(fc[i][1]),
            f64::from(f_mask[i][0]),
            f64::from(f_mask[i][1]),
            stol2[i],
            hf,
            kf,
            lf,
        );

        let r = fo - y_calc;
        wssr += w * w * r * r;
    }
    wssr
}

/// Build the normal equations `J^T W J` and `J^T W r` for the LM step.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::suboptimal_flops
)]
fn build_normal_equations(
    params: &[f64],
    reflections: &[Reflection],
    fc: &[[f32; 2]],
    f_mask: &[[f32; 2]],
    stol2: &[f64],
    work_idx: &[usize],
    basis: &[[f64; 6]],
    n_params: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    let k_overall = params[0];
    let k_sol = params[1];
    let b_sol = params[2];
    let b_tensor = reconstruct_b(params, basis);

    let mut jtj = vec![0.0_f64; n_params * n_params];
    let mut jtr = vec![0.0_f64; n_params];
    let mut wssr = 0.0_f64;

    for &i in work_idx {
        let refl = &reflections[i];
        let fo = f64::from(refl.f_obs);
        let w = 1.0 / f64::from(refl.sigma_f);
        let hf = f64::from(refl.h);
        let kf = f64::from(refl.k);
        let lf = f64::from(refl.l);
        let s2 = stol2[i];

        let fc_re = f64::from(fc[i][0]);
        let fc_im = f64::from(fc[i][1]);
        let fm_re = f64::from(f_mask[i][0]);
        let fm_im = f64::from(f_mask[i][1]);

        let ka = kaniso(&b_tensor, hf, kf, lf);
        let solv_b = (-b_sol * s2).exp();
        let total_re = fc_re + k_sol * solv_b * fm_re;
        let total_im = fc_im + k_sol * solv_b * fm_im;
        let fc_total_amp = total_re.hypot(total_im);

        // Avoid division by zero for vanishing Fc.
        if fc_total_amp < 1e-30 {
            continue;
        }

        let y_calc = k_overall * ka * fc_total_amp;
        let r = fo - y_calc;
        wssr += w * w * r * r;

        // Jacobian components.
        let mut jac = vec![0.0_f64; n_params];

        // dy/dk_overall = ka * |fc_total|
        jac[0] = ka * fc_total_amp;

        // Common factor for solvent derivatives.
        // Re(conj(fc_total) . Fmask) / |fc_total|
        let conj_dot_re = (total_re * fm_re + total_im * fm_im) / fc_total_amp;

        // dy/dk_sol
        jac[1] = solv_b * conj_dot_re * k_overall * ka;

        // dy/dB_sol
        jac[2] = -s2 * k_sol * solv_b * conj_dot_re * k_overall * ka;

        // Anisotropic B derivatives.
        // du/dB_ij for the 6 unique components:
        let du = [
            -0.25 * y_calc * hf * hf, // B11
            -0.25 * y_calc * kf * kf, // B22
            -0.25 * y_calc * lf * lf, // B33
            -0.50 * y_calc * hf * kf, // B12
            -0.50 * y_calc * hf * lf, // B13
            -0.50 * y_calc * kf * lf, // B23
        ];

        for (j, bv) in basis.iter().enumerate() {
            jac[3 + j] = dot6(bv, &du);
        }

        // Accumulate into J^T W J and J^T W r.
        let w2 = w * w;
        for p in 0..n_params {
            jtr[p] += w2 * jac[p] * r;
            for q in 0..n_params {
                jtj[p * n_params + q] += w2 * jac[p] * jac[q];
            }
        }
    }

    (jtj, jtr, wssr)
}

#[cfg(test)]
#[allow(
    clippy::many_single_char_names,
    clippy::excessive_nesting,
    clippy::suboptimal_flops
)]
#[path = "scaling_tests.rs"]
mod tests;
