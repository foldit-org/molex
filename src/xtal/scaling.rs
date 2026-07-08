//! Staged bounded scaling of calculated structure factors against observed.
//!
//! Fits four scalars — overall scale, isotropic B, and the bulk-solvent
//! `(k_sol, b_sol)` — to the model
//!
//! ```text
//! |F_model(h)| = k_overall * exp(-B_iso * s²)
//!                * |Fc(h) + k_sol * exp(-b_sol * s²) * Fmask(h)|
//! ```
//!
//! with `s² = d*²/4`. The fit runs in stages: a closed-form Wilson estimate of
//! `(k_overall, B_iso)` with solvent off, a bounded grid scan plus local refine
//! of `(k_sol, b_sol)` at fixed scale, and an optional joint polish. Every
//! stage is box-clamped and only accepted when it lowers the weighted residual,
//! so the result never scores worse than the Wilson-only solution.

use super::types::{Reflection, UnitCell};

// ── Result type ──────────────────────────────────────────────────────

/// Result of the bulk-solvent + isotropic scaling fit.
#[derive(Debug, Clone)]
pub struct ScalingResult {
    /// Overall isotropic scale factor.
    pub k_overall: f64,
    /// Isotropic B-factor (Å²) applied as `exp(-B_iso * s²)`.
    pub b_iso: f64,
    /// Bulk-solvent scale factor.
    pub k_sol: f64,
    /// Bulk-solvent B factor (Å²).
    pub b_sol: f64,
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
    let b_clamped = b_iso.clamp(B_ISO_LO, B_ISO_HI);

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

// ── Staged bounded fit ───────────────────────────────────────────────

/// Convergence threshold on relative change in weighted sum of squared
/// residuals.
const REL_TOL: f64 = 1e-5;
/// Maximum LM iterations per refine stage.
const MAX_LM_ITER: usize = 40;

/// Box bounds. `k_overall` is only floored; the others are two-sided.
const K_OVERALL_MIN: f64 = 1e-6;
const B_ISO_LO: f64 = 1.0;
const B_ISO_HI: f64 = 200.0;
const K_SOL_LO: f64 = 0.0;
const K_SOL_HI: f64 = 0.6;
const B_SOL_LO: f64 = 10.0;
const B_SOL_HI: f64 = 200.0;

/// Bulk-solvent grid-scan nodes (canonical coarse bracket around the physical
/// basin) used to seed the local `(k_sol, b_sol)` refine.
const K_SOL_GRID: [f64; 13] = [
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
];
const B_SOL_GRID: [f64; 12] = [
    10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
];

/// The four fitted scalars in optimizer order: `[k_overall, b_iso, k_sol,
/// b_sol]`.
type Params = [f64; 4];

/// Clamp a parameter vector into the physical box.
fn clamp_params(p: &mut Params) {
    p[0] = p[0].max(K_OVERALL_MIN);
    p[1] = p[1].clamp(B_ISO_LO, B_ISO_HI);
    p[2] = p[2].clamp(K_SOL_LO, K_SOL_HI);
    p[3] = p[3].clamp(B_SOL_LO, B_SOL_HI);
}

/// Fit the overall scale, isotropic B, and bulk-solvent parameters to the
/// observed data with a staged, box-bounded protocol.
///
/// # Arguments
///
/// * `reflections` - observed reflection data
/// * `fc` - complex calculated structure factors `[re, im]` per reflection
/// * `f_mask` - complex bulk-solvent mask structure factors `[re, im]` per
///   reflection
/// * `unit_cell` - crystallographic unit cell
///
/// Only working-set reflections (where `free_flag == false`) contribute to the
/// fit. The returned result never scores a worse weighted residual than the
/// closed-form Wilson (scale + isotropic B, solvent off) solution.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn fit_scaling(
    reflections: &[Reflection],
    fc: &[[f32; 2]],
    f_mask: &[[f32; 2]],
    unit_cell: &UnitCell,
) -> ScalingResult {
    // Working set and per-reflection stol2.
    let mut work_idx: Vec<usize> = Vec::new();
    for (i, refl) in reflections.iter().enumerate() {
        if !refl.free_flag && refl.sigma_f > 0.0 {
            work_idx.push(i);
        }
    }
    let stol2: Vec<f64> = reflections
        .iter()
        .map(|r| unit_cell.d_star_sq(r.h, r.k, r.l) * 0.25)
        .collect();

    // Stage A: Wilson scale + isotropic B, solvent off. Closed-form and stable.
    let fc_amps: Vec<f64> = fc
        .iter()
        .map(|c| f64::from(c[0]).hypot(f64::from(c[1])))
        .collect();
    let (k_init, b_init) = wilson_estimate(reflections, &fc_amps, unit_cell);

    let mut best: Params = [k_init, b_init, 0.0, 46.0];
    let mut best_wssr =
        compute_wssr(&best, reflections, fc, f_mask, &stol2, &work_idx);

    // Stage B: bulk solvent by bounded grid scan (scale/B fixed), then a local
    // bounded LM refine over (k_sol, b_sol) only.
    let mut grid_best = best;
    let mut grid_wssr = best_wssr;
    for &ks in &K_SOL_GRID {
        for &bs in &B_SOL_GRID {
            let cand: Params = [best[0], best[1], ks, bs];
            let w =
                compute_wssr(&cand, reflections, fc, f_mask, &stol2, &work_idx);
            if w < grid_wssr {
                grid_wssr = w;
                grid_best = cand;
            }
        }
    }
    let (b_params, b_wssr) = lm_refine(
        grid_best,
        [false, false, true, true],
        reflections,
        fc,
        f_mask,
        &stol2,
        &work_idx,
    );
    if b_wssr < best_wssr {
        best = b_params;
        best_wssr = b_wssr;
    }

    // Stage C: joint polish over all four, box-clamped each step. Seeded near
    // the basin by A/B, so it only sharpens.
    let (c_params, c_wssr) = lm_refine(
        best,
        [true, true, true, true],
        reflections,
        fc,
        f_mask,
        &stol2,
        &work_idx,
    );
    if c_wssr < best_wssr {
        best = c_params;
    }

    ScalingResult {
        k_overall: best[0],
        b_iso: best[1],
        k_sol: best[2],
        b_sol: best[3],
    }
}

/// Bounded Levenberg-Marquardt refine over the parameters flagged in `active`.
///
/// Frozen parameters keep their seed value; every accepted step is box-clamped.
/// Returns the refined parameters and their weighted residual. The seed's
/// residual is the floor: a stage that cannot improve returns the seed
/// unchanged.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::too_many_lines
)]
fn lm_refine(
    seed: Params,
    active: [bool; 4],
    reflections: &[Reflection],
    fc: &[[f32; 2]],
    f_mask: &[[f32; 2]],
    stol2: &[f64],
    work_idx: &[usize],
) -> (Params, f64) {
    let idxs: Vec<usize> = (0..4).filter(|&i| active[i]).collect();
    let m = idxs.len();
    if m == 0 {
        let w = compute_wssr(&seed, reflections, fc, f_mask, stol2, work_idx);
        return (seed, w);
    }

    let mut params = seed;
    let mut best_wssr =
        compute_wssr(&params, reflections, fc, f_mask, stol2, work_idx);
    let mut lambda = 1e-3_f64;

    for _ in 0..MAX_LM_ITER {
        // Normal equations over the active subset only.
        let mut jtj = vec![0.0_f64; m * m];
        let mut jtr = vec![0.0_f64; m];
        for &i in work_idx {
            let refl = &reflections[i];
            let fo = f64::from(refl.f_obs);
            let Some((y, jac)) = model_and_jac(
                &params,
                f64::from(fc[i][0]),
                f64::from(fc[i][1]),
                f64::from(f_mask[i][0]),
                f64::from(f_mask[i][1]),
                stol2[i],
            ) else {
                continue;
            };
            let r = fo - y;
            for (a, &pa) in idxs.iter().enumerate() {
                jtr[a] += jac[pa] * r;
                for (b, &pb) in idxs.iter().enumerate() {
                    jtj[a * m + b] += jac[pa] * jac[pb];
                }
            }
        }

        // Marquardt damping on the diagonal.
        let mut damped = jtj.clone();
        for a in 0..m {
            let d = jtj[a * m + a];
            damped[a * m + a] = d + lambda * d.max(1e-12);
        }
        let Some(delta) = solve_spd(&damped, &jtr, m) else {
            lambda *= 10.0;
            if lambda > 1e15 {
                break;
            }
            continue;
        };

        let mut trial = params;
        for (a, &pa) in idxs.iter().enumerate() {
            trial[pa] += delta[a];
        }
        clamp_params(&mut trial);

        let tw = compute_wssr(&trial, reflections, fc, f_mask, stol2, work_idx);
        if tw < best_wssr {
            let rel = (best_wssr - tw) / best_wssr.max(1e-30);
            params = trial;
            best_wssr = tw;
            lambda = (lambda * 0.1).max(1e-15);
            if rel < REL_TOL {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e15 {
                break;
            }
        }
    }

    (params, best_wssr)
}

// ── Model + derivatives ──────────────────────────────────────────────

/// Compute model amplitude for one reflection given parameters.
///
/// This is the full scaled model the fit optimizes:
/// `k_overall * exp(-B_iso * s²) * |Fc + k_sol * exp(-b_sol * s²) * Fmask|`.
/// The R-factor metrics reuse it so they score the same model rather than a
/// bare `k_overall * |Fc|`.
#[allow(clippy::similar_names, clippy::too_many_arguments)]
pub(crate) fn model_amplitude(
    k_overall: f64,
    k_sol: f64,
    b_sol: f64,
    b_iso: f64,
    fc_re: f64,
    fc_im: f64,
    fm_re: f64,
    fm_im: f64,
    stol2: f64,
) -> f64 {
    let dw = (-b_iso * stol2).exp();
    let solv_b = (-b_sol * stol2).exp();
    let total_re = fc_re + k_sol * solv_b * fm_re;
    let total_im = fc_im + k_sol * solv_b * fm_im;
    k_overall * dw * total_re.hypot(total_im)
}

/// Model amplitude and its Jacobian `[d/dk_overall, d/dB_iso, d/dk_sol,
/// d/db_sol]` for one reflection. Returns `None` when the model amplitude
/// vanishes (the solvent derivatives divide by it).
#[allow(clippy::similar_names, clippy::too_many_arguments)]
fn model_and_jac(
    p: &Params,
    fc_re: f64,
    fc_im: f64,
    fm_re: f64,
    fm_im: f64,
    stol2: f64,
) -> Option<(f64, [f64; 4])> {
    let (k_overall, b_iso, k_sol, b_sol) = (p[0], p[1], p[2], p[3]);
    let dw = (-b_iso * stol2).exp();
    let solv_b = (-b_sol * stol2).exp();
    let total_re = fc_re + k_sol * solv_b * fm_re;
    let total_im = fc_im + k_sol * solv_b * fm_im;
    let amp = total_re.hypot(total_im);
    if amp < 1e-30 {
        return None;
    }
    let y = k_overall * dw * amp;

    // Re(conj(total) · Fmask) / |total|.
    let conj_dot = (total_re * fm_re + total_im * fm_im) / amp;
    let jac = [
        dw * amp,                                            // d/dk_overall
        -stol2 * y,                                          // d/dB_iso
        k_overall * dw * solv_b * conj_dot,                  // d/dk_sol
        -stol2 * k_sol * solv_b * conj_dot * k_overall * dw, // d/db_sol
    ];
    Some((y, jac))
}

/// Sum of squared amplitude residuals over the working set.
///
/// Unit weights (not `1/sigma`): the goal is a scale that minimizes the
/// unweighted `|Fo - F_model|` the R-factor reports, and `1/sigma` weighting
/// on this dataset up-weights the strong low-resolution reflections enough to
/// pull the overall scale off the R-optimal value.
#[allow(clippy::too_many_arguments)]
fn compute_wssr(
    p: &Params,
    reflections: &[Reflection],
    fc: &[[f32; 2]],
    f_mask: &[[f32; 2]],
    stol2: &[f64],
    work_idx: &[usize],
) -> f64 {
    let mut wssr = 0.0_f64;
    for &i in work_idx {
        let refl = &reflections[i];
        let fo = f64::from(refl.f_obs);
        let y = model_amplitude(
            p[0],
            p[2],
            p[3],
            p[1],
            f64::from(fc[i][0]),
            f64::from(fc[i][1]),
            f64::from(f_mask[i][0]),
            f64::from(f_mask[i][1]),
            stol2[i],
        );
        let r = fo - y;
        wssr += r * r;
    }
    wssr
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::suboptimal_flops)]
#[path = "scaling_tests.rs"]
mod tests;
