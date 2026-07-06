//! Atomic density splatting onto a 3D grid using Gaussian form factors.
//!
//! Implements real-space density calculation from IT92 scattering factors,
//! symmetry-based grid averaging, and reciprocal-space deblurring.

use std::f64::consts::PI;

use super::form_factors::FormFactor;
use super::types::{DensityGrid, GroupOps, UnitCell, DEN};

// ── Gaussian precalculation ──────────────────────────────────────────

/// Number of radial samples in the per-atom density lookup table. The isotropic
/// kernel is a function of `r_sq` only and fixed per atom, so it is tabulated
/// once and then read (with linear interpolation) at every covered voxel.
const LUT_BINS: usize = 1024;

/// Build the five-Gaussian real-space kernel coefficients `(a, b)` for one atom
/// with effective B-factor `b_atom + blur`: four IT92 terms plus the constant
/// term. The kernel value at squared distance `r_sq` is `Σ
/// a[i]*exp(b[i]*r_sq)`.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn kernel_coeffs(
    ff: &FormFactor,
    b_atom: f64,
    blur: f64,
) -> ([f64; 5], [f64; 5]) {
    let four_pi = 4.0 * PI;
    let mut a = [0.0_f64; 5];
    let mut b = [0.0_f64; 5];

    for i in 0..4 {
        let t = four_pi / (ff.b[i] + b_atom + blur);
        a[i] = ff.a[i] * t.powf(1.5);
        b[i] = -t * PI;
    }

    // Constant term (i = 4).
    let t_c = four_pi / (b_atom + blur);
    a[4] = ff.c * t_c.powf(1.5);
    b[4] = -t_c * PI;

    (a, b)
}

/// Exact (non-LUT) five-Gaussian kernel value at squared distance `r_sq`,
/// before occupancy: `Σ a[i]*exp(b[i]*r_sq)`. Callers multiply by occupancy.
#[must_use]
pub fn kernel_eval(ff: &FormFactor, b_atom: f64, blur: f64, r_sq: f64) -> f64 {
    let (a, b) = kernel_coeffs(ff, b_atom, blur);
    let mut sum = 0.0;
    for i in 0..5 {
        sum = a[i].mul_add((b[i] * r_sq).exp(), sum);
    }
    sum
}

/// Precalculated radial density lookup table for one atom.
///
/// The five-Gaussian kernel (4 IT92 terms + constant) is evaluated once into a
/// table indexed by squared distance over `[0, r_cut_sq]`; each voxel then
/// costs a table lookup plus linear interpolation instead of five `exp` calls.
struct GaussianPrecalc {
    /// Tabulated kernel value at `r_sq_k = k / inv_bin`, before occupancy.
    lut: [f64; LUT_BINS],
    /// Bins per unit `r_sq`: `(LUT_BINS - 1) / r_cut_sq`.
    inv_bin: f64,
    /// Squared cutoff radius; the table spans `[0, r_cut_sq]`.
    r_cut_sq: f64,
}

impl GaussianPrecalc {
    /// Build from a form factor, the atom's isotropic B-factor, the blur, and
    /// the squared cutoff radius (the span the table covers).
    #[allow(clippy::cast_precision_loss)]
    fn new(ff: &FormFactor, b_atom: f64, blur: f64, cutoff_sq: f64) -> Self {
        let (a, b) = kernel_coeffs(ff, b_atom, blur);

        // Tabulate the kernel over [0, cutoff_sq] once per atom; this is the
        // only place `exp` runs.
        let step = cutoff_sq / (LUT_BINS - 1) as f64;
        let mut lut = [0.0_f64; LUT_BINS];
        for (k, slot) in lut.iter_mut().enumerate() {
            let r_sq_k = k as f64 * step;
            let mut sum = 0.0;
            for i in 0..5 {
                sum = a[i].mul_add((b[i] * r_sq_k).exp(), sum);
            }
            *slot = sum;
        }

        Self {
            lut,
            inv_bin: (LUT_BINS - 1) as f64 / cutoff_sq,
            r_cut_sq: cutoff_sq,
        }
    }

    /// Evaluate the Gaussian density at squared distance `r_sq`, scaled by
    /// occupancy, via table lookup with linear interpolation.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn density_at(&self, r_sq: f64, occupancy: f64) -> f64 {
        if r_sq >= self.r_cut_sq {
            return 0.0;
        }
        let f = r_sq * self.inv_bin;
        let k = (f as usize).min(LUT_BINS - 2);
        let frac = f - k as f64;
        let v = self.lut[k] + frac * (self.lut[k + 1] - self.lut[k]);
        occupancy * v
    }
}

// ── Public helpers ───────────────────────────────────────────────────

/// Compute the isotropic blur to apply to all atoms so that the density
/// is adequately sampled on a grid with spacing `d_min / (2 * rate)`.
///
/// Returns zero when the atoms already have sufficient B-factors.
#[must_use]
#[allow(clippy::suboptimal_flops)]
pub fn compute_blur(d_min: f64, rate: f64, b_min: f64) -> f64 {
    let spacing = d_min / (2.0 * rate);
    let raw = 8.0 * PI * PI / 1.1 * spacing * spacing - b_min;
    raw.max(0.0)
}

/// Compute the real-space cutoff radius (Angstroms) for an atom with
/// effective B-factor `b_eff = B_atom + blur`.
///
/// The formula balances accuracy against speed by keeping ~99.9% of the
/// Gaussian integral within the cutoff sphere.
#[must_use]
pub fn cutoff_radius(b_eff: f64) -> f64 {
    (0.075f64.mul_add(b_eff, 8.5)) / (0.0045f64.mul_add(b_eff, 2.4))
}

// ── Density splatting ────────────────────────────────────────────────

/// Parameters for splatting atomic density onto a grid.
pub struct SplatParams<'a> {
    /// Fractional coordinates of asymmetric-unit atoms.
    pub positions: &'a [[f64; 3]],
    /// IT92 scattering form factor for each atom.
    pub form_factors: &'a [&'a FormFactor],
    /// Isotropic B-factor (A^2) for each atom.
    pub b_factors: &'a [f64],
    /// Site occupancy for each atom.
    pub occupancies: &'a [f64],
    /// The crystallographic unit cell.
    pub unit_cell: &'a UnitCell,
    /// Additional isotropic blur applied to all atoms.
    pub blur: f64,
}

/// Walk every grid voxel covered by one atom copy centered at fractional
/// position `frac_pos`, invoking `visit(idx, kernel_value)` for each.
///
/// This is the single box/cutoff/wrap traversal shared by the density splat
/// (which writes the kernel into a grid) and the gradient gather (which reads a
/// map at each covered voxel). `kernel_value` already carries the atom's
/// occupancy and Debye-Waller factor via [`GaussianPrecalc`].
#[allow(
    clippy::too_many_arguments,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn for_each_covered_voxel(
    frac_pos: [f64; 3],
    ff: &FormFactor,
    b_atom: f64,
    occupancy: f64,
    unit_cell: &UnitCell,
    nu: usize,
    nv: usize,
    nw: usize,
    blur: f64,
    mut visit: impl FnMut(usize, f64),
) {
    let orth_n = build_orth_n(unit_cell, [nu, nv, nw]);
    let dims = [nu as f64, nv as f64, nw as f64];

    let cutoff = cutoff_radius(b_atom + blur);
    let cutoff_sq = cutoff * cutoff;
    let precalc = GaussianPrecalc::new(ff, b_atom, blur, cutoff_sq);

    // Nearest grid point (fractional -> grid index, rounded).
    let u0f = frac_pos[0] * dims[0];
    let v0f = frac_pos[1] * dims[1];
    let w0f = frac_pos[2] * dims[2];

    let u0 = u0f.round() as i64;
    let v0 = v0f.round() as i64;
    let w0 = w0f.round() as i64;

    // Fractional offsets of the atom from the nearest grid point (in grid
    // units) -- needed for sub-grid-point accuracy.
    let frac_offset = [u0f - u0 as f64, v0f - v0 as f64, w0f - w0 as f64];

    // Bounding box in grid units.
    let max_radius_u = estimate_grid_radius(&orth_n, cutoff, 0);
    let max_radius_v = estimate_grid_radius(&orth_n, cutoff, 1);
    let max_radius_w = estimate_grid_radius(&orth_n, cutoff, 2);

    for du in -max_radius_u..=max_radius_u {
        for dv in -max_radius_v..=max_radius_v {
            for dw in -max_radius_w..=max_radius_w {
                // Exact delta in grid units, accounting for sub-grid offset.
                let delta = [
                    f64::from(du) - frac_offset[0],
                    f64::from(dv) - frac_offset[1],
                    f64::from(dw) - frac_offset[2],
                ];

                // Cartesian distance via orth_n.
                let r_sq = cart_dist_sq(&orth_n, &delta);

                if r_sq > cutoff_sq {
                    continue;
                }

                let val = precalc.density_at(r_sq, occupancy);

                let iu = wrap(u0 + i64::from(du), nu);
                let iv = wrap(v0 + i64::from(dv), nv);
                let iw = wrap(w0 + i64::from(dw), nw);

                let idx = (iu * nv + iv) * nw + iw;
                visit(idx, val);
            }
        }
    }
}

/// Splat atomic electron density onto a 3D grid.
///
/// Each atom is represented as a sum of five Gaussians (four IT92 terms plus
/// the constant). For every atom the contribution is added to all grid points
/// within a per-atom cutoff radius, using periodic boundary conditions.
pub fn splat_density(grid: &mut DensityGrid, params: &SplatParams<'_>) {
    let nu = grid.nu;
    let nv = grid.nv;
    let nw = grid.nw;

    let n_atoms = params.positions.len();
    for atom in 0..n_atoms {
        let ff = params.form_factors[atom];
        let b_atom = params.b_factors[atom];
        let occ = params.occupancies[atom];
        let frac_pos = params.positions[atom];

        for_each_covered_voxel(
            frac_pos,
            ff,
            b_atom,
            occ,
            params.unit_cell,
            nu,
            nv,
            nw,
            params.blur,
            |idx, val| {
                #[allow(clippy::cast_possible_truncation)]
                {
                    grid.data[idx] += val as f32;
                }
            },
        );
    }
}

/// Accumulate the real-space gradient gather for one atom copy centered at
/// fractional position `frac_pos`.
///
/// Mirrors one atom's box/cutoff/wrap iteration in [`splat_density`], but
/// instead of writing the Gaussian kernel into a grid it reads `g_map` at each
/// covered voxel and accumulates `g_map[voxel] * kernel(voxel) * occupancy`.
/// The kernel carries the atom's Debye-Waller factor (`b_atom + blur`) and
/// element form factor via [`GaussianPrecalc`], so a per-reflection
/// `exp(+blur * s²/4)` deblur factor on the map coefficients nets against the
/// blur baked in here to leave `exp(-b_atom * s²/4)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn gather_gradient(
    g_map: &[f32],
    frac_pos: [f64; 3],
    ff: &FormFactor,
    b_atom: f64,
    occupancy: f64,
    unit_cell: &UnitCell,
    nu: usize,
    nv: usize,
    nw: usize,
    blur: f64,
) -> f64 {
    let mut sum = 0.0_f64;
    for_each_covered_voxel(
        frac_pos,
        ff,
        b_atom,
        occupancy,
        unit_cell,
        nu,
        nv,
        nw,
        blur,
        |idx, val| {
            sum += f64::from(g_map[idx]) * val;
        },
    );
    sum
}

/// Build the grid-step-to-Cartesian matrix that converts grid-index deltas to
/// Cartesian distances: `orth_n[i][j] = orth[i][j] / dim[j]`.
#[allow(clippy::cast_precision_loss)]
pub(super) fn build_orth_n(cell: &UnitCell, dims: [usize; 3]) -> [[f64; 3]; 3] {
    let dim = [dims[0] as f64, dims[1] as f64, dims[2] as f64];
    let mut orth_n = [[0.0_f64; 3]; 3];
    for (i, orth_row) in orth_n.iter_mut().enumerate() {
        for j in 0..3 {
            orth_row[j] = cell.orth[i][j] / dim[j];
        }
    }
    orth_n
}

/// Compute the squared Cartesian distance for a grid-delta vector `d` via the
/// orthogonalization-per-grid-step matrix `orth_n`.
#[inline]
fn cart_dist_sq(orth_n: &[[f64; 3]; 3], d: &[f64; 3]) -> f64 {
    let cx = orth_n[0][0]
        .mul_add(d[0], orth_n[0][1].mul_add(d[1], orth_n[0][2] * d[2]));
    let cy = orth_n[1][0]
        .mul_add(d[0], orth_n[1][1].mul_add(d[1], orth_n[1][2] * d[2]));
    let cz = orth_n[2][0]
        .mul_add(d[0], orth_n[2][1].mul_add(d[1], orth_n[2][2] * d[2]));
    cx.mul_add(cx, cy.mul_add(cy, cz * cz))
}

/// Estimate the maximum number of grid steps along axis `axis` that could
/// fall within `cutoff` Angstroms of the origin, given the grid-to-Cartesian
/// matrix `orth_n`.
///
/// Uses the inverse of the column length as an upper bound on the grid
/// spacing in that direction.
pub(super) fn estimate_grid_radius(
    orth_n: &[[f64; 3]; 3],
    cutoff: f64,
    axis: usize,
) -> i32 {
    // Length of the column vector corresponding to this axis.
    let col_len = orth_n[0][axis]
        .mul_add(
            orth_n[0][axis],
            orth_n[1][axis]
                .mul_add(orth_n[1][axis], orth_n[2][axis] * orth_n[2][axis]),
        )
        .sqrt();

    if col_len < 1e-15 {
        return 0;
    }

    // Each grid step covers at least `col_len` Angstroms in that direction,
    // so the maximum number of steps is cutoff / col_len, rounded up, plus
    // one for sub-grid offset safety.
    #[allow(clippy::cast_possible_truncation)]
    let r = (cutoff / col_len).ceil() as i32 + 1;
    r
}

/// Map an integer grid coordinate into `[0, n)` with periodic wrapping.
#[inline]
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]
fn wrap(idx: i64, n: usize) -> usize {
    let n_i = n as i64;
    (((idx % n_i) + n_i) % n_i) as usize
}

// ── Symmetrize ───────────────────────────────────────────────────────

/// Average the density grid over all space-group symmetry mates.
///
/// For each grid point, the orbit under all symmetry and centering operations
/// is computed. The density values at all orbit points are summed, and the sum
/// is written back to every point in the orbit. A visited bitmap avoids
/// redundant work.
#[allow(clippy::cast_precision_loss, clippy::excessive_nesting)]
pub fn symmetrize_sum(grid: &mut DensityGrid, group: &GroupOps) {
    let nu = grid.nu;
    let nv = grid.nv;
    let nw = grid.nw;
    let total = nu * nv * nw;
    let den_f = f64::from(DEN);

    // Visited bitmap: one bit per grid point.
    let mut visited = vec![false; total];

    for u in 0..nu {
        for v in 0..nv {
            for w in 0..nw {
                let base_idx = (u * nv + v) * nw + w;
                if visited[base_idx] {
                    continue;
                }

                // Fractional coordinates of this grid point.
                let frac = [
                    u as f64 / nu as f64,
                    v as f64 / nv as f64,
                    w as f64 / nw as f64,
                ];

                // Collect all orbit indices and sum their values.
                let mut orbit = Vec::new();
                let mut sum = 0.0_f64;

                for sym_op in &group.sym_ops {
                    let sym_frac = sym_op.apply_to_frac(frac);

                    for cen in &group.cen_ops {
                        let cf = [
                            sym_frac[0] + f64::from(cen[0]) / den_f,
                            sym_frac[1] + f64::from(cen[1]) / den_f,
                            sym_frac[2] + f64::from(cen[2]) / den_f,
                        ];

                        let gu = frac_to_grid(cf[0], nu);
                        let gv = frac_to_grid(cf[1], nv);
                        let gw = frac_to_grid(cf[2], nw);

                        let idx = (gu * nv + gv) * nw + gw;
                        orbit.push(idx);
                        sum += f64::from(grid.data[idx]);
                    }
                }

                // Write sum back to all orbit points and mark visited.
                #[allow(clippy::cast_possible_truncation)]
                let sum_f32 = sum as f32;
                for &idx in &orbit {
                    grid.data[idx] = sum_f32;
                    visited[idx] = true;
                }
            }
        }
    }
}

/// Convert a fractional coordinate to a grid index in `[0, n)`, with wrapping.
#[inline]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn frac_to_grid(f: f64, n: usize) -> usize {
    let raw = (f * n as f64).round() as i64;
    wrap(raw, n)
}

// ── Deblur ───────────────────────────────────────────────────────────

/// Remove the artificial blur from Fourier coefficients after FFT.
///
/// For each Miller index `(h, k, l)` mapped from the grid indices, multiply
/// the complex Fc by `exp(blur * d*^2 / 4)`.
///
/// `fc` is a flat array of `[re, im]` pairs in row-major `(u, v, w)` order
/// over the full complex spectrum (length `nu * nv * nw`).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_arguments
)]
pub fn deblur_fc(
    fc: &mut [[f32; 2]],
    unit_cell: &UnitCell,
    nu: usize,
    nv: usize,
    nw: usize,
    blur: f64,
) {
    for u in 0..nu {
        let h = if u <= nu / 2 {
            u as i32
        } else {
            u as i32 - nu as i32
        };

        for v in 0..nv {
            let k = if v <= nv / 2 {
                v as i32
            } else {
                v as i32 - nv as i32
            };

            for w in 0..nw {
                let l = if w <= nw / 2 {
                    w as i32
                } else {
                    w as i32 - nw as i32
                };

                let d_star_sq = unit_cell.d_star_sq(h, k, l);
                // Clamp exponent to avoid f32 overflow (f32::MAX ≈ 3.4e38,
                // ln(3.4e38) ≈ 88). Beyond this, the coefficient is
                // effectively zero after blurring and deblur would amplify
                // noise, so capping is physically reasonable.
                let exponent = (blur * d_star_sq / 4.0).min(80.0);
                let scale = exponent.exp() as f32;

                let idx = (u * nv + v) * nw + w;
                fc[idx][0] *= scale;
                fc[idx][1] *= scale;
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xtal::form_factors::FormFactor;
    use crate::xtal::types::UnitCell;

    /// Carbon IT92 form factor for testing.
    fn carbon_ff() -> FormFactor {
        FormFactor {
            a: [2.31, 1.02, 1.5886, 0.865],
            b: [20.8439, 10.2075, 0.5687, 51.6512],
            c: 0.2156,
        }
    }

    /// The precomputed density LUT reproduces a direct 5-Gaussian `exp`
    /// evaluation to within a small relative error across the tabulated range.
    #[test]
    fn lut_density_matches_exact_exp() {
        let ff = carbon_ff();
        let b_atom = 20.0;
        let blur = 5.0;
        let cutoff = cutoff_radius(b_atom + blur);
        let cutoff_sq = cutoff * cutoff;
        let precalc = GaussianPrecalc::new(&ff, b_atom, blur, cutoff_sq);

        let mut max_rel = 0.0_f64;
        for n in 0..=256 {
            let r_sq = cutoff_sq * (f64::from(n) / 256.0) * 0.999;
            let approx = precalc.density_at(r_sq, 1.0);
            let ex = kernel_eval(&ff, b_atom, blur, r_sq);
            let rel = (approx - ex).abs() / ex.abs().max(1e-30);
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_rel < 1e-3,
            "LUT vs exact max relative error {max_rel} exceeds 1e-3"
        );
    }

    #[test]
    fn density_at_origin_positive_and_decays() {
        let cell = UnitCell::new(20.0, 20.0, 20.0, 90.0, 90.0, 90.0);
        let nu = 32;
        let nv = 32;
        let nw = 32;
        let mut grid = DensityGrid {
            data: vec![0.0; nu * nv * nw],
            nu,
            nv,
            nw,
        };

        let ff = carbon_ff();
        let params = SplatParams {
            positions: &[[0.0, 0.0, 0.0]],
            form_factors: &[&ff],
            b_factors: &[30.0],
            occupancies: &[1.0],
            unit_cell: &cell,
            blur: 10.0,
        };

        splat_density(&mut grid, &params);

        // Density at the origin (grid point 0,0,0) should be positive.
        let origin_val = grid.data[0];
        assert!(
            origin_val > 0.0,
            "density at origin should be positive, got {origin_val}"
        );

        // Density should decay with distance from the origin.
        let far_idx = 8 * nv * nw;
        let far_val = grid.data[far_idx];
        assert!(
            far_val < origin_val,
            "density should decay: origin={origin_val}, far={far_val}"
        );
    }

    #[test]
    fn compute_blur_zero_when_b_min_large() {
        let result = compute_blur(2.0, 1.5, 1000.0);
        assert!(
            result.abs() < 1e-12,
            "blur should be 0 with large b_min, got {result}"
        );
    }

    #[test]
    fn compute_blur_positive_for_typical_values() {
        let result = compute_blur(2.0, 1.5, 5.0);
        assert!(
            result > 0.0,
            "blur should be positive for typical parameters, got {result}"
        );
    }

    #[test]
    fn cutoff_radius_increases_with_b() {
        let r1 = cutoff_radius(10.0);
        let r2 = cutoff_radius(50.0);
        let r3 = cutoff_radius(200.0);

        assert!(r2 > r1, "cutoff should increase: r(10)={r1}, r(50)={r2}");
        assert!(r3 > r2, "cutoff should increase: r(50)={r2}, r(200)={r3}");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn integrated_density_approximately_z() {
        // For carbon, Z = 6. The integrated density over the unit cell should
        // be approximately 6 (the number of electrons).
        let a = 15.0;
        let cell = UnitCell::new(a, a, a, 90.0, 90.0, 90.0);
        let nu = 48;
        let nv = 48;
        let nw = 48;
        let mut grid = DensityGrid {
            data: vec![0.0; nu * nv * nw],
            nu,
            nv,
            nw,
        };

        let ff = carbon_ff();
        let b_factor = 20.0;
        let blur = compute_blur(2.0, 1.5, b_factor);

        let params = SplatParams {
            positions: &[[0.5, 0.5, 0.5]],
            form_factors: &[&ff],
            b_factors: &[b_factor],
            occupancies: &[1.0],
            unit_cell: &cell,
            blur,
        };

        splat_density(&mut grid, &params);

        // Volume element per grid point.
        let voxel_volume = cell.volume / (nu * nv * nw) as f64;

        // Integrate: sum of all density values * voxel volume.
        let total: f64 =
            grid.data.iter().map(|&v| f64::from(v)).sum::<f64>() * voxel_volume;

        let z_carbon = 6.0;
        let rel_error = (total - z_carbon).abs() / z_carbon;
        assert!(
            rel_error < 0.10,
            "integrated density {total:.3} should be within 10% of Z=6, \
             rel_error={rel_error:.4}"
        );
    }

    /// deblur_fc must not produce Inf/NaN for high blur × high resolution.
    #[test]
    fn deblur_fc_no_overflow() {
        let cell = UnitCell::new(10.0, 10.0, 10.0, 90.0, 90.0, 90.0);
        let nu = 8;
        let nv = 8;
        let nw = 8;
        let n = nu * nv * nw;
        let mut fc: Vec<[f32; 2]> = vec![[1.0, 0.5]; n];

        // Large blur * high-res reflections would overflow without clamping.
        deblur_fc(&mut fc, &cell, nu, nv, nw, 200.0);

        for (i, c) in fc.iter().enumerate() {
            assert!(
                c[0].is_finite(),
                "deblur_fc[{i}][0] is not finite: {}",
                c[0]
            );
            assert!(
                c[1].is_finite(),
                "deblur_fc[{i}][1] is not finite: {}",
                c[1]
            );
        }
    }
}
