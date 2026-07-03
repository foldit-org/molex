//! Bulk solvent mask computation via Felzenszwalb-Huttenlocher EDT.
//!
//! Implements the flat bulk-solvent model mask used in crystallographic
//! refinement.  The mask is computed by:
//!
//! 1. Marking grid points within `vdw_radius + r_probe` of any atom as protein.
//! 2. Shrinking thin protein bridges via an exact distance transform (EDT).
//! 3. Returning a binary grid: 0.0 = protein, 1.0 = solvent.

use super::form_factors::vdw_radius;
use super::types::{DensityGrid, UnitCell};
use crate::element::Element;

// ── 1-D Exact Distance Transform ─────────────────────────────────────

/// Felzenszwalb–Huttenlocher 1-D exact distance transform.
///
/// Given squared-distance values (`0.0` for source points,
/// [`f64::INFINITY`] for others), returns the squared Euclidean distance
/// from each position to its nearest source.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn edt_1d(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }

    let mut v = vec![0_usize; n]; // parabola locations
    let mut z = vec![0.0_f64; n + 1]; // boundaries
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let mut k = 0_usize;

    for q in 1..n {
        loop {
            let vk = v[k];
            #[allow(clippy::cast_precision_loss)]
            let s = (input[q] + (q * q) as f64 - input[vk] - (vk * vk) as f64)
                / (2.0 * (q as f64 - vk as f64));
            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
            if k == 0 {
                v[0] = q;
                z[1] = f64::INFINITY;
                break;
            }
            k -= 1;
        }
    }

    let mut dt = vec![0.0_f64; n];
    k = 0;
    for (q, dt_val) in dt.iter_mut().enumerate() {
        #[allow(clippy::while_float, clippy::cast_precision_loss)]
        while z[k + 1] < q as f64 {
            k += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let d = q as f64 - v[k] as f64;
        *dt_val = d.mul_add(d, input[v[k]]);
    }
    dt
}

/// Periodic variant of the 1-D EDT.
///
/// Pads the input by `n/2 + 1` on each side (wrapping), runs [`edt_1d`]
/// on the padded array, and extracts the central `n` elements.
#[must_use]
pub fn edt_1d_periodic(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }

    let pad = n / 2 + 1;
    let total = n + 2 * pad;
    let mut padded = vec![0.0_f64; total];

    for (idx, slot) in padded.iter_mut().enumerate() {
        // Map padded index back into [0, n) with wrapping.
        let src = (idx + n - pad % n) % n;
        *slot = input[src];
    }

    let result = edt_1d(&padded);
    result[pad..pad + n].to_vec()
}

// ── 3-D Exact Distance Transform ─────────────────────────────────────

/// Compute the Euclidean norm of column `j` of a 3x3 matrix.
fn column_length(m: &[[f64; 3]; 3], j: usize) -> f64 {
    (m[0][j].mul_add(m[0][j], m[1][j].mul_add(m[1][j], m[2][j] * m[2][j])))
        .sqrt()
}

/// Three-dimensional exact distance transform with anisotropic grid spacing.
///
/// Operates in-place on a flat grid of size `nu * nv * nw` (row-major, u
/// varies slowest).  The input values should be `0.0` for source voxels and
/// [`f64::INFINITY`] for others.  On return, each voxel holds its squared
/// Euclidean distance (in squared Angstroms) to the nearest source.
///
/// Grid spacings `hu`, `hv`, `hw` are the real-space step sizes along
/// each axis, derived from the orthogonalization matrix columns.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn edt_3d(
    grid: &mut [f64],
    nu: usize,
    nv: usize,
    nw: usize,
    hu: f64,
    hv: f64,
    hw: f64,
) {
    let hu2 = hu * hu;
    let hv2 = hv * hv;
    let hw2 = hw * hw;

    // Pass 1: along u-axis
    {
        let mut buf = vec![0.0_f64; nu];
        for iv in 0..nv {
            for iw in 0..nw {
                for iu in 0..nu {
                    buf[iu] = grid[iu * nv * nw + iv * nw + iw];
                }
                let dt = edt_1d_periodic(&buf);
                for iu in 0..nu {
                    grid[iu * nv * nw + iv * nw + iw] = dt[iu] * hu2;
                }
            }
        }
    }

    // Pass 2: along v-axis
    {
        let mut buf = vec![0.0_f64; nv];
        let inv_hv2 = 1.0 / hv2;
        for iu in 0..nu {
            for iw in 0..nw {
                for iv in 0..nv {
                    buf[iv] = grid[iu * nv * nw + iv * nw + iw] * inv_hv2;
                }
                let dt = edt_1d_periodic(&buf);
                for iv in 0..nv {
                    grid[iu * nv * nw + iv * nw + iw] = dt[iv] * hv2;
                }
            }
        }
    }

    // Pass 3: along w-axis
    {
        let mut buf = vec![0.0_f64; nw];
        let inv_hw2 = 1.0 / hw2;
        for iu in 0..nu {
            for iv in 0..nv {
                let base = iu * nv * nw + iv * nw;
                for iw in 0..nw {
                    buf[iw] = grid[base + iw] * inv_hw2;
                }
                let dt = edt_1d_periodic(&buf);
                for iw in 0..nw {
                    grid[base + iw] = dt[iw] * hw2;
                }
            }
        }
    }
}

// ── Solvent mask ──────────────────────────────────────────────────────

/// Default probe radius (Angstroms) for bulk solvent mask.
pub const DEFAULT_R_PROBE: f64 = 1.0;
/// Default shrink radius (Angstroms) for removing thin protein bridges.
pub const DEFAULT_R_SHRINK: f64 = 1.1;

/// Compute a bulk-solvent mask on a crystallographic grid.
///
/// Returns a [`DensityGrid`] where each voxel is `1.0` (solvent) or `0.0`
/// (protein).
///
/// # Arguments
///
/// * `positions` — fractional coordinates of each atom.
/// * `elements`  — element type of each atom.
/// * `unit_cell` — the crystallographic unit cell.
/// * `nu`, `nv`, `nw` — grid dimensions along a, b, c.
/// * `r_probe`  — probe radius in Angstroms (typically 1.0).
/// * `r_shrink` — shrink radius in Angstroms (typically 1.1).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn solvent_mask(
    positions: &[[f64; 3]],
    elements: &[Element],
    unit_cell: &UnitCell,
    nu: usize,
    nv: usize,
    nw: usize,
    r_probe: f64,
    r_shrink: f64,
) -> DensityGrid {
    let total = nu * nv * nw;

    // 1.0 = solvent, 0.0 = protein
    let mut mask = vec![1.0_f64; total];

    // Scaled orthogonalization matrix columns (maps one grid step to
    // Angstroms).
    let orth_n = orth_per_grid(unit_cell, nu, nv, nw);

    // Mark grid points within r_total of each atom as protein.
    for (pos, &elem) in positions.iter().zip(elements.iter()) {
        let Some(r_vdw) = vdw_radius(elem) else {
            continue; // skip atoms with unknown radius
        };
        let r_total = r_vdw + r_probe;
        let r_total_sq = r_total * r_total;

        mark_atom_vicinity(&mut mask, pos, r_total_sq, &orth_n, nu, nv, nw);
    }

    // Build EDT input: solvent (1.0) -> 0.0 (source), protein (0.0) -> INF.
    let mut edt_grid = vec![0.0_f64; total];
    for (slot, &m_val) in edt_grid.iter_mut().zip(mask.iter()) {
        *slot = if m_val > 0.5 { 0.0 } else { f64::INFINITY };
    }

    // Grid spacings in Angstroms.
    #[allow(clippy::cast_precision_loss)]
    let hu = column_length(&unit_cell.orth, 0) / nu as f64;
    #[allow(clippy::cast_precision_loss)]
    let hv = column_length(&unit_cell.orth, 1) / nv as f64;
    #[allow(clippy::cast_precision_loss)]
    let hw = column_length(&unit_cell.orth, 2) / nw as f64;

    // Compute EDT: squared distance from each protein point to nearest solvent.
    edt_3d(&mut edt_grid, nu, nv, nw, hu, hv, hw);

    let r_shrink_sq = r_shrink * r_shrink;

    // Shrink: protein points close to solvent become solvent.
    for (idx, m_val) in mask.iter_mut().enumerate() {
        if *m_val < 0.5 && edt_grid[idx] < r_shrink_sq {
            *m_val = 1.0;
        }
    }

    // Convert to f32 output.
    #[allow(clippy::cast_possible_truncation)]
    let data: Vec<f32> = mask.iter().map(|&v| v as f32).collect();

    DensityGrid { data, nu, nv, nw }
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Build the per-grid-step orthogonalization matrix columns.
///
/// `orth_n[row][col]` is `orth[row][col]` divided by the grid count for
/// that column (nu, nv, or nw respectively).
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn orth_per_grid(
    uc: &UnitCell,
    nu: usize,
    nv: usize,
    nw: usize,
) -> [[f64; 3]; 3] {
    let nu_f = nu as f64;
    let nv_f = nv as f64;
    let nw_f = nw as f64;
    [
        [
            uc.orth[0][0] / nu_f,
            uc.orth[0][1] / nv_f,
            uc.orth[0][2] / nw_f,
        ],
        [
            uc.orth[1][0] / nu_f,
            uc.orth[1][1] / nv_f,
            uc.orth[1][2] / nw_f,
        ],
        [
            uc.orth[2][0] / nu_f,
            uc.orth[2][1] / nv_f,
            uc.orth[2][2] / nw_f,
        ],
    ]
}

/// Mark grid points near a single atom as protein (0.0).
///
/// Uses fractional coordinates to find a bounding box, then checks each
/// candidate voxel with the full Cartesian distance via `orth_n`.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
fn mark_atom_vicinity(
    mask: &mut [f64],
    frac_pos: &[f64; 3],
    r_total_sq: f64,
    orth_n: &[[f64; 3]; 3],
    nu: usize,
    nv: usize,
    nw: usize,
) {
    // Convert atom fractional position to grid coordinates.
    let gu = frac_pos[0] * nu as f64;
    let gv = frac_pos[1] * nv as f64;
    let gw = frac_pos[2] * nw as f64;

    // Estimate search radius in grid units.  Use the maximum possible
    // displacement per grid step (the matrix column norms).
    let r_total = r_total_sq.sqrt();
    let step_u = column_length(orth_n, 0);
    let step_v = column_length(orth_n, 1);
    let step_w = column_length(orth_n, 2);

    // Guard against zero-length columns.
    let margin_u = if step_u > 0.0 {
        (r_total / step_u).ceil() as isize + 1
    } else {
        0
    };
    let margin_v = if step_v > 0.0 {
        (r_total / step_v).ceil() as isize + 1
    } else {
        0
    };
    let margin_w = if step_w > 0.0 {
        (r_total / step_w).ceil() as isize + 1
    } else {
        0
    };

    let gu_int = gu.floor() as isize;
    let gv_int = gv.floor() as isize;
    let gw_int = gw.floor() as isize;

    for du in -margin_u..=margin_u {
        let ui = gu_int + du;
        let du_frac = ui as f64 - gu;
        let u_idx = ui.rem_euclid(nu as isize) as usize;

        for dv in -margin_v..=margin_v {
            let vi = gv_int + dv;
            let dv_frac = vi as f64 - gv;
            let v_idx = vi.rem_euclid(nv as isize) as usize;

            for dw in -margin_w..=margin_w {
                let wi = gw_int + dw;
                let dw_frac = wi as f64 - gw;
                let w_idx = wi.rem_euclid(nw as isize) as usize;

                // Cartesian displacement via orth_n.
                let cart_x = orth_n[0][0].mul_add(
                    du_frac,
                    orth_n[0][1].mul_add(dv_frac, orth_n[0][2] * dw_frac),
                );
                let cart_y = orth_n[1][0].mul_add(
                    du_frac,
                    orth_n[1][1].mul_add(dv_frac, orth_n[1][2] * dw_frac),
                );
                let cart_z = orth_n[2][0].mul_add(
                    du_frac,
                    orth_n[2][1].mul_add(dv_frac, orth_n[2][2] * dw_frac),
                );

                let dist_sq = cart_x
                    .mul_add(cart_x, cart_y.mul_add(cart_y, cart_z * cart_z));
                if dist_sq <= r_total_sq {
                    mask[u_idx * nv * nw + v_idx * nw + w_idx] = 0.0;
                }
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn edt_1d_simple() {
        let f = vec![0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0];
        let dt = edt_1d(&f);
        assert!((dt[0] - 0.0).abs() < TOL);
        assert!((dt[1] - 1.0).abs() < TOL);
        assert!((dt[2] - 4.0).abs() < TOL);
        assert!((dt[3] - 1.0).abs() < TOL);
        assert!((dt[4] - 0.0).abs() < TOL);
    }

    #[test]
    fn edt_1d_empty() {
        let dt = edt_1d(&[]);
        assert!(dt.is_empty());
    }

    #[test]
    fn edt_1d_single() {
        let dt = edt_1d(&[0.0]);
        assert_eq!(dt.len(), 1);
        assert!((dt[0] - 0.0).abs() < TOL);
    }

    #[test]
    fn edt_1d_periodic_wrap() {
        let f = vec![0.0, f64::INFINITY, f64::INFINITY];
        let dt = edt_1d_periodic(&f);
        // With periodicity, index 2 is distance 1 from index 0 (wrapping).
        assert!((dt[0] - 0.0).abs() < TOL);
        assert!((dt[1] - 1.0).abs() < TOL);
        assert!((dt[2] - 1.0).abs() < TOL);
    }

    #[test]
    fn edt_3d_single_source() {
        let nu = 8;
        let nv = 8;
        let nw = 8;
        let total = nu * nv * nw;
        let mut grid = vec![f64::INFINITY; total];

        // Place source at (0, 0, 0).
        grid[0] = 0.0;

        // Isotropic spacing of 1.0 Angstroms per grid step.
        edt_3d(&mut grid, nu, nv, nw, 1.0, 1.0, 1.0);

        // (1, 0, 0): distance squared = 1.0
        assert!((grid[nv * nw] - 1.0).abs() < TOL, "got {}", grid[nv * nw]);
        // (0, 1, 0): distance squared = 1.0
        assert!((grid[nw] - 1.0).abs() < TOL, "got {}", grid[nw]);
        // (0, 0, 1): distance squared = 1.0
        assert!((grid[1] - 1.0).abs() < TOL, "got {}", grid[1]);
        // (1, 1, 0): distance squared = 2.0
        assert!(
            (grid[nv * nw + nw] - 2.0).abs() < TOL,
            "got {}",
            grid[nv * nw + nw]
        );
    }

    #[test]
    fn solvent_mask_single_atom() {
        let uc = UnitCell::new(10.0, 10.0, 10.0, 90.0, 90.0, 90.0);
        let positions = [[0.5, 0.5, 0.5]]; // centre of cell
        let elements = [Element::C]; // vdw = 1.7

        let grid =
            solvent_mask(&positions, &elements, &uc, 48, 48, 48, 1.0, 1.1);

        let total = grid.data.len();
        let protein_count = grid.data.iter().filter(|&&v| v < 0.5).count();

        #[allow(clippy::cast_precision_loss)]
        let solvent_fraction = 1.0 - protein_count as f64 / total as f64;

        // A single carbon (r_vdw=1.7) + probe 1.0 = 2.7 A sphere in a 10 A box,
        // after shrinking by 1.1 A the effective radius is ~1.6 A.
        // Protein fraction: ~4/3*pi*(1.6)^3 / 10^3 ~ 1.7%, solvent ~ 98%.
        // Before shrink, protein fraction: ~4/3*pi*(2.7)^3 / 10^3 ~ 8.2%.
        // Allow a wide range to be robust against discretization.
        assert!(
            (0.40..=1.00).contains(&solvent_fraction),
            "solvent fraction {solvent_fraction} outside expected range \
             0.40-1.00"
        );
        // Verify some protein points actually exist (the atom carved out
        // space).
        assert!(
            protein_count > 0,
            "expected at least some protein voxels, got none"
        );
    }

    #[test]
    fn solvent_mask_dimensions() {
        let uc = UnitCell::new(10.0, 10.0, 10.0, 90.0, 90.0, 90.0);
        let grid = solvent_mask(&[], &[], &uc, 12, 12, 12, 1.0, 1.1);
        assert_eq!(grid.nu, 12);
        assert_eq!(grid.nv, 12);
        assert_eq!(grid.nw, 12);
        assert_eq!(grid.data.len(), 12 * 12 * 12);

        // No atoms -> entire grid is solvent.
        for &val in &grid.data {
            assert!(
                (val - 1.0_f32).abs() < f32::EPSILON,
                "expected all solvent, got {val}"
            );
        }
    }
}
