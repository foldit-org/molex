//! Crystallographic refinement: electron density maps and maximum-likelihood
//! target functions.
//!
//! This module implements the full maximum-likelihood crystallographic
//! refinement pipeline: atomic density splatting, bulk solvent mask,
//! anisotropic scaling, sigma-A estimation, weighted map coefficient
//! computation, and B-factor refinement.
//!
//! Enable with the `xtal` Cargo feature.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use molex::xtal::{XtalRefinement, UnitCell, SpaceGroup, Reflection};
//!
//! let refinement = XtalRefinement::new(unit_cell, space_group, reflections, grid_dims);
//! let density = refinement.compute_map(&atoms, &elements, &b_factors, &occupancies);
//! ```

// The refinement kernels accumulate products in explicit multiply-add loops;
// keeping them as written preserves the numerical intent over clippy's
// `mul_add` rewrite, whose fused rounding would perturb R-factor outputs.
#![allow(clippy::suboptimal_flops)]

mod bessel;
#[cfg(feature = "minimization")]
mod bfactor_refine;
mod density;
mod experimental;
mod fft_cpu;
mod form_factors;
#[cfg(all(feature = "xtal", feature = "gpu"))]
mod gpu;
mod map_coefficients;
mod refinement;
mod scaling;
mod sigma_a;
mod solvent_mask;
mod targets;
mod types;

// ── Public re-exports ───────────────────────────────────────────────

pub use bessel::log_bessel_i0;
pub use density::{
    compute_blur, cutoff_radius, kernel_eval, splat_density, SplatParams,
};
pub use experimental::{ExperimentalData, RefinementInputs};
pub use fft_cpu::FftPrecision;
pub use form_factors::{form_factor, FormFactor};
pub use map_coefficients::MapCoefficients;
use ndarray::Array3;
#[cfg(all(feature = "xtal", any(test, feature = "testutil")))]
pub(crate) use refinement::ForwardSymmetry;
pub use refinement::{StencilBackend, XtalRefinement};
pub use scaling::ScalingResult;
pub use sigma_a::SigmaAResult;
pub use targets::maximum_likelihood_target;
pub use types::{
    derive_grid, epsilon_factor, grid_factors, has_small_factorization,
    is_centric, is_space_group_supported, is_systematically_absent,
    requires_equal_uv, round_up_to_pow2, round_up_to_smooth, space_group,
    space_group_number_from_name, supported_space_groups, CrystalSystem,
    DensityGrid, GroupOps, Reflection, SpaceGroup, Symop, UnitCell, DEN,
};

use crate::adapters::cif::extract as cif;
use crate::entity::surface::density::{Density, VoxelGrid};

// ── SF-CIF → xtal reflection conversion ─────────────────────────────

/// FNV-1a hash of a key (e.g. a PDB code), used to seed the free-flag
/// partition deterministically per dataset.
#[must_use]
pub fn deterministic_free_flag_seed(key: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for b in key.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Convert CIF reflection data into xtal [`Reflection`] values.
///
/// This is the bridge between the CIF parser's [`cif::ReflectionData`] and the
/// xtal module's [`Reflection`] type. It:
///
/// 1. Filters out reflections with missing Fobs (no measured amplitude).
/// 2. Converts `f64` → `f32` for Fobs and sigma.
/// 3. If the file did not provide R-free flags (`free_flags_from_file ==
///    false`), randomly assigns approximately `free_fraction` of reflections as
///    the test set using a deterministic PRNG seeded by `seed`. Each reflection
///    is assigned independently by comparing a per-reflection xorshift draw
///    against a single global threshold; the partition is not stratified by
///    resolution.
///
/// The `seed` should be fixed per dataset (e.g. hash of the PDB code) so the
/// free set is reproducible across refinement runs.
///
/// Returns an empty `Vec` if no reflections have valid Fobs.
///
/// # Known limitations
///
/// **Anomalous (Bijvoet) data**: SF-CIF files that contain separate F(+) and
/// F(−) columns (`_refln.pdbx_F_plus` / `_refln.pdbx_F_minus`) instead of
/// merged amplitudes are not currently handled. Only merged `_refln.F_meas_au`
/// or `_refln.F_squared_meas` columns are read. If the input contains
/// unmerged anomalous data, reflections will appear to have missing Fobs and
/// will be silently dropped.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn reflections_from_cif(
    data: &cif::ReflectionData,
    free_fraction: f64,
    seed: u64,
) -> Vec<Reflection> {
    let frac = free_fraction.clamp(0.01, 0.50);

    // First pass: collect indices of reflections with valid Fobs.
    let mut valid: Vec<usize> = Vec::new();
    for (i, r) in data.reflections.iter().enumerate() {
        if r.f_meas.is_some() {
            valid.push(i);
        }
    }

    if valid.is_empty() {
        return Vec::new();
    }

    // If file had free flags, just convert directly.
    if data.free_flags_from_file {
        return valid
            .iter()
            .map(|&i| {
                let r = &data.reflections[i];
                Reflection {
                    h: r.h,
                    k: r.k,
                    l: r.l,
                    f_obs: r.f_meas.unwrap_or(0.0) as f32,
                    sigma_f: r.sigma_f_meas.unwrap_or(1.0) as f32,
                    free_flag: r.free_flag,
                }
            })
            .collect();
    }

    // No flags from file: assign free set with deterministic PRNG.
    let mut rng_state: u64 = seed | 1; // ensure nonzero

    valid
        .iter()
        .map(|&i| {
            let r = &data.reflections[i];

            // Advance xorshift64.
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let rand_val =
                (rng_state & 0x1F_FFFF_FFFF_FFFF) as f64 / (1_u64 << 53) as f64;

            Reflection {
                h: r.h,
                k: r.k,
                l: r.l,
                f_obs: r.f_meas.unwrap_or(0.0) as f32,
                sigma_f: r.sigma_f_meas.unwrap_or(1.0) as f32,
                free_flag: rand_val < frac,
            }
        })
        .collect()
}

// ── DensityGrid → Density promotion ─────────────────────────────────

/// Promote a bare [`DensityGrid`] (values plus grid dimensions) into a full
/// [`Density`] by attaching the metadata a whole-unit-cell map implies.
///
/// A crystallographic map samples the entire fractional cell, so the grid
/// dimensions equal the sampling intervals, the start indices are zero, and
/// the origin sits at the cell corner. Min/max/mean/RMS are computed from the
/// data; the space group is passed through.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "cell params narrow f64->f32; sg number is non-negative"
)]
pub fn density_from_grid(
    grid: &DensityGrid,
    unit_cell: &UnitCell,
    space_group: i32,
) -> Density {
    let dims = (grid.nu, grid.nv, grid.nw);
    // Normalize length to nu*nv*nw so the reshape is infallible; the
    // DensityGrid contract already guarantees this, and pad/truncate degrades
    // gracefully rather than panicking if a caller violates it.
    let mut values = grid.data.clone();
    values.resize(grid.nu * grid.nv * grid.nw, 0.0);
    let data = Array3::from_shape_vec(dims, values)
        .unwrap_or_else(|_| Array3::zeros(dims));

    let (dmin, dmax, dmean, rms) = grid_statistics(&grid.data);

    Density {
        grid: VoxelGrid {
            nx: grid.nu,
            ny: grid.nv,
            nz: grid.nw,
            nxstart: 0,
            nystart: 0,
            nzstart: 0,
            mx: grid.nu,
            my: grid.nv,
            mz: grid.nw,
            cell_dims: [
                unit_cell.a as f32,
                unit_cell.b as f32,
                unit_cell.c as f32,
            ],
            cell_angles: [
                unit_cell.alpha as f32,
                unit_cell.beta as f32,
                unit_cell.gamma as f32,
            ],
            origin: [0.0, 0.0, 0.0],
            data,
        },
        dmin,
        dmax,
        dmean,
        rms,
        space_group: space_group as u32,
    }
}

/// Compute `(dmin, dmax, dmean, rms)` over a density grid.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "voxel count to f64 for averaging; mean/rms narrow back to f32"
)]
fn grid_statistics(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut dmin = f32::INFINITY;
    let mut dmax = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in data {
        dmin = dmin.min(v);
        dmax = dmax.max(v);
        sum += f64::from(v);
    }
    let n = data.len() as f64;
    let mean = sum / n;
    let var = data
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (dmin, dmax, mean as f32, var.sqrt() as f32)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── reflections_from_cif tests ──────────────────────────────────

    fn make_cif_reflection_data(with_flags: bool) -> cif::ReflectionData {
        let mut reflections = Vec::new();
        for h in 1..=4_i32 {
            for k in 0..=2_i32 {
                for l in 0..=2_i32 {
                    reflections.push(cif::Reflection {
                        h,
                        k,
                        l,
                        f_meas: Some(100.0 / f64::from(h)),
                        sigma_f_meas: Some(1.0),
                        free_flag: with_flags && (h + k + l) % 7 == 0,
                    });
                }
            }
        }
        cif::ReflectionData {
            cell: cif::UnitCell {
                a: 50.0,
                b: 60.0,
                c: 70.0,
                alpha: 90.0,
                beta: 90.0,
                gamma: 90.0,
            },
            spacegroup: Some("P 21 21 21".to_owned()),
            reflections,
            obs_data_type: cif::ObsDataType::Amplitude,
            free_flags_from_file: with_flags,
        }
    }

    #[test]
    fn density_from_grid_populates_metadata() {
        let grid = DensityGrid {
            data: (0..24u16).map(f32::from).collect(),
            nu: 2,
            nv: 3,
            nw: 4,
        };
        let cell = UnitCell::new(30.0, 40.0, 50.0, 90.0, 90.0, 90.0);
        let density = density_from_grid(&grid, &cell, 19);

        assert_eq!(density.nx, 2);
        assert_eq!(density.ny, 3);
        assert_eq!(density.nz, 4);
        assert_eq!(density.nxstart, 0);
        assert_eq!(density.nystart, 0);
        assert_eq!(density.nzstart, 0);
        assert_eq!(density.mx, 2);
        assert_eq!(density.my, 3);
        assert_eq!(density.mz, 4);
        assert!((density.cell_dims[0] - 30.0).abs() < 1e-4);
        assert!((density.cell_dims[1] - 40.0).abs() < 1e-4);
        assert!((density.cell_dims[2] - 50.0).abs() < 1e-4);
        assert!((density.cell_angles[0] - 90.0).abs() < 1e-4);
        assert!((density.cell_angles[1] - 90.0).abs() < 1e-4);
        assert!((density.cell_angles[2] - 90.0).abs() < 1e-4);
        assert!(density.origin.iter().all(|&o| o.abs() < 1e-9));
        assert_eq!(density.space_group, 19);

        // Stats over the values 0..=23.
        assert!((density.dmin - 0.0).abs() < 1e-6);
        assert!((density.dmax - 23.0).abs() < 1e-6);
        assert!((density.dmean - 11.5).abs() < 1e-4);
        assert!(density.rms > 0.0);

        // Row-major (u, v, w) maps directly onto data[[u, v, w]].
        assert!((density.data[[0, 0, 0]] - 0.0).abs() < 1e-6);
        assert!((density.data[[0, 0, 1]] - 1.0).abs() < 1e-6);
        assert!((density.data[[1, 2, 3]] - 23.0).abs() < 1e-6);
    }

    #[test]
    fn reflections_from_cif_preserves_file_flags() {
        let data = make_cif_reflection_data(true);
        let refls = reflections_from_cif(&data, 0.05, 42);

        assert_eq!(refls.len(), data.reflections.len());

        // Flags should match the source.
        for (xtal_r, cif_r) in refls.iter().zip(data.reflections.iter()) {
            assert_eq!(xtal_r.free_flag, cif_r.free_flag);
        }
    }

    #[test]
    fn reflections_from_cif_assigns_flags_when_missing() {
        let data = make_cif_reflection_data(false);
        let refls = reflections_from_cif(&data, 0.05, 42);

        assert_eq!(refls.len(), data.reflections.len());

        // Some should be free, some working.
        let free_count = refls.iter().filter(|r| r.free_flag).count();
        assert!(free_count > 0, "expected some free reflections");
        assert!(
            free_count < refls.len(),
            "expected some working reflections"
        );

        // Deterministic: same seed → same result.
        let refls2 = reflections_from_cif(&data, 0.05, 42);
        let flags1: Vec<bool> = refls.iter().map(|r| r.free_flag).collect();
        let flags2: Vec<bool> = refls2.iter().map(|r| r.free_flag).collect();
        assert_eq!(flags1, flags2);
    }

    #[test]
    fn reflections_from_cif_filters_missing_fobs() {
        let mut data = make_cif_reflection_data(true);
        // Set some f_meas to None.
        data.reflections[0].f_meas = None;
        data.reflections[3].f_meas = None;

        let refls = reflections_from_cif(&data, 0.05, 42);

        assert_eq!(
            refls.len(),
            data.reflections.len() - 2,
            "should filter out 2 reflections with missing Fobs"
        );
    }

    #[test]
    fn reflections_from_cif_converts_types() {
        let data = make_cif_reflection_data(true);
        let refls = reflections_from_cif(&data, 0.05, 42);

        let first = &refls[0];
        let cif_first = &data.reflections[0];
        assert_eq!(first.h, cif_first.h);
        assert_eq!(first.k, cif_first.k);
        assert_eq!(first.l, cif_first.l);
        #[allow(clippy::cast_possible_truncation)]
        let expected_fobs = cif_first.f_meas.unwrap_or(0.0) as f32;
        assert!((first.f_obs - expected_fobs).abs() < 1e-6);
    }
}
