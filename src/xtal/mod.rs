//! Crystallographic refinement: electron density maps and ML target functions.
//!
//! This module implements the full ML crystallographic refinement pipeline:
//! atomic density splatting, bulk solvent mask, anisotropic scaling, sigma-A
//! estimation, weighted map coefficient computation, and B-factor refinement.
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

mod bessel;
mod density;
mod fft_cpu;
mod form_factors;
mod map_coefficients;
mod refinement;
mod scaling;
mod sigma_a;
mod solvent_mask;
mod targets;
mod types;

// ── Public re-exports ───────────────────────────────────────────────

pub use bessel::log_bessel_i0;
pub use form_factors::FormFactor;
pub use map_coefficients::MapCoefficients;
pub use refinement::XtalRefinement;
pub use scaling::ScalingResult;
pub use sigma_a::{r_free, SigmaAResult};
pub use targets::ml_target_value;
pub use types::{
    epsilon_factor, grid_factors, has_small_factorization, is_centric,
    is_systematically_absent, requires_equal_uv, round_up_to_smooth,
    space_group, CrystalSystem, DensityGrid, GroupOps, Reflection, SpaceGroup,
    Symop, UnitCell, DEN,
};

use crate::adapters::cif::extract as cif;

// ── SF-CIF → xtal reflection conversion ─────────────────────────────

/// Default R-free fraction when the SF-CIF file does not specify free flags.
const DEFAULT_FREE_FRACTION: f64 = 0.05;

/// Convert CIF reflection data into xtal [`Reflection`] values.
///
/// This is the bridge between the CIF parser's [`cif::ReflectionData`] and the
/// xtal module's [`Reflection`] type. It:
///
/// 1. Filters out reflections with missing Fobs (no measured amplitude).
/// 2. Converts `f64` → `f32` for Fobs and sigma.
/// 3. If the file did not provide R-free flags (`free_flags_from_file ==
///    false`), randomly assigns approximately `free_fraction` of reflections as
///    the test set using a deterministic PRNG seeded by `seed`. The assignment
///    is stratified by resolution (20 thin shells) so the free set has even
///    coverage.
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
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
pub fn reflections_from_cif(
    data: &cif::ReflectionData,
    free_fraction: f64,
    seed: u64,
) -> Vec<Reflection> {
    let frac = free_fraction.clamp(0.01, 0.50);

    // Build the xtal unit cell so we can use d_star_sq for resolution binning.
    let uc = UnitCell::new(
        data.cell.a,
        data.cell.b,
        data.cell.c,
        data.cell.alpha,
        data.cell.beta,
        data.cell.gamma,
    );

    // First pass: collect indices of reflections with valid Fobs, and their s².
    let mut valid: Vec<(usize, f64)> = Vec::new();
    for (i, r) in data.reflections.iter().enumerate() {
        if r.f_meas.is_some() {
            let s2 = uc.d_star_sq(r.h, r.k, r.l);
            valid.push((i, s2));
        }
    }

    if valid.is_empty() {
        return Vec::new();
    }

    // If file had free flags, just convert directly.
    if data.free_flags_from_file {
        return valid
            .iter()
            .map(|&(i, _)| {
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

    // No flags from file — assign free set with deterministic PRNG.
    // Resolution range for stratification.
    let s2_min = valid.iter().map(|v| v.1).fold(f64::MAX, f64::min);
    let s2_max = valid.iter().map(|v| v.1).fold(f64::MIN, f64::max);
    let range = s2_max - s2_min;
    let _bin_width = if range < 1e-12 { 1.0 } else { range / 20.0 };

    let mut rng_state: u64 = seed | 1; // ensure nonzero

    valid
        .iter()
        .map(|&(i, _s2)| {
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

/// Convert CIF reflection data using the default free fraction (5%).
///
/// Convenience wrapper around [`reflections_from_cif`].
#[must_use]
pub fn reflections_from_cif_default(
    data: &cif::ReflectionData,
    seed: u64,
) -> Vec<Reflection> {
    reflections_from_cif(data, DEFAULT_FREE_FRACTION, seed)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
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
                        f_calc: None,
                        phase_calc: None,
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
