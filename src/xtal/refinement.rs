//! `XtalRefinement` — main orchestrator for crystallographic ML refinement.

use super::form_factors::{self, FormFactor};
use super::types::{
    CrystalSystem, DensityGrid, GroupOps, Reflection, SpaceGroup,
};
use super::{
    density, fft_cpu, map_coefficients, scaling, sigma_a, solvent_mask,
    targets, ScalingResult, SigmaAResult,
};
use crate::element::Element;

/// Wrap a signed Miller index into the range `[0, size)` using modular
/// arithmetic.  Safe for any index magnitude (unlike the bare
/// `(idx + size) as usize` pattern which overflows when `|idx| > size`).
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn wrap_miller_index(idx: i32, size: usize) -> usize {
    let s = size as i32;
    let wrapped = ((idx % s) + s) % s;
    wrapped as usize
}

/// Main orchestrator for crystallographic ML refinement.
///
/// Holds unit cell, space group, reflection data, and grid dimensions. Provides
/// methods to run the full pipeline (density -> FFT -> mask -> scale -> sigma-A
/// -> map coefficients -> IFFT) and to refine B-factors.
pub struct XtalRefinement {
    /// The crystallographic unit cell.
    pub unit_cell: super::UnitCell,
    /// The space group operations.
    pub group_ops: GroupOps,
    /// Crystal system for anisotropic scaling constraints.
    pub crystal_system: CrystalSystem,
    /// Observed reflection data.
    pub reflections: Vec<Reflection>,
    /// Grid dimensions `[nu, nv, nw]`.
    pub grid_dims: [usize; 3],
    /// Current scaling result (fitted parameters).
    pub scaling: ScalingResult,
    /// Current sigma-A estimates.
    pub sigma_a: Option<SigmaAResult>,
}

impl XtalRefinement {
    /// Create a new refinement state.
    ///
    /// # Arguments
    ///
    /// * `unit_cell` — the crystallographic unit cell
    /// * `sg` — the space group (consumed; ops and crystal system are
    ///   extracted)
    /// * `reflections` — observed reflection data
    /// * `grid_dims` — `[nu, nv, nw]` grid dimensions for the density map
    #[must_use]
    pub fn new(
        unit_cell: super::UnitCell,
        sg: SpaceGroup,
        reflections: Vec<Reflection>,
        grid_dims: [usize; 3],
    ) -> Self {
        // Filter out (0,0,0) and systematic absences — these must not
        // participate in scaling, sigma-A, R-factors, or ML targets.
        let filtered: Vec<Reflection> = reflections
            .into_iter()
            .filter(|r| {
                // Skip the origin reflection (unmeasurable, causes 1/d = Inf).
                if r.h == 0 && r.k == 0 && r.l == 0 {
                    return false;
                }
                // Skip systematic absences for this space group.
                !super::is_systematically_absent([r.h, r.k, r.l], &sg.ops)
            })
            .collect();

        Self {
            unit_cell,
            group_ops: sg.ops,
            crystal_system: sg.crystal_system,
            reflections: filtered,
            grid_dims,
            scaling: ScalingResult {
                k_overall: 1.0,
                b_aniso: [0.0; 6],
                k_sol: 0.35,
                b_sol: 46.0,
            },
            sigma_a: None,
        }
    }

    /// Run the full map computation pipeline.
    ///
    /// 1. Splat atomic density onto the grid
    /// 2. FFT -> Fc
    /// 3. Deblur Fc
    /// 4. Compute solvent mask -> FFT -> Fmask
    /// 5. Fit scaling parameters
    /// 6. Estimate sigma-A
    /// 7. Compute map coefficients
    /// 8. Inverse FFT -> real-space density
    ///
    /// Returns the 2mFo-DFc electron density grid, or `None` if the pipeline
    /// fails (e.g. FFT dimension mismatch).
    ///
    /// # Arguments
    ///
    /// * `positions` — fractional coordinates of ASU atoms
    /// * `elements` — element type per atom
    /// * `b_factors` — isotropic B-factor per atom
    /// * `occupancies` — site occupancy per atom
    #[allow(clippy::too_many_lines)]
    pub fn compute_map(
        &mut self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> Option<DensityGrid> {
        let [nu, nv, nw] = self.grid_dims;

        // Resolve form factors for each atom.
        let ffs: Vec<&FormFactor> = elements
            .iter()
            .filter_map(|e| form_factors::form_factor(*e))
            .collect();
        if ffs.len() != positions.len() {
            return None; // some elements have no form factor
        }

        // Compute blur from minimum B-factor and resolution.
        let b_min = b_factors.iter().copied().fold(f64::MAX, f64::min);
        let d_min = self.estimate_d_min();
        let blur = density::compute_blur(d_min, 1.5, b_min);

        // Step 1: Splat density.
        let mut grid = DensityGrid {
            data: vec![0.0; nu * nv * nw],
            nu,
            nv,
            nw,
        };
        let splat_params = density::SplatParams {
            positions,
            form_factors: &ffs,
            b_factors,
            occupancies,
            unit_cell: &self.unit_cell,
            blur,
        };
        density::splat_density(&mut grid, &splat_params);

        // Symmetrize.
        density::symmetrize_sum(&mut grid, &self.group_ops);

        // Step 2: FFT -> Fc.
        let fc_complex =
            fft_cpu::fft_3d_forward(&grid.data, nu, nv, nw).ok()?;

        // Step 3: Deblur.
        let mut fc_deblurred = fc_complex;
        density::deblur_fc(
            &mut fc_deblurred,
            &self.unit_cell,
            nu,
            nv,
            nw,
            blur,
        );

        // Step 4: Solvent mask -> Fmask.
        let mask = solvent_mask::solvent_mask(
            positions,
            elements,
            &self.unit_cell,
            nu,
            nv,
            nw,
            solvent_mask::DEFAULT_R_PROBE,
            solvent_mask::DEFAULT_R_SHRINK,
        );
        let fmask_complex =
            fft_cpu::fft_3d_forward(&mask.data, nu, nv, nw).ok()?;

        // Extract per-reflection Fc and Fmask values.
        let (refl_fc, refl_fmask) = self.extract_reflection_values(
            &fc_deblurred,
            &fmask_complex,
            nu,
            nv,
            nw,
        );

        // Fc amplitudes for scaling/sigma-A.
        let fc_amps: Vec<f64> = refl_fc
            .iter()
            .map(|c| f64::from(c[0]).hypot(f64::from(c[1])))
            .collect();

        // Step 5: Fit scaling.
        self.scaling = scaling::fit_scaling(
            &self.reflections,
            &refl_fc,
            &refl_fmask,
            &self.unit_cell,
            &self.crystal_system,
        );

        // Step 6: Sigma-A estimation.
        let sa = sigma_a::estimate_sigma_a(
            &self.reflections,
            &fc_amps,
            &self.unit_cell,
        );
        self.sigma_a = Some(sa.clone());

        // Step 7: Map coefficients.
        let map_coeffs = map_coefficients::compute_map_coefficients(
            &self.reflections,
            &refl_fc,
            &sa,
            &self.unit_cell,
        );

        // Step 8: Inverse FFT -> density.
        let density_data = map_coefficients::map_from_coefficients(
            &map_coeffs.two_fo_fc,
            &self.reflections,
            nu,
            nv,
            nw,
        )
        .ok()?;

        Some(DensityGrid {
            data: density_data,
            nu,
            nv,
            nw,
        })
    }

    /// Run one macro-cycle of B-factor refinement.
    ///
    /// Computes the B-factor gradient map, samples it at atomic positions, and
    /// takes a projected gradient descent step. Returns the updated B-factors.
    ///
    /// Requires that `compute_map` has been called at least once so that
    /// sigma-A estimates are available.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn refine_b_factors(
        &self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &mut [f64],
        occupancies: &[f64],
        learning_rate: f64,
    ) -> Option<()> {
        let [nu, nv, nw] = self.grid_dims;
        let sa = self.sigma_a.as_ref()?;

        // Resolve form factors.
        let ffs: Vec<&FormFactor> = elements
            .iter()
            .filter_map(|e| form_factors::form_factor(*e))
            .collect();
        if ffs.len() != positions.len() {
            return None;
        }

        // Compute Fc for current B-factors.
        let b_min = b_factors.iter().copied().fold(f64::MAX, f64::min);
        let d_min = self.estimate_d_min();
        let blur = density::compute_blur(d_min, 1.5, b_min);

        let mut grid = DensityGrid {
            data: vec![0.0; nu * nv * nw],
            nu,
            nv,
            nw,
        };
        let splat_params = density::SplatParams {
            positions,
            form_factors: &ffs,
            b_factors,
            occupancies,
            unit_cell: &self.unit_cell,
            blur,
        };
        density::splat_density(&mut grid, &splat_params);
        density::symmetrize_sum(&mut grid, &self.group_ops);

        let fc_complex =
            fft_cpu::fft_3d_forward(&grid.data, nu, nv, nw).ok()?;
        let mut fc_deblurred = fc_complex;
        density::deblur_fc(
            &mut fc_deblurred,
            &self.unit_cell,
            nu,
            nv,
            nw,
            blur,
        );

        let (refl_fc, _) = self.extract_reflection_values(
            &fc_deblurred,
            &fc_deblurred,
            nu,
            nv,
            nw,
        );

        // Compute B-factor gradient map.
        let grad_map = targets::b_factor_gradient_map(
            &self.reflections,
            &refl_fc,
            sa,
            &self.unit_cell,
            nu,
            nv,
            nw,
        )
        .ok()?;

        // Sample gradients at atom positions.
        let gradients = targets::per_atom_b_gradient(
            &grad_map,
            positions,
            occupancies,
            nu,
            nv,
            nw,
            self.unit_cell.volume,
        );

        // Gradient descent step.
        targets::refine_b_factors_step(b_factors, &gradients, learning_rate);

        Some(())
    }

    /// Run the full refinement loop for `n_macro_cycles`.
    ///
    /// Each cycle: recompute map, refine B-factors, check convergence.
    /// Returns `(r_work, r_free)` after the final cycle, or `None` on failure.
    #[allow(clippy::too_many_arguments)]
    pub fn refine(
        &mut self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &mut [f64],
        occupancies: &[f64],
        n_macro_cycles: usize,
        learning_rate: f64,
    ) -> Option<(f64, f64)> {
        for _cycle in 0..n_macro_cycles {
            // Recompute map (updates scaling + sigma-A).
            let _map =
                self.compute_map(positions, elements, b_factors, occupancies)?;

            // Refine B-factors.
            self.refine_b_factors(
                positions,
                elements,
                b_factors,
                occupancies,
                learning_rate,
            )?;
        }

        // Final R-factors.
        let r = self.r_factors(positions, elements, b_factors, occupancies)?;
        Some(r)
    }

    /// Compute R-work and R-free for the current model.
    ///
    /// Returns `(r_work, r_free)`, or `None` if the pipeline fails.
    pub fn r_factors(
        &self,
        positions: &[[f64; 3]],
        elements: &[Element],
        b_factors: &[f64],
        occupancies: &[f64],
    ) -> Option<(f64, f64)> {
        let [nu, nv, nw] = self.grid_dims;

        let ffs: Vec<&FormFactor> = elements
            .iter()
            .filter_map(|e| form_factors::form_factor(*e))
            .collect();
        if ffs.len() != positions.len() {
            return None;
        }

        let b_min = b_factors.iter().copied().fold(f64::MAX, f64::min);
        let d_min = self.estimate_d_min();
        let blur = density::compute_blur(d_min, 1.5, b_min);

        let mut grid = DensityGrid {
            data: vec![0.0; nu * nv * nw],
            nu,
            nv,
            nw,
        };
        let splat_params = density::SplatParams {
            positions,
            form_factors: &ffs,
            b_factors,
            occupancies,
            unit_cell: &self.unit_cell,
            blur,
        };
        density::splat_density(&mut grid, &splat_params);
        density::symmetrize_sum(&mut grid, &self.group_ops);

        let fc_complex =
            fft_cpu::fft_3d_forward(&grid.data, nu, nv, nw).ok()?;
        let mut fc_deblurred = fc_complex;
        density::deblur_fc(
            &mut fc_deblurred,
            &self.unit_cell,
            nu,
            nv,
            nw,
            blur,
        );

        let (refl_fc, _) = self.extract_reflection_values(
            &fc_deblurred,
            &fc_deblurred,
            nu,
            nv,
            nw,
        );

        let fc_amps: Vec<f64> = refl_fc
            .iter()
            .map(|c| f64::from(c[0]).hypot(f64::from(c[1])))
            .collect();

        let rw = targets::r_work(
            &self.reflections,
            &fc_amps,
            self.scaling.k_overall,
        );
        let rf = targets::r_free_value(
            &self.reflections,
            &fc_amps,
            self.scaling.k_overall,
        );

        Some((rw, rf))
    }

    /// Estimate the minimum d-spacing from the reflection data.
    fn estimate_d_min(&self) -> f64 {
        let mut max_s2 = 0.0_f64;
        for r in &self.reflections {
            let s2 = self.unit_cell.d_star_sq(r.h, r.k, r.l);
            if s2 > max_s2 {
                max_s2 = s2;
            }
        }
        if max_s2 > 0.0 {
            1.0 / max_s2.sqrt()
        } else {
            2.0 // default fallback
        }
    }

    /// Extract per-reflection complex values from a full 3D grid.
    ///
    /// Maps each reflection's Miller indices to the grid and reads the complex
    /// value at that position. Indices that exceed the grid dimensions are
    /// treated as zero (missing coefficient).
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_arguments,
        clippy::cast_possible_truncation
    )]
    fn extract_reflection_values(
        &self,
        fc_grid: &[[f32; 2]],
        fmask_grid: &[[f32; 2]],
        nu: usize,
        nv: usize,
        nw: usize,
    ) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
        let mut refl_fc = Vec::with_capacity(self.reflections.len());
        let mut refl_fmask = Vec::with_capacity(self.reflections.len());

        for r in &self.reflections {
            let u = wrap_miller_index(r.h, nu);
            let v = wrap_miller_index(r.k, nv);
            let w = wrap_miller_index(r.l, nw);

            let idx = (u * nv + v) * nw + w;
            if idx < fc_grid.len() {
                refl_fc.push(fc_grid[idx]);
            } else {
                refl_fc.push([0.0, 0.0]);
            }
            if idx < fmask_grid.len() {
                refl_fmask.push(fmask_grid[idx]);
            } else {
                refl_fmask.push([0.0, 0.0]);
            }
        }

        (refl_fc, refl_fmask)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::super::types::{Symop, UnitCell};
    use super::super::{space_group, SpaceGroup};
    use super::*;
    use crate::element::Element;

    /// Smoke test: construct an XtalRefinement and compute a map.
    #[test]
    fn compute_map_smoke() {
        let cell = UnitCell::new(20.0, 20.0, 20.0, 90.0, 90.0, 90.0);
        let sg = space_group(1); // P1
        let sg = sg.unwrap_or_else(|| SpaceGroup {
            number: 1,
            hm: "P 1",
            ops: GroupOps {
                sym_ops: vec![Symop {
                    rot: [[24, 0, 0], [0, 24, 0], [0, 0, 24]],
                    tran: [0, 0, 0],
                }],
                cen_ops: vec![[0, 0, 0]],
            },
            crystal_system: CrystalSystem::Triclinic,
        });

        // Simple reflections.
        let mut reflections = Vec::new();
        for h in 1..=3_i32 {
            for k in 0..=2_i32 {
                for l in 0..=2_i32 {
                    reflections.push(Reflection {
                        h,
                        k,
                        l,
                        f_obs: 10.0,
                        sigma_f: 1.0,
                        free_flag: false,
                    });
                }
            }
        }

        let grid_dims = [16, 16, 16];
        let mut refinement =
            XtalRefinement::new(cell, sg, reflections, grid_dims);

        let positions = [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]];
        let elements = [Element::C, Element::N];
        let b_factors = [20.0, 25.0];
        let occupancies = [1.0, 1.0];

        let result = refinement.compute_map(
            &positions,
            &elements,
            &b_factors,
            &occupancies,
        );

        assert!(result.is_some(), "compute_map should succeed");
        let grid = result.expect("verified above");
        assert_eq!(grid.data.len(), 16 * 16 * 16);

        // Grid should have non-zero values.
        let has_nonzero = grid.data.iter().any(|&v| v.abs() > 1e-10);
        assert!(has_nonzero, "density grid should have non-zero values");
    }

    /// Verify that r_factors returns something reasonable.
    #[test]
    fn r_factors_smoke() {
        let cell = UnitCell::new(20.0, 20.0, 20.0, 90.0, 90.0, 90.0);
        let sg = space_group(1).expect("P1 should exist");

        let mut reflections = Vec::new();
        for h in 1..=2_i32 {
            for k in 0..=1_i32 {
                for l in 0..=1_i32 {
                    reflections.push(Reflection {
                        h,
                        k,
                        l,
                        f_obs: 10.0,
                        sigma_f: 1.0,
                        free_flag: false,
                    });
                }
            }
        }

        let grid_dims = [16, 16, 16];
        let mut refinement =
            XtalRefinement::new(cell, sg, reflections, grid_dims);

        let positions = [[0.5, 0.5, 0.5]];
        let elements = [Element::C];
        let b_factors = [20.0];
        let occupancies = [1.0];

        // Must compute map first to set scaling.
        let _ = refinement.compute_map(
            &positions,
            &elements,
            &b_factors,
            &occupancies,
        );

        let r = refinement.r_factors(
            &positions,
            &elements,
            &b_factors,
            &occupancies,
        );
        assert!(r.is_some(), "r_factors should return Some");
    }

    /// (0,0,0) and systematic absences are filtered out at construction.
    #[test]
    fn new_filters_origin_and_systematic_absences() {
        let cell = UnitCell::new(50.0, 60.0, 70.0, 90.0, 90.0, 90.0);
        // C 1 2 1 (SG 5): C-centering requires h+k even.
        let sg = space_group(5).expect("C 1 2 1 should exist");

        let reflections = vec![
            // (0,0,0) — should be filtered (origin)
            Reflection {
                h: 0,
                k: 0,
                l: 0,
                f_obs: 999.0,
                sigma_f: 1.0,
                free_flag: false,
            },
            // (1,0,1) — h+k = 1 (odd) → systematically absent in C lattice
            Reflection {
                h: 1,
                k: 0,
                l: 1,
                f_obs: 50.0,
                sigma_f: 1.0,
                free_flag: false,
            },
            // (2,0,1) — h+k = 2 (even) → valid
            Reflection {
                h: 2,
                k: 0,
                l: 1,
                f_obs: 30.0,
                sigma_f: 1.0,
                free_flag: false,
            },
            // (1,1,1) — h+k = 2 (even) → valid
            Reflection {
                h: 1,
                k: 1,
                l: 1,
                f_obs: 20.0,
                sigma_f: 1.0,
                free_flag: false,
            },
        ];

        let grid_dims = [16, 16, 16];
        let refinement = XtalRefinement::new(cell, sg, reflections, grid_dims);

        // Should have 2 reflections: (2,0,1) and (1,1,1)
        assert_eq!(
            refinement.reflections.len(),
            2,
            "expected 2 reflections after filtering, got {}",
            refinement.reflections.len()
        );
        // Verify the survivors
        assert_eq!(refinement.reflections[0].h, 2);
        assert_eq!(refinement.reflections[1].h, 1);
        assert_eq!(refinement.reflections[1].k, 1);
    }
}
