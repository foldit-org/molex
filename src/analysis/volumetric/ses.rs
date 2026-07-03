//! Solvent-Excluded Surface (SES / Connolly surface) signed distance
//! field generation.
//!
//! Uses the EDTSurf algorithm:
//! 1. Voxelize the SAS solid (binary: inside any atom's vdW + probe)
//! 2. 3D Euclidean Distance Transform on the solid interior
//! 3. Carve voxels where EDT < probe (probe can reach from outside)
//! 4. Seal interior cavities back into the solid (so marching cubes produces
//!    only the outer envelope)
//! 5. Convert the resulting SES solid to a signed distance field
//!
//! The output is a [`ScalarVoxelGrid`] with the standard SDF sign
//! convention: **negative inside, positive outside**. Downstream
//! mesh extraction that wants the opposite sign (e.g. marching cubes
//! configured for positive-inside) should negate the data at use site.

use glam::Vec3;

use super::grid::{binary_to_sdf, detect_cavity_mask, edt_3d, voxelize_sas};
use super::{GridSpec, ScalarVoxelGrid};

/// Default solvent probe radius (water) in Angstroms.
const DEFAULT_PROBE_RADIUS: f32 = 1.4;

/// Compute the SES signed distance field for a set of atoms.
///
/// - `positions`: atom world-space positions (Angstroms)
/// - `radii`: per-atom van der Waals radii (Angstroms)
/// - `probe_radius`: solvent probe radius; defaults to 1.4 A
/// - `resolution`: grid spacing in Angstroms (lower = finer; 0.5-1.0 typical)
///
/// Returns a [`ScalarVoxelGrid`] with the SDF sampled on a uniform
/// orthogonal grid. Sign convention: **negative inside, positive
/// outside** (flip at use site if your mesher wants the opposite).
/// Returns an empty grid (zero dims) for empty input.
#[must_use]
pub fn compute_ses_sdf(
    positions: &[Vec3],
    radii: &[f32],
    probe_radius: Option<f32>,
    resolution: f32,
) -> ScalarVoxelGrid {
    if positions.is_empty() {
        return ScalarVoxelGrid {
            dims: [0, 0, 0],
            origin: [0.0; 3],
            spacing: [0.0; 3],
            data: Vec::new(),
        };
    }

    let probe = probe_radius.unwrap_or(DEFAULT_PROBE_RADIUS);
    let padding = probe + 2.0;

    // Compute grid bounds
    let mut min_bound = positions[0];
    let mut max_bound = positions[0];
    for &p in positions {
        min_bound = min_bound.min(p);
        max_bound = max_bound.max(p);
    }
    let max_radius = radii.iter().copied().fold(0.0f32, f32::max);
    let expand = max_radius + probe + padding;
    let spec = GridSpec::from_bounds(
        min_bound - Vec3::splat(expand),
        max_bound + Vec3::splat(expand),
        resolution,
    );

    // Step 1: Build binary SAS solid
    let sas_solid = voxelize_sas(positions, radii, probe, &spec);

    // Step 2: EDT on interior -> distance to nearest outside voxel
    let interior_edt = edt_3d(&sas_solid, spec.dims, &spec.spacing);

    // Step 3: Carve. Voxels where EDT < probe become outside
    let total = spec.voxel_count();
    let mut ses_solid = vec![false; total];
    for i in 0..total {
        ses_solid[i] = sas_solid[i] && interior_edt[i] >= probe;
    }

    // Step 3b: Fill interior voids so a mesher produces only the outer
    // envelope. The cavity detection uses the same primitive; here we
    // just consume the mask to seal up the solid.
    let cavity_mask = detect_cavity_mask(&ses_solid, spec.dims);
    for (i, &in_cavity) in cavity_mask.iter().enumerate() {
        if in_cavity {
            ses_solid[i] = true;
        }
    }

    // Step 4: Convert SES solid to signed distance field (negative
    // inside, positive outside).
    let sdf = binary_to_sdf(&ses_solid, spec.dims, &spec.spacing);

    ScalarVoxelGrid {
        dims: spec.dims,
        origin: spec.origin,
        spacing: spec.spacing,
        data: sdf,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn ses_empty_input() {
        let g = compute_ses_sdf(&[], &[], None, 1.0);
        assert_eq!(g.dims, [0, 0, 0]);
        assert!(g.data.is_empty());
    }

    #[test]
    fn ses_single_atom_produces_field() {
        let pos = vec![Vec3::ZERO];
        let radii = vec![1.5];
        let g = compute_ses_sdf(&pos, &radii, Some(1.4), 0.5);
        assert!(g.dims[0] > 0);
        assert_eq!(g.data.len(), g.dims[0] * g.dims[1] * g.dims[2]);
        // There should be some negative (inside) voxels near the atom.
        assert!(
            g.data.iter().any(|&v| v < 0.0),
            "expected at least one interior voxel (negative SDF)"
        );
    }
}
