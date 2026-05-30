//! Gaussian molecular surface scalar field generation.
//!
//! Each atom contributes a 3D Gaussian blob to a scalar field; the
//! isosurface of the summed field produces a smooth, blobby surface
//! that envelops the molecule. Resolution controls smoothness: low
//! values give a coarse overview, high values resolve individual
//! atoms.
//!
//! This module only produces the scalar field. Isosurface extraction
//! (marching cubes, etc.) and vertex formatting are the caller's job.

use glam::Vec3;

use super::{GridSpec, ScalarVoxelGrid};

/// Compute the Gaussian molecular surface scalar field.
///
/// - `positions`: atom world-space positions (Angstroms)
/// - `radii`: per-atom van der Waals radii (Angstroms)
/// - `resolution`: grid spacing in Angstroms (lower = finer; 0.5-2.0 typical)
///
/// Each atom contributes `amplitude * exp(-r^2 / (2*sigma^2))` with
/// `sigma = 0.7*vdW` and `amplitude = vdW`. Gaussian contributions are
/// clipped at `3*sigma` for efficiency.
///
/// Returns an empty grid (zero dims) for empty input. The field is
/// zero-initialized on unaffected voxels; the isosurface threshold
/// `level` chosen by the caller determines where the surface lies.
#[must_use]
pub fn compute_gaussian_field(
    positions: &[Vec3],
    radii: &[f32],
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

    let padding = 3.0 * resolution;

    // Bounding box
    let mut min = positions[0];
    let mut max = positions[0];
    for &p in positions {
        min = min.min(p);
        max = max.max(p);
    }
    let max_radius = radii.iter().copied().fold(0.0f32, f32::max);
    let expand = max_radius + padding;
    let spec = GridSpec::from_bounds(
        min - Vec3::splat(expand),
        max + Vec3::splat(expand),
        resolution,
    );

    let mut grid = vec![0.0f32; spec.voxel_count()];

    for (i, &pos) in positions.iter().enumerate() {
        splat_gaussian(&mut grid, &spec, pos, radii[i]);
    }

    ScalarVoxelGrid {
        dims: spec.dims,
        origin: spec.origin,
        spacing: spec.spacing,
        data: grid,
    }
}

/// Splat a single atom's Gaussian blob onto the grid.
///
/// World->voxel index casts are clamped to `[0, dim - 1]` before the
/// `as usize` conversion, and voxel indices are bounded by the grid
/// dimensions (which fit f32 mantissa precision).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "world->voxel casts are clamped before truncation and voxel \
              indices are bounded by grid dims (<< 2^2^4, fits f32 mantissa)"
)]
fn splat_gaussian(
    grid: &mut [f32],
    spec: &GridSpec,
    pos: Vec3,
    vdw_radius: f32,
) {
    let [nx, ny, nz] = spec.dims;
    let origin = spec.origin;
    let spacing = spec.spacing;
    let sigma = vdw_radius * 0.7;
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let cutoff = 3.0 * sigma;
    let cutoff2 = cutoff * cutoff;
    let amplitude = vdw_radius;

    let gx0 =
        ((pos.x - cutoff - origin[0]) / spacing[0]).floor().max(0.0) as usize;
    let gy0 =
        ((pos.y - cutoff - origin[1]) / spacing[1]).floor().max(0.0) as usize;
    let gz0 =
        ((pos.z - cutoff - origin[2]) / spacing[2]).floor().max(0.0) as usize;
    let gx1 = (((pos.x + cutoff - origin[0]) / spacing[0]).ceil() as usize)
        .min(nx - 1);
    let gy1 = (((pos.y + cutoff - origin[1]) / spacing[1]).ceil() as usize)
        .min(ny - 1);
    let gz1 = (((pos.z + cutoff - origin[2]) / spacing[2]).ceil() as usize)
        .min(nz - 1);

    for ix in gx0..=gx1 {
        let dx = (ix as f32).mul_add(spacing[0], origin[0]) - pos.x;
        for iy in gy0..=gy1 {
            let dy = (iy as f32).mul_add(spacing[1], origin[1]) - pos.y;
            let dxy2 = dx.mul_add(dx, dy * dy);
            if dxy2 > cutoff2 {
                continue;
            }
            for iz in gz0..=gz1 {
                let dz = (iz as f32).mul_add(spacing[2], origin[2]) - pos.z;
                let r2 = dz.mul_add(dz, dxy2);
                if r2 <= cutoff2 {
                    grid[spec.lin(ix, iy, iz)] = amplitude.mul_add(
                        (-r2 * inv_2sigma2).exp(),
                        grid[spec.lin(ix, iy, iz)],
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_empty_input() {
        let g = compute_gaussian_field(&[], &[], 1.0);
        assert_eq!(g.dims, [0, 0, 0]);
        assert!(g.data.is_empty());
    }

    #[test]
    fn gaussian_single_atom_nonzero_at_center() {
        let g = compute_gaussian_field(&[Vec3::ZERO], &[1.5], 0.5);
        assert_eq!(g.data.len(), g.dims[0] * g.dims[1] * g.dims[2]);
        // Field should peak somewhere inside the grid.
        let max = g.data.iter().copied().fold(0.0f32, f32::max);
        assert!(max > 0.0, "gaussian field should have positive peak");
    }
}
