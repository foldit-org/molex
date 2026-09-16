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

/// Width, amplitude, and optional smooth cutoff for each atom's density.
#[derive(Debug, Clone, Copy)]
pub struct GaussianOptions {
    /// Gaussian standard deviation divided by the atom radius.
    pub sigma_scale: f32,
    /// Constant peak density; `None` uses each atom's radius.
    pub amplitude: Option<f32>,
    /// Join the Gaussian at `2*sigma` to a quadratic ending at `3*sigma`.
    pub quadratic_tail: bool,
}

impl Default for GaussianOptions {
    fn default() -> Self {
        Self {
            sigma_scale: 0.7,
            amplitude: None,
            quadratic_tail: false,
        }
    }
}

/// Failure to construct a Gaussian grid.
#[derive(Debug, thiserror::Error)]
pub enum GaussianGridError {
    /// An input is invalid or cannot be represented by the FP32 rasterizer.
    #[error("{0}")]
    InvalidInput(&'static str),
    /// The requested grid cannot be allocated.
    #[error("cannot allocate Gaussian grid: {0}")]
    Allocation(#[from] std::collections::TryReserveError),
}

/// Rasterize atoms onto an explicit grid, in x-major, y, z order.
///
/// With the tail enabled, density beyond `2*sigma` is
/// `amplitude * exp(-2) * (3 - distance/sigma)^2`, ending at `3*sigma`.
/// Empty atoms yield a zero grid.
///
/// # Errors
/// Returns an error for mismatched arrays, nonfinite inputs, nonpositive
/// radii/width/spacing, unrepresentable grid geometry, or allocation failure.
pub fn compute_gaussian_field_on_grid(
    positions: &[Vec3],
    radii: &[f32],
    spec: GridSpec,
    options: GaussianOptions,
) -> Result<ScalarVoxelGrid, GaussianGridError> {
    use GaussianGridError::InvalidInput;

    if positions.len() != radii.len() {
        return Err(InvalidInput(
            "positions and radii must have equal lengths",
        ));
    }
    if !options.sigma_scale.is_finite()
        || options.sigma_scale <= 0.0
        || options.amplitude.is_some_and(|a| !a.is_finite())
    {
        return Err(InvalidInput(
            "width must be finite and positive; amplitude must be finite",
        ));
    }
    let count = checked_voxel_count(spec)?;
    for (&pos, &radius) in positions.iter().zip(radii) {
        let sigma = radius * options.sigma_scale;
        let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
        let cutoff = 3.0 * sigma;
        if !pos.is_finite()
            || !radius.is_finite()
            || radius <= 0.0
            || !sigma.is_finite()
            || sigma <= 0.0
            || !inv_2sigma2.is_finite()
            || inv_2sigma2 <= 0.0
            || !(cutoff * cutoff).is_finite()
        {
            return Err(InvalidInput(
                "positions must be finite and radii must produce a finite \
                 positive width and kernel",
            ));
        }
    }

    let mut grid = Vec::new();
    grid.try_reserve_exact(count)?;
    grid.resize(count, 0.0);
    for (&pos, &radius) in positions.iter().zip(radii) {
        splat_gaussian(&mut grid, &spec, pos, radius, options);
    }
    Ok(ScalarVoxelGrid {
        dims: spec.dims,
        origin: spec.origin,
        spacing: spec.spacing,
        data: grid,
    })
}

#[allow(clippy::cast_precision_loss, reason = "dimensions are bounded by 2^24")]
fn checked_voxel_count(spec: GridSpec) -> Result<usize, GaussianGridError> {
    use GaussianGridError::InvalidInput;

    let mut count = 1usize;
    for axis in 0..3 {
        let dim = spec.dims[axis];
        let spacing = spec.spacing[axis];
        let origin = spec.origin[axis];
        if dim == 0
            || dim > (1 << 24)
            || !origin.is_finite()
            || !spacing.is_finite()
            || spacing <= 0.0
        {
            return Err(InvalidInput(
                "grid dimensions must be in 1..=2^24; origin and positive \
                 spacing must be finite",
            ));
        }
        if !((dim - 1) as f32).mul_add(spacing, origin).is_finite() {
            return Err(InvalidInput("grid endpoint must be finite"));
        }
        count = count
            .checked_mul(dim)
            .ok_or(InvalidInput("grid voxel count overflows"))?;
    }
    Ok(count)
}

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
        splat_gaussian(
            &mut grid,
            &spec,
            pos,
            radii[i],
            GaussianOptions::default(),
        );
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
    options: GaussianOptions,
) {
    let [nx, ny, nz] = spec.dims;
    let origin = spec.origin;
    let spacing = spec.spacing;
    let sigma = vdw_radius * options.sigma_scale;
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let cutoff = 3.0 * sigma;
    let cutoff2 = cutoff * cutoff;
    let amplitude = options.amplitude.unwrap_or(vdw_radius);
    let join2 = (2.0 * sigma) * (2.0 * sigma);
    let tail_scale = (-2.0f32).exp();

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
                if r2 > cutoff2 || r2.is_nan() {
                    continue;
                }
                let density = if options.quadratic_tail && r2 > join2 {
                    let remaining = (3.0 - r2.sqrt() / sigma).max(0.0);
                    tail_scale * remaining * remaining
                } else {
                    (-r2 * inv_2sigma2).exp()
                };
                grid[spec.lin(ix, iy, iz)] =
                    amplitude.mul_add(density, grid[spec.lin(ix, iy, iz)]);
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

    #[test]
    fn quadratic_tail_matches_piecewise_formula(
    ) -> Result<(), GaussianGridError> {
        for (distance, quadratic_tail, expected) in [
            (0.0, true, 1.0),
            (2.0, true, (-2.0f32).exp()),
            (2.5, true, 0.25 * (-2.0f32).exp()),
            (2.5, false, (-3.125f32).exp()),
            (3.0, true, 0.0),
        ] {
            let spec = GridSpec {
                dims: [1; 3],
                origin: [distance, 0.0, 0.0],
                spacing: [1.0; 3],
            };
            let options = GaussianOptions {
                sigma_scale: 0.5,
                amplitude: Some(1.0),
                quadratic_tail,
            };
            let grid = compute_gaussian_field_on_grid(
                &[Vec3::ZERO],
                &[2.0],
                spec,
                options,
            )?;
            assert!(
                (grid.data[0] - expected).abs() <= 1e-7,
                "d={distance} tail={quadratic_tail}: {} != {expected}",
                grid.data[0]
            );
        }
        Ok(())
    }
}
