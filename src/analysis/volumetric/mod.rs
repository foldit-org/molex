//! Volumetric analysis over uniform voxel grids.
//!
//! Primitives and pipelines that rasterize atoms into a voxel grid and
//! run distance-field / connectivity analyses on the result. Everything
//! in this module is pure CPU computation with no rendering concepts;
//! the output is a scalar voxel field or a list of detected cavities,
//! ready to be consumed by any mesher or downstream analysis.
//!
//! **Grid primitives** ([`crate::analysis::volumetric::grid`]):
//! solvent-accessible-surface (SAS) rasterization,
//! Felzenszwalb-Huttenlocher 3D Euclidean distance transform,
//! binary-to-SDF conversion, exterior flood-fill for cavity masks.
//!
//! **Detection pipelines**:
//! - [`crate::analysis::volumetric::detect_cavities`] ->
//!   [`Vec<DetectedCavity>`]: internal cavities found via SES erosion + flood
//!   fill + connected components.
//! - [`crate::analysis::volumetric::compute_ses_sdf`] -> [`ScalarVoxelGrid`]:
//!   the solvent-excluded-surface signed distance field.
//! - [`crate::analysis::volumetric::compute_gaussian_field`] ->
//!   [`ScalarVoxelGrid`]: summed Gaussian blobs for smooth molecular surface
//!   rendering.

pub mod cavity;
pub mod gaussian;
pub mod grid;
pub mod ses;

pub use cavity::{detect_cavities, DetectedCavity, VoxelBbox};
pub use gaussian::compute_gaussian_field;
use glam::Vec3;
pub use grid::binary_to_sdf;
pub use ses::compute_ses_sdf;

/// Geometry of a uniform orthogonal voxel grid: dimensions, world-space
/// origin, and per-axis spacing.
///
/// The three fields always travel together when rasterizing world-space
/// quantities into voxel indices and vice versa. Bundled to avoid
/// six-argument helper functions and to centralize the grid-sizing
/// arithmetic via [`GridSpec::from_bounds`].
///
/// Indexing convention is `x-major -> y -> z`; use [`GridSpec::lin`] to
/// compute linear indices.
#[derive(Debug, Clone, Copy)]
pub struct GridSpec {
    /// `[nx, ny, nz]`: grid dimensions in voxels.
    pub dims: [usize; 3],
    /// World-space position of voxel `(0, 0, 0)` (Angstroms).
    pub origin: [f32; 3],
    /// Angstroms per voxel along each axis.
    pub spacing: [f32; 3],
}

impl GridSpec {
    /// Build a grid that brackets `[min, max]` at the requested
    /// `resolution` (Angstroms / voxel).
    ///
    /// Each axis is rounded up so the requested extent fits with at
    /// least 2 voxels; spacing is then chosen so voxels span the
    /// extent inclusively (`spacing = extent / (n - 1)`).
    ///
    /// The `usize -> f32` cast on dimensions is precision-safe: voxel
    /// counts are bounded by available memory (<< 2^2^4) and so always fit
    /// in f32's 23-bit mantissa exactly.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "voxel count is bounded by memory (<< 2^2^4), always fits \
                  f32 mantissa and round-tripping through usize is the \
                  natural representation"
    )]
    pub fn from_bounds(min: Vec3, max: Vec3, resolution: f32) -> Self {
        let extent = max - min;
        let nx = ((extent.x / resolution).ceil() as usize).max(2);
        let ny = ((extent.y / resolution).ceil() as usize).max(2);
        let nz = ((extent.z / resolution).ceil() as usize).max(2);
        let spacing = [
            extent.x / (nx - 1) as f32,
            extent.y / (ny - 1) as f32,
            extent.z / (nz - 1) as f32,
        ];
        Self {
            dims: [nx, ny, nz],
            origin: [min.x, min.y, min.z],
            spacing,
        }
    }

    /// Total number of voxels in this grid.
    #[must_use]
    pub const fn voxel_count(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Linear index for `(ix, iy, iz)` in x-major, y, z order.
    ///
    /// Caller is responsible for `ix < dims[0]` etc.; this is a
    /// hot-path helper and does not bounds-check.
    #[must_use]
    #[inline]
    pub const fn lin(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix * self.dims[1] * self.dims[2] + iy * self.dims[2] + iz
    }
}

/// A scalar value on a uniform orthogonal voxel grid.
///
/// Lightweight counterpart to [`crate::entity::surface::VoxelGrid`],
/// which carries the full crystallographic cell parameters needed for
/// density maps. This type is for the common case where the grid is
/// axis-aligned and uniformly spaced: just dimensions, origin, and
/// spacing.
///
/// Indexing is `x-major -> y -> z`:
/// `index = ix * dims[1] * dims[2] + iy * dims[2] + iz`.
#[derive(Debug, Clone)]
pub struct ScalarVoxelGrid {
    /// `[nx, ny, nz]`: grid dimensions in voxels.
    pub dims: [usize; 3],
    /// World-space position of voxel `(0, 0, 0)` (Angstroms).
    pub origin: [f32; 3],
    /// Angstroms per voxel along each axis.
    pub spacing: [f32; 3],
    /// Scalar values, one per voxel, laid out `x-major -> y -> z`.
    pub data: Vec<f32>,
}

impl ScalarVoxelGrid {
    /// Convert integer voxel coordinates to world-space (Angstroms).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn voxel_to_world(&self, ix: usize, iy: usize, iz: usize) -> [f32; 3] {
        [
            (ix as f32).mul_add(self.spacing[0], self.origin[0]),
            (iy as f32).mul_add(self.spacing[1], self.origin[1]),
            (iz as f32).mul_add(self.spacing[2], self.origin[2]),
        ]
    }

    /// Convert fractional voxel coordinates to world-space (Angstroms).
    ///
    /// Useful for sub-voxel interpolation (e.g. marching-cubes edge
    /// crossings).
    #[must_use]
    pub fn voxel_to_world_f32(&self, gx: f32, gy: f32, gz: f32) -> [f32; 3] {
        [
            gx.mul_add(self.spacing[0], self.origin[0]),
            gy.mul_add(self.spacing[1], self.origin[1]),
            gz.mul_add(self.spacing[2], self.origin[2]),
        ]
    }
}
