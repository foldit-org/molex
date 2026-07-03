//! Volumetric grid and electron density types.

use ndarray::Array3;

/// A generic 3D voxel grid with spatial mapping.
///
/// Stores a 3D array of scalar values with the information needed to map
/// between grid indices and Cartesian coordinates. This is the reusable
/// foundation for any volumetric data: electron density, cryo-EM
/// reconstructions, electrostatic potentials, distance fields, etc.
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    /// Grid dimension along X.
    pub nx: usize,
    /// Grid dimension along Y.
    pub ny: usize,
    /// Grid dimension along Z.
    pub nz: usize,
    /// Grid start index along X.
    pub nxstart: i32,
    /// Grid start index along Y.
    pub nystart: i32,
    /// Grid start index along Z.
    pub nzstart: i32,
    /// Unit cell sampling intervals along X.
    pub mx: usize,
    /// Unit cell sampling intervals along Y.
    pub my: usize,
    /// Unit cell sampling intervals along Z.
    pub mz: usize,
    /// Unit cell dimensions a, b, c in angstroms.
    pub cell_dims: [f32; 3],
    /// Unit cell angles alpha, beta, gamma in degrees.
    pub cell_angles: [f32; 3],
    /// Origin in angstroms.
    pub origin: [f32; 3],
    /// 3D grid of scalar values, indexed as `data[[x, y, z]]`.
    pub data: Array3<f32>,
}

impl VoxelGrid {
    /// Angstroms per voxel along each axis.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn voxel_size(&self) -> [f32; 3] {
        [
            self.cell_dims[0] / self.mx as f32,
            self.cell_dims[1] / self.my as f32,
            self.cell_dims[2] / self.mz as f32,
        ]
    }

    /// Build the 3x3 fractional-to-Cartesian matrix.
    ///
    /// For orthogonal cells (alpha=beta=gamma=90 deg) this is diagonal with
    /// (a,b,c). For non-orthogonal cells the off-diagonal terms handle the
    /// skew.
    #[must_use]
    pub fn frac_to_cart_matrix(&self) -> [[f32; 3]; 3] {
        let [a, b, c] = self.cell_dims;
        let alpha = self.cell_angles[0].to_radians();
        let beta = self.cell_angles[1].to_radians();
        let gamma = self.cell_angles[2].to_radians();

        let cos_a = alpha.cos();
        let cos_b = beta.cos();
        let cos_g = gamma.cos();
        let sin_g = gamma.sin();

        let xi = cos_b.mul_add(-cos_g, cos_a) / sin_g;
        let sin_b = beta.sin();
        let zeta = sin_b.mul_add(sin_b, -(xi * xi)).max(0.0).sqrt();

        [
            [a, b * cos_g, c * cos_b],
            [0.0, b * sin_g, c * xi],
            [0.0, 0.0, c * zeta],
        ]
    }

    /// Convert grid indices to Cartesian coordinates in angstroms.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn grid_to_cartesian(
        &self,
        ix: usize,
        iy: usize,
        iz: usize,
    ) -> [f32; 3] {
        self.grid_to_cartesian_f32(ix as f32, iy as f32, iz as f32)
    }

    /// Convert fractional grid positions to Cartesian coordinates.
    ///
    /// Accepts fractional grid positions for sub-voxel interpolation
    /// (e.g. from marching cubes edge interpolation).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn grid_to_cartesian_f32(&self, gx: f32, gy: f32, gz: f32) -> [f32; 3] {
        let fx = (self.nxstart as f32 + gx) / self.mx as f32;
        let fy = (self.nystart as f32 + gy) / self.my as f32;
        let fz = (self.nzstart as f32 + gz) / self.mz as f32;

        let m = self.frac_to_cart_matrix();
        [
            m[0][0].mul_add(fx, m[0][1].mul_add(fy, m[0][2] * fz))
                + self.origin[0],
            m[1][1].mul_add(fy, m[1][2] * fz) + self.origin[1],
            m[2][2].mul_add(fz, self.origin[2]),
        ]
    }

    /// Convert Cartesian coordinates back to fractional grid indices.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cartesian_to_grid(&self, cart: [f32; 3]) -> [f32; 3] {
        let cx = cart[0] - self.origin[0];
        let cy = cart[1] - self.origin[1];
        let cz = cart[2] - self.origin[2];

        let m = self.frac_to_cart_matrix();
        let fz = cz / m[2][2];
        let fy = m[1][2].mul_add(-fz, cy) / m[1][1];
        let fx = m[0][2].mul_add(-fz, m[0][1].mul_add(-fy, cx)) / m[0][0];

        [
            fx.mul_add(self.mx as f32, -(self.nxstart as f32)),
            fy.mul_add(self.my as f32, -(self.nystart as f32)),
            fz.mul_add(self.mz as f32, -(self.nzstart as f32)),
        ]
    }
}

/// Electron density map parsed from MRC/CCP4 format.
///
/// Wraps a [`VoxelGrid`] with density-specific statistics (min, max, mean,
/// RMS) and a space group number. Works for both X-ray crystallography
/// and cryo-EM maps (cryo-EM uses space group P1).
#[derive(Debug, Clone)]
pub struct Density {
    /// The underlying voxel grid.
    pub grid: VoxelGrid,
    /// Minimum density value.
    pub dmin: f32,
    /// Maximum density value.
    pub dmax: f32,
    /// Mean density value.
    pub dmean: f32,
    /// RMS deviation from mean density.
    pub rms: f32,
    /// Space group number (1 = P1 for cryo-EM).
    pub space_group: u32,
}

impl std::ops::Deref for Density {
    type Target = VoxelGrid;

    fn deref(&self) -> &VoxelGrid {
        &self.grid
    }
}

impl std::ops::DerefMut for Density {
    fn deref_mut(&mut self) -> &mut VoxelGrid {
        &mut self.grid
    }
}

impl Density {
    /// Density threshold at a given sigma level: `dmean + sigma * rms`.
    #[must_use]
    pub fn sigma_level(&self, sigma: f32) -> f32 {
        sigma.mul_add(self.rms, self.dmean)
    }
}

/// Errors that can occur when parsing a density map.
#[derive(Debug, thiserror::Error)]
pub enum DensityError {
    /// The file header or data layout is invalid.
    #[error("invalid density map format: {0}")]
    InvalidFormat(String),

    /// The MRC data mode is not supported.
    #[error("unsupported MRC data mode: {0}")]
    UnsupportedMode(i32),

    /// An I/O error occurred while reading the map file.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
