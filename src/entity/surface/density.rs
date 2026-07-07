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

    /// Crop a whole-cell map to a shifted-origin sub-block enclosing `points`
    /// (Cartesian angstroms), padded by `pad_cells` grid cells per side.
    ///
    /// The bounding box is taken in absolute grid-index space and padded, then
    /// extracted into a smaller grid whose `nxstart`/`nx` carry the offset
    /// while `mx`/`cell`/`origin` stay unchanged, so the sub-block renders in
    /// place over the model. Deposited coordinates routinely sit outside the
    /// `[0,1)` fractional box (a molecule longer than a cell edge spans more
    /// than one repeat along that axis), so the window is free to reach past
    /// `mx`; the periodic (`pos_mod`) fill then wraps the density back onto
    /// those atoms. Within the window every voxel farther than
    /// [`MASK_RADIUS_A`] from all `points` is zeroed so the block hugs the
    /// model rather than showing a full periodic slab of symmetry mates. The
    /// statistics (`dmin`/`dmax`/`dmean`/`rms`) are recomputed over the masked
    /// sub-block so the sigma contour is relative to it.
    ///
    /// The source is assumed to be a full-cell map (`nxstart == 0` and
    /// `nx == mx` per axis); a non-full-cell map is returned unchanged.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn crop_to_points(
        &self,
        points: &[[f32; 3]],
        pad_cells: usize,
    ) -> Density {
        let g = &self.grid;
        let full_cell = g.nxstart == 0
            && g.nystart == 0
            && g.nzstart == 0
            && g.nx == g.mx
            && g.ny == g.my
            && g.nz == g.mz;
        debug_assert!(full_cell, "crop_to_points expects a full-cell map");
        if !full_cell || points.is_empty() {
            return self.clone();
        }

        let mx = [g.mx, g.my, g.mz];
        let (start, ext) = crop_bounds(g, points, pad_cells);

        let mut data = Array3::<f32>::zeros(ext);
        for i in 0..ext[0] {
            let sx = pos_mod(start[0] + i as i64, mx[0]);
            for j in 0..ext[1] {
                let sy = pos_mod(start[1] + j as i64, mx[1]);
                for k in 0..ext[2] {
                    let sz = pos_mod(start[2] + k as i64, mx[2]);
                    data[[i, j, k]] = g.data[[sx, sy, sz]];
                }
            }
        }

        mask_to_model(&mut data, g, points, start, ext);
        let (dmin, dmax, dmean, rms) = density_statistics(&data);

        Density {
            grid: VoxelGrid {
                nx: ext[0],
                ny: ext[1],
                nz: ext[2],
                nxstart: start[0] as i32,
                nystart: start[1] as i32,
                nzstart: start[2] as i32,
                mx: g.mx,
                my: g.my,
                mz: g.mz,
                cell_dims: g.cell_dims,
                cell_angles: g.cell_angles,
                origin: g.origin,
                data,
            },
            dmin,
            dmax,
            dmean,
            rms,
            space_group: self.space_group,
        }
    }
}

/// Absolute grid-index bounding box for `points`, padded by `pad_cells` cells
/// per side. Returns per-axis `(start, extent)`.
///
/// The extent is deliberately **not** clamped to one cell. Deposited atoms can
/// lie outside the `[0,1)` fractional box, and a molecule longer than a cell
/// edge spans more than one repeat along that axis; clamping to `mx` there
/// collapses the crop to a full periodic slab (all symmetry mates) that also
/// fails to enclose the out-of-box atoms. Letting the window run past `mx`
/// keeps it tight around the model, and [`crop_to_points`]'s `pos_mod` fill
/// wraps the periodic density back onto those atoms.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn crop_bounds(
    g: &VoxelGrid,
    points: &[[f32; 3]],
    pad_cells: usize,
) -> ([i64; 3], [usize; 3]) {
    let mut min_abs = [f32::INFINITY; 3];
    let mut max_abs = [f32::NEG_INFINITY; 3];
    for p in points {
        // For a full-cell map nxstart == 0, so the local index the grid math
        // returns already equals the absolute index.
        let gi = g.cartesian_to_grid(*p);
        for k in 0..3 {
            min_abs[k] = min_abs[k].min(gi[k]);
            max_abs[k] = max_abs[k].max(gi[k]);
        }
    }

    let pad = pad_cells as i64;
    let mut start = [0i64; 3];
    let mut ext = [0usize; 3];
    for k in 0..3 {
        let lo = min_abs[k].floor() as i64;
        let hi = max_abs[k].ceil() as i64;
        start[k] = lo - pad;
        ext[k] = ((hi - lo) + 2 * pad).max(1) as usize;
    }
    (start, ext)
}

/// Cartesian radius (angstroms) around each model atom kept by the render
/// crop's proximity mask. Voxels farther than this from every atom are zeroed
/// so the sub-block hugs the model instead of a periodic slab. Wide enough to
/// keep the bonded-density envelope connected, tight enough to drop
/// symmetry-mate and bulk-solvent blobs that fall inside the bounding box.
const MASK_RADIUS_A: f32 = 3.0;

/// Zero every voxel of `data` farther than [`MASK_RADIUS_A`] from all model
/// `points`, so the extracted sub-block hugs the model.
///
/// A voxel's Cartesian position is its in-place render position,
/// `g.grid_to_cartesian_f32(start + local)`. Because the window is free to
/// reach past one cell, that position lands on the model even for atoms
/// deposited outside the `[0,1)` box, so a plain nearest-atom distance is exact
/// without minimum-image folding. Each atom stamps only the voxels inside its
/// cutoff sphere, so the cost is `O(atoms * sphere_voxels)` rather than
/// `O(voxels * atoms)`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn mask_to_model(
    data: &mut Array3<f32>,
    g: &VoxelGrid,
    points: &[[f32; 3]],
    start: [i64; 3],
    ext: [usize; 3],
) {
    let mut keep = Array3::<bool>::from_elem(ext, false);
    for p in points {
        stamp_atom_sphere(&mut keep, g, *p, start, ext);
    }

    for i in 0..ext[0] {
        for j in 0..ext[1] {
            for k in 0..ext[2] {
                if !keep[[i, j, k]] {
                    data[[i, j, k]] = 0.0;
                }
            }
        }
    }
}

/// Mark `keep[i,j,k] = true` for block voxels within [`MASK_RADIUS_A`] of atom
/// `p`, scanning only the cutoff-sphere half-window around the atom's grid
/// position and clamping to the block bounds.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn stamp_atom_sphere(
    keep: &mut Array3<bool>,
    g: &VoxelGrid,
    p: [f32; 3],
    start: [i64; 3],
    ext: [usize; 3],
) {
    let r2 = MASK_RADIUS_A * MASK_RADIUS_A;
    let vs = g.voxel_size();
    // Grid-index half-window covering the cutoff sphere per axis. The `+ 2`
    // margin absorbs the round-to-nearest centering and mild cell skew (grid
    // steps run along the cell axes, so a Cartesian sphere is not perfectly
    // axis-aligned in index space); the exact distance test below is the real
    // gate, this only bounds the loop.
    let rv = [
        (MASK_RADIUS_A / vs[0]).ceil() as i64 + 2,
        (MASK_RADIUS_A / vs[1]).ceil() as i64 + 2,
        (MASK_RADIUS_A / vs[2]).ceil() as i64 + 2,
    ];
    let ci = [
        (g.cartesian_to_grid(p)[0] - start[0] as f32).round() as i64,
        (g.cartesian_to_grid(p)[1] - start[1] as f32).round() as i64,
        (g.cartesian_to_grid(p)[2] - start[2] as f32).round() as i64,
    ];
    let bound = |c: i64, r: i64, e: usize| {
        ((c - r).max(0) as usize)..((c + r + 1).max(0) as usize).min(e)
    };
    for i in bound(ci[0], rv[0], ext[0]) {
        for j in bound(ci[1], rv[1], ext[1]) {
            for k in bound(ci[2], rv[2], ext[2]) {
                if keep[[i, j, k]] {
                    continue;
                }
                let vc = g.grid_to_cartesian_f32(
                    (start[0] + i as i64) as f32,
                    (start[1] + j as i64) as f32,
                    (start[2] + k as i64) as f32,
                );
                let dx = vc[0] - p[0];
                let dy = vc[1] - p[1];
                let dz = vc[2] - p[2];
                if dx.mul_add(dx, dy.mul_add(dy, dz * dz)) <= r2 {
                    keep[[i, j, k]] = true;
                }
            }
        }
    }
}

/// Positive modulo mapping an absolute grid index into `[0, m)`.
#[inline]
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn pos_mod(a: i64, m: usize) -> usize {
    let m = m as i64;
    (((a % m) + m) % m) as usize
}

/// Compute `(dmin, dmax, dmean, rms)` over a density grid.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "voxel count to f64 for averaging; mean/rms narrow back to f32"
)]
fn density_statistics(data: &Array3<f32>) -> (f32, f32, f32, f32) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Full-cell 10x10x10 orthogonal map with `data[[x,y,z]] = x*100+y*10+z`.
    fn synthetic_full_cell() -> Density {
        let n = 10;
        let data = Array3::from_shape_fn((n, n, n), |(x, y, z)| {
            (x * 100 + y * 10 + z) as f32
        });
        let (dmin, dmax, dmean, rms) = density_statistics(&data);
        Density {
            grid: VoxelGrid {
                nx: n,
                ny: n,
                nz: n,
                nxstart: 0,
                nystart: 0,
                nzstart: 0,
                mx: n,
                my: n,
                mz: n,
                cell_dims: [10.0, 10.0, 10.0],
                cell_angles: [90.0, 90.0, 90.0],
                origin: [0.0, 0.0, 0.0],
                data,
            },
            dmin,
            dmax,
            dmean,
            rms,
            space_group: 1,
        }
    }

    #[test]
    fn crop_bbox_offset_and_in_place() {
        let src = synthetic_full_cell();
        // Orthogonal 10 A cell with mx=10 => grid index == cartesian value.
        let points = [[3.0f32, 3.0, 3.0], [5.0, 5.0, 5.0]];
        let cropped = src.crop_to_points(&points, 1);

        // bbox [3,5] per axis, pad 1: start = floor(3)-1 = 2,
        // ext = (ceil(5)-floor(3)) + 2*1 = 4.
        assert_eq!(cropped.nxstart, 2);
        assert_eq!(cropped.nystart, 2);
        assert_eq!(cropped.nzstart, 2);
        assert_eq!([cropped.nx, cropped.ny, cropped.nz], [4, 4, 4]);

        // mx/cell/origin unchanged so the sub-block renders in place.
        assert_eq!([cropped.mx, cropped.my, cropped.mz], [10, 10, 10]);
        let sub_start = cropped.grid_to_cartesian_f32(0.0, 0.0, 0.0);
        let src_start = src.grid_to_cartesian(2, 2, 2);
        for k in 0..3 {
            assert!((sub_start[k] - src_start[k]).abs() < 1e-4);
        }

        // Data copied from the absolute source index at the sub-block start.
        assert_eq!(cropped.data[[0, 0, 0]], src.data[[2, 2, 2]]);
        assert_eq!(cropped.data[[1, 2, 3]], src.data[[3, 4, 5]]);
    }

    #[test]
    fn crop_wraps_periodically_for_negative_start() {
        let src = synthetic_full_cell();
        // A point near the cell origin drives a negative padded start, which
        // must wrap to the far side of the periodic map.
        let cropped = src.crop_to_points(&[[0.0f32, 0.0, 0.0]], 2);
        assert_eq!(cropped.nxstart, -2);
        // Local (1,1,1) sits at absolute index -1 -> source index
        // pos_mod(-1,10)=9, and 1.73 A from the atom so the mask keeps it.
        assert_eq!(cropped.data[[1, 1, 1]], src.data[[9, 9, 9]]);
        // The (2,2,2) voxel is the atom itself (absolute 0).
        assert_eq!(cropped.data[[2, 2, 2]], src.data[[0, 0, 0]]);
    }

    /// 12x12x12 full-cell map whose value encodes the absolute source index so
    /// the periodic (wrapped) fetch is verifiable: `data = x*10000+y*100+z`.
    fn indexed_full_cell(n: usize) -> Density {
        let data = Array3::from_shape_fn((n, n, n), |(x, y, z)| {
            (x * 10_000 + y * 100 + z) as f32
        });
        let (dmin, dmax, dmean, rms) = density_statistics(&data);
        Density {
            grid: VoxelGrid {
                nx: n,
                ny: n,
                nz: n,
                nxstart: 0,
                nystart: 0,
                nzstart: 0,
                mx: n,
                my: n,
                mz: n,
                // 1 A per voxel keeps grid index == Cartesian coordinate.
                cell_dims: [n as f32, n as f32, n as f32],
                cell_angles: [90.0, 90.0, 90.0],
                origin: [0.0, 0.0, 0.0],
                data,
            },
            dmin,
            dmax,
            dmean,
            rms,
            space_group: 1,
        }
    }

    #[test]
    fn crop_bounds_does_not_clamp_when_model_exceeds_cell() {
        // A run of atoms along z spanning frac -0.08 .. 1.17 (span 15 cells on
        // a 12-cell axis): the model is longer than the c edge. The old clamp
        // collapsed z to mz=12 (a full periodic slab); the window must instead
        // reach past the cell to enclose every atom.
        let g = &indexed_full_cell(12).grid;
        let pts: Vec<[f32; 3]> = [-1.0f32, 3.0, 7.0, 11.0, 14.0]
            .iter()
            .map(|&z| [6.0, 6.0, z])
            .collect();
        let (start, ext) = crop_bounds(g, &pts, 2);

        // z: lo=-1, hi=14, pad 2 -> start=-3, ext=(14-(-1))+4=19. Unclamped and
        // strictly greater than mz=12 (the old code returned 12).
        assert_eq!(start[2], -3);
        assert_eq!(ext[2], 19);
        assert!(
            ext[2] > g.mz,
            "z window must exceed one cell, got {}",
            ext[2]
        );
        // x,y are a single point at index ~6: tight window, well under the
        // cell (index rounds to 5 or 6 in f32; the point is it is not a slab).
        assert!(ext[0] < g.mx, "x window should stay tight, got {}", ext[0]);
        assert!(ext[1] < g.my, "y window should stay tight, got {}", ext[1]);
    }

    #[test]
    fn crop_renders_out_of_cell_atom_in_place() {
        // Atom at z=14 on a 12-cell axis is at frac 1.17, i.e. one cell out.
        // Its density lives at source z index pos_mod(14,12)=2; the crop must
        // fetch it wrapped yet render it at Cartesian z=14 (over the atom).
        let src = indexed_full_cell(12);
        let atom = [6.0f32, 6.0, 14.0];
        let cropped = src.crop_to_points(&[atom], 2);

        // Locate the voxel rendering the atom's Cartesian position.
        let gi = cropped.cartesian_to_grid(atom); // local fractional index
        let (li, lj, lk) = (
            gi[0].round() as usize,
            gi[1].round() as usize,
            gi[2].round() as usize,
        );
        // Round-trips to the atom Cartesian, confirming render-in-place.
        let back = cropped.grid_to_cartesian(li, lj, lk);
        for c in 0..3 {
            assert!((back[c] - atom[c]).abs() < 1e-3);
        }
        // Value is the wrapped source density: absolute (6,6,14) -> (6,6,2).
        assert_eq!(cropped.data[[li, lj, lk]], src.data[[6, 6, 2]]);
        assert!(cropped.data[[li, lj, lk]] > 0.0);
    }

    #[test]
    fn crop_mask_zeroes_voxels_far_from_atoms() {
        // Two compact atoms; the block corner is far from both and must be
        // masked to zero, while the atom-centered voxels survive.
        let src = indexed_full_cell(20);
        // 1 A voxels, mask radius 3 A: keep within 3 cells of an atom.
        let a = [8.0f32, 8.0, 8.0];
        let b = [10.0f32, 10.0, 10.0];
        let cropped = src.crop_to_points(&[a, b], 4);

        // start = floor(8)-4 = 4, ext = (10-8)+8 = 10 per axis. The corner
        // voxel (0,0,0) is absolute (4,4,4): sqrt(48)=6.9 A from atom a and
        // farther from b, so it is zeroed.
        assert_eq!([cropped.nx, cropped.ny, cropped.nz], [10, 10, 10]);
        assert_eq!(cropped.data[[0, 0, 0]], 0.0);
        // Atom a sits at local (4,4,4); its voxel keeps the source density.
        assert_eq!(cropped.data[[4, 4, 4]], src.data[[8, 8, 8]]);
        assert!(cropped.data[[4, 4, 4]] > 0.0);
        // Atom b sits at local (6,6,6).
        assert_eq!(cropped.data[[6, 6, 6]], src.data[[10, 10, 10]]);
    }
}
