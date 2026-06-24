//! Internal cavity detection via SES erosion + connected components.
//!
//! Pipeline:
//! 1. Voxelize the SAS solid from atom positions
//! 2. EDT on the solid interior -> erode by `probe` -> SES solid
//! 3. Flood-fill from grid exterior -> cavity mask (non-solid voxels
//!    unreachable from the outside)
//! 4. Connected-component label the cavity mask -> per-cavity sub-grids
//!
//! Each returned [`DetectedCavity`] carries a padded binary mask on its
//! own sub-grid, ready to be handed to a mesher (marching cubes,
//! splatting, volumetric raymarch, whatever) without further analysis.

use glam::Vec3;

use super::grid::{detect_cavity_mask, edt_3d, voxelize_sas, NEIGHBOR_OFFSETS};
use super::GridSpec;

/// Default solvent probe radius (water) in Angstroms.
const DEFAULT_PROBE_RADIUS: f32 = 1.4;

/// Axis-aligned bounding box in voxel coordinates (inclusive on both ends).
#[derive(Debug, Clone, Copy)]
pub struct VoxelBbox {
    /// Minimum corner `(ix, iy, iz)`.
    pub min: [usize; 3],
    /// Maximum corner `(ix, iy, iz)` (inclusive).
    pub max: [usize; 3],
}

/// A single detected cavity represented as a padded binary mask on its
/// own sub-grid.
///
/// The sub-grid is a local slice of the full voxelization, padded by 1
/// voxel in every direction so a downstream mesher can close surfaces
/// against the boundary. Only voxels belonging to **this** cavity are
/// marked `true` in `sub_mask`; other cavities and the exterior are
/// `false`.
#[derive(Debug, Clone)]
pub struct DetectedCavity {
    /// Binary mask indexed `x-major -> y -> z` with dimensions `sub_dims`.
    pub sub_mask: Vec<bool>,
    /// Sub-grid dimensions in voxels.
    pub sub_dims: [usize; 3],
    /// World-space position of sub-grid voxel `(0, 0, 0)` (Angstroms).
    pub sub_origin: [f32; 3],
    /// Angstroms per voxel along each axis (same as the parent grid).
    pub spacing: [f32; 3],
    /// World-space centroid of this cavity's voxel bounding box
    /// (Angstroms). Useful for effects anchored at the cavity's center.
    pub centroid: [f32; 3],
}

/// Detect internal cavities in a set of atom positions.
///
/// - `positions`: atom world-space positions (Angstroms)
/// - `radii`: per-atom van der Waals radii (Angstroms)
/// - `probe_radius`: solvent probe radius; defaults to 1.4 A
/// - `resolution`: grid spacing in Angstroms (lower = finer; 0.5-1.0 typical)
///
/// Returns one [`DetectedCavity`] per connected component, in label
/// order (no guaranteed ordering otherwise).
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "1-based cavity id `bboxes.len() + 1` cannot exceed u32 range \
              for any realistic voxel grid"
)]
pub fn detect_cavities(
    positions: &[Vec3],
    radii: &[f32],
    probe_radius: Option<f32>,
    resolution: f32,
) -> Vec<DetectedCavity> {
    if positions.is_empty() {
        return Vec::new();
    }

    let probe = probe_radius.unwrap_or(DEFAULT_PROBE_RADIUS);
    // Load-bearing: the padding must be large enough that every grid-
    // face voxel is outside the SAS envelope, or the exterior flood
    // fill in `detect_cavity_mask` will miss atoms near the grid edge.
    let padding = probe + 2.0;

    // Compute grid bounds from atom positions
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

    // Step 1: Binary SAS solid
    let sas_solid = voxelize_sas(positions, radii, probe, &spec);

    // Step 2: SES carve; voxels where EDT < probe become outside
    let interior_edt = edt_3d(&sas_solid, spec.dims, &spec.spacing);
    let total = spec.voxel_count();
    let mut ses_solid = vec![false; total];
    for i in 0..total {
        ses_solid[i] = sas_solid[i] && interior_edt[i] >= probe;
    }

    // Step 3: Extract cavity mask (non-solid voxels not reachable from
    // the grid exterior)
    let cavity_mask = detect_cavity_mask(&ses_solid, spec.dims);

    // Step 4: Label connected components -> per-cavity IDs + bboxes
    let (labels, bboxes) = label_connected_components(&cavity_mask, spec.dims);
    if bboxes.is_empty() {
        return Vec::new();
    }

    // Step 5: Extract padded sub-grid mask for each cavity
    bboxes
        .iter()
        .enumerate()
        .map(|(cavity_idx, bbox)| {
            let id = cavity_idx as u32 + 1;
            extract_sub_mask(id, bbox, &labels, &spec)
        })
        .collect()
}

/// Build a padded per-cavity sub-grid mask.
#[allow(
    clippy::cast_precision_loss,
    reason = "voxel index -> f32 cast: indices bounded by grid dims, fit f32 \
              mantissa exactly"
)]
fn extract_sub_mask(
    id: u32,
    bbox: &VoxelBbox,
    labels: &[u32],
    spec: &GridSpec,
) -> DetectedCavity {
    let [nx, ny, nz] = spec.dims;
    let origin = spec.origin;
    let spacing = spec.spacing;

    // Pad the bbox by 1 voxel in each direction so a downstream mesher
    // has neighboring "outside" cells to close the surface against.
    let sub_min = [
        bbox.min[0].saturating_sub(1),
        bbox.min[1].saturating_sub(1),
        bbox.min[2].saturating_sub(1),
    ];
    let sub_max = [
        (bbox.max[0] + 1).min(nx - 1),
        (bbox.max[1] + 1).min(ny - 1),
        (bbox.max[2] + 1).min(nz - 1),
    ];
    let sub_dims = [
        sub_max[0] - sub_min[0] + 1,
        sub_max[1] - sub_min[1] + 1,
        sub_max[2] - sub_min[2] + 1,
    ];

    // Build per-cavity binary mask in local sub-grid coordinates. Only
    // voxels carrying this cavity's label become solid; other cavities
    // and exterior are treated as outside.
    let sub_total = sub_dims[0] * sub_dims[1] * sub_dims[2];
    let mut sub_mask = vec![false; sub_total];
    let cells = (0..sub_dims[0]).flat_map(|lx| {
        (0..sub_dims[1])
            .flat_map(move |ly| (0..sub_dims[2]).map(move |lz| (lx, ly, lz)))
    });
    for (lx, ly, lz) in cells {
        let gx = sub_min[0] + lx;
        let gy = sub_min[1] + ly;
        let gz = sub_min[2] + lz;
        if labels[spec.lin(gx, gy, gz)] == id {
            sub_mask[lx * sub_dims[1] * sub_dims[2] + ly * sub_dims[2] + lz] =
                true;
        }
    }

    let sub_origin = [
        (sub_min[0] as f32).mul_add(spacing[0], origin[0]),
        (sub_min[1] as f32).mul_add(spacing[1], origin[1]),
        (sub_min[2] as f32).mul_add(spacing[2], origin[2]),
    ];

    // World-space centroid of the cavity's voxel bbox.
    let centroid = [
        ((bbox.min[0] + bbox.max[0]) as f32 * 0.5)
            .mul_add(spacing[0], origin[0]),
        ((bbox.min[1] + bbox.max[1]) as f32 * 0.5)
            .mul_add(spacing[1], origin[1]),
        ((bbox.min[2] + bbox.max[2]) as f32 * 0.5)
            .mul_add(spacing[2], origin[2]),
    ];

    DetectedCavity {
        sub_mask,
        sub_dims,
        sub_origin,
        spacing,
        centroid,
    }
}

/// 6-connected flood-fill labeling over a binary cavity mask.
///
/// Returns `(labels, bboxes)` where `labels[i]` is 0 for non-cavity
/// voxels and a positive 1-based cavity ID for cavity voxels.
/// `bboxes[id-1]` is the inclusive voxel-space bounding box of cavity
/// `id`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "1-based cavity id `bboxes.len() + 1` cannot exceed u32 range \
              for any realistic voxel grid"
)]
fn label_connected_components(
    cavity_mask: &[bool],
    dims: [usize; 3],
) -> (Vec<u32>, Vec<VoxelBbox>) {
    let [nx, ny, nz] = dims;
    let total = nx * ny * nz;
    let idx = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;

    let mut labels = vec![0u32; total];
    let mut bboxes: Vec<VoxelBbox> = Vec::new();

    let cells = (0..nx).flat_map(|ix| {
        (0..ny).flat_map(move |iy| (0..nz).map(move |iz| (ix, iy, iz)))
    });
    for (ix, iy, iz) in cells {
        let i = idx(ix, iy, iz);
        if !cavity_mask[i] || labels[i] != 0 {
            continue;
        }
        let label = bboxes.len() as u32 + 1;
        let bbox = label_component(
            (ix, iy, iz),
            label,
            cavity_mask,
            &mut labels,
            dims,
        );
        bboxes.push(bbox);
    }

    (labels, bboxes)
}

/// Label the connected component containing `start`, marking every
/// connected cavity voxel with `label` and returning the bounding box.
fn label_component(
    start: (usize, usize, usize),
    label: u32,
    cavity_mask: &[bool],
    labels: &mut [u32],
    dims: [usize; 3],
) -> VoxelBbox {
    let [nx, ny, nz] = dims;
    let idx = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;

    let (sx, sy, sz) = start;
    let mut bbox = VoxelBbox {
        min: [sx, sy, sz],
        max: [sx, sy, sz],
    };
    labels[idx(sx, sy, sz)] = label;

    let mut stack = vec![start];
    while let Some((x, y, z)) = stack.pop() {
        bbox.min[0] = bbox.min[0].min(x);
        bbox.min[1] = bbox.min[1].min(y);
        bbox.min[2] = bbox.min[2].min(z);
        bbox.max[0] = bbox.max[0].max(x);
        bbox.max[1] = bbox.max[1].max(y);
        bbox.max[2] = bbox.max[2].max(z);

        for (dx, dy, dz) in NEIGHBOR_OFFSETS {
            let Some(nx2) = x.checked_add_signed(dx) else {
                continue;
            };
            let Some(ny2) = y.checked_add_signed(dy) else {
                continue;
            };
            let Some(nz2) = z.checked_add_signed(dz) else {
                continue;
            };
            if nx2 >= nx || ny2 >= ny || nz2 >= nz {
                continue;
            }
            let j = idx(nx2, ny2, nz2);
            if cavity_mask[j] && labels[j] == 0 {
                labels[j] = label;
                stack.push((nx2, ny2, nz2));
            }
        }
    }

    bbox
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn lin(dims: [usize; 3], x: usize, y: usize, z: usize) -> usize {
        x * dims[1] * dims[2] + y * dims[2] + z
    }

    fn mask_from<const N: usize>(
        dims: [usize; 3],
        cells: [(usize, usize, usize); N],
    ) -> Vec<bool> {
        let mut m = vec![false; dims[0] * dims[1] * dims[2]];
        for (x, y, z) in cells {
            m[lin(dims, x, y, z)] = true;
        }
        m
    }

    #[test]
    fn label_empty_mask() {
        let dims = [4usize; 3];
        let mask = vec![false; 64];
        let (labels, bboxes) = label_connected_components(&mask, dims);
        assert!(labels.iter().all(|&l| l == 0));
        assert!(bboxes.is_empty());
    }

    #[test]
    fn label_single_cavity() {
        let dims = [4usize; 3];
        let mask = mask_from(dims, [(1, 1, 1), (1, 1, 2), (1, 2, 1)]);
        let (labels, bboxes) = label_connected_components(&mask, dims);
        assert_eq!(bboxes.len(), 1);
        assert_eq!(labels[lin(dims, 1, 1, 1)], 1);
        assert_eq!(labels[lin(dims, 1, 1, 2)], 1);
        assert_eq!(labels[lin(dims, 1, 2, 1)], 1);
        assert_eq!(bboxes[0].min, [1, 1, 1]);
        assert_eq!(bboxes[0].max, [1, 2, 2]);
    }

    #[test]
    fn label_two_separated_cavities() {
        let dims = [5usize; 3];
        let mask = mask_from(dims, [(1, 1, 1), (3, 3, 3)]);
        let (labels, bboxes) = label_connected_components(&mask, dims);
        assert_eq!(bboxes.len(), 2);
        assert_eq!(labels[lin(dims, 1, 1, 1)], 1);
        assert_eq!(labels[lin(dims, 3, 3, 3)], 2);
    }

    #[test]
    fn label_adjacent_cavities_merge() {
        let dims = [4usize; 3];
        let mask = mask_from(dims, [(1, 1, 1), (2, 1, 1)]);
        let (_labels, bboxes) = label_connected_components(&mask, dims);
        assert_eq!(bboxes.len(), 1);
        assert_eq!(bboxes[0].min, [1, 1, 1]);
        assert_eq!(bboxes[0].max, [2, 1, 1]);
    }

    #[test]
    fn detect_cavities_empty_atoms() {
        let result = detect_cavities(&[], &[], None, 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn detect_cavities_single_atom_has_none() {
        // A lone atom is a solid blob with no interior voids.
        let result = detect_cavities(&[Vec3::ZERO], &[1.5], Some(1.4), 0.5);
        assert!(result.is_empty());
    }
}
