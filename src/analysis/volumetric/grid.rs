//! Shared signed-distance-field primitives over a uniform voxel grid.
//!
//! These primitives are consumed by every surface/cavity flavor built on
//! top of an atom-rasterized distance field:
//!
//! - [`voxelize_sas`]: rasterize a binary SAS solid from atom positions
//! - [`edt_1d`] / [`edt_3d`]: Felzenszwalb-Huttenlocher Euclidean distance
//!   transform
//! - [`binary_to_sdf`]: binary mask -> signed distance field
//! - [`detect_cavity_mask`]: flood-fill from grid boundary to identify
//!   non-solid voxels that are NOT reachable from the exterior (i.e. internal
//!   cavities)
//!
//! Everything here is pure, thread-safe, and has no rendering or GPU
//! concepts. Downstream consumers (cavity detection, SES, Gaussian
//! surfaces) build on top of these by wrapping them in specialized
//! pipelines.

use glam::Vec3;

use super::GridSpec;

/// Voxelize the SAS: voxel is `true` if within any atom's (vdW + probe).
#[must_use]
pub(crate) fn voxelize_sas(
    positions: &[Vec3],
    radii: &[f32],
    probe: f32,
    spec: &GridSpec,
) -> Vec<bool> {
    let mut solid = vec![false; spec.voxel_count()];
    for (i, &pos) in positions.iter().enumerate() {
        splat_sas_atom(&mut solid, spec, pos, radii[i] + probe);
    }
    solid
}

/// Mark voxels within radius `r` of `pos` as solid.
///
/// World->voxel index casts are clamped to `[0, dim - 1]` before the
/// `as usize` conversion, and voxel indices are bounded by the grid
/// dimensions (which fit f32 mantissa precision); the cast lints below
/// flag arithmetic that is correct by construction in this layer.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "world->voxel casts are clamped before truncation and voxel \
              indices are bounded by grid dims (<< 2^2^4, fits f32 mantissa)"
)]
fn splat_sas_atom(solid: &mut [bool], spec: &GridSpec, pos: Vec3, r: f32) {
    let [nx, ny, nz] = spec.dims;
    let origin = spec.origin;
    let spacing = spec.spacing;
    let r2 = r * r;

    let gx0 = ((pos.x - r - origin[0]) / spacing[0]).floor().max(0.0) as usize;
    let gy0 = ((pos.y - r - origin[1]) / spacing[1]).floor().max(0.0) as usize;
    let gz0 = ((pos.z - r - origin[2]) / spacing[2]).floor().max(0.0) as usize;
    let gx1 =
        (((pos.x + r - origin[0]) / spacing[0]).ceil() as usize).min(nx - 1);
    let gy1 =
        (((pos.y + r - origin[1]) / spacing[1]).ceil() as usize).min(ny - 1);
    let gz1 =
        (((pos.z + r - origin[2]) / spacing[2]).ceil() as usize).min(nz - 1);

    for ix in gx0..=gx1 {
        let dx = (ix as f32).mul_add(spacing[0], origin[0]) - pos.x;
        for iy in gy0..=gy1 {
            let dy = (iy as f32).mul_add(spacing[1], origin[1]) - pos.y;
            let dxy2 = dx.mul_add(dx, dy * dy);
            if dxy2 > r2 {
                continue;
            }
            for iz in gz0..=gz1 {
                let dz = (iz as f32).mul_add(spacing[2], origin[2]) - pos.z;
                if dz.mul_add(dz, dxy2) <= r2 {
                    solid[spec.lin(ix, iy, iz)] = true;
                }
            }
        }
    }
}

/// Felzenszwalb & Huttenlocher 1D EDT on squared distances.
///
/// `f[i]` holds the squared distance value at position `i` (with positions
/// scaled by `spacing`). On return, `f[i]` holds the minimum over all `q`
/// of `f[q] + (spacing * (i - q))^2`.
///
/// Single-character names match the variable names in the published
/// algorithm (Felzenszwalb & Huttenlocher 2012, "Distance Transforms of
/// Sampled Functions"): `v` = parabola locations, `z` = boundaries
/// between parabolas, `d` = output distances, `k` = parabola index,
/// `q` = query position, `f` = input.
#[allow(
    clippy::while_float,
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    reason = "names match the Felzenszwalb-Huttenlocher EDT paper; loop \
              counters use f32 boundaries by construction; `q as f32` and `vk \
              as f32` are bounded by grid dim (fits f32 mantissa)"
)]
pub(crate) fn edt_1d(f: &mut [f32], spacing: f32) {
    let n = f.len();
    if n <= 1 {
        return;
    }

    let sp2 = spacing * spacing;
    // Parabola envelope
    let mut v = vec![0usize; n]; // locations of parabolas
    let mut z = vec![0.0f32; n + 1]; // boundaries between parabolas
    let mut d = vec![0.0f32; n]; // output

    let mut k = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        loop {
            let vk = v[k];
            let diff = q as f32 - vk as f32;
            let sum = q as f32 + vk as f32;
            // (f[q] - f[vk] + sp2 * diff * sum) / (2 * sp2 * diff)
            let s =
                (sp2 * diff).mul_add(sum, f[q] - f[vk]) / (2.0 * sp2 * diff);
            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
            if k == 0 {
                v[0] = q;
                z[1] = f32::INFINITY;
                break;
            }
            k -= 1;
        }
    }

    k = 0;
    for (q, d_q) in d.iter_mut().enumerate().take(n) {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let diff = q as f32 - v[k] as f32;
        // sp2 * diff * diff + f[v[k]]
        *d_q = (sp2 * diff).mul_add(diff, f[v[k]]);
    }

    f.copy_from_slice(&d);
}

/// 3D EDT on a binary solid. Returns Euclidean distance from each inside
/// voxel to the nearest outside voxel (outside voxels get 0).
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "(nx + ny + nz) bounded by grid memory, fits f32 mantissa"
)]
pub(crate) fn edt_3d(
    solid: &[bool],
    dims: [usize; 3],
    spacing: &[f32; 3],
) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let inf = (nx + ny + nz) as f32 * (spacing[0] + spacing[1] + spacing[2]);
    let inf2 = inf * inf;

    // Initialize: inside = INF^2, outside = 0 (squared distances)
    let mut dt: Vec<f32> =
        solid.iter().map(|&s| if s { inf2 } else { 0.0 }).collect();

    // Pass 1: along X for each (y, z) line
    let mut buf = vec![0.0f32; nx.max(ny).max(nz)];
    for iy in 0..ny {
        for iz in 0..nz {
            for ix in 0..nx {
                buf[ix] = dt[ix * ny * nz + iy * nz + iz];
            }
            edt_1d(&mut buf[..nx], spacing[0]);
            for ix in 0..nx {
                dt[ix * ny * nz + iy * nz + iz] = buf[ix];
            }
        }
    }

    // Pass 2: along Y for each (x, z) line
    for ix in 0..nx {
        for iz in 0..nz {
            for iy in 0..ny {
                buf[iy] = dt[ix * ny * nz + iy * nz + iz];
            }
            edt_1d(&mut buf[..ny], spacing[1]);
            for iy in 0..ny {
                dt[ix * ny * nz + iy * nz + iz] = buf[iy];
            }
        }
    }

    // Pass 3: along Z for each (x, y) line
    for ix in 0..nx {
        for iy in 0..ny {
            let base = ix * ny * nz + iy * nz;
            buf[..nz].copy_from_slice(&dt[base..base + nz]);
            edt_1d(&mut buf[..nz], spacing[2]);
            dt[base..base + nz].copy_from_slice(&buf[..nz]);
        }
    }

    // sqrt to get actual Euclidean distances
    for v in &mut dt {
        *v = v.sqrt();
    }

    dt
}

/// Flood-fill from grid-face voxels through non-solid voxels and return
/// a cavity mask.
///
/// The returned mask is `true` at non-solid voxels that are NOT
/// reachable from the exterior, i.e. internal cavities. The SES
/// pipeline uses this to fill cavities back into the solid before
/// extracting the outer envelope; the cavity pipeline uses it as the
/// input to connected-component labeling + per-cavity mesh extraction.
#[must_use]
pub(crate) fn detect_cavity_mask(
    solid: &[bool],
    dims: [usize; 3],
) -> Vec<bool> {
    let [nx, ny, nz] = dims;
    let total = nx * ny * nz;

    // `exterior[i] = true` means this voxel is reachable from the grid
    // boundary through non-solid voxels.
    let mut exterior = vec![false; total];
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();

    seed_face_voxels(solid, dims, &mut exterior, &mut stack);
    flood_exterior(solid, dims, &mut exterior, &mut stack);

    // Cavity mask: non-solid AND non-exterior
    let mut cavity = vec![false; total];
    for i in 0..total {
        cavity[i] = !solid[i] && !exterior[i];
    }
    cavity
}

/// Mark every grid-face voxel that is non-solid as exterior, seeding
/// the flood-fill stack with those voxels.
fn seed_face_voxels(
    solid: &[bool],
    dims: [usize; 3],
    exterior: &mut [bool],
    stack: &mut Vec<(usize, usize, usize)>,
) {
    let [nx, ny, nz] = dims;
    let idx = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;

    let mut seed = |ix: usize, iy: usize, iz: usize| {
        let i = idx(ix, iy, iz);
        if !solid[i] && !exterior[i] {
            exterior[i] = true;
            stack.push((ix, iy, iz));
        }
    };

    // +/-Z faces
    for ix in 0..nx {
        for iy in 0..ny {
            for &iz in &[0, nz - 1] {
                seed(ix, iy, iz);
            }
        }
    }
    // +/-Y faces (excluding the +/-Z edges already seeded)
    for ix in 0..nx {
        for &iy in &[0, ny - 1] {
            for iz in 1..nz - 1 {
                seed(ix, iy, iz);
            }
        }
    }
    // +/-X faces (excluding the edges already seeded)
    for &ix in &[0, nx - 1] {
        for iy in 1..ny - 1 {
            for iz in 1..nz - 1 {
                seed(ix, iy, iz);
            }
        }
    }
}

/// 6-connected flood fill from `stack`, marking reached non-solid
/// voxels in `exterior`.
fn flood_exterior(
    solid: &[bool],
    dims: [usize; 3],
    exterior: &mut [bool],
    stack: &mut Vec<(usize, usize, usize)>,
) {
    let [nx, ny, nz] = dims;
    let idx = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;

    while let Some((x, y, z)) = stack.pop() {
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
            let i = idx(nx2, ny2, nz2);
            if !solid[i] && !exterior[i] {
                exterior[i] = true;
                stack.push((nx2, ny2, nz2));
            }
        }
    }
}

/// 6-connected neighbor offsets for flood fill.
pub(super) const NEIGHBOR_OFFSETS: [(isize, isize, isize); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Convert a binary solid to a signed distance field.
///
/// Runs EDT from both sides of the boundary:
/// - Outside voxels: +distance to nearest inside
/// - Inside voxels: -distance to nearest outside
#[must_use]
pub fn binary_to_sdf(
    solid: &[bool],
    dims: [usize; 3],
    spacing: &[f32; 3],
) -> Vec<f32> {
    let total = solid.len();

    // EDT of inside (distance from outside to nearest inside)
    let inverted: Vec<bool> = solid.iter().map(|&s| !s).collect();
    let dist_outside = edt_3d(&inverted, dims, spacing);

    // EDT of outside (distance from inside to nearest outside)
    let dist_inside = edt_3d(solid, dims, spacing);

    // SDF: negative inside, positive outside
    let mut sdf = vec![0.0f32; total];
    for i in 0..total {
        sdf[i] = if solid[i] {
            -dist_inside[i]
        } else {
            dist_outside[i]
        };
    }

    sdf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edt_1d_basic() {
        // Single inside voxel at position 2 in a line of 5
        let inf2 = 1e10f32;
        let mut f = [0.0, 0.0, inf2, 0.0, 0.0];
        edt_1d(&mut f, 1.0);
        // Position 2 should have distance 1.0 squared = 1.0
        assert!((f[2] - 1.0).abs() < 1e-6, "got {}", f[2]);
    }

    #[test]
    fn detect_cavity_mask_simple_cavity() {
        // 5x5x5 grid. Make an outer shell of solid with a single hollow
        // interior voxel at the center. The center voxel should be
        // detected as a cavity.
        let dims = [5usize; 3];
        let total = 125;
        let mut solid = vec![false; total];
        let idx = |x: usize, y: usize, z: usize| x * 25 + y * 5 + z;

        // Fill a 3x3x3 solid shell (x,y,z in 1..=3), hollow center at (2,2,2).
        let cells = (1..=3).flat_map(|ix| {
            (1..=3).flat_map(move |iy| (1..=3).map(move |iz| (ix, iy, iz)))
        });
        for (ix, iy, iz) in cells {
            if (ix, iy, iz) == (2, 2, 2) {
                continue;
            }
            solid[idx(ix, iy, iz)] = true;
        }

        let cavity = detect_cavity_mask(&solid, dims);

        // Center should be a cavity
        assert!(cavity[idx(2, 2, 2)], "center voxel should be a cavity");
        // Boundary voxels should not be cavities
        assert!(!cavity[idx(0, 0, 0)], "boundary voxel should not be cavity");
        assert!(
            !cavity[idx(4, 4, 4)],
            "opposite boundary voxel should not be cavity"
        );
        // Solid voxels should not be cavities
        assert!(!cavity[idx(1, 1, 1)], "solid voxel should not be cavity");
    }

    #[test]
    fn detect_cavity_mask_no_cavity() {
        // A solid blob with no interior holes should produce no cavity voxels.
        let dims = [5usize; 3];
        let mut solid = vec![false; 125];
        let idx = |x: usize, y: usize, z: usize| x * 25 + y * 5 + z;
        solid[idx(2, 2, 2)] = true;

        let cavity = detect_cavity_mask(&solid, dims);
        assert!(cavity.iter().all(|&c| !c), "no cavities expected");
    }
}
