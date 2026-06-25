//! Kabsch alignment, SVD, and coordinate transformation utilities.

use glam::{Mat3, Vec3};

/// Compute centroid of a point set.
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "point count fits in f32")]
pub(crate) fn centroid(points: &[Vec3]) -> Vec3 {
    if points.is_empty() {
        return Vec3::ZERO;
    }
    let sum: Vec3 = points.iter().copied().sum();
    sum / points.len() as f32
}

/// Kabsch algorithm: find optimal rotation and translation to align target to
/// reference.
///
/// Returns (rotation_matrix, translation) such that: aligned =
/// rotation * target + translation
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn kabsch_alignment(
    reference: &[Vec3],
    target: &[Vec3],
) -> Option<(Mat3, Vec3)> {
    if reference.len() != target.len() || reference.len() < 3 {
        return None;
    }

    let ref_centroid = centroid(reference);
    let tgt_centroid = centroid(target);

    let ref_centered: Vec<Vec3> =
        reference.iter().map(|p| *p - ref_centroid).collect();
    let tgt_centered: Vec<Vec3> =
        target.iter().map(|p| *p - tgt_centroid).collect();

    let mut h = [[0.0f32; 3]; 3];
    for k in 0..reference.len() {
        let t = tgt_centered[k];
        let r = ref_centered[k];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = t[i].mul_add(r[j], h[i][j]);
            }
        }
    }

    let (u, _s, v) = svd_3x3(h);

    let u_mat = Mat3::from_cols(
        Vec3::new(u[0][0], u[1][0], u[2][0]),
        Vec3::new(u[0][1], u[1][1], u[2][1]),
        Vec3::new(u[0][2], u[1][2], u[2][2]),
    );
    let v_mat = Mat3::from_cols(
        Vec3::new(v[0][0], v[1][0], v[2][0]),
        Vec3::new(v[0][1], v[1][1], v[2][1]),
        Vec3::new(v[0][2], v[1][2], v[2][2]),
    );

    let mut rotation = v_mat * u_mat.transpose();

    if rotation.determinant() < 0.0 {
        let v_flipped =
            Mat3::from_cols(v_mat.col(0), v_mat.col(1), -v_mat.col(2));
        rotation = v_flipped * u_mat.transpose();
    }

    let translation = ref_centroid - rotation * tgt_centroid;

    Some((rotation, translation))
}

/// Optimal-superposition RMSD between two equal-length point sets.
///
/// Runs [`kabsch_alignment`] to find the rigid transform that best maps `a`
/// onto `b`, applies it to `a`, and returns the root-mean-square deviation
/// against `b`. The result is invariant to any rigid motion of either set.
///
/// Returns `None` when the sets differ in length or have fewer than three
/// points (the same precondition [`kabsch_alignment`] enforces), so callers
/// get a sentinel rather than a panic.
#[must_use]
pub fn rmsd(a: &[Vec3], b: &[Vec3]) -> Option<f32> {
    let (rotation, translation) = kabsch_alignment(b, a)?;

    #[allow(clippy::cast_precision_loss, reason = "point count fits in f32")]
    let n = a.len() as f32;
    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(p, q)| (rotation * *p + translation - *q).length_squared())
        .sum();
    Some((sum_sq / n).sqrt())
}

// SVD Implementation (Jacobi iteration for 3x3 matrices)

fn svd_3x3(a: [[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3], [[f32; 3]; 3]) {
    let ata = compute_ata(a);
    let (eigenvalues, v) = jacobi_eigendecomposition(ata);

    let s = [
        eigenvalues[0].max(0.0).sqrt(),
        eigenvalues[1].max(0.0).sqrt(),
        eigenvalues[2].max(0.0).sqrt(),
    ];

    let mut u = compute_u_from_av(a, &v, &s);
    orthonormalize(&mut u);

    (u, s, v)
}

fn compute_ata(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut ata = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for row in &a {
                ata[i][j] = row[i].mul_add(row[j], ata[i][j]);
            }
        }
    }
    ata
}

fn compute_u_from_av(
    a: [[f32; 3]; 3],
    v: &[[f32; 3]; 3],
    s: &[f32; 3],
) -> [[f32; 3]; 3] {
    let mut u = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            if s[j] > 1e-10 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum = a[i][k].mul_add(v[k][j], sum);
                }
                u[i][j] = sum / s[j];
            }
        }
    }
    u
}

#[allow(clippy::many_single_char_names)]
fn jacobi_eigendecomposition(
    mut a: [[f32; 3]; 3],
) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut v = [[0.0f32; 3]; 3];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    const MAX_ITER: usize = 50;
    for _ in 0..MAX_ITER {
        let Some((p, q)) = find_max_off_diagonal(&a) else {
            break;
        };
        apply_jacobi_rotation(&mut a, &mut v, p, q);
    }

    sort_eigenpairs(a, v)
}

fn find_max_off_diagonal(a: &[[f32; 3]; 3]) -> Option<(usize, usize)> {
    let mut max_val = 0.0f32;
    let mut p = 0;
    let mut q = 1;
    for (i, row) in a.iter().enumerate() {
        for (j, &val) in row.iter().enumerate().skip(i + 1) {
            if val.abs() > max_val {
                max_val = val.abs();
                p = i;
                q = j;
            }
        }
    }
    (max_val >= 1e-10).then_some((p, q))
}

#[allow(clippy::many_single_char_names)]
fn apply_jacobi_rotation(
    a: &mut [[f32; 3]; 3],
    v: &mut [[f32; 3]; 3],
    p: usize,
    q: usize,
) {
    let diff = a[q][q] - a[p][p];
    let theta = if diff.abs() < 1e-10 {
        std::f32::consts::FRAC_PI_4
    } else {
        0.5 * (2.0 * a[p][q] / diff).atan()
    };

    let c = theta.cos();
    let s = theta.sin();

    let mut new_a = *a;
    new_a[p][p] = c.mul_add(
        c * a[p][p],
        (-2.0 * s).mul_add(c * a[p][q], s * s * a[q][q]),
    );
    new_a[q][q] =
        s.mul_add(s * a[p][p], (2.0 * s).mul_add(c * a[p][q], c * c * a[q][q]));
    new_a[p][q] = 0.0;
    new_a[q][p] = 0.0;

    for i in 0..3 {
        if i != p && i != q {
            new_a[i][p] = c.mul_add(a[i][p], -(s * a[i][q]));
            new_a[p][i] = new_a[i][p];
            new_a[i][q] = s.mul_add(a[i][p], c * a[i][q]);
            new_a[q][i] = new_a[i][q];
        }
    }
    *a = new_a;

    for row in v.iter_mut() {
        let vip = row[p];
        let viq = row[q];
        row[p] = c.mul_add(vip, -(s * viq));
        row[q] = s.mul_add(vip, c * viq);
    }
}

fn sort_eigenpairs(
    a: [[f32; 3]; 3],
    v: [[f32; 3]; 3],
) -> ([f32; 3], [[f32; 3]; 3]) {
    let eigenvalues = [a[0][0], a[1][1], a[2][2]];

    let mut indices = [0usize, 1, 2];
    indices.sort_by(|&i, &j| {
        eigenvalues[j]
            .partial_cmp(&eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues = [
        eigenvalues[indices[0]],
        eigenvalues[indices[1]],
        eigenvalues[indices[2]],
    ];

    let mut sorted_v = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            sorted_v[i][j] = v[i][indices[j]];
        }
    }

    (sorted_eigenvalues, sorted_v)
}

fn orthonormalize(m: &mut [[f32; 3]; 3]) {
    let mut norm: f32 = m.iter().map(|row| row[0] * row[0]).sum();
    norm = norm.sqrt();
    if norm > 1e-10 {
        for row in m.iter_mut() {
            row[0] /= norm;
        }
    }

    let mut dot: f32 = m.iter().map(|row| row[1] * row[0]).sum();
    for row in m.iter_mut() {
        row[1] = dot.mul_add(-row[0], row[1]);
    }
    norm = m.iter().map(|row| row[1] * row[1]).sum();
    norm = norm.sqrt();
    if norm > 1e-10 {
        for row in m.iter_mut() {
            row[1] /= norm;
        }
    }

    dot = m.iter().map(|row| row[2] * row[0]).sum();
    for row in m.iter_mut() {
        row[2] = dot.mul_add(-row[0], row[2]);
    }
    dot = m.iter().map(|row| row[2] * row[1]).sum();
    for row in m.iter_mut() {
        row[2] = dot.mul_add(-row[1], row[2]);
    }
    norm = m.iter().map(|row| row[2] * row[2]).sum();
    norm = norm.sqrt();
    if norm > 1e-10 {
        for row in m.iter_mut() {
            row[2] /= norm;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_centroid() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let c = centroid(&points);
        assert!((c.x - 0.667).abs() < 0.01);
        assert!((c.y - 0.667).abs() < 0.01);
        assert!(c.z.abs() < 0.01);
    }

    #[test]
    fn test_kabsch_identity() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let (rotation, translation) =
            kabsch_alignment(&points, &points).unwrap();
        assert!((rotation.determinant() - 1.0).abs() < 0.01);
        assert!(translation.length() < 0.01);
    }

    // Rigorous correctness tests for the hand-rolled 3x3 SVD / Kabsch.

    /// Compare two matrices element-by-element to a tolerance.
    fn mat_close(a: Mat3, b: Mat3, tol: f32) -> bool {
        (0..3).all(|c| (a.col(c) - b.col(c)).length() < tol)
    }

    /// A non-symmetric, non-degenerate point cloud (no three points
    /// collinear, no special symmetry) so that the optimal alignment is
    /// uniquely determined.
    fn asymmetric_cloud() -> Vec<Vec3> {
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.3, 0.2, -0.4),
            Vec3::new(-0.7, 1.1, 0.9),
            Vec3::new(2.1, -0.6, 1.7),
            Vec3::new(0.4, 2.3, -1.2),
            Vec3::new(-1.5, -0.9, 2.0),
        ]
    }

    /// A proper rotation about an arbitrary, non-axis-aligned axis.
    fn known_rotation() -> Mat3 {
        let axis = Vec3::new(1.0, -2.0, 3.0).normalize();
        Mat3::from_axis_angle(axis, 0.97) // ~55.6 degrees
    }

    /// KNOWN-ROTATION RECOVERY: kabsch must recover the exact rotation and
    /// translation that map the target cloud onto the reference cloud.
    /// This is the test that catches a transposed or otherwise wrong R.
    #[test]
    fn test_kabsch_recovers_known_rotation_and_translation() {
        let rotation = known_rotation();
        let translation = Vec3::new(3.5, -1.2, 0.8);

        // reference = R * target + t, so kabsch(reference, target) must
        // return (R, t). Build target freely, derive reference.
        let target = asymmetric_cloud();
        let reference: Vec<Vec3> =
            target.iter().map(|p| rotation * *p + translation).collect();

        let (r_rec, t_rec) = kabsch_alignment(&reference, &target).unwrap();

        // Recovered transform equals the planted one.
        assert!(
            mat_close(r_rec, rotation, 1e-5),
            "recovered rotation differs: {r_rec:?} vs {rotation:?}"
        );
        assert!(
            (t_rec - translation).length() < 1e-4,
            "recovered translation differs: {t_rec:?} vs {translation:?}"
        );

        // And it must actually map every target point onto its reference.
        for (tgt, refp) in target.iter().zip(reference.iter()) {
            let mapped = r_rec * *tgt + t_rec;
            assert!(
                (mapped - *refp).length() < 1e-4,
                "point not aligned: {mapped:?} vs {refp:?}"
            );
        }
    }

    /// REFLECTION / DETERMINANT FIX: when the target is a mirror image of
    /// the reference, a naive SVD yields a det = -1 reflection. The
    /// determinant-correction branch must instead return a proper rotation
    /// (det ~= +1) while still minimising RMSD.
    #[test]
    fn test_kabsch_rejects_reflection() {
        let reference = asymmetric_cloud();
        // Mirror across the x=0 plane -> a pure reflection (det = -1).
        let target: Vec<Vec3> = reference
            .iter()
            .map(|p| Vec3::new(-p.x, p.y, p.z))
            .collect();

        let (rotation, translation) =
            kabsch_alignment(&reference, &target).unwrap();

        // Result is a PROPER rotation, not a reflection.
        assert!(
            (rotation.determinant() - 1.0).abs() < 1e-4,
            "expected det ~= +1, got {}",
            rotation.determinant()
        );

        // The proper-rotation fit cannot match a mirror perfectly, but it
        // must achieve the minimal-RMSD fit: residual must be finite and the
        // rotation must be orthonormal (R^T R = I).
        let should_be_identity = rotation.transpose() * rotation;
        assert!(
            mat_close(should_be_identity, Mat3::IDENTITY, 1e-4),
            "rotation not orthonormal: {should_be_identity:?}"
        );

        let mut rmsd_sq = 0.0f32;
        for (tgt, refp) in target.iter().zip(reference.iter()) {
            let mapped = rotation * *tgt + translation;
            rmsd_sq += (mapped - *refp).length_squared();
            assert!(mapped.is_finite(), "non-finite mapped point: {mapped:?}");
        }
        assert!(rmsd_sq.is_finite());
    }

    /// DEGENERATE INPUT: collinear points make H rank-deficient, which can
    /// stress the SVD's small-singular-value handling. The output rotation
    /// must contain no NaN/inf even though the alignment is under-determined.
    #[test]
    fn test_kabsch_collinear_no_nan() {
        // All points on the x-axis -> rank-1 covariance.
        let reference: Vec<Vec3> = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(|x| Vec3::new(x, 0.0, 0.0))
            .collect();
        let rotation = known_rotation();
        let translation = Vec3::new(-2.0, 4.0, 1.5);
        let target: Vec<Vec3> = reference
            .iter()
            .map(|p| rotation * *p + translation)
            .collect();

        let (r_rec, t_rec) = kabsch_alignment(&reference, &target).unwrap();

        for c in 0..3 {
            let col = r_rec.col(c);
            assert!(
                col.is_finite(),
                "rotation column {c} has NaN/inf: {col:?}"
            );
        }
        assert!(t_rec.is_finite(), "translation has NaN/inf: {t_rec:?}");
    }

    /// RMSD onto a rigidly rotated + translated copy of a point set must be
    /// ~0: the superposition undoes the planted motion exactly.
    #[test]
    fn test_rmsd_rigid_moved_self_is_zero() {
        let a = asymmetric_cloud();
        let rotation = known_rotation();
        let translation = Vec3::new(12.3, -4.5, 7.8);
        let b: Vec<Vec3> =
            a.iter().map(|p| rotation * *p + translation).collect();

        let r = rmsd(&a, &b).unwrap();
        assert!(r <= 1e-4, "rmsd of rigid-moved self should be ~0, got {r}");
    }

    /// Length mismatch and fewer-than-three points return `None`, never panic.
    #[test]
    fn test_rmsd_invalid_inputs_return_none() {
        let a = asymmetric_cloud();
        assert!(rmsd(&a, &a[..a.len() - 1]).is_none());
        assert!(rmsd(&a[..2], &a[..2]).is_none());
    }
}
