//! Kabsch alignment, SVD, and coordinate transformation utilities.

use glam::{Mat3, Vec3};

use super::extract::{centroid, extract_ca_positions};
use crate::entity::molecule::MoleculeEntity;
use crate::ops::codec::AdapterError;

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

/// Kabsch-Umeyama algorithm: find optimal rotation, translation, AND scale.
#[must_use]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::cast_precision_loss, reason = "point count fits in f32")]
pub fn kabsch_alignment_with_scale(
    reference: &[Vec3],
    target: &[Vec3],
) -> Option<(Mat3, Vec3, f32)> {
    if reference.len() != target.len() || reference.len() < 3 {
        return None;
    }

    let ref_centroid = centroid(reference);
    let tgt_centroid = centroid(target);

    let ref_centered: Vec<Vec3> =
        reference.iter().map(|p| *p - ref_centroid).collect();
    let tgt_centered: Vec<Vec3> =
        target.iter().map(|p| *p - tgt_centroid).collect();

    let _ref_var: f32 =
        ref_centered.iter().map(|p| p.length_squared()).sum::<f32>()
            / reference.len() as f32;
    let tgt_var: f32 =
        tgt_centered.iter().map(|p| p.length_squared()).sum::<f32>()
            / target.len() as f32;

    if tgt_var < 1e-10 {
        return None;
    }

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

    let (u, s, v) = svd_3x3(h);

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

    let (rotation, sign) = if (v_mat * u_mat.transpose()).determinant() < 0.0 {
        let v_flipped =
            Mat3::from_cols(v_mat.col(0), v_mat.col(1), -v_mat.col(2));
        (v_flipped * u_mat.transpose(), -1.0f32)
    } else {
        (v_mat * u_mat.transpose(), 1.0f32)
    };

    let trace_sd = sign.mul_add(s[2], s[0] + s[1]);
    let scale =
        (trace_sd / (tgt_var * reference.len() as f32)).clamp(0.1, 10.0);

    let translation = ref_centroid - scale * (rotation * tgt_centroid);

    Some((rotation, translation, scale))
}

/// Apply rotation + translation to all atoms in a set of entities.
pub fn transform_entities(
    entities: &mut [MoleculeEntity],
    rotation: Mat3,
    translation: Vec3,
) {
    for entity in entities.iter_mut() {
        for atom in entity.atom_set_mut() {
            atom.position = rotation * atom.position + translation;
        }
    }
}

/// Apply rotation + translation + scale to all atoms in a set of entities.
pub fn transform_entities_with_scale(
    entities: &mut [MoleculeEntity],
    rotation: Mat3,
    translation: Vec3,
    scale: f32,
) {
    for entity in entities.iter_mut() {
        for atom in entity.atom_set_mut() {
            atom.position = rotation * (atom.position * scale) + translation;
        }
    }
}

/// Align entities to match reference CA positions using Kabsch algorithm.
///
/// # Errors
///
/// Returns `AdapterError::InvalidFormat` if CA count differs between reference
/// and entities, or if the Kabsch alignment fails.
pub fn align_to_reference(
    entities: &mut [MoleculeEntity],
    reference_ca: &[Vec3],
) -> Result<(), AdapterError> {
    let predicted_ca = extract_ca_positions(entities);

    if predicted_ca.len() != reference_ca.len() {
        return Err(AdapterError::InvalidFormat(format!(
            "CA count mismatch: reference={}, entities={}",
            reference_ca.len(),
            predicted_ca.len()
        )));
    }

    let (rotation, translation) = kabsch_alignment(reference_ca, &predicted_ca)
        .ok_or_else(|| {
            AdapterError::InvalidFormat("Kabsch alignment failed".to_owned())
        })?;

    transform_entities(entities, rotation, translation);
    Ok(())
}

// ============================================================================
// SVD Implementation (Jacobi iteration for 3x3 matrices)
// ============================================================================

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
}
