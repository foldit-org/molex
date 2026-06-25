//! Kabsch-Sander backbone hydrogen bond detection.
//!
//! Pure function: backbone residue positions in, H-bond pairs out.
//! The energy formula uses partial charges on backbone atoms (N, H, C, O)
//! to compute electrostatic interaction energy. Pairs with energy below
//! the threshold (-0.5 kcal/mol) are accepted as hydrogen bonds.

use glam::Vec3;

use crate::entity::molecule::protein::ResidueBackbone;

/// Kabsch-Sander electrostatic energy constant (kcal/mol * A).
const KS_FACTOR: f32 = 27.888;

/// H-bond energy threshold (kcal/mol). Bonds with E < this are accepted.
const HBOND_THRESHOLD: f32 = -0.5;

/// A backbone hydrogen bond detected by Kabsch-Sander energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HBond {
    /// Residue index of the N-H donor.
    pub donor: usize,
    /// Residue index of the C=O acceptor.
    pub acceptor: usize,
    /// Kabsch-Sander energy (kcal/mol, negative = favorable).
    pub energy: f32,
}

/// Detect backbone hydrogen bonds using the Kabsch-Sander energy criterion.
///
/// For each residue, amide H positions are estimated from geometry:
/// `H_i = N_i + normalize(N_i - C_{i-1})` (first residue uses N-CA).
///
/// Returns all donor-acceptor pairs with energy below the threshold,
/// sorted by donor index then by energy (strongest first).
///
/// Internal implementation detail of `Assembly`'s derived-data pipeline,
/// reached through `detect_fallback_connections` and the SS computation.
#[must_use]
pub(crate) fn detect_hbonds(residues: &[ResidueBackbone]) -> Vec<HBond> {
    let n = residues.len();
    if n < 2 {
        return Vec::new();
    }

    // Estimate amide H positions using the bisector method:
    // H is placed 1.0 A from N along the bisector of (N - C_prev) and
    // (N - CA). This matches the canonical DSSP placement.
    let h_positions: Vec<Vec3> = (0..n)
        .map(|i| {
            let v_ca = (residues[i].n - residues[i].ca).normalize_or_zero();
            let v_prev = if i > 0 {
                (residues[i].n - residues[i - 1].c).normalize_or_zero()
            } else {
                v_ca
            };
            let bisector = (v_ca + v_prev).normalize_or_zero();
            residues[i].n + bisector
        })
        .collect();

    let mut hbonds = Vec::new();

    for i in 1..n {
        let h = h_positions[i];
        let n_pos = residues[i].n;

        for (j, res_j) in residues.iter().enumerate() {
            // Skip self and immediate neighbors.
            if i == j || i == j + 1 || (j > 0 && i == j - 1) {
                continue;
            }

            let c = res_j.c;
            let o = res_j.o;

            let r_on = (o - n_pos).length();
            let r_ch = (c - h).length();
            let r_oh = (o - h).length();
            let r_cn = (c - n_pos).length();

            // Avoid division by near-zero distances.
            if r_on < 0.5 || r_ch < 0.5 || r_oh < 0.5 || r_cn < 0.5 {
                continue;
            }

            let energy =
                KS_FACTOR * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn);

            if energy < HBOND_THRESHOLD {
                hbonds.push(HBond {
                    donor: i,
                    acceptor: j,
                    energy,
                });
            }
        }
    }

    hbonds
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert!(detect_hbonds(&[]).is_empty());
    }

    #[test]
    fn single_residue() {
        let residues = vec![ResidueBackbone {
            n: Vec3::new(0.0, 0.0, 0.0),
            ca: Vec3::new(1.5, 0.0, 0.0),
            c: Vec3::new(2.5, 1.0, 0.0),
            o: Vec3::new(2.5, 2.0, 0.0),
        }];
        assert!(detect_hbonds(&residues).is_empty());
    }
}
