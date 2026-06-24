//! Structural analysis: bond detection, secondary structure, and geometry.

/// Axis-aligned bounding box.
pub mod aabb;
/// Bond detection: covalent, hydrogen, and disulfide.
pub mod bonds;
/// Shrake-Rupley solvent-accessible surface area.
pub mod sasa;
/// Secondary structure detection and parsing.
pub mod ss;
/// Volumetric analysis: voxel grids, SDFs, cavity detection, surfaces.
pub mod volumetric;

pub use aabb::Aabb;
pub use bonds::{
    detect_disulfides, infer_bonds, BondOrder, HBond, InferredBond,
    DEFAULT_TOLERANCE,
};
pub use sasa::{assembly_sasa, shrake_rupley};
pub use volumetric::{
    binary_to_sdf, compute_gaussian_field, compute_ses_sdf, detect_cavities,
    detect_cavity_mask, edt_1d, edt_3d, voxelize_sas, DetectedCavity,
    ScalarVoxelGrid, VoxelBbox,
};

/// Q3 secondary structure classification for a single residue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SSType {
    /// Alpha helix.
    Helix,
    /// Beta sheet / strand.
    Sheet,
    /// Coil / loop (neither helix nor sheet).
    Coil,
}

impl SSType {
    /// Get the color for this SS type (RGB, 0-1 range).
    #[must_use]
    pub fn color(&self) -> [f32; 3] {
        match self {
            SSType::Helix => [0.9, 0.3, 0.5],
            SSType::Sheet => [0.95, 0.85, 0.3],
            SSType::Coil => [0.6, 0.85, 0.6],
        }
    }
}

/// Convert isolated 1-residue helix/sheet runs to Coil.
///
/// These are too short for ribbon rendering and would leave residues
/// with no backbone geometry.
#[must_use]
pub fn merge_short_segments(ss_types: &[SSType]) -> Vec<SSType> {
    let mut result = ss_types.to_vec();
    for i in 0..result.len() {
        if result[i] != SSType::Coil {
            let prev_same = i > 0 && result[i - 1] == result[i];
            let next_same = i + 1 < result.len() && result[i + 1] == result[i];
            if !prev_same && !next_same {
                result[i] = SSType::Coil;
            }
        }
    }
    result
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    // -- SSType::color --

    #[test]
    fn sstype_helix_color() {
        let c = SSType::Helix.color();
        assert_eq!(c, [0.9, 0.3, 0.5]);
    }

    #[test]
    fn sstype_sheet_color() {
        let c = SSType::Sheet.color();
        assert_eq!(c, [0.95, 0.85, 0.3]);
    }

    #[test]
    fn sstype_coil_color() {
        let c = SSType::Coil.color();
        assert_eq!(c, [0.6, 0.85, 0.6]);
    }

    #[test]
    fn sstype_all_colors_in_range() {
        for ss in &[SSType::Helix, SSType::Sheet, SSType::Coil] {
            let color = ss.color();
            for &val in &color {
                assert!(
                    (0.0..=1.0).contains(&val),
                    "{ss:?} color component {val} out of [0,1]"
                );
            }
        }
    }

    // -- merge_short_segments --

    #[test]
    fn merge_short_segments_empty() {
        let result = merge_short_segments(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn merge_short_segments_all_coil() {
        let input = vec![SSType::Coil; 5];
        let result = merge_short_segments(&input);
        assert_eq!(result, input);
    }

    #[test]
    fn merge_short_segments_isolated_helix_becomes_coil() {
        // C H C -> C C C
        let input = vec![SSType::Coil, SSType::Helix, SSType::Coil];
        let result = merge_short_segments(&input);
        assert_eq!(result, vec![SSType::Coil, SSType::Coil, SSType::Coil]);
    }

    #[test]
    fn merge_short_segments_isolated_sheet_becomes_coil() {
        // C S C -> C C C
        let input = vec![SSType::Coil, SSType::Sheet, SSType::Coil];
        let result = merge_short_segments(&input);
        assert_eq!(result, vec![SSType::Coil, SSType::Coil, SSType::Coil]);
    }

    #[test]
    fn merge_short_segments_pair_survives() {
        // H H -> H H (two adjacent helices are not isolated)
        let input = vec![SSType::Helix, SSType::Helix];
        let result = merge_short_segments(&input);
        assert_eq!(result, vec![SSType::Helix, SSType::Helix]);
    }

    #[test]
    fn merge_short_segments_long_run_survives() {
        // H H H H H
        let input = vec![SSType::Helix; 5];
        let result = merge_short_segments(&input);
        assert_eq!(result, vec![SSType::Helix; 5]);
    }

    #[test]
    fn merge_short_segments_mixed() {
        // C H H H C S C H H C
        let input = vec![
            SSType::Coil,
            SSType::Helix,
            SSType::Helix,
            SSType::Helix,
            SSType::Coil,
            SSType::Sheet, // isolated
            SSType::Coil,
            SSType::Helix,
            SSType::Helix,
            SSType::Coil,
        ];
        let result = merge_short_segments(&input);
        assert_eq!(result[1], SSType::Helix);
        assert_eq!(result[2], SSType::Helix);
        assert_eq!(result[3], SSType::Helix);
        assert_eq!(result[5], SSType::Coil); // isolated sheet -> coil
        assert_eq!(result[7], SSType::Helix);
        assert_eq!(result[8], SSType::Helix);
    }

    #[test]
    fn merge_short_segments_single_element() {
        assert_eq!(merge_short_segments(&[SSType::Helix]), vec![SSType::Coil]);
        assert_eq!(merge_short_segments(&[SSType::Coil]), vec![SSType::Coil]);
    }

    #[test]
    fn merge_short_segments_boundary_helix_at_start() {
        // H C C -> C C C (isolated at start)
        let input = vec![SSType::Helix, SSType::Coil, SSType::Coil];
        let result = merge_short_segments(&input);
        assert_eq!(result[0], SSType::Coil);
    }

    #[test]
    fn merge_short_segments_boundary_helix_at_end() {
        // C C H -> C C C (isolated at end)
        let input = vec![SSType::Coil, SSType::Coil, SSType::Helix];
        let result = merge_short_segments(&input);
        assert_eq!(result[2], SSType::Coil);
    }
}
