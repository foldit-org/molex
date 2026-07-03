//! Mass-weighted whole-assembly geometry: center of mass and radius of
//! gyration.

use glam::Vec3;

use crate::assembly::Assembly;
use crate::entity::molecule::protein::dihedral_deg;

/// Signed dihedral angle about the `p1->p2` bond for the four points
/// `p0, p1, p2, p3`, in degrees in the range (-180, 180].
///
/// Uses the IUPAC-sign convention; the same kernel backbone φ/ψ and sidechain
/// χ angles use.
#[must_use]
pub fn dihedral(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> f32 {
    dihedral_deg(p0, p1, p2, p3)
}

/// Mass-weighted center of mass and total mass of every atom in `assembly`,
/// gathered through the zero-copy `positions()`/`elements()` accessors.
///
/// Returns `(Vec3::ZERO, 0.0)` for an empty assembly.
fn com_and_mass(assembly: &Assembly) -> (Vec3, f32) {
    let mut weighted = Vec3::ZERO;
    let mut total = 0.0_f32;
    for entity in assembly.entities() {
        for (pos, element) in entity.positions().iter().zip(entity.elements()) {
            let m = element.mass();
            weighted += *pos * m;
            total += m;
        }
    }
    if total == 0.0 {
        return (Vec3::ZERO, 0.0);
    }
    (weighted / total, total)
}

impl Assembly {
    /// Mass-weighted center of mass (Angstroms) over every atom of every
    /// entity, using standard atomic weights ([`crate::Element::mass`]).
    ///
    /// An empty assembly returns the origin.
    #[must_use]
    pub fn center_of_mass(&self) -> Vec3 {
        com_and_mass(self).0
    }

    /// Mass-weighted radius of gyration (Angstroms): the RMS atom distance
    /// from the center of mass, `sqrt(sum(m_i * |r_i - com|^2) / sum(m_i))`.
    ///
    /// An empty assembly returns `0.0`.
    #[must_use]
    pub fn radius_of_gyration(&self) -> f32 {
        let (com, total) = com_and_mass(self);
        if total == 0.0 {
            return 0.0;
        }
        let mut weighted_sq = 0.0_f32;
        for entity in self.entities() {
            for (pos, element) in
                entity.positions().iter().zip(entity.elements())
            {
                weighted_sq = element
                    .mass()
                    .mul_add((*pos - com).length_squared(), weighted_sq);
            }
        }
        (weighted_sq / total).sqrt()
    }

    /// Per-atom provenance over every atom of every entity, in the same flat
    /// order [`Self::center_of_mass`] and the atom-count sum walk: `true` for
    /// a parsed atom, `false` for one fabricated by completion. The mask
    /// length equals the assembly's total atom count.
    #[must_use]
    pub fn observed_mask(&self) -> Vec<bool> {
        let mut mask = Vec::new();
        for entity in self.entities() {
            mask.extend_from_slice(&entity.columns().observed);
        }
        mask
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use glam::Vec3;

    use crate::assembly::Assembly;

    #[test]
    fn empty_assembly_is_origin_and_zero() {
        let asm = Assembly::new(Vec::new());
        assert_eq!(asm.center_of_mass(), Vec3::ZERO);
        assert_eq!(asm.radius_of_gyration(), 0.0);
    }

    #[test]
    fn dihedral_matches_kernel() {
        // Pure pass-through to the protein torsion kernel.
        let p0 = Vec3::new(0.0, 1.0, 0.0);
        let p1 = Vec3::ZERO;
        let p2 = Vec3::new(1.0, 0.0, 0.0);
        let p3 = p2 + Vec3::new(0.0, 0.0, 1.0);
        let d = super::dihedral(p0, p1, p2, p3);
        assert!((d.abs() - 90.0).abs() < 1e-3, "got {d}");
    }
}
