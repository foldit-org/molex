//! Backbone-independent sidechain rotamer data and connectivity helpers.
//!
//! Modal sidechain-torsion (χ) values per residue type, drawn from the
//! Penultimate Rotamer Library (Lovell, Word, Richardson & Richardson,
//! "The Penultimate Rotamer Library", Proteins 40:389-408, 2000). Each
//! residue carries an ordered list of its common rotamers (most populated
//! first) so a placer can try them in descending modal probability.
//!
//! Symmetric terminal torsions are given once in their canonical half:
//! Asp χ2, Glu χ3, and Phe/Tyr χ2 carry a 180-degree degeneracy
//! (carboxylate / aromatic-ring two-fold), so the geometric twin (value
//! ± 180) is equivalent and is not enumerated. Amide and imidazole flips
//! are NOT degenerate (they change hydrogen-bonding), so Asn χ2, Gln χ3,
//! and His χ2 list the flipped orientation as its own rotamer.
//!
//! The angles are canonical staggered modal values (g+ ≈ 62, t ≈ 180,
//! g- ≈ -65) refined toward the library's per-residue modes; they seat a
//! chemically sane, refinable starting sidechain, not a scored optimum.

use super::amino_acids::AminoAcid;
use super::atom_name::AtomName;

/// Common rotamers for `aa`, each inner slice holding χ1..χN in degrees,
/// ordered most-populated first. Returns an empty slice for residues with
/// no rotatable placement search: Ala and Gly (no χ) and Pro (its ring
/// closure is fixed, so the template pucker is used as-is rather than
/// driven by [`crate::entity::molecule::protein`]'s χ rotation).
pub(crate) const fn modal_rotamers(aa: AminoAcid) -> &'static [&'static [f32]] {
    match aa {
        AminoAcid::Ala | AminoAcid::Gly | AminoAcid::Pro => &[],
        AminoAcid::Ser => SER_ROT,
        AminoAcid::Cys => CYS_ROT,
        AminoAcid::Thr => THR_ROT,
        AminoAcid::Val => VAL_ROT,
        AminoAcid::Leu => LEU_ROT,
        AminoAcid::Ile => ILE_ROT,
        AminoAcid::Asp => ASP_ROT,
        AminoAcid::Asn => ASN_ROT,
        AminoAcid::His => HIS_ROT,
        AminoAcid::Phe => PHE_ROT,
        AminoAcid::Tyr => TYR_ROT,
        AminoAcid::Trp => TRP_ROT,
        AminoAcid::Met => MET_ROT,
        AminoAcid::Glu => GLU_ROT,
        AminoAcid::Gln => GLN_ROT,
        AminoAcid::Lys => LYS_ROT,
        AminoAcid::Arg => ARG_ROT,
    }
}

/// Sidechain atom names that rotate when the bond `near`-`far` is turned:
/// every atom reachable from `far` in the residue's heavy-atom bond graph
/// without crossing back through `near`. `near` and `far` themselves are
/// excluded (both lie on the rotation axis).
///
/// The graph is [`AminoAcid::bonds`], which omits the proline CD-N ring
/// closure, so every residue's sidechain is a tree here and the walk
/// terminates without revisiting the axis.
pub(crate) fn distal_atoms(
    aa: AminoAcid,
    near: AtomName,
    far: AtomName,
) -> Vec<AtomName> {
    let bonds = aa.bonds();
    let mut visited = vec![near, far];
    let mut frontier = vec![far];
    let mut out = Vec::new();
    while let Some(cur) = frontier.pop() {
        for &(a, b) in bonds {
            let next = if a == cur {
                Some(b)
            } else if b == cur {
                Some(a)
            } else {
                None
            };
            if let Some(n) = next {
                if !visited.contains(&n) {
                    visited.push(n);
                    out.push(n);
                    frontier.push(n);
                }
            }
        }
    }
    out
}

// Penultimate-library common rotamers, most-populated first. See the module
// docs for the symmetric/flip convention.

const SER_ROT: &[&[f32]] = &[&[-65.0], &[62.0], &[-177.0]];
const CYS_ROT: &[&[f32]] = &[&[-65.0], &[-177.0], &[62.0]];
const THR_ROT: &[&[f32]] = &[&[62.0], &[-60.0], &[-175.0]];
const VAL_ROT: &[&[f32]] = &[&[175.0], &[-60.0], &[63.0]];

const LEU_ROT: &[&[f32]] = &[&[-65.0, 175.0], &[-172.0, 62.0], &[-85.0, -58.0]];
const ILE_ROT: &[&[f32]] = &[
    &[-65.0, 170.0],
    &[-65.0, -60.0],
    &[62.0, 170.0],
    &[-178.0, 170.0],
];

// Asp χ2 / Glu χ3 are carboxylate-symmetric: one value, twin omitted.
const ASP_ROT: &[&[f32]] = &[&[-65.0, -15.0], &[-170.0, -10.0], &[62.0, -10.0]];
const GLU_ROT: &[&[f32]] = &[
    &[-65.0, 180.0, -10.0],
    &[-65.0, -65.0, -30.0],
    &[-178.0, 180.0, -10.0],
    &[62.0, 180.0, -10.0],
];

// Asn χ2 / Gln χ3 amide flips and His χ2 ring flips are distinct rotamers.
const ASN_ROT: &[&[f32]] = &[
    &[-65.0, -20.0],
    &[-65.0, 120.0],
    &[-170.0, -15.0],
    &[62.0, -10.0],
];
const GLN_ROT: &[&[f32]] = &[
    &[-65.0, 180.0, -25.0],
    &[-65.0, -65.0, -40.0],
    &[-178.0, 180.0, -25.0],
    &[62.0, 180.0, -25.0],
];
const HIS_ROT: &[&[f32]] = &[
    &[-65.0, -70.0],
    &[-65.0, 165.0],
    &[-178.0, -80.0],
    &[62.0, 80.0],
];

// Phe / Tyr χ2 are ring-symmetric: one value, twin omitted. Trp is not.
const PHE_ROT: &[&[f32]] = &[&[-65.0, -85.0], &[-178.0, 80.0], &[62.0, 90.0]];
const TYR_ROT: &[&[f32]] = &[&[-65.0, -85.0], &[-178.0, 80.0], &[62.0, 90.0]];
const TRP_ROT: &[&[f32]] = &[
    &[-65.0, -90.0],
    &[-65.0, 95.0],
    &[-178.0, -105.0],
    &[62.0, 90.0],
];

const MET_ROT: &[&[f32]] = &[
    &[-65.0, -65.0, -70.0],
    &[-65.0, 180.0, 75.0],
    &[-65.0, 180.0, 180.0],
    &[-178.0, 180.0, -75.0],
];
const LYS_ROT: &[&[f32]] = &[
    &[-65.0, 180.0, 180.0, 180.0],
    &[-178.0, 180.0, 180.0, 180.0],
    &[-65.0, 180.0, 62.0, 180.0],
    &[-178.0, 180.0, -65.0, 180.0],
];
const ARG_ROT: &[&[f32]] = &[
    &[-65.0, 180.0, 180.0, 85.0],
    &[-178.0, 180.0, 180.0, 85.0],
    &[-65.0, 180.0, 180.0, -85.0],
    &[-178.0, 180.0, -65.0, 180.0],
];

#[cfg(test)]
mod tests {
    use super::*;

    fn n(s: &[u8]) -> AtomName {
        AtomName::from_bytes(s)
    }

    #[test]
    fn rotamer_width_matches_chi_count() {
        // Every rotamer for a residue must carry exactly one value per χ.
        for &aa in &[
            AminoAcid::Ser,
            AminoAcid::Val,
            AminoAcid::Leu,
            AminoAcid::Phe,
            AminoAcid::Met,
            AminoAcid::Gln,
            AminoAcid::Lys,
            AminoAcid::Arg,
        ] {
            let width = aa.chi_atoms().len();
            for rot in modal_rotamers(aa) {
                assert_eq!(rot.len(), width, "{aa:?} rotamer width");
            }
        }
    }

    #[test]
    fn no_chi_search_for_ala_gly_pro() {
        assert!(modal_rotamers(AminoAcid::Ala).is_empty());
        assert!(modal_rotamers(AminoAcid::Gly).is_empty());
        assert!(modal_rotamers(AminoAcid::Pro).is_empty());
    }

    #[test]
    fn distal_set_for_leucine_chi1_is_branch_beyond_cb() {
        // χ1 turns the CA-CB bond; everything past CB rotates.
        let moving = distal_atoms(AminoAcid::Leu, n(b"CA"), n(b"CB"));
        for atom in [n(b"CG"), n(b"CD1"), n(b"CD2")] {
            assert!(moving.contains(&atom), "{atom} should rotate");
        }
        assert!(!moving.contains(&n(b"CB")), "axis atom CB must not move");
        assert!(!moving.contains(&n(b"CA")), "axis atom CA must not move");
    }

    #[test]
    fn distal_set_for_leucine_chi2_excludes_cg() {
        // χ2 turns the CB-CG bond; only the terminal carbons rotate.
        let moving = distal_atoms(AminoAcid::Leu, n(b"CB"), n(b"CG"));
        assert!(moving.contains(&n(b"CD1")));
        assert!(moving.contains(&n(b"CD2")));
        assert!(!moving.contains(&n(b"CG")));
        assert!(!moving.contains(&n(b"CB")));
    }

    #[test]
    fn distal_set_for_phenylalanine_chi2_is_whole_ring() {
        // The aromatic ring rotates rigidly about CB-CG.
        let moving = distal_atoms(AminoAcid::Phe, n(b"CB"), n(b"CG"));
        for atom in [n(b"CD1"), n(b"CD2"), n(b"CE1"), n(b"CE2"), n(b"CZ")] {
            assert!(moving.contains(&atom), "{atom} should rotate with ring");
        }
    }
}
