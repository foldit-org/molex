//! Molecule-type vocabulary maps and the columnar de-vocab layer shared by the
//! Python interchange.
//!
//! Holds the `MoleculeType` <-> AtomWorks `chain_type`/`mol_type` mappings and
//! the [`columns`] de-vocab builders that turn a native [`AtomTable`] into the
//! vocab-bearing per-atom columns [`PyAtomTable`] exposes to Python. No runtime
//! dependency on AtomWorks or Biotite: the maps are static lookups consumed
//! entirely in-process.
//!
//! [`AtomTable`]: crate::adapters::table::AtomTable
//! [`PyAtomTable`]: crate::python::PyAtomTable

pub mod columns;

use crate::entity::molecule::MoleculeType;

// Molecule type <-> AtomWorks chain type mapping

/// AtomWorks `ChainType` enum values (from `atomworks.enums.ChainType`).
///
/// These are the integer codes AtomWorks uses to classify PN units:
///   0=CyclicPseudoPeptide, 1=OtherPolymer, 2=PeptideNucleicAcid,
///   3=DNA, 4=DNA_RNA_HYBRID, 5=POLYPEPTIDE_D, 6=POLYPEPTIDE_L, 7=RNA,
///   8=NON_POLYMER, 9=WATER, 10=BRANCHED, 11=MACROLIDE
///
/// We map to these from `MoleculeType` when building annotations.
fn molecule_type_to_chain_type_id(mt: MoleculeType) -> u8 {
    match mt {
        // POLYPEPTIDE_L (default; D-peptides need explicit flag)
        MoleculeType::Protein => 6,
        MoleculeType::DNA => 3,
        MoleculeType::RNA => 7,
        MoleculeType::Ligand
        | MoleculeType::Ion
        | MoleculeType::Lipid
        | MoleculeType::Cofactor => 8, // NON_POLYMER
        MoleculeType::Water | MoleculeType::Solvent => 9,
    }
}

pub(crate) fn chain_type_id_to_molecule_type(ct: u8) -> MoleculeType {
    match ct {
        3 | 4 => MoleculeType::DNA,     // DNA, DNA_RNA_HYBRID
        5 | 6 => MoleculeType::Protein, // POLYPEPTIDE_D, POLYPEPTIDE_L
        7 => MoleculeType::RNA,
        9 => MoleculeType::Water,
        // NON_POLYMER (refined later by residue name), BRANCHED, conservative
        // default
        _ => MoleculeType::Ligand,
    }
}

/// AtomWorks `mol_type` string annotation values.
fn molecule_type_to_mol_type_str(mt: MoleculeType) -> &'static str {
    match mt {
        MoleculeType::Protein => "protein",
        MoleculeType::DNA => "dna",
        MoleculeType::RNA => "rna",
        MoleculeType::Ligand | MoleculeType::Cofactor | MoleculeType::Lipid => {
            "ligand"
        }
        MoleculeType::Ion => "ion",
        MoleculeType::Water | MoleculeType::Solvent => "water",
    }
}

pub(crate) fn mol_type_str_to_molecule_type(s: &str) -> MoleculeType {
    match s.to_lowercase().as_str() {
        "protein" | "polypeptide_l" | "polypeptide_d" => MoleculeType::Protein,
        "dna" => MoleculeType::DNA,
        "rna" => MoleculeType::RNA,
        "water" => MoleculeType::Water,
        "ion" => MoleculeType::Ion,
        // "ligand", "non_polymer", "branched", and anything else
        _ => MoleculeType::Ligand,
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_molecule_type_roundtrip() {
        // Verify that our mapping is at least self-consistent for the common
        // types
        let types = vec![
            MoleculeType::Protein,
            MoleculeType::DNA,
            MoleculeType::RNA,
            MoleculeType::Ligand,
            MoleculeType::Water,
        ];

        for mt in types {
            let ct = molecule_type_to_chain_type_id(mt);
            let back = chain_type_id_to_molecule_type(ct);
            assert_eq!(
                mt, back,
                "Round-trip failed for {mt:?} -> ct={ct} -> {back:?}",
            );
        }
    }

    #[test]
    fn test_mol_type_str_roundtrip() {
        let types = vec![
            (MoleculeType::Protein, "protein"),
            (MoleculeType::DNA, "dna"),
            (MoleculeType::RNA, "rna"),
            (MoleculeType::Ligand, "ligand"),
            (MoleculeType::Water, "water"),
        ];

        for (mt, expected_str) in types {
            let s = molecule_type_to_mol_type_str(mt);
            assert_eq!(s, expected_str);
            let back = mol_type_str_to_molecule_type(s);
            assert_eq!(mt, back);
        }
    }
}
