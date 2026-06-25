//! AtomWorks adapter for RC-Foundry / ModelForge models.
//!
//! Bidirectional conversion between molex's `Vec<MoleculeEntity>` and
//! AtomWorks-annotated Biotite `AtomArray` objects.
//!
//! This adapter:
//!
//! - Operates on **entities**, preserving molecule type, entity ID, and chain
//!   grouping through the round-trip.
//! - Populates **bonds** from distance inference.
//! - Sets AtomWorks-specific per-atom annotations (`entity_id`, `mol_type`,
//!   `pn_unit_iid`) so structures can feed directly into Foundry model
//!   pipelines (RF3, RFdiffusion3, LigandMPNN) without re-parsing.
//! - Can optionally invoke `atomworks.io.parser.parse()` on the Python side to
//!   get the full cleaning pipeline (leaving group removal, charge correction,
//!   missing atom imputation, etc.).
//!
//! # Usage from Python (via PyO3)
//!
//! ```python
//! import molex
//! from atomworks.io.parser import parse as aw_parse
//!
//! # -- molex -> AtomWorks (for model inference) --
//! atom_array = molex.entities_to_atom_array(assembly_bytes)
//! atom_array_plus = molex.entities_to_atom_array_plus(assembly_bytes)
//!
//! # -- AtomArray -> molex (after model prediction) --
//! assembly_bytes = molex.atom_array_to_entities(atom_array)
//!
//! # -- Full AtomWorks cleaning pipeline --
//! atom_array = molex.entities_to_atom_array_parsed(assembly_bytes, "3nez.cif.gz")
//! ```

mod from_array;
mod to_array;

// Re-export all public items so they remain accessible at the same paths.
pub use from_array::{
    atom_array_to_entities, atom_array_to_entity_vec, parse_file_full,
    parse_file_to_entities,
};
use pyo3::prelude::*;
pub(crate) use to_array::for_each_flat_atom;
pub use to_array::{
    assembly_bytes_to_atom_array, assembly_bytes_to_atom_array_plus,
    entities_to_atom_array, entities_to_atom_array_parsed,
    entities_to_atom_array_plus,
};
// The shared per-atom column collector and the canonical atom walk it
// drives. Used by the Biotite bridge in `to_array.rs` and by the native
// numpy read path in `python::arrays` (the walk also maps bond endpoints
// into flat space). `collect_atom_data`/`AtomData` are public so the
// egress benchmark can time the pure-Rust core without a Python
// interpreter.
pub use to_array::{collect_atom_data, AtomData};

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

fn chain_type_id_to_molecule_type(ct: u8) -> MoleculeType {
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

fn mol_type_str_to_molecule_type(s: &str) -> MoleculeType {
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

// Helpers

/// Try to get an optional annotation array from an AtomArray.
/// Returns `None` if the annotation doesn't exist.
fn get_annotation_opt<'py>(
    atom_array: &Bound<'py, PyAny>,
    name: &str,
) -> Option<Bound<'py, PyAny>> {
    atom_array.call_method1("get_annotation", (name,)).ok()
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
