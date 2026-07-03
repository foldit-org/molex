//! Enum-to-string renderers shared by the `Entity` object accessors:
//! `entity_kind` / `molecule_type` map their molex enums to the stable
//! Python-facing strings. A submodule of the `python` module, split out to
//! keep each file under the 800-line source cap enforced by `just
//! file-lengths`.

use crate::entity::molecule::{EntityKind, MoleculeType};

pub(super) fn entity_kind_str(kind: EntityKind) -> &'static str {
    match kind {
        EntityKind::Protein => "Protein",
        EntityKind::NucleicAcid => "NucleicAcid",
        EntityKind::SmallMolecule => "SmallMolecule",
        EntityKind::Bulk => "Bulk",
    }
}

pub(super) fn molecule_type_str(mol_type: MoleculeType) -> &'static str {
    match mol_type {
        MoleculeType::Protein => "Protein",
        MoleculeType::DNA => "DNA",
        MoleculeType::RNA => "RNA",
        MoleculeType::Ligand => "Ligand",
        MoleculeType::Ion => "Ion",
        MoleculeType::Water => "Water",
        MoleculeType::Lipid => "Lipid",
        MoleculeType::Cofactor => "Cofactor",
        MoleculeType::Solvent => "Solvent",
    }
}
