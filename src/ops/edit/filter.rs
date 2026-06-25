//! Whole-assembly sanitation filters that project to a fresh `Assembly`.
//!
//! Both methods are ML-input preparation: a model that should never see
//! crystallographic waters or explicit hydrogens gets a clean view without
//! mutating the host's source of truth. Neither fabricates atoms.

use std::sync::Arc;

use crate::assembly::Assembly;
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::bulk::BulkEntity;
use crate::entity::molecule::nucleic_acid::NAEntity;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::small_molecule::SmallMoleculeEntity;
use crate::entity::molecule::{Completion, MoleculeEntity, MoleculeType};

impl Assembly {
    /// Project to a fresh `Assembly` with bulk water and solvent entities
    /// dropped, every other entity passed through unchanged.
    ///
    /// Whole-entity drop keyed on [`MoleculeEntity::molecule_type`]: an
    /// entity classified [`MoleculeType::Water`] or [`MoleculeType::Solvent`]
    /// is removed; survivors keep their `Arc`, so this is O(entities) of
    /// refcount bumps with no atom rebuild. For ML inputs that must not see
    /// crystallographic solvent. Does not mutate `self`.
    #[must_use]
    pub fn without_waters(&self) -> Assembly {
        let kept: Vec<Arc<MoleculeEntity>> = self
            .entities()
            .iter()
            .filter(|e| {
                !matches!(
                    e.molecule_type(),
                    MoleculeType::Water | MoleculeType::Solvent
                )
            })
            .map(Arc::clone)
            .collect();
        Assembly::from_arcs(kept)
    }

    /// Project to a fresh `Assembly` with every hydrogen atom dropped,
    /// keeping all heavy atoms exactly as observed.
    ///
    /// Each entity is rebuilt with its hydrogens dropped and nothing
    /// fabricated, so a residue stays exactly as heavy as it was parsed.
    /// Polymer entities go through the non-fabricating heavy-only strip
    /// (`Completion::HeavyRaw`), which canonicalizes the heavy atoms without
    /// running the rigid-fit completion pass; an entity that already carries
    /// no hydrogen keeps its `Arc`. Does not mutate `self`. For ML inputs
    /// that consume heavy atoms only.
    #[must_use]
    pub fn heavy_only(&self) -> Assembly {
        let entities: Vec<Arc<MoleculeEntity>> = self
            .entities()
            .iter()
            .map(|e| {
                if e.elements().iter().all(|el| *el != Element::H) {
                    Arc::clone(e)
                } else {
                    Arc::new(strip_hydrogens(e))
                }
            })
            .collect();
        Assembly::from_arcs(entities)
    }
}

/// Rebuild `entity` without its hydrogen atoms, fabricating nothing.
///
/// Polymer entities hand their full atom set and residues to the completing
/// constructor at [`Completion::HeavyRaw`], which skips the rigid-fit pass
/// (so no atom is invented) but drops parsed hydrogens in the canonicalize
/// reorder. Non-polymer entities filter their flat atom list and rebuild
/// directly.
fn strip_hydrogens(entity: &MoleculeEntity) -> MoleculeEntity {
    match entity {
        MoleculeEntity::Protein(p) => {
            MoleculeEntity::Protein(ProteinEntity::new_normalized(
                p.id,
                p.columns.to_atoms(),
                p.residues.clone(),
                p.pdb_chain_id.clone(),
                Completion::HeavyRaw,
            ))
        }
        MoleculeEntity::NucleicAcid(n) => {
            MoleculeEntity::NucleicAcid(NAEntity::new_normalized(
                n.id,
                n.na_type,
                n.columns.to_atoms(),
                n.residues.clone(),
                n.pdb_chain_id.clone(),
                Completion::HeavyRaw,
            ))
        }
        MoleculeEntity::SmallMolecule(s) => {
            MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
                s.id,
                s.mol_type,
                heavy_atoms(entity),
                s.residue_name,
            ))
        }
        MoleculeEntity::Bulk(b) => MoleculeEntity::Bulk(BulkEntity::new(
            b.id,
            b.mol_type,
            heavy_atoms(entity),
            b.residue_name,
            b.molecule_count,
        )),
    }
}

/// Gather an entity's heavy atoms (`element != Element::H`) into an owned
/// `Vec<Atom>`, in storage order.
fn heavy_atoms(entity: &MoleculeEntity) -> Vec<Atom> {
    entity
        .columns()
        .to_atoms()
        .into_iter()
        .filter(|a| a.element != Element::H)
        .collect()
}
