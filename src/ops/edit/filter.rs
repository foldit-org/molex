//! Whole-assembly sanitation filters that project to a fresh `Assembly`.
//!
//! Both methods are ML-input preparation: a model that should never see
//! crystallographic waters or explicit hydrogens gets a clean view without
//! mutating the host's source of truth. Neither fabricates atoms.

use std::sync::Arc;

use crate::assembly::Assembly;
use crate::element::Element;
use crate::entity::molecule::atom::{Atom, AtomColumns};
use crate::entity::molecule::bulk::BulkEntity;
use crate::entity::molecule::nucleic_acid::NAEntity;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::small_molecule::SmallMoleculeEntity;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};

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
    /// Each entity is rebuilt from its non-hydrogen (`element != Element::H`)
    /// atoms through the pure, non-completing constructors, so no atom is
    /// fabricated: a residue stays exactly as heavy as it was parsed.
    /// Polymer entities re-index their residues over the stripped atom list;
    /// an entity that already carries no hydrogen keeps its `Arc`. Does not
    /// mutate `self`. For ML inputs that consume heavy atoms only.
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
/// Polymer entities re-gather each residue's heavy atoms into a contiguous
/// atom list and hand the re-ranged residues to the pure `new` constructor
/// (`Completion::Raw`: skips completion, so no atom is invented). Non-polymer
/// entities filter their flat atom list and rebuild directly.
fn strip_hydrogens(entity: &MoleculeEntity) -> MoleculeEntity {
    match entity {
        MoleculeEntity::Protein(p) => {
            let (atoms, residues) =
                heavy_polymer_atoms(&p.columns, &p.residues);
            MoleculeEntity::Protein(ProteinEntity::new(
                p.id,
                atoms,
                residues,
                p.pdb_chain_id.clone(),
            ))
        }
        MoleculeEntity::NucleicAcid(n) => {
            let (atoms, residues) =
                heavy_polymer_atoms(&n.columns, &n.residues);
            MoleculeEntity::NucleicAcid(NAEntity::new(
                n.id,
                n.na_type,
                atoms,
                residues,
                n.pdb_chain_id.clone(),
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

/// Re-gather a polymer's heavy atoms residue by residue, emitting a fresh
/// atom list and residues whose `atom_range` indexes the stripped list.
///
/// Only atoms referenced by some residue are carried, matching the entity's
/// own model where atoms of dropped residues are left unreferenced. The
/// re-ranged residues let the pure `new` constructor re-partition without
/// fabrication.
fn heavy_polymer_atoms(
    columns: &AtomColumns,
    residues: &[Residue],
) -> (Vec<Atom>, Vec<Residue>) {
    let mut atoms: Vec<Atom> = Vec::with_capacity(columns.len());
    let mut out: Vec<Residue> = Vec::with_capacity(residues.len());
    for residue in residues {
        let start = atoms.len();
        for idx in residue.atom_range.clone() {
            if columns.element[idx] != Element::H {
                atoms.push(columns.gather(idx));
            }
        }
        out.push(Residue {
            atom_range: start..atoms.len(),
            ..residue.clone()
        });
    }
    (atoms, out)
}
