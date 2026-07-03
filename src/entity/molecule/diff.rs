//! Structural diff between two views of the same entity.
//!
//! [`MoleculeEntity::diff`] is the inverse of
//! [`Assembly::apply_edits`](crate::Assembly::apply_edits): given a prior
//! and a new view of one entity, it produces the minimal `AssemblyEdit`
//! list that turns one into the other. This is the single
//! "compute the delta between two states" primitive every subsystem shares
//! (the host's plugin broadcast, viso's animation sequencer) instead of
//! each hand-rolling its own comparison.
//!
//! Round-trip contract: for a representable change,
//! `apply_edits(prior.diff(new)?)` reproduces `new`'s positions, residue
//! identities, and variants. Atom occupancy / b-factor are intentionally
//! ignored (they don't ride the edit vocabulary), matching the steady-state
//! delta path's policy that lifecycle mutations only touch positions,
//! identities, and variants.

use thiserror::Error;

use super::atom::AtomColumns;
use super::polymer::Residue;
use super::{EntityId, MoleculeEntity};
use crate::ops::edit::AssemblyEdit;

/// Why two entities can't be expressed as an [`AssemblyEdit`] delta.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EntityDiffError {
    /// The two entities are different molecule varieties (e.g. protein vs
    /// ligand); there is no incremental edit between them.
    #[error("cannot diff incompatible entity varieties")]
    IncompatibleVariety,
    /// A polymer's residue count changed (a residue insert/delete). The
    /// edit vocabulary has no per-residue insert/delete, so callers fall
    /// back to a full snapshot.
    #[error("residue count changed ({prior} -> {new}); requires full rebuild")]
    ResidueCountChanged {
        /// Residue count in the prior entity.
        prior: usize,
        /// Residue count in the new entity.
        new: usize,
    },
    /// A non-polymer's atom count changed; with no residue addressing this
    /// can't be expressed incrementally, so callers fall back to a full
    /// snapshot.
    #[error("atom count changed ({prior} -> {new}); requires full rebuild")]
    AtomCountChanged {
        /// Atom count in the prior entity.
        prior: usize,
        /// Atom count in the new entity.
        new: usize,
    },
}

impl MoleculeEntity {
    /// The minimal [`AssemblyEdit`] list transforming `self` (prior) into
    /// `new`, scoped to `new`'s [`EntityId`]. The inverse of
    /// [`Assembly::apply_edits`](crate::Assembly::apply_edits).
    ///
    /// `self` and `new` are expected to be two views of the same entity
    /// (same id and molecule variety). Atom occupancy / b-factor changes
    /// are not reported.
    ///
    /// # Errors
    ///
    /// - [`EntityDiffError::IncompatibleVariety`] if the two entities are
    ///   different molecule varieties.
    /// - [`EntityDiffError::ResidueCountChanged`] for a residue insert/delete
    ///   (not representable as an edit).
    /// - [`EntityDiffError::AtomCountChanged`] for a non-polymer atom-count
    ///   change.
    pub fn diff(
        &self,
        new: &MoleculeEntity,
    ) -> Result<Vec<AssemblyEdit>, EntityDiffError> {
        if std::mem::discriminant(self) != std::mem::discriminant(new) {
            return Err(EntityDiffError::IncompatibleVariety);
        }
        let entity = new.id();
        match (self.residues(), new.residues()) {
            (Some(a_res), Some(b_res)) => {
                if a_res.len() != b_res.len() {
                    return Err(EntityDiffError::ResidueCountChanged {
                        prior: a_res.len(),
                        new: b_res.len(),
                    });
                }
                Ok(diff_polymer(entity, self, new))
            }
            // Non-polymer: only whole-entity coordinates can change.
            (None, None) => {
                if self.atom_count() != new.atom_count() {
                    return Err(EntityDiffError::AtomCountChanged {
                        prior: self.atom_count(),
                        new: new.atom_count(),
                    });
                }
                let mut edits = Vec::new();
                if self.positions() != new.positions() {
                    edits.push(AssemblyEdit::SetEntityCoords {
                        entity,
                        coords: new.positions().to_vec(),
                    });
                }
                Ok(edits)
            }
            // The discriminant check above guarantees both polymer or both
            // non-polymer; this arm is unreachable in practice.
            _ => Err(EntityDiffError::IncompatibleVariety),
        }
    }
}

/// Per-residue classification of what changed between prior and new.
struct ResidueChange {
    /// Identity changed: name, atom count, or any atom's identity
    /// (element / name / formal charge) differs. Captured wholesale by a
    /// [`AssemblyEdit::MutateResidue`].
    mutated: bool,
    /// Variant list differs (and the residue is not otherwise mutated).
    variants_changed: bool,
    /// Only atom positions moved (residue otherwise identical).
    moved: bool,
}

fn classify_residue(
    a: &Residue,
    b: &Residue,
    a_atoms: &AtomColumns,
    b_atoms: &AtomColumns,
) -> ResidueChange {
    let mutated = a.name != b.name
        || a.atom_range.len() != b.atom_range.len()
        || a.atom_range
            .clone()
            .map(|i| a_atoms.atom_ref(i))
            .zip(b.atom_range.clone().map(|i| b_atoms.atom_ref(i)))
            .any(|(x, y)| {
                x.element != y.element
                    || x.name != y.name
                    || x.formal_charge != y.formal_charge
            });
    if mutated {
        return ResidueChange {
            mutated: true,
            variants_changed: false,
            moved: false,
        };
    }
    let variants_changed = a.variants != b.variants;
    let moved = a
        .atom_range
        .clone()
        .map(|i| a_atoms.atom_ref(i))
        .zip(b.atom_range.clone().map(|i| b_atoms.atom_ref(i)))
        .any(|(x, y)| x.position != y.position);
    ResidueChange {
        mutated: false,
        variants_changed,
        moved,
    }
}

fn diff_polymer(
    entity: EntityId,
    prior: &MoleculeEntity,
    new: &MoleculeEntity,
) -> Vec<AssemblyEdit> {
    // The caller has already verified both are polymers with equal residue
    // counts.
    let (Some(a_res), Some(b_res)) = (prior.residues(), new.residues()) else {
        return Vec::new();
    };
    let a_atoms = prior.columns();
    let b_atoms = new.columns();
    let changes: Vec<ResidueChange> = a_res
        .iter()
        .zip(b_res)
        .map(|(a, b)| classify_residue(a, b, a_atoms, b_atoms))
        .collect();

    let mut edits = Vec::new();
    if changes.iter().all(|c| !c.mutated) {
        // No identity changes: per-residue variant edits, plus one
        // whole-entity coord edit (byte-identical to the old coord-only
        // delta path).
        for (i, c) in changes.iter().enumerate() {
            if c.variants_changed {
                edits.push(AssemblyEdit::SetVariants {
                    entity,
                    residue_idx: i,
                    variants: b_res[i].variants.clone(),
                });
            }
        }
        if changes.iter().any(|c| c.moved) {
            edits.push(AssemblyEdit::SetEntityCoords {
                entity,
                coords: new.positions().to_vec(),
            });
        }
        return edits;
    }

    // Some residues changed identity: emit per-residue edits (a whole-entity
    // coord edit can't be used once atom counts shift mid-list). Edits are
    // emitted in ascending residue index; each `MutateResidue` shifts only
    // the atom ranges of later residues, which subsequent edits re-read by
    // index, so the order round-trips.
    for (i, c) in changes.iter().enumerate() {
        let rb = b_res[i].atom_range.clone();
        if c.mutated {
            edits.push(AssemblyEdit::MutateResidue {
                entity,
                residue_idx: i,
                new_name: b_res[i].name,
                new_atoms: rb.map(|j| b_atoms.gather(j)).collect(),
                new_variants: b_res[i].variants.clone(),
            });
        } else {
            if c.variants_changed {
                edits.push(AssemblyEdit::SetVariants {
                    entity,
                    residue_idx: i,
                    variants: b_res[i].variants.clone(),
                });
            }
            if c.moved {
                edits.push(AssemblyEdit::SetResidueCoords {
                    entity,
                    residue_idx: i,
                    coords: b_atoms.position[rb].to_vec(),
                });
            }
        }
    }
    edits
}

#[cfg(test)]
#[path = "diff_tests.rs"]
mod tests;
