//! Typed Assembly edits — the in-memory representation of incremental
//! updates exchanged between the foldit host and its plugins.
//!
//! `AssemblyEdit` covers the steady-state mutation vocabulary:
//! coordinate updates (per-entity or per-residue), residue mutations
//! (sequence design / mutate-residue), variant changes (protonation,
//! disulfide, terminus patches), and entity-level topology
//! (add/remove). Per-residue topology (insert/delete a residue inside
//! an existing entity) is intentionally not yet enumerated — bridge
//! consumers fall back to full pose rebuild for that class of change.
//!
//! `Assembly::apply_edit` and `Assembly::apply_edits` dispatch each
//! variant to the appropriate primitive. Each successful apply bumps
//! the Assembly's generation counter (via `after_mutation`). Secondary
//! structure is not recomputed here; callers opt in via
//! `Assembly::recompute_ss`.

use std::sync::Arc;

use glam::Vec3;
use thiserror::Error;

use crate::assembly::Assembly;
use crate::chemistry::variant::VariantTag;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::id::EntityId;
use crate::entity::molecule::MoleculeEntity;

/// Typed Assembly edit.
///
/// Each variant carries enough information for a receiver to apply the
/// change to its own Assembly mirror without consulting the
/// originator. Residues are addressed by index within their parent
/// entity's residue list; `EntityId` is the stable identifier across
/// edits.
#[derive(Debug, Clone)]
pub enum AssemblyEdit {
    /// Replace every atom position in the target entity. `coords` must
    /// have length equal to the entity's `atom_count`.
    SetEntityCoords {
        /// Entity whose atoms get repositioned.
        entity: EntityId,
        /// New positions, one per atom in declaration order.
        coords: Vec<Vec3>,
    },

    /// Replace atom positions for a single residue within a polymer
    /// entity. `coords` must have length equal to the residue's
    /// current atom count (the edit does not change residue shape).
    SetResidueCoords {
        /// Polymer entity that owns the target residue.
        entity: EntityId,
        /// Residue index within the entity's residue list.
        residue_idx: usize,
        /// New positions, one per atom in the residue's atom range.
        coords: Vec<Vec3>,
    },

    /// Replace a residue's identity and atoms within a polymer entity.
    /// Used for sequence-design output and single-residue mutate ops.
    /// `new_atoms` is the full atom list for the new residue (may
    /// differ in count from the prior residue). Subsequent residues in
    /// the entity have their `atom_range`s shifted by the new size
    /// delta automatically.
    MutateResidue {
        /// Polymer entity that owns the target residue.
        entity: EntityId,
        /// Residue index within the entity's residue list.
        residue_idx: usize,
        /// New 3-letter residue name.
        new_name: [u8; 3],
        /// New atom list for the residue (may differ in count from the
        /// prior residue's atoms).
        new_atoms: Vec<Atom>,
        /// New variant tags for the residue.
        new_variants: Vec<VariantTag>,
    },

    /// Replace a residue's variant list within a polymer entity.
    SetVariants {
        /// Polymer entity that owns the target residue.
        entity: EntityId,
        /// Residue index within the entity's residue list.
        residue_idx: usize,
        /// New variant tag list.
        variants: Vec<VariantTag>,
    },

    /// Append a new entity to the assembly. The contained entity's
    /// `id()` must not collide with any existing entity in the target.
    AddEntity {
        /// Entity to append.
        entity: MoleculeEntity,
    },

    /// Remove an entity from the assembly. The apply path returns
    /// `Err(UnknownEntity)` if the id is unknown so callers can
    /// distinguish "removed-as-requested" from "wasn't there to begin
    /// with".
    RemoveEntity {
        /// Entity to remove.
        entity: EntityId,
    },
}

/// Wraps an [`EditError`] with the index of the offending edit when
/// it surfaces from a bulk [`Assembly::apply_edits`] call.
#[derive(Debug, Error)]
#[error("edit at index {index} failed: {source}")]
pub struct BulkEditError {
    /// Index of the edit in the input slice that triggered the
    /// failure.
    pub index: usize,
    /// Underlying error.
    #[source]
    pub source: EditError,
}

/// Failure modes when applying an [`AssemblyEdit`].
#[derive(Debug, Error)]
pub enum EditError {
    /// The target entity id is not present in the assembly.
    #[error("unknown entity {0}")]
    UnknownEntity(EntityId),
    /// The target residue index is out of range for the entity's
    /// residue list.
    #[error("entity {entity} has no residue at index {residue_idx}")]
    UnknownResidue {
        /// Entity that was addressed.
        entity: EntityId,
        /// Residue index that was out of range.
        residue_idx: usize,
    },
    /// The supplied coord/atom count doesn't match the target's
    /// current count (only enforced for edits where mismatch is
    /// invalid — `MutateResidue` explicitly allows count changes).
    #[error(
        "count mismatch on entity {entity}: expected {expected}, got {got}"
    )]
    CountMismatch {
        /// Entity that was addressed.
        entity: EntityId,
        /// Atom or coordinate count the entity currently holds.
        expected: usize,
        /// Atom or coordinate count supplied by the edit.
        got: usize,
    },
    /// The edit targets a polymer-only operation (residue-level
    /// addressing) on a non-polymer entity (SmallMolecule, Bulk).
    #[error("entity {entity} is not a polymer; cannot apply per-residue edit")]
    NotPolymer {
        /// Entity that was addressed.
        entity: EntityId,
    },
    /// `AddEntity` was issued with an id that already exists in the
    /// target assembly.
    #[error("entity id {0} already present; AddEntity rejected")]
    DuplicateEntity(EntityId),
}

impl Assembly {
    /// Apply a single [`AssemblyEdit`] to this assembly.
    ///
    /// On success the generation counter is incremented. Secondary
    /// structure is not recomputed; `ss_types` stays as-is until a caller
    /// opts in via [`Assembly::recompute_ss`].
    ///
    /// # Errors
    ///
    /// See [`EditError`] for the failure modes.
    pub fn apply_edit(&mut self, edit: &AssemblyEdit) -> Result<(), EditError> {
        match edit {
            AssemblyEdit::SetEntityCoords { entity, coords } => {
                apply_set_entity_coords(self, *entity, coords)
            }
            AssemblyEdit::SetResidueCoords {
                entity,
                residue_idx,
                coords,
            } => apply_set_residue_coords(self, *entity, *residue_idx, coords),
            AssemblyEdit::MutateResidue {
                entity,
                residue_idx,
                new_name,
                new_atoms,
                new_variants,
            } => apply_mutate_residue(
                self,
                *entity,
                *residue_idx,
                *new_name,
                new_atoms,
                new_variants,
            ),
            AssemblyEdit::SetVariants {
                entity,
                residue_idx,
                variants,
            } => apply_set_variants(self, *entity, *residue_idx, variants),
            AssemblyEdit::AddEntity { entity } => {
                apply_add_entity(self, entity.clone())
            }
            AssemblyEdit::RemoveEntity { entity } => {
                apply_remove_entity(self, *entity)
            }
        }
    }

    /// Apply multiple edits in order. Stops at the first error and
    /// returns it; edits applied before the failing one stay applied,
    /// each having bumped the generation counter.
    ///
    /// # Errors
    ///
    /// Returns a [`BulkEditError`] carrying the index of the failing
    /// edit and the underlying [`EditError`].
    pub fn apply_edits(
        &mut self,
        edits: &[AssemblyEdit],
    ) -> Result<(), BulkEditError> {
        for (index, edit) in edits.iter().enumerate() {
            self.apply_edit(edit)
                .map_err(|source| BulkEditError { index, source })?;
        }
        Ok(())
    }
}

fn entity_index(assembly: &Assembly, id: EntityId) -> Result<usize, EditError> {
    assembly
        .entities()
        .iter()
        .position(|e| e.id() == id)
        .ok_or(EditError::UnknownEntity(id))
}

fn apply_set_entity_coords(
    assembly: &mut Assembly,
    entity: EntityId,
    coords: &[Vec3],
) -> Result<(), EditError> {
    let idx = entity_index(assembly, entity)?;
    let entities = assembly.entities_mut();
    let target = &mut entities[idx];
    if target.atom_count() != coords.len() {
        return Err(EditError::CountMismatch {
            entity,
            expected: target.atom_count(),
            got: coords.len(),
        });
    }
    let atoms = Arc::make_mut(target).atom_set_mut();
    for (atom, &pos) in atoms.iter_mut().zip(coords.iter()) {
        atom.position = pos;
    }
    assembly.after_mutation_pub();
    Ok(())
}

fn apply_set_residue_coords(
    assembly: &mut Assembly,
    entity: EntityId,
    residue_idx: usize,
    coords: &[Vec3],
) -> Result<(), EditError> {
    let idx = entity_index(assembly, entity)?;
    let entities = assembly.entities_mut();
    let target = Arc::make_mut(&mut entities[idx]);
    let Some((atoms, residues)) = target.polymer_parts_mut() else {
        return Err(EditError::NotPolymer { entity });
    };
    let residue =
        residues.get(residue_idx).ok_or(EditError::UnknownResidue {
            entity,
            residue_idx,
        })?;
    let range = residue.atom_range.clone();
    if range.len() != coords.len() {
        return Err(EditError::CountMismatch {
            entity,
            expected: range.len(),
            got: coords.len(),
        });
    }
    for (atom_idx, &pos) in range.zip(coords.iter()) {
        atoms[atom_idx].position = pos;
    }
    assembly.after_mutation_pub();
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "arguments mirror the AssemblyEdit::MutateResidue variant's \
              fields; collapsing into a struct just renames the same count \
              and breaks the symmetry with apply_edit's dispatch."
)]
fn apply_mutate_residue(
    assembly: &mut Assembly,
    entity: EntityId,
    residue_idx: usize,
    new_name: [u8; 3],
    new_atoms: &[Atom],
    new_variants: &[VariantTag],
) -> Result<(), EditError> {
    let idx = entity_index(assembly, entity)?;
    let entities = assembly.entities_mut();
    let target = Arc::make_mut(&mut entities[idx]);
    let Some((atoms, residues)) = target.polymer_parts_mut() else {
        return Err(EditError::NotPolymer { entity });
    };
    if residue_idx >= residues.len() {
        return Err(EditError::UnknownResidue {
            entity,
            residue_idx,
        });
    }

    let old_range = residues[residue_idx].atom_range.clone();
    let old_len = old_range.len();
    let new_len = new_atoms.len();

    // Splice the entity's atom list: drop `old_range`, insert
    // `new_atoms` in its place. Drop the returned iterator immediately
    // — `Vec::splice` only runs the replacement when the iterator is
    // consumed/dropped.
    drop(atoms.splice(old_range.clone(), new_atoms.iter().cloned()));

    // Reshape the target residue.
    let new_end = old_range.start + new_len;
    residues[residue_idx].name = new_name;
    residues[residue_idx].atom_range = old_range.start..new_end;
    residues[residue_idx].variants = new_variants.to_vec();

    // Shift subsequent residues' atom_range by the size delta.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        reason = "atom count fits in isize for any plausible structure"
    )]
    let delta = new_len as isize - old_len as isize;
    if delta != 0 {
        for r in &mut residues[residue_idx + 1..] {
            #[allow(
                clippy::cast_possible_wrap,
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "atom_range offsets fit in isize/usize"
            )]
            {
                let new_start = (r.atom_range.start as isize + delta) as usize;
                let new_endv = (r.atom_range.end as isize + delta) as usize;
                r.atom_range = new_start..new_endv;
            }
        }
    }

    assembly.after_mutation_pub();
    Ok(())
}

fn apply_set_variants(
    assembly: &mut Assembly,
    entity: EntityId,
    residue_idx: usize,
    variants: &[VariantTag],
) -> Result<(), EditError> {
    let idx = entity_index(assembly, entity)?;
    let entities = assembly.entities_mut();
    let target = Arc::make_mut(&mut entities[idx]);
    let Some((_, residues)) = target.polymer_parts_mut() else {
        return Err(EditError::NotPolymer { entity });
    };
    let residue =
        residues
            .get_mut(residue_idx)
            .ok_or(EditError::UnknownResidue {
                entity,
                residue_idx,
            })?;
    residue.variants = variants.to_vec();
    assembly.after_mutation_pub();
    Ok(())
}

fn apply_add_entity(
    assembly: &mut Assembly,
    entity: MoleculeEntity,
) -> Result<(), EditError> {
    let id = entity.id();
    if assembly.entities().iter().any(|e| e.id() == id) {
        return Err(EditError::DuplicateEntity(id));
    }
    assembly.add_entity(entity);
    Ok(())
}

fn apply_remove_entity(
    assembly: &mut Assembly,
    entity: EntityId,
) -> Result<(), EditError> {
    let _ = entity_index(assembly, entity)?;
    assembly.remove_entity(entity);
    Ok(())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
