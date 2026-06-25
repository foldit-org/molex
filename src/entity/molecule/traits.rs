//! Entity and Polymer traits defining shared behavior across entity types.

use glam::Vec3;

use super::atom::{Atom, AtomRef};
use super::id::EntityId;
use super::{MoleculeType, Residue};
use crate::bond::CovalentBond;

/// Common behavior for all molecular entities.
pub trait Entity {
    /// Unique identifier for this entity.
    fn id(&self) -> EntityId;

    /// Classification of this entity's molecule type.
    fn molecule_type(&self) -> MoleculeType;

    /// Reference to the underlying atom data.
    fn atoms(&self) -> &[Atom];

    /// One atom by index, gathered by value. Panics if `i` is out of
    /// bounds, matching slice indexing.
    fn atom(&self, i: usize) -> Atom {
        self.atoms()[i].clone()
    }

    /// Layout-agnostic per-atom views over every atom, in storage order.
    fn atoms_iter(&self) -> impl Iterator<Item = AtomRef<'_>> {
        self.atoms().iter().map(AtomRef::from_atom)
    }

    /// All atom positions as `Vec3`.
    fn positions(&self) -> Vec<Vec3> {
        self.atoms().iter().map(|a| a.position).collect()
    }

    /// Number of atoms in this entity.
    fn atom_count(&self) -> usize {
        self.atoms().len()
    }

    /// Intra-entity covalent bonds, with `AtomId` endpoints into this
    /// entity's atoms. Entity types that carry no bond topology (bulk
    /// solvent) keep the default empty slice.
    fn bonds(&self) -> &[CovalentBond] {
        &[]
    }
}

/// Shared behavior for polymer entities (protein, DNA, RNA).
///
/// A polymer is a single chain instance that may contain backbone
/// gaps (missing residues), producing multiple continuous segments.
/// Segment breaks are indices into the residue array where the
/// backbone is discontinuous.
pub trait Polymer: Entity {
    /// Ordered residues in this polymer.
    fn residues(&self) -> &[Residue];

    /// Number of residues.
    fn residue_count(&self) -> usize {
        self.residues().len()
    }

    /// Indices into `residues()` where backbone segments begin a new
    /// continuous run. An empty vec means the entire chain is continuous
    /// (one segment). `[47]` means residues 0..47 are segment 0 and
    /// 47..N are segment 1.
    fn segment_breaks(&self) -> &[usize];

    /// Number of continuous backbone segments.
    fn segment_count(&self) -> usize {
        self.segment_breaks().len() + 1
    }

    /// Residue range for the given segment index.
    fn segment_range(&self, idx: usize) -> std::ops::Range<usize> {
        let breaks = self.segment_breaks();
        let n = self.residue_count();
        let start = if idx == 0 {
            0
        } else {
            *breaks.get(idx - 1).unwrap_or(&n)
        };
        let end = breaks.get(idx).copied().unwrap_or(n);
        start..end
    }

    /// Residues for the given segment index.
    fn segment_residues(&self, idx: usize) -> &[Residue] {
        let range = self.segment_range(idx);
        &self.residues()[range]
    }
}
