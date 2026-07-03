//! Bulk entity: many identical small molecules (water, solvent).

use super::atom::{Atom, AtomColumns};
use super::id::EntityId;
use super::traits::Entity;
use super::MoleculeType;

/// A group of identical small molecules (water, solvent).
#[derive(Debug, Clone)]
pub struct BulkEntity {
    /// Unique entity identifier.
    pub id: EntityId,
    /// Molecule type (Water or Solvent).
    pub mol_type: MoleculeType,
    /// Atom data for all molecules in this group, stored as parallel
    /// columns.
    pub columns: AtomColumns,
    /// 3-character residue code (e.g. b"HOH").
    pub residue_name: [u8; 3],
    /// Number of individual molecules in this group.
    pub molecule_count: usize,
    /// PDB chain identifier the molecules came in on. Empty when the source
    /// carried no chain.
    pub pdb_chain_id: String,
}

impl BulkEntity {
    /// Construct from a list of atoms. `molecule_count` is supplied by
    /// the caller (typically the number of residues in the source).
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "non-polymer construction carries the source chain alongside \
                  the atom group keys"
    )]
    pub fn new(
        id: EntityId,
        mol_type: MoleculeType,
        atoms: Vec<Atom>,
        residue_name: [u8; 3],
        molecule_count: usize,
        chain_id: String,
    ) -> Self {
        Self {
            id,
            mol_type,
            columns: AtomColumns::from_atoms(atoms),
            residue_name,
            molecule_count,
            pdb_chain_id: chain_id,
        }
    }
}

impl Entity for BulkEntity {
    fn id(&self) -> EntityId {
        self.id
    }
    fn molecule_type(&self) -> MoleculeType {
        self.mol_type
    }
    fn columns(&self) -> &AtomColumns {
        &self.columns
    }
}
