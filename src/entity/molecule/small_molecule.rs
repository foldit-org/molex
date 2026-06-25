//! Small molecule entity: a single non-polymer molecule (ligand, ion,
//! cofactor, lipid).

use super::atom::{Atom, AtomColumns};
use super::id::EntityId;
use super::traits::Entity;
use super::MoleculeType;
use crate::analysis::{infer_bonds, DEFAULT_TOLERANCE};
use crate::atom_id::AtomId;
use crate::bond::CovalentBond;

/// A single non-polymer molecule.
#[derive(Debug, Clone)]
pub struct SmallMoleculeEntity {
    /// Unique entity identifier.
    pub id: EntityId,
    /// Molecule type (Ligand, Ion, Cofactor, or Lipid).
    pub mol_type: MoleculeType,
    /// Atom data, stored as parallel columns.
    pub columns: AtomColumns,
    /// 3-character residue code (e.g. b"ATP").
    pub residue_name: [u8; 3],
    /// Human-readable name for display (e.g. "Chlorophyll A").
    pub display_name: String,
    /// Intra-entity covalent bonds with `AtomId` endpoints. Populated
    /// at construction by distance-based [`infer_bonds`] over the
    /// entity's atoms.
    pub bonds: Vec<CovalentBond>,
}

impl SmallMoleculeEntity {
    /// Construct from a list of atoms. Bonds are inferred via
    /// distance-based [`infer_bonds`]. The display name is derived from
    /// `residue_name` and `mol_type`.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "atom indices bounded by small-molecule atom count"
    )]
    pub fn new(
        id: EntityId,
        mol_type: MoleculeType,
        atoms: Vec<Atom>,
        residue_name: [u8; 3],
    ) -> Self {
        let rn_str = std::str::from_utf8(&residue_name).unwrap_or("???").trim();
        let display_name =
            super::classify::small_molecule_display_name(mol_type, rn_str);
        let bonds = infer_bonds(&atoms, DEFAULT_TOLERANCE)
            .into_iter()
            .map(|b| CovalentBond {
                a: AtomId {
                    entity: id,
                    index: b.atom_a as u32,
                },
                b: AtomId {
                    entity: id,
                    index: b.atom_b as u32,
                },
                order: b.order,
            })
            .collect();
        Self {
            id,
            mol_type,
            columns: AtomColumns::from_atoms(atoms),
            residue_name,
            display_name,
            bonds,
        }
    }
}

impl Entity for SmallMoleculeEntity {
    fn id(&self) -> EntityId {
        self.id
    }
    fn molecule_type(&self) -> MoleculeType {
        self.mol_type
    }
    fn columns(&self) -> &AtomColumns {
        &self.columns
    }
    fn bonds(&self) -> &[CovalentBond] {
        &self.bonds
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::element::Element;
    use crate::entity::molecule::id::EntityIdAllocator;

    fn atom_at(name: &str, element: Element, x: f32, y: f32, z: f32) -> Atom {
        let mut n = [b' '; 4];
        for (i, b) in name.bytes().take(4).enumerate() {
            n[i] = b;
        }
        Atom {
            position: Vec3::new(x, y, z),
            occupancy: 1.0,
            b_factor: 0.0,
            element,
            name: n,
            formal_charge: 0,
            observed: true,
        }
    }

    fn res_bytes(s: &str) -> [u8; 3] {
        let mut n = [b' '; 3];
        for (i, b) in s.bytes().take(3).enumerate() {
            n[i] = b;
        }
        n
    }

    #[test]
    fn small_molecule_populates_bonds_from_positions() {
        let atoms = vec![
            atom_at("O1", Element::O, 0.0, 0.0, 0.0),
            atom_at("H1", Element::H, 0.95, 0.0, 0.0),
            atom_at("H2", Element::H, -0.24, 0.92, 0.0),
        ];
        let id = EntityIdAllocator::new().allocate();
        let sm = SmallMoleculeEntity::new(
            id,
            MoleculeType::Ligand,
            atoms,
            res_bytes("HOL"),
        );
        assert_eq!(sm.bonds.len(), 2);
        for b in &sm.bonds {
            assert_eq!(b.a.entity, sm.id);
            assert_eq!(b.b.entity, sm.id);
        }
    }
}
