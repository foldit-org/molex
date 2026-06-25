//! Distance-based covalent bond inference for small molecules.
//!
//! Infers covalent bonds from atom positions and element covalent radii.
//! Used for ligands, waters, and other non-protein entities where
//! bond topology is not provided by a dictionary.

use crate::element::Element;
use crate::entity::molecule::atom::Atom;

/// Bond order classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondOrder {
    /// Single covalent bond.
    Single,
    /// Double covalent bond.
    Double,
    /// Triple covalent bond.
    Triple,
    /// Aromatic bond.
    Aromatic,
}

/// An inferred bond between two atoms.
#[derive(Debug, Clone)]
pub struct InferredBond {
    /// Index of the first atom in the input slice.
    pub atom_a: usize,
    /// Index of the second atom in the input slice.
    pub atom_b: usize,
    /// Inferred bond order.
    pub order: BondOrder,
}

/// Infer covalent bonds from atom positions and element types.
///
/// O(n^2) is fine for small molecules (ligands typically <100 atoms).
#[must_use]
pub fn infer_bonds(atoms: &[Atom], tolerance: f32) -> Vec<InferredBond> {
    let n = atoms.len();
    if n < 2 {
        return Vec::new();
    }

    let mut bonds = Vec::new();

    for i in 0..n {
        let elem_i = atoms[i].element;
        if elem_i == Element::H {
            continue;
        }
        let cov_i = elem_i.covalent_radius();

        for j in (i + 1)..n {
            let elem_j = atoms[j].element;
            if elem_i == Element::H && elem_j == Element::H {
                continue;
            }
            let cov_j = elem_j.covalent_radius();

            let dist = atoms[i].position.distance(atoms[j].position);
            let sum_cov = cov_i + cov_j;
            let single_threshold = sum_cov + tolerance;

            if dist <= single_threshold && dist > 0.4 {
                let order = if dist < sum_cov * 0.9 {
                    BondOrder::Double
                } else {
                    BondOrder::Single
                };

                bonds.push(InferredBond {
                    atom_a: i,
                    atom_b: j,
                    order,
                });
            }
        }
    }

    bonds
}

/// Default tolerance for bond inference (0.4 angstroms).
pub const DEFAULT_TOLERANCE: f32 = 0.4;

impl crate::Assembly {
    /// All intra-entity covalent bonds across this assembly, aggregated
    /// from each entity's stored bond topology
    /// ([`MoleculeEntity::bonds`](crate::entity::molecule::MoleculeEntity::bonds)),
    /// in entity-declaration order. Inter-entity bonds (e.g. disulfides)
    /// are not included — see [`crate::Assembly::disulfides`].
    #[must_use]
    pub fn covalent_bonds(&self) -> Vec<crate::bond::CovalentBond> {
        self.entities()
            .iter()
            .flat_map(|e| e.bonds().iter().cloned())
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use glam::Vec3;

    use super::*;

    fn make_atom(x: f32, y: f32, z: f32, element: Element) -> Atom {
        Atom {
            position: Vec3::new(x, y, z),
            occupancy: 1.0,
            b_factor: 0.0,
            element,
            name: *b"X   ",
            formal_charge: 0,
        }
    }

    #[test]
    fn test_simple_bond() {
        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, Element::C),
            make_atom(1.5, 0.0, 0.0, Element::C),
        ];
        let bonds = infer_bonds(&atoms, DEFAULT_TOLERANCE);
        assert_eq!(bonds.len(), 1);
        assert_eq!(bonds[0].atom_a, 0);
        assert_eq!(bonds[0].atom_b, 1);
        assert_eq!(bonds[0].order, BondOrder::Single);
    }

    #[test]
    fn test_double_bond() {
        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, Element::C),
            make_atom(1.23, 0.0, 0.0, Element::O),
        ];
        let bonds = infer_bonds(&atoms, DEFAULT_TOLERANCE);
        assert_eq!(bonds.len(), 1);
        assert_eq!(bonds[0].order, BondOrder::Double);
    }

    #[test]
    fn test_no_bond_far_apart() {
        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, Element::C),
            make_atom(5.0, 0.0, 0.0, Element::C),
        ];
        let bonds = infer_bonds(&atoms, DEFAULT_TOLERANCE);
        assert!(bonds.is_empty());
    }

    #[test]
    fn test_water_bonds() {
        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, Element::O),
            make_atom(0.757, 0.586, 0.0, Element::H),
            make_atom(-0.757, 0.586, 0.0, Element::H),
        ];
        let bonds = infer_bonds(&atoms, DEFAULT_TOLERANCE);
        assert_eq!(bonds.len(), 2);
    }
}
