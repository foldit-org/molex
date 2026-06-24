//! Disulfide bond detection between cysteine residues.
//!
//! Detects Cys-Cys disulfide bridges by finding SG-SG atom pairs within
//! the expected distance range (~2.05 A).

use std::borrow::Borrow;

use crate::analysis::BondOrder;
use crate::atom_id::AtomId;
use crate::bond::CovalentBond;
use crate::entity::molecule::MoleculeEntity;

/// Maximum SG-SG distance (angstroms) for a disulfide bond.
const MAX_SS_DISTANCE: f32 = 2.5;
/// Minimum SG-SG distance (angstroms) to avoid clashes.
const MIN_SS_DISTANCE: f32 = 1.5;

/// Detect disulfide (Cys SG-SG) bonds across a set of molecule entities.
///
/// Scans every [`MoleculeEntity::Protein`] for atoms named `"SG"` in
/// residues named `"CYS"`, then emits one [`CovalentBond`] for each
/// pair (intra- or inter-entity) whose SG-SG distance falls within the
/// 1.5-2.5 A disulfide range.
///
/// Endpoints use [`AtomId`] so bonds remain addressable after entities
/// are reordered or recomposed.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "atom indices are bounded by entity size (< u32::MAX)"
)]
pub fn detect_disulfides<E: Borrow<MoleculeEntity>>(
    entities: &[E],
) -> Vec<CovalentBond> {
    #[derive(Clone, Copy)]
    struct SgAtom {
        id: AtomId,
        position: glam::Vec3,
    }

    let mut sg_atoms: Vec<SgAtom> = Vec::new();
    for entity in entities {
        let MoleculeEntity::Protein(protein) = entity.borrow() else {
            continue;
        };
        for residue in &protein.residues {
            if trimmed_residue_name(&residue.name) != b"CYS" {
                continue;
            }
            for idx in residue.atom_range.clone() {
                let atom = &protein.atoms[idx];
                if trimmed_atom_name_bytes(&atom.name) == b"SG" {
                    sg_atoms.push(SgAtom {
                        id: AtomId {
                            entity: protein.id,
                            index: idx as u32,
                        },
                        position: atom.position,
                    });
                }
            }
        }
    }

    let mut bonds = Vec::new();
    for (i, a) in sg_atoms.iter().enumerate() {
        for b in &sg_atoms[i + 1..] {
            let dist = a.position.distance(b.position);
            if (MIN_SS_DISTANCE..=MAX_SS_DISTANCE).contains(&dist) {
                bonds.push(CovalentBond {
                    a: a.id,
                    b: b.id,
                    order: BondOrder::Single,
                });
            }
        }
    }
    bonds
}

fn trimmed_residue_name(name: &[u8; 3]) -> &[u8] {
    let mut end = 3;
    while end > 0 && (name[end - 1] == b' ' || name[end - 1] == 0) {
        end -= 1;
    }
    &name[..end]
}

fn trimmed_atom_name_bytes(name: &[u8; 4]) -> &[u8] {
    let mut end = 4;
    while end > 0 && (name[end - 1] == b' ' || name[end - 1] == 0) {
        end -= 1;
    }
    let mut start = 0;
    while start < end && (name[start] == b' ' || name[start] == 0) {
        start += 1;
    }
    &name[start..end]
}

impl crate::Assembly {
    /// Cys SG-SG disulfide bonds across this assembly. See
    /// [`detect_disulfides`].
    #[must_use]
    pub fn disulfides(&self) -> Vec<CovalentBond> {
        detect_disulfides(self.entities())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::element::Element;
    use crate::entity::molecule::atom::Atom;
    use crate::entity::molecule::id::EntityIdAllocator;
    use crate::entity::molecule::protein::ProteinEntity;
    use crate::entity::molecule::{MoleculeEntity, Residue};

    fn mk_atom(name: [u8; 4], el: Element, pos: Vec3) -> Atom {
        Atom {
            position: pos,
            occupancy: 1.0,
            b_factor: 0.0,
            element: el,
            name,
            formal_charge: 0,
        }
    }

    fn cys_residue(pos: Vec3) -> (Vec<Atom>, Residue) {
        // Minimum atom set: N, CA, C, O, CB, SG.
        let atoms = vec![
            mk_atom(*b"N   ", Element::N, pos),
            mk_atom(*b"CA  ", Element::C, pos + Vec3::new(1.0, 0.0, 0.0)),
            mk_atom(*b"C   ", Element::C, pos + Vec3::new(2.0, 0.0, 0.0)),
            mk_atom(*b"O   ", Element::O, pos + Vec3::new(2.0, 1.0, 0.0)),
            mk_atom(*b"CB  ", Element::C, pos + Vec3::new(1.0, -1.0, 0.0)),
            mk_atom(*b"SG  ", Element::S, pos + Vec3::new(1.0, -2.5, 0.0)),
        ];
        let residue = Residue {
            name: *b"CYS",
            label_seq_id: 1,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: 0..atoms.len(),
            variants: Vec::new(),
        };
        (atoms, residue)
    }

    #[test]
    fn detect_disulfides_single_interchain_pair() {
        let (mut atoms_a, res_a) = cys_residue(Vec3::new(0.0, 0.0, 0.0));
        // Chain B SG positioned ~2.03 A from chain A SG.
        // Chain A SG is at (1.0, -2.5, 0.0). Put chain B SG at
        // (1.0, -4.53, 0.0); distance 2.03.
        // For that, place chain B origin so that SG = (1.0, -4.53, 0.0);
        // SG offset is (1.0, -2.5, 0.0), so origin = (0.0, -2.03, 0.0).
        let (mut atoms_b, res_b) = cys_residue(Vec3::new(0.0, -2.03, 0.0));

        let mut alloc = EntityIdAllocator::new();
        let id_a = alloc.allocate();
        let id_b = alloc.allocate();

        let ent_a = ProteinEntity::new(
            id_a,
            std::mem::take(&mut atoms_a),
            vec![res_a],
            "A".to_owned(),
        );
        let ent_b = ProteinEntity::new(
            id_b,
            std::mem::take(&mut atoms_b),
            vec![res_b],
            "B".to_owned(),
        );
        let entities = vec![
            MoleculeEntity::Protein(ent_a),
            MoleculeEntity::Protein(ent_b),
        ];

        let bonds = detect_disulfides(&entities);
        assert_eq!(bonds.len(), 1);
        let b = &bonds[0];
        assert_ne!(b.a.entity, b.b.entity);
    }

    #[test]
    fn detect_disulfides_skips_non_cys_sg() {
        // Met SG-like sulfur at disulfide distance must not produce a
        // bond: detector keys on residue name == CYS.
        let mut atoms = vec![
            mk_atom(*b"N   ", Element::N, Vec3::new(0.0, 0.0, 0.0)),
            mk_atom(*b"CA  ", Element::C, Vec3::new(1.0, 0.0, 0.0)),
            mk_atom(*b"C   ", Element::C, Vec3::new(2.0, 0.0, 0.0)),
            mk_atom(*b"O   ", Element::O, Vec3::new(2.0, 1.0, 0.0)),
            mk_atom(*b"SG  ", Element::S, Vec3::new(1.0, -2.5, 0.0)),
            mk_atom(*b"SG  ", Element::S, Vec3::new(1.0, -4.53, 0.0)),
        ];
        let residue = Residue {
            name: *b"MET",
            label_seq_id: 1,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: 0..atoms.len(),
            variants: Vec::new(),
        };
        let mut alloc = EntityIdAllocator::new();
        let id = alloc.allocate();
        let ent = ProteinEntity::new(
            id,
            std::mem::take(&mut atoms),
            vec![residue],
            "A".to_owned(),
        );
        let entities = vec![MoleculeEntity::Protein(ent)];
        assert!(detect_disulfides(&entities).is_empty());
    }
}
