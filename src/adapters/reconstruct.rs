//! Shared entity-reconstruction core for the conversion adapters.
//!
//! Both the AtomWorks Biotite bridge and the ML model-output adapter build
//! `MoleculeEntity` values from a flat, ordered list of per-atom rows. This
//! module owns that grouping logic so neither adapter reaches into the other:
//! given a `Vec<ReconstructRow>` (atom plus its residue/chain context), it
//! splits the run into residues on `res_num` changes and dispatches by
//! [`MoleculeType`] into the right entity variant.

use std::collections::HashSet;

use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::bulk::BulkEntity;
use crate::entity::molecule::id::EntityId;
use crate::entity::molecule::nucleic_acid::NAEntity;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::small_molecule::SmallMoleculeEntity;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};

/// One atom decoded from an adapter source, paired with the per-atom
/// residue/chain context used to group atoms into residues.
pub(crate) struct ReconstructRow {
    pub atom: Atom,
    pub chain_id: String,
    pub res_name: [u8; 3],
    pub res_num: i32,
}

/// Build a single `MoleculeEntity` from an ordered group of rows.
///
/// Rows must already be sorted into residue order (atoms of one residue
/// contiguous); residue boundaries are taken from `res_num` changes.
pub(crate) fn build_entity(
    id: EntityId,
    mol_type: MoleculeType,
    rows: Vec<ReconstructRow>,
) -> MoleculeEntity {
    let pdb_chain_id =
        rows.first().map_or_else(|| " ".to_owned(), |r| r.chain_id.clone());

    match mol_type {
        MoleculeType::Protein => {
            let (atoms, residues) = into_atoms_and_residues(rows);
            MoleculeEntity::Protein(ProteinEntity::new(
                id,
                atoms,
                residues,
                pdb_chain_id,
            ))
        }
        MoleculeType::DNA | MoleculeType::RNA => {
            let (atoms, residues) = into_atoms_and_residues(rows);
            MoleculeEntity::NucleicAcid(NAEntity::new(
                id,
                mol_type,
                atoms,
                residues,
                pdb_chain_id,
            ))
        }
        MoleculeType::Water | MoleculeType::Solvent => {
            let residue_name = rows.first().map_or([b' '; 3], |r| r.res_name);
            let mut seen = HashSet::new();
            for row in &rows {
                let _ = seen.insert((row.chain_id.clone(), row.res_num));
            }
            let molecule_count = seen.len();
            let atoms = rows.into_iter().map(|r| r.atom).collect();
            MoleculeEntity::Bulk(BulkEntity::new(
                id,
                mol_type,
                atoms,
                residue_name,
                molecule_count,
            ))
        }
        MoleculeType::Ligand
        | MoleculeType::Ion
        | MoleculeType::Cofactor
        | MoleculeType::Lipid => {
            let residue_name = rows.first().map_or([b' '; 3], |r| r.res_name);
            let atoms = rows.into_iter().map(|r| r.atom).collect();
            MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
                id,
                mol_type,
                atoms,
                residue_name,
            ))
        }
    }
}

/// Split an ordered row list into a flat atom vec plus the residues that
/// index into it. A new residue starts on each `res_num` change; the atom
/// range of each residue is contiguous and the ranges cover all atoms with
/// no gap or overlap.
pub(crate) fn into_atoms_and_residues(
    rows: Vec<ReconstructRow>,
) -> (Vec<Atom>, Vec<Residue>) {
    let mut atoms = Vec::with_capacity(rows.len());
    let mut residues = Vec::new();
    let mut current_res_num: Option<i32> = None;
    let mut current_res_name: [u8; 3] = [b' '; 3];
    let mut current_start = 0usize;

    for row in rows {
        let new_residue = current_res_num.is_none_or(|n| n != row.res_num);
        if new_residue {
            if let Some(rn) = current_res_num {
                residues.push(Residue {
                    name: current_res_name,
                    label_seq_id: rn,
                    auth_seq_id: None,
                    auth_comp_id: None,
                    ins_code: None,
                    atom_range: current_start..atoms.len(),
                    variants: Vec::new(),
                });
            }
            current_start = atoms.len();
            current_res_num = Some(row.res_num);
            current_res_name = row.res_name;
        }
        atoms.push(row.atom);
    }
    if let Some(rn) = current_res_num {
        residues.push(Residue {
            name: current_res_name,
            label_seq_id: rn,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: current_start..atoms.len(),
            variants: Vec::new(),
        });
    }

    (atoms, residues)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "tests: coords are exact f32 copies, no arithmetic"
)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::element::Element;

    #[allow(
        clippy::too_many_arguments,
        reason = "test row constructor mirrors the per-atom annotation set"
    )]
    fn mk_row(
        name: [u8; 4],
        element: Element,
        pos: Vec3,
        chain_id: u8,
        res_name: [u8; 3],
        res_num: i32,
    ) -> ReconstructRow {
        ReconstructRow {
            atom: Atom {
                position: pos,
                occupancy: 1.0,
                b_factor: 0.0,
                element,
                name,
                formal_charge: 0,
            },
            chain_id: String::from(chain_id as char),
            res_name,
            res_num,
        }
    }

    #[test]
    fn into_atoms_and_residues_empty_yields_nothing() {
        let (atoms, residues) = into_atoms_and_residues(Vec::new());
        assert!(atoms.is_empty());
        assert!(residues.is_empty());
    }

    #[test]
    fn build_entity_empty_rows_uses_fill_defaults() {
        // Edge case: an entity group with zero atoms must not panic and must
        // fall back to the documented fill values rather than indexing past
        // the (empty) row vec. A wrong fill here would silently corrupt the
        // residue/chain metadata of an empty group.
        let lig =
            build_entity(EntityId::from_raw(0), MoleculeType::Ligand, vec![]);
        let MoleculeEntity::SmallMolecule(e) = lig else {
            unreachable!("Ligand mol_type must build a SmallMolecule");
        };
        assert!(e.atoms.is_empty());
        assert_eq!(e.residue_name, [b' '; 3]);

        let water =
            build_entity(EntityId::from_raw(1), MoleculeType::Water, vec![]);
        assert!(matches!(water, MoleculeEntity::Bulk(_)));
    }

    #[test]
    fn into_atoms_and_residues_groups_by_res_num() {
        // Three atoms in residue 1, two in residue 2; the split must land
        // exactly on the res_num change (catches an off-by-one in the
        // atom_range boundary).
        let rows = vec![
            mk_row(*b"N   ", Element::N, Vec3::ZERO, b'A', *b"ALA", 1),
            mk_row(*b"CA  ", Element::C, Vec3::X, b'A', *b"ALA", 1),
            mk_row(*b"C   ", Element::C, Vec3::Y, b'A', *b"ALA", 1),
            mk_row(*b"N   ", Element::N, Vec3::Z, b'A', *b"GLY", 2),
            mk_row(*b"CA  ", Element::C, Vec3::ONE, b'A', *b"GLY", 2),
        ];
        let (atoms, residues) = into_atoms_and_residues(rows);

        assert_eq!(atoms.len(), 5);
        assert_eq!(residues.len(), 2);

        assert_eq!(residues[0].name, *b"ALA");
        assert_eq!(residues[0].label_seq_id, 1);
        assert_eq!(residues[0].atom_range, 0..3);

        assert_eq!(residues[1].name, *b"GLY");
        assert_eq!(residues[1].label_seq_id, 2);
        assert_eq!(residues[1].atom_range, 3..5);

        // atom_range slices index back into the contiguous atom vec correctly.
        assert_eq!(atoms[residues[1].atom_range.clone()].len(), 2);
    }

    #[test]
    fn into_atoms_and_residues_single_residue() {
        let rows = vec![
            mk_row(*b"C1  ", Element::C, Vec3::ZERO, b'A', *b"LIG", 7),
            mk_row(*b"O1  ", Element::O, Vec3::X, b'A', *b"LIG", 7),
        ];
        let (atoms, residues) = into_atoms_and_residues(rows);
        assert_eq!(atoms.len(), 2);
        assert_eq!(residues.len(), 1);
        assert_eq!(residues[0].atom_range, 0..2);
    }

    #[test]
    fn build_entity_dispatches_by_mol_type() {
        let protein_rows = vec![
            mk_row(*b"N   ", Element::N, Vec3::ZERO, b'A', *b"ALA", 1),
            mk_row(*b"CA  ", Element::C, Vec3::X, b'A', *b"ALA", 1),
        ];
        let id = EntityId::from_raw(0);
        let e = build_entity(id, MoleculeType::Protein, protein_rows);
        assert!(matches!(e, MoleculeEntity::Protein(_)));
        assert_eq!(e.id(), id);

        let lig_rows =
            vec![mk_row(*b"C1  ", Element::C, Vec3::ZERO, b'B', *b"LIG", 1)];
        let lig =
            build_entity(EntityId::from_raw(1), MoleculeType::Ligand, lig_rows);
        assert!(matches!(lig, MoleculeEntity::SmallMolecule(_)));

        let water_rows =
            vec![mk_row(*b"O   ", Element::O, Vec3::ZERO, b'C', *b"HOH", 1)];
        let water = build_entity(
            EntityId::from_raw(2),
            MoleculeType::Water,
            water_rows,
        );
        assert!(matches!(water, MoleculeEntity::Bulk(_)));
    }

    /// The grouping logic that decides atom order, residue grouping, and
    /// coordinate placement: a two-residue protein-like group must round-trip
    /// with exact coords and contiguous, gap-free residue ranges.
    #[test]
    fn reconstruct_multi_residue_entity_roundtrip() {
        let coords = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.2, 0.0, 0.0),
            Vec3::new(4.2, 0.0, 0.0),
        ];
        let names: [[u8; 4]; 5] =
            [*b"N   ", *b"CA  ", *b"C   ", *b"N   ", *b"CA  "];
        let res_nums = [1, 1, 1, 2, 2];
        let res_names: [[u8; 3]; 5] =
            [*b"ALA", *b"ALA", *b"ALA", *b"GLY", *b"GLY"];

        let rows: Vec<ReconstructRow> = (0..5)
            .map(|i| {
                mk_row(
                    names[i],
                    Element::from_atom_name(
                        std::str::from_utf8(&names[i]).unwrap(),
                    ),
                    coords[i],
                    b'A',
                    res_names[i],
                    res_nums[i],
                )
            })
            .collect();

        let (atoms, residues) = into_atoms_and_residues(rows);

        assert_eq!(atoms.len(), 5);
        for (i, a) in atoms.iter().enumerate() {
            assert_eq!(a.position, coords[i]);
        }
        assert_eq!(residues.len(), 2);
        assert_eq!(residues[0].atom_range, 0..3);
        assert_eq!(residues[1].atom_range, 3..5);
        assert_eq!(
            residues[0].atom_range.end, residues[1].atom_range.start,
            "residue ranges must be contiguous"
        );
        assert_eq!(residues.last().unwrap().atom_range.end, atoms.len());
    }
}
