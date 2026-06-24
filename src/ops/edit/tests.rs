#![allow(clippy::unwrap_used, clippy::float_cmp)]

use glam::Vec3;

use super::{AssemblyEdit, EditError};
use crate::assembly::Assembly;
use crate::chemistry::variant::{ProtonationState, VariantTag};
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::id::EntityIdAllocator;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::MoleculeEntity;

fn atom_at(name: &str, element: Element, x: f32) -> Atom {
    let mut n = [b' '; 4];
    for (i, b) in name.bytes().take(4).enumerate() {
        n[i] = b;
    }
    Atom {
        position: Vec3::new(x, 0.0, 0.0),
        occupancy: 1.0,
        b_factor: 0.0,
        element,
        name: n,
        formal_charge: 0,
    }
}

fn res_bytes(s: &str) -> [u8; 3] {
    let mut n = [b' '; 3];
    for (i, b) in s.bytes().take(3).enumerate() {
        n[i] = b;
    }
    n
}

fn residue(name: &str, seq: i32, range: std::ops::Range<usize>) -> Residue {
    Residue {
        name: res_bytes(name),
        label_seq_id: seq,
        auth_seq_id: None,
        auth_comp_id: None,
        ins_code: None,
        atom_range: range,
        variants: Vec::new(),
    }
}

fn ala_atoms(start_x: f32) -> Vec<Atom> {
    vec![
        atom_at("N", Element::N, start_x),
        atom_at("CA", Element::C, start_x + 1.0),
        atom_at("C", Element::C, start_x + 2.0),
        atom_at("O", Element::O, start_x + 3.0),
    ]
}

fn trp_atoms(start_x: f32) -> Vec<Atom> {
    // 14 atoms — different count from ALA's 4.
    (0u16..14)
        .map(|i| atom_at("CA", Element::C, start_x + f32::from(i)))
        .collect()
}

fn small_protein() -> Assembly {
    let id = EntityIdAllocator::new().allocate();
    let atoms = {
        let mut v = ala_atoms(0.0);
        v.extend(ala_atoms(10.0));
        v
    };
    // Residues named UNK so the missing-atom completion pass is a no-op;
    // these coordinate-edit tests fix the entity at exactly two 4-atom
    // residues and are not about chemistry completion.
    let residues = vec![residue("UNK", 1, 0..4), residue("UNK", 2, 4..8)];
    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        atoms,
        residues,
        "A".to_owned(),
    ));
    Assembly::new(vec![protein])
}

#[test]
fn set_entity_coords_updates_all_positions_and_bumps_generation() {
    let mut a = small_protein();
    let id = a.entities()[0].id();
    let start_gen = a.generation();

    let new_coords: Vec<Vec3> = (0u16..8)
        .map(|i| Vec3::new(100.0 + f32::from(i), 0.0, 0.0))
        .collect();
    a.apply_edit(&AssemblyEdit::SetEntityCoords {
        entity: id,
        coords: new_coords.clone(),
    })
    .unwrap();

    assert_eq!(a.generation(), start_gen + 1);
    let positions: Vec<Vec3> = a.entities()[0].positions();
    assert_eq!(positions, new_coords);
}

#[test]
fn set_entity_coords_count_mismatch_rejected() {
    let mut a = small_protein();
    let id = a.entities()[0].id();
    let start_gen = a.generation();

    let err = a
        .apply_edit(&AssemblyEdit::SetEntityCoords {
            entity: id,
            coords: vec![Vec3::ZERO; 3],
        })
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::CountMismatch {
            expected: 8,
            got: 3,
            ..
        }
    ));
    // Generation must NOT advance on a rejected edit.
    assert_eq!(a.generation(), start_gen);
}

#[test]
fn set_residue_coords_touches_only_target_residue() {
    let mut a = small_protein();
    let id = a.entities()[0].id();

    let before = a.entities()[0].positions();
    let new_r0_coords = vec![Vec3::new(1.0, 1.0, 1.0); 4];
    a.apply_edit(&AssemblyEdit::SetResidueCoords {
        entity: id,
        residue_idx: 0,
        coords: new_r0_coords.clone(),
    })
    .unwrap();

    let after = a.entities()[0].positions();
    assert_eq!(&after[0..4], new_r0_coords.as_slice());
    // Residue 1's atoms (indices 4..8) untouched.
    assert_eq!(&after[4..8], &before[4..8]);
}

#[test]
fn mutate_residue_changes_atom_count_and_shifts_subsequent_ranges() {
    let mut a = small_protein();
    let id = a.entities()[0].id();

    // Mutate residue 0 (ALA, 4 atoms) -> TRP (14 atoms). Residue 1's
    // atom_range should shift from 4..8 to 14..18.
    a.apply_edit(&AssemblyEdit::MutateResidue {
        entity: id,
        residue_idx: 0,
        new_name: res_bytes("TRP"),
        new_atoms: trp_atoms(0.0),
        new_variants: vec![],
    })
    .unwrap();

    let p = a.entities()[0].as_protein().unwrap();
    assert_eq!(p.residues[0].name, res_bytes("TRP"));
    assert_eq!(p.residues[0].atom_range, 0..14);
    assert_eq!(p.residues[1].atom_range, 14..18);
    assert_eq!(p.atoms.len(), 18);
}

#[test]
fn set_variants_replaces_residue_variants() {
    let mut a = small_protein();
    let id = a.entities()[0].id();

    a.apply_edit(&AssemblyEdit::SetVariants {
        entity: id,
        residue_idx: 0,
        variants: vec![
            VariantTag::Protonation(ProtonationState::HisEpsilon),
            VariantTag::Disulfide,
        ],
    })
    .unwrap();

    let p = a.entities()[0].as_protein().unwrap();
    assert_eq!(p.residues[0].variants.len(), 2);
    assert!(matches!(
        p.residues[0].variants[0],
        VariantTag::Protonation(ProtonationState::HisEpsilon),
    ));
}

#[test]
fn unknown_entity_returns_error_without_bumping_generation() {
    let mut a = small_protein();
    let start_gen = a.generation();
    // Burn through the global allocator to mint an id that won't
    // appear in the assembly.
    let mut spare = EntityIdAllocator::new();
    let _ = spare.allocate();
    let _ = spare.allocate();
    let bogus = spare.allocate(); // a different namespace than the assembly's

    let err = a
        .apply_edit(&AssemblyEdit::SetEntityCoords {
            entity: bogus,
            coords: vec![Vec3::ZERO; 8],
        })
        .unwrap_err();
    assert!(matches!(err, EditError::UnknownEntity(_)));
    assert_eq!(a.generation(), start_gen);
}

#[test]
fn unknown_residue_index_returns_error() {
    let mut a = small_protein();
    let id = a.entities()[0].id();

    let err = a
        .apply_edit(&AssemblyEdit::SetResidueCoords {
            entity: id,
            residue_idx: 99,
            coords: vec![Vec3::ZERO; 4],
        })
        .unwrap_err();
    assert!(matches!(
        err,
        EditError::UnknownResidue {
            residue_idx: 99,
            ..
        }
    ));
}

#[test]
fn apply_edits_stops_at_first_error_and_reports_index() {
    let mut a = small_protein();
    let id = a.entities()[0].id();
    let start_gen = a.generation();

    let edits = vec![
        AssemblyEdit::SetResidueCoords {
            entity: id,
            residue_idx: 0,
            coords: vec![Vec3::ZERO; 4],
        },
        // Bad: wrong count.
        AssemblyEdit::SetEntityCoords {
            entity: id,
            coords: vec![Vec3::ZERO; 99],
        },
        // Should not be reached.
        AssemblyEdit::SetResidueCoords {
            entity: id,
            residue_idx: 1,
            coords: vec![Vec3::ONE; 4],
        },
    ];
    let err = a.apply_edits(&edits).unwrap_err();
    assert_eq!(err.index, 1);
    assert!(matches!(err.source, EditError::CountMismatch { .. }));
    // First edit was applied; gen advanced by exactly 1.
    assert_eq!(a.generation(), start_gen + 1);
}
