#![allow(clippy::unwrap_used, clippy::float_cmp)]

use glam::Vec3;

use super::EntityDiffError;
use crate::assembly::Assembly;
use crate::chemistry::variant::{ProtonationState, VariantTag};
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::id::EntityIdAllocator;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::small_molecule::SmallMoleculeEntity;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};

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

/// Five-atom residue (N, CA, C, O, CB) — a real sidechain stub, distinct
/// atom count from glycine-style 4-atom residues.
fn ser_atoms(start_x: f32) -> Vec<Atom> {
    let mut v = ala_atoms(start_x);
    v.push(atom_at("CB", Element::C, start_x + 4.0));
    v
}

/// One protein chain from the given per-residue (name, atoms) list.
fn protein(residues: &[(&str, Vec<Atom>)]) -> MoleculeEntity {
    let id = EntityIdAllocator::new().allocate();
    let mut atoms = Vec::new();
    let mut res = Vec::new();
    for (seq, (name, ratoms)) in (1i32..).zip(residues) {
        let start = atoms.len();
        atoms.extend(ratoms.iter().cloned());
        res.push(residue(name, seq, start..atoms.len()));
    }
    MoleculeEntity::Protein(ProteinEntity::new(id, atoms, res, "A".to_owned()))
}

/// Apply `prior.diff(new)` to an assembly holding `prior` and assert the
/// result reproduces `new`'s positions, residue names, and variants.
fn assert_round_trips(prior: &MoleculeEntity, new: &MoleculeEntity) {
    let edits = prior.diff(new).unwrap();
    let mut asm = Assembly::new(vec![prior.clone()]);
    asm.apply_edits(&edits).unwrap();
    let got = &asm.entities()[0];
    assert_eq!(got.positions(), new.positions(), "positions mismatch");
    let (Some(g_res), Some(n_res)) = (got.residues(), new.residues()) else {
        return;
    };
    assert_eq!(g_res.len(), n_res.len());
    for (g, n) in g_res.iter().zip(n_res.iter()) {
        assert_eq!(g.name, n.name, "residue name mismatch");
        assert_eq!(g.variants, n.variants, "residue variants mismatch");
    }
}

#[test]
fn no_change_yields_no_edits() {
    let a = protein(&[("ALA", ala_atoms(0.0)), ("ALA", ala_atoms(10.0))]);
    assert!(a.diff(&a).unwrap().is_empty());
}

#[test]
fn coord_only_yields_single_set_entity_coords() {
    let a = protein(&[("ALA", ala_atoms(0.0)), ("ALA", ala_atoms(10.0))]);
    let b = protein(&[("ALA", ala_atoms(5.0)), ("ALA", ala_atoms(15.0))]);
    let edits = a.diff(&b).unwrap();
    assert_eq!(edits.len(), 1);
    assert!(matches!(
        edits[0],
        crate::ops::edit::AssemblyEdit::SetEntityCoords { .. }
    ));
    assert_round_trips(&a, &b);
}

#[test]
fn mutation_changes_atom_count_and_round_trips() {
    // ALA (4 atoms) -> SER (5 atoms) at residue 0: an atom-count-changing
    // mutation, the animation's core case.
    let a = protein(&[("ALA", ala_atoms(0.0)), ("ALA", ala_atoms(10.0))]);
    let b = protein(&[("SER", ser_atoms(0.0)), ("ALA", ala_atoms(10.0))]);
    let edits = a.diff(&b).unwrap();
    assert!(edits.iter().any(|e| matches!(
        e,
        crate::ops::edit::AssemblyEdit::MutateResidue { .. }
    )));
    assert_round_trips(&a, &b);
}

#[test]
fn mutation_plus_unchanged_residue_move_round_trips() {
    // Residue 0 mutates (ALA->SER) AND residue 1 just moves: the diff must
    // emit MutateResidue(0) + SetResidueCoords(1), and round-trip.
    let a = protein(&[("ALA", ala_atoms(0.0)), ("ALA", ala_atoms(10.0))]);
    let b = protein(&[("SER", ser_atoms(0.0)), ("ALA", ala_atoms(20.0))]);
    assert_round_trips(&a, &b);
}

#[test]
fn variant_change_yields_set_variants_and_round_trips() {
    let a = protein(&[("ALA", ala_atoms(0.0))]);
    let mut b = protein(&[("ALA", ala_atoms(0.0))]);
    if let Some((_, residues)) = b.polymer_columns_mut() {
        residues[0].variants =
            vec![VariantTag::Protonation(ProtonationState::HisEpsilon)];
    }
    let edits = a.diff(&b).unwrap();
    assert_eq!(edits.len(), 1);
    assert!(matches!(
        edits[0],
        crate::ops::edit::AssemblyEdit::SetVariants { .. }
    ));
    assert_round_trips(&a, &b);
}

#[test]
fn residue_count_change_is_indel_error() {
    let a = protein(&[("ALA", ala_atoms(0.0)), ("ALA", ala_atoms(10.0))]);
    let b = protein(&[("ALA", ala_atoms(0.0))]);
    assert_eq!(
        a.diff(&b).unwrap_err(),
        EntityDiffError::ResidueCountChanged { prior: 2, new: 1 }
    );
}

#[test]
fn incompatible_variety_errors() {
    let a = protein(&[("ALA", ala_atoms(0.0))]);
    let lig_id = EntityIdAllocator::new().allocate();
    let lig = MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
        lig_id,
        MoleculeType::Ligand,
        vec![atom_at("C1", Element::C, 0.0)],
        res_bytes("LIG"),
    ));
    assert_eq!(
        a.diff(&lig).unwrap_err(),
        EntityDiffError::IncompatibleVariety
    );
}
