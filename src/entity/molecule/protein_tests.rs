//! Tests for `protein.rs`. Split into a sibling file to keep the primary
//! module under the 800-line source cap enforced by `just file-lengths`.

#![allow(clippy::unwrap_used, clippy::float_cmp)]

use super::*;
use crate::element::Element;
use crate::entity::molecule::id::EntityIdAllocator;
use crate::entity::molecule::polymer::Residue;

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

fn build_protein(atoms: Vec<Atom>, residues: Vec<Residue>) -> ProteinEntity {
    let id = EntityIdAllocator::new().allocate();
    ProteinEntity::new(id, atoms, residues, b'A', None)
}

/// Two ALA-GLY residues with a 10 A C->N gap between them (segment break).
fn two_residue_protein_with_break() -> ProteinEntity {
    let atoms = vec![
        atom_at("N", Element::N, 1.0, 2.0, 3.0),
        atom_at("CA", Element::C, 4.0, 5.0, 6.0),
        atom_at("C", Element::C, 7.0, 8.0, 9.0),
        atom_at("O", Element::O, 10.0, 11.0, 12.0),
        atom_at("N", Element::N, 13.0, 14.0, 15.0),
        atom_at("CA", Element::C, 16.0, 17.0, 18.0),
        atom_at("C", Element::C, 19.0, 20.0, 21.0),
        atom_at("O", Element::O, 22.0, 23.0, 24.0),
    ];
    let residues = vec![residue("ALA", 1, 0..4), residue("GLY", 2, 4..8)];
    build_protein(atoms, residues)
}

// -- Sidechain tests --

#[test]
fn sidechain_empty() {
    let sc = Sidechain::empty();
    assert!(sc.is_empty());
    assert!(sc.atoms.is_empty());
    assert!(sc.bonds.is_empty());
}

#[test]
fn sidechain_is_empty_with_atoms() {
    let sc = Sidechain {
        atoms: vec![Atom {
            position: Vec3::ZERO,
            occupancy: 1.0,
            b_factor: 0.0,
            element: Element::C,
            name: *b"CB  ",
            formal_charge: 0,
        }],
        bonds: Vec::new(),
    };
    assert!(!sc.is_empty());
}

// -- to_backbone tests --

#[test]
fn to_backbone_returns_two_entries() {
    let protein = two_residue_protein_with_break();
    let backbone = protein.to_backbone();
    assert_eq!(backbone.len(), 2);
}

#[test]
fn to_backbone_first_residue_positions() {
    let protein = two_residue_protein_with_break();
    let bb0 = &protein.to_backbone()[0];
    assert!((bb0.n.x - 1.0).abs() < 1e-6);
    assert!((bb0.ca.x - 4.0).abs() < 1e-6);
    assert!((bb0.c.x - 7.0).abs() < 1e-6);
    assert!((bb0.o.x - 10.0).abs() < 1e-6);
}

#[test]
fn to_backbone_second_residue_positions() {
    let protein = two_residue_protein_with_break();
    let bb1 = &protein.to_backbone()[1];
    assert!((bb1.n.x - 13.0).abs() < 1e-6);
    assert!((bb1.ca.x - 16.0).abs() < 1e-6);
    assert!((bb1.c.x - 19.0).abs() < 1e-6);
    assert!((bb1.o.x - 22.0).abs() < 1e-6);
}

// -- segment breaks --

#[test]
fn segment_break_detected_for_large_gap() {
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("CA", Element::C, 1.0, 0.0, 0.0),
        atom_at("C", Element::C, 2.0, 0.0, 0.0),
        atom_at("O", Element::O, 2.0, 1.0, 0.0),
        atom_at("N", Element::N, 20.0, 0.0, 0.0),
        atom_at("CA", Element::C, 21.0, 0.0, 0.0),
        atom_at("C", Element::C, 22.0, 0.0, 0.0),
        atom_at("O", Element::O, 22.0, 1.0, 0.0),
    ];
    let residues = vec![residue("ALA", 1, 0..4), residue("ALA", 2, 4..8)];
    let protein = build_protein(atoms, residues);
    assert!(
        !protein.segment_breaks.is_empty(),
        "should have segment break at large gap"
    );
}

#[test]
fn no_segment_break_for_close_residues() {
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("CA", Element::C, 1.5, 0.0, 0.0),
        atom_at("C", Element::C, 2.5, 0.0, 0.0),
        atom_at("O", Element::O, 2.5, 1.0, 0.0),
        atom_at("N", Element::N, 3.8, 0.0, 0.0),
        atom_at("CA", Element::C, 5.0, 0.0, 0.0),
        atom_at("C", Element::C, 6.0, 0.0, 0.0),
        atom_at("O", Element::O, 6.0, 1.0, 0.0),
    ];
    let residues = vec![residue("ALA", 1, 0..4), residue("ALA", 2, 4..8)];
    let protein = build_protein(atoms, residues);
    assert!(
        protein.segment_breaks.is_empty(),
        "should have no segment break for peptide-bonded residues"
    );
}

// -- to_interleaved_segments --

#[test]
fn interleaved_segments_for_single_segment() {
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("CA", Element::C, 1.5, 0.0, 0.0),
        atom_at("C", Element::C, 2.5, 0.0, 0.0),
        atom_at("O", Element::O, 2.5, 1.0, 0.0),
        atom_at("N", Element::N, 3.8, 0.0, 0.0),
        atom_at("CA", Element::C, 5.0, 0.0, 0.0),
        atom_at("C", Element::C, 6.0, 0.0, 0.0),
        atom_at("O", Element::O, 6.0, 1.0, 0.0),
    ];
    let residues = vec![residue("ALA", 1, 0..4), residue("ALA", 2, 4..8)];
    let protein = build_protein(atoms, residues);
    let segments = protein.to_interleaved_segments();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].len(), 6);
}

// -- Canonical ordering + bond population --

/// ALA input atoms in non-canonical order: CB, O, N, HA, CA, C.
fn alanine_scrambled_protein() -> ProteinEntity {
    let atoms = vec![
        atom_at("CB", Element::C, 50.0, 0.0, 0.0),
        atom_at("O", Element::O, 10.0, 11.0, 12.0),
        atom_at("N", Element::N, 1.0, 2.0, 3.0),
        atom_at("HA", Element::H, 60.0, 0.0, 0.0),
        atom_at("CA", Element::C, 4.0, 5.0, 6.0),
        atom_at("C", Element::C, 7.0, 8.0, 9.0),
    ];
    let residues = vec![residue("ALA", 1, 0..6)];
    build_protein(atoms, residues)
}

#[test]
fn canonical_ordering_reorders_scrambled_input() {
    let protein = alanine_scrambled_protein();
    assert_eq!(protein.residues.len(), 1);
    let r = &protein.residues[0];
    let names: Vec<&str> = r
        .atom_range
        .clone()
        .map(|i| std::str::from_utf8(&protein.atoms[i].name).unwrap().trim())
        .collect();
    assert_eq!(names, vec!["N", "CA", "C", "O", "CB", "HA"]);
}

#[test]
fn canonical_ordering_drops_residue_missing_backbone() {
    // GLY has only N, CA, C (missing O); must be dropped.
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("CA", Element::C, 1.0, 0.0, 0.0),
        atom_at("C", Element::C, 2.0, 0.0, 0.0),
        atom_at("O", Element::O, 3.0, 0.0, 0.0),
        atom_at("N", Element::N, 4.0, 0.0, 0.0),
        atom_at("CA", Element::C, 5.0, 0.0, 0.0),
        atom_at("C", Element::C, 6.0, 0.0, 0.0),
    ];
    let residues = vec![residue("ALA", 1, 0..4), residue("GLY", 2, 4..7)];
    let protein = build_protein(atoms, residues);
    assert_eq!(protein.residues.len(), 1);
    assert_eq!(&protein.residues[0].name, b"ALA");
    assert_eq!(protein.atoms.len(), 7);
}

#[test]
fn gly_backbone_bonds_are_three_sidechain_empty() {
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("CA", Element::C, 1.5, 0.0, 0.0),
        atom_at("C", Element::C, 2.5, 0.0, 0.0),
        atom_at("O", Element::O, 2.5, 1.0, 0.0),
    ];
    let residues = vec![residue("GLY", 1, 0..4)];
    let protein = build_protein(atoms, residues);
    assert_eq!(protein.backbone_bonds().count(), 3);
    assert_eq!(protein.sidechain_bonds().count(), 0);
}

#[test]
fn ala_sidechain_bonds_is_ca_cb_only() {
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("CA", Element::C, 1.5, 0.0, 0.0),
        atom_at("C", Element::C, 2.5, 0.0, 0.0),
        atom_at("O", Element::O, 2.5, 1.0, 0.0),
        atom_at("CB", Element::C, 1.5, -1.5, 0.0),
    ];
    let residues = vec![residue("ALA", 1, 0..5)];
    let protein = build_protein(atoms, residues);
    let sidechain: Vec<_> = protein.sidechain_bonds().collect();
    assert_eq!(sidechain.len(), 1);
    let b = sidechain[0];
    let a_name = std::str::from_utf8(&protein.atoms[b.a.index as usize].name)
        .unwrap()
        .trim();
    let c_name = std::str::from_utf8(&protein.atoms[b.b.index as usize].name)
        .unwrap()
        .trim();
    let pair = (a_name, c_name);
    assert!(
        pair == ("CA", "CB") || pair == ("CB", "CA"),
        "expected CA-CB sidechain anchor, got {pair:?}"
    );
}

#[test]
fn peptide_bond_connects_consecutive_residues() {
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("CA", Element::C, 1.5, 0.0, 0.0),
        atom_at("C", Element::C, 2.5, 0.0, 0.0),
        atom_at("O", Element::O, 2.5, 1.0, 0.0),
        atom_at("N", Element::N, 3.8, 0.0, 0.0),
        atom_at("CA", Element::C, 5.0, 0.0, 0.0),
        atom_at("C", Element::C, 6.0, 0.0, 0.0),
        atom_at("O", Element::O, 6.0, 1.0, 0.0),
    ];
    let residues = vec![residue("ALA", 1, 0..4), residue("ALA", 2, 4..8)];
    let protein = build_protein(atoms, residues);
    // 3 backbone * 2 residues + 1 peptide = 7 backbone bonds total.
    assert_eq!(
        protein.backbone_bonds().count(),
        7,
        "3 per residue + 1 peptide"
    );
}

#[test]
fn dropped_residue_emits_log_warning() {
    testing_logger::setup();
    // ALA residue missing CA; must be dropped and logged.
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("C", Element::C, 1.5, 0.0, 0.0),
        atom_at("O", Element::O, 2.5, 1.0, 0.0),
    ];
    let residues = vec![residue("ALA", 7, 0..3)];
    let _ = build_protein(atoms, residues);
    testing_logger::validate(|captured_logs| {
        let warn_bodies: Vec<&str> = captured_logs
            .iter()
            .filter(|l| l.level == log::Level::Warn)
            .map(|l| l.body.as_str())
            .collect();
        assert!(
            !warn_bodies.is_empty(),
            "expected at least one warn-level log entry"
        );
        assert!(
            warn_bodies.iter().any(|b| b.contains("dropping residue")),
            "warning should mention dropping a residue; got {warn_bodies:?}"
        );
    });
}

#[test]
fn peptide_bond_skipped_across_segment_break() {
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("CA", Element::C, 1.5, 0.0, 0.0),
        atom_at("C", Element::C, 2.5, 0.0, 0.0),
        atom_at("O", Element::O, 2.5, 1.0, 0.0),
        atom_at("N", Element::N, 20.0, 0.0, 0.0),
        atom_at("CA", Element::C, 21.5, 0.0, 0.0),
        atom_at("C", Element::C, 22.5, 0.0, 0.0),
        atom_at("O", Element::O, 22.5, 1.0, 0.0),
    ];
    let residues = vec![residue("ALA", 1, 0..4), residue("ALA", 2, 4..8)];
    let protein = build_protein(atoms, residues);
    // No peptide bond; just 3 backbone per residue = 6.
    assert_eq!(protein.backbone_bonds().count(), 6);
}
