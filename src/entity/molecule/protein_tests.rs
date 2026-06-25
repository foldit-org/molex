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
    ProteinEntity::new(id, atoms, residues, "A".to_owned())
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

// -- Construction completion vs pure-assembly contrast --

/// Trimmed atom names referenced by a residue's range, in order.
fn residue_atom_names(atoms: &[Atom], r: &Residue) -> Vec<String> {
    r.atom_range
        .clone()
        .map(|i| {
            std::str::from_utf8(trimmed_atom_name(&atoms[i].name))
                .unwrap()
                .to_owned()
        })
        .collect()
}

/// One backbone-only ALA residue (N, CA, C, O; CB absent). Positioned by
/// a rigid transform of the ideal backbone so the completion rigid fit
/// can place CB. Returns the atoms and the matching residue.
fn backbone_only_alanine() -> (Vec<Atom>, Vec<Residue>) {
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("CA", Element::C, 1.46, 0.0, 0.0),
        atom_at("C", Element::C, 2.0, 1.4, 0.0),
        atom_at("O", Element::O, 1.4, 2.5, 0.0),
    ];
    let residues = vec![residue("ALA", 1, 0..4)];
    (atoms, residues)
}

#[test]
fn new_does_not_fabricate_missing_atoms() {
    // Pure construction assembles the atoms exactly as given: a
    // sidechain-incomplete residue gains nothing, so the atom count is
    // unchanged.
    let (atoms, residues) = backbone_only_alanine();
    let input_count = atoms.len();
    let id = EntityIdAllocator::new().allocate();
    let protein = ProteinEntity::new(id, atoms, residues, "A".to_owned());
    assert_eq!(
        protein.residues.len(),
        1,
        "ALA must survive (backbone intact)"
    );
    assert_eq!(
        protein.atoms.len(),
        input_count,
        "pure construction must fabricate no atoms"
    );
    let names = residue_atom_names(&protein.atoms, &protein.residues[0]);
    assert!(
        !names.iter().any(|n| n == "CB"),
        "CB must not be fabricated on the pure path; got {names:?}"
    );
}

#[test]
fn new_normalized_completes_missing_sidechain_atoms() {
    // The opt-in completing constructor runs heavy-atom completion at
    // construction, so the same backbone-only ALA gains its CB.
    let (atoms, residues) = backbone_only_alanine();
    let id = EntityIdAllocator::new().allocate();
    let protein = ProteinEntity::new_normalized(
        id,
        atoms,
        residues,
        "A".to_owned(),
        Completion::Heavy,
    );
    assert_eq!(protein.residues.len(), 1, "ALA must survive completion");
    let names = residue_atom_names(&protein.atoms, &protein.residues[0]);
    assert!(
        names.iter().any(|n| n == "CB"),
        "CB must be fabricated by new_normalized; got {names:?}"
    );
}

/// ALA with N, C, O present but the backbone CA absent. Backbone-
/// incomplete (canonicalize needs N, CA, C, O, so it drops this on the
/// pure path), yet anchorable: three matched atoms (N, C, O), and CA is
/// in-scope for completion (not a terminus/leaving atom), so completion
/// can fabricate CA and rescue the residue.
fn backbone_incomplete_anchorable_alanine() -> (Vec<Atom>, Vec<Residue>) {
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("C", Element::C, 2.0, 1.4, 0.0),
        atom_at("O", Element::O, 1.4, 2.5, 0.0),
    ];
    let residues = vec![residue("ALA", 1, 0..3)];
    (atoms, residues)
}

#[test]
fn new_drops_backbone_incomplete_residue() {
    // Pure construction does not complete, so a residue missing a backbone
    // atom (here CA) cannot be rescued and is dropped by canonicalize. The
    // load-bearing drop stays intact on the pure path.
    let (atoms, residues) = backbone_incomplete_anchorable_alanine();
    let id = EntityIdAllocator::new().allocate();
    let protein = ProteinEntity::new(id, atoms, residues, "A".to_owned());
    assert_eq!(
        protein.residues.len(),
        0,
        "backbone-incomplete residue must drop on the pure path"
    );
}

#[test]
fn new_normalized_rescues_backbone_incomplete_residue() {
    // Completion runs before the canonicalize drop, so the same residue
    // (N, C, O present; CA absent) has its backbone CA fabricated from the
    // template and survives, whereas pure `new` would drop it.
    let (atoms, residues) = backbone_incomplete_anchorable_alanine();
    let id = EntityIdAllocator::new().allocate();
    let protein = ProteinEntity::new_normalized(
        id,
        atoms,
        residues,
        "A".to_owned(),
        Completion::Heavy,
    );
    assert_eq!(
        protein.residues.len(),
        1,
        "completion must rescue the backbone-incomplete residue"
    );
    let names = residue_atom_names(&protein.atoms, &protein.residues[0]);
    assert!(
        names.iter().any(|n| n == "CA"),
        "the missing backbone CA must be fabricated; got {names:?}"
    );
}

#[test]
fn normalize_completes_surviving_residue_sidechain() {
    // A pure-built ALA whose backbone is intact but sidechain (CB) is
    // absent survives construction. normalize() re-runs heavy completion
    // on that survivor and fills in the CB.
    let (atoms, residues) = backbone_only_alanine();
    let id = EntityIdAllocator::new().allocate();
    let pure = ProteinEntity::new(id, atoms, residues, "A".to_owned());
    assert_eq!(pure.residues.len(), 1, "backbone-intact ALA must survive");
    let pure_names = residue_atom_names(&pure.atoms, &pure.residues[0]);
    assert!(
        !pure_names.iter().any(|n| n == "CB"),
        "pure construction must not fabricate CB; got {pure_names:?}"
    );

    let completed = pure.normalize();
    let names = residue_atom_names(&completed.atoms, &completed.residues[0]);
    assert!(
        names.iter().any(|n| n == "CB"),
        "normalize must fabricate the missing CB on the survivor; got \
         {names:?}"
    );
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

// -- phi/psi dihedrals --

/// A geometry with a known +90-degree dihedral about the p1->p2 bond: p0 and
/// p3 sit on perpendicular axes either side of a z-aligned central bond, so
/// the two bond half-planes are a quarter turn apart in the IUPAC-sign sense
/// (the convention validated against 1UBQ in the oracle gate).
#[test]
fn dihedral_known_ninety_degrees() {
    let angle = dihedral_deg(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
    );
    assert!((angle - 90.0).abs() < 1e-3, "expected +90 deg, got {angle}");
}

/// Mirroring p3 across the dihedral plane flips the sign.
#[test]
fn dihedral_sign_flips_with_mirror() {
    let angle = dihedral_deg(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, -1.0, 1.0),
    );
    assert!((angle + 90.0).abs() < 1e-3, "expected -90 deg, got {angle}");
}

#[test]
fn phi_psi_len_equals_residue_count_and_termini_none() {
    let protein = two_residue_protein_with_break();
    let pp = protein.phi_psi();
    assert_eq!(pp.len(), protein.residues.len());
    // First residue has no preceding C -> phi None.
    assert!(pp[0].0.is_none(), "first phi must be None");
    // Last residue has no following N -> psi None.
    assert!(pp[1].1.is_none(), "last psi must be None");
    // The two residues are separated by a segment break, so the bond-
    // crossing angles (psi of res 0, phi of res 1) are also None.
    assert!(pp[0].1.is_none(), "psi across a break must be None");
    assert!(pp[1].0.is_none(), "phi across a break must be None");
}

/// Three residues placed at ideal trans-peptide backbone coordinates so the
/// middle residue has a computable (phi, psi). The coordinates are the
/// canonical N/CA/C/O positions of three consecutive residues of an extended
/// strand, giving phi/psi near the beta region.
#[test]
fn phi_psi_middle_residue_is_computable() {
    // Backbone atoms for three peptide-bonded residues. Positions are taken
    // from a real extended-strand backbone (close C->N spacing so no segment
    // break falls between them).
    let atoms = vec![
        // residue 1
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("CA", Element::C, 1.458, 0.0, 0.0),
        atom_at("C", Element::C, 2.009, 1.420, 0.0),
        atom_at("O", Element::O, 1.251, 2.390, 0.0),
        // residue 2
        atom_at("N", Element::N, 3.332, 1.512, 0.0),
        atom_at("CA", Element::C, 3.988, 2.806, 0.0),
        atom_at("C", Element::C, 5.486, 2.651, 0.183),
        atom_at("O", Element::O, 5.983, 1.531, 0.300),
        // residue 3
        atom_at("N", Element::N, 6.201, 3.770, 0.197),
        atom_at("CA", Element::C, 7.649, 3.770, 0.351),
        atom_at("C", Element::C, 8.121, 5.209, 0.500),
        atom_at("O", Element::O, 7.369, 6.180, 0.460),
    ];
    let residues = vec![
        residue("ALA", 1, 0..4),
        residue("ALA", 2, 4..8),
        residue("ALA", 3, 8..12),
    ];
    let protein = build_protein(atoms, residues);
    assert!(
        protein.segment_breaks.is_empty(),
        "residues must be peptide-bonded (no break)"
    );
    let pp = protein.phi_psi();
    assert_eq!(pp.len(), 3);
    // Termini: phi of first None, psi of last None.
    assert!(pp[0].0.is_none());
    assert!(pp[2].1.is_none());
    // Middle residue has both a preceding C and a following N -> both Some,
    // and they are finite, in-range angles.
    let (phi, psi) = pp[1];
    let phi = phi.unwrap();
    let psi = psi.unwrap();
    assert!(phi.is_finite() && phi > -180.0 && phi <= 180.0, "phi={phi}");
    assert!(psi.is_finite() && psi > -180.0 && psi <= 180.0, "psi={psi}");
    // First residue still has a psi (bond to residue 2 present); last has phi.
    assert!(pp[0].1.is_some(), "first psi present (bonded to res 2)");
    assert!(pp[2].0.is_some(), "last phi present (bonded from res 2)");
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
    // The four backbone atoms come first, then CB, then the parsed HA.
    // Completion is heavy-only by default and this residue is already
    // heavy-complete, so no atoms are fabricated.
    assert_eq!(names, vec!["N", "CA", "C", "O", "CB", "HA"]);
}

#[test]
fn canonical_ordering_drops_residue_missing_backbone() {
    // GLY has only N, CA, C (missing O); it cannot have its carbonyl
    // oxygen rebuilt (O is a C-terminal template atom, out of scope for
    // completion) so it must still be dropped. The complete ALA is kept.
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
    // 3 backbone * 2 residues + 1 peptide = 7 bonds total.
    assert_eq!(protein.bonds.len(), 7, "3 per residue + 1 peptide");
}

#[test]
fn dropped_residue_emits_log_warning() {
    testing_logger::setup();
    // ALA residue with only N and O: missing CA and C, and with just two
    // present atoms it is below the three-atom anchor floor, so the
    // completion pass cannot rebuild the missing backbone. The residue
    // must be dropped and logged.
    let atoms = vec![
        atom_at("N", Element::N, 0.0, 0.0, 0.0),
        atom_at("O", Element::O, 2.5, 1.0, 0.0),
    ];
    let residues = vec![residue("ALA", 7, 0..2)];
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
    assert_eq!(protein.bonds.len(), 6);
}
