//! Tests for the missing-atom completion pass.
//!
//! Split into a sibling file to mirror the `protein_tests.rs` pattern and
//! keep the primary module focused on the algorithm.

#![allow(clippy::unwrap_used, clippy::float_cmp)]

use std::collections::BTreeSet;

use super::*;
use crate::chemistry::amino_acids::AminoAcid;
use crate::element::Element;
use crate::entity::molecule::id::EntityIdAllocator;
use crate::entity::molecule::nucleic_acid::NAEntity;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::MoleculeType;

fn atom_at(name: &str, element: Element, p: Vec3) -> Atom {
    Atom {
        position: p,
        occupancy: 1.0,
        b_factor: 0.0,
        element,
        name: pad_atom_name(name),
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

fn residue(name: &str, range: std::ops::Range<usize>) -> Residue {
    Residue {
        name: res_bytes(name),
        label_seq_id: 1,
        auth_seq_id: None,
        auth_comp_id: None,
        ins_code: None,
        atom_range: range,
        variants: Vec::new(),
    }
}

/// Position of the (first) atom with the given trimmed name within a
/// residue's range. Unwraps (panics) if absent.
fn position_by_name(atoms: &[Atom], r: &Residue, name: &str) -> Vec3 {
    let idx = r
        .atom_range
        .clone()
        .find(|&i| trimmed_atom_name(&atoms[i].name) == name.as_bytes())
        .unwrap();
    atoms[idx].position
}

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

/// The all-atom completion target set for an amino acid built as a single
/// residue, i.e. both chain termini. Interior body atoms plus the
/// terminus-only atoms admitted at a terminus (the C-terminal carboxylate
/// OXT/HXT and the extra N-terminal proton); leaving atoms that are not a
/// terminus extra stay excluded.
fn in_scope_template_names(aa: AminoAcid) -> BTreeSet<String> {
    aa.template()
        .atoms
        .iter()
        .filter(|a| {
            let interior =
                !a.is_leaving && !a.is_n_terminal && !a.is_c_terminal;
            interior || a.is_c_terminal || a.is_n_terminal
        })
        .map(|a| a.name.as_str().to_owned())
        .collect()
}

/// A fixed non-trivial rigid transform (90 deg about Z, then a shift) to
/// place a residue away from the dictionary's local frame, so a correct
/// rigid fit must recover real geometry rather than coincide with the
/// identity.
fn sample_transform(p: Vec3) -> Vec3 {
    let rot = Mat3::from_cols(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    rot * p + Vec3::new(10.0, -5.0, 3.0)
}

/// Build a parsed atom whose position is the named template atom's ideal
/// run through [`sample_transform`]. Panics if the name is absent.
fn placed_from_template(aa: AminoAcid, name: &str, element: Element) -> Atom {
    let t = aa
        .template()
        .atom(AtomName::from_bytes(name.as_bytes()))
        .unwrap();
    atom_at(name, element, sample_transform(t.ideal))
}

/// Heavy-atom subset of [`in_scope_template_names`]: the same single-residue
/// (both-termini) set with hydrogens removed, i.e. what the default
/// heavy-only path fabricates. The C-terminal carboxylate OXT survives
/// (heavy); HXT and the N-terminal protons drop as hydrogens.
fn in_scope_heavy_template_names(aa: AminoAcid) -> BTreeSet<String> {
    aa.template()
        .atoms
        .iter()
        .filter(|a| {
            let interior =
                !a.is_leaving && !a.is_n_terminal && !a.is_c_terminal;
            let kept = interior || a.is_c_terminal || a.is_n_terminal;
            kept && a.element != Element::H
        })
        .map(|a| a.name.as_str().to_owned())
        .collect()
}

/// Build a single-residue protein from the given parsed atoms, running
/// heavy-atom completion (this module tests the completion pass, so it
/// builds through the completing constructor).
fn build_protein(name: &str, atoms: Vec<Atom>) -> ProteinEntity {
    let n = atoms.len();
    let id = EntityIdAllocator::new().allocate();
    ProteinEntity::new_normalized(
        id,
        atoms,
        vec![residue(name, 0..n)],
        b'A',
        None,
        CompletionMode::HeavyOnly,
    )
}

#[test]
fn backbone_only_alanine_builds_cb_near_ca() {
    // Only the four backbone atoms present, positioned by a known rigid
    // transform of the ideal backbone; CB and the hydrogens are absent
    // and must be placed by the rigid fit at chemically sane positions.
    let atoms = vec![
        placed_from_template(AminoAcid::Ala, "N", Element::N),
        placed_from_template(AminoAcid::Ala, "CA", Element::C),
        placed_from_template(AminoAcid::Ala, "C", Element::C),
        placed_from_template(AminoAcid::Ala, "O", Element::O),
    ];
    let protein = build_protein("ALA", atoms);
    assert_eq!(protein.residues.len(), 1, "ALA must survive completion");
    let r = &protein.residues[0];
    let names = residue_atom_names(&protein.atoms, r);
    assert!(names.contains(&"CB".to_owned()), "CB should be built");

    let ca = r.atom_range.start + 1; // canonical CA offset
    let ca_pos = protein.atoms[ca].position;
    let cb_idx = r
        .atom_range
        .clone()
        .find(|&i| trimmed_atom_name(&protein.atoms[i].name) == b"CB")
        .unwrap();
    let cb_pos = protein.atoms[cb_idx].position;
    assert!(cb_pos.is_finite(), "CB position must be finite");
    let d = (cb_pos - ca_pos).length();
    assert!(
        (1.3..1.7).contains(&d),
        "CA-CB distance should be ~1.5 A, got {d}"
    );
}

#[test]
fn heavy_only_alanine_builds_cb_but_no_hydrogens() {
    // Backbone-only ALA: CB (heavy) and HB1/HB2/HB3 plus HA (hydrogen)
    // are all in-scope and absent. The default heavy-only path must build
    // the heavy CB and fabricate no hydrogens at all.
    let atoms = vec![
        placed_from_template(AminoAcid::Ala, "N", Element::N),
        placed_from_template(AminoAcid::Ala, "CA", Element::C),
        placed_from_template(AminoAcid::Ala, "C", Element::C),
        placed_from_template(AminoAcid::Ala, "O", Element::O),
    ];
    let (out_atoms, out_res) = complete_protein_residues(
        &atoms,
        &[residue("ALA", 0..4)],
        CompletionMode::HeavyOnly,
    );
    let r = &out_res[0];
    let names: BTreeSet<String> =
        residue_atom_names(&out_atoms, r).into_iter().collect();

    // The missing heavy atom was built.
    assert!(names.contains("CB"), "heavy-only must still build CB");
    // The completed set is the preserved backbone plus the heavy in-scope
    // set (no hydrogens), and no atom in the residue is a hydrogen.
    let want = {
        let mut s = in_scope_heavy_template_names(AminoAcid::Ala);
        for kept in ["N", "C", "O"] {
            let _ = s.insert(kept.to_owned());
        }
        s
    };
    assert_eq!(
        names, want,
        "heavy-only completion should add only the heavy in-scope atoms"
    );
    for i in r.atom_range.clone() {
        assert_ne!(
            out_atoms[i].element,
            Element::H,
            "heavy-only must fabricate no hydrogens"
        );
    }
}

#[test]
fn all_atom_alanine_builds_cb_and_hydrogens() {
    // Same backbone-only ALA, but through the all-atom entry: CB and the
    // sidechain hydrogens HB1/HB2/HB3 (and HA) are all fabricated.
    let atoms = vec![
        placed_from_template(AminoAcid::Ala, "N", Element::N),
        placed_from_template(AminoAcid::Ala, "CA", Element::C),
        placed_from_template(AminoAcid::Ala, "C", Element::C),
        placed_from_template(AminoAcid::Ala, "O", Element::O),
    ];
    let (out_atoms, out_res) = complete_protein_residues(
        &atoms,
        &[residue("ALA", 0..4)],
        CompletionMode::AllAtom,
    );
    let r = &out_res[0];
    let names: BTreeSet<String> =
        residue_atom_names(&out_atoms, r).into_iter().collect();

    assert!(names.contains("CB"), "all-atom must build CB");
    for h in ["HB1", "HB2", "HB3"] {
        assert!(names.contains(h), "all-atom must build hydrogen {h}");
    }
    // The all-atom completion is the preserved backbone plus the full
    // in-scope set (heavy atoms and hydrogens).
    let want = {
        let mut s = in_scope_template_names(AminoAcid::Ala);
        for kept in ["N", "C", "O"] {
            let _ = s.insert(kept.to_owned());
        }
        s
    };
    assert_eq!(
        names, want,
        "all-atom completion should add the full in-scope set"
    );
}

#[test]
fn truncated_leucine_builds_to_full_in_scope_set() {
    // LEU missing the two distal CD carbons and all hydrogens; the six
    // present atoms (including the terminal-flagged N/C/O that real
    // residues carry) are positioned by the known rigid transform.
    // Exercised through the all-atom entry, which fabricates hydrogens
    // too, so the result is the complete in-scope set.
    let atoms = vec![
        placed_from_template(AminoAcid::Leu, "N", Element::N),
        placed_from_template(AminoAcid::Leu, "CA", Element::C),
        placed_from_template(AminoAcid::Leu, "C", Element::C),
        placed_from_template(AminoAcid::Leu, "O", Element::O),
        placed_from_template(AminoAcid::Leu, "CB", Element::C),
        placed_from_template(AminoAcid::Leu, "CG", Element::C),
    ];
    let (out_atoms, out_res) = complete_protein_residues(
        &atoms,
        &[residue("LEU", 0..6)],
        CompletionMode::AllAtom,
    );
    let r = &out_res[0];
    let got: BTreeSet<String> =
        residue_atom_names(&out_atoms, r).into_iter().collect();
    // Present atoms are preserved (including the terminal-flagged N/C/O),
    // and every in-scope template atom is now present.
    let want = {
        let mut s = in_scope_template_names(AminoAcid::Leu);
        for kept in ["N", "C", "O"] {
            let _ = s.insert(kept.to_owned());
        }
        s
    };
    assert_eq!(
        got, want,
        "completed LEU should hold present atoms plus the in-scope set"
    );
    // The two distal carbons specifically must be built.
    assert!(got.contains("CD1") && got.contains("CD2"));
    for p in r.atom_range.clone() {
        assert!(out_atoms[p].position.is_finite());
    }
}

#[test]
fn already_complete_residue_gains_no_atoms() {
    // Provide every in-scope ALA heavy atom plus the input hydrogens a
    // protonated parse carries; heavy-only completion must fabricate
    // nothing, must not duplicate, and must STRIP the input hydrogens
    // (the file-ingest entity is heavy-only).
    let mut atoms = vec![
        atom_at("N", Element::N, Vec3::new(0.0, 0.0, 0.0)),
        atom_at("CA", Element::C, Vec3::new(1.46, 0.0, 0.0)),
        atom_at("C", Element::C, Vec3::new(2.0, 1.4, 0.0)),
        atom_at("O", Element::O, Vec3::new(1.3, 2.4, 0.0)),
        atom_at("CB", Element::C, Vec3::new(2.0, -1.2, 0.6)),
        atom_at("HA", Element::H, Vec3::new(0.6, -0.7, 0.0)),
        atom_at("HB1", Element::H, Vec3::new(2.9, -1.1, 0.0)),
        atom_at("HB2", Element::H, Vec3::new(1.6, -2.1, 0.3)),
        atom_at("HB3", Element::H, Vec3::new(2.1, -1.3, 1.7)),
    ];
    // Add the terminus atoms a complete chain end carries: the N-terminal
    // proton and the C-terminal carboxylate OXT (this single residue is
    // both chain termini, so heavy-only completion would otherwise build
    // OXT).
    atoms.push(atom_at("H", Element::H, Vec3::new(-0.5, 0.5, 0.0)));
    atoms.push(atom_at("OXT", Element::O, Vec3::new(3.3, 1.4, 0.0)));
    let protein = build_protein("ALA", atoms);
    let r = &protein.residues[0];
    // The six heavy atoms survive (N, CA, C, O, CB, OXT); every input
    // hydrogen is dropped under heavy-only.
    let names = residue_atom_names(&protein.atoms, r);
    let kept: BTreeSet<&String> = names.iter().collect();
    let expected: BTreeSet<String> = ["N", "CA", "C", "O", "CB", "OXT"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    assert_eq!(
        kept,
        expected.iter().collect(),
        "heavy-only keeps exactly the heavy atoms, no fabrication"
    );
    assert!(
        r.atom_range
            .clone()
            .all(|i| protein.atoms[i].element != Element::H),
        "no hydrogen survives heavy-only ingest"
    );
    // No duplicated names within the residue.
    assert_eq!(kept.len(), names.len(), "no duplicate atom names");
}

#[test]
fn protonated_input_strips_h_heavy_only_keeps_h_all_atom() {
    // A protonated ALA parse (backbone + CB + the variously-named input
    // protons): under HeavyOnly the canonical entity must carry NO
    // hydrogen, and under AllAtom the same input keeps/places them.
    let protonated = || {
        vec![
            atom_at("N", Element::N, Vec3::new(0.0, 0.0, 0.0)),
            atom_at("CA", Element::C, Vec3::new(1.46, 0.0, 0.0)),
            atom_at("C", Element::C, Vec3::new(2.0, 1.4, 0.0)),
            atom_at("O", Element::O, Vec3::new(1.3, 2.4, 0.0)),
            atom_at("CB", Element::C, Vec3::new(2.0, -1.2, 0.6)),
            atom_at("HA", Element::H, Vec3::new(0.6, -0.7, 0.0)),
            atom_at("HB1", Element::H, Vec3::new(2.9, -1.1, 0.0)),
            // Numeric-prefix proton (e.g. `1H`): the parser's `H`-first /
            // element-H rule classifies it as a hydrogen all the same.
            atom_at("1H", Element::H, Vec3::new(-0.5, 0.5, 0.0)),
        ]
    };
    let n = protonated().len();
    let id = EntityIdAllocator::new().allocate();

    let heavy = ProteinEntity::new_normalized(
        id,
        protonated(),
        vec![residue("ALA", 0..n)],
        b'A',
        None,
        CompletionMode::HeavyOnly,
    );
    let r = &heavy.residues[0];
    assert!(
        r.atom_range
            .clone()
            .all(|i| heavy.atoms[i].element != Element::H),
        "heavy-only ingest strips every input hydrogen"
    );

    let all = ProteinEntity::new_normalized(
        id,
        protonated(),
        vec![residue("ALA", 0..n)],
        b'A',
        None,
        CompletionMode::AllAtom,
    );
    let r = &all.residues[0];
    assert!(
        r.atom_range
            .clone()
            .any(|i| all.atoms[i].element == Element::H),
        "all-atom keeps/places hydrogens for the same input"
    );
}

#[test]
fn sparse_residue_passes_through_unchanged() {
    // Two atoms only: fewer than the three needed to anchor a rigid fit.
    // The residue still has N, CA, C, O? No: give just N and CA so it is
    // both sparse and (intentionally) backbone-incomplete; completion
    // must fabricate nothing.
    let atoms = vec![
        atom_at("N", Element::N, Vec3::new(0.0, 0.0, 0.0)),
        atom_at("CA", Element::C, Vec3::new(1.46, 0.0, 0.0)),
    ];
    let (out_atoms, out_res) = complete_protein_residues(
        &atoms,
        &[residue("ALA", 0..2)],
        CompletionMode::HeavyOnly,
    );
    assert_eq!(out_atoms.len(), 2, "no atoms fabricated for sparse residue");
    assert_eq!(out_res[0].atom_range, 0..2);
    let names = residue_atom_names(&out_atoms, &out_res[0]);
    assert_eq!(names, vec!["N".to_owned(), "CA".to_owned()]);
}

#[test]
fn backbone_missing_residue_still_drops() {
    // ALA with only N and C present: missing CA and O, and only two
    // matched atoms, so completion cannot anchor. The residue must still
    // be dropped by canonicalize.
    let atoms = vec![
        atom_at("N", Element::N, Vec3::new(0.0, 0.0, 0.0)),
        atom_at("C", Element::C, Vec3::new(2.0, 1.4, 0.0)),
    ];
    let protein = build_protein("ALA", atoms);
    assert_eq!(
        protein.residues.len(),
        0,
        "backbone-incomplete, unanchorable residue must drop"
    );
}

#[test]
fn c_terminal_oxt_only_on_last_residue() {
    // Three backbone-only ALA residues in N->C order. Heavy-only completion
    // must fabricate the C-terminal carboxylate OXT on the last residue
    // only; the two interior residues must not gain OXT.
    let backbone = |offset: f32| {
        let shift = Vec3::new(offset, 0.0, 0.0);
        vec![
            atom_at("N", Element::N, sample_transform(Vec3::ZERO) + shift),
            atom_at(
                "CA",
                Element::C,
                sample_transform(Vec3::new(1.46, 0.0, 0.0)) + shift,
            ),
            atom_at(
                "C",
                Element::C,
                sample_transform(Vec3::new(2.0, 1.4, 0.0)) + shift,
            ),
            atom_at(
                "O",
                Element::O,
                sample_transform(Vec3::new(1.3, 2.4, 0.0)) + shift,
            ),
        ]
    };
    let mut atoms = Vec::new();
    let mut residues = Vec::new();
    for (i, offset) in [0.0f32, 20.0, 40.0].into_iter().enumerate() {
        let start = atoms.len();
        atoms.extend(backbone(offset));
        residues.push(Residue {
            name: res_bytes("ALA"),
            label_seq_id: i32::try_from(i).unwrap() + 1,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: start..atoms.len(),
            variants: Vec::new(),
        });
    }
    let (out_atoms, out_res) =
        complete_protein_residues(&atoms, &residues, CompletionMode::HeavyOnly);
    let has_oxt = |r: &Residue| {
        residue_atom_names(&out_atoms, r).iter().any(|n| n == "OXT")
    };
    assert!(
        !has_oxt(&out_res[0]),
        "interior residue 0 must not gain OXT"
    );
    assert!(
        !has_oxt(&out_res[1]),
        "interior residue 1 must not gain OXT"
    );
    assert!(has_oxt(&out_res[2]), "C-terminal residue must gain OXT");
}

#[test]
fn na_completion_fills_dna_base_and_keeps_present_atoms() {
    // A DA residue carrying its backbone + a couple of base atoms; the
    // completion pass should add the remaining in-scope DA atoms while
    // leaving the present ones intact, and the entity must survive.
    let atoms = vec![
        atom_at("P", Element::P, Vec3::new(0.0, 0.0, 0.0)),
        atom_at("O5'", Element::O, Vec3::new(1.5, 0.0, 0.0)),
        atom_at("C5'", Element::C, Vec3::new(2.5, 1.0, 0.0)),
        atom_at("C4'", Element::C, Vec3::new(3.5, 0.5, 0.5)),
        atom_at("C3'", Element::C, Vec3::new(4.5, 1.5, 0.0)),
        atom_at("O3'", Element::O, Vec3::new(5.5, 1.0, 0.5)),
        atom_at("C1'", Element::C, Vec3::new(3.8, -0.8, 0.0)),
        atom_at("O4'", Element::O, Vec3::new(3.0, -0.5, 1.0)),
    ];
    let n = atoms.len();
    let id = EntityIdAllocator::new().allocate();
    let na = NAEntity::new_normalized(
        id,
        MoleculeType::DNA,
        atoms,
        vec![residue("DA", 0..n)],
        b'A',
        None,
        CompletionMode::HeavyOnly,
    );
    assert_eq!(na.residues.len(), 1, "DA must survive completion");
    let r = &na.residues[0];
    let names: BTreeSet<String> =
        residue_atom_names(&na.atoms, r).into_iter().collect();
    // Present sugar/phosphate atoms preserved.
    for kept in ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "O4'"] {
        assert!(names.contains(kept), "lost present atom {kept}");
    }
    // Base atoms newly built.
    for built in ["N9", "C8", "N7", "C4", "N6"] {
        assert!(names.contains(built), "DA base atom {built} not built");
    }
}

#[test]
fn na_all_atom_fills_dna_base_and_hydrogens() {
    // Same DA sugar/phosphate core as the heavy NA test, run through the
    // all-atom NA entry. It must build the base heavy atoms AND template
    // hydrogens, exercising the fourth completion entry point.
    let atoms = vec![
        atom_at("P", Element::P, Vec3::new(0.0, 0.0, 0.0)),
        atom_at("O5'", Element::O, Vec3::new(1.5, 0.0, 0.0)),
        atom_at("C5'", Element::C, Vec3::new(2.5, 1.0, 0.0)),
        atom_at("C4'", Element::C, Vec3::new(3.5, 0.5, 0.5)),
        atom_at("C3'", Element::C, Vec3::new(4.5, 1.5, 0.0)),
        atom_at("O3'", Element::O, Vec3::new(5.5, 1.0, 0.5)),
        atom_at("C1'", Element::C, Vec3::new(3.8, -0.8, 0.0)),
        atom_at("O4'", Element::O, Vec3::new(3.0, -0.5, 1.0)),
    ];
    let n = atoms.len();
    let (out_atoms, out_res) = complete_na_residues(
        &atoms,
        &[residue("DA", 0..n)],
        CompletionMode::AllAtom,
        MoleculeType::DNA,
    );
    let r = &out_res[0];
    let names: BTreeSet<String> =
        residue_atom_names(&out_atoms, r).into_iter().collect();
    // Heavy base atoms present.
    for built in ["N9", "C8", "N7", "C4", "N6"] {
        assert!(names.contains(built), "DA base atom {built} not built");
    }
    // At least one template hydrogen was fabricated by the all-atom path.
    let has_h = r
        .atom_range
        .clone()
        .any(|i| out_atoms[i].element == Element::H);
    assert!(has_h, "all-atom NA completion must fabricate hydrogens");
}

/// Build a DA residue from real template ideal geometry, keeping every
/// in-scope heavy atom whose name passes `keep`. Used to seed a free
/// 5'-OH nucleotide (drop the phosphate group) at coordinates a correct
/// rigid fit can recover.
fn da_heavy_atoms(keep: impl Fn(&str) -> bool) -> Vec<Atom> {
    let template = Nucleotide::A.template(MoleculeType::DNA).unwrap();
    template
        .atoms
        .iter()
        .filter(|a| a.element != Element::H && !a.is_leaving)
        .filter(|a| keep(a.name.as_str()))
        .map(|a| atom_at(a.name.as_str(), a.element, a.ideal))
        .collect()
}

#[test]
fn na_five_prime_oh_fabricates_no_phosphate() {
    // A free 5'-OH DA: real sugar/backbone + base, but no P/OP1/OP2. As
    // the chain-first residue, the absent 5'-phosphate group must NOT be
    // fabricated (additive policy: never forge a terminus the structure
    // lacks). Base atoms missing from the parse still complete normally.
    let phosphate = |n: &str| matches!(n, "P" | "OP1" | "OP2" | "OP3");
    let atoms = da_heavy_atoms(|n| !phosphate(n) && n != "N6");
    let n = atoms.len();
    let (out_atoms, out_res) = complete_na_residues(
        &atoms,
        &[residue("DA", 0..n)],
        CompletionMode::HeavyOnly,
        MoleculeType::DNA,
    );
    let r = &out_res[0];
    let names: BTreeSet<String> =
        residue_atom_names(&out_atoms, r).into_iter().collect();

    for forged in ["P", "OP1", "OP2", "OP3"] {
        assert!(
            !names.contains(forged),
            "5'-OH residue must not fabricate {forged}; got {names:?}"
        );
    }
    // The interior base atom that was absent still completes.
    assert!(names.contains("N6"), "absent base atom N6 should complete");
}

#[test]
fn na_five_prime_partial_phosphate_completes() {
    // A 5'-phosphate that IS present but missing one OP oxygen: because P
    // is present, the phosphate group is real and the missing OP2
    // completes normally (the skip applies only when the whole group is
    // absent).
    let atoms = da_heavy_atoms(|n| n != "OP2");
    let n = atoms.len();
    let (out_atoms, out_res) = complete_na_residues(
        &atoms,
        &[residue("DA", 0..n)],
        CompletionMode::HeavyOnly,
        MoleculeType::DNA,
    );
    let r = &out_res[0];
    let names: BTreeSet<String> =
        residue_atom_names(&out_atoms, r).into_iter().collect();
    assert!(
        names.contains("OP2"),
        "a present phosphate must complete its missing OP2; got {names:?}"
    );
    assert!(names.contains("P") && names.contains("OP1"));
}

#[test]
fn na_five_prime_skip_does_not_apply_to_interior_residue() {
    // The phosphate-group skip is gated on the chain-first residue. An
    // interior P-less residue (idx > 0) still has its phosphate
    // fabricated, since the rule keys only on the 5' terminus.
    let phosphate = |n: &str| matches!(n, "P" | "OP1" | "OP2" | "OP3");
    let first = da_heavy_atoms(|_| true); // full 5' residue (has P)
    let second = da_heavy_atoms(|n| !phosphate(n)); // P-less interior
    let first_len = first.len();
    let mut atoms = first;
    atoms.extend(second);
    let residues = vec![
        residue("DA", 0..first_len),
        residue("DA", first_len..atoms.len()),
    ];
    let (out_atoms, out_res) = complete_na_residues(
        &atoms,
        &residues,
        CompletionMode::HeavyOnly,
        MoleculeType::DNA,
    );
    let interior_names: BTreeSet<String> =
        residue_atom_names(&out_atoms, &out_res[1])
            .into_iter()
            .collect();
    assert!(
        interior_names.contains("P"),
        "interior P-less residue must still gain its phosphate; got \
         {interior_names:?}"
    );
}

#[test]
fn non_template_atom_is_preserved_with_original_coords() {
    // A backbone-only glycine carrying an extra atom whose name is NOT in
    // the GLY template (CB; glycine has no beta carbon, and an arbitrary
    // ZZ name). Both must survive completion untouched, with their exact
    // input coordinates, while the pass still builds whatever GLY needs.
    let cb_pos = Vec3::new(42.0, -7.0, 3.5);
    let zz_pos = Vec3::new(-1.0, 100.0, 0.25);
    let mut atoms = vec![
        placed_from_template(AminoAcid::Gly, "N", Element::N),
        placed_from_template(AminoAcid::Gly, "CA", Element::C),
        placed_from_template(AminoAcid::Gly, "C", Element::C),
        placed_from_template(AminoAcid::Gly, "O", Element::O),
    ];
    atoms.push(atom_at("CB", Element::C, cb_pos));
    atoms.push(atom_at("ZZ", Element::C, zz_pos));
    let n = atoms.len();
    let (out_atoms, out_res) = complete_protein_residues(
        &atoms,
        &[residue("GLY", 0..n)],
        CompletionMode::HeavyOnly,
    );
    let r = &out_res[0];
    let names: BTreeSet<String> =
        residue_atom_names(&out_atoms, r).into_iter().collect();

    // The off-template atoms survived, at their original coordinates.
    assert!(names.contains("CB"), "off-template CB must be preserved");
    assert!(names.contains("ZZ"), "off-template ZZ must be preserved");
    assert_eq!(
        position_by_name(&out_atoms, r, "CB"),
        cb_pos,
        "preserved CB must keep its original coordinates"
    );
    assert_eq!(
        position_by_name(&out_atoms, r, "ZZ"),
        zz_pos,
        "preserved ZZ must keep its original coordinates"
    );
    // Completion still ran: the GLY backbone hydrogens (e.g. HA2) are in
    // scope for all-atom only, but the heavy-only path here at minimum
    // leaves the backbone intact and adds no off-template atoms.
    for kept in ["N", "CA", "C", "O"] {
        assert!(names.contains(kept), "backbone atom {kept} must survive");
    }
}

#[test]
fn all_atom_places_hydrogen_at_real_bond_length() {
    // Backbone-only ALA through the all-atom entry. Beyond name presence,
    // the fabricated HA must sit ~1.09 A from CA: a mis-rotated H would
    // land at the wrong distance and fail here, where a name-only check
    // would pass.
    let atoms = vec![
        placed_from_template(AminoAcid::Ala, "N", Element::N),
        placed_from_template(AminoAcid::Ala, "CA", Element::C),
        placed_from_template(AminoAcid::Ala, "C", Element::C),
        placed_from_template(AminoAcid::Ala, "O", Element::O),
    ];
    let (out_atoms, out_res) = complete_protein_residues(
        &atoms,
        &[residue("ALA", 0..4)],
        CompletionMode::AllAtom,
    );
    let r = &out_res[0];

    let ca = position_by_name(&out_atoms, r, "CA");
    let ha = position_by_name(&out_atoms, r, "HA");
    let d_ca_ha = (ha - ca).length();
    assert!(
        (0.9..1.3).contains(&d_ca_ha),
        "CA-HA bond should be ~1.09 A, got {d_ca_ha}"
    );

    // And a sidechain C-H: HB1 must sit ~1.09 A from CB.
    let cb = position_by_name(&out_atoms, r, "CB");
    let hb1 = position_by_name(&out_atoms, r, "HB1");
    let d_cb_hb1 = (hb1 - cb).length();
    assert!(
        (0.9..1.3).contains(&d_cb_hb1),
        "CB-HB1 bond should be ~1.09 A, got {d_cb_hb1}"
    );
}

#[test]
fn collinear_anchors_fabricate_nothing() {
    // Three present atoms forced onto a single line (the x-axis): the
    // matched anchors are collinear, so no unique rigid fit exists and the
    // call-site collinearity guard must skip the residue, fabricating no
    // missing atom. We use ALA, whose template wants CB (and more) built;
    // none of it may appear.
    let atoms = vec![
        atom_at("N", Element::N, Vec3::new(0.0, 0.0, 0.0)),
        atom_at("CA", Element::C, Vec3::new(1.5, 0.0, 0.0)),
        atom_at("C", Element::C, Vec3::new(3.0, 0.0, 0.0)),
    ];
    let (out_atoms, out_res) = complete_protein_residues(
        &atoms,
        &[residue("ALA", 0..3)],
        CompletionMode::HeavyOnly,
    );
    let r = &out_res[0];
    // Exactly the three input atoms remain; nothing fabricated.
    assert_eq!(
        r.atom_range.len(),
        3,
        "collinear anchors must fabricate no atoms"
    );
    let names = residue_atom_names(&out_atoms, r);
    assert_eq!(names, vec!["N".to_owned(), "CA".to_owned(), "C".to_owned()]);
    assert!(
        !names.contains(&"CB".to_owned()),
        "collinear-anchored CB must NOT be fabricated"
    );
}
