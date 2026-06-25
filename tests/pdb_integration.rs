//! Integration tests for the PDB adapter against on-disk fixtures under
//! `tests/data/pdb/`.
//!
//! Covers behaviors that require real files (filename-based BeEM
//! detection, parse-or-refuse drift guard) or that the inline unit tests
//! don't exercise (altLoc dedup, hybrid-36 fields in full atom rows,
//! element-column fallback, AlphaFold-style pLDDT in B-factor).

#![allow(clippy::unwrap_used, clippy::missing_panics_doc)]

use std::path::PathBuf;

use molex::adapters::pdb::pdb_file_to_entities;
use molex::{MoleculeEntity, MoleculeType};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/pdb")
        .join(name)
}

fn load(name: &str) -> Vec<MoleculeEntity> {
    pdb_file_to_entities(&fixture(name)).unwrap()
}

fn atom_names(entity: &MoleculeEntity) -> Vec<String> {
    entity
        .columns()
        .to_atoms()
        .iter()
        .map(|a| {
            std::str::from_utf8(&a.name)
                .unwrap_or("?")
                .trim()
                .to_owned()
        })
        .collect()
}

#[test]
fn with_ligand_classifies_atp_as_small_molecule() {
    let entities = load("with_ligand.pdb");
    let has_ligand = entities.iter().any(|e| {
        matches!(
            e.molecule_type(),
            MoleculeType::Ligand | MoleculeType::Cofactor
        )
    });
    assert!(has_ligand, "ATP should classify as Ligand or Cofactor");
}

#[test]
fn altloc_a_beats_b_when_occupancy_is_higher() {
    let entities = load("altloc_AB.pdb");
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap();
    let names = atom_names(protein);
    let cb_count = names.iter().filter(|n| *n == "CB").count();
    assert_eq!(cb_count, 1);
    let atoms = protein.columns().to_atoms();
    let cb_atom = atoms
        .iter()
        .find(|a| std::str::from_utf8(&a.name).unwrap_or("").trim() == "CB")
        .unwrap();
    assert!((cb_atom.position.x - 2.0).abs() < 0.01);
}

#[test]
fn altloc_numeric_picks_higher_occupancy() {
    let entities = load("altloc_numeric.pdb");
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap();
    let atoms = protein.columns().to_atoms();
    let cb_atom = atoms
        .iter()
        .find(|a| std::str::from_utf8(&a.name).unwrap_or("").trim() == "CB")
        .unwrap();
    // altLoc 1 has occ 0.55, altLoc 2 has occ 0.45 → 1 wins.
    assert!((cb_atom.occupancy - 0.55).abs() < 0.01);
}

#[test]
fn altloc_ties_break_alphabetically() {
    let entities = load("altloc_ties.pdb");
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap();
    let atoms = protein.columns().to_atoms();
    let cb_atom = atoms
        .iter()
        .find(|a| std::str::from_utf8(&a.name).unwrap_or("").trim() == "CB")
        .unwrap();
    // A and B both have occ 0.50 → A wins, x=2.0.
    assert!((cb_atom.position.x - 2.0).abs() < 0.01);
}

#[test]
fn altloc_blank_backbone_kept_with_nonblank_sidechain() {
    let entities = load("altloc_blank_kept.pdb");
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap();
    // Backbone (blank altLoc) + one CB after sidechain dedup.
    assert_eq!(protein.atom_count(), 5);
}

#[test]
fn hybrid36_atom_serial_parses() {
    // molex doesn't store atom serial, so this just verifies the parser
    // doesn't reject the hybrid-36 atom serial field.
    let entities = load("hybrid36_atom_serial.pdb");
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap();
    assert_eq!(protein.atom_count(), 4);
}

#[test]
fn hybrid36_resid_decodes_to_10000() {
    let entities = load("hybrid36_resid.pdb");
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap()
        .as_protein()
        .unwrap();
    assert_eq!(protein.residues.len(), 1);
    assert_eq!(protein.residues[0].label_seq_id, 10_000);
}

#[test]
fn hybrid36_mixed_serial_and_decimal_resid() {
    let entities = load("hybrid36_mixed.pdb");
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap()
        .as_protein()
        .unwrap();
    assert_eq!(protein.residues.len(), 1);
    assert_eq!(protein.residues[0].label_seq_id, 100);
}

#[test]
fn no_element_column_falls_back_to_atom_name_inference() {
    use molex::Element;
    let entities = load("no_element_col.pdb");
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap();
    let atoms = protein.columns().to_atoms();
    assert_eq!(atoms[0].element, Element::N);
    assert_eq!(atoms[1].element, Element::C);
    assert_eq!(atoms[2].element, Element::C);
    assert_eq!(atoms[3].element, Element::O);
}

#[test]
fn alphafold_style_pldbt_in_bfactor_column() {
    let entities = load("alphafold_style.pdb");
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap();
    let atoms = protein.columns().to_atoms();
    assert!(atoms[0].b_factor > 50.0);
    assert!(atoms[0].b_factor < 100.0);
}

#[test]
fn beem_bundle_refused_with_mmcif_redirect() {
    // Exercises the filename-based BeEM detection — only reachable via
    // pdb_file_to_entities (the string entry point can't see the
    // filename pattern).
    let path = fixture("1xyz-pdb-bundle1.pdb");
    let err = pdb_file_to_entities(&path).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("BeEM"), "msg: {msg}");
    assert!(msg.contains("1xyz") || msg.contains("1XYZ"), "msg: {msg}");
    assert!(msg.contains("cif"), "msg: {msg}");
}

/// Drift guard: every `.pdb` under `tests/data/pdb/` either parses
/// non-empty or refuses cleanly (bundle files only). Catches silent
/// regressions if fixtures are added later without a matching test.
#[test]
fn all_fixtures_parse_or_refuse_cleanly() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pdb");
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("pdb") {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let is_bundle = name.contains("-pdb-bundle");
        match pdb_file_to_entities(&path) {
            Ok(es) => {
                assert!(
                    !is_bundle,
                    "{name}: bundle file should refuse but parsed"
                );
                assert!(!es.is_empty(), "{name}: empty entity list");
            }
            Err(e) => {
                assert!(is_bundle, "{name}: unexpected parse failure: {e}");
            }
        }
    }
}
