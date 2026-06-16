//! Integration tests for the mmCIF adapter against on-disk fixtures
//! under `tests/data/cif/`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss
)]

use std::path::PathBuf;

use molex::adapters::cif::{mmcif_file_to_all_models, mmcif_file_to_entities};
use molex::{MoleculeEntity, MoleculeType};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/cif")
        .join(name)
}

fn load(name: &str) -> Vec<MoleculeEntity> {
    mmcif_file_to_entities(&fixture(name)).unwrap()
}

fn protein_count(es: &[MoleculeEntity]) -> usize {
    es.iter()
        .filter(|e| e.molecule_type() == MoleculeType::Protein)
        .count()
}

fn first_protein(es: &[MoleculeEntity]) -> &MoleculeEntity {
    es.iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .expect("expected at least one protein entity")
}

#[test]
fn modified_amino_acids_stay_in_protein_chain() {
    let entities = load("modified_amino_acids.cif");
    assert_eq!(protein_count(&entities), 1);
    let protein = first_protein(&entities).as_protein().unwrap();
    assert_eq!(protein.residues.len(), 4);
    let names: Vec<String> = protein
        .residues
        .iter()
        .map(|r| std::str::from_utf8(&r.name).unwrap().trim().to_owned())
        .collect();
    assert!(names.contains(&"SEP".to_owned()));
    assert!(names.contains(&"MSE".to_owned()));
    assert!(names.contains(&"PTR".to_owned()));
}

#[test]
fn modified_nucleotides_classify_as_rna() {
    let entities = load("modified_nucleotides.cif");
    let rna = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::RNA)
        .expect("expected an RNA entity");
    let na = rna.as_nucleic_acid().unwrap();
    assert_eq!(na.residues.len(), 3);
}

#[test]
fn designed_protein_keeps_noncanonical_residues_in_chain() {
    let entities = load("designed_protein_noncanonical.cif");
    let protein = first_protein(&entities).as_protein().unwrap();
    assert_eq!(protein.residues.len(), 3);
    let n_ligands = entities
        .iter()
        .filter(|e| matches!(e.molecule_type(), MoleculeType::Ligand))
        .count();
    assert_eq!(n_ligands, 0);
}

#[test]
fn auth_chain_id_visible_on_writeback() {
    let entities = load("auth_vs_label_diverges.cif");
    let proteins: Vec<_> = entities
        .iter()
        .filter(|e| e.molecule_type() == MoleculeType::Protein)
        .collect();
    assert_eq!(proteins.len(), 2);
    let label_chains: Vec<char> = proteins
        .iter()
        .map(|e| e.as_protein().unwrap().pdb_chain_id as char)
        .collect();
    let auth_chains: Vec<Option<u8>> = proteins
        .iter()
        .map(|e| e.as_protein().unwrap().auth_asym_id)
        .collect();
    let first = auth_chains[0].expect("auth_asym_id should be populated");
    assert!(
        auth_chains.iter().all(|c| *c == Some(first)),
        "homodimer chains should share auth byte; got {auth_chains:?}"
    );
    assert!(
        label_chains.contains(&'A') && label_chains.contains(&'B'),
        "label chains should keep A/B distinction; got {label_chains:?}"
    );
}

fn find_atom_by_name<'a>(
    entity: &'a MoleculeEntity,
    name: &str,
) -> Option<&'a molex::Atom> {
    entity
        .atom_set()
        .iter()
        .find(|a| std::str::from_utf8(&a.name).unwrap_or("").trim() == name)
}

#[test]
fn single_model_api_returns_first_model() {
    let entities = load("multi_model_3.cif");
    let protein_entity = first_protein(&entities);
    let n = find_atom_by_name(protein_entity, "N").unwrap();
    assert!(n.position.x.abs() < 0.5, "got x={}", n.position.x);
}

#[test]
fn all_models_returns_three_buckets() {
    let models =
        mmcif_file_to_all_models(&fixture("multi_model_3.cif")).unwrap();
    assert_eq!(models.len(), 3);
    for (i, model) in models.iter().enumerate() {
        let protein_entity = first_protein(model);
        let n = find_atom_by_name(protein_entity, "N").unwrap();
        let expected_x = (i as f32) * 10.0;
        assert!(
            (n.position.x - expected_x).abs() < 0.5,
            "model {} expected x={}, got {}",
            i + 1,
            expected_x,
            n.position.x
        );
    }
}

#[test]
fn altloc_a_wins_over_b_by_occupancy() {
    let entities = load("altloc_AB.cif");
    let protein_entity = first_protein(&entities);
    let cb_atoms: Vec<_> = protein_entity
        .atom_set()
        .iter()
        .filter(|a| std::str::from_utf8(&a.name).unwrap().trim() == "CB")
        .collect();
    assert_eq!(cb_atoms.len(), 1);
    assert!((cb_atoms[0].position.x - 2.0).abs() < 0.01);
}

#[test]
fn ins_code_disambiguates_inserts() {
    let entities = load("with_ins_code.cif");
    let protein = first_protein(&entities).as_protein().unwrap();
    assert_eq!(protein.residues.len(), 3);
    let ins_codes: Vec<Option<u8>> =
        protein.residues.iter().map(|r| r.ins_code).collect();
    assert!(ins_codes.contains(&Some(b'A')));
    assert!(ins_codes.contains(&Some(b'B')));
}

#[test]
fn formal_charge_field_populated() {
    let entities = load("formal_charge.cif");
    let charges: Vec<i8> = entities
        .iter()
        .flat_map(|e| e.atom_set().iter().map(|a| a.formal_charge))
        .collect();
    assert!(charges.contains(&2));
    assert!(charges.contains(&-1));
    assert!(charges.contains(&1));
}

#[test]
fn multi_block_input_refuses_with_clear_message() {
    let err = mmcif_file_to_entities(&fixture("multi_block.cif")).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("data blocks"), "msg: {msg}");
}

#[test]
fn branched_entities_collapse_without_crash() {
    let entities = load("branched_collapses.cif");
    let has_protein = entities
        .iter()
        .any(|e| e.molecule_type() == MoleculeType::Protein);
    assert!(has_protein);
    let has_non_polymer = entities.iter().any(|e| {
        matches!(
            e.molecule_type(),
            MoleculeType::Ligand | MoleculeType::Cofactor | MoleculeType::Lipid
        )
    });
    assert!(
        has_non_polymer,
        "branched NAG should land in some Bulk/Ligand"
    );
}

#[test]
fn fast_path_quotes_with_embedded_apostrophe_parse_cleanly() {
    let entities = load("fast_path_quotes.cif");
    let na_entity = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::RNA)
        .unwrap();
    let atom_names: Vec<String> = na_entity
        .atom_set()
        .iter()
        .map(|a| std::str::from_utf8(&a.name).unwrap().trim().to_owned())
        .collect();
    assert!(atom_names.iter().any(|n| n == "O5'"));
    assert!(atom_names.iter().any(|n| n == "C5'"));
}

#[test]
fn fast_path_semicolon_text_field_does_not_desync() {
    let entities = load("fast_path_semicolon.cif");
    let protein_entity = first_protein(&entities);
    let protein = protein_entity.as_protein().unwrap();
    assert_eq!(protein.residues.len(), 1);
    // Desync proxy: the four parsed backbone rows survive in order. The
    // missing-atom completion pass then appends sidechain/hydrogen atoms,
    // so assert on the parsed prefix rather than the total atom count.
    let names: Vec<String> = protein_entity
        .atom_set()
        .iter()
        .take(4)
        .map(|a| std::str::from_utf8(&a.name).unwrap().trim().to_owned())
        .collect();
    assert_eq!(names, ["N", "CA", "C", "O"]);
}

#[test]
fn dom_path_dot_in_type_symbol_falls_back_to_atom_name() {
    use molex::Element;
    let entities = load("dom_dot_in_type_symbol.cif");
    let protein_entity = first_protein(&entities);
    let atoms = protein_entity.atom_set();
    assert_eq!(atoms[0].element, Element::N);
    assert_eq!(atoms[1].element, Element::C);
    assert_eq!(atoms[2].element, Element::C);
    assert_eq!(atoms[3].element, Element::O);
}

#[test]
fn all_cif_fixtures_parse_or_refuse_cleanly() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/cif");
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("cif") {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let expect_refuse = name == "multi_block.cif";
        match mmcif_file_to_entities(&path) {
            Ok(es) => {
                assert!(!expect_refuse, "{name}: expected refuse, got Ok");
                assert!(!es.is_empty(), "{name}: empty entity list");
            }
            Err(e) => {
                assert!(expect_refuse, "{name}: unexpected parse failure: {e}");
            }
        }
    }
}
