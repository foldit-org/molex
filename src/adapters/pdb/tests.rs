//! Inline unit tests for the PDB adapter.

#![allow(clippy::unwrap_used, clippy::float_cmp)]

use super::parse::{
    decode_hybrid36_resseq, decode_hybrid36_serial, normalize_name,
    parse_pdb_charge, parse_seq_field,
};
use super::refuse::extract_pdb_id_from_bundle_filename;
use super::{
    entities_to_pdb, pdb_str_to_all_models, pdb_str_to_entities,
    pdb_str_to_entities_with,
};
use crate::entity::molecule::MoleculeType;
use crate::{Completion, MoleculeEntity};

/// Minimal PDB with one backbone-only residue (N, CA, C, O). Named MSE
/// so the missing-atom completion pass is a no-op and these parse-path
/// tests keep their fixed 4-atom counts.
const MINIMAL_PDB: &str = "\
ATOM      1  N   MSE A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  MSE A   1       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  C   MSE A   1       7.000   8.000   9.000  1.00  0.00           C
ATOM      4  O   MSE A   1      10.000  11.000  12.000  1.00  0.00           O
END
";

#[test]
fn pdb_str_to_entities_minimal() {
    let entities = pdb_str_to_entities(MINIMAL_PDB).unwrap();
    assert!(!entities.is_empty());
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein);
    assert!(protein.is_some());
    assert_eq!(protein.unwrap().atom_count(), 4);
}

/// One ALA residue with only its four backbone atoms at real (non-collinear)
/// peptide geometry, so the rigid fit anchors and heavy completion can
/// fabricate the missing CB (and the lone-residue C-terminal OXT).
const SPARSE_ALA_PDB: &str = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
END
";

fn protein_atom_count(entities: &[MoleculeEntity]) -> usize {
    entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .map_or(0, MoleculeEntity::atom_count)
}

#[test]
fn raw_parse_skips_completion_heavy_fabricates_and_complete_recovers() {
    let raw =
        pdb_str_to_entities_with(SPARSE_ALA_PDB, Completion::Raw).unwrap();
    let heavy =
        pdb_str_to_entities_with(SPARSE_ALA_PDB, Completion::Heavy).unwrap();

    let raw_count = protein_atom_count(&raw);
    let heavy_count = protein_atom_count(&heavy);

    // Raw is the four parsed backbone atoms verbatim; Heavy additionally
    // fabricates CB and the lone-residue C-terminal OXT.
    assert_eq!(raw_count, 4, "Raw must keep only the parsed atoms");
    assert!(
        raw_count < heavy_count,
        "Heavy must fabricate atoms Raw omits ({raw_count} vs {heavy_count})",
    );

    // Default parse equals Heavy (today's behavior, unchanged).
    assert_eq!(
        protein_atom_count(&pdb_str_to_entities(SPARSE_ALA_PDB).unwrap()),
        heavy_count
    );

    // Re-completing the Raw result to Heavy recovers the Heavy atom set.
    let recovered: Vec<MoleculeEntity> =
        raw.iter().map(|e| e.complete(Completion::Heavy)).collect();
    assert_eq!(
        protein_atom_count(&recovered),
        heavy_count,
        "complete(Heavy) on the Raw parse must match the Heavy parse",
    );
}

#[test]
fn pdb_str_to_entities_preserves_positions() {
    let entities = pdb_str_to_entities(MINIMAL_PDB).unwrap();
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap();
    let pos = &protein.atom_set()[0].position;
    assert!((pos.x - 1.0).abs() < 0.01);
    assert!((pos.y - 2.0).abs() < 0.01);
    assert!((pos.z - 3.0).abs() < 0.01);
}

#[test]
fn pdb_str_to_entities_two_chains() {
    let pdb = "\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  C   ALA A   1       7.000   8.000   9.000  1.00  0.00           C
ATOM      4  N   GLY B   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      5  CA  GLY B   1       4.000   5.000   6.000  1.00  0.00           C
ATOM      6  C   GLY B   1       7.000   8.000   9.000  1.00  0.00           C
END
";
    let entities = pdb_str_to_entities(pdb).unwrap();
    let protein_count = entities
        .iter()
        .filter(|e| e.molecule_type() == MoleculeType::Protein)
        .count();
    assert_eq!(protein_count, 2);
}

#[test]
fn pdb_str_to_entities_water() {
    let pdb = "\
ATOM      1  O   HOH A 100       1.000   2.000   3.000  1.00  0.00           O
ATOM      2  O   HOH A 101       4.000   5.000   6.000  1.00  0.00           O
END
";
    let entities = pdb_str_to_entities(pdb).unwrap();
    let water = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Water);
    assert!(water.is_some());
    assert_eq!(water.unwrap().atom_count(), 2);
}

#[test]
fn entities_to_pdb_produces_valid_output() {
    let entities = pdb_str_to_entities(MINIMAL_PDB).unwrap();
    let pdb_output = entities_to_pdb(&entities).unwrap();
    assert!(pdb_output.contains("ATOM"));
    assert!(pdb_output.contains("MSE"));
    assert!(pdb_output.ends_with("END\n"));
    let atom_line_count =
        pdb_output.lines().filter(|l| l.starts_with("ATOM")).count();
    assert_eq!(atom_line_count, 4);
}

#[test]
fn entities_to_pdb_preserves_coordinates_in_output() {
    let entities = pdb_str_to_entities(MINIMAL_PDB).unwrap();
    let pdb_output = entities_to_pdb(&entities).unwrap();
    assert!(pdb_output.contains("1.000"));
    assert!(pdb_output.contains("2.000"));
    assert!(pdb_output.contains("3.000"));
    assert!(pdb_output.contains('A'));
    assert!(pdb_output.contains('1'));
}

#[test]
fn pdb_str_empty_produces_error() {
    let result = pdb_str_to_entities("");
    assert!(result.is_err());
}

#[test]
fn parser_silently_skips_remarks_and_misc_records() {
    // MSE residue so missing-atom completion is a no-op; this test fixes
    // the parsed atom count and is about record skipping, not chemistry.
    let pdb = "\
HEADER    SOME HEADER
REMARK no leading digit accepted
TITLE     A LONELY PROTEIN
ATOM      1  N   MSE A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  MSE A   1       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  C   MSE A   1       7.000   8.000   9.000  1.00  0.00           C
ATOM      4  O   MSE A   1      10.000  11.000  12.000  1.00  0.00           O
CONECT    1    2
END
";
    let entities = pdb_str_to_entities(pdb).unwrap();
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap();
    assert_eq!(protein.atom_count(), 4);
}

#[test]
fn normalize_name_left_justifies_with_spaces() {
    assert_eq!(normalize_name::<4>(b" CA "), [b'C', b'A', b' ', b' ']);
    assert_eq!(normalize_name::<4>(b"FE  "), [b'F', b'E', b' ', b' ']);
    assert_eq!(normalize_name::<4>(b"1HD1"), [b'1', b'H', b'D', b'1']);
    assert_eq!(normalize_name::<3>(b"ALA"), [b'A', b'L', b'A']);
}

#[test]
fn parse_pdb_charge_handles_signed_digit_pairs() {
    assert_eq!(parse_pdb_charge(b"  "), 0);
    assert_eq!(parse_pdb_charge(b"2+"), 2);
    assert_eq!(parse_pdb_charge(b"1-"), -1);
    assert_eq!(parse_pdb_charge(b""), 0);
}

#[test]
fn decode_hybrid36_serial_decimal_boundary() {
    // "A0000" is the first hybrid-36 value, encoding atom serial 100000.
    assert_eq!(decode_hybrid36_serial(b"A0000"), Some(100_000));
    // Last value still inside i32 range. ZZZZZ = 100_000 + 26 * 36^4 - 1
    // = 43_770_015.
    assert_eq!(decode_hybrid36_serial(b"ZZZZZ"), Some(43_770_015));
    // Lower-case range starts at 100_000 + 26 * 36^4 = 43_770_016.
    assert_eq!(decode_hybrid36_serial(b"a0000"), Some(43_770_016));
    // Wrong width.
    assert_eq!(decode_hybrid36_serial(b"A000"), None);
    // Non-alphanumeric.
    assert_eq!(decode_hybrid36_serial(b"A0 00"), None);
    // Pure decimal: caller's responsibility, helper rejects it.
    assert_eq!(decode_hybrid36_serial(b"12345"), None);
}

#[test]
fn decode_hybrid36_resseq_decimal_boundary() {
    // "A000" is the first hybrid-36 value, encoding resseq 10000.
    assert_eq!(decode_hybrid36_resseq(b"A000"), Some(10_000));
    // Worked example: pure base-36 of "ZZZZ" = 35*(36^3+36^2+36+1) =
    // 1_679_615; result = pure - 10*36^3 + 10^4 = 1_223_055.
    assert_eq!(decode_hybrid36_resseq(b"ZZZZ"), Some(1_223_055));
    // Lower-case range starts one above the upper range.
    assert_eq!(decode_hybrid36_resseq(b"a000"), Some(1_223_056));
}

#[test]
fn parse_seq_field_per_column_decimal_vs_hybrid() {
    // Decimal with leading space: falls to plain int parse.
    assert_eq!(parse_seq_field(b" 184", 4), Some(184));
    // All decimal digits, exactly width chars: decimal.
    assert_eq!(parse_seq_field(b"9999", 4), Some(9999));
    // Hybrid-36 detected (exact width, all alphanumeric, has letter).
    assert_eq!(parse_seq_field(b"A000", 4), Some(10_000));
    // Has a non-alphanumeric char -> not hybrid; decimal parse of
    // signed integer still works.
    assert_eq!(parse_seq_field(b"-184", 4), Some(-184));
}

#[test]
fn all_models_returns_one_entry_for_single_model_file() {
    let models = pdb_str_to_all_models(MINIMAL_PDB).unwrap();
    assert_eq!(models.len(), 1);
    assert!(!models[0].is_empty());
}

#[test]
fn all_models_returns_one_entry_per_model_block() {
    // MSE residues so missing-atom completion is a no-op; this test fixes
    // the per-model atom count and is about MODEL/ENDMDL block handling.
    let pdb = "\
MODEL        1
ATOM      1  N   MSE A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  MSE A   1       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  C   MSE A   1       7.000   8.000   9.000  1.00  0.00           C
ATOM      4  O   MSE A   1      10.000  11.000  12.000  1.00  0.00           O
ENDMDL
MODEL        2
ATOM      1  N   MSE A   1       1.500   2.500   3.500  1.00  0.00           N
ATOM      2  CA  MSE A   1       4.500   5.500   6.500  1.00  0.00           C
ATOM      3  C   MSE A   1       7.500   8.500   9.500  1.00  0.00           C
ATOM      4  O   MSE A   1      10.500  11.500  12.500  1.00  0.00           O
ENDMDL
END
";
    let models = pdb_str_to_all_models(pdb).unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0][0].atom_count(), 4);
    assert_eq!(models[1][0].atom_count(), 4);
    let m0_x = models[0][0].atom_set()[0].position.x;
    let m1_x = models[1][0].atom_set()[0].position.x;
    assert!((m0_x - 1.0).abs() < 0.01);
    assert!((m1_x - 1.5).abs() < 0.01);
}

#[test]
fn single_model_api_returns_first_model_only() {
    let pdb = "\
MODEL        1
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  C   ALA A   1       7.000   8.000   9.000  1.00  0.00           C
ATOM      4  O   ALA A   1      10.000  11.000  12.000  1.00  0.00           O
ENDMDL
MODEL        2
ATOM      1  N   ALA A   1       1.500   2.500   3.500  1.00  0.00           N
ATOM      2  CA  ALA A   1       4.500   5.500   6.500  1.00  0.00           C
ATOM      3  C   ALA A   1       7.500   8.500   9.500  1.00  0.00           C
ATOM      4  O   ALA A   1      10.500  11.500  12.500  1.00  0.00           O
ENDMDL
END
";
    let entities = pdb_str_to_entities(pdb).unwrap();
    let protein = entities
        .iter()
        .find(|e| e.molecule_type() == MoleculeType::Protein)
        .unwrap();
    let pos = &protein.atom_set()[0].position;
    assert!((pos.x - 1.0).abs() < 0.01);
}

#[test]
fn writer_emits_hetatm_for_water_bulk() {
    let pdb = "\
ATOM      1  O   HOH A 100       1.000   2.000   3.000  1.00  0.00           O
ATOM      2  O   HOH A 101       4.000   5.000   6.000  1.00  0.00           O
END
";
    let entities = pdb_str_to_entities(pdb).unwrap();
    let out = entities_to_pdb(&entities).unwrap();
    let hetatm_count = out.lines().filter(|l| l.starts_with("HETATM")).count();
    let atom_count = out.lines().filter(|l| l.starts_with("ATOM  ")).count();
    assert_eq!(hetatm_count, 2);
    assert_eq!(atom_count, 0);
}

#[test]
fn writer_emits_ter_after_polymer_chains() {
    let pdb = "\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  C   ALA A   1       7.000   8.000   9.000  1.00  0.00           C
ATOM      4  O   ALA A   1      10.000  11.000  12.000  1.00  0.00           O
ATOM      5  N   GLY B   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      6  CA  GLY B   1       4.000   5.000   6.000  1.00  0.00           C
ATOM      7  C   GLY B   1       7.000   8.000   9.000  1.00  0.00           C
ATOM      8  O   GLY B   1      10.000  11.000  12.000  1.00  0.00           O
END
";
    let entities = pdb_str_to_entities(pdb).unwrap();
    let out = entities_to_pdb(&entities).unwrap();
    let ter_count = out.lines().filter(|l| l.starts_with("TER")).count();
    assert_eq!(ter_count, 2);
    let ter_lines: Vec<&str> =
        out.lines().filter(|l| l.starts_with("TER")).collect();
    assert!(ter_lines[0].contains("ALA A"));
    assert!(ter_lines[1].contains("GLY B"));
}

#[test]
fn writer_aligns_atom_name_by_element_symbol() {
    let entities = pdb_str_to_entities(MINIMAL_PDB).unwrap();
    let out = entities_to_pdb(&entities).unwrap();
    let ca_line = out
        .lines()
        .find(|l| l.starts_with("ATOM") && l.contains(" CA "))
        .unwrap();
    assert_eq!(&ca_line[12..16], " CA ");
}

#[test]
fn beem_bundle_detected_by_header_text() {
    let pdb = "\
HEADER    PDB BUNDLE FOR PDB ID 1XYZ                          12-JAN-26   1XYZ
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
END
";
    let err = pdb_str_to_entities(pdb).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("BeEM"), "msg: {msg}");
    assert!(msg.contains("mmCIF"), "msg: {msg}");
}

#[test]
fn beem_bundle_detected_by_filename_pattern() {
    let id = extract_pdb_id_from_bundle_filename("1xyz-pdb-bundle1.pdb");
    assert_eq!(id.as_deref(), Some("1xyz"));
    let id = extract_pdb_id_from_bundle_filename("1XYZ-pdb-bundle10.pdb");
    assert_eq!(id.as_deref(), Some("1XYZ"));
    assert!(extract_pdb_id_from_bundle_filename("1xyz.pdb").is_none());
    assert!(
        extract_pdb_id_from_bundle_filename("1xyz-pdb-bundle1a.pdb").is_none()
    );
}

#[test]
fn writer_refuses_overflowing_atom_count() {
    use crate::entity::molecule::bulk::BulkEntity;
    use crate::entity::molecule::id::EntityIdAllocator;
    use crate::entity::molecule::Atom;
    use crate::Element;
    let mut alloc = EntityIdAllocator::new();
    let atoms: Vec<Atom> = (0..100_001)
        .map(|_| Atom {
            position: glam::Vec3::ZERO,
            occupancy: 1.0,
            b_factor: 0.0,
            element: Element::O,
            name: *b"O   ",
            formal_charge: 0,
        })
        .collect();
    let bulk = BulkEntity::new(
        alloc.allocate(),
        MoleculeType::Water,
        atoms,
        *b"HOH",
        100_001,
    );
    let entities = vec![MoleculeEntity::Bulk(bulk)];
    let err = entities_to_pdb(&entities).unwrap_err();
    assert!(err.to_string().contains("legacy PDB format"));
}
