#![allow(clippy::unwrap_used, clippy::float_cmp)]

use glam::Vec3;

use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::bulk::BulkEntity;
use crate::entity::molecule::id::EntityIdAllocator;
use crate::entity::molecule::nucleic_acid::NAEntity;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::small_molecule::SmallMoleculeEntity;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};
use crate::ops::wire::{assembly_bytes, deserialize_assembly};

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

fn ala_residue_atoms(start_x: f32) -> Vec<Atom> {
    vec![
        atom_at("N", Element::N, start_x),
        atom_at("CA", Element::C, start_x + 1.0),
        atom_at("C", Element::C, start_x + 2.0),
        atom_at("O", Element::O, start_x + 3.0),
    ]
}

#[test]
fn assembly_bytes_roundtrip_mixed() {
    let mut allocator = EntityIdAllocator::new();

    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        allocator.allocate(),
        ala_residue_atoms(1.0),
        vec![residue("ALA", 1, 0..4)],
        "A".to_owned(),
    ));
    let ligand = MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
        allocator.allocate(),
        MoleculeType::Ligand,
        vec![atom_at("C1", Element::C, 10.0)],
        res_bytes("ATP"),
    ));
    let zinc = MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
        allocator.allocate(),
        MoleculeType::Ion,
        vec![atom_at("ZN", Element::Zn, 20.0)],
        res_bytes("ZN"),
    ));

    let entities = vec![protein, ligand, zinc];
    let bytes = assembly_bytes(&entities).unwrap();
    assert_eq!(&bytes[0..8], b"ASSEMBLY");
    assert_eq!(bytes[8], 2);

    let roundtripped = deserialize_assembly(&bytes).unwrap();
    assert_eq!(roundtripped.entities().len(), entities.len());

    for (orig, rt) in entities.iter().zip(roundtripped.entities().iter()) {
        assert_eq!(orig.molecule_type(), rt.molecule_type());
        assert_eq!(orig.atom_count(), rt.atom_count());
    }
}

#[test]
fn assembly_bytes_protein_only() {
    let id = EntityIdAllocator::new().allocate();
    // This test fixes the serialized atom count and is about wire layout,
    // not chemistry. Construction is pure, so the four backbone atoms
    // round-trip unchanged regardless of residue name.
    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        ala_residue_atoms(1.0),
        vec![residue("UNK", 1, 0..4)],
        "A".to_owned(),
    ));
    let entities = vec![protein];

    let bytes = assembly_bytes(&entities).unwrap();
    let roundtripped = deserialize_assembly(&bytes).unwrap();
    assert_eq!(roundtripped.entities().len(), 1);
    assert_eq!(
        roundtripped.entities()[0].molecule_type(),
        MoleculeType::Protein
    );
    assert_eq!(roundtripped.entities()[0].atom_count(), 4);
}

#[test]
fn deserialize_assembly_is_pure_adds_no_atoms() {
    // The wire path constructs entities purely: a sidechain-incomplete
    // residue (backbone-only ALA, CB absent) must survive serialization
    // and deserialization with its atom count unchanged. Completion would
    // fabricate CB; the wire path must not run it.
    let id = EntityIdAllocator::new().allocate();
    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        ala_residue_atoms(1.0),
        vec![residue("ALA", 1, 0..4)],
        "A".to_owned(),
    ));
    let input_count = protein.atom_count();
    assert_eq!(input_count, 4, "backbone-only ALA starts with 4 atoms");

    let bytes = assembly_bytes(&[protein]).unwrap();
    let roundtripped = deserialize_assembly(&bytes).unwrap();
    assert_eq!(
        roundtripped.entities()[0].atom_count(),
        input_count,
        "wire round-trip must fabricate no atoms (CB stays absent)"
    );
}

#[test]
fn assembly_bytes_single_atom_ion() {
    let id = EntityIdAllocator::new().allocate();
    let ion = MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
        id,
        MoleculeType::Ion,
        vec![atom_at("ZN", Element::Zn, 5.5)],
        res_bytes("ZN"),
    ));
    let entities = vec![ion];
    let bytes = assembly_bytes(&entities).unwrap();
    let roundtripped = deserialize_assembly(&bytes).unwrap();
    assert_eq!(roundtripped.entities().len(), 1);
    assert_eq!(
        roundtripped.entities()[0].molecule_type(),
        MoleculeType::Ion
    );
    assert!(
        (roundtripped.entities()[0].atom_set()[0].position.x - 5.5).abs()
            < 1e-6
    );
}

#[test]
fn assembly_bytes_empty_entities() {
    let entities: Vec<MoleculeEntity> = Vec::new();
    let bytes = assembly_bytes(&entities).unwrap();
    let roundtripped = deserialize_assembly(&bytes).unwrap();
    assert!(roundtripped.entities().is_empty());
}

#[test]
fn assembly_byte_layout() {
    let id = EntityIdAllocator::new().allocate();
    // Residue named UNK so the missing-atom completion pass is a no-op
    // and the exact serialized byte length below stays fixed at 4 atoms.
    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        ala_residue_atoms(1.0),
        vec![residue("UNK", 1, 0..4)],
        "A".to_owned(),
    ));
    let entities = vec![protein];
    let bytes = assembly_bytes(&entities).unwrap();

    // 8 magic + 1 version + 4 count + per-entity header (9 fixed + 2 chain
    // len + 1 chain byte "A") + 4 atoms * 25 + 4 variant-count u32 (zero, no
    // variants on this residue).
    assert_eq!(&bytes[0..8], b"ASSEMBLY");
    assert_eq!(bytes[8], 2); // version
    assert_eq!(u32::from_be_bytes(bytes[9..13].try_into().unwrap()), 1);
    assert_eq!(bytes[13], 0); // Protein
    assert_eq!(u32::from_be_bytes(bytes[14..18].try_into().unwrap()), 4);
    // bytes[18..22] is the 4-byte entity_id; bytes[22..24] the u16 chain
    // length (1); bytes[24] the chain byte (b'A').
    assert_eq!(u16::from_be_bytes(bytes[22..24].try_into().unwrap()), 1);
    assert_eq!(bytes[24], b'A');
    assert_eq!(bytes.len(), 8 + 1 + 4 + (9 + 2 + 1) + 4 * 25 + 4);
}

#[test]
fn assembly_bytes_polymer_roundtrip_preserves_residues() {
    let id = EntityIdAllocator::new().allocate();
    let atoms = {
        let mut v = ala_residue_atoms(1.0);
        v.extend(ala_residue_atoms(5.0));
        v
    };
    let residues = vec![residue("ALA", 1, 0..4), residue("ALA", 2, 4..8)];
    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        atoms,
        residues,
        "A".to_owned(),
    ));
    let entities = vec![protein];

    let bytes = assembly_bytes(&entities).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();
    let rt_protein = rt.entities()[0].as_protein().unwrap();
    assert_eq!(rt_protein.residues.len(), 2);
    assert_eq!(rt_protein.residues[0].label_seq_id, 1);
    assert_eq!(rt_protein.residues[1].label_seq_id, 2);
}

#[test]
fn assembly_bytes_na_roundtrip() {
    let atoms = vec![
        atom_at("P", Element::P, 0.0),
        atom_at("O5'", Element::O, 1.0),
        atom_at("C5'", Element::C, 2.0),
        atom_at("C4'", Element::C, 3.0),
        atom_at("C3'", Element::C, 4.0),
        atom_at("O3'", Element::O, 5.0),
        atom_at("N9", Element::N, 6.0),
    ];
    // Residue named UNK so the missing-atom completion pass is a no-op;
    // the molecule type comes from the DNA argument, not the residue name.
    let residues = vec![residue("UNK", 1, 0..7)];
    let id = EntityIdAllocator::new().allocate();
    let na = MoleculeEntity::NucleicAcid(NAEntity::new(
        id,
        MoleculeType::DNA,
        atoms,
        residues,
        "A".to_owned(),
    ));
    let entities = vec![na];

    let bytes = assembly_bytes(&entities).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();
    assert_eq!(rt.entities()[0].molecule_type(), MoleculeType::DNA);
    assert_eq!(rt.entities()[0].atom_count(), 7);
}

#[test]
fn assembly_bytes_water_roundtrip() {
    let atoms =
        vec![atom_at("O", Element::O, 0.0), atom_at("O", Element::O, 1.0)];
    let id = EntityIdAllocator::new().allocate();
    let water = MoleculeEntity::Bulk(BulkEntity::new(
        id,
        MoleculeType::Water,
        atoms,
        res_bytes("HOH"),
        2,
    ));
    let entities = vec![water];

    let bytes = assembly_bytes(&entities).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();
    assert_eq!(rt.entities()[0].molecule_type(), MoleculeType::Water);
    assert_eq!(rt.entities()[0].atom_count(), 2);
    assert_eq!(rt.entities()[0].as_bulk().unwrap().molecule_count, 2);
}

#[test]
fn deserialize_assembly_empty_bytes() {
    assert!(deserialize_assembly(&[]).is_err());
}

#[test]
fn deserialize_assembly_wrong_magic() {
    assert!(deserialize_assembly(b"BADMAGIC\x00\x00\x00\x00").is_err());
}

#[test]
fn deserialize_assembly_truncated_atom_data() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"ASSEMBLY");
    bytes.push(1); // version
    bytes.extend_from_slice(&1u32.to_be_bytes()); // 1 entity
    bytes.push(0); // Protein type
    bytes.extend_from_slice(&1u32.to_be_bytes()); // 1 atom
    bytes.extend_from_slice(&0u32.to_be_bytes()); // entity_id, no atom data
    assert!(deserialize_assembly(&bytes).is_err());
}

#[test]
fn deserialize_assembly_rejects_unknown_version() {
    // A well-formed header with a future version byte must be a clean
    // error, not a panic or a misparse. Versions 1 and 2 are supported;
    // 3 is not.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"ASSEMBLY");
    bytes.push(3); // unsupported version
    bytes.extend_from_slice(&0u32.to_be_bytes()); // 0 entities
    assert!(deserialize_assembly(&bytes).is_err());
}

#[test]
fn deserialize_assembly_corrupt_buffers_error_not_panic() {
    // Valid magic + version + entity count, but the per-entity header is
    // cut off (claims one entity, then ends).
    let mut truncated_header = Vec::new();
    truncated_header.extend_from_slice(b"ASSEMBLY");
    truncated_header.push(1); // version
    truncated_header.extend_from_slice(&1u32.to_be_bytes()); // 1 entity
    truncated_header.push(0); // mol_type byte, nothing after it
    assert!(deserialize_assembly(&truncated_header).is_err());

    // Valid magic + version + an absurd entity count with no entity data.
    let mut overstated_count = Vec::new();
    overstated_count.extend_from_slice(b"ASSEMBLY");
    overstated_count.push(1); // version
    overstated_count.extend_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
    assert!(deserialize_assembly(&overstated_count).is_err());

    // Arbitrary garbage of header length: bad magic, must error cleanly.
    let garbage = [0xABu8; 32];
    assert!(deserialize_assembly(&garbage).is_err());
}

#[test]
fn preserves_entity_id_across_roundtrip() {
    // Allocate a few EntityIds so the test entity has a non-zero id;
    // round-trip must preserve it.
    let mut alloc = EntityIdAllocator::new();
    let _ = alloc.allocate();
    let _ = alloc.allocate();
    let id = alloc.allocate(); // EntityId(2)

    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        ala_residue_atoms(0.0),
        vec![residue("ALA", 1, 0..4)],
        "A".to_owned(),
    ));

    let bytes = assembly_bytes(&[protein]).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();
    assert_eq!(rt.entities()[0].id(), id);
}

#[test]
fn variants_roundtrip_through_wire() {
    use crate::chemistry::variant::{ProtonationState, VariantTag};

    let id = EntityIdAllocator::new().allocate();
    let atoms = {
        let mut v = ala_residue_atoms(1.0);
        v.extend(ala_residue_atoms(5.0));
        v
    };
    // Residue 1: protonation HisEpsilon + Disulfide.
    // Residue 2: NTerminus + Other("custom-tag").
    let mut r1 = residue("HIS", 1, 0..4);
    r1.variants = vec![
        VariantTag::Protonation(ProtonationState::HisEpsilon),
        VariantTag::Disulfide,
    ];
    let mut r2 = residue("ALA", 2, 4..8);
    r2.variants = vec![
        VariantTag::NTerminus,
        VariantTag::Other("custom-tag".to_owned()),
    ];

    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        atoms,
        vec![r1, r2],
        "A".to_owned(),
    ));

    let bytes = assembly_bytes(&[protein]).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();
    let rt_protein = rt.entities()[0].as_protein().unwrap();

    assert_eq!(rt_protein.residues[0].variants.len(), 2);
    assert!(matches!(
        rt_protein.residues[0].variants[0],
        VariantTag::Protonation(ProtonationState::HisEpsilon),
    ));
    assert!(matches!(
        rt_protein.residues[0].variants[1],
        VariantTag::Disulfide,
    ));

    assert_eq!(rt_protein.residues[1].variants.len(), 2);
    assert!(matches!(
        rt_protein.residues[1].variants[0],
        VariantTag::NTerminus,
    ));
    assert!(
        matches!(&rt_protein.residues[1].variants[1], VariantTag::Other(s) if s == "custom-tag"),
    );
}

#[test]
fn variants_protonation_custom_string_roundtrip() {
    use crate::chemistry::variant::{ProtonationState, VariantTag};

    let id = EntityIdAllocator::new().allocate();
    let atoms = ala_residue_atoms(1.0);
    let mut r = residue("ASP", 1, 0..4);
    r.variants = vec![VariantTag::Protonation(ProtonationState::Custom(
        "ASP_PROTONATED".to_owned(),
    ))];

    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        atoms,
        vec![r],
        "A".to_owned(),
    ));

    let bytes = assembly_bytes(&[protein]).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();
    let rt_protein = rt.entities()[0].as_protein().unwrap();

    assert!(matches!(
        &rt_protein.residues[0].variants[0],
        VariantTag::Protonation(ProtonationState::Custom(s))
            if s == "ASP_PROTONATED",
    ));
}

#[test]
fn multi_char_chain_id_survives_roundtrip() {
    // A multi-character chain id ("AA") is impossible under the old
    // single-byte model; the version-2 header chain table carries it.
    let id = EntityIdAllocator::new().allocate();
    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        ala_residue_atoms(1.0),
        vec![residue("ALA", 1, 0..4)],
        "AA".to_owned(),
    ));

    let bytes = assembly_bytes(&[protein]).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();
    assert_eq!(rt.entities()[0].pdb_chain_id(), Some("AA"));
}

#[test]
fn over_ninety_chains_survive_roundtrip() {
    // The old 90-chain cap is gone: serialize 120 distinct multi-character
    // chains, deserialize, and assert every chain id is preserved.
    let mut alloc = EntityIdAllocator::new();
    let entities: Vec<MoleculeEntity> = (0..120u16)
        .map(|i| {
            MoleculeEntity::Protein(ProteinEntity::new(
                alloc.allocate(),
                ala_residue_atoms(f32::from(i)),
                vec![residue("ALA", 1, 0..4)],
                format!("CH{i}"),
            ))
        })
        .collect();

    let bytes = assembly_bytes(&entities).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();
    assert_eq!(rt.entities().len(), 120);
    for (i, e) in rt.entities().iter().enumerate() {
        assert_eq!(e.pdb_chain_id(), Some(format!("CH{i}").as_str()));
    }
}

#[test]
fn version_one_payload_still_decodes() {
    // A version-1 payload (per-atom chain byte, 26-byte rows, 9-byte
    // headers) must still decode after the version-2 bump. Hand-build one
    // ALA backbone atom on chain 'A'.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"ASSEMBLY");
    bytes.push(1); // version 1
    bytes.extend_from_slice(&1u32.to_be_bytes()); // 1 entity
                                                  // Entity header (v1): mol_type, atom_count, entity_id.
    bytes.push(0); // Protein
    bytes.extend_from_slice(&1u32.to_be_bytes()); // 1 atom
    bytes.extend_from_slice(&0u32.to_be_bytes()); // entity_id 0
                                                  // One v1 atom row (26 bytes): xyz, chain byte, res_name, res_num,
                                                  // atom_name, element symbol.
    bytes.extend_from_slice(&0.0f32.to_be_bytes());
    bytes.extend_from_slice(&0.0f32.to_be_bytes());
    bytes.extend_from_slice(&0.0f32.to_be_bytes());
    bytes.push(b'A'); // chain byte
    bytes.extend_from_slice(b"ALA"); // res_name
    bytes.extend_from_slice(&1i32.to_be_bytes()); // res_num
    bytes.extend_from_slice(b"CA  "); // atom_name
    bytes.push(b'C'); // element symbol byte 0
    bytes.push(0); // element symbol byte 1
                   // Variants section: 1 entity, zero variant blocks.
    bytes.extend_from_slice(&0u32.to_be_bytes());

    let rt = deserialize_assembly(&bytes).unwrap();
    assert_eq!(rt.entities().len(), 1);
    assert_eq!(rt.entities()[0].pdb_chain_id(), Some("A"));
    assert_eq!(rt.entities()[0].atom_count(), 1);
}
