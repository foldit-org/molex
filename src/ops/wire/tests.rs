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
        b'A',
        None,
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
    assert_eq!(&bytes[0..8], b"ASSEM02\0");

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
        b'A',
        None,
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
        b'A',
        None,
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
        b'A',
        None,
    ));
    let entities = vec![protein];
    let bytes = assembly_bytes(&entities).unwrap();

    // 8 magic + 4 count + 9 per-entity header + 4 atoms * 26
    // + 4 variant-count u32 (zero, no variants on this residue).
    assert_eq!(&bytes[0..8], b"ASSEM02\0");
    assert_eq!(u32::from_be_bytes(bytes[8..12].try_into().unwrap()), 1);
    assert_eq!(bytes[12], 0); // Protein
    assert_eq!(u32::from_be_bytes(bytes[13..17].try_into().unwrap()), 4);
    // bytes[17..21] is the 4-byte entity_id (originator's raw value).
    assert_eq!(bytes.len(), 8 + 4 + 9 + 4 * 26 + 4);
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
        id, atoms, residues, b'A', None,
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
        b'A',
        None,
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
    bytes.extend_from_slice(b"ASSEM01\0");
    bytes.extend_from_slice(&1u32.to_be_bytes()); // 1 entity
    bytes.push(0); // Protein type
    bytes.extend_from_slice(&1u32.to_be_bytes()); // 1 atom, no atom data
    assert!(deserialize_assembly(&bytes).is_err());
}

#[test]
fn deserialize_assembly_corrupt_buffers_error_not_panic() {
    // Valid magic + entity count, but the per-entity header is cut off
    // (claims one entity, then ends).
    let mut truncated_header = Vec::new();
    truncated_header.extend_from_slice(b"ASSEM02\0");
    truncated_header.extend_from_slice(&1u32.to_be_bytes()); // 1 entity
    truncated_header.push(0); // mol_type byte, nothing after it
    assert!(deserialize_assembly(&truncated_header).is_err());

    // Valid magic + an absurd entity count with no entity data following.
    let mut overstated_count = Vec::new();
    overstated_count.extend_from_slice(b"ASSEM02\0");
    overstated_count.extend_from_slice(&0xFFFF_FFFFu32.to_be_bytes());
    assert!(deserialize_assembly(&overstated_count).is_err());

    // Arbitrary garbage of header length: bad magic, must error cleanly.
    let garbage = [0xABu8; 32];
    assert!(deserialize_assembly(&garbage).is_err());
}

#[test]
fn assem01_back_compat_read_succeeds_with_empty_variants() {
    // Construct a minimal ASSEM01 byte stream by hand (5-byte
    // per-entity header, no entity_id, no variants trailer) and verify
    // the deserializer accepts it under the back-compat path.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"ASSEM01\0");
    bytes.extend_from_slice(&1u32.to_be_bytes()); // entity_count = 1
    bytes.push(0); // mol_type = Protein
    bytes.extend_from_slice(&4u32.to_be_bytes()); // atom_count = 4

    // 4 ASSEM01 atom rows for an ALA residue (26 bytes each).
    for atom in ala_residue_atoms(1.0) {
        bytes.extend_from_slice(&atom.position.x.to_be_bytes());
        bytes.extend_from_slice(&atom.position.y.to_be_bytes());
        bytes.extend_from_slice(&atom.position.z.to_be_bytes());
        bytes.push(b'A'); // chain_id
        bytes.extend_from_slice(&res_bytes("ALA")); // res_name
        bytes.extend_from_slice(&1i32.to_be_bytes()); // res_num (= label_seq_id)
        bytes.extend_from_slice(&atom.name); // atom_name (4 bytes)
        let sym = atom.element.symbol().as_bytes();
        bytes.push(sym.first().copied().unwrap_or(b'X'));
        bytes.push(sym.get(1).copied().unwrap_or(0));
    }

    let rt = deserialize_assembly(&bytes).unwrap();
    let rt_protein = rt.entities()[0].as_protein().unwrap();
    assert_eq!(rt_protein.residues.len(), 1);
    assert!(rt_protein.residues[0].variants.is_empty());
}

#[test]
fn assem02_preserves_entity_id_across_roundtrip() {
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
        b'A',
        None,
    ));

    let bytes = assembly_bytes(&[protein]).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();
    assert_eq!(rt.entities()[0].id(), id);
}

#[test]
fn variants_roundtrip_through_assem02() {
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
        b'A',
        None,
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
        b'A',
        None,
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
