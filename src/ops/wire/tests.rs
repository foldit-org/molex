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
        observed: true,
    }
}

fn atom_bo(
    name: &str,
    element: Element,
    x: f32,
    b_factor: f32,
    occupancy: f32,
) -> Atom {
    Atom {
        occupancy,
        b_factor,
        ..atom_at(name, element, x)
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
        String::from("B"),
    ));
    let zinc = MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
        allocator.allocate(),
        MoleculeType::Ion,
        vec![atom_at("ZN", Element::Zn, 20.0)],
        res_bytes("ZN"),
        String::from("C"),
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
        String::from("B"),
    ));
    let entities = vec![ion];
    let bytes = assembly_bytes(&entities).unwrap();
    let roundtripped = deserialize_assembly(&bytes).unwrap();
    assert_eq!(roundtripped.entities().len(), 1);
    assert_eq!(
        roundtripped.entities()[0].molecule_type(),
        MoleculeType::Ion
    );
    assert!((roundtripped.entities()[0].positions()[0].x - 5.5).abs() < 1e-6);
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
    // len + 1 chain byte "A") + 4 atoms * 33 (v2 row) + 4 variant-count u32
    // (zero, no variants on this residue).
    assert_eq!(&bytes[0..8], b"ASSEMBLY");
    assert_eq!(bytes[8], 2); // version
    assert_eq!(u32::from_be_bytes(bytes[9..13].try_into().unwrap()), 1);
    assert_eq!(bytes[13], 0); // Protein
    assert_eq!(u32::from_be_bytes(bytes[14..18].try_into().unwrap()), 4);
    // bytes[18..22] is the 4-byte entity_id; bytes[22..24] the u16 chain
    // length (1); bytes[24] the chain byte (b'A').
    assert_eq!(u16::from_be_bytes(bytes[22..24].try_into().unwrap()), 1);
    assert_eq!(bytes[24], b'A');
    assert_eq!(bytes.len(), 8 + 1 + 4 + (9 + 2 + 1) + 4 * 33 + 4);
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
fn nonempty_dropped_tail_polymer_roundtrips() {
    // A polymer with a non-empty dropped-residue tail: one complete ALA
    // (N/CA/C/O) followed by a backbone-incomplete residue (CB+CG only).
    // canonicalize drops the incomplete residue from the residue list but
    // leaves its two atoms in the storage columns, unreferenced, so
    // columns().len() (6) exceeds the sum of residue atom_range lengths (4).
    // The body walks residue-range atoms (4); the header must declare 4, not
    // 6, or deserialize reads past the atom data and fails "Data too short".
    let id = EntityIdAllocator::new().allocate();
    let atoms = {
        let mut v = ala_residue_atoms(1.0);
        v.push(atom_at("CB", Element::C, 5.0));
        v.push(atom_at("CG", Element::C, 6.0));
        v
    };
    let residues = vec![residue("ALA", 1, 0..4), residue("XYZ", 2, 4..6)];
    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        atoms,
        residues,
        "A".to_owned(),
    ));

    // canonicalize kept residue 1 (4 atoms) and dropped residue 2 to the tail.
    let p = protein.as_protein().unwrap();
    assert_eq!(p.residues.len(), 1, "incomplete residue dropped");
    assert_eq!(p.columns.len(), 6, "dropped atoms stay in the columns");
    let referenced: usize = p.residues.iter().map(|r| r.atom_range.len()).sum();
    assert_eq!(referenced, 4, "non-empty tail: 6 stored, 4 referenced");

    // Round-trip must not error: the header now declares the 4 body rows.
    let bytes = assembly_bytes(&[protein]).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();

    let rt_protein = rt.entities()[0].as_protein().unwrap();
    assert_eq!(rt_protein.residues.len(), 1);
    assert_eq!(rt_protein.residues[0].label_seq_id, 1);
    // The body emits only the kept residue's atoms; the tail is dropped on the
    // wire (no residue references it), so the rebuilt entity carries 4 atoms.
    assert_eq!(rt.entities()[0].atom_count(), 4);
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
        String::from("W"),
    ));
    let entities = vec![water];

    let bytes = assembly_bytes(&entities).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();
    assert_eq!(rt.entities()[0].molecule_type(), MoleculeType::Water);
    assert_eq!(rt.entities()[0].atom_count(), 2);
    assert_eq!(rt.entities()[0].as_bulk().unwrap().molecule_count, 2);
}

#[test]
fn assembly_bytes_preserves_nonpolymer_chains() {
    // Regression guard: a non-polymer's chain must survive the assembly-bytes
    // round-trip. A water and a ligand on chains distinct from the chain-A
    // protein must not collapse onto "A" on egress (the collision that made a
    // downstream `(chain, res_id)` tokenizer see two residues at A/1).
    let mut allocator = EntityIdAllocator::new();

    let protein = MoleculeEntity::Protein(ProteinEntity::new(
        allocator.allocate(),
        ala_residue_atoms(1.0),
        vec![residue("MET", 1, 0..4)],
        "A".to_owned(),
    ));
    let water = MoleculeEntity::Bulk(BulkEntity::new(
        allocator.allocate(),
        MoleculeType::Water,
        vec![
            atom_at("O", Element::O, 30.0),
            atom_at("O", Element::O, 31.0),
        ],
        res_bytes("HOH"),
        2,
        String::from("W"),
    ));
    let ligand = MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
        allocator.allocate(),
        MoleculeType::Ligand,
        vec![atom_at("C1", Element::C, 40.0)],
        res_bytes("ATP"),
        String::from("B"),
    ));

    let entities = vec![protein, water, ligand];
    let bytes = assembly_bytes(&entities).unwrap();
    let rt = deserialize_assembly(&bytes).unwrap();

    let chains: Vec<Option<&str>> =
        rt.entities().iter().map(|e| e.pdb_chain_id()).collect();
    assert_eq!(chains, vec![Some("A"), Some("W"), Some("B")]);
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
    // A well-formed header with an unsupported version byte must be a clean
    // error, not a panic or a misparse. Versions 1 and 2 are valid.
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
    // A multi-character chain id ("AA") is impossible under a single-byte
    // model; the per-entity header chain string carries it.
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
fn assembly_roundtrip_preserves_b_factor_and_occupancy() {
    let id = EntityIdAllocator::new().allocate();
    let ligand = MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
        id,
        MoleculeType::Ligand,
        vec![
            atom_bo("C1", Element::C, 10.0, 12.5, 0.75),
            atom_bo("N2", Element::N, 11.0, 33.25, 0.5),
            atom_bo("O3", Element::O, 12.0, 7.125, 1.0),
        ],
        res_bytes("ATP"),
        String::from("B"),
    ));

    let bytes = assembly_bytes(&[ligand]).unwrap();
    assert_eq!(bytes[8], 2);
    let rt = deserialize_assembly(&bytes).unwrap();
    let cols = rt.entities()[0].columns();

    let mut got: Vec<([u8; 4], f32, f32)> = (0..cols.len())
        .map(|i| (cols.name[i], cols.b_factor[i], cols.occupancy[i]))
        .collect();
    got.sort_by_key(|(n, _, _)| *n);
    assert_eq!(
        got,
        vec![
            (*b"C1  ", 12.5, 0.75),
            (*b"N2  ", 33.25, 0.5),
            (*b"O3  ", 7.125, 1.0),
        ],
    );
}

#[test]
fn assembly_v1_payload_decodes_with_defaults() {
    // A v1 payload (25-byte rows, no b_factor/occupancy) must still decode;
    // the two absent fields fall back to their defaults (0.0 / 1.0).
    use crate::entity::molecule::atom::AtomRef;
    use crate::ops::wire::serialize::write_atom_row;
    use crate::ops::wire::RowLayout;

    // The source atom carries non-default values, but the v1 row cannot hold
    // them, so the decoded atom must show the defaults instead.
    let atom = atom_bo("ZN", Element::Zn, 5.5, 42.0, 0.5);

    let mut buf = Vec::new();
    buf.extend_from_slice(b"ASSEMBLY");
    buf.push(1); // v1
    buf.extend_from_slice(&1u32.to_be_bytes()); // 1 entity
    buf.push(4); // Ion molecule-type wire byte
    buf.extend_from_slice(&1u32.to_be_bytes()); // atom_count = 1
    buf.extend_from_slice(&0u32.to_be_bytes()); // entity_id = 0
    buf.extend_from_slice(&1u16.to_be_bytes()); // chain_len = 1
    buf.push(b'B'); // chain byte
    write_atom_row(
        AtomRef::from_atom(&atom),
        res_bytes("ZN"),
        1,
        RowLayout::V1,
        &mut buf,
    );
    buf.extend_from_slice(&0u32.to_be_bytes()); // variants: zero for the entity

    let rt = deserialize_assembly(&buf).unwrap();
    let cols = rt.entities()[0].columns();
    assert_eq!(cols.b_factor[0], 0.0);
    assert_eq!(cols.occupancy[0], 1.0);
    assert_eq!(cols.position[0].x, 5.5);
}

#[test]
fn delta_mutate_residue_preserves_b_factor_and_occupancy() {
    use crate::ops::edit::AssemblyEdit;
    use crate::ops::wire::delta::{deserialize_edits, serialize_edits};

    let id = EntityIdAllocator::new().allocate();
    let new_atoms = vec![
        atom_bo("N", Element::N, 0.0, 11.0, 0.9),
        atom_bo("CA", Element::C, 1.0, 22.5, 0.8),
    ];
    let edits = vec![AssemblyEdit::MutateResidue {
        entity: id,
        residue_idx: 3,
        new_name: res_bytes("HIS"),
        new_atoms,
        new_variants: Vec::new(),
    }];

    let bytes = serialize_edits(&edits).unwrap();
    assert_eq!(bytes[8], 2);
    let rt = deserialize_edits(&bytes).unwrap();
    let AssemblyEdit::MutateResidue {
        new_atoms: rt_atoms,
        ..
    } = &rt[0]
    else {
        unreachable!("unexpected edit: {:?}", rt[0]);
    };
    assert_eq!(rt_atoms[0].b_factor, 11.0);
    assert_eq!(rt_atoms[0].occupancy, 0.9);
    assert_eq!(rt_atoms[1].b_factor, 22.5);
    assert_eq!(rt_atoms[1].occupancy, 0.8);
}

#[test]
fn delta_v1_payload_decodes_with_defaults() {
    // A v1 delta MutateResidue payload (25-byte rows) must still decode; the
    // absent b_factor/occupancy fall back to their defaults (0.0 / 1.0).
    use crate::entity::molecule::atom::AtomRef;
    use crate::ops::edit::AssemblyEdit;
    use crate::ops::wire::delta::{deserialize_edits, DELTA_MAGIC};
    use crate::ops::wire::serialize::write_atom_row;
    use crate::ops::wire::RowLayout;

    let atoms = vec![
        atom_bo("N", Element::N, 0.0, 11.0, 0.9),
        atom_bo("CA", Element::C, 1.0, 22.5, 0.8),
    ];

    let mut buf = Vec::new();
    buf.extend_from_slice(DELTA_MAGIC);
    buf.push(1); // v1
    buf.extend_from_slice(&1u32.to_be_bytes()); // 1 edit
    buf.push(0x03); // TAG_MUTATE_RESIDUE
    buf.extend_from_slice(&0u32.to_be_bytes()); // entity_id raw
    buf.extend_from_slice(&3u32.to_be_bytes()); // residue_idx
    buf.extend_from_slice(&res_bytes("HIS")); // new_name (3 bytes)
    buf.extend_from_slice(&2u32.to_be_bytes()); // atom_count = 2
    for a in &atoms {
        write_atom_row(
            AtomRef::from_atom(a),
            res_bytes("HIS"),
            0,
            RowLayout::V1,
            &mut buf,
        );
    }
    buf.extend_from_slice(&0u32.to_be_bytes()); // variant_count = 0

    let rt = deserialize_edits(&buf).unwrap();
    let AssemblyEdit::MutateResidue {
        new_atoms: rt_atoms,
        ..
    } = &rt[0]
    else {
        unreachable!("unexpected edit: {:?}", rt[0]);
    };
    assert_eq!(rt_atoms.len(), 2);
    for a in rt_atoms {
        assert_eq!(a.b_factor, 0.0);
        assert_eq!(a.occupancy, 1.0);
    }
}
