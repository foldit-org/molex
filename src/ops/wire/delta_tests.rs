#![allow(clippy::unwrap_used, clippy::float_cmp)]

use glam::Vec3;

use super::{
    deserialize_edits, serialize_edits, DeltaSerializeError, DELTA_MAGIC,
    DELTA_VERSION,
};
use crate::chemistry::variant::{ProtonationState, VariantTag};
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::id::EntityIdAllocator;
use crate::ops::edit::AssemblyEdit;

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

#[test]
fn magic_header_is_emitted_first() {
    let bytes = serialize_edits(&[]).unwrap();
    assert_eq!(&bytes[0..8], DELTA_MAGIC);
    assert_eq!(bytes[8], DELTA_VERSION);
    // Empty list -> 0 edits.
    assert_eq!(u32::from_be_bytes(bytes[9..13].try_into().unwrap()), 0);
    assert_eq!(bytes.len(), 13);
}

#[test]
fn set_entity_coords_roundtrips() {
    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();

    let coords = vec![
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(-7.5, 8.25, -9.125),
    ];
    let edits = vec![AssemblyEdit::SetEntityCoords {
        entity: id,
        coords: coords.clone(),
    }];

    let bytes = serialize_edits(&edits).unwrap();
    let rt = deserialize_edits(&bytes).unwrap();
    assert_eq!(rt.len(), 1);
    let AssemblyEdit::SetEntityCoords {
        entity: rt_entity,
        coords: rt_coords,
    } = &rt[0]
    else {
        unreachable!("unexpected edit: {:?}", rt[0]);
    };
    assert_eq!(*rt_entity, id);
    assert_eq!(rt_coords, &coords);
}

#[test]
fn set_residue_coords_roundtrips() {
    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();

    let edits = vec![AssemblyEdit::SetResidueCoords {
        entity: id,
        residue_idx: 17,
        coords: vec![Vec3::ONE, Vec3::Y, Vec3::Z],
    }];

    let bytes = serialize_edits(&edits).unwrap();
    let rt = deserialize_edits(&bytes).unwrap();
    let AssemblyEdit::SetResidueCoords {
        entity: rt_entity,
        residue_idx,
        coords,
    } = &rt[0]
    else {
        unreachable!("unexpected edit: {:?}", rt[0]);
    };
    assert_eq!(*rt_entity, id);
    assert_eq!(*residue_idx, 17);
    assert_eq!(coords.len(), 3);
}

#[test]
fn mutate_residue_roundtrips_with_variants() {
    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();

    let new_atoms = vec![
        atom_at("N", Element::N, 0.0),
        atom_at("CA", Element::C, 1.0),
        atom_at("C", Element::C, 2.0),
    ];
    let edits = vec![AssemblyEdit::MutateResidue {
        entity: id,
        residue_idx: 4,
        new_name: res_bytes("HIS"),
        new_atoms,
        new_variants: vec![VariantTag::Protonation(
            ProtonationState::HisEpsilon,
        )],
    }];

    let bytes = serialize_edits(&edits).unwrap();
    let rt = deserialize_edits(&bytes).unwrap();
    let AssemblyEdit::MutateResidue {
        entity: rt_entity,
        residue_idx,
        new_name,
        new_atoms: rt_atoms,
        new_variants,
    } = &rt[0]
    else {
        unreachable!("unexpected edit: {:?}", rt[0]);
    };
    assert_eq!(*rt_entity, id);
    assert_eq!(*residue_idx, 4);
    assert_eq!(new_name, &res_bytes("HIS"));
    assert_eq!(rt_atoms.len(), 3);
    assert!(matches!(
        new_variants[0],
        VariantTag::Protonation(ProtonationState::HisEpsilon),
    ));
}

#[test]
fn set_variants_roundtrips() {
    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();

    let edits = vec![AssemblyEdit::SetVariants {
        entity: id,
        residue_idx: 0,
        variants: vec![
            VariantTag::NTerminus,
            VariantTag::Other("custom".to_owned()),
        ],
    }];

    let bytes = serialize_edits(&edits).unwrap();
    let rt = deserialize_edits(&bytes).unwrap();
    let AssemblyEdit::SetVariants {
        entity: rt_entity,
        residue_idx,
        variants,
    } = &rt[0]
    else {
        unreachable!("unexpected edit: {:?}", rt[0]);
    };
    assert_eq!(*rt_entity, id);
    assert_eq!(*residue_idx, 0);
    assert_eq!(variants.len(), 2);
    assert!(matches!(variants[0], VariantTag::NTerminus));
    assert!(matches!(&variants[1], VariantTag::Other(s) if s == "custom"));
}

#[test]
fn multiple_edits_roundtrip_in_order() {
    let mut alloc = EntityIdAllocator::new();
    let a = alloc.allocate();
    let b = alloc.allocate();

    let edits = vec![
        AssemblyEdit::SetEntityCoords {
            entity: a,
            coords: vec![Vec3::ZERO],
        },
        AssemblyEdit::SetVariants {
            entity: b,
            residue_idx: 2,
            variants: vec![VariantTag::Disulfide],
        },
        AssemblyEdit::SetResidueCoords {
            entity: a,
            residue_idx: 0,
            coords: vec![Vec3::ONE; 4],
        },
    ];

    let bytes = serialize_edits(&edits).unwrap();
    let rt = deserialize_edits(&bytes).unwrap();
    assert_eq!(rt.len(), 3);
    assert!(matches!(rt[0], AssemblyEdit::SetEntityCoords { .. }));
    assert!(matches!(rt[1], AssemblyEdit::SetVariants { .. }));
    assert!(matches!(rt[2], AssemblyEdit::SetResidueCoords { .. }));
}

#[test]
fn topology_edits_are_rejected_at_serialize_time() {
    use crate::entity::molecule::polymer::Residue;
    use crate::entity::molecule::protein::ProteinEntity;
    use crate::entity::molecule::MoleculeEntity;

    let mut alloc = EntityIdAllocator::new();
    let id = alloc.allocate();
    let residues = vec![Residue {
        name: res_bytes("ALA"),
        label_seq_id: 1,
        auth_seq_id: None,
        auth_comp_id: None,
        ins_code: None,
        atom_range: 0..1,
        variants: Vec::new(),
    }];
    let entity = MoleculeEntity::Protein(ProteinEntity::new(
        id,
        vec![atom_at("CA", Element::C, 0.0)],
        residues,
        "A".to_owned(),
    ));

    let edits = vec![
        AssemblyEdit::SetEntityCoords {
            entity: id,
            coords: vec![Vec3::ZERO],
        },
        AssemblyEdit::AddEntity { entity },
    ];

    let err = serialize_edits(&edits).unwrap_err();
    assert!(matches!(
        err,
        DeltaSerializeError::TopologyEditNotSupported { index: 1 },
    ));
}

#[test]
fn deserialize_rejects_bad_magic() {
    let mut bytes = vec![0u8; 13];
    bytes[..8].copy_from_slice(b"BADMAGIC");
    assert!(deserialize_edits(&bytes).is_err());
}

#[test]
fn deserialize_rejects_unknown_version() {
    let mut bytes = vec![0u8; 13];
    bytes[..8].copy_from_slice(DELTA_MAGIC);
    bytes[8] = DELTA_VERSION.wrapping_add(1); // unsupported version
    assert!(deserialize_edits(&bytes).is_err());
}

#[test]
fn deserialize_rejects_truncated_header() {
    assert!(deserialize_edits(&[0u8; 4]).is_err());
}
