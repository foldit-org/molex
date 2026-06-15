//! Tests for `assembly.rs`. Split into a sibling file to keep the primary
//! module under the 800-line source cap enforced by `just file-lengths`.

#![allow(clippy::unwrap_used, clippy::float_cmp, clippy::cast_precision_loss)]

use glam::Vec3;

use super::*;
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::id::EntityIdAllocator;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::Residue;

fn mk_atom(name: [u8; 4], el: Element, pos: Vec3) -> Atom {
    Atom {
        position: pos,
        occupancy: 1.0,
        b_factor: 0.0,
        element: el,
        name,
        formal_charge: 0,
    }
}

/// Build a minimal two-residue protein (ALA-GLY) with backbone +
/// one sidechain heavy atom, laid out in canonical order after
/// `ProteinEntity::new`.
fn make_dipeptide(
    alloc: &mut EntityIdAllocator,
    chain: u8,
    origin: Vec3,
) -> MoleculeEntity {
    make_dipeptide_with_id(alloc.allocate(), chain, origin)
}

fn make_dipeptide_with_id(
    id: EntityId,
    chain: u8,
    origin: Vec3,
) -> MoleculeEntity {
    let atoms = vec![
        mk_atom(*b"N   ", Element::N, origin),
        mk_atom(*b"CA  ", Element::C, origin + Vec3::new(1.0, 0.0, 0.0)),
        mk_atom(*b"C   ", Element::C, origin + Vec3::new(2.0, 0.0, 0.0)),
        mk_atom(*b"O   ", Element::O, origin + Vec3::new(2.0, 1.0, 0.0)),
        mk_atom(*b"CB  ", Element::C, origin + Vec3::new(1.0, -1.0, 0.0)),
        mk_atom(*b"N   ", Element::N, origin + Vec3::new(3.2, 0.0, 0.0)),
        mk_atom(*b"CA  ", Element::C, origin + Vec3::new(4.2, 0.0, 0.0)),
        mk_atom(*b"C   ", Element::C, origin + Vec3::new(5.2, 0.0, 0.0)),
        mk_atom(*b"O   ", Element::O, origin + Vec3::new(5.2, 1.0, 0.0)),
    ];
    let residues = vec![
        Residue {
            name: *b"ALA",
            label_seq_id: 1,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: 0..5,
            variants: Vec::new(),
        },
        Residue {
            name: *b"GLY",
            label_seq_id: 2,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: 5..9,
            variants: Vec::new(),
        },
    ];
    MoleculeEntity::Protein(ProteinEntity::new(
        id, atoms, residues, chain, None,
    ))
}

/// A single cysteine residue with a bondable SG at a given position.
fn cys_residue_with_sg(
    alloc: &mut EntityIdAllocator,
    chain: u8,
    sg_pos: Vec3,
) -> MoleculeEntity {
    cys_residue_with_sg_with_id(alloc.allocate(), chain, sg_pos)
}

fn cys_residue_with_sg_with_id(
    id: EntityId,
    chain: u8,
    sg_pos: Vec3,
) -> MoleculeEntity {
    let atoms = vec![
        mk_atom(*b"N   ", Element::N, Vec3::new(0.0, 0.0, 0.0)),
        mk_atom(*b"CA  ", Element::C, Vec3::new(1.0, 0.0, 0.0)),
        mk_atom(*b"C   ", Element::C, Vec3::new(2.0, 0.0, 0.0)),
        mk_atom(*b"O   ", Element::O, Vec3::new(2.0, 1.0, 0.0)),
        mk_atom(*b"CB  ", Element::C, Vec3::new(1.0, -1.0, 0.0)),
        mk_atom(*b"SG  ", Element::S, sg_pos),
    ];
    let residues = vec![Residue {
        name: *b"CYS",
        label_seq_id: 1,
        auth_seq_id: None,
        auth_comp_id: None,
        ins_code: None,
        atom_range: 0..atoms.len(),
        variants: Vec::new(),
    }];
    MoleculeEntity::Protein(ProteinEntity::new(
        id, atoms, residues, chain, None,
    ))
}

// -- Construction + generation --

#[test]
fn new_starts_at_generation_zero() {
    let mut alloc = EntityIdAllocator::new();
    let dipep = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let assembly = Assembly::new(vec![dipep]);
    assert_eq!(assembly.generation(), 0);
}

#[test]
fn new_exposes_all_entities() {
    let mut alloc = EntityIdAllocator::new();
    let a = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let b = make_dipeptide(&mut alloc, b'B', Vec3::new(20.0, 0.0, 0.0));
    let assembly = Assembly::new(vec![a, b]);
    assert_eq!(assembly.entities().len(), 2);
}

#[test]
fn from_arcs_matches_new_over_same_entities() {
    // `from_arcs` is the Arc-preserving sibling of `new`: over the same
    // entity set it must produce identical observable and derived state.
    // Build both dipeptides once, then compare an Assembly built from
    // pre-`Arc`ed clones against one built from the owned values.
    let mut alloc = EntityIdAllocator::new();
    let a = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let b = make_dipeptide(&mut alloc, b'B', Vec3::new(20.0, 0.0, 0.0));
    let a_id = a.id();
    let b_id = b.id();

    let from_arcs =
        Assembly::from_arcs(vec![Arc::new(a.clone()), Arc::new(b.clone())]);
    let fresh = Assembly::new(vec![a, b]);

    // Same entity count and ids, in order.
    assert_eq!(from_arcs.entities().len(), fresh.entities().len());
    let from_arcs_ids: Vec<EntityId> =
        from_arcs.entities().iter().map(|e| e.id()).collect();
    let fresh_ids: Vec<EntityId> =
        fresh.entities().iter().map(|e| e.id()).collect();
    assert_eq!(from_arcs_ids, fresh_ids);

    // Generation starts at 0, same as `new`.
    assert_eq!(from_arcs.generation(), 0);
    assert_eq!(from_arcs.generation(), fresh.generation());

    // Same derived outputs.
    assert_eq!(from_arcs.ss_types(a_id).len(), fresh.ss_types(a_id).len());
    assert_eq!(from_arcs.ss_types(b_id).len(), fresh.ss_types(b_id).len());
}

// -- Mutation generation + recompute --

#[test]
fn add_entity_bumps_generation_exactly_once() {
    let mut alloc = EntityIdAllocator::new();
    let mut assembly =
        Assembly::new(vec![make_dipeptide(&mut alloc, b'A', Vec3::ZERO)]);
    let before = assembly.generation();
    assembly.add_entity(make_dipeptide(
        &mut alloc,
        b'B',
        Vec3::new(20.0, 0.0, 0.0),
    ));
    assert_eq!(assembly.generation(), before + 1);
}

#[test]
fn remove_entity_bumps_generation_exactly_once() {
    let mut alloc = EntityIdAllocator::new();
    let a = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let b_id;
    let b = {
        let e = make_dipeptide(&mut alloc, b'B', Vec3::new(20.0, 0.0, 0.0));
        b_id = e.id();
        e
    };
    let mut assembly = Assembly::new(vec![a, b]);
    let before = assembly.generation();
    assembly.remove_entity(b_id);
    assert_eq!(assembly.generation(), before + 1);
    assert_eq!(assembly.entities().len(), 1);
}

#[test]
fn mutation_result_matches_fresh_build() {
    // A fresh Assembly over the same final entity set must produce
    // the same derived outputs as a mutated Assembly: that is the
    // contract "derived data is recomputed on every mutation".
    let mut alloc = EntityIdAllocator::new();
    let a = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let b = make_dipeptide(&mut alloc, b'B', Vec3::new(20.0, 0.0, 0.0));
    let a_id = a.id();

    // Mutation path: start with just `a`, add `b`.
    let mut mutated = Assembly::new(vec![a.clone()]);
    mutated.add_entity(b.clone());

    // Fresh path: build from both at once.
    let fresh = Assembly::new(vec![a, b]);

    assert_eq!(mutated.ss_types(a_id).len(), fresh.ss_types(a_id).len());
}

// -- Connections (rendering metadata) --

#[test]
fn connections_default_empty() {
    let mut alloc = EntityIdAllocator::new();
    let dipep = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let assembly = Assembly::new(vec![dipep]);
    assert!(assembly.connections().is_empty());
}

#[test]
fn set_connections_roundtrips() {
    use crate::connection::{AtomEnd, AtomLink, ConnectionType};

    let mut alloc = EntityIdAllocator::new();
    let dipep = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let eid = dipep.id();
    let mut assembly = Assembly::new(vec![dipep]);

    let mut map: HashMap<ConnectionType, Vec<AtomLink>> = HashMap::new();
    let link = AtomLink::new(
        AtomEnd::Atom(AtomId {
            entity: eid,
            index: 0,
        }),
        AtomEnd::Atom(AtomId {
            entity: eid,
            index: 2,
        }),
    );
    let _ = map.insert(ConnectionType::Clash, vec![link]);
    assembly.set_connections(map);

    let got = assembly.connections();
    assert_eq!(got.len(), 1);
    assert_eq!(got.get(&ConnectionType::Clash), Some(&vec![link]));
}

#[test]
fn detect_fallback_connections_finds_disulfide() {
    use crate::connection::{AtomEnd, ConnectionType};

    let mut alloc = EntityIdAllocator::new();
    let sg_a = Vec3::new(0.0, 0.0, 0.0);
    let sg_b = Vec3::new(2.03, 0.0, 0.0);
    let ca = cys_residue_with_sg(&mut alloc, b'A', sg_a);
    let cb = cys_residue_with_sg(&mut alloc, b'B', sg_b);
    let assembly = Assembly::new(vec![ca, cb]);

    let fallback = assembly.detect_fallback_connections();
    let disulfides = fallback.get(&ConnectionType::Disulfide).unwrap();
    assert_eq!(disulfides.len(), 1);
    let link = disulfides[0];
    assert!(matches!(link.a, AtomEnd::Atom(_)));
    assert!(matches!(link.b, AtomEnd::Atom(_)));
    // Fallback geometry carries no intensity.
    assert_eq!(link.magnitude, None);
}

#[test]
fn with_magnitude_carries_scalar() {
    use crate::connection::{AtomEnd, AtomLink};

    let mut alloc = EntityIdAllocator::new();
    let end = AtomEnd::Atom(AtomId {
        entity: alloc.allocate(),
        index: 0,
    });
    let link = AtomLink::with_magnitude(end, end, 0.75);
    assert_eq!(link.magnitude, Some(0.75));
}
