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
    assert_eq!(from_arcs.hbonds().len(), fresh.hbonds().len());
    assert_eq!(
        from_arcs.cross_entity_bonds().len(),
        fresh.cross_entity_bonds().len(),
    );
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
fn update_positions_bumps_generation_and_moves_atoms() {
    let mut alloc = EntityIdAllocator::new();
    let entity = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let id = entity.id();
    let n_atoms = entity.atom_count();
    let mut assembly = Assembly::new(vec![entity]);
    let before = assembly.generation();

    let shifted: Vec<Vec3> = (0..n_atoms)
        .map(|i| Vec3::new(i as f32, 100.0, 0.0))
        .collect();
    assembly.update_positions(id, &shifted);

    assert_eq!(assembly.generation(), before + 1);
    let updated = assembly.entity(id).unwrap();
    assert!(
        (updated.atom_set()[0].position - Vec3::new(0.0, 100.0, 0.0)).length()
            < 1e-6
    );
}

#[test]
fn replace_entity_preserves_vec_index() {
    let mut alloc = EntityIdAllocator::new();
    let a = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let b = make_dipeptide(&mut alloc, b'B', Vec3::new(20.0, 0.0, 0.0));
    let c = make_dipeptide(&mut alloc, b'C', Vec3::new(40.0, 0.0, 0.0));
    let a_id = a.id();
    let b_id = b.id();
    let c_id = c.id();
    let mut assembly = Assembly::new(vec![a, b, c]);

    // Replace B with a fresh body at a moved origin. Same id, new bytes.
    let b_replacement =
        make_dipeptide_with_id(b_id, b'B', Vec3::new(60.0, 0.0, 0.0));
    assembly.replace_entity(b_id, b_replacement);

    // Order must be unchanged: A at 0, B at 1, C at 2.
    assert_eq!(assembly.entities()[0].id(), a_id);
    assert_eq!(assembly.entities()[1].id(), b_id);
    assert_eq!(assembly.entities()[2].id(), c_id);

    // The replacement bytes are visible at index 1.
    let b_after = assembly.entities()[1].as_protein().unwrap();
    assert!(
        (b_after.atoms[0].position - Vec3::new(60.0, 0.0, 0.0)).length() < 1e-6,
    );
}

#[test]
fn replace_entity_unknown_id_is_a_no_op() {
    let mut alloc = EntityIdAllocator::new();
    let entity = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let known_id = entity.id();
    let mut assembly = Assembly::new(vec![entity]);
    let before = assembly.generation();

    // Mint an id that isn't in this assembly.
    let mut other_alloc = EntityIdAllocator::new();
    let unknown_id = other_alloc.from_raw(9_999);
    let stranger =
        make_dipeptide_with_id(unknown_id, b'Z', Vec3::new(99.0, 0.0, 0.0));
    assembly.replace_entity(unknown_id, stranger);

    // Generation unchanged; entity set unchanged.
    assert_eq!(assembly.generation(), before);
    assert_eq!(assembly.entities().len(), 1);
    assert_eq!(assembly.entities()[0].id(), known_id);
}

#[test]
fn replace_entity_bumps_generation_and_recomputes_derived() {
    let mut alloc = EntityIdAllocator::new();
    let sg_a = Vec3::new(0.0, 0.0, 0.0);
    let sg_b = Vec3::new(2.03, 0.0, 0.0);
    let ca = cys_residue_with_sg(&mut alloc, b'A', sg_a);
    let cb = cys_residue_with_sg(&mut alloc, b'B', sg_b);
    let cb_id = cb.id();

    let mut assembly = Assembly::new(vec![ca, cb]);
    assert_eq!(assembly.cross_entity_bonds().len(), 1);
    let before = assembly.generation();

    // Replace B's CYS with one whose SG is too far for a disulfide.
    let cb_far =
        cys_residue_with_sg_with_id(cb_id, b'B', Vec3::new(50.0, 0.0, 0.0));
    assembly.replace_entity(cb_id, cb_far);

    assert_eq!(assembly.generation(), before + 1);
    assert!(assembly.cross_entity_bonds().is_empty());
}

#[test]
fn update_positions_length_mismatch_is_a_no_op() {
    let mut alloc = EntityIdAllocator::new();
    let entity = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let id = entity.id();
    let mut assembly = Assembly::new(vec![entity]);
    let before = assembly.generation();

    // Too few coords: must be rejected without bumping generation.
    assembly.update_positions(id, &[Vec3::ZERO]);
    assert_eq!(assembly.generation(), before);
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

    assert_eq!(mutated.hbonds().len(), fresh.hbonds().len());
    assert_eq!(
        mutated.cross_entity_bonds().len(),
        fresh.cross_entity_bonds().len(),
    );
    assert_eq!(mutated.ss_types(a_id).len(), fresh.ss_types(a_id).len());
}

// -- Disulfide handling --

#[test]
fn disulfides_filtered_and_cross_bonds_populated() {
    let mut alloc = EntityIdAllocator::new();
    // Two cysteine entities with SGs at ~2.03 A separation.
    let sg_a = Vec3::new(0.0, 0.0, 0.0);
    let sg_b = Vec3::new(2.03, 0.0, 0.0);
    let ca = cys_residue_with_sg(&mut alloc, b'A', sg_a);
    let cb = cys_residue_with_sg(&mut alloc, b'B', sg_b);

    let assembly = Assembly::new(vec![ca, cb]);
    assert_eq!(assembly.cross_entity_bonds().len(), 1);
    assert_eq!(assembly.disulfides().count(), 1);
}

#[test]
fn remove_entity_purges_touching_cross_bonds() {
    let mut alloc = EntityIdAllocator::new();
    let sg_a = Vec3::new(0.0, 0.0, 0.0);
    let sg_b = Vec3::new(2.03, 0.0, 0.0);
    let ca = cys_residue_with_sg(&mut alloc, b'A', sg_a);
    let cb = cys_residue_with_sg(&mut alloc, b'B', sg_b);
    let cb_id = cb.id();

    let mut assembly = Assembly::new(vec![ca, cb]);
    assert_eq!(assembly.cross_entity_bonds().len(), 1);

    assembly.remove_entity(cb_id);
    assert!(assembly.cross_entity_bonds().is_empty());
    assert_eq!(assembly.disulfides().count(), 0);
}

// -- bonds_touching --

#[test]
fn bonds_touching_walks_both_intra_and_cross() {
    let mut alloc = EntityIdAllocator::new();
    let sg_a = Vec3::new(0.0, 0.0, 0.0);
    let sg_b = Vec3::new(2.03, 0.0, 0.0);
    let ca = cys_residue_with_sg(&mut alloc, b'A', sg_a);
    let ca_id = ca.id();
    let cb = cys_residue_with_sg(&mut alloc, b'B', sg_b);
    let assembly = Assembly::new(vec![ca, cb]);

    // SG in chain A sits at index 5 after canonical ordering
    // (N, CA, C, O, CB, SG).
    let sg_atom = AtomId {
        entity: ca_id,
        index: 5,
    };
    let neighbors: Vec<AtomId> = assembly.bonds_touching(sg_atom).collect();

    // Must include the CB neighbor (intra) and the SG partner on
    // chain B (cross). That is at least two.
    assert!(
        neighbors.len() >= 2,
        "expected >=2 neighbors for SG, got {}: {neighbors:?}",
        neighbors.len()
    );
}

// -- CoordinateSnapshot roundtrip --

#[test]
fn coordinate_snapshot_roundtrip_restores_positions() {
    let mut alloc = EntityIdAllocator::new();
    let entity = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let id = entity.id();
    let mut assembly = Assembly::new(vec![entity]);
    let original = CoordinateSnapshot::from_assembly(&assembly);

    // Shift every atom by +10 in z via set_coordinate_snapshot.
    let shifted_positions: Vec<Vec3> = assembly
        .entity(id)
        .unwrap()
        .atom_set()
        .iter()
        .map(|a| a.position + Vec3::new(0.0, 0.0, 10.0))
        .collect();
    let mut per_entity = HashMap::new();
    let _ = per_entity.insert(id, shifted_positions);
    assembly.set_coordinate_snapshot(CoordinateSnapshot::new(per_entity));
    assert!(
        (assembly.entity(id).unwrap().atom_set()[0].position.z - 10.0).abs()
            < 1e-6
    );

    // Restore and confirm we're back.
    assembly.set_coordinate_snapshot(original);
    assert!(assembly.entity(id).unwrap().atom_set()[0].position.z.abs() < 1e-6);
}

#[test]
fn coordinate_snapshot_skips_length_mismatch_but_applies_matching() {
    let mut alloc = EntityIdAllocator::new();
    let a = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let b = make_dipeptide(&mut alloc, b'B', Vec3::new(20.0, 0.0, 0.0));
    let a_id = a.id();
    let b_id = b.id();
    let b_original_pos = b.atom_set()[0].position;
    let mut assembly = Assembly::new(vec![a, b]);
    let gen_before = assembly.generation();

    let a_count = assembly.entity(a_id).unwrap().atom_count();
    let shifted_a: Vec<Vec3> = (0..a_count)
        .map(|i| Vec3::new(i as f32, 50.0, 0.0))
        .collect();
    let too_short_b = vec![Vec3::ZERO];

    let mut per_entity = HashMap::new();
    let _ = per_entity.insert(a_id, shifted_a);
    let _ = per_entity.insert(b_id, too_short_b);
    assembly.set_coordinate_snapshot(CoordinateSnapshot::new(per_entity));

    assert_eq!(assembly.generation(), gen_before + 1);
    assert!(
        (assembly.entity(a_id).unwrap().atom_set()[0].position
            - Vec3::new(0.0, 50.0, 0.0))
        .length()
            < 1e-6
    );
    assert!(
        (assembly.entity(b_id).unwrap().atom_set()[0].position
            - b_original_pos)
            .length()
            < 1e-6
    );
}

#[test]
fn coordinate_snapshot_with_unknown_ids_still_bumps_generation() {
    let mut alloc = EntityIdAllocator::new();
    let a = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let a_id = a.id();
    let original_pos = a.atom_set()[0].position;
    let mut assembly = Assembly::new(vec![a]);
    let gen_before = assembly.generation();

    let mut other_alloc = EntityIdAllocator::new();
    let unknown_id = other_alloc.from_raw(10_000);

    let mut per_entity = HashMap::new();
    let _ = per_entity.insert(unknown_id, vec![Vec3::ZERO; 9]);
    assembly.set_coordinate_snapshot(CoordinateSnapshot::new(per_entity));

    assert_eq!(assembly.generation(), gen_before + 1);
    assert!(
        (assembly.entity(a_id).unwrap().atom_set()[0].position - original_pos)
            .length()
            < 1e-6
    );
}
