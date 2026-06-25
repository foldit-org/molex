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
        id,
        atoms,
        residues,
        String::from(chain as char),
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
        id,
        atoms,
        residues,
        String::from(chain as char),
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

    let mut from_arcs =
        Assembly::from_arcs(vec![Arc::new(a.clone()), Arc::new(b.clone())]);
    let mut fresh = Assembly::new(vec![a, b]);

    // Secondary structure is opt-in; populate both before comparing.
    from_arcs.recompute_ss();
    fresh.recompute_ss();

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
    // A fresh Assembly over the same final entity set must produce the
    // same secondary structure as a mutated Assembly once both opt in:
    // recompute_ss reads the current entity set regardless of how the
    // assembly was assembled.
    let mut alloc = EntityIdAllocator::new();
    let a = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let b = make_dipeptide(&mut alloc, b'B', Vec3::new(20.0, 0.0, 0.0));
    let a_id = a.id();

    // Mutation path: start with just `a`, add `b`.
    let mut mutated = Assembly::new(vec![a.clone()]);
    mutated.add_entity(b.clone());
    mutated.recompute_ss();

    // Fresh path: build from both at once.
    let mut fresh = Assembly::new(vec![a, b]);
    fresh.recompute_ss();

    assert_eq!(mutated.ss_types(a_id).len(), fresh.ss_types(a_id).len());
}

// -- Secondary structure is opt-in --

#[test]
fn construction_is_ss_free_by_default() {
    // A protein that WOULD have a per-residue SS vector after
    // recompute_ss must come out of `new` / `from_arcs` with empty
    // ss_types: construction does not run DSSP.
    let mut alloc = EntityIdAllocator::new();
    let protein = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let id = protein.id();

    let fresh = Assembly::new(vec![protein.clone()]);
    assert!(
        fresh.ss_types(id).is_empty(),
        "Assembly::new must not populate ss_types"
    );

    let from_arcs = Assembly::from_arcs(vec![Arc::new(protein)]);
    assert!(
        from_arcs.ss_types(id).is_empty(),
        "Assembly::from_arcs must not populate ss_types"
    );
}

#[test]
fn recompute_ss_populates_secondary_structure() {
    // After the explicit opt-in, the protein gets a per-residue SS
    // vector (one entry per residue).
    let mut alloc = EntityIdAllocator::new();
    let protein = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let id = protein.id();

    let mut assembly = Assembly::new(vec![protein]);
    assert!(assembly.ss_types(id).is_empty());

    assembly.recompute_ss();
    assert_eq!(
        assembly.ss_types(id).len(),
        2,
        "recompute_ss must populate one SS entry per residue"
    );
}

#[test]
fn carry_ss_from_copies_matching_entities() {
    // A fresh (SS-free) snapshot over the same entity set inherits the
    // source's SS for entities whose residue count matches, leaving its
    // own ss_types entry populated without running DSSP.
    let mut alloc = EntityIdAllocator::new();
    let protein = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let id = protein.id();

    let mut src = Assembly::new(vec![protein.clone()]);
    src.recompute_ss();
    assert_eq!(src.ss_types(id).len(), 2);

    let mut dst = Assembly::new(vec![protein]);
    assert!(dst.ss_types(id).is_empty());

    dst.carry_ss_from(&src);
    assert_eq!(
        dst.ss_types(id),
        src.ss_types(id),
        "matching-length entity must inherit the source SS"
    );
}

#[test]
fn carry_ss_from_skips_residue_count_mismatch() {
    // A length mismatch (a different entity in the source, or an indel)
    // leaves the target's entry untouched: a stale SS can never be carried
    // onto the wrong residues.
    let mut alloc = EntityIdAllocator::new();
    let dst_protein = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let id = dst_protein.id();

    // Source has SS under the SAME id but a different (single-residue)
    // length, so the guard must reject it.
    let src_protein = cys_residue_with_sg_with_id(id, b'A', Vec3::ZERO);
    let mut src = Assembly::new(vec![src_protein]);
    src.recompute_ss();

    let mut dst = Assembly::new(vec![dst_protein]);
    dst.carry_ss_from(&src);
    assert!(
        dst.ss_types(id).is_empty(),
        "a residue-count mismatch must leave ss_types empty"
    );
}

#[test]
fn deserialize_assembly_is_ss_free() {
    // The wire path (serialize -> deserialize_assembly -> Assembly::new)
    // is the per-frame streaming hot path; it must not run DSSP. Every
    // entity comes back with empty ss_types.
    use crate::ops::wire::{assembly_bytes, deserialize_assembly};

    let mut alloc = EntityIdAllocator::new();
    let protein = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let id = protein.id();

    let bytes = assembly_bytes(&[protein]).unwrap();
    let roundtripped = deserialize_assembly(&bytes).unwrap();

    assert!(
        roundtripped.ss_types(id).is_empty(),
        "deserialize_assembly must yield empty ss_types"
    );
}

// -- All-atom projection --

/// Assert every source heavy atom survives the all-atom projection
/// unchanged: per residue, the projected residue's source heavy atoms
/// match by name and position, in order. The projection may additionally
/// fabricate the C-terminal carboxylate OXT on the last residue (the one
/// new heavy atom completion adds), so the projected heavy set is the
/// source heavy set with an optional trailing OXT.
fn assert_heavy_atoms_preserved(
    heavy: &ProteinEntity,
    all_atom: &ProteinEntity,
) {
    use crate::entity::molecule::protein::trimmed_atom_name;
    use crate::entity::molecule::traits::Entity;

    let heavy_all = heavy.columns().to_atoms();
    let all_atom_all = all_atom.columns().to_atoms();
    let last = all_atom.residues.len().saturating_sub(1);
    for (ri, (h_res, a_res)) in
        heavy.residues.iter().zip(&all_atom.residues).enumerate()
    {
        let heavy_atoms: Vec<_> =
            h_res.atom_range.clone().map(|i| &heavy_all[i]).collect();
        let aa_heavy: Vec<_> = a_res
            .atom_range
            .clone()
            .map(|i| &all_atom_all[i])
            .filter(|a| a.element != Element::H)
            .collect();
        // Source heavy atoms come first, preserved by name and position.
        for (ha, aa) in heavy_atoms.iter().zip(&aa_heavy) {
            assert_eq!(
                trimmed_atom_name(&ha.name),
                trimmed_atom_name(&aa.name),
                "heavy atom name must be preserved"
            );
            assert_eq!(
                ha.position, aa.position,
                "heavy atom position must be preserved"
            );
        }
        // The only extra heavy atom the projection may add is OXT, and
        // only on the C-terminal residue.
        let extra: Vec<_> = aa_heavy[heavy_atoms.len()..]
            .iter()
            .map(|a| trimmed_atom_name(&a.name))
            .collect();
        if ri == last {
            assert!(
                extra.is_empty() || extra == [b"OXT".as_slice()],
                "C-terminal residue may gain only OXT, got {extra:?}"
            );
        } else {
            assert!(
                extra.is_empty(),
                "interior residue must gain no heavy atoms, got {extra:?}"
            );
        }
    }
}

#[test]
fn to_all_atom_adds_hydrogens_keeps_heavy_and_is_idempotent() {
    let mut alloc = EntityIdAllocator::new();
    // ALA-GLY dipeptide built heavy-only: both residues resolve to a
    // standard template, so all-atom completion has hydrogens to add.
    let dipep = make_dipeptide(&mut alloc, b'A', Vec3::ZERO);
    let eid = dipep.id();
    let heavy = Assembly::new(vec![dipep]);
    let heavy_protein =
        heavy.entity(eid).unwrap().as_protein().unwrap().clone();
    assert!(
        heavy_protein
            .columns
            .element
            .iter()
            .all(|e| *e != Element::H),
        "heavy-only build must carry no hydrogens"
    );

    // (a) Projection gains hydrogens: atom count rises and at least one
    // H is present.
    let all_atom = heavy.to_all_atom();
    let aa_protein =
        all_atom.entity(eid).unwrap().as_protein().unwrap().clone();
    assert!(
        aa_protein.columns.len() > heavy_protein.columns.len(),
        "all-atom projection must add atoms (got {} vs {})",
        aa_protein.columns.len(),
        heavy_protein.columns.len()
    );
    assert!(
        aa_protein.columns.element.contains(&Element::H),
        "all-atom projection must fabricate hydrogens"
    );

    // (b) Every source heavy atom survives unchanged.
    assert_heavy_atoms_preserved(&heavy_protein, &aa_protein);

    // (c) Idempotent: projecting twice equals projecting once (the second
    // pass finds every template atom already present and adds nothing).
    let twice = all_atom.to_all_atom();
    let twice_protein =
        twice.entity(eid).unwrap().as_protein().unwrap().clone();
    assert_eq!(
        twice_protein.columns.len(),
        aa_protein.columns.len(),
        "to_all_atom must be idempotent"
    );
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
