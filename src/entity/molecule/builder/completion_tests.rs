//! Guard for missing-atom completion through the file-parse path.
//!
//! Split into a sibling file to mirror the `tests` / `roundtrip_tests`
//! pattern and keep `tests.rs` under the file-length gate. Reuses the
//! `RowBuilder` helpers `tests` exposes via `pub(super)`.

#![allow(clippy::unwrap_used)]

use super::tests::RowBuilder;
use super::*;

/// The file-parse path runs through `EntityBuilder` -> classify, which
/// builds polymers via the completing constructor (`new_normalized` with
/// `Heavy`). An ALA with its four backbone atoms but no CB must come
/// out with CB fabricated; if a classify site reverts to the pure
/// constructor, the CB is absent and this fails.
///
/// Backbone coordinates are real (non-collinear) peptide geometry, not
/// the straight-line placeholder `push_protein_residue` uses, so the
/// rigid fit is fully determined and completion runs.
#[test]
fn hinted_protein_path_completes_missing_cb() {
    let entities = build_backbone_only_ala(Some(ExpectedEntityType::Protein));
    assert_has_cb(
        &entities,
        "hinted protein path (emit_polymer_chain) must fabricate CB; \
         classify.rs must use the completing constructor",
    );
}

/// Same backbone-only ALA with no entity hint, so the chain resolves to
/// `Unknown` and routes through `emit_unknown_chain` ->
/// `emit_unknown_polymer`, the PDB-heuristic polymer emit site. It too
/// must use the completing constructor.
#[test]
fn unknown_chain_path_completes_missing_cb() {
    let entities = build_backbone_only_ala(None);
    assert_has_cb(
        &entities,
        "unknown-chain path (emit_unknown_polymer) must fabricate CB; \
         classify.rs must use the completing constructor",
    );
}

/// Build a single backbone-complete-but-CB-missing ALA chain. With a
/// hint, the chain routes through `emit_polymer_chain`; with `None` the
/// hint resolves to `Unknown` and routes through `emit_unknown_polymer`.
fn build_backbone_only_ala(
    hint: Option<ExpectedEntityType>,
) -> Vec<MoleculeEntity> {
    let backbone = [
        ("N", Element::N, (0.0_f32, 0.0_f32, 0.0_f32)),
        ("CA", Element::C, (1.458, 0.0, 0.0)),
        ("C", Element::C, (2.009, 1.420, 0.0)),
        ("O", Element::O, (1.251, 2.390, 0.0)),
    ];
    let mut b = EntityBuilder::new();
    let entity_id = hint.map(|h| {
        b.register_entity("1", h);
        "1"
    });
    for (name, el, (x, y, z)) in backbone {
        let mut row = RowBuilder::new("A", 1, "ALA", name).at(x, y, z).elem(el);
        if let Some(e) = entity_id {
            row = row.entity(e);
        }
        b.push_atom(row.build()).unwrap();
    }
    b.finish().unwrap()
}

fn assert_has_cb(entities: &[MoleculeEntity], msg: &str) {
    use crate::entity::molecule::traits::Entity;
    let protein = entities[0].as_protein().unwrap();
    let has_cb = protein
        .columns()
        .to_atoms()
        .iter()
        .any(|a| std::str::from_utf8(&a.name).unwrap().trim() == "CB");
    assert!(has_cb, "{msg}");
}
