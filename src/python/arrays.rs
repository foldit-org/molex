//! Flat-atom index lookup for mapping bond endpoints into coordinate-row space.
//!
//! Bond and disulfide detection report endpoints as `(entity, per-entity atom
//! index)` pairs; [`FlatAtoms`] maps those to the atom's position in molex's
//! canonical flat atom order, so the indices `PyAssembly::covalent_bonds` /
//! `disulfides` return line up with the coordinate rows a caller reads.

use std::collections::HashMap;

use crate::adapters::table::AtomTable;
use crate::entity::molecule::MoleculeEntity;

/// Lookup from per-entity atom identity to flat atom-index space.
///
/// `flat_of` maps `(entity raw id, atom index within that entity)` to the
/// atom's position in molex's canonical flat order (the order
/// [`AtomTable::from_entities`] lays atoms out, the single source of that
/// ordering).
pub(crate) struct FlatAtoms {
    pub flat_of: HashMap<(u32, u32), u32>,
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "flat atom count fits in u32 for valid structures"
)]
pub(crate) fn flat_atoms<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
) -> FlatAtoms {
    let flat_of = AtomTable::flat_source_indices(entities)
        .into_iter()
        .enumerate()
        .map(|(flat, src)| (src, flat as u32))
        .collect();

    FlatAtoms { flat_of }
}
