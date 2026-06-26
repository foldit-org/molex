//! The de-vocab marshaling boundary: native [`AtomTable`] columns -> the
//! vocab-bearing per-atom columns [`PyAtomTable`] exposes to Python.
//!
//! Every builder takes the canonical [`AtomTable`] (the one flatten, via
//! [`AtomTable::from_entities`]) plus a half-open flat-atom `range`, and
//! returns one owned column over that range with the egress de-vocab applied:
//! the empty chain -> `"A"` fallback, the res-name / atom-name utf8-trim, the
//! element symbol, and the `mol_type` -> str / `chain_type` -> i32 maps.
//! Numeric columns hand back owned `Vec`s the
//! numpy path moves in via `into_pyarray`; string columns hand back
//! `Vec<String>` the consumer wraps with `numpy.array`.
//!
//! [`PyAtomTable`]: crate::python::PyAtomTable

use super::{molecule_type_to_chain_type_id, molecule_type_to_mol_type_str};
use crate::adapters::table::AtomTable;

/// XYZ coordinates over `range`, three `f32` per atom, row-major.
#[must_use]
pub fn coords_flat(
    table: &AtomTable,
    range: std::ops::Range<usize>,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(range.len() * 3);
    for i in range {
        let pos = table.position[i];
        out.push(pos.x);
        out.push(pos.y);
        out.push(pos.z);
    }
    out
}

/// Chain ids over `range`. Polymer atoms carry the real `label_asym_id`;
/// non-polymer atoms arrive with an empty chain and fall back to `"A"` so the
/// egress consumer always sees a non-empty chain string.
#[must_use]
pub fn chain_ids(
    table: &AtomTable,
    range: std::ops::Range<usize>,
) -> Vec<String> {
    range
        .map(|i| {
            let chain = table.chain_id[i].as_str();
            if chain.is_empty() {
                "A".to_owned()
            } else {
                chain.to_owned()
            }
        })
        .collect()
}

/// Residue sequence numbers over `range`.
#[must_use]
pub fn res_ids(table: &AtomTable, range: std::ops::Range<usize>) -> Vec<i32> {
    range.map(|i| table.res_id[i]).collect()
}

/// Residue names over `range`, utf8-decoded and trimmed.
#[must_use]
pub fn res_names(
    table: &AtomTable,
    range: std::ops::Range<usize>,
) -> Vec<String> {
    range
        .map(|i| {
            std::str::from_utf8(&table.res_name[i])
                .unwrap_or("UNK")
                .trim()
                .to_owned()
        })
        .collect()
}

/// Atom names over `range`, utf8-decoded and trimmed.
#[must_use]
pub fn atom_names(
    table: &AtomTable,
    range: std::ops::Range<usize>,
) -> Vec<String> {
    range
        .map(|i| {
            std::str::from_utf8(&table.name[i])
                .unwrap_or("X")
                .trim()
                .to_owned()
        })
        .collect()
}

/// Element symbols over `range`.
#[must_use]
pub fn elements(
    table: &AtomTable,
    range: std::ops::Range<usize>,
) -> Vec<String> {
    range
        .map(|i| table.element[i].symbol().to_owned())
        .collect()
}

/// Occupancies over `range`.
#[must_use]
pub fn occupancies(
    table: &AtomTable,
    range: std::ops::Range<usize>,
) -> Vec<f32> {
    range.map(|i| table.occupancy[i]).collect()
}

/// B-factors over `range`.
#[must_use]
pub fn b_factors(table: &AtomTable, range: std::ops::Range<usize>) -> Vec<f32> {
    range.map(|i| table.b_factor[i]).collect()
}

/// AtomWorks entity ids over `range` (the native `u32` reinterpreted signed).
#[allow(
    clippy::cast_possible_wrap,
    reason = "entity ids fit in i32 for valid structures"
)]
#[must_use]
pub fn entity_ids(
    table: &AtomTable,
    range: std::ops::Range<usize>,
) -> Vec<i32> {
    range.map(|i| table.entity_id[i].cast_signed()).collect()
}

/// AtomWorks `mol_type` strings over `range`, via the vocab map.
#[must_use]
pub fn mol_types(
    table: &AtomTable,
    range: std::ops::Range<usize>,
) -> Vec<String> {
    range
        .map(|i| molecule_type_to_mol_type_str(table.mol_type[i]).to_owned())
        .collect()
}

/// AtomWorks `chain_type` ids over `range`, via the vocab map.
#[must_use]
pub fn chain_types(
    table: &AtomTable,
    range: std::ops::Range<usize>,
) -> Vec<i32> {
    range
        .map(|i| i32::from(molecule_type_to_chain_type_id(table.mol_type[i])))
        .collect()
}
