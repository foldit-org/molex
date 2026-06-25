//! ML model-output adapter: host numpy arrays -> assembly wire bytes.
//!
//! Folding model plugins (SimpleFold and other Boltz-`Structure` producers)
//! emit a structured-array description of the predicted complex: a flat atom
//! table (coordinates, presence mask, encoded atom names) plus a residue table
//! (per-residue atom range and 3-letter name). This adapter builds molex
//! entities directly from those host-resident columns through the shared
//! [`AtomTable`] partition core, with no Biotite `AtomArray` in between.
//!
//! The conversion logic lives in a pure-Rust core over native slices, so it is
//! unit-testable without a Python interpreter. The
//! [`boltz_structure_to_entities`] `#[pyfunction]` is a thin shim that borrows
//! the numpy arrays and forwards to the core.

use compact_str::CompactString;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::adapters::table::AtomTable;
use crate::assembly::Assembly;
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::MoleculeType;
use crate::ops::wire::serialize_assembly;

/// Canonical atom37 ordering (AF2 / OpenFold standard).
///
/// Used only for the packed-atom37 layout, where the flat atom table has no
/// per-atom names and each atom's name is its slot in this table modulo 37.
const ATOM37_NAMES: [&str; 37] = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "OD1", "OD2",
    "CD", "CD1", "CD2", "ND1", "ND2", "OE1", "OE2", "NE", "NE1", "NE2", "CE",
    "CE1", "CE2", "CE3", "NH1", "NH2", "CZ", "CZ2", "CZ3", "CH2", "OH", "SG",
    "SD", "NZ", "X",
];

/// All the per-atom and per-residue columns of a Boltz `Structure`, already
/// extracted to host-resident slices.
///
/// `coords`, `atom_names`, and `is_present` are per-atom (length `N`);
/// `res_atom_idx`, `res_atom_num`, and `res_names` are per-residue (length
/// `R`); `plddt`, when present, is per-residue confidence in the 0-1 range.
pub(crate) struct BoltzColumns<'a> {
    pub coords: &'a [[f32; 3]],
    pub atom_names: &'a [[i8; 4]],
    pub is_present: &'a [bool],
    pub res_atom_idx: &'a [i32],
    pub res_atom_num: &'a [i32],
    pub res_names: &'a [String],
    pub plddt: Option<&'a [f32]>,
}

/// Build assembly wire bytes from Boltz structured-array columns.
///
/// Mirrors the reference `boltz_structure_to_atom_array` conversion, but emits
/// molex entities directly. Atoms are grouped into one protein entity in
/// source order; residue boundaries follow the per-residue atom ranges. Atoms
/// that are absent or have a non-finite coordinate are dropped. Atom names are
/// taken from the packed atom37 table when the atom count is exactly `37 * R`,
/// otherwise decoded from the Boltz `int8[4]` (ASCII-minus-32) encoding.
///
/// # Errors
///
/// Returns `AdapterError` if assembly serialization fails.
pub(crate) fn boltz_columns_to_assembly_bytes(
    columns: &BoltzColumns<'_>,
) -> Result<Vec<u8>, crate::ops::error::AdapterError> {
    let table = boltz_columns_to_table(columns);
    if table.is_empty() {
        return serialize_assembly(&Assembly::new(vec![]));
    }
    serialize_assembly(&Assembly::new(table.into_entities()))
}

/// Decode the Boltz columns into an ordered single-protein-entity
/// [`AtomTable`].
///
/// The conversion logic in isolation, before [`AtomTable::into_entities`]
/// groups the rows into a protein entity (which canonicalizes atom order and
/// may drop backbone-incomplete residues). Absent atoms and atoms with a
/// non-finite coordinate are skipped, so the result has at most one row per
/// present, finite source atom. Every row is tagged entity 0, chain "A",
/// [`MoleculeType::Protein`].
fn boltz_columns_to_table(columns: &BoltzColumns<'_>) -> AtomTable {
    let n_atoms = columns.coords.len();
    let mut table = AtomTable::with_capacity(n_atoms);
    if n_atoms == 0 {
        return table;
    }
    let n_residues = columns.res_names.len();
    let atom_to_residue = map_atoms_to_residues(columns, n_atoms);
    let is_atom37 = n_atoms == n_residues.saturating_mul(37);

    for (i, coord) in columns.coords.iter().enumerate() {
        let present = columns.is_present.get(i).copied().unwrap_or(false);
        if !present || coord.iter().any(|c| c.is_nan()) {
            continue;
        }

        let res_i = atom_to_residue[i];
        let res_name = residue_name_bytes(columns.res_names, res_i);
        let atom_name = if is_atom37 {
            atom37_name(i)
        } else {
            decode_boltz_atom_name(columns.atom_names.get(i))
        };
        let b_factor = plddt_to_b_factor(columns.plddt, res_i);

        let an_str = std::str::from_utf8(&atom_name).unwrap_or("");
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "residue index fits in i32 for model outputs"
        )]
        let res_num = res_i as i32;
        table.push_row(
            &Atom {
                position: glam::Vec3::new(coord[0], coord[1], coord[2]),
                occupancy: 1.0,
                b_factor,
                element: Element::from_atom_name(an_str),
                name: atom_name,
                formal_charge: 0,
                observed: true,
            },
            0,
            CompactString::new("A"),
            MoleculeType::Protein,
            res_num,
            res_name,
        );
    }
    table
}

/// Per-atom residue index, derived from the per-residue `(atom_idx, atom_num)`
/// ranges. Atoms outside every range stay assigned to residue 0.
fn map_atoms_to_residues(
    columns: &BoltzColumns<'_>,
    n_atoms: usize,
) -> Vec<usize> {
    let mut atom_to_residue = vec![0usize; n_atoms];
    let n_residues = columns.res_atom_idx.len().min(columns.res_atom_num.len());
    for res_i in 0..n_residues {
        #[allow(
            clippy::cast_sign_loss,
            reason = "negated by .max(0): the value is non-negative"
        )]
        let start = columns.res_atom_idx[res_i].max(0) as usize;
        #[allow(
            clippy::cast_sign_loss,
            reason = "negated by .max(0): the value is non-negative"
        )]
        let count = columns.res_atom_num[res_i].max(0) as usize;
        let end = start.saturating_add(count).min(n_atoms);
        for slot in atom_to_residue.iter_mut().take(end).skip(start) {
            *slot = res_i;
        }
    }
    atom_to_residue
}

/// 3-byte residue name for residue `res_i`, space-padded; `UNK` when the
/// residue is out of range or its name is blank.
fn residue_name_bytes(res_names: &[String], res_i: usize) -> [u8; 3] {
    let name = res_names.get(res_i).map_or("", |s| s.trim());
    let name = if name.is_empty() { "UNK" } else { name };
    let mut out = [b' '; 3];
    for (slot, byte) in out.iter_mut().zip(name.bytes().take(3)) {
        *slot = byte;
    }
    out
}

/// Atom name for the packed atom37 layout: the slot is the atom index modulo
/// 37, looked up in [`ATOM37_NAMES`].
fn atom37_name(atom_index: usize) -> [u8; 4] {
    let slot = atom_index % 37;
    let name = ATOM37_NAMES.get(slot).copied().unwrap_or("X");
    let mut out = [b' '; 4];
    for (slot, byte) in out.iter_mut().zip(name.bytes().take(4)) {
        *slot = byte;
    }
    out
}

/// Decode a Boltz `int8[4]` atom name: each non-zero byte is an ASCII code
/// offset down by 32, clamped into the printable range. Empty decodes to `X`.
fn decode_boltz_atom_name(encoded: Option<&[i8; 4]>) -> [u8; 4] {
    let mut out = [b' '; 4];
    let mut len = 0usize;
    if let Some(encoded) = encoded {
        for &b in encoded {
            if b == 0 {
                continue;
            }
            let ch = (i32::from(b) + 32).clamp(32, 126);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "ch is clamped to 32..=126, fits in u8"
            )]
            let ch = ch as u8;
            if ch != b' ' {
                out[len] = ch;
                len += 1;
            }
        }
    }
    if len == 0 {
        return [b'X', b' ', b' ', b' '];
    }
    out
}

/// Convert per-residue pLDDT (0-1) to a pseudo B-factor, matching the
/// reference `(1 - plddt) * 100`. Missing confidence yields 0.
fn plddt_to_b_factor(plddt: Option<&[f32]>, res_i: usize) -> f32 {
    match plddt.and_then(|v| v.get(res_i)) {
        Some(&p) => (1.0 - p) * 100.0,
        None => 0.0,
    }
}

/// Convert a Boltz `Structure`'s host-resident columns to assembly wire bytes.
///
/// The folding-model entry point: the plugin extracts the structured-array
/// columns to numpy arrays (already host-resident, never a live tensor) and
/// passes them here. Returns assembly wire bytes, matching the contract of the
/// AtomWorks `atom_array_to_entities` bridge.
///
/// `coords` is `(N, 3)` float32, `atom_names` is `(N, 4)` int8 (Boltz
/// ASCII-minus-32 encoding), `is_present` is `(N,)` bool; `res_atom_idx` and
/// `res_atom_num` are `(R,)` int32 residue atom ranges, `res_names` is the
/// list of `R` 3-letter residue names, and `plddt` is an optional `(R,)`
/// float32 per-residue confidence.
///
/// # Errors
///
/// Returns `PyErr` if a numpy array is not contiguous or if assembly
/// serialization fails.
#[pyfunction]
#[pyo3(signature = (
    coords,
    atom_names,
    is_present,
    res_atom_idx,
    res_atom_num,
    res_names,
    plddt = None,
))]
#[allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    reason = "one parameter per Boltz Structure column; pyo3 entry point"
)]
pub fn boltz_structure_to_entities(
    coords: PyReadonlyArray2<'_, f32>,
    atom_names: PyReadonlyArray2<'_, i8>,
    is_present: PyReadonlyArray1<'_, bool>,
    res_atom_idx: PyReadonlyArray1<'_, i32>,
    res_atom_num: PyReadonlyArray1<'_, i32>,
    res_names: Vec<String>,
    plddt: Option<PyReadonlyArray1<'_, f32>>,
) -> PyResult<Vec<u8>> {
    let coords = rows3_from_flat(coords.as_slice()?);
    let atom_names = rows4_from_flat(atom_names.as_slice()?);
    let is_present = is_present.as_slice()?;
    let res_atom_idx = res_atom_idx.as_slice()?;
    let res_atom_num = res_atom_num.as_slice()?;
    let plddt_slice = match plddt.as_ref() {
        Some(arr) => Some(arr.as_slice()?),
        None => None,
    };

    let columns = BoltzColumns {
        coords: &coords,
        atom_names: &atom_names,
        is_present,
        res_atom_idx,
        res_atom_num,
        res_names: &res_names,
        plddt: plddt_slice,
    };

    boltz_columns_to_assembly_bytes(&columns).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
    })
}

/// Reshape a contiguous `3 * N` slice into `[[f32; 3]; N]`. Any trailing
/// remainder (a non-multiple-of-3 length) is dropped.
fn rows3_from_flat(flat: &[f32]) -> Vec<[f32; 3]> {
    flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Reshape a contiguous `4 * N` slice into `[[i8; 4]; N]`. Any trailing
/// remainder (a non-multiple-of-4 length) is dropped.
fn rows4_from_flat(flat: &[i8]) -> Vec<[i8; 4]> {
    flat.chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "tests: coords are exact f32 copies, no arithmetic"
)]
mod tests {
    use super::*;
    use crate::entity::molecule::traits::Entity;
    use crate::entity::molecule::MoleculeEntity;
    use crate::ops::wire::deserialize_assembly;

    /// Encode an ASCII atom name into the Boltz `int8[4]` form (each byte
    /// minus 32), the inverse of [`decode_boltz_atom_name`].
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "test inputs are printable ASCII; offset stays in i8 range"
    )]
    fn encode_boltz_name(s: &[u8]) -> [i8; 4] {
        let mut out = [0i8; 4];
        for (slot, &b) in out.iter_mut().zip(s) {
            *slot = (i32::from(b) - 32) as i8;
        }
        out
    }

    #[test]
    fn boltz_two_residues_decode_names_coords_grouping_and_residue_names() {
        // Two residues: ALA with N/CA/C, GLY with N/CA. Atom names are in the
        // Boltz int8[4] encoding (ASCII minus 32). This exercises the decode
        // and grouping logic directly, before protein canonicalization.
        let nm = encode_boltz_name;
        let coords = [
            [0.0_f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.2, 0.0, 0.0],
            [4.2, 0.0, 0.0],
        ];
        let atom_names = [nm(b"N"), nm(b"CA"), nm(b"C"), nm(b"N"), nm(b"CA")];
        let is_present = [true, true, true, true, true];
        let res_atom_idx = [0i32, 3];
        let res_atom_num = [3i32, 2];
        let res_names = [String::from("ALA"), String::from("GLY")];

        let columns = BoltzColumns {
            coords: &coords,
            atom_names: &atom_names,
            is_present: &is_present,
            res_atom_idx: &res_atom_idx,
            res_atom_num: &res_atom_num,
            res_names: &res_names,
            plddt: None,
        };
        let table = boltz_columns_to_table(&columns);

        assert_eq!(table.len(), 5);

        // Atom names decoded from the int8[4] encoding, in source order.
        assert_eq!(table.name[0], *b"N   ");
        assert_eq!(table.name[1], *b"CA  ");
        assert_eq!(table.name[2], *b"C   ");
        assert_eq!(table.name[3], *b"N   ");
        assert_eq!(table.name[4], *b"CA  ");

        // Elements inferred from atom names.
        assert_eq!(table.element[0], Element::N);
        assert_eq!(table.element[1], Element::C);

        // Coordinates survive in order, exact f32.
        for (pos, coord) in table.position.iter().zip(&coords) {
            assert_eq!(pos.x, coord[0]);
            assert_eq!(pos.y, coord[1]);
            assert_eq!(pos.z, coord[2]);
        }

        // Residue grouping: the first three atoms map to residue 0 (ALA),
        // the last two to residue 1 (GLY); the residue index is the row res_id.
        assert_eq!([table.res_id[0], table.res_id[1], table.res_id[2]], [0; 3]);
        assert_eq!([table.res_id[3], table.res_id[4]], [1, 1]);
        assert_eq!(table.res_name[0], *b"ALA");
        assert_eq!(table.res_name[3], *b"GLY");
    }

    #[test]
    fn boltz_drops_absent_and_nan_atoms() {
        // Five slots, but the middle one is absent and the last has a NaN
        // coordinate; both must be dropped, leaving three rows.
        let nm = encode_boltz_name;
        let coords = [
            [0.0_f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [f32::NAN, 0.0, 0.0],
        ];
        let atom_names = [nm(b"N"), nm(b"CA"), nm(b"C"), nm(b"O"), nm(b"CB")];
        let is_present = [true, true, false, true, true];
        let res_atom_idx = [0i32];
        let res_atom_num = [5i32];
        let res_names = [String::from("ALA")];

        let columns = BoltzColumns {
            coords: &coords,
            atom_names: &atom_names,
            is_present: &is_present,
            res_atom_idx: &res_atom_idx,
            res_atom_num: &res_atom_num,
            res_names: &res_names,
            plddt: None,
        };
        let table = boltz_columns_to_table(&columns);

        // Slot 2 (absent) and slot 4 (NaN) dropped: N, CA, O survive.
        assert_eq!(table.len(), 3);
        assert_eq!(table.name[0], *b"N   ");
        assert_eq!(table.name[1], *b"CA  ");
        assert_eq!(table.name[2], *b"O   ");
    }

    #[test]
    fn boltz_atom37_packed_uses_ordering_table() {
        // n_atoms == 37 * n_residues triggers the packed branch; atom names
        // come from ATOM37_NAMES[i % 37] and the int8[4] table is ignored.
        let n = 37usize;
        let coords: Vec<[f32; 3]> = (0..n)
            .map(|i| [f32::from(u8::try_from(i).unwrap()), 0.0, 0.0])
            .collect();
        let atom_names = vec![[0i8; 4]; n];
        let is_present = vec![true; n];
        let res_atom_idx = [0i32];
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let res_atom_num = [n as i32];
        let res_names = [String::from("ALA")];

        let columns = BoltzColumns {
            coords: &coords,
            atom_names: &atom_names,
            is_present: &is_present,
            res_atom_idx: &res_atom_idx,
            res_atom_num: &res_atom_num,
            res_names: &res_names,
            plddt: None,
        };
        let table = boltz_columns_to_table(&columns);

        assert_eq!(table.len(), 37);
        // Slots follow the canonical atom37 ordering, not the (zeroed) int8[4]
        // names.
        assert_eq!(table.name[0], *b"N   ");
        assert_eq!(table.name[1], *b"CA  ");
        assert_eq!(table.name[2], *b"C   ");
        assert_eq!(table.name[3], *b"CB  ");
        assert_eq!(table.name[4], *b"O   ");
        assert_eq!(table.name[36], *b"X   ");
    }

    #[test]
    fn boltz_plddt_maps_to_b_factor() {
        let coords = [[0.0_f32, 0.0, 0.0]];
        let atom_names = [encode_boltz_name(b"CA")];
        let is_present = [true];
        let res_atom_idx = [0i32];
        let res_atom_num = [1i32];
        let res_names = [String::from("ALA")];
        let plddt = [0.75_f32];

        let columns = BoltzColumns {
            coords: &coords,
            atom_names: &atom_names,
            is_present: &is_present,
            res_atom_idx: &res_atom_idx,
            res_atom_num: &res_atom_num,
            res_names: &res_names,
            plddt: Some(&plddt),
        };
        let table = boltz_columns_to_table(&columns);
        // (1 - 0.75) * 100 == 25.
        assert_eq!(table.b_factor[0], 25.0);
    }

    #[test]
    fn boltz_blank_residue_name_falls_back_to_unk() {
        let coords = [[0.0_f32, 0.0, 0.0]];
        let atom_names = [encode_boltz_name(b"CA")];
        let is_present = [true];
        let res_atom_idx = [0i32];
        let res_atom_num = [1i32];
        let res_names = [String::from("   ")];

        let columns = BoltzColumns {
            coords: &coords,
            atom_names: &atom_names,
            is_present: &is_present,
            res_atom_idx: &res_atom_idx,
            res_atom_num: &res_atom_num,
            res_names: &res_names,
            plddt: None,
        };
        let table = boltz_columns_to_table(&columns);
        assert_eq!(table.res_name[0], *b"UNK");
    }

    #[test]
    fn boltz_empty_yields_empty_assembly() {
        let columns = BoltzColumns {
            coords: &[],
            atom_names: &[],
            is_present: &[],
            res_atom_idx: &[],
            res_atom_num: &[],
            res_names: &[],
            plddt: None,
        };
        let bytes = boltz_columns_to_assembly_bytes(&columns).unwrap();
        let assembly = deserialize_assembly(&bytes).unwrap();
        assert!(assembly.entities().is_empty());
    }

    /// End-to-end: complete-backbone residues survive protein construction
    /// and round-trip through assembly serialization with the expected
    /// entity/residue/atom counts, residue names, and per-atom name->coord
    /// pairing. Atoms are given in canonical N, CA, C, O order so no residue
    /// is dropped for a missing backbone atom.
    #[test]
    fn boltz_complete_residues_serialize_to_two_residue_protein() {
        let nm = encode_boltz_name;
        let coords = [
            // ALA: N, CA, C, O
            [0.0_f32, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 1.0, 0.0],
            [2.5, 2.2, 0.0],
            // GLY: N, CA, C, O (placed near ALA's C so the chain is one
            // segment).
            [3.8, 1.0, 0.0],
            [4.8, 1.0, 0.0],
            [5.8, 2.0, 0.0],
            [5.8, 3.2, 0.0],
        ];
        let atom_names = [
            nm(b"N"),
            nm(b"CA"),
            nm(b"C"),
            nm(b"O"),
            nm(b"N"),
            nm(b"CA"),
            nm(b"C"),
            nm(b"O"),
        ];
        let is_present = [true; 8];
        let res_atom_idx = [0i32, 4];
        let res_atom_num = [4i32, 4];
        let res_names = [String::from("ALA"), String::from("GLY")];

        let columns = BoltzColumns {
            coords: &coords,
            atom_names: &atom_names,
            is_present: &is_present,
            res_atom_idx: &res_atom_idx,
            res_atom_num: &res_atom_num,
            res_names: &res_names,
            plddt: None,
        };
        let bytes = boltz_columns_to_assembly_bytes(&columns).unwrap();

        let assembly = deserialize_assembly(&bytes).unwrap();
        assert_eq!(assembly.entities().len(), 1);
        let MoleculeEntity::Protein(protein) = &*assembly.entities()[0] else {
            unreachable!("Boltz path builds a protein entity");
        };

        assert_eq!(protein.residues.len(), 2);
        assert_eq!(protein.residues[0].name, *b"ALA");
        assert_eq!(protein.residues[1].name, *b"GLY");
        let atoms = protein.columns().to_atoms();
        assert_eq!(atoms.len(), 8);

        // Per-residue name->coord pairing survives canonicalization: look each
        // expected atom up by name within its residue range.
        let coord_by_name = |range: std::ops::Range<usize>, want: &[u8; 4]| {
            atoms[range]
                .iter()
                .find(|a| &a.name == want)
                .map(|a| a.position)
                .unwrap()
        };
        let ala = protein.residues[0].atom_range.clone();
        assert_eq!(coord_by_name(ala.clone(), b"N   ").x, 0.0);
        assert_eq!(coord_by_name(ala.clone(), b"CA  ").x, 1.5);
        assert_eq!(coord_by_name(ala, b"O   ").y, 2.2);
        let gly = protein.residues[1].atom_range.clone();
        assert_eq!(coord_by_name(gly.clone(), b"N   ").x, 3.8);
        assert_eq!(coord_by_name(gly, b"O   ").y, 3.2);
    }
}
