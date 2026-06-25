//! C ABI for typed Assembly edits (`AssemblyEdit`) and the delta
//! wire format.
//!
//! Mirrors the Rust surface in [`crate::ops::edit`] and
//! [`crate::ops::wire::delta`]. See `feedback_molex_api_parity.md` for
//! the cross-binding shape rules.
//!
//! Ownership / lifetime notes:
//! - `molex_EditList*` is owned by the caller; free with [`molex_edits_free`].
//! - `molex_AtomRow` / `molex_Variant` are flat structs passed by pointer +
//!   length. Their backing memory is borrowed for the duration of the call
//!   only.
//! - Byte buffers returned via `out_buf` / `out_len` from
//!   [`molex_edits_to_delta`] are heap-allocated by Rust; caller frees with
//!   `molex_free_bytes`.

#![allow(non_camel_case_types)]

use std::ptr;

use glam::Vec3;

use super::{
    assembly_inner_mut, clear_last_error, molex_Assembly, set_last_error,
    slice_from_raw, vec_to_out_buffer, MOLEX_ERR, MOLEX_ERR_NULL, MOLEX_OK,
};
use crate::chemistry::variant::{ProtonationState, VariantTag};
use crate::element::Element;
use crate::entity::molecule::atom::Atom;
use crate::entity::molecule::id::EntityIdAllocator;
use crate::ops::edit::AssemblyEdit;
use crate::ops::wire::delta::{
    deserialize_edits, serialize_edits, DeltaSerializeError,
};

pub mod read;

// Opaque list handle

/// Owned, ordered list of typed Assembly edits.
///
/// Free with [`molex_edits_free`]. Push typed entries via the
/// `molex_edits_push_*` constructors; or dump to wire bytes with
/// [`molex_edits_to_delta`].
pub struct molex_EditList;

/// Construct an empty edit list.
#[no_mangle]
pub extern "C" fn molex_edits_new() -> *mut molex_EditList {
    let v: Vec<AssemblyEdit> = Vec::new();
    Box::into_raw(Box::new(v)).cast::<molex_EditList>()
}

/// Free an edit list returned by `molex_edits_new` or
/// `molex_delta_to_edits`. Safe to call with a null pointer (no-op).
#[no_mangle]
pub extern "C" fn molex_edits_free(list: *mut molex_EditList) {
    if list.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(list.cast::<Vec<AssemblyEdit>>()) });
}

/// Number of edits currently in the list. Returns 0 if `list` is null.
#[no_mangle]
pub extern "C" fn molex_edits_count(list: *const molex_EditList) -> usize {
    let Some(inner) = edit_list_inner(list) else {
        return 0;
    };
    inner.len()
}

fn edit_list_inner<'a>(
    list: *const molex_EditList,
) -> Option<&'a Vec<AssemblyEdit>> {
    if list.is_null() {
        return None;
    }
    Some(unsafe { &*list.cast::<Vec<AssemblyEdit>>() })
}

fn edit_list_inner_mut<'a>(
    list: *mut molex_EditList,
) -> Option<&'a mut Vec<AssemblyEdit>> {
    if list.is_null() {
        return None;
    }
    Some(unsafe { &mut *list.cast::<Vec<AssemblyEdit>>() })
}

// Variant / atom payload structs

/// Variant tag kind discriminant on the C boundary.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum molex_VariantKind {
    /// Chain N-terminus patch.
    NTerminus = 1,
    /// Chain C-terminus patch.
    CTerminus = 2,
    /// Participates in a disulfide bond.
    Disulfide = 3,
    /// Non-canonical protonation. Reads
    /// `molex_Variant::protonation`.
    Protonation = 4,
    /// Open-ended tag carrying a UTF-8 string in
    /// `molex_Variant::{str_ptr, str_len}`.
    Other = 0xFE,
}

/// Protonation state discriminant. Read only when `kind ==
/// Protonation`. For `Custom`, the string in
/// `molex_Variant::{str_ptr, str_len}` carries the variant name.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum molex_ProtonationKind {
    /// `HID` — delta-protonated histidine.
    HisDelta = 1,
    /// `HIE` — epsilon-protonated histidine.
    HisEpsilon = 2,
    /// `HIP` — doubly-protonated histidine.
    HisDoubly = 3,
    /// Anything else; payload string carries the variant name.
    Custom = 0xFF,
    /// Default placeholder when `kind != Protonation`. The push
    /// functions ignore this field unless the variant kind is
    /// `Protonation`.
    Unused = 0,
}

/// One variant tag as crossed-over to C.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct molex_Variant {
    /// Discriminant.
    pub kind: molex_VariantKind,
    /// Protonation sub-discriminant. Read only when `kind ==
    /// Protonation`.
    pub protonation: molex_ProtonationKind,
    /// UTF-8 bytes for `Other` and `Protonation::Custom`. Borrowed for
    /// the duration of the call.
    pub str_ptr: *const u8,
    /// Length of `str_ptr` in bytes.
    pub str_len: usize,
}

/// One atom row crossed-over to C. Layout mirrors the assembly wire atom
/// row (12 + 4 + 2 = 18 bytes excluding chain/residue context which
/// is conveyed separately).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct molex_AtomRow {
    /// X coordinate.
    pub position_x: f32,
    /// Y coordinate.
    pub position_y: f32,
    /// Z coordinate.
    pub position_z: f32,
    /// PDB atom name, space-padded.
    pub name: [u8; 4],
    /// Element symbol, null-padded if 1 character (e.g. `['C', 0]`).
    pub element: [u8; 2],
}

// Push entry points

fn build_coord_list(
    coords_xyz: *const f32,
    coord_count: usize,
) -> Option<Vec<Vec3>> {
    let triples = slice_from_raw(coords_xyz, coord_count.checked_mul(3)?)?;
    let mut out = Vec::with_capacity(coord_count);
    for c in triples.chunks_exact(3) {
        out.push(Vec3::new(c[0], c[1], c[2]));
    }
    Some(out)
}

fn build_atom_list(
    atoms_ptr: *const molex_AtomRow,
    atom_count: usize,
) -> Option<Vec<Atom>> {
    let rows = slice_from_raw(atoms_ptr, atom_count)?;
    let mut out = Vec::with_capacity(atom_count);
    for r in rows {
        let sym = std::str::from_utf8(&r.element)
            .unwrap_or("")
            .trim_matches('\0')
            .trim();
        let element = Element::from_symbol(sym);
        out.push(Atom {
            position: Vec3::new(r.position_x, r.position_y, r.position_z),
            occupancy: 1.0,
            b_factor: 0.0,
            element,
            name: r.name,
            formal_charge: 0,
            observed: true,
        });
    }
    Some(out)
}

fn build_variant_list(
    variants_ptr: *const molex_Variant,
    variant_count: usize,
) -> Result<Vec<VariantTag>, String> {
    let slice =
        slice_from_raw(variants_ptr, variant_count).ok_or_else(|| {
            "variants pointer is null with nonzero count".to_owned()
        })?;
    let mut out = Vec::with_capacity(variant_count);
    for v in slice {
        out.push(c_variant_to_tag(v)?);
    }
    Ok(out)
}

fn c_variant_to_tag(v: &molex_Variant) -> Result<VariantTag, String> {
    fn payload_str(v: &molex_Variant) -> Result<String, String> {
        if v.str_ptr.is_null() {
            return Err("variant payload string pointer is null".to_owned());
        }
        let bytes = unsafe { std::slice::from_raw_parts(v.str_ptr, v.str_len) };
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|e| format!("variant payload not valid UTF-8: {e}"))
    }
    match v.kind {
        molex_VariantKind::NTerminus => Ok(VariantTag::NTerminus),
        molex_VariantKind::CTerminus => Ok(VariantTag::CTerminus),
        molex_VariantKind::Disulfide => Ok(VariantTag::Disulfide),
        molex_VariantKind::Protonation => {
            let state = match v.protonation {
                molex_ProtonationKind::HisDelta => ProtonationState::HisDelta,
                molex_ProtonationKind::HisEpsilon => {
                    ProtonationState::HisEpsilon
                }
                molex_ProtonationKind::HisDoubly => ProtonationState::HisDoubly,
                molex_ProtonationKind::Custom => {
                    ProtonationState::Custom(payload_str(v)?)
                }
                molex_ProtonationKind::Unused => {
                    return Err(
                        "Protonation variant has Unused sub-kind".to_owned()
                    );
                }
            };
            Ok(VariantTag::Protonation(state))
        }
        molex_VariantKind::Other => Ok(VariantTag::Other(payload_str(v)?)),
    }
}

fn make_entity_id(raw: u32) -> crate::entity::molecule::id::EntityId {
    // Construct via a throwaway allocator: receiver-side edits refer to
    // existing entity ids, not freshly allocated ones.
    EntityIdAllocator::new().from_raw(raw)
}

/// Append a `SetEntityCoords` edit to `list`. `coords_xyz` must point
/// to `coord_count * 3` floats (x,y,z,x,y,z,...).
#[no_mangle]
pub extern "C" fn molex_edits_push_set_entity_coords(
    list: *mut molex_EditList,
    entity_id: u32,
    coords_xyz: *const f32,
    coord_count: usize,
) -> i32 {
    let Some(inner) = edit_list_inner_mut(list) else {
        set_last_error(&"molex_edits_push_set_entity_coords: null list");
        return MOLEX_ERR_NULL;
    };
    let Some(coords) = build_coord_list(coords_xyz, coord_count) else {
        set_last_error(
            &"molex_edits_push_set_entity_coords: invalid coords pointer",
        );
        return MOLEX_ERR_NULL;
    };
    inner.push(AssemblyEdit::SetEntityCoords {
        entity: make_entity_id(entity_id),
        coords,
    });
    clear_last_error();
    MOLEX_OK
}

/// Append a `SetResidueCoords` edit to `list`.
#[no_mangle]
pub extern "C" fn molex_edits_push_set_residue_coords(
    list: *mut molex_EditList,
    entity_id: u32,
    residue_idx: usize,
    coords_xyz: *const f32,
    coord_count: usize,
) -> i32 {
    let Some(inner) = edit_list_inner_mut(list) else {
        set_last_error(&"molex_edits_push_set_residue_coords: null list");
        return MOLEX_ERR_NULL;
    };
    let Some(coords) = build_coord_list(coords_xyz, coord_count) else {
        set_last_error(
            &"molex_edits_push_set_residue_coords: invalid coords pointer",
        );
        return MOLEX_ERR_NULL;
    };
    inner.push(AssemblyEdit::SetResidueCoords {
        entity: make_entity_id(entity_id),
        residue_idx,
        coords,
    });
    clear_last_error();
    MOLEX_OK
}

/// Append a `MutateResidue` edit to `list`. `new_name` is a 3-byte
/// (space-padded) residue name. `atoms` carries the new atom list;
/// `variants` carries the residue's variant tags.
#[no_mangle]
pub extern "C" fn molex_edits_push_mutate_residue(
    list: *mut molex_EditList,
    entity_id: u32,
    residue_idx: usize,
    new_name: *const u8, // 3-byte buffer
    atoms: *const molex_AtomRow,
    atom_count: usize,
    variants: *const molex_Variant,
    variant_count: usize,
) -> i32 {
    let Some(inner) = edit_list_inner_mut(list) else {
        set_last_error(&"molex_edits_push_mutate_residue: null list");
        return MOLEX_ERR_NULL;
    };
    if new_name.is_null() {
        set_last_error(&"molex_edits_push_mutate_residue: null new_name");
        return MOLEX_ERR_NULL;
    }
    let mut name_buf = [b' '; 3];
    unsafe {
        ptr::copy_nonoverlapping(new_name, name_buf.as_mut_ptr(), 3);
    }
    let Some(new_atoms) = build_atom_list(atoms, atom_count) else {
        set_last_error(&"molex_edits_push_mutate_residue: invalid atoms");
        return MOLEX_ERR_NULL;
    };
    let new_variants = if variant_count == 0 {
        Vec::new()
    } else {
        match build_variant_list(variants, variant_count) {
            Ok(v) => v,
            Err(msg) => {
                set_last_error(&msg);
                return MOLEX_ERR;
            }
        }
    };
    inner.push(AssemblyEdit::MutateResidue {
        entity: make_entity_id(entity_id),
        residue_idx,
        new_name: name_buf,
        new_atoms,
        new_variants,
    });
    clear_last_error();
    MOLEX_OK
}

/// Append a `SetVariants` edit to `list`.
#[no_mangle]
pub extern "C" fn molex_edits_push_set_variants(
    list: *mut molex_EditList,
    entity_id: u32,
    residue_idx: usize,
    variants: *const molex_Variant,
    variant_count: usize,
) -> i32 {
    let Some(inner) = edit_list_inner_mut(list) else {
        set_last_error(&"molex_edits_push_set_variants: null list");
        return MOLEX_ERR_NULL;
    };
    let new_variants = if variant_count == 0 {
        Vec::new()
    } else {
        match build_variant_list(variants, variant_count) {
            Ok(v) => v,
            Err(msg) => {
                set_last_error(&msg);
                return MOLEX_ERR;
            }
        }
    };
    inner.push(AssemblyEdit::SetVariants {
        entity: make_entity_id(entity_id),
        residue_idx,
        variants: new_variants,
    });
    clear_last_error();
    MOLEX_OK
}

// Serialize / deserialize delta

/// Serialize the edit list to delta bytes.
///
/// On success returns [`MOLEX_OK`] and writes the heap-allocated
/// buffer pointer + length to `out_buf` / `out_len`; caller frees with
/// `molex_free_bytes`.
#[no_mangle]
pub extern "C" fn molex_edits_to_delta(
    list: *const molex_EditList,
    out_buf: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if out_buf.is_null() || out_len.is_null() {
        set_last_error(&"molex_edits_to_delta: null out pointer");
        return MOLEX_ERR_NULL;
    }
    let Some(inner) = edit_list_inner(list) else {
        set_last_error(&"molex_edits_to_delta: null list");
        return MOLEX_ERR_NULL;
    };
    match serialize_edits(inner) {
        Ok(bytes) => {
            clear_last_error();
            vec_to_out_buffer(bytes, out_buf, out_len)
        }
        Err(DeltaSerializeError::TopologyEditNotSupported { index }) => {
            set_last_error(&format!(
                "molex_edits_to_delta: topology edit at index {index} cannot \
                 be encoded as delta"
            ));
            MOLEX_ERR
        }
    }
}

/// Decode delta bytes into a new edit list.
///
/// Returns null on failure with the error message available via
/// `molex_last_error_message`. Caller frees with `molex_edits_free`.
#[no_mangle]
pub extern "C" fn molex_delta_to_edits(
    bytes_ptr: *const u8,
    len: usize,
) -> *mut molex_EditList {
    let Some(bytes) = slice_from_raw(bytes_ptr, len) else {
        set_last_error(&"molex_delta_to_edits: null input pointer");
        return ptr::null_mut();
    };
    match deserialize_edits(bytes) {
        Ok(v) => {
            clear_last_error();
            Box::into_raw(Box::new(v)).cast::<molex_EditList>()
        }
        Err(e) => {
            set_last_error(&e);
            ptr::null_mut()
        }
    }
}

// Apply

/// Apply every edit in `list` to `assembly` in order. Stops at the
/// first failure and returns nonzero; edits applied before the
/// failing one stay applied (each having bumped the generation
/// counter).
#[no_mangle]
pub extern "C" fn molex_assembly_apply_edits(
    assembly: *mut molex_Assembly,
    list: *const molex_EditList,
) -> i32 {
    let Some(asm) = assembly_inner_mut(assembly) else {
        set_last_error(&"molex_assembly_apply_edits: null assembly");
        return MOLEX_ERR_NULL;
    };
    let Some(edits) = edit_list_inner(list) else {
        set_last_error(&"molex_assembly_apply_edits: null list");
        return MOLEX_ERR_NULL;
    };
    match asm.apply_edits(edits) {
        Ok(()) => {
            clear_last_error();
            MOLEX_OK
        }
        Err(e) => {
            set_last_error(&e);
            MOLEX_ERR
        }
    }
}

/// Decode delta bytes and apply them to `assembly`.
///
/// Equivalent to calling `molex_delta_to_edits` then
/// `molex_assembly_apply_edits` then `molex_edits_free`, but avoids
/// the round-trip through a separate list handle on the apply hot
/// path.
#[no_mangle]
pub extern "C" fn molex_assembly_apply_delta(
    assembly: *mut molex_Assembly,
    bytes_ptr: *const u8,
    len: usize,
) -> i32 {
    let Some(asm) = assembly_inner_mut(assembly) else {
        set_last_error(&"molex_assembly_apply_delta: null assembly");
        return MOLEX_ERR_NULL;
    };
    let Some(bytes) = slice_from_raw(bytes_ptr, len) else {
        set_last_error(&"molex_assembly_apply_delta: null input pointer");
        return MOLEX_ERR_NULL;
    };
    let edits = match deserialize_edits(bytes) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&e);
            return MOLEX_ERR;
        }
    };
    match asm.apply_edits(&edits) {
        Ok(()) => {
            clear_last_error();
            MOLEX_OK
        }
        Err(e) => {
            set_last_error(&e);
            MOLEX_ERR
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests assert against literal float coordinates that round-trip \
              through serialization byte-for-byte; bit-exact equality is the \
              correct check."
)]
mod tests {
    use super::*;
    use crate::entity::molecule::polymer::Residue;
    use crate::entity::molecule::protein::ProteinEntity;
    use crate::entity::molecule::MoleculeEntity;
    use crate::Assembly;

    fn ala_atoms() -> Vec<Atom> {
        let mk = |sym: &str, x: f32, elem: Element| {
            let mut name = [b' '; 4];
            for (i, b) in sym.bytes().take(4).enumerate() {
                name[i] = b;
            }
            Atom {
                position: Vec3::new(x, 0.0, 0.0),
                occupancy: 1.0,
                b_factor: 0.0,
                element: elem,
                name,
                formal_charge: 0,
                observed: true,
            }
        };
        vec![
            mk("N", 0.0, Element::N),
            mk("CA", 1.0, Element::C),
            mk("C", 2.0, Element::C),
            mk("O", 3.0, Element::O),
        ]
    }

    fn make_assembly_handle() -> (*mut molex_Assembly, u32) {
        let mut alloc = EntityIdAllocator::new();
        let id = alloc.allocate();
        // Residue named MSE so the missing-atom completion pass is a
        // no-op; these coordinate-edit tests fix the entity at four atoms
        // and are about the edit wire path, not chemistry completion.
        let residues = vec![Residue {
            name: *b"MSE",
            label_seq_id: 1,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: 0..4,
            variants: Vec::new(),
        }];
        let protein = MoleculeEntity::Protein(ProteinEntity::new(
            id,
            ala_atoms(),
            residues,
            "A".to_owned(),
        ));
        let assembly = Assembly::new(vec![protein]);
        let ptr = Box::into_raw(Box::new(assembly)).cast::<molex_Assembly>();
        (ptr, id.raw())
    }

    #[test]
    fn push_set_entity_coords_serialize_deserialize_apply_roundtrip() {
        let (asm_ptr, entity_raw) = make_assembly_handle();

        let list = molex_edits_new();
        let new_coords: [f32; 12] = [
            10.0, 20.0, 30.0, 11.0, 21.0, 31.0, 12.0, 22.0, 32.0, 13.0, 23.0,
            33.0,
        ];
        let status = molex_edits_push_set_entity_coords(
            list,
            entity_raw,
            new_coords.as_ptr(),
            4,
        );
        assert_eq!(status, MOLEX_OK);
        assert_eq!(molex_edits_count(list), 1);

        let mut out_buf: *mut u8 = ptr::null_mut();
        let mut out_len: usize = 0;
        let status =
            molex_edits_to_delta(list, &raw mut out_buf, &raw mut out_len);
        assert_eq!(status, MOLEX_OK);
        assert!(!out_buf.is_null());
        assert!(out_len > 12);

        // Round-trip the bytes back through molex_delta_to_edits then
        // apply.
        let back = molex_delta_to_edits(out_buf, out_len);
        assert!(!back.is_null());
        assert_eq!(molex_edits_count(back), 1);

        let status = molex_assembly_apply_edits(asm_ptr, back);
        assert_eq!(status, MOLEX_OK);

        // Verify positions actually moved.
        let asm_ref = unsafe { &*asm_ptr.cast::<Assembly>() };
        let positions = asm_ref.entities()[0].positions();
        assert_eq!(positions[0], Vec3::new(10.0, 20.0, 30.0));
        assert_eq!(positions[3], Vec3::new(13.0, 23.0, 33.0));

        // Cleanup.
        super::super::molex_free_bytes(out_buf, out_len);
        molex_edits_free(list);
        molex_edits_free(back);
        super::super::molex_assembly_free(asm_ptr);
    }

    #[test]
    fn apply_delta_combined_entry_point() {
        let (asm_ptr, entity_raw) = make_assembly_handle();

        let list = molex_edits_new();
        let coords: [f32; 12] = [
            50.0, 0.0, 0.0, 51.0, 0.0, 0.0, 52.0, 0.0, 0.0, 53.0, 0.0, 0.0,
        ];
        assert_eq!(
            molex_edits_push_set_entity_coords(
                list,
                entity_raw,
                coords.as_ptr(),
                4,
            ),
            MOLEX_OK,
        );

        let mut out_buf: *mut u8 = ptr::null_mut();
        let mut out_len: usize = 0;
        assert_eq!(
            molex_edits_to_delta(list, &raw mut out_buf, &raw mut out_len,),
            MOLEX_OK,
        );

        // The combined apply path - decode + apply in one call.
        let status = molex_assembly_apply_delta(asm_ptr, out_buf, out_len);
        assert_eq!(status, MOLEX_OK);

        let asm_ref = unsafe { &*asm_ptr.cast::<Assembly>() };
        assert_eq!(asm_ref.entities()[0].positions()[0].x, 50.0);

        super::super::molex_free_bytes(out_buf, out_len);
        molex_edits_free(list);
        super::super::molex_assembly_free(asm_ptr);
    }

    #[test]
    fn null_list_rejected_with_error_code() {
        let coords = [0.0f32; 3];
        let status = molex_edits_push_set_entity_coords(
            ptr::null_mut(),
            0,
            coords.as_ptr(),
            1,
        );
        assert_eq!(status, MOLEX_ERR_NULL);
    }

    #[test]
    fn push_variants_with_protonation_custom_string() {
        let list = molex_edits_new();
        let payload = b"ASP_PROTONATED";
        let variant = molex_Variant {
            kind: molex_VariantKind::Protonation,
            protonation: molex_ProtonationKind::Custom,
            str_ptr: payload.as_ptr(),
            str_len: payload.len(),
        };
        let status =
            molex_edits_push_set_variants(list, 0, 0, &raw const variant, 1);
        assert_eq!(status, MOLEX_OK);
        molex_edits_free(list);
    }
}
