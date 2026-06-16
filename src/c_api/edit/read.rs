//! Read accessors over `molex_EditList`.
//!
//! The push entry points in the parent module let C callers build edit
//! lists. The accessors here let them read the lists back out --
//! needed by the rosetta bridge dispatching delta-decoded edits
//! per-variant onto rosetta API calls.
//!
//! Coords are exposed as borrowed pointers into the EditList's owned
//! storage (`glam::Vec3` is `#[repr(C)]` with three contiguous `f32`s,
//! so a `Vec<Vec3>::as_ptr() as *const f32` yields the canonical
//! flat-xyz triple layout the push side accepts).
//!
//! Atoms and variants are NOT layout-compatible with their internal
//! storage (`Atom` is richer than `molex_AtomRow`; `VariantTag` is a
//! Rust enum with owned `String` payloads rather than the C tagged
//! struct). Their read accessors therefore allocate a fresh temporary
//! `Vec<molex_AtomRow>` / `Vec<molex_Variant>`, leak it as a raw
//! pointer for the caller, and rely on the caller to free via
//! [`molex_atom_rows_free`] / [`molex_variants_free`].  For variants
//! the temporary's `str_ptr` fields point INTO the EditList's owned
//! `String`s; those pointers remain valid only as long as the parent
//! EditList is alive and unmutated, which is the same lifetime
//! contract the coord-borrowing accessors already imply.

#![allow(non_camel_case_types)]

use std::ptr;

use super::{
    edit_list_inner, molex_AtomRow, molex_EditList, molex_ProtonationKind,
    molex_Variant, molex_VariantKind,
};
use crate::c_api::{
    clear_last_error, set_last_error, MOLEX_ERR, MOLEX_ERR_NULL, MOLEX_OK,
};
use crate::chemistry::variant::{ProtonationState, VariantTag};
use crate::entity::molecule::atom::Atom;
use crate::ops::edit::AssemblyEdit;

/// Per-edit kind discriminant.
///
/// Returned by [`molex_edits_kind_at`]; caller dispatches on the
/// value to pick the right per-variant getter. `AddEntity` /
/// `RemoveEntity` are listed for completeness but never appear in
/// lists obtained from `molex_delta_to_edits` (the delta
/// serializer rejects topology edits up front).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum molex_EditKind {
    /// Sentinel returned when the list pointer is null or the index is
    /// out of bounds.
    Invalid = 0,
    /// `SetEntityCoords` — bulk per-entity coordinate update. Read via
    /// [`molex_edits_set_entity_coords_at`].
    SetEntityCoords = 1,
    /// `SetResidueCoords` — per-residue coordinate update inside a
    /// polymer entity. Read via
    /// [`molex_edits_set_residue_coords_at`].
    SetResidueCoords = 2,
    /// `MutateResidue` — residue identity + atoms + variants
    /// replacement. Read via [`molex_edits_mutate_residue_at`].
    MutateResidue = 3,
    /// `SetVariants` — replace a residue's variant tag list. Read
    /// via [`molex_edits_set_variants_at`].
    SetVariants = 4,
    /// `AddEntity` — topology edit; appears only in Rust-side edit
    /// lists, never in lists decoded from delta.
    AddEntity = 5,
    /// `RemoveEntity` — topology edit; appears only in Rust-side edit
    /// lists, never in lists decoded from delta.
    RemoveEntity = 6,
}

/// Return the kind of the edit at `index`. Returns
/// [`molex_EditKind::Invalid`] when `list` is null or `index >=
/// molex_edits_count(list)`.
#[no_mangle]
pub extern "C" fn molex_edits_kind_at(
    list: *const molex_EditList,
    index: usize,
) -> molex_EditKind {
    let Some(inner) = edit_list_inner(list) else {
        return molex_EditKind::Invalid;
    };
    let Some(edit) = inner.get(index) else {
        return molex_EditKind::Invalid;
    };
    match edit {
        AssemblyEdit::SetEntityCoords { .. } => molex_EditKind::SetEntityCoords,
        AssemblyEdit::SetResidueCoords { .. } => {
            molex_EditKind::SetResidueCoords
        }
        AssemblyEdit::MutateResidue { .. } => molex_EditKind::MutateResidue,
        AssemblyEdit::SetVariants { .. } => molex_EditKind::SetVariants,
        AssemblyEdit::AddEntity { .. } => molex_EditKind::AddEntity,
        AssemblyEdit::RemoveEntity { .. } => molex_EditKind::RemoveEntity,
    }
}

/// Read the fields of a `SetEntityCoords` edit at `index`.
///
/// On success returns [`MOLEX_OK`] and writes the entity id plus a
/// borrowed coord pointer (3 * `*out_coord_count` `f32`s in
/// x,y,z,x,y,z order). The coord pointer is valid until the parent
/// EditList is freed or mutated.
///
/// Returns [`MOLEX_ERR_NULL`] on null inputs; returns [`MOLEX_ERR`]
/// when `index` is out of range or the edit at `index` is not a
/// `SetEntityCoords` (caller should consult [`molex_edits_kind_at`]
/// before calling).
#[no_mangle]
pub extern "C" fn molex_edits_set_entity_coords_at(
    list: *const molex_EditList,
    index: usize,
    out_entity_id: *mut u32,
    out_coords_xyz: *mut *const f32,
    out_coord_count: *mut usize,
) -> i32 {
    if out_entity_id.is_null()
        || out_coords_xyz.is_null()
        || out_coord_count.is_null()
    {
        set_last_error(&"molex_edits_set_entity_coords_at: null out pointer");
        return MOLEX_ERR_NULL;
    }
    let Some(inner) = edit_list_inner(list) else {
        set_last_error(&"molex_edits_set_entity_coords_at: null list");
        return MOLEX_ERR_NULL;
    };
    let Some(edit) = inner.get(index) else {
        set_last_error(&"molex_edits_set_entity_coords_at: index out of range");
        return MOLEX_ERR;
    };
    let AssemblyEdit::SetEntityCoords { entity, coords } = edit else {
        set_last_error(
            &"molex_edits_set_entity_coords_at: edit at index is not \
              SetEntityCoords",
        );
        return MOLEX_ERR;
    };
    unsafe {
        *out_entity_id = entity.raw();
        *out_coords_xyz = coords.as_ptr().cast::<f32>();
        *out_coord_count = coords.len();
    }
    clear_last_error();
    MOLEX_OK
}

/// Read the fields of a `SetResidueCoords` edit at `index`. Same
/// lifetime contract on `out_coords_xyz` as
/// [`molex_edits_set_entity_coords_at`].
#[no_mangle]
pub extern "C" fn molex_edits_set_residue_coords_at(
    list: *const molex_EditList,
    index: usize,
    out_entity_id: *mut u32,
    out_residue_idx: *mut usize,
    out_coords_xyz: *mut *const f32,
    out_coord_count: *mut usize,
) -> i32 {
    if out_entity_id.is_null()
        || out_residue_idx.is_null()
        || out_coords_xyz.is_null()
        || out_coord_count.is_null()
    {
        set_last_error(&"molex_edits_set_residue_coords_at: null out pointer");
        return MOLEX_ERR_NULL;
    }
    let Some(inner) = edit_list_inner(list) else {
        set_last_error(&"molex_edits_set_residue_coords_at: null list");
        return MOLEX_ERR_NULL;
    };
    let Some(edit) = inner.get(index) else {
        set_last_error(
            &"molex_edits_set_residue_coords_at: index out of range",
        );
        return MOLEX_ERR;
    };
    let AssemblyEdit::SetResidueCoords {
        entity,
        residue_idx,
        coords,
    } = edit
    else {
        set_last_error(
            &"molex_edits_set_residue_coords_at: edit at index is not \
              SetResidueCoords",
        );
        return MOLEX_ERR;
    };
    unsafe {
        *out_entity_id = entity.raw();
        *out_residue_idx = *residue_idx;
        *out_coords_xyz = coords.as_ptr().cast::<f32>();
        *out_coord_count = coords.len();
    }
    clear_last_error();
    MOLEX_OK
}

/// Read the fields of a `MutateResidue` edit at `index`.
///
/// `out_new_name` receives a borrowed 3-byte pointer (lifetime tied
/// to the parent EditList).
///
/// `out_atoms` / `out_atom_count` receive a freshly-allocated
/// `molex_AtomRow` array derived from the edit's internal `Atom`
/// storage. Caller MUST free with [`molex_atom_rows_free`]. The
/// conversion is lossy on the `occupancy` / `b_factor` /
/// `formal_charge` fields (those don't ride delta and aren't
/// representable in `molex_AtomRow`).
///
/// `out_variants` / `out_variant_count` receive a freshly-allocated
/// `molex_Variant` array; the `str_ptr` fields inside it point INTO
/// the parent EditList's owned `String`s (valid until the list is
/// freed or mutated). Caller MUST free the outer array with
/// [`molex_variants_free`].
#[no_mangle]
pub extern "C" fn molex_edits_mutate_residue_at(
    list: *const molex_EditList,
    index: usize,
    out_entity_id: *mut u32,
    out_residue_idx: *mut usize,
    out_new_name: *mut *const u8,
    out_atoms: *mut *const molex_AtomRow,
    out_atom_count: *mut usize,
    out_variants: *mut *const molex_Variant,
    out_variant_count: *mut usize,
) -> i32 {
    if out_entity_id.is_null()
        || out_residue_idx.is_null()
        || out_new_name.is_null()
        || out_atoms.is_null()
        || out_atom_count.is_null()
        || out_variants.is_null()
        || out_variant_count.is_null()
    {
        set_last_error(&"molex_edits_mutate_residue_at: null out pointer");
        return MOLEX_ERR_NULL;
    }
    let Some(inner) = edit_list_inner(list) else {
        set_last_error(&"molex_edits_mutate_residue_at: null list");
        return MOLEX_ERR_NULL;
    };
    let Some(edit) = inner.get(index) else {
        set_last_error(&"molex_edits_mutate_residue_at: index out of range");
        return MOLEX_ERR;
    };
    let AssemblyEdit::MutateResidue {
        entity,
        residue_idx,
        new_name,
        new_atoms,
        new_variants,
    } = edit
    else {
        set_last_error(
            &"molex_edits_mutate_residue_at: edit at index is not \
              MutateResidue",
        );
        return MOLEX_ERR;
    };

    let atom_rows: Vec<molex_AtomRow> =
        new_atoms.iter().map(atom_to_row).collect();
    let variants: Vec<molex_Variant> =
        new_variants.iter().map(tag_to_c_variant).collect();

    let (atoms_ptr, atoms_len) = leak_vec(atom_rows);
    let (variants_ptr, variants_len) = leak_vec(variants);

    unsafe {
        *out_entity_id = entity.raw();
        *out_residue_idx = *residue_idx;
        *out_new_name = new_name.as_ptr();
        *out_atoms = atoms_ptr;
        *out_atom_count = atoms_len;
        *out_variants = variants_ptr;
        *out_variant_count = variants_len;
    }
    clear_last_error();
    MOLEX_OK
}

/// Read the fields of a `SetVariants` edit at `index`. Same lifetime
/// contract on `out_variants` as
/// [`molex_edits_mutate_residue_at`]; caller frees via
/// [`molex_variants_free`].
#[no_mangle]
pub extern "C" fn molex_edits_set_variants_at(
    list: *const molex_EditList,
    index: usize,
    out_entity_id: *mut u32,
    out_residue_idx: *mut usize,
    out_variants: *mut *const molex_Variant,
    out_variant_count: *mut usize,
) -> i32 {
    if out_entity_id.is_null()
        || out_residue_idx.is_null()
        || out_variants.is_null()
        || out_variant_count.is_null()
    {
        set_last_error(&"molex_edits_set_variants_at: null out pointer");
        return MOLEX_ERR_NULL;
    }
    let Some(inner) = edit_list_inner(list) else {
        set_last_error(&"molex_edits_set_variants_at: null list");
        return MOLEX_ERR_NULL;
    };
    let Some(edit) = inner.get(index) else {
        set_last_error(&"molex_edits_set_variants_at: index out of range");
        return MOLEX_ERR;
    };
    let AssemblyEdit::SetVariants {
        entity,
        residue_idx,
        variants,
    } = edit
    else {
        set_last_error(
            &"molex_edits_set_variants_at: edit at index is not SetVariants",
        );
        return MOLEX_ERR;
    };

    let variants_out: Vec<molex_Variant> =
        variants.iter().map(tag_to_c_variant).collect();
    let (variants_ptr, variants_len) = leak_vec(variants_out);

    unsafe {
        *out_entity_id = entity.raw();
        *out_residue_idx = *residue_idx;
        *out_variants = variants_ptr;
        *out_variant_count = variants_len;
    }
    clear_last_error();
    MOLEX_OK
}

/// Free a `molex_AtomRow` array returned by
/// [`molex_edits_mutate_residue_at`]. Safe to call with a null
/// pointer (no-op). `count` MUST match the value originally written
/// to `out_atom_count`.
#[no_mangle]
pub extern "C" fn molex_atom_rows_free(ptr: *mut molex_AtomRow, count: usize) {
    if ptr.is_null() {
        return;
    }
    // Safety: `(ptr, count)` came from `leak_vec`'s `Box<[T]>` round
    // trip. Reconstruct the boxed slice and drop it.
    unsafe {
        let raw = ptr::slice_from_raw_parts_mut(ptr, count);
        drop(Box::from_raw(raw));
    }
}

/// Free a `molex_Variant` array.
///
/// Returned by [`molex_edits_mutate_residue_at`] or
/// [`molex_edits_set_variants_at`]. Safe to call with a null pointer
/// (no-op). The borrowed string payloads pointed at by each entry's
/// `str_ptr` are NOT freed by this call -- they belong to the parent
/// EditList.
#[no_mangle]
pub extern "C" fn molex_variants_free(ptr: *mut molex_Variant, count: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let raw = ptr::slice_from_raw_parts_mut(ptr, count);
        drop(Box::from_raw(raw));
    }
}

// Internal helpers for the read accessors

/// Build a `molex_AtomRow` from an internal `Atom`. Loses occupancy /
/// b_factor / formal_charge (not representable in the C row struct).
fn atom_to_row(a: &Atom) -> molex_AtomRow {
    let sym = a.element.symbol();
    let bytes = sym.as_bytes();
    let mut element = [0u8; 2];
    if !bytes.is_empty() {
        element[0] = bytes[0];
    }
    if bytes.len() > 1 {
        element[1] = bytes[1];
    }
    molex_AtomRow {
        position_x: a.position.x,
        position_y: a.position.y,
        position_z: a.position.z,
        name: a.name,
        element,
    }
}

/// Build a `molex_Variant` from an internal `VariantTag`. The
/// returned struct's `str_ptr` (when populated) borrows from the
/// `VariantTag`'s owned `String`; caller must respect that lifetime.
fn tag_to_c_variant(tag: &VariantTag) -> molex_Variant {
    let (kind, protonation, str_ptr, str_len) = match tag {
        VariantTag::NTerminus => (
            molex_VariantKind::NTerminus,
            molex_ProtonationKind::Unused,
            ptr::null::<u8>(),
            0usize,
        ),
        VariantTag::CTerminus => (
            molex_VariantKind::CTerminus,
            molex_ProtonationKind::Unused,
            ptr::null::<u8>(),
            0usize,
        ),
        VariantTag::Disulfide => (
            molex_VariantKind::Disulfide,
            molex_ProtonationKind::Unused,
            ptr::null::<u8>(),
            0usize,
        ),
        VariantTag::Protonation(state) => match state {
            ProtonationState::HisDelta => (
                molex_VariantKind::Protonation,
                molex_ProtonationKind::HisDelta,
                ptr::null::<u8>(),
                0usize,
            ),
            ProtonationState::HisEpsilon => (
                molex_VariantKind::Protonation,
                molex_ProtonationKind::HisEpsilon,
                ptr::null::<u8>(),
                0usize,
            ),
            ProtonationState::HisDoubly => (
                molex_VariantKind::Protonation,
                molex_ProtonationKind::HisDoubly,
                ptr::null::<u8>(),
                0usize,
            ),
            ProtonationState::Custom(s) => (
                molex_VariantKind::Protonation,
                molex_ProtonationKind::Custom,
                s.as_ptr(),
                s.len(),
            ),
        },
        VariantTag::Other(s) => (
            molex_VariantKind::Other,
            molex_ProtonationKind::Unused,
            s.as_ptr(),
            s.len(),
        ),
    };
    molex_Variant {
        kind,
        protonation,
        str_ptr,
        str_len,
    }
}

/// Leak a Vec as a raw `(ptr, len)` pair. Goes through `Box<[T]>` so
/// the underlying allocation is sized exactly to the element count --
/// needed because the caller-side free functions rebuild the boxed
/// slice from `(ptr, len)` and the allocator requires the
/// deallocation size to match the original allocation size.
/// (`Vec::shrink_to_fit` is "best-effort" per std docs and is NOT
/// safe to round-trip in this way.)
fn leak_vec<T>(v: Vec<T>) -> (*const T, usize) {
    let boxed: Box<[T]> = v.into_boxed_slice();
    let len = boxed.len();
    let ptr: *mut T = Box::into_raw(boxed).cast();
    (ptr.cast_const(), len)
}

#[cfg(test)]
mod tests;
