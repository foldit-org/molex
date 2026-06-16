//! Tests for `molex_edits_*_at` read accessors.

#![allow(
    clippy::float_cmp,
    reason = "tests assert against literal float coordinates that round-trip \
              through serialization byte-for-byte; bit-exact equality is the \
              correct check."
)]

use std::ptr;

use super::*;
use crate::c_api::edit::{
    molex_delta_to_edits, molex_edits_count, molex_edits_free, molex_edits_new,
    molex_edits_push_mutate_residue, molex_edits_push_set_entity_coords,
    molex_edits_push_set_residue_coords, molex_edits_push_set_variants,
    molex_edits_to_delta,
};
use crate::c_api::molex_free_bytes;
#[test]
fn kind_at_reports_invalid_on_null_and_out_of_range() {
    assert_eq!(molex_edits_kind_at(ptr::null(), 0), molex_EditKind::Invalid);
    let list = molex_edits_new();
    assert_eq!(molex_edits_kind_at(list, 0), molex_EditKind::Invalid);
    molex_edits_free(list);
}

#[test]
fn read_set_entity_coords_round_trip() {
    let list = molex_edits_new();
    let coords: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    assert_eq!(
        molex_edits_push_set_entity_coords(list, 42, coords.as_ptr(), 3),
        MOLEX_OK
    );
    assert_eq!(
        molex_edits_kind_at(list, 0),
        molex_EditKind::SetEntityCoords
    );

    let mut entity_id: u32 = 0;
    let mut out_coords: *const f32 = ptr::null();
    let mut out_count: usize = 0;
    let status = molex_edits_set_entity_coords_at(
        list,
        0,
        &raw mut entity_id,
        &raw mut out_coords,
        &raw mut out_count,
    );
    assert_eq!(status, MOLEX_OK);
    assert_eq!(entity_id, 42);
    assert_eq!(out_count, 3);
    unsafe {
        let slice = std::slice::from_raw_parts(out_coords, 9);
        assert_eq!(slice, &coords[..]);
    }
    molex_edits_free(list);
}

#[test]
fn read_set_residue_coords_round_trip() {
    let list = molex_edits_new();
    let coords: [f32; 6] = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    assert_eq!(
        molex_edits_push_set_residue_coords(list, 7, 5, coords.as_ptr(), 2),
        MOLEX_OK
    );
    assert_eq!(
        molex_edits_kind_at(list, 0),
        molex_EditKind::SetResidueCoords
    );

    let mut entity_id: u32 = 0;
    let mut residue_idx: usize = 0;
    let mut out_coords: *const f32 = ptr::null();
    let mut out_count: usize = 0;
    let status = molex_edits_set_residue_coords_at(
        list,
        0,
        &raw mut entity_id,
        &raw mut residue_idx,
        &raw mut out_coords,
        &raw mut out_count,
    );
    assert_eq!(status, MOLEX_OK);
    assert_eq!(entity_id, 7);
    assert_eq!(residue_idx, 5);
    assert_eq!(out_count, 2);
    unsafe {
        let slice = std::slice::from_raw_parts(out_coords, 6);
        assert_eq!(slice, &coords[..]);
    }
    molex_edits_free(list);
}

/// Push a MutateResidue edit fixture for the read-back tests below.
/// Returns the list handle plus the expected new_name bytes (which
/// the caller verifies are accessible via the read accessor).
fn push_mutate_residue_fixture() -> (*mut molex_EditList, &'static [u8; 3]) {
    let list = molex_edits_new();
    let atoms = [
        molex_AtomRow {
            position_x: 0.5,
            position_y: 1.5,
            position_z: 2.5,
            name: *b"N   ",
            element: [b'N', 0],
        },
        molex_AtomRow {
            position_x: 1.5,
            position_y: 2.5,
            position_z: 3.5,
            name: *b"CA  ",
            element: [b'C', 0],
        },
    ];
    let variant = molex_Variant {
        kind: molex_VariantKind::Protonation,
        protonation: molex_ProtonationKind::HisDelta,
        str_ptr: ptr::null(),
        str_len: 0,
    };
    let new_name = b"HIS";
    let status = molex_edits_push_mutate_residue(
        list,
        3,
        9,
        new_name.as_ptr(),
        atoms.as_ptr(),
        atoms.len(),
        &raw const variant,
        1,
    );
    assert_eq!(status, MOLEX_OK);
    (list, new_name)
}

/// Read the MutateResidue at index 0 via the C accessor, copying
/// the scalar fields out and returning the raw payload pointers
/// for the caller to inspect + free.
struct MutateResidueRead {
    entity_id: u32,
    residue_idx: usize,
    name_ptr: *const u8,
    atoms_ptr: *const molex_AtomRow,
    atom_count: usize,
    variants_ptr: *const molex_Variant,
    variant_count: usize,
}

fn read_mutate_residue_at_zero(
    list: *const molex_EditList,
) -> MutateResidueRead {
    let mut r = MutateResidueRead {
        entity_id: 0,
        residue_idx: 0,
        name_ptr: ptr::null(),
        atoms_ptr: ptr::null(),
        atom_count: 0,
        variants_ptr: ptr::null(),
        variant_count: 0,
    };
    let status = molex_edits_mutate_residue_at(
        list,
        0,
        &raw mut r.entity_id,
        &raw mut r.residue_idx,
        &raw mut r.name_ptr,
        &raw mut r.atoms_ptr,
        &raw mut r.atom_count,
        &raw mut r.variants_ptr,
        &raw mut r.variant_count,
    );
    assert_eq!(status, MOLEX_OK);
    r
}

#[test]
fn read_mutate_residue_round_trip_scalar_fields() {
    let (list, new_name) = push_mutate_residue_fixture();
    assert_eq!(molex_edits_kind_at(list, 0), molex_EditKind::MutateResidue);
    let r = read_mutate_residue_at_zero(list);
    assert_eq!(r.entity_id, 3);
    assert_eq!(r.residue_idx, 9);
    assert_eq!(r.atom_count, 2);
    assert_eq!(r.variant_count, 1);
    unsafe {
        let name_slice = std::slice::from_raw_parts(r.name_ptr, 3);
        assert_eq!(name_slice, new_name);
    }
    molex_atom_rows_free(r.atoms_ptr.cast_mut(), r.atom_count);
    molex_variants_free(r.variants_ptr.cast_mut(), r.variant_count);
    molex_edits_free(list);
}

#[test]
fn read_mutate_residue_round_trip_payload_arrays() {
    let (list, _new_name) = push_mutate_residue_fixture();
    let r = read_mutate_residue_at_zero(list);
    unsafe {
        let rows = std::slice::from_raw_parts(r.atoms_ptr, r.atom_count);
        assert_eq!(rows[0].position_x, 0.5);
        assert_eq!(rows[0].name, *b"N   ");
        assert_eq!(rows[0].element, [b'N', 0]);
        assert_eq!(rows[1].position_x, 1.5);
        assert_eq!(rows[1].name, *b"CA  ");
        let vs = std::slice::from_raw_parts(r.variants_ptr, r.variant_count);
        assert_eq!(vs[0].kind, molex_VariantKind::Protonation);
        assert_eq!(vs[0].protonation, molex_ProtonationKind::HisDelta);
        assert!(vs[0].str_ptr.is_null());
        assert_eq!(vs[0].str_len, 0);
    }
    molex_atom_rows_free(r.atoms_ptr.cast_mut(), r.atom_count);
    molex_variants_free(r.variants_ptr.cast_mut(), r.variant_count);
    molex_edits_free(list);
}

#[test]
fn read_set_variants_round_trip_with_custom_string_payload() {
    let list = molex_edits_new();
    let payload = b"LYS_DEPROT";
    let variant = molex_Variant {
        kind: molex_VariantKind::Other,
        protonation: molex_ProtonationKind::Unused,
        str_ptr: payload.as_ptr(),
        str_len: payload.len(),
    };
    assert_eq!(
        molex_edits_push_set_variants(list, 11, 2, &raw const variant, 1),
        MOLEX_OK
    );
    assert_eq!(molex_edits_kind_at(list, 0), molex_EditKind::SetVariants);

    let mut entity_id: u32 = 0;
    let mut residue_idx: usize = 0;
    let mut out_variants: *const molex_Variant = ptr::null();
    let mut out_variant_count: usize = 0;

    let status = molex_edits_set_variants_at(
        list,
        0,
        &raw mut entity_id,
        &raw mut residue_idx,
        &raw mut out_variants,
        &raw mut out_variant_count,
    );
    assert_eq!(status, MOLEX_OK);
    assert_eq!(entity_id, 11);
    assert_eq!(residue_idx, 2);
    assert_eq!(out_variant_count, 1);
    unsafe {
        let vs = std::slice::from_raw_parts(out_variants, out_variant_count);
        assert_eq!(vs[0].kind, molex_VariantKind::Other);
        assert_eq!(vs[0].str_len, payload.len());
        // The borrowed string bytes round-trip through the
        // EditList's owned `String`; pointer identity is NOT
        // required (the push path copies into a new String), but
        // the bytes must match.
        let s = std::slice::from_raw_parts(vs[0].str_ptr, vs[0].str_len);
        assert_eq!(s, payload);
    }
    molex_variants_free(out_variants.cast_mut(), out_variant_count);
    molex_edits_free(list);
}

#[test]
fn read_kind_mismatch_rejected() {
    let list = molex_edits_new();
    let coords: [f32; 3] = [0.0, 0.0, 0.0];
    let _ = molex_edits_push_set_entity_coords(list, 0, coords.as_ptr(), 1);

    // The edit is SetEntityCoords; reading it via the
    // MutateResidue accessor should fail.
    let mut entity_id: u32 = 0;
    let mut residue_idx: usize = 0;
    let mut out_name: *const u8 = ptr::null();
    let mut out_atoms: *const molex_AtomRow = ptr::null();
    let mut out_atom_count: usize = 0;
    let mut out_variants: *const molex_Variant = ptr::null();
    let mut out_variant_count: usize = 0;

    let status = molex_edits_mutate_residue_at(
        list,
        0,
        &raw mut entity_id,
        &raw mut residue_idx,
        &raw mut out_name,
        &raw mut out_atoms,
        &raw mut out_atom_count,
        &raw mut out_variants,
        &raw mut out_variant_count,
    );
    assert_eq!(status, MOLEX_ERR);
    molex_edits_free(list);
}

/// Two-edit fixture: build a list with one `SetEntityCoords` plus
/// one `SetResidueCoords`, serialize to delta, decode, return
/// the decoded list + the original byte buffer (caller frees
/// both). The original push-side list is dropped before return --
/// only the decoded one survives, so any assertion about its
/// contents has to hit delta round-trip semantics.
fn build_decoded_two_edit_list(
    coords_a: &[f32],
    coords_b: &[f32],
) -> (*mut molex_EditList, *mut u8, usize) {
    let list = molex_edits_new();
    let _ = molex_edits_push_set_entity_coords(list, 1, coords_a.as_ptr(), 1);
    let _ =
        molex_edits_push_set_residue_coords(list, 2, 7, coords_b.as_ptr(), 1);
    let mut buf: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    assert_eq!(
        molex_edits_to_delta(list, &raw mut buf, &raw mut len),
        MOLEX_OK
    );
    let decoded = molex_delta_to_edits(buf, len);
    assert!(!decoded.is_null());
    molex_edits_free(list);
    (decoded, buf, len)
}

#[test]
fn delta_decoded_list_reports_correct_kinds() {
    let coords_a: [f32; 3] = [1.0, 2.0, 3.0];
    let coords_b: [f32; 3] = [4.0, 5.0, 6.0];
    let (decoded, buf, len) = build_decoded_two_edit_list(&coords_a, &coords_b);
    assert_eq!(molex_edits_count(decoded), 2);
    assert_eq!(
        molex_edits_kind_at(decoded, 0),
        molex_EditKind::SetEntityCoords
    );
    assert_eq!(
        molex_edits_kind_at(decoded, 1),
        molex_EditKind::SetResidueCoords
    );
    molex_free_bytes(buf, len);
    molex_edits_free(decoded);
}

#[test]
fn delta_decoded_list_round_trips_per_edit_fields() {
    let coords_a: [f32; 3] = [1.0, 2.0, 3.0];
    let coords_b: [f32; 3] = [4.0, 5.0, 6.0];
    let (decoded, buf, len) = build_decoded_two_edit_list(&coords_a, &coords_b);

    let mut eid: u32 = 0;
    let mut cptr: *const f32 = ptr::null();
    let mut ccount: usize = 0;
    assert_eq!(
        molex_edits_set_entity_coords_at(
            decoded,
            0,
            &raw mut eid,
            &raw mut cptr,
            &raw mut ccount,
        ),
        MOLEX_OK
    );
    assert_eq!(eid, 1);
    assert_eq!(ccount, 1);
    unsafe {
        let s = std::slice::from_raw_parts(cptr, 3);
        assert_eq!(s, &coords_a[..]);
    }

    let mut ridx: usize = 0;
    assert_eq!(
        molex_edits_set_residue_coords_at(
            decoded,
            1,
            &raw mut eid,
            &raw mut ridx,
            &raw mut cptr,
            &raw mut ccount,
        ),
        MOLEX_OK
    );
    assert_eq!(eid, 2);
    assert_eq!(ridx, 7);
    assert_eq!(ccount, 1);
    unsafe {
        let s = std::slice::from_raw_parts(cptr, 3);
        assert_eq!(s, &coords_b[..]);
    }

    molex_free_bytes(buf, len);
    molex_edits_free(decoded);
}
