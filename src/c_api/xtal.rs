//! C ABI surface for the resident crystallographic workflow.
//!
//! Mirrors the Python `ExperimentalData` binding: an opaque
//! [`molex_ExperimentalData`] handle parses an sf.cif once and stays resident,
//! then drives density-map and B-factor-refinement queries against the atoms of
//! a `molex_Assembly`. Each map/refine call rebuilds a fresh `XtalRefinement`
//! from the resident data; the parse cost is paid once.
//!
//! The atom source is a `molex_Assembly` handle. Atoms are flattened via
//! `AtomTable::from_entities`; the flat order of that projection is the
//! canonical order the returned B-factor array aligns to.
//!
//! Both the density splat and the B-factor refinement exclude hydrogens: an
//! X-ray density map carries almost no hydrogen signal, so hydrogens are
//! dropped before structure factors are computed. `refine_b_factors` scatters
//! the refined non-H B-factors back into a full-length array aligned 1:1 with
//! the flat atom order, leaving hydrogen B-factors at their input values.
//!
//! Float buffers returned via `out_data` / `out_b` are heap-allocated by Rust
//! and must be released with [`molex_free_floats`].

use std::ffi::c_char;

use super::{
    assembly_inner, clear_last_error, molex_Assembly, set_last_error,
    slice_from_raw, MOLEX_ERR, MOLEX_ERR_NULL, MOLEX_OK,
};
use crate::adapters::table::AtomTable;
use crate::xtal::ExperimentalData;

/// Owned handle to resident experimental crystallographic data. Free with
/// [`molex_experimental_data_free`].
pub struct molex_ExperimentalData;

// Buffer helpers (f32 analogs of the u8 `vec_to_out_buffer` /
// `molex_free_bytes` in the parent module; the density and B-factor paths are
// f32).

fn vec_f32_to_out_buffer(
    data: Vec<f32>,
    out_buf: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    let len = data.len();
    let mut boxed = data.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    unsafe {
        *out_buf = ptr;
        *out_len = len;
    }
    MOLEX_OK
}

/// Free a float buffer previously returned via `out_data` / `out_b`.
///
/// Safe to call with a null pointer (no-op). `len` must match the element
/// count the producing call reported (grid `nu*nv*nw` for density, atom count
/// for refined B-factors).
#[no_mangle]
pub extern "C" fn molex_free_floats(ptr: *mut f32, len: usize) {
    if ptr.is_null() {
        return;
    }
    drop(unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len))
    });
}

// Resident experimental data handle.

fn box_experimental_data(
    inner: ExperimentalData,
) -> *mut molex_ExperimentalData {
    Box::into_raw(Box::new(inner)).cast::<molex_ExperimentalData>()
}

fn experimental_data_inner<'a>(
    data: *const molex_ExperimentalData,
) -> Option<&'a ExperimentalData> {
    if data.is_null() {
        return None;
    }
    Some(unsafe { &*data.cast::<ExperimentalData>() })
}

/// Parse an sf.cif string into resident experimental data, resolving the space
/// group from the file's own symmetry tag.
///
/// Returns null on failure (missing symmetry/cell/reflections or unsupported
/// space group) with the message available via `molex_last_error_message`. The
/// caller owns the returned handle and must free it with
/// [`molex_experimental_data_free`].
#[no_mangle]
pub extern "C" fn molex_experimental_data_from_sf_cif(
    sf_ptr: *const c_char,
    len: usize,
    free_fraction: f64,
    seed: u64,
) -> *mut molex_ExperimentalData {
    let Some(bytes) = slice_from_raw(sf_ptr.cast::<u8>(), len) else {
        set_last_error(
            &"molex_experimental_data_from_sf_cif: null input pointer",
        );
        return std::ptr::null_mut();
    };
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in sf.cif input: {e}"));
            return std::ptr::null_mut();
        }
    };
    let Some(data) = ExperimentalData::from_sf_cif(s, free_fraction, seed)
    else {
        set_last_error(
            &"failed to parse sf.cif (missing symmetry/cell/reflections or \
              unsupported space group)",
        );
        return std::ptr::null_mut();
    };
    clear_last_error();
    box_experimental_data(data)
}

/// Parse an sf.cif string with an externally supplied International Tables
/// space-group number, for the common case where the sf.cif omits symmetry.
///
/// Returns null on failure (missing cell/reflections or unsupported space
/// group) with the message available via `molex_last_error_message`. The caller
/// owns the returned handle and must free it with
/// [`molex_experimental_data_free`].
#[no_mangle]
pub extern "C" fn molex_experimental_data_from_sf_cif_with_spacegroup(
    sf_ptr: *const c_char,
    len: usize,
    sg_number: u16,
    free_fraction: f64,
    seed: u64,
) -> *mut molex_ExperimentalData {
    let Some(bytes) = slice_from_raw(sf_ptr.cast::<u8>(), len) else {
        set_last_error(
            &"molex_experimental_data_from_sf_cif_with_spacegroup: null input \
              pointer",
        );
        return std::ptr::null_mut();
    };
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in sf.cif input: {e}"));
            return std::ptr::null_mut();
        }
    };
    let Some(data) = ExperimentalData::from_sf_cif_with_spacegroup(
        s,
        sg_number,
        free_fraction,
        seed,
    ) else {
        set_last_error(
            &"failed to parse sf.cif (missing cell/reflections or unsupported \
              space group)",
        );
        return std::ptr::null_mut();
    };
    clear_last_error();
    box_experimental_data(data)
}

/// Free a resident experimental-data handle.
///
/// Safe to call with a null pointer (no-op).
#[no_mangle]
pub extern "C" fn molex_experimental_data_free(
    data: *mut molex_ExperimentalData,
) {
    if data.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(data.cast::<ExperimentalData>()) });
}

/// Minimum Bragg spacing (Å) over the reflection set. Returns 0 for a null
/// handle.
#[no_mangle]
pub extern "C" fn molex_experimental_data_d_min(
    data: *const molex_ExperimentalData,
) -> f64 {
    experimental_data_inner(data).map_or(0.0, ExperimentalData::d_min)
}

/// The space-group International Tables number. Returns 0 for a null handle.
#[no_mangle]
pub extern "C" fn molex_experimental_data_space_group_number(
    data: *const molex_ExperimentalData,
) -> i32 {
    experimental_data_inner(data)
        .map_or(0, ExperimentalData::space_group_number)
}

/// Number of reflections in the resident set. Returns 0 for a null handle.
#[no_mangle]
pub extern "C" fn molex_experimental_data_num_reflections(
    data: *const molex_ExperimentalData,
) -> usize {
    experimental_data_inner(data).map_or(0, |d| d.reflections.len())
}

/// Write the density-map grid dimensions `(nu, nv, nw)` into the 3-element
/// output array. No-op if `out_dims` is null or the handle is invalid.
#[no_mangle]
pub extern "C" fn molex_experimental_data_grid_dims(
    data: *const molex_ExperimentalData,
    out_dims: *mut usize,
) {
    if out_dims.is_null() {
        return;
    }
    let Some(inner) = experimental_data_inner(data) else {
        return;
    };
    let [nu, nv, nw] = inner.derive_grid();
    unsafe {
        *out_dims.add(0) = nu;
        *out_dims.add(1) = nv;
        *out_dims.add(2) = nw;
    }
}

/// Compute the 2mFo-DFc electron density map from the assembly's atoms and the
/// resident data.
///
/// On success returns [`MOLEX_OK`], writes the grid dimensions `(nu, nv, nw)`
/// into `out_dims`, and writes the heap-allocated row-major `(u, v, w)` float32
/// grid pointer into `out_data`; the caller frees it with
/// [`molex_free_floats`] passing `nu*nv*nw` as the length. Hydrogens are
/// excluded from the splat.
///
/// Returns [`MOLEX_ERR_NULL`] on a null handle or output pointer, or
/// [`MOLEX_ERR`] if density computation fails (unsupported space group or empty
/// data); the message is available via `molex_last_error_message`.
#[no_mangle]
pub extern "C" fn molex_experimental_data_compute_density(
    data: *const molex_ExperimentalData,
    assembly: *const molex_Assembly,
    out_data: *mut *mut f32,
    out_dims: *mut usize,
) -> i32 {
    if out_data.is_null() || out_dims.is_null() {
        set_last_error(
            &"molex_experimental_data_compute_density: null pointer argument",
        );
        return MOLEX_ERR_NULL;
    }
    let Some(inner) = experimental_data_inner(data) else {
        set_last_error(
            &"molex_experimental_data_compute_density: null pointer argument",
        );
        return MOLEX_ERR_NULL;
    };
    let Some(assembly) = assembly_inner(assembly) else {
        set_last_error(
            &"molex_experimental_data_compute_density: null pointer argument",
        );
        return MOLEX_ERR_NULL;
    };

    let table = AtomTable::from_entities(assembly.entities());
    let Some(grid) = crate::xtal::density_from_atom_table(inner, &table) else {
        set_last_error(
            &"density computation failed (unsupported space group or empty \
              data)",
        );
        return MOLEX_ERR;
    };

    unsafe {
        *out_dims.add(0) = grid.nu;
        *out_dims.add(1) = grid.nv;
        *out_dims.add(2) = grid.nw;
    }
    clear_last_error();
    let mut written_len: usize = 0;
    vec_f32_to_out_buffer(grid.data, out_data, &raw mut written_len)
}

/// Refine per-atom isotropic B-factors against the resident data.
///
/// On success returns [`MOLEX_OK`], writes the full-length refined B-factor
/// buffer pointer + count into `out_b` / `out_n`, and writes the R-factors into
/// `out_r_work` / `out_r_free`. The buffer is aligned 1:1 to the
/// `AtomTable::from_entities` flat atom order and is freed with
/// [`molex_free_floats`] passing `out_n` as the length.
///
/// Hydrogens are excluded from the refinement; hydrogen atoms keep their input
/// B-factor in the returned array while non-hydrogen atoms carry the refined
/// value.
///
/// Returns [`MOLEX_ERR_NULL`] on a null handle or output pointer, or
/// [`MOLEX_ERR`] if the refinement pipeline fails; the message is available via
/// `molex_last_error_message`.
#[cfg(feature = "minimization")]
#[no_mangle]
pub extern "C" fn molex_experimental_data_refine_b_factors(
    data: *const molex_ExperimentalData,
    assembly: *const molex_Assembly,
    n_macro_cycles: usize,
    out_b: *mut *mut f32,
    out_n: *mut usize,
    out_r_work: *mut f64,
    out_r_free: *mut f64,
) -> i32 {
    if out_b.is_null()
        || out_n.is_null()
        || out_r_work.is_null()
        || out_r_free.is_null()
    {
        set_last_error(
            &"molex_experimental_data_refine_b_factors: null pointer argument",
        );
        return MOLEX_ERR_NULL;
    }
    let Some(inner) = experimental_data_inner(data) else {
        set_last_error(
            &"molex_experimental_data_refine_b_factors: null pointer argument",
        );
        return MOLEX_ERR_NULL;
    };
    let Some(assembly) = assembly_inner(assembly) else {
        set_last_error(
            &"molex_experimental_data_refine_b_factors: null pointer argument",
        );
        return MOLEX_ERR_NULL;
    };

    let table = AtomTable::from_entities(assembly.entities());
    let Some((full_b, r_work, r_free)) = crate::xtal::refine_b_from_atom_table(
        inner,
        &table,
        n_macro_cycles,
        |_, _, _| true,
    ) else {
        set_last_error(&"B-factor refinement failed");
        return MOLEX_ERR;
    };

    unsafe {
        *out_r_work = r_work;
        *out_r_free = r_free;
    }
    clear_last_error();
    vec_f32_to_out_buffer(full_b, out_b, out_n)
}
