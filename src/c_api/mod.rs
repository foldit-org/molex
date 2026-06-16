//! C ABI surface for molex.
//!
//! Gated behind the `c-api` cargo feature. Bridges `Assembly` and its
//! associated read accessors plus PDB/CIF/BCIF read + PDB write across the
//! Rust/C FFI boundary via opaque-pointer handles and heap-allocated
//! output buffers.
//!
//! # Naming
//!
//! Types exposed here use bare names (`Assembly`, `Entity`, `Residue`,
//! `Atom`, `molex_MoleculeType`). C++ consumers wrap them in `namespace molex`
//! for disambiguation; pure C consumers accept the global-namespace
//! placement. Free functions keep a `molex_` prefix because C has no
//! namespace mechanism and the prefix is part of the function name
//! itself rather than a per-layer transformation.
//!
//! # Ownership model
//!
//! - `*mut molex_Assembly` returned by parser entry points is owned by the
//!   caller and must be released with [`molex_assembly_free`].
//! - `*const molex_Entity`, `*const molex_Residue`, `*const molex_Atom` handles
//!   are non-owning views into the parent `Assembly`. They are valid for the
//!   lifetime of the parent and must not outlive it.
//! - Byte buffers returned via `out_buf` / `out_len` (e.g. from
//!   [`molex_assembly_to_pdb`]) are heap-allocated by Rust and must be freed
//!   with [`molex_free_bytes`].
//! - `const char*` + length pairs passed *into* parsers are borrowed for the
//!   duration of the call only.
//!
//! # Error reporting
//!
//! Fallible entry points either return a null handle (parsers) or a
//! nonzero status code (writers). On failure, a human-readable message
//! is recorded in thread-local storage and can be retrieved with
//! [`molex_last_error_message`]. Successful calls do not clear the
//! error state, so consumers should only read it immediately after a
//! failed call.

#![allow(
    clippy::missing_safety_doc,
    reason = "FFI surface: every fn that takes raw pointers documents its \
              preconditions in prose; safety contract is on the caller side \
              rather than each unsafe block."
)]
#![allow(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "FFI entry points receive raw pointers from C/C++ callers; \
              dereferencing them after a null check is the whole point."
)]
#![allow(
    non_camel_case_types,
    reason = "FFI types intentionally match their C-side spelling \
              (`molex_Assembly` etc.) so the same identifier appears in Rust \
              source, cbindgen output, and C consumer code."
)]

pub mod edit;
pub mod walk;

use std::cell::RefCell;
use std::ffi::c_char;

use crate::adapters::{bcif, cif, pdb};
use crate::assembly::Assembly;
use crate::entity::molecule::{EntityKind, MoleculeEntity, MoleculeType};
use crate::ops::wire::{deserialize_assembly, serialize_assembly};

thread_local! {
    static LAST_ERROR: RefCell<Option<Vec<u8>>> = const { RefCell::new(None) };
}

fn set_last_error<E: std::fmt::Display>(err: &E) {
    let msg = format!("{err}").into_bytes();
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(msg);
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------
//
// These are zero-sized tag types. Pointers in the FFI are tagged with
// these but always cast back to the real inner type before dereference.
// Defining them as 0-sized with no fields stops cbindgen from drilling
// into the inner Rust types and leaking molex internals into `molex.h`.

/// Owned, top-level assembly handle. Free with [`molex_assembly_free`].
pub struct molex_Assembly;

/// Non-owning view of a single entity within an assembly.
pub struct molex_Entity;

/// Non-owning view of a single polymer residue within an entity.
pub struct molex_Residue;

/// Non-owning view of a single atom within an entity.
pub struct molex_Atom;

/// Discriminant for an entity's molecule classification across the FFI
/// boundary. Stable integer codes so C consumers can pattern-match
/// without depending on Rust's enum layout.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum molex_MoleculeType {
    /// Amino acid polymer.
    Protein = 0,
    /// Deoxyribonucleic acid polymer.
    Dna = 1,
    /// Ribonucleic acid polymer.
    Rna = 2,
    /// Non-polymer small molecule (drug, substrate).
    Ligand = 3,
    /// Single-atom metal or halide ion.
    Ion = 4,
    /// Water molecule.
    Water = 5,
    /// Lipid or detergent molecule.
    Lipid = 6,
    /// Enzyme cofactor (heme, NAD, FAD, Fe-S cluster).
    Cofactor = 7,
    /// Crystallization solvent or buffer artifact.
    Solvent = 8,
}

impl From<MoleculeType> for molex_MoleculeType {
    fn from(value: MoleculeType) -> Self {
        match value {
            MoleculeType::Protein => molex_MoleculeType::Protein,
            MoleculeType::DNA => molex_MoleculeType::Dna,
            MoleculeType::RNA => molex_MoleculeType::Rna,
            MoleculeType::Ligand => molex_MoleculeType::Ligand,
            MoleculeType::Ion => molex_MoleculeType::Ion,
            MoleculeType::Water => molex_MoleculeType::Water,
            MoleculeType::Lipid => molex_MoleculeType::Lipid,
            MoleculeType::Cofactor => molex_MoleculeType::Cofactor,
            MoleculeType::Solvent => molex_MoleculeType::Solvent,
        }
    }
}

/// Structural discriminant for an entity across the FFI boundary.
///
/// The four-way taxonomy of the underlying variant, 1:1 with the Rust
/// `EntityKind`. Distinct from the finer [`molex_MoleculeType`]
/// classification. Stable integer codes so C consumers can pattern-match
/// without depending on Rust's enum layout.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum molex_EntityKind {
    /// A single protein chain.
    Protein = 0,
    /// A single DNA or RNA chain.
    NucleicAcid = 1,
    /// A single non-polymer molecule.
    SmallMolecule = 2,
    /// A group of identical small molecules.
    Bulk = 3,
}

impl From<EntityKind> for molex_EntityKind {
    fn from(value: EntityKind) -> Self {
        match value {
            EntityKind::Protein => molex_EntityKind::Protein,
            EntityKind::NucleicAcid => molex_EntityKind::NucleicAcid,
            EntityKind::SmallMolecule => molex_EntityKind::SmallMolecule,
            EntityKind::Bulk => molex_EntityKind::Bulk,
        }
    }
}

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

/// Success status returned by writer-style entry points.
pub const MOLEX_OK: i32 = 0;
/// Generic failure status returned by writer-style entry points.
pub const MOLEX_ERR: i32 = -1;
/// One of the input pointers was null.
pub const MOLEX_ERR_NULL: i32 = -2;

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------

/// Pointer to the most recent error message recorded on this thread.
///
/// `out_len` receives the length in bytes (excluding any null terminator;
/// the returned pointer is *not* guaranteed to be null-terminated). When
/// no error has been recorded, returns null and writes 0 to `out_len`.
///
/// The returned pointer is valid until the next FFI call that records or
/// clears the thread-local error state on the same thread. Copy it
/// immediately if the caller needs to retain the message.
#[no_mangle]
pub extern "C" fn molex_last_error_message(
    out_len: *mut usize,
) -> *const c_char {
    LAST_ERROR.with(|slot| {
        let borrow = slot.borrow();
        match borrow.as_ref() {
            None => {
                if !out_len.is_null() {
                    unsafe {
                        *out_len = 0;
                    }
                }
                std::ptr::null()
            }
            Some(msg) => {
                if !out_len.is_null() {
                    unsafe {
                        *out_len = msg.len();
                    }
                }
                msg.as_ptr().cast::<c_char>()
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Buffer / handle lifecycle
// ---------------------------------------------------------------------------

/// Free a buffer previously returned via `out_buf` / `out_len` from a
/// writer-style entry point.
///
/// Safe to call with a null pointer (no-op). `len` must match the value
/// originally written to `out_len`.
#[no_mangle]
pub extern "C" fn molex_free_bytes(bytes: *mut u8, len: usize) {
    if bytes.is_null() {
        return;
    }
    drop(unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(bytes, len))
    });
}

/// Free an assembly handle returned by a parser entry point.
///
/// Safe to call with a null pointer (no-op). Borrowed sub-handles
/// (`*const molex_Entity`, `*const molex_Residue`, `*const molex_Atom`)
/// returned from the assembly are invalidated and must not be used after this
/// call.
#[no_mangle]
pub extern "C" fn molex_assembly_free(assembly: *mut molex_Assembly) {
    if assembly.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(assembly.cast::<Assembly>()) });
}

// ---------------------------------------------------------------------------
// Parser entry points (PDB / CIF / BCIF -> Assembly)
// ---------------------------------------------------------------------------

fn assembly_from_entities(
    entities: Vec<MoleculeEntity>,
) -> *mut molex_Assembly {
    let inner = Assembly::new(entities);
    Box::into_raw(Box::new(inner)).cast::<molex_Assembly>()
}

pub(crate) fn assembly_inner<'a>(
    assembly: *const molex_Assembly,
) -> Option<&'a Assembly> {
    if assembly.is_null() {
        return None;
    }
    Some(unsafe { &*assembly.cast::<Assembly>() })
}

pub(crate) fn assembly_inner_mut<'a>(
    assembly: *mut molex_Assembly,
) -> Option<&'a mut Assembly> {
    if assembly.is_null() {
        return None;
    }
    Some(unsafe { &mut *assembly.cast::<Assembly>() })
}

/// Project an assembly to a fresh heavy-complete handle: polymer entities
/// gain their missing heavy atoms (see [`molex::Assembly::normalize`]). The
/// source handle is left untouched.
///
/// Completion runs on surviving residues only; residues dropped at
/// construction are not resurrected.
///
/// Returns null on a null input with the error message available via
/// [`molex_last_error_message`]. The returned handle is caller-owned and
/// must be freed with [`molex_assembly_free`]; freeing it is independent
/// of the source handle.
#[no_mangle]
pub extern "C" fn molex_assembly_normalize(
    assembly: *const molex_Assembly,
) -> *mut molex_Assembly {
    let Some(inner) = assembly_inner(assembly) else {
        set_last_error(&"molex_assembly_normalize: null input pointer");
        return std::ptr::null_mut();
    };
    clear_last_error();
    Box::into_raw(Box::new(inner.normalize())).cast::<molex_Assembly>()
}

/// Project an assembly to a fresh all-atom handle: polymer entities gain
/// their template hydrogens (see [`molex::Assembly::to_all_atom`]). The
/// source handle is left untouched.
///
/// Returns null on a null input with the error message available via
/// [`molex_last_error_message`]. The returned handle is caller-owned and
/// must be freed with [`molex_assembly_free`]; freeing it is independent
/// of the source handle.
#[no_mangle]
pub extern "C" fn molex_assembly_to_all_atom(
    assembly: *const molex_Assembly,
) -> *mut molex_Assembly {
    let Some(inner) = assembly_inner(assembly) else {
        set_last_error(&"molex_assembly_to_all_atom: null input pointer");
        return std::ptr::null_mut();
    };
    clear_last_error();
    Box::into_raw(Box::new(inner.to_all_atom())).cast::<molex_Assembly>()
}

fn slice_from_raw<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]> {
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Parse a PDB-format string into an `Assembly`.
///
/// Returns null on failure with the error message available via
/// [`molex_last_error_message`]. The caller owns the returned handle
/// and must free it with [`molex_assembly_free`].
#[no_mangle]
pub extern "C" fn molex_pdb_str_to_assembly(
    str_ptr: *const c_char,
    len: usize,
) -> *mut molex_Assembly {
    let Some(bytes) = slice_from_raw(str_ptr.cast::<u8>(), len) else {
        set_last_error(&"molex_pdb_str_to_assembly: null input pointer");
        return std::ptr::null_mut();
    };
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in PDB input: {e}"));
            return std::ptr::null_mut();
        }
    };
    match pdb::pdb_str_to_entities(s) {
        Ok(entities) => {
            clear_last_error();
            assembly_from_entities(entities)
        }
        Err(e) => {
            set_last_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Parse an mmCIF-format string into an `Assembly`.
///
/// Returns null on failure with the error message available via
/// [`molex_last_error_message`]. The caller owns the returned handle
/// and must free it with [`molex_assembly_free`].
#[no_mangle]
pub extern "C" fn molex_cif_str_to_assembly(
    str_ptr: *const c_char,
    len: usize,
) -> *mut molex_Assembly {
    let Some(bytes) = slice_from_raw(str_ptr.cast::<u8>(), len) else {
        set_last_error(&"molex_cif_str_to_assembly: null input pointer");
        return std::ptr::null_mut();
    };
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8 in mmCIF input: {e}"));
            return std::ptr::null_mut();
        }
    };
    match cif::mmcif_str_to_entities(s) {
        Ok(entities) => {
            clear_last_error();
            assembly_from_entities(entities)
        }
        Err(e) => {
            set_last_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Decode assembly binary bytes into an `Assembly`.
///
/// Bytes must start with the `ASSEMBLY` magic header and version byte
/// followed by the entity / atom payload (see `crate::ops::wire`). Returns
/// null on failure with the error message available via
/// [`molex_last_error_message`]. The caller owns the returned handle and
/// must free it with [`molex_assembly_free`].
#[no_mangle]
pub extern "C" fn molex_bytes_to_assembly(
    bytes_ptr: *const u8,
    len: usize,
) -> *mut molex_Assembly {
    let Some(bytes) = slice_from_raw(bytes_ptr, len) else {
        set_last_error(&"molex_bytes_to_assembly: null input pointer");
        return std::ptr::null_mut();
    };
    match deserialize_assembly(bytes) {
        Ok(assembly) => {
            clear_last_error();
            Box::into_raw(Box::new(assembly)).cast::<molex_Assembly>()
        }
        Err(e) => {
            set_last_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Decode BinaryCIF bytes into an `Assembly`.
///
/// Returns null on failure with the error message available via
/// [`molex_last_error_message`]. The caller owns the returned handle
/// and must free it with [`molex_assembly_free`].
#[no_mangle]
pub extern "C" fn molex_bcif_to_assembly(
    bytes_ptr: *const u8,
    len: usize,
) -> *mut molex_Assembly {
    let Some(bytes) = slice_from_raw(bytes_ptr, len) else {
        set_last_error(&"molex_bcif_to_assembly: null input pointer");
        return std::ptr::null_mut();
    };
    match bcif::bcif_to_entities(bytes) {
        Ok(entities) => {
            clear_last_error();
            assembly_from_entities(entities)
        }
        Err(e) => {
            set_last_error(&e);
            std::ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------------------------
// Writer entry points (Assembly -> PDB)
// ---------------------------------------------------------------------------

fn vec_to_out_buffer(
    bytes: Vec<u8>,
    out_buf: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    let len = bytes.len();
    let mut boxed = bytes.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    unsafe {
        *out_buf = ptr;
        *out_len = len;
    }
    MOLEX_OK
}

/// Emit an `Assembly` as assembly binary bytes.
///
/// On success returns [`MOLEX_OK`] and writes the heap-allocated buffer
/// pointer + length to `out_buf` / `out_len`; the caller frees with
/// [`molex_free_bytes`]. On failure returns a nonzero status and the
/// error message is available via [`molex_last_error_message`].
#[no_mangle]
pub extern "C" fn molex_assembly_to_bytes(
    assembly: *const molex_Assembly,
    out_buf: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if out_buf.is_null() || out_len.is_null() {
        set_last_error(&"molex_assembly_to_bytes: null pointer argument");
        return MOLEX_ERR_NULL;
    }
    let Some(assembly) = assembly_inner(assembly) else {
        set_last_error(&"molex_assembly_to_bytes: null pointer argument");
        return MOLEX_ERR_NULL;
    };
    match serialize_assembly(assembly) {
        Ok(bytes) => {
            clear_last_error();
            vec_to_out_buffer(bytes, out_buf, out_len)
        }
        Err(e) => {
            set_last_error(&e);
            MOLEX_ERR
        }
    }
}

/// Emit an `Assembly` as a PDB-format byte buffer.
///
/// On success returns [`MOLEX_OK`] and writes the heap-allocated buffer
/// pointer + length to `out_buf` / `out_len`; the caller frees with
/// [`molex_free_bytes`]. On failure returns a nonzero status and the
/// error message is available via [`molex_last_error_message`].
#[no_mangle]
pub extern "C" fn molex_assembly_to_pdb(
    assembly: *const molex_Assembly,
    out_buf: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if out_buf.is_null() || out_len.is_null() {
        set_last_error(&"molex_assembly_to_pdb: null pointer argument");
        return MOLEX_ERR_NULL;
    }
    let Some(assembly) = assembly_inner(assembly) else {
        set_last_error(&"molex_assembly_to_pdb: null pointer argument");
        return MOLEX_ERR_NULL;
    };
    match pdb::assembly_to_pdb(assembly) {
        Ok(s) => {
            clear_last_error();
            vec_to_out_buffer(s.into_bytes(), out_buf, out_len)
        }
        Err(e) => {
            set_last_error(&e);
            MOLEX_ERR
        }
    }
}
