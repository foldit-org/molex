#ifndef MOLEX_H
#define MOLEX_H

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

/**
 * Default solvent probe radius (water) in Angstroms.
 */
#define DEFAULT_PROBE_RADIUS 1.4

/**
 * Default number of test points per atom (golden-spiral lattice).
 */
#define DEFAULT_N_POINTS 960

/**
 * Wire format version written after [`ASSEMBLY_MAGIC`].
 */
#define ASSEMBLY_VERSION 2

/**
 * Wire format version written after [`DELTA_MAGIC`].
 */
#define DELTA_VERSION 2

/**
 * Default probe radius (Angstroms) for bulk solvent mask.
 */
#define DEFAULT_R_PROBE 1.0

/**
 * Default shrink radius (Angstroms) for removing thin protein bridges.
 */
#define DEFAULT_R_SHRINK 1.1

/**
 * Common denominator for integer-encoded symmetry operations.
 */
#define DEN 24

/**
 * Success status returned by writer-style entry points.
 */
#define MOLEX_OK 0

/**
 * Generic failure status returned by writer-style entry points.
 */
#define MOLEX_ERR -1

/**
 * One of the input pointers was null.
 */
#define MOLEX_ERR_NULL -2

/**
 * Completion level across the FFI boundary: `Raw < Heavy < AllAtom`.
 *
 * Mirrors `molex::Completion`. Stable integer codes so C consumers can pass
 * a level without depending on Rust's enum layout. `Raw` keeps atoms exactly
 * as given (no fabricated heavy atoms, input hydrogens preserved); `Heavy`
 * fabricates absent heavy template atoms and drops input hydrogens (the
 * file-ingest default that the no-level parse entry points use); `AllAtom`
 * additionally places template hydrogens.
 */
enum molex_Completion
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : int32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * Atoms exactly as given: skip completion, keep input hydrogens.
   */
  MOLEX_COMPLETION_RAW = 0,
  /**
   * Fabricate absent heavy template atoms (the file-ingest level).
   */
  MOLEX_COMPLETION_HEAVY = 1,
  /**
   * Fabricate absent heavy atoms and template hydrogens.
   */
  MOLEX_COMPLETION_ALL_ATOM = 2,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum molex_Completion molex_Completion;
#else
typedef int32_t molex_Completion;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Variant tag kind discriminant on the C boundary.
 */
enum molex_VariantKind
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : int32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * Chain N-terminus patch.
   */
  MOLEX_VARIANT_KIND_N_TERMINUS = 1,
  /**
   * Chain C-terminus patch.
   */
  MOLEX_VARIANT_KIND_C_TERMINUS = 2,
  /**
   * Participates in a disulfide bond.
   */
  MOLEX_VARIANT_KIND_DISULFIDE = 3,
  /**
   * Non-canonical protonation. Reads
   * `molex_Variant::protonation`.
   */
  MOLEX_VARIANT_KIND_PROTONATION = 4,
  /**
   * Open-ended tag carrying a UTF-8 string in
   * `molex_Variant::{str_ptr, str_len}`.
   */
  MOLEX_VARIANT_KIND_OTHER = 254,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum molex_VariantKind molex_VariantKind;
#else
typedef int32_t molex_VariantKind;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Protonation state discriminant. Read only when `kind ==
 * Protonation`. For `Custom`, the string in
 * `molex_Variant::{str_ptr, str_len}` carries the variant name.
 */
enum molex_ProtonationKind
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : int32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * `HID`: delta-protonated histidine.
   */
  MOLEX_PROTONATION_KIND_HIS_DELTA = 1,
  /**
   * `HIE`: epsilon-protonated histidine.
   */
  MOLEX_PROTONATION_KIND_HIS_EPSILON = 2,
  /**
   * `HIP`: doubly-protonated histidine.
   */
  MOLEX_PROTONATION_KIND_HIS_DOUBLY = 3,
  /**
   * Anything else; payload string carries the variant name.
   */
  MOLEX_PROTONATION_KIND_CUSTOM = 255,
  /**
   * Default placeholder when `kind != Protonation`. The push
   * functions ignore this field unless the variant kind is
   * `Protonation`.
   */
  MOLEX_PROTONATION_KIND_UNUSED = 0,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum molex_ProtonationKind molex_ProtonationKind;
#else
typedef int32_t molex_ProtonationKind;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Per-edit kind discriminant.
 *
 * Returned by [`molex_edits_kind_at`]; caller dispatches on the
 * value to pick the right per-variant getter. `AddEntity` /
 * `RemoveEntity` are listed for completeness but never appear in
 * lists obtained from `molex_delta_to_edits` (the delta
 * serializer rejects topology edits up front).
 */
enum molex_EditKind
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : int32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * Sentinel returned when the list pointer is null or the index is
   * out of bounds.
   */
  MOLEX_EDIT_KIND_INVALID = 0,
  /**
   * `SetEntityCoords`: bulk per-entity coordinate update. Read via
   * [`molex_edits_set_entity_coords_at`].
   */
  MOLEX_EDIT_KIND_SET_ENTITY_COORDS = 1,
  /**
   * `SetResidueCoords`: per-residue coordinate update inside a
   * polymer entity. Read via
   * [`molex_edits_set_residue_coords_at`].
   */
  MOLEX_EDIT_KIND_SET_RESIDUE_COORDS = 2,
  /**
   * `MutateResidue`: residue identity + atoms + variants
   * replacement. Read via [`molex_edits_mutate_residue_at`].
   */
  MOLEX_EDIT_KIND_MUTATE_RESIDUE = 3,
  /**
   * `SetVariants`: replace a residue's variant tag list. Read
   * via [`molex_edits_set_variants_at`].
   */
  MOLEX_EDIT_KIND_SET_VARIANTS = 4,
  /**
   * `AddEntity`: topology edit; appears only in Rust-side edit
   * lists, never in lists decoded from delta.
   */
  MOLEX_EDIT_KIND_ADD_ENTITY = 5,
  /**
   * `RemoveEntity`: topology edit; appears only in Rust-side edit
   * lists, never in lists decoded from delta.
   */
  MOLEX_EDIT_KIND_REMOVE_ENTITY = 6,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum molex_EditKind molex_EditKind;
#else
typedef int32_t molex_EditKind;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Discriminant for an entity's molecule classification across the FFI
 * boundary. Stable integer codes so C consumers can pattern-match
 * without depending on Rust's enum layout.
 */
enum molex_MoleculeType
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : int32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * Amino acid polymer.
   */
  MOLEX_MOLECULE_TYPE_PROTEIN = 0,
  /**
   * Deoxyribonucleic acid polymer.
   */
  MOLEX_MOLECULE_TYPE_DNA = 1,
  /**
   * Ribonucleic acid polymer.
   */
  MOLEX_MOLECULE_TYPE_RNA = 2,
  /**
   * Non-polymer small molecule (drug, substrate).
   */
  MOLEX_MOLECULE_TYPE_LIGAND = 3,
  /**
   * Single-atom metal or halide ion.
   */
  MOLEX_MOLECULE_TYPE_ION = 4,
  /**
   * Water molecule.
   */
  MOLEX_MOLECULE_TYPE_WATER = 5,
  /**
   * Lipid or detergent molecule.
   */
  MOLEX_MOLECULE_TYPE_LIPID = 6,
  /**
   * Enzyme cofactor (heme, NAD, FAD, Fe-S cluster).
   */
  MOLEX_MOLECULE_TYPE_COFACTOR = 7,
  /**
   * Crystallization solvent or buffer artifact.
   */
  MOLEX_MOLECULE_TYPE_SOLVENT = 8,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum molex_MoleculeType molex_MoleculeType;
#else
typedef int32_t molex_MoleculeType;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Structural discriminant for an entity across the FFI boundary.
 *
 * The four-way taxonomy of the underlying variant, 1:1 with the Rust
 * `EntityKind`. Distinct from the finer [`molex_MoleculeType`]
 * classification. Stable integer codes so C consumers can pattern-match
 * without depending on Rust's enum layout.
 */
enum molex_EntityKind
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : int32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  /**
   * A single protein chain.
   */
  MOLEX_ENTITY_KIND_PROTEIN = 0,
  /**
   * A single DNA or RNA chain.
   */
  MOLEX_ENTITY_KIND_NUCLEIC_ACID = 1,
  /**
   * A single non-polymer molecule.
   */
  MOLEX_ENTITY_KIND_SMALL_MOLECULE = 2,
  /**
   * A group of identical small molecules.
   */
  MOLEX_ENTITY_KIND_BULK = 3,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum molex_EntityKind molex_EntityKind;
#else
typedef int32_t molex_EntityKind;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

/**
 * Owned, top-level assembly handle. Free with [`molex_assembly_free`].
 */
typedef struct molex_Assembly molex_Assembly;

/**
 * Owned, ordered list of typed Assembly edits.
 *
 * Free with [`molex_edits_free`]. Push typed entries via the
 * `molex_edits_push_*` constructors; or dump to wire bytes with
 * [`molex_edits_to_delta`].
 */
typedef struct molex_EditList molex_EditList;

/**
 * Non-owning view of a single entity within an assembly.
 */
typedef struct molex_Entity molex_Entity;

/**
 * Owned handle to resident experimental crystallographic data. Free with
 * [`molex_experimental_data_free`].
 */
typedef struct molex_ExperimentalData molex_ExperimentalData;

/**
 * Non-owning view of a single polymer residue within an entity.
 */
typedef struct molex_Residue molex_Residue;

/**
 * One atom row crossed-over to C. Layout mirrors the assembly wire atom
 * row (12 + 4 + 2 = 18 bytes excluding chain/residue context which
 * is conveyed separately).
 */
typedef struct {
  /**
   * X coordinate.
   */
  float position_x;
  /**
   * Y coordinate.
   */
  float position_y;
  /**
   * Z coordinate.
   */
  float position_z;
  /**
   * PDB atom name, space-padded.
   */
  uint8_t name[4];
  /**
   * Element symbol, null-padded if 1 character (e.g. `['C', 0]`).
   */
  uint8_t element[2];
} molex_AtomRow;

/**
 * One variant tag as crossed-over to C.
 */
typedef struct {
  /**
   * Discriminant.
   */
  molex_VariantKind kind;
  /**
   * Protonation sub-discriminant. Read only when `kind ==
   * Protonation`.
   */
  molex_ProtonationKind protonation;
  /**
   * UTF-8 bytes for `Other` and `Protonation::Custom`. Borrowed for
   * the duration of the call.
   */
  const uint8_t *str_ptr;
  /**
   * Length of `str_ptr` in bytes.
   */
  uintptr_t str_len;
} molex_Variant;

/**
 * By-value handle locating a single atom within an entity's columns.
 *
 * Carries a borrowed `entity` pointer plus the atom's `index`. A null
 * `entity` is the invalid/out-of-bounds sentinel; the `molex_atom_*`
 * accessors return 0/null for it.
 */
typedef struct {
  /**
   * Entity the atom belongs to (borrowed; null means invalid).
   */
  const molex_Entity *entity;
  /**
   * Atom index into the entity's columns.
   */
  uintptr_t index;
} molex_Atom;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Pointer to the most recent error message recorded on this thread.
 *
 * `out_len` receives the length in bytes (excluding any null terminator;
 * the returned pointer is *not* guaranteed to be null-terminated). When
 * no error has been recorded, returns null and writes 0 to `out_len`.
 *
 * The returned pointer is valid until the next FFI call that records or
 * clears the thread-local error state on the same thread. Copy it
 * immediately if the caller needs to retain the message.
 */

const char *molex_last_error_message(uintptr_t *out_len)
;

/**
 * Free a buffer previously returned via `out_buf` / `out_len` from a
 * writer-style entry point.
 *
 * Safe to call with a null pointer (no-op). `len` must match the value
 * originally written to `out_len`.
 */

void molex_free_bytes(uint8_t *bytes,
                      uintptr_t len)
;

/**
 * Free an assembly handle returned by a parser entry point.
 *
 * Safe to call with a null pointer (no-op). Borrowed sub-handles
 * (`*const molex_Entity`, `*const molex_Residue`, `*const molex_Atom`)
 * returned from the assembly are invalidated and must not be used after this
 * call.
 */

void molex_assembly_free(molex_Assembly *assembly)
;

/**
 * Project an assembly to a fresh heavy-complete handle: polymer entities
 * gain their missing heavy atoms (see [`crate::Assembly::normalize`]). The
 * source handle is left untouched.
 *
 * Completion runs on surviving residues only; residues dropped at
 * construction are not resurrected.
 *
 * Returns null on a null input with the error message available via
 * [`molex_last_error_message`]. The returned handle is caller-owned and
 * must be freed with [`molex_assembly_free`]; freeing it is independent
 * of the source handle.
 */

molex_Assembly *molex_assembly_normalize(const molex_Assembly *assembly)
;

/**
 * Project an assembly to a fresh all-atom handle: polymer entities gain
 * their template hydrogens (see [`crate::Assembly::to_all_atom`]). The
 * source handle is left untouched.
 *
 * Returns null on a null input with the error message available via
 * [`molex_last_error_message`]. The returned handle is caller-owned and
 * must be freed with [`molex_assembly_free`]; freeing it is independent
 * of the source handle.
 */

molex_Assembly *molex_assembly_to_all_atom(const molex_Assembly *assembly)
;

/**
 * Project an assembly to a fresh handle re-completed at `level`.
 *
 * Polymer entities are rebuilt (see [`crate::Assembly::complete`]); the
 * source handle is left untouched. [`molex_assembly_normalize`] and
 * [`molex_assembly_to_all_atom`] are the `Heavy` / `AllAtom` shortcuts.
 *
 * Completion runs on surviving residues only; residues dropped at
 * construction are not resurrected.
 *
 * Returns null on a null input with the error message available via
 * [`molex_last_error_message`]. The returned handle is caller-owned and
 * must be freed with [`molex_assembly_free`]; freeing it is independent
 * of the source handle.
 */

molex_Assembly *molex_assembly_complete(const molex_Assembly *assembly,
                                        molex_Completion level)
;

/**
 * Parse a PDB-format string into an `Assembly` at [`Completion::Heavy`]
 * (missing heavy atoms filled, input hydrogens dropped). For another level
 * use [`molex_pdb_str_to_assembly_with_completion`].
 *
 * Returns null on failure with the error message available via
 * [`molex_last_error_message`]. The caller owns the returned handle
 * and must free it with [`molex_assembly_free`].
 */

molex_Assembly *molex_pdb_str_to_assembly(const char *str_ptr,
                                          uintptr_t len)
;

/**
 * Parse a PDB-format string into an `Assembly` at the given completion
 * `level`. Like [`molex_pdb_str_to_assembly`] but selects the level.
 *
 * Returns null on failure with the error message available via
 * [`molex_last_error_message`]. The caller owns the returned handle
 * and must free it with [`molex_assembly_free`].
 */

molex_Assembly *molex_pdb_str_to_assembly_with_completion(const char *str_ptr,
                                                          uintptr_t len,
                                                          molex_Completion level)
;

/**
 * Parse an mmCIF-format string into an `Assembly` at [`Completion::Heavy`].
 * For another level use [`molex_cif_str_to_assembly_with_completion`].
 *
 * Returns null on failure with the error message available via
 * [`molex_last_error_message`]. The caller owns the returned handle
 * and must free it with [`molex_assembly_free`].
 */

molex_Assembly *molex_cif_str_to_assembly(const char *str_ptr,
                                          uintptr_t len)
;

/**
 * Parse an mmCIF-format string into an `Assembly` at the given completion
 * `level`. Like [`molex_cif_str_to_assembly`] but selects the level.
 *
 * Returns null on failure with the error message available via
 * [`molex_last_error_message`]. The caller owns the returned handle
 * and must free it with [`molex_assembly_free`].
 */

molex_Assembly *molex_cif_str_to_assembly_with_completion(const char *str_ptr,
                                                          uintptr_t len,
                                                          molex_Completion level)
;

/**
 * Decode assembly binary bytes into an `Assembly`.
 *
 * Bytes must start with the `ASSEMBLY` magic header and version byte
 * followed by the entity / atom payload (see `crate::ops::wire`). Returns
 * null on failure with the error message available via
 * [`molex_last_error_message`]. The caller owns the returned handle and
 * must free it with [`molex_assembly_free`].
 */

molex_Assembly *molex_bytes_to_assembly(const uint8_t *bytes_ptr,
                                        uintptr_t len)
;

/**
 * Decode BinaryCIF bytes into an `Assembly` at [`Completion::Heavy`]. For
 * another level use [`molex_bcif_to_assembly_with_completion`].
 *
 * Returns null on failure with the error message available via
 * [`molex_last_error_message`]. The caller owns the returned handle
 * and must free it with [`molex_assembly_free`].
 */

molex_Assembly *molex_bcif_to_assembly(const uint8_t *bytes_ptr,
                                       uintptr_t len)
;

/**
 * Decode BinaryCIF bytes into an `Assembly` at the given completion `level`.
 * Like [`molex_bcif_to_assembly`] but selects the level.
 *
 * Returns null on failure with the error message available via
 * [`molex_last_error_message`]. The caller owns the returned handle
 * and must free it with [`molex_assembly_free`].
 */

molex_Assembly *molex_bcif_to_assembly_with_completion(const uint8_t *bytes_ptr,
                                                       uintptr_t len,
                                                       molex_Completion level)
;

/**
 * Emit an `Assembly` as assembly binary bytes.
 *
 * On success returns [`MOLEX_OK`] and writes the heap-allocated buffer
 * pointer + length to `out_buf` / `out_len`; the caller frees with
 * [`molex_free_bytes`]. On failure returns a nonzero status and the
 * error message is available via [`molex_last_error_message`].
 */

int32_t molex_assembly_to_bytes(const molex_Assembly *assembly,
                                uint8_t **out_buf,
                                uintptr_t *out_len)
;

/**
 * Emit an `Assembly` as a PDB-format byte buffer.
 *
 * On success returns [`MOLEX_OK`] and writes the heap-allocated buffer
 * pointer + length to `out_buf` / `out_len`; the caller frees with
 * [`molex_free_bytes`]. On failure returns a nonzero status and the
 * error message is available via [`molex_last_error_message`].
 */

int32_t molex_assembly_to_pdb(const molex_Assembly *assembly,
                              uint8_t **out_buf,
                              uintptr_t *out_len)
;

/**
 * Emit an `Assembly` as an mmCIF-format byte buffer.
 *
 * On success returns [`MOLEX_OK`] and writes the heap-allocated buffer
 * pointer + length to `out_buf` / `out_len`; the caller frees with
 * [`molex_free_bytes`]. On failure (null argument) returns a nonzero
 * status and the error message is available via
 * [`molex_last_error_message`].
 */

int32_t molex_assembly_to_mmcif(const molex_Assembly *assembly,
                                uint8_t **out_buf,
                                uintptr_t *out_len)
;

/**
 * Construct an empty edit list.
 */

molex_EditList *molex_edits_new(void)
;

/**
 * Free an edit list returned by `molex_edits_new` or
 * `molex_delta_to_edits`. Safe to call with a null pointer (no-op).
 */

void molex_edits_free(molex_EditList *list)
;

/**
 * Number of edits currently in the list. Returns 0 if `list` is null.
 */

uintptr_t molex_edits_count(const molex_EditList *list)
;

/**
 * Append a `SetEntityCoords` edit to `list`. `coords_xyz` must point
 * to `coord_count * 3` floats (x,y,z,x,y,z,...).
 */

int32_t molex_edits_push_set_entity_coords(molex_EditList *list,
                                           uint32_t entity_id,
                                           const float *coords_xyz,
                                           uintptr_t coord_count)
;

/**
 * Append a `SetResidueCoords` edit to `list`.
 */

int32_t molex_edits_push_set_residue_coords(molex_EditList *list,
                                            uint32_t entity_id,
                                            uintptr_t residue_idx,
                                            const float *coords_xyz,
                                            uintptr_t coord_count)
;

/**
 * Append a `MutateResidue` edit to `list`. `new_name` is a 3-byte
 * (space-padded) residue name. `atoms` carries the new atom list;
 * `variants` carries the residue's variant tags.
 */

int32_t molex_edits_push_mutate_residue(molex_EditList *list,
                                        uint32_t entity_id,
                                        uintptr_t residue_idx,
                                        const uint8_t *new_name,
                                        const molex_AtomRow *atoms,
                                        uintptr_t atom_count,
                                        const molex_Variant *variants,
                                        uintptr_t variant_count)
;

/**
 * Append a `SetVariants` edit to `list`.
 */

int32_t molex_edits_push_set_variants(molex_EditList *list,
                                      uint32_t entity_id,
                                      uintptr_t residue_idx,
                                      const molex_Variant *variants,
                                      uintptr_t variant_count)
;

/**
 * Serialize the edit list to delta bytes.
 *
 * On success returns [`MOLEX_OK`] and writes the heap-allocated
 * buffer pointer + length to `out_buf` / `out_len`; caller frees with
 * `molex_free_bytes`.
 */

int32_t molex_edits_to_delta(const molex_EditList *list,
                             uint8_t **out_buf,
                             uintptr_t *out_len)
;

/**
 * Decode delta bytes into a new edit list.
 *
 * Returns null on failure with the error message available via
 * `molex_last_error_message`. Caller frees with `molex_edits_free`.
 */

molex_EditList *molex_delta_to_edits(const uint8_t *bytes_ptr,
                                     uintptr_t len)
;

/**
 * Apply every edit in `list` to `assembly` in order. Stops at the
 * first failure and returns nonzero; edits applied before the
 * failing one stay applied (each having bumped the generation
 * counter).
 */

int32_t molex_assembly_apply_edits(molex_Assembly *assembly,
                                   const molex_EditList *list)
;

/**
 * Decode delta bytes and apply them to `assembly`.
 *
 * Equivalent to calling `molex_delta_to_edits` then
 * `molex_assembly_apply_edits` then `molex_edits_free`, but avoids
 * the round-trip through a separate list handle on the apply hot
 * path.
 */

int32_t molex_assembly_apply_delta(molex_Assembly *assembly,
                                   const uint8_t *bytes_ptr,
                                   uintptr_t len)
;

/**
 * Return the kind of the edit at `index`. Returns
 * [`molex_EditKind::Invalid`] when `list` is null or `index >=
 * molex_edits_count(list)`.
 */

molex_EditKind molex_edits_kind_at(const molex_EditList *list,
                                   uintptr_t index)
;

/**
 * Read the fields of a `SetEntityCoords` edit at `index`.
 *
 * On success returns [`MOLEX_OK`] and writes the entity id plus a
 * borrowed coord pointer (3 * `*out_coord_count` `f32`s in
 * x,y,z,x,y,z order). The coord pointer is valid until the parent
 * EditList is freed or mutated.
 *
 * Returns [`MOLEX_ERR_NULL`] on null inputs; returns [`MOLEX_ERR`]
 * when `index` is out of range or the edit at `index` is not a
 * `SetEntityCoords` (caller should consult [`molex_edits_kind_at`]
 * before calling).
 */

int32_t molex_edits_set_entity_coords_at(const molex_EditList *list,
                                         uintptr_t index,
                                         uint32_t *out_entity_id,
                                         const float **out_coords_xyz,
                                         uintptr_t *out_coord_count)
;

/**
 * Read the fields of a `SetResidueCoords` edit at `index`. Same
 * lifetime contract on `out_coords_xyz` as
 * [`molex_edits_set_entity_coords_at`].
 */

int32_t molex_edits_set_residue_coords_at(const molex_EditList *list,
                                          uintptr_t index,
                                          uint32_t *out_entity_id,
                                          uintptr_t *out_residue_idx,
                                          const float **out_coords_xyz,
                                          uintptr_t *out_coord_count)
;

/**
 * Read the fields of a `MutateResidue` edit at `index`.
 *
 * `out_new_name` receives a borrowed 3-byte pointer (lifetime tied
 * to the parent EditList).
 *
 * `out_atoms` / `out_atom_count` receive a freshly-allocated
 * `molex_AtomRow` array derived from the edit's internal `Atom`
 * storage. Caller MUST free with [`molex_atom_rows_free`]. The
 * conversion is lossy on the `occupancy` / `b_factor` /
 * `formal_charge` fields (those don't ride delta and aren't
 * representable in `molex_AtomRow`).
 *
 * `out_variants` / `out_variant_count` receive a freshly-allocated
 * `molex_Variant` array; the `str_ptr` fields inside it point INTO
 * the parent EditList's owned `String`s (valid until the list is
 * freed or mutated). Caller MUST free the outer array with
 * [`molex_variants_free`].
 */

int32_t molex_edits_mutate_residue_at(const molex_EditList *list,
                                      uintptr_t index,
                                      uint32_t *out_entity_id,
                                      uintptr_t *out_residue_idx,
                                      const uint8_t **out_new_name,
                                      const molex_AtomRow **out_atoms,
                                      uintptr_t *out_atom_count,
                                      const molex_Variant **out_variants,
                                      uintptr_t *out_variant_count)
;

/**
 * Read the fields of a `SetVariants` edit at `index`. Same lifetime
 * contract on `out_variants` as
 * [`molex_edits_mutate_residue_at`]; caller frees via
 * [`molex_variants_free`].
 */

int32_t molex_edits_set_variants_at(const molex_EditList *list,
                                    uintptr_t index,
                                    uint32_t *out_entity_id,
                                    uintptr_t *out_residue_idx,
                                    const molex_Variant **out_variants,
                                    uintptr_t *out_variant_count)
;

/**
 * Free a `molex_AtomRow` array returned by
 * [`molex_edits_mutate_residue_at`]. Safe to call with a null
 * pointer (no-op). `count` MUST match the value originally written
 * to `out_atom_count`.
 */

void molex_atom_rows_free(molex_AtomRow *ptr,
                          uintptr_t count)
;

/**
 * Free a `molex_Variant` array.
 *
 * Returned by [`molex_edits_mutate_residue_at`] or
 * [`molex_edits_set_variants_at`]. Safe to call with a null pointer
 * (no-op). The borrowed string payloads pointed at by each entry's
 * `str_ptr` are NOT freed by this call -- they belong to the parent
 * EditList.
 */

void molex_variants_free(molex_Variant *ptr,
                         uintptr_t count)
;

/**
 * Monotonic generation counter; increments on every mutation.
 *
 * Use this for cheap change detection on consumer side without
 * snapshotting the full atom set.
 */

uint64_t molex_assembly_generation(const molex_Assembly *assembly)
;

/**
 * Number of entities in the assembly. Returns 0 if `assembly` is null.
 */

uintptr_t molex_assembly_num_entities(const molex_Assembly *assembly)
;

/**
 * Borrow a non-owning view of the i-th entity. Returns null when
 * `assembly` is null or `i` is out of bounds.
 */

const molex_Entity *molex_assembly_entity(const molex_Assembly *assembly,
                                          uintptr_t i)
;

/**
 * Raw entity id (`u32`). Returns 0 if `entity` is null.
 */

uint32_t molex_entity_id(const molex_Entity *entity)
;

/**
 * Molecule type discriminant. Returns [`molex_MoleculeType::Solvent`]'s
 * integer value (8) as a placeholder if `entity` is null - callers
 * should null-check the handle first.
 */

molex_MoleculeType molex_entity_molecule_type(const molex_Entity *entity)
;

/**
 * Structural discriminant (the four-way variant taxonomy). Returns
 * [`molex_EntityKind::Bulk`]'s integer value (3) as a placeholder if
 * `entity` is null - callers should null-check the handle first.
 */

molex_EntityKind molex_entity_kind(const molex_Entity *entity)
;

/**
 * First PDB chain-identifier byte for an entity (polymer or non-polymer).
 *
 * Returns -1 when the entity has no chain id, when `entity` is null, or
 * when the chain id is multi-character and so cannot be represented as a
 * single byte.
 *
 * Chain ids are arbitrary strings (mmCIF `label_asym_id`); ribosome and
 * capsid assemblies use multi-character ids ("AA", "AB"). Use
 * [`molex_entity_chain_id`] for the full string. This single-byte
 * accessor is retained for ABI stability and returns the first byte only
 * when the id is exactly one byte.
 */

int32_t molex_entity_pdb_chain_id(const molex_Entity *entity)
;

/**
 * Pointer to this entity's full PDB chain-identifier UTF-8 bytes.
 *
 * The chain id is the mmCIF `label_asym_id`. Writes the byte length to
 * `out_len` on success. Returns null and writes 0 for an entity with no
 * chain id or a null `entity`.
 *
 * The pointer borrows the entity's owned chain string and is valid for
 * the entity's lifetime; the buffer is not NUL-terminated, so callers
 * must use `out_len`.
 */

const uint8_t *molex_entity_chain_id(const molex_Entity *entity,
                                     uintptr_t *out_len)
;

/**
 * Total atom count in this entity. Returns 0 if `entity` is null.
 */

uintptr_t molex_entity_num_atoms(const molex_Entity *entity)
;

/**
 * A by-value handle locating the i-th atom in this entity's flat atom
 * list. Returns the null-`entity` sentinel when `entity` is null or `i`
 * is out of bounds.
 */

molex_Atom molex_entity_atom(const molex_Entity *entity,
                             uintptr_t i)
;

/**
 * Pointer to the single 3-byte residue name carried by a non-polymer
 * entity (`SmallMolecule` / `Bulk`). Writes 3 to `out_len` on success;
 * returns null and writes 0 for polymers or a null `entity`.
 *
 * The buffer is space-padded to 3 bytes; callers should strip trailing
 * ASCII spaces if needed.
 */

const uint8_t *molex_entity_residue_name_single(const molex_Entity *entity,
                                                uintptr_t *out_len)
;

/**
 * Number of equal-sized molecule chunks the atom set should be split into.
 *
 * For non-polymer entities only: returns 1 for `SmallMolecule`,
 * `BulkEntity::molecule_count` for `Bulk`, and 0 for polymers or a null
 * `entity`.
 */

uintptr_t molex_entity_molecule_count(const molex_Entity *entity)
;

/**
 * Number of indexable residues in this entity.
 *
 * Returns the residue count for protein and nucleic acid entities; 0 for
 * small-molecule and bulk entities (which do not expose individual
 * residue records). Returns 0 if `entity` is null.
 */

uintptr_t molex_entity_num_residues(const molex_Entity *entity)
;

/**
 * Borrow a non-owning view of the i-th residue in this entity.
 *
 * Returns null when `entity` is null, the entity is not a polymer, or
 * `i` is out of bounds.
 */

const molex_Residue *molex_entity_residue(const molex_Entity *entity,
                                          uintptr_t i)
;

/**
 * Number of atoms in the i-th residue of this entity. Returns 0 if
 * `entity` is null, the entity has no residue records, or `i` is out
 * of bounds.
 */

uintptr_t molex_entity_residue_num_atoms(const molex_Entity *entity,
                                         uintptr_t i)
;

/**
 * A by-value handle locating the j-th atom in the i-th residue of this
 * entity. Returns the null-`entity` sentinel on any out-of-bounds access
 * or null `entity` argument.
 */

molex_Atom molex_entity_residue_atom(const molex_Entity *entity,
                                     uintptr_t residue_idx,
                                     uintptr_t atom_idx)
;

/**
 * Pointer to this residue's 3-byte name (e.g. b"ALA"). Writes 3 to
 * `out_len` on success. Returns null and writes 0 if `residue` is null.
 *
 * The buffer is space-padded to 3 bytes; callers that want a trimmed
 * name should strip trailing ASCII spaces.
 */

const uint8_t *molex_residue_name(const molex_Residue *residue,
                                  uintptr_t *out_len)
;

/**
 * Author-side sequence id, falling back to the structural-side
 * (`label_seq_id`) when the author id is absent.
 */

int32_t molex_residue_seq_id(const molex_Residue *residue)
;

/**
 * Structural-side sequence id (`label_seq_id`). Use this for stable
 * internal ordering rather than display.
 */

int32_t molex_residue_label_seq_id(const molex_Residue *residue)
;

/**
 * PDB insertion code byte (`iCode` / `pdbx_PDB_ins_code`). Returns 0
 * when the residue has no insertion code or `residue` is null.
 */

uint8_t molex_residue_ins_code(const molex_Residue *residue)
;

/**
 * Pointer to this atom's 4-byte PDB-style name (e.g. b"CA  "). Writes
 * 4 to `out_len` on success. Returns null and writes 0 if the handle is
 * invalid.
 *
 * The buffer is space-padded; callers that want a trimmed atom name
 * should strip ASCII spaces.
 */

const uint8_t *molex_atom_name(molex_Atom atom,
                               uintptr_t *out_len)
;

/**
 * Atomic number for this atom's element, or 0 for
 * [`Element::Unknown`] / an invalid handle.
 */

uint8_t molex_atom_atomic_number(molex_Atom atom)
;

/**
 * Write this atom's `(x, y, z)` position into the 3-float output array.
 * No-op if `out_xyz` is null or the handle is invalid.
 */

void molex_atom_position(molex_Atom atom,
                         float *out_xyz)
;

/**
 * Crystallographic occupancy (0.0 to 1.0). Returns 0 if the handle is
 * invalid.
 */

float molex_atom_occupancy(molex_Atom atom)
;

/**
 * Temperature factor (B-factor) in square angstroms. Returns 0 if the
 * handle is invalid.
 */

float molex_atom_b_factor(molex_Atom atom)
;

/**
 * Signed formal charge (0 means neutral). Returns 0 if the handle is
 * invalid.
 */

int8_t molex_atom_formal_charge(molex_Atom atom)
;

/**
 * Free a float buffer previously returned via `out_data` / `out_b`.
 *
 * Safe to call with a null pointer (no-op). `len` must match the element
 * count the producing call reported (grid `nu*nv*nw` for density, atom count
 * for refined B-factors).
 */

void molex_free_floats(float *ptr,
                       uintptr_t len)
;

/**
 * Parse an sf.cif string into resident experimental data, resolving the space
 * group from the file's own symmetry tag.
 *
 * Returns null on failure (missing symmetry/cell/reflections or unsupported
 * space group) with the message available via `molex_last_error_message`. The
 * caller owns the returned handle and must free it with
 * [`molex_experimental_data_free`].
 */

molex_ExperimentalData *molex_experimental_data_from_sf_cif(const char *sf_ptr,
                                                            uintptr_t len,
                                                            double free_fraction,
                                                            uint64_t seed)
;

/**
 * Parse an sf.cif string with an externally supplied International Tables
 * space-group number, for the common case where the sf.cif omits symmetry.
 *
 * Returns null on failure (missing cell/reflections or unsupported space
 * group) with the message available via `molex_last_error_message`. The caller
 * owns the returned handle and must free it with
 * [`molex_experimental_data_free`].
 */

molex_ExperimentalData *molex_experimental_data_from_sf_cif_with_spacegroup(const char *sf_ptr,
                                                                            uintptr_t len,
                                                                            uint16_t sg_number,
                                                                            double free_fraction,
                                                                            uint64_t seed)
;

/**
 * Free a resident experimental-data handle.
 *
 * Safe to call with a null pointer (no-op).
 */

void molex_experimental_data_free(molex_ExperimentalData *data)
;

/**
 * Minimum Bragg spacing (Å) over the reflection set. Returns 0 for a null
 * handle.
 */

double molex_experimental_data_d_min(const molex_ExperimentalData *data)
;

/**
 * The space-group International Tables number. Returns 0 for a null handle.
 */

int32_t molex_experimental_data_space_group_number(const molex_ExperimentalData *data)
;

/**
 * Number of reflections in the resident set. Returns 0 for a null handle.
 */

uintptr_t molex_experimental_data_num_reflections(const molex_ExperimentalData *data)
;

/**
 * Write the density-map grid dimensions `(nu, nv, nw)` into the 3-element
 * output array. No-op if `out_dims` is null or the handle is invalid.
 */

void molex_experimental_data_grid_dims(const molex_ExperimentalData *data,
                                       uintptr_t *out_dims)
;

/**
 * Compute the 2mFo-DFc electron density map from the assembly's atoms and the
 * resident data.
 *
 * On success returns [`MOLEX_OK`], writes the grid dimensions `(nu, nv, nw)`
 * into `out_dims`, and writes the heap-allocated row-major `(u, v, w)` float32
 * grid pointer into `out_data`; the caller frees it with
 * [`molex_free_floats`] passing `nu*nv*nw` as the length. Hydrogens are
 * excluded from the splat.
 *
 * Returns [`MOLEX_ERR_NULL`] on a null handle or output pointer, or
 * [`MOLEX_ERR`] if density computation fails (unsupported space group or empty
 * data); the message is available via `molex_last_error_message`.
 */

int32_t molex_experimental_data_compute_density(const molex_ExperimentalData *data,
                                                const molex_Assembly *assembly,
                                                float **out_data,
                                                uintptr_t *out_dims)
;

/**
 * Refine per-atom isotropic B-factors against the resident data.
 *
 * On success returns [`MOLEX_OK`], writes the full-length refined B-factor
 * buffer pointer + count into `out_b` / `out_n`, and writes the R-factors into
 * `out_r_work` / `out_r_free`. The buffer is aligned 1:1 to the
 * `AtomTable::from_entities` flat atom order and is freed with
 * [`molex_free_floats`] passing `out_n` as the length.
 *
 * Hydrogens are excluded from the refinement; hydrogen atoms keep their input
 * B-factor in the returned array while non-hydrogen atoms carry the refined
 * value.
 *
 * Returns [`MOLEX_ERR_NULL`] on a null handle or output pointer, or
 * [`MOLEX_ERR`] if the refinement pipeline fails; the message is available via
 * `molex_last_error_message`.
 */

int32_t molex_experimental_data_refine_b_factors(const molex_ExperimentalData *data,
                                                 const molex_Assembly *assembly,
                                                 uintptr_t n_macro_cycles,
                                                 float **out_b,
                                                 uintptr_t *out_n,
                                                 double *out_r_work,
                                                 double *out_r_free)
;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* MOLEX_H */
