# C API

`molex::c_api` (source: `src/c_api/`) is the C ABI surface, gated behind the
`c-api` cargo feature. It bridges `Assembly` and its read accessors plus
PDB/CIF/BCIF read and PDB/mmCIF write across the Rust/C boundary through
opaque-pointer handles and heap-allocated output buffers. It is the C/C++
counterpart of the Python object graph; the two surfaces parallel each other.

## Naming

Handle types use bare names (`molex_Assembly`, `molex_Entity`,
`molex_Residue`, `molex_Atom`). Free functions keep a `molex_` prefix because
C has no namespace mechanism. C++ consumers wrap the types in
`namespace molex`.

## Ownership

- A `*mut molex_Assembly` returned by a parser entry point is caller-owned and
  must be released with `molex_assembly_free`.
- `*const molex_Entity` / `*const molex_Residue` / `molex_Atom` handles are
  non-owning views into the parent assembly; they must not outlive it.
- Byte buffers returned via `out_buf` / `out_len` are heap-allocated by Rust
  and freed with `molex_free_bytes`.

## Entry points

- **Parsers**: `molex_pdb_str_to_assembly`, `molex_cif_str_to_assembly`,
  `molex_bcif_to_assembly`, `molex_bytes_to_assembly`, plus
  `*_with_completion` variants that take a `molex_Completion` level (the
  no-level entries default to `Heavy`).
- **Writers**: `molex_assembly_to_pdb`, `molex_assembly_to_mmcif`,
  `molex_assembly_to_bytes`.
- **Projections**: `molex_assembly_normalize`, `molex_assembly_to_all_atom`,
  `molex_assembly_complete` return fresh re-completed handles (see
  [Completion](completion.md)).
- **Walk** (`c_api::walk`): entity / residue / atom accessors for reading an
  assembly's structure.
- **Edits** (`c_api::edit`): the handle-based edit and delta surface that the
  Python `EditList` parallels.
- **Crystallographic refinement** (`c_api::xtal`, gated on `c-api` + `xtal`):
  `molex_experimental_data_from_sf_cif` and its `*_with_spacegroup` variant,
  `molex_experimental_data_compute_density`, and
  `molex_experimental_data_refine_b_factors`.

## Error reporting

Fallible entry points return either a null handle (parsers) or a nonzero
status code (writers). On failure a human-readable message is recorded in
thread-local storage and retrieved with `molex_last_error_message`; successful
calls do not clear it, so read it immediately after a failed call.
