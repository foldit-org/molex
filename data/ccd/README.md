# Vendored PDB Chemical Component Dictionary entries

These `*.cif` files are individual component definitions from the RCSB
PDB Chemical Component Dictionary (CCD). Each carries an ordered
all-atom (heavy + hydrogen) ideal conformation used as the source
geometry for completing residues that are missing atoms.

## Provenance

- Source: RCSB PDB ligand download endpoint.
- URL pattern: `https://files.rcsb.org/ligands/download/<CODE>.cif`
- Fetched: 2026-06-15 (HTTP 200 for every file).

## License

The PDB Chemical Component Dictionary is released under CC0 1.0
(Creative Commons Public Domain Dedication), so these files are freely
redistributable. See <https://creativecommons.org/publicdomain/zero/1.0/>.

## Components (28)

20 standard amino acids:

    ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE
    LEU LYS MET PHE PRO SER THR TRP TYR VAL

4 DNA (deoxyribose) bases:

    DA DC DG DT

4 RNA (ribose) bases:

    A C G U

The RNA components carry the ribose 2'-OH (`O2'`/`HO2'`); the DNA
components are deoxy and do not. Template resolution is strand-aware,
so an RNA chain completes against ribose geometry and a DNA chain
against deoxyribose. The same CC0 provenance applies to all four RNA
files (HTTP 200 on fetch). Note the `C` and `U` ligand CIFs omit the
`pdbx_backbone_atom_flag` / `pdbx_n_terminal_atom_flag` /
`pdbx_c_terminal_atom_flag` columns; the generator treats an absent
role-flag column as unflagged (false), which matches the all-`N` rows
the components that do carry those columns emit.

## Regenerating the const table

`src/chemistry/completion_data.rs` is generated from these files. After
adding or refreshing a CIF, regenerate with:

    cargo run --example gen_completion_table

The generator parses each `_chem_comp_atom` loop (resolving column
positions from the header, honoring double-quoted primed atom names) and
emits a deterministic, fmt-clean const table. Do not hand-edit the
generated file.
