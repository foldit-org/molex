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

## Components (24)

20 standard amino acids:

    ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE
    LEU LYS MET PHE PRO SER THR TRP TYR VAL

4 DNA bases:

    DA DC DG DT

RNA bases are not vendored yet; RNA uracil has no template.

## Regenerating the const table

`src/chemistry/completion_data.rs` is generated from these files. After
adding or refreshing a CIF, regenerate with:

    cargo run --example gen_completion_table

The generator parses each `_chem_comp_atom` loop (resolving column
positions from the header, honoring double-quoted primed atom names) and
emits a deterministic, fmt-clean const table. Do not hand-edit the
generated file.
