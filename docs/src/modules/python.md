# Python Bindings

Python bindings are available when molex is built with the `python` feature. The module is exposed as `import molex` via PyO3.

## Installation

```bash
cd crates/molex
maturin develop --release --features python
```

## Assembly-bytes IO helpers (`python.rs`)

These operate on serialized assembly bytes (entity-aware binary format).

```python
import molex

# Parse PDB string to assembly bytes
assembly_bytes = molex.pdb_to_assembly_bytes(pdb_string)

# Parse mmCIF string to assembly bytes
assembly_bytes = molex.mmcif_to_assembly_bytes(cif_string)

# Convert assembly bytes to PDB string
pdb_string = molex.assembly_bytes_to_pdb(assembly_bytes)

# Round-trip validation
validated = molex.deserialize_assembly_bytes(assembly_bytes)
```

## Biotite-agnostic columnar interchange (`PyAtomTable`)

`PyAtomTable` round-trips atom data through molex as per-atom numpy columns,
with no Biotite import on either side. A plugin builds the Biotite (or any
other) representation from these columns itself.

```python
import molex

# From assembly bytes -> columns
t = molex.PyAtomTable.from_assembly_bytes(assembly_bytes)
coords = t.coords            # (N, 3) float32
atom_names = t.atom_names    # numpy 'S4' byte strings, trailing-space trimmed
res_names = t.res_names      # numpy 'S3' byte strings
res_ids = t.res_ids          # (N,) int32
observed = t.observed_mask   # (N,) bool
bonds = t.bonds()            # [(i, j, order), ...] inferred ligand/ion bonds

# Columns -> molex (after a model edits the coordinates, say)
t2 = molex.PyAtomTable.from_columns(
    coords, atom_names, t.elements, res_ids, res_names, t.chain_ids,
    t.occupancies, t.b_factors, t.mol_types,
    entity_ids=t.entity_ids, observed=observed,
)
assembly_bytes = t2.to_assembly_bytes()
```

`atom_names`/`res_names`/`elements`/`mol_types` accept either `str` or
fixed-width `bytes` (a numpy `'S'` array works). Only `coords`, `atom_names`,
`elements`, `res_ids`, `res_names`, and `chain_ids` are required; absent
`occupancies`/`b_factors` default to `1.0`/`0.0` and absent `observed` to
`True`.

Each atom's molecule type follows a three-tier precedence: `chain_types` (int32
codes) if given, else `mol_types` strings, else classification of the residue
name. So annotation-free input (hand-built backbone frames, concatenated
templates, folding-model outputs) classifies a protein backbone as protein
rather than ligand. When `entity_ids` is omitted, entity ids are synthesized
from each atom's `(chain_id, mol_type)` pair in first-appearance order.
