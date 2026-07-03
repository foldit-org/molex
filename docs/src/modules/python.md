# Python Bindings

Python bindings are available when molex is built with the `python` feature. The module is exposed as `import molex` via PyO3.

## Installation

```bash
cd crates/molex
maturin develop --release --features python
```

The bindings are a directory module (`src/python/`), not a single file.

## Object graph (`Assembly` / `Entity` / `Residue`)

The documented default for a normal caller is the object API: parse a
structure, then walk entities and residues.

```python
import molex

asm = molex.Assembly.from_pdb(pdb_string)          # default Completion.Heavy
asm = molex.Assembly.from_mmcif(cif_string)
asm = molex.Assembly.from_bcif(bcif_bytes)

for entity in asm:                                  # __len__ / __getitem__
    print(entity.id, entity.kind, entity.chain_id, entity.residue_count)
    for residue in entity.residues():
        print(residue.name, residue.seq_id, residue.ins_code)
```

`from_pdb` / `from_mmcif` / `from_bcif` take an optional `completion`
argument (`Completion.Raw` / `Heavy` / `AllAtom`, default `Heavy`). `Entity`
exposes `id`, `kind`, `molecule_type`, `chain_id`, `label`, `atom_count`,
`residue_count`, `residues()`, and `phi_psi()` (per-residue backbone torsions
in degrees). `Residue` exposes `name`, `seq_id`, `label_seq_id`, and
`ins_code`.

### Completion and analysis

```python
heavy = asm.normalize()         # Completion.Heavy projection
all_atom = asm.to_all_atom()    # Completion.AllAtom projection
custom = asm.complete(molex.Completion.Raw)

asm.recompute_ss()              # opt-in DSSP (ss starts empty)
ss = asm.secondary_structure()  # one 'H'/'E'/'C' string per entity

area = asm.sasa()               # total protein SASA (Angstrom^2)
bonds = asm.covalent_bonds()    # [(i, j), ...] intra-entity, flat atom indices
ss_bonds = asm.disulfides()     # [(i, j), ...] Cys SG-SG pairs

cif = asm.to_mmcif()            # mmCIF string (round-trips via from_mmcif)
```

The `complete` projections re-index atoms, so externally held atom indices do
not carry across them. `rmsd(a, b)` is a free function: optimal-superposition
RMSD between two `(N, 3)` coordinate sets.

### Edits and deltas

`EditList` builds a typed edit batch (parallel to the C edit surface);
`Assembly.apply_edits` applies it, and the delta bytes round-trip the
coordinate/residue/variant edits (topology edits are not representable on the
delta wire).

```python
edits = molex.EditList()
edits.push_set_entity_coords(entity_id, new_coords)        # [(x, y, z), ...]
edits.push_set_residue_coords(entity_id, residue_idx, new_coords)
edits.push_set_variants(entity_id, residue_idx, [molex.Variant.protonation_his_delta()])
asm.apply_edits(edits)

delta = edits.to_delta()        # bytes
asm.apply_delta(delta)          # decode + apply in one call
```

## Assembly-bytes IO helpers

These free functions operate on serialized assembly bytes (the entity-aware
binary format); they are transport helpers for the wire protocol, not the
default path.

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
