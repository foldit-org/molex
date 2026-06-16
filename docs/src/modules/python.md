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

## AtomWorks interop (`adapters::atomworks`)

Entity-aware conversions between molex's assembly binary format and Biotite `AtomArray` objects with AtomWorks annotations (`entity_id`, `mol_type`, `pn_unit_iid`, `chain_type`).

### molex -> AtomWorks

```python
import molex

# Convert assembly bytes to AtomArray (basic annotations)
atom_array = molex.entities_to_atom_array(assembly_bytes)

# Convert with full bond list and chain type annotations
atom_array = molex.entities_to_atom_array_plus(assembly_bytes)

# Convert with AtomWorks cleaning pipeline (leaving group removal,
# charge correction, missing atom imputation)
atom_array = molex.entities_to_atom_array_parsed(assembly_bytes, "3nez.cif.gz")

# Decode assembly bytes directly into an AtomArray
atom_array = molex.assembly_bytes_to_atom_array(assembly_bytes)
atom_array = molex.assembly_bytes_to_atom_array_plus(assembly_bytes)
```

### AtomWorks -> molex

```python
# Convert AtomArray back to assembly bytes (preserves entity metadata)
assembly_bytes = molex.atom_array_to_entities(atom_array)
```

### File-based shortcuts

```python
# Parse structure file directly to assembly bytes via AtomWorks
assembly_bytes = molex.parse_file_to_entities("3nez.cif.gz")

# Parse to AtomArray with full AtomWorks pipeline
atom_array = molex.parse_file_full("3nez.cif.gz")
```
