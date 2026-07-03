# molex

**Mol**ecular **ex**change: a Rust library for parsing, analyzing,
and serializing molecular structure data.

## Features

- **Parse** PDB, mmCIF, BinaryCIF, MRC/CCP4 density maps, and DCD trajectories
- **Entity model**: proteins, nucleic acids, ligands, ions, waters, and other small molecules as typed entities
- **Analyze**: DSSP secondary structure, hydrogen bonds, covalent bonds, disulfide bridges, SASA, contacts
- **Transform**: Kabsch alignment and superposition RMSD
- **Serialize**: compact binary wire format for FFI and IPC
- **Python bindings**: PyO3 module with AtomWorks/Biotite interop

## Quick start

```rust
use molex::adapters::pdb::pdb_file_to_entities;

let entities = pdb_file_to_entities("1ubq.pdb".as_ref())?;
for e in &entities {
    println!("{}: {} atoms", e.label(), e.atom_count());
}
```

## Python

```bash
pip install molex
```

The Python API is object-centric: parse a structure into an `Assembly`, walk
its entities and residues, and read atoms as numpy columns.

```python
import molex

asm = molex.Assembly.from_pdb(open("1ubq.pdb").read())
for e in asm.entities():
    print(e.kind, e.chain_id, e.residue_count)

asm.recompute_ss()       # opt-in DSSP secondary structure

# Per-atom numpy columns (no Biotite required) come from PyAtomTable:
table = molex.PyAtomTable.from_assembly_bytes(asm.to_assembly_bytes())
coords = table.coords    # (N, 3) float32
```

Atom completion (filling missing heavy atoms) happens at parse time, during
file ingest; the default is `Completion.Heavy`. `from_mmcif` and `from_bcif`
parse the other formats. The `*_to_assembly_bytes` / `assembly_bytes_to_*`
free functions are transport helpers for the wire protocol, not the default
path. Type stubs (`molex.pyi`) ship with the wheel.

## Optional features

| Feature  | Description |
|----------|-------------|
| `python` | PyO3 bindings for use from Python |

## Documentation

- [**Guide**](https://petridecus.github.io/molex/): architecture, modules, and examples
- [**API reference**](https://petridecus.github.io/molex/api/molex/): generated rustdoc

## License

MIT
