# molex

**Mol**ecular **ex**change — a Rust library for parsing, analyzing,
and serializing molecular structure data.

## Features

- **Parse** PDB, mmCIF, BinaryCIF, MRC/CCP4 density maps, and DCD trajectories
- **Entity model** — proteins, nucleic acids, ligands, ions, waters, and cofactors as typed entities
- **Analyze** — DSSP secondary structure, hydrogen bonds, covalent bonds, disulfide bridges
- **Transform** — Kabsch alignment, CA extraction, backbone segments
- **Serialize** — compact binary format (ASSEM02) for FFI and IPC
- **Python bindings** — PyO3 module with AtomWorks/Biotite interop

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

```python
import molex

assembly_bytes = molex.pdb_to_assembly_bytes(open("1ubq.pdb").read())
pdb_string = molex.assembly_bytes_to_pdb(assembly_bytes)
```

## Optional features

| Feature  | Description |
|----------|-------------|
| `python` | PyO3 bindings for use from Python |

## Documentation

- [**Guide**](https://petridecus.github.io/molex/) — architecture, modules, and examples
- [**API reference**](https://petridecus.github.io/molex/api/molex/) — generated rustdoc

## License

MIT
