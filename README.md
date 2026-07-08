# molex

**Mol**ecular **ex**change: a Rust library for parsing, transforming, analyzing,
and serializing molecular structure data, with Python and C bindings.

## Features

- **Parse** PDB, mmCIF, BinaryCIF, MRC/CCP4 density maps, and DCD trajectories
- **Entity model**: proteins, nucleic acids, ligands, ions, waters, and other
  small molecules as typed entities under a single `Assembly`
- **Analyze**: DSSP secondary structure, hydrogen bonds, covalent bonds,
  disulfide bridges, SASA, contacts
- **Transform**: Kabsch alignment and superposition RMSD, heavy-atom completion,
  all-atom projection, structured edits and deltas
- **Crystallography** (`xtal`): maximum-likelihood refinement from structure
  factors: density synthesis, bulk-solvent masking, anisotropic scaling,
  sigma-A estimation, and B-factor refinement, with an optional GPU backend
- **Serialize**: a compact binary wire format for FFI and IPC
- **Bindings**: a PyO3 Python module (Biotite-free numpy interchange) and a
  C ABI static library (`libmolex.a`) for embedding in native hosts

## Quick start (Rust)

```rust
use std::path::Path;
use molex::Assembly;

let assembly = Assembly::from_file(Path::new("1ubq.pdb"))?;
for e in assembly.entities() {
    println!("{:?}: {} atoms", e.molecule_type(), e.atom_count());
}
```

`Assembly::from_file` reads PDB or mmCIF by extension; `from_pdb` / `from_mmcif`
/ `from_bcif` take the source as a string, each with a `_with(..., Completion)`
variant. Heavy-atom completion runs at parse time (default `Completion::Heavy`).
Reach atoms through `assembly.entities()`, and write back with
`assembly.to_pdb()`.

## Python

```bash
pip install molex
```

The Python API is object-centric: parse into an `Assembly`, walk its entities
and residues, and read atoms as numpy columns.

```python
import molex

asm = molex.Assembly.from_pdb(open("1ubq.pdb").read())
for e in asm.entities():
    print(e.kind, e.chain_id, e.residue_count)

asm.recompute_ss()   # opt-in DSSP secondary structure

# Biotite-free per-atom numpy columns via PyAtomTable:
table = molex.PyAtomTable.from_assembly_bytes(asm.to_assembly_bytes())
coords = table.coords          # (N, 3) float32
mol_types = table.mol_types    # AtomWorks-style vocabulary columns
```

`from_mmcif` / `from_bcif` parse the other formats. `PyAtomTable` exposes plain
numpy columns (coords, atom/residue names, elements, b-factors, occupancies,
chain/residue ids, and AtomWorks-style vocabulary) with **no Biotite
dependency**. Type stubs (`molex.pyi`) ship with the wheel. Crystallographic
refinement is available through `molex.PyExperimentalData` (`from_sf_cif`,
`compute_density`, `refine_b_factors`).

## C API

With the `c-api` feature, molex builds a static library (`libmolex.a`) plus a
cbindgen-generated header (`include/molex.h`) exposing an opaque-handle C ABI:
parse PDB/mmCIF/BinaryCIF into an `Assembly`, walk entities/residues/atoms,
apply edits, and (with `xtal`) drive crystallographic refinement. This is the
interface native hosts embed molex through.

## Optional features

| Feature | Enables |
| --- | --- |
| *(default)* | Pure-Rust core: parsing, entity model, analysis, wire format |
| `serde` | `Serialize` / `Deserialize` on the core types |
| `specta` | TypeScript type-export derives |
| `python` | PyO3 bindings + numpy interchange |
| `extension-module` | Build the Python extension as a loadable wheel (with maturin) |
| `c-api` | C ABI, `libmolex.a`, and the generated `include/molex.h` |
| `xtal` | Crystallographic refinement pipeline (density, scaling, sigma-A, FFT) |
| `minimization` | B-factor refinement (argmin), on top of `xtal` |
| `gpu` | GPU density/refinement backend (cubecl + wgpu), on top of `xtal` |
| `testutil` | Crystallographic test fixtures for external bench/integration crates |

## Documentation

- [**Guide**](https://foldit-org.github.io/molex/): architecture, modules, and examples
- [**API reference**](https://foldit-org.github.io/molex/api/molex/): generated rustdoc

## License

MIT
