# molex

**molex** (molecular exchange) is a Rust library for parsing, analyzing, transforming, and serializing molecular structure data. It supports PDB, mmCIF, BinaryCIF, MRC/CCP4 density maps, and DCD trajectories.

## Key concepts

- **`MoleculeEntity`** represents a single molecule: a protein chain, a DNA/RNA strand, a ligand, an ion, or a group of waters. Parsing a structure file produces a `Vec<MoleculeEntity>`.

- **`Assembly`** is the top-level host container: it owns the entities, opt-in per-entity DSSP secondary structure (empty until `recompute_ss()`), and owner-set rendering connections, with a generation counter that increments on every mutation. Disulfide and backbone-H-bond geometry is computed on demand, not stored.

- **`Atom`** holds a position, element, atom name, occupancy, and B-factor. Residue and chain context live on the entity that contains the atom.

- **`AtomId`** and **`CovalentBond`** are the cross-cutting identifiers: bonds reference atoms by `AtomId { entity, index }` so they remain addressable across reorderings.

- The **assembly wire format** is a compact binary serialization format for FFI and IPC that round-trips per-entity molecule-type metadata alongside coordinates.

- **Analysis** includes covalent bond inference, DSSP hydrogen bond detection, disulfide bridges, secondary structure classification, AABBs, and volumetric/SES utilities.

- **`VoxelGrid`** and **`Density`** represent 3D volumetric data (electron density, cryo-EM maps).

- Beyond the pure-Rust core, molex ships a feature-gated crystallographic-refinement subsystem (`xtal`) and C / Python bindings for cross-language consumers.

## Crate features

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

## API documentation

For the full Rust API reference, run:

```bash
cargo doc --open --document-private-items
```
