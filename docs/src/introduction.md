# molex

**molex** (molecular exchange) is a Rust library for parsing, analyzing, transforming, and serializing molecular structure data. It supports PDB, mmCIF, BinaryCIF, MRC/CCP4 density maps, and DCD trajectories.

## Key concepts

- **`MoleculeEntity`** represents a single molecule: a protein chain, a DNA/RNA strand, a ligand, an ion, or a group of waters. Parsing a structure file produces a `Vec<MoleculeEntity>`.

- **`Assembly`** is the top-level host container: it owns the entities, cross-entity bonds (disulfides), per-entity DSSP secondary structure, and backbone H-bonds, with a generation counter that increments on every mutation.

- **`Atom`** holds a position, element, atom name, occupancy, and B-factor. Residue and chain context live on the entity that contains the atom.

- **`AtomId`** and **`CovalentBond`** are the cross-cutting identifiers: bonds reference atoms by `AtomId { entity, index }` so they remain addressable across reorderings.

- The **assembly wire format** is a compact binary serialization format for FFI and IPC that round-trips per-entity molecule-type metadata alongside coordinates.

- **Analysis** includes covalent bond inference, DSSP hydrogen bond detection, disulfide bridges, secondary structure classification, AABBs, and volumetric/SES utilities.

- **`VoxelGrid`** and **`Density`** represent 3D volumetric data (electron density, cryo-EM maps).

## Crate features

| Feature  | Description |
|----------|-------------|
| `default` | Core Rust library (no Python) |
| `python` | PyO3 bindings + AtomWorks interop |

## API documentation

For the full Rust API reference, run:

```bash
cargo doc --open --document-private-items
```
