# Architecture Overview

## Module layout

```text
molex/src/
├── adapters/         File format parsers (PDB, mmCIF, BinaryCIF, MRC, DCD, AtomWorks)
├── analysis/         Structural analysis (bonds, secondary structure, AABB, volumetric)
├── chemistry/        Static residue tables (amino acids, nucleotides, atom names)
├── entity/           Entity system
│   ├── molecule/     MoleculeEntity enum + subtypes (protein, nucleic acid, small molecule, bulk)
│   └── surface/      Surface types (VoxelGrid, Density)
├── ops/              Operations
│   ├── codec/        AdapterError, ca_positions helper
│   ├── transform/    Kabsch alignment, CA extraction, backbone segments
│   └── wire/         assembly binary wire format encoder/decoder
├── assembly.rs       Top-level Assembly container with eagerly-computed derived data
├── atom_id.rs        Cross-cutting AtomId (entity + index)
├── bond.rs           Cross-cutting CovalentBond (AtomId endpoints)
├── element.rs        Element enum (symbols, covalent radii, colors)
├── python.rs         PyO3 bindings (feature = "python")
└── lib.rs            Crate root, re-exports
```

## Entity-first design

Entities (`Vec<MoleculeEntity>`) are the primary data model.

**Adapters** parse files into entities. The `*_to_entities` functions are the primary API; the `*_to_all_models` variants return one entity list per MODEL block for NMR ensembles or multi-state trajectories.

**Analysis** operates on `&[Atom]`, `&[ResidueBackbone]`, or `&[MoleculeEntity]`.

## Assembly

`Assembly` (in `src/assembly.rs`) is the host-owned structural source of truth. It bundles:

- a `Vec<MoleculeEntity>`
- cross-entity `CovalentBond`s (currently disulfides)
- per-entity `SSType` arrays (DSSP)
- backbone `HBond`s
- a generation counter that bumps on every mutation

Any `&mut Assembly` mutation (add/remove entity, update positions, restore from `CoordinateSnapshot`) recomputes all derived data before returning, so any snapshot a reader holds is internally consistent.

## Entity classification

When a file is parsed, atoms are grouped into entities by chain ID, residue name, and molecule type. The `classify_residue` function maps 3-letter residue codes to `MoleculeType` values using lookup tables:

| MoleculeType | Examples |
|---|---|
| `Protein` | Standard amino acids (ALA, GLY, ...) |
| `DNA` | DA, DT, DC, DG |
| `RNA` | A, U, C, G |
| `Ligand` | ATP, HEM, NAG, ... |
| `Ion` | ZN, MG, CA, FE, ... |
| `Water` | HOH, WAT, DOD |
| `Lipid` | OLC, PLM, ... |
| `Cofactor` | HEM, NAD, FAD, ... |
| `Solvent` | GOL, EDO, PEG, ... |

## Type hierarchy

```text
MoleculeEntity (enum)
├── Protein(ProteinEntity)      -- chain with residues, segment breaks
├── NucleicAcid(NAEntity)       -- DNA/RNA chain with residues
├── SmallMolecule(SmallMoleculeEntity) -- single ligand, ion, cofactor, lipid
└── Bulk(BulkEntity)            -- grouped water or solvent molecules

Entity trait   -- id(), molecule_type(), atoms(), positions(), atom_count()
Polymer trait  -- residues(), segment_breaks(), segment_count(), segment_range()
```

`ProteinEntity` and `NAEntity` implement both `Entity` and `Polymer`. `SmallMoleculeEntity` and `BulkEntity` implement `Entity` only.

## Surface types

The `entity::surface` module provides volumetric data types:

- **`VoxelGrid`** -- a generic 3D grid (`ndarray::Array3<f32>`) with crystallographic cell metadata. Handles fractional-to-Cartesian coordinate conversion.
- **`Density`** -- wraps `VoxelGrid` with density-specific metadata. Constructed by the MRC adapter.

## Binary format

The **assembly format** is molex's compact binary IPC format: a fixed 8-byte magic (`ASSEMBLY`) followed by a version byte that selects the payload layout (only version 1 exists today). It is entity-aware: each entity is preceded by a 9-byte header (molecule-type byte, atom count, and `EntityId`) so the decoder reconstructs `MoleculeEntity` variants without re-running residue classification, and a trailing variants section carries per-residue variant tags. See the [Wire Format](../modules/wire.md) page for the full byte layout.
