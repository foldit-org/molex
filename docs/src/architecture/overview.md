# Architecture Overview

## Module layout

```text
molex/src/
├── adapters/         File format parsers (PDB, mmCIF, BinaryCIF, MRC, DCD, AtomWorks)
├── analysis/         Structural analysis (bonds, secondary structure, SASA, contacts, volumetric)
├── chemistry/        Static residue tables + missing-atom completion
├── entity/           Entity system
│   ├── molecule/     MoleculeEntity enum + subtypes (protein, nucleic acid, small molecule, bulk)
│   └── surface/      Surface types (VoxelGrid, Density)
├── ops/              Operations
│   ├── edit/         Typed AssemblyEdit + apply path
│   ├── error/        AdapterError
│   ├── transform/    Kabsch alignment, superposition RMSD
│   └── wire/         Assembly + delta binary wire format encoder/decoder
├── assembly.rs       Top-level Assembly container (entities + opt-in SS + connections)
├── atom_id.rs        Cross-cutting AtomId (entity + index)
├── bond.rs           Cross-cutting CovalentBond (AtomId endpoints)
├── connection.rs     Inter-entity rendering connections (ConnectionType, AtomEnd, AtomLink)
├── element.rs        Element enum (symbols, covalent radii, colors)
├── xtal/             Crystallographic refinement pipeline (feature = "xtal")
├── c_api/            C ABI surface (feature = "c-api")
├── python/           PyO3 bindings (feature = "python")
└── lib.rs            Crate root, re-exports
```

## Entity-first design

Entities (`Vec<MoleculeEntity>`) are the primary data model.

**Adapters** parse files into an `Assembly`. The `Assembly::from_file` / `from_pdb` / `from_mmcif` / `from_bcif` constructors are the primary API; the `*_to_all_models` variants return one entity list per MODEL block for NMR ensembles or multi-state trajectories.

**Analysis** operates on `&[Atom]`, `&[ResidueBackbone]`, or `&[MoleculeEntity]`.

## Assembly

`Assembly` (in `src/assembly.rs`) is the host-owned structural source of truth. It bundles:

- a `Vec<Arc<MoleculeEntity>>`
- per-entity `SSType` arrays (`ss_types`), empty until `recompute_ss()` is called
- owner-set rendering connections keyed by `ConnectionType`
- a generation counter that bumps on every mutation

Construction is secondary-structure-free and mutations only bump the
generation counter; secondary structure is opt-in via `recompute_ss()`, and
disulfide / backbone-H-bond geometry is computed on demand by
`detect_fallback_connections()` rather than stored. See the
[Assembly](../modules/assembly.md) page for the full contract.

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

The feature-gated `xtal` refinement layer sits above these `VoxelGrid` / `Density` primitives: it synthesizes density from atoms, scales against experimental structure factors, and refines B-factors.

## Binary format

The **assembly format** is molex's compact binary IPC format: a fixed 8-byte magic (`ASSEMBLY`) followed by a version byte that selects the payload layout (only version 1 exists today). It is entity-aware: each entity is preceded by a 9-byte header (molecule-type byte, atom count, and `EntityId`) so the decoder reconstructs `MoleculeEntity` variants without re-running residue classification, and a trailing variants section carries per-residue variant tags. See the [Wire Format](../modules/wire.md) page for the full byte layout.
