# Data Flow

## Overview

```text
                       ┌──────────────┐
 PDB / mmCIF / BCIF ──>│              ├──> Vec<MoleculeEntity>
 MRC / CCP4         ──>│   Adapters   ├──> Density (wraps VoxelGrid)
 DCD                ──>│              ├──> Vec<DcdFrame>
                       └──────┬───────┘
                              │
                              v
                  ┌───────────────────────┐
                  │       Assembly        │
                  │  (entities + derived: │
                  │   ss, hbonds, S-S)    │
                  └──┬────────┬────────┬──┘
                     │        │        │
                     v        v        v
              ┌──────────┐ ┌──────────┐ ┌───────────┐
              │ Analysis │ │Transform │ │   Wire    │
              │          │ │          │ │           │
              │ dssp     │ │ kabsch   │ │ Assembly  │
              │ bonds    │ │ align    │ │ serialize │
              │ disulfide│ │ extract  │ │    /      │
              │ sasa     │ │   ca     │ │deserialize│
              └──────────┘ └──────────┘ └─────┬─────┘
                                              │
                                              v
                                     FFI / IPC / Python
```

Analysis, Transform, and Wire are independent; use any combination depending on what you need.

## 1. Parsing

Every structure adapter returns `Vec<MoleculeEntity>`:

```rust,ignore
let entities = pdb_file_to_entities(Path::new("1ubq.pdb"))?;
let entities = mmcif_file_to_entities(Path::new("3nez.cif"))?;
// BinaryCIF parses from bytes (no path-taking entry point):
let bytes = std::fs::read("1ubq.bcif")?;
let entities = Assembly::from_bcif(&bytes)?.into_entities();
```

Density and trajectory adapters return their own types:

```rust,ignore
let density = mrc_file_to_density(Path::new("emd_1234.map"))?;
let frames = dcd_file_to_frames(Path::new("trajectory.dcd"))?;
```

## 2. Entity construction

Each parser tokenizes its input and pushes one `AtomRow` per atom into an
`EntityBuilder`. `EntityBuilder` is the single point of classification: it
groups rows by chain plus residue scope, joins mmCIF `_entity` /
`_entity_poly` hints when present, and emits one `MoleculeEntity` per
logical molecule (protein chain, NA chain, ligand instance, water bulk,
solvent bulk) at `finish()`. Each emitted entity carries a freshly
allocated `EntityId`.

For NMR ensembles or multi-model trajectories, the adapter-level
`*_to_all_models` entry points (`pdb_str_to_all_models`,
`mmcif_str_to_all_models`, `bcif_to_all_models`, plus the matching
`_file_*` variants) return one `Vec<MoleculeEntity>` per MODEL.

## 3. Analysis

```rust,ignore
use molex::{detect_disulfides, Assembly};
use molex::analysis::{infer_bonds, DEFAULT_TOLERANCE};

// Top-level pipeline: build an Assembly to get DSSP secondary
// structure eagerly computed and kept in sync.
let assembly = Assembly::new(entities);
let ss     = assembly.ss_types(entity_id);

// Disulfide + backbone-H-bond geometry is computed on demand for
// rendering, not stored on the Assembly:
let connections = assembly.detect_fallback_connections();

// Or call the building blocks directly:
let bonds      = infer_bonds(atoms, DEFAULT_TOLERANCE);   // distance-based
let disulfides = detect_disulfides(&entities);             // CYS SG-SG
```

## 4. Transforms

```rust,ignore
let (rotation, translation) = kabsch_alignment(&reference_ca, &target_ca)?;
transform_entities(&mut entities, rotation, translation);
let ca_positions = extract_ca_positions(&entities);
```

## 5. Serialization

For sending to C/C++/Python consumers:

```rust,ignore
use molex::ops::wire::{assembly_bytes, serialize_assembly};

// Serialize a raw entity slice
let bytes = assembly_bytes(&entities)?;

// Or serialize a fully-built Assembly (derived data is recomputed on
// the deserialize side via Assembly::new)
let bytes = serialize_assembly(&assembly)?;
```
