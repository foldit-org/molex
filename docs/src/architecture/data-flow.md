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
                  │  (entities + opt-in   │
                  │   ss + connections)   │
                  └──┬────────┬────────┬──┘
                     │        │        │
                     v        v        v
              ┌──────────┐ ┌──────────┐ ┌───────────┐
              │ Analysis │ │Transform │ │   Wire    │
              │          │ │          │ │           │
              │ dssp     │ │ kabsch   │ │ assembly  │
              │ bonds    │ │ align    │ │  bytes    │
              │ disulfide│ │ rmsd     │ │    +      │
              │ sasa     │ │          │ │  delta    │
              └──────────┘ └──────────┘ └─────┬─────┘
                                              │
                                              v
                                     FFI / IPC / Python
```

Analysis, Transform, and Wire are independent; use any combination depending on what you need.

## 1. Parsing

Every structure adapter returns `Vec<MoleculeEntity>`:

```rust,ignore
let entities = Assembly::from_file(Path::new("1ubq.pdb"))?.into_entities();
let entities = Assembly::from_file(Path::new("3nez.cif"))?.into_entities(); // mmCIF by extension
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

// Secondary structure is opt-in: build, then recompute.
let mut assembly = Assembly::new(entities);
assembly.recompute_ss();
let ss = assembly.ss_types(entity_id);

// Disulfide + backbone-H-bond geometry is computed on demand for
// rendering, not stored on the Assembly:
let connections = assembly.detect_fallback_connections();

// Or call the building blocks directly:
let bonds      = infer_bonds(atoms, DEFAULT_TOLERANCE);   // distance-based
let disulfides = detect_disulfides(&entities);             // CYS SG-SG
```

## 4. Transforms

`molex::ops` exposes Kabsch alignment and superposition RMSD; nothing else
lives in the transform surface.

```rust,ignore
use molex::ops::kabsch_alignment;
use molex::ops::transform::rmsd;

let aligned = kabsch_alignment(&reference, &target); // Option<(Mat3, Vec3)>
let deviation = rmsd(&a, &b);                         // Option<f32>
```

## 5. Serialization

For sending to C/C++/Python consumers, `assembly_bytes` is the only public
wire entry point (it serializes a raw entity slice). Decode with
`Assembly::from_bytes`, which returns a secondary-structure-free assembly;
call `recompute_ss()` if you need SS.

```rust,ignore
use molex::ops::wire::assembly_bytes;

let bytes = assembly_bytes(&entities)?;
let mut assembly = molex::Assembly::from_bytes(&bytes)?; // ss_types empty
assembly.recompute_ss();
```
