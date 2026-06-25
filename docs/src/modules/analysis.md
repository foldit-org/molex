# Analysis

Structural analysis lives in `molex::analysis` (source: `src/analysis/`). Most analysis functions operate on entity-level types (`&[Atom]`, `&[ResidueBackbone]`, `&[MoleculeEntity]`).

The eagerly-computed pipeline (DSSP secondary structure + backbone H-bonds + disulfides) is exposed through [`Assembly`](../architecture/overview.md); see `Assembly::ss_types`, `Assembly::hbonds`, and `Assembly::disulfides`. The standalone functions documented below are the building blocks underneath.

## Secondary structure (`analysis::ss`)

DSSP-based secondary structure classification.

```rust,ignore
use molex::analysis::ss::{classify, from_string};
use molex::analysis::{HBond, SSType};

// Classify residues from a precomputed H-bond list.
let ss_types: Vec<SSType> = classify(&hbonds, n_residues);

// Parse a secondary-structure string like "HHHCCCEEE" into Vec<SSType>.
let ss_types = from_string("HHHCCCEEE");
```

`SSType` is a Q3 classification:

```rust,ignore
pub enum SSType { Helix, Sheet, Coil }
```

Each variant has a `.color()` method returning an RGB `[f32; 3]` for rendering.

`analysis::merge_short_segments` converts isolated 1-residue helix/sheet runs to `Coil`.

For most callers the recommended path is to construct an `Assembly` and read `assembly.ss_types(entity_id)`. The assembly internally runs H-bond detection and `classify` for every protein entity.

## Bond detection (`analysis::bonds`)

### Covalent bonds (distance-based)

```rust,ignore
use molex::analysis::{infer_bonds, InferredBond, BondOrder, DEFAULT_TOLERANCE};

let bonds: Vec<InferredBond> = infer_bonds(atoms, DEFAULT_TOLERANCE);
// InferredBond { atom_a: usize, atom_b: usize, order: BondOrder }
// BondOrder: Single, Double, Triple, Aromatic
```

Distance-based inference using element covalent radii with a configurable tolerance. Used for ligands and other non-protein entities where bond topology isn't supplied by a chemistry table. Protein and nucleic-acid bonds are populated from the chemistry tables at entity construction time and live on `ProteinEntity::bonds` / `NAEntity::bonds`.

### Hydrogen bonds

Backbone H-bond detection is an in-crate helper; the function `analysis::bonds::hydrogen::detect_hbonds(&[ResidueBackbone])` is `pub(crate)`. `Assembly` does not store H-bonds; the viewer-facing geometry is computed on demand via `assembly.detect_fallback_connections()`, which returns backbone H-bonds (and disulfides) keyed by `ConnectionType`:

```rust,ignore
use molex::{Assembly, ConnectionType};

let assembly = Assembly::new(entities);
if let Some(hbonds) = assembly.detect_fallback_connections().get(&ConnectionType::HBond) {
    for link in hbonds {
        println!("a={:?} b={:?}", link.a, link.b);
    }
}
```

`HBond` is `{ donor: usize, acceptor: usize, energy: f32 }` (Kabsch-Sander electrostatic energy in kcal/mol).

### Disulfide bonds

```rust,ignore
use molex::{detect_disulfides, CovalentBond};

let disulfides: Vec<CovalentBond> = detect_disulfides(&entities);
```

Scans every protein entity for CYS SG atoms and emits one `CovalentBond` (with `AtomId` endpoints) per SG-SG pair within 1.5 to 2.5 angstroms. `Assembly` does not store disulfides; the same pairs surface for rendering via `assembly.detect_fallback_connections()` under `ConnectionType::Disulfide`.

## Volumetric (`analysis::volumetric`)

Voxel-grid analysis used for surface and cavity work:

```rust,ignore
use molex::analysis::{
    binary_to_sdf, compute_gaussian_field, compute_ses_sdf, detect_cavities,
    detect_cavity_mask, edt_1d, edt_3d, voxelize_sas,
    DetectedCavity, ScalarVoxelGrid, VoxelBbox,
};
```

These power Gaussian density approximations, solvent-excluded surface SDFs, and cavity detection.

## Transforms (`ops::transform`)

Structural alignment and extraction utilities. Re-exported from
`molex::ops`.

```rust,ignore
use molex::ops::{
    align_to_reference, centroid, extract_backbone_segments,
    extract_ca_from_chains, extract_ca_positions, kabsch_alignment,
    kabsch_alignment_with_scale, transform_entities,
    transform_entities_with_scale,
};

// Kabsch alignment (minimize RMSD between point sets)
let (rotation, translation) = kabsch_alignment(&source, &target)?;
let (rotation, translation, scale) =
    kabsch_alignment_with_scale(&source, &target)?;

// Apply transform to entities
transform_entities(&mut entities, rotation, translation);
transform_entities_with_scale(&mut entities, rotation, translation, scale);

// Align entities to a reference structure
align_to_reference(&mut mobile, &reference);

// Extract alpha-carbon positions
let ca_positions: Vec<Vec3> = extract_ca_positions(&entities);
let ca_by_chain: Vec<Vec<Vec3>> = extract_ca_from_chains(&entities);

// Get continuous backbone segments
let segments: Vec<Vec<Vec3>> = extract_backbone_segments(&entities);

// Compute centroid
let center: Vec3 = centroid(&positions);
```

