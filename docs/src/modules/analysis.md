# Analysis

Structural analysis lives in `molex::analysis` (source: `src/analysis/`). Most analysis functions operate on entity-level types (`&[Atom]`, `&[ResidueBackbone]`, `&[MoleculeEntity]`).

`Assembly` exposes a few of these analyses as methods: `ss_types` (after
`recompute_ss`), `detect_fallback_connections` (disulfide + backbone-H-bond
geometry), `covalent_bonds`, `contacts`, and `sasa`. The standalone functions
documented below are the building blocks underneath. Note that secondary
structure is opt-in (it is empty until `recompute_ss` runs) and H-bonds /
disulfides are computed on demand rather than stored on the `Assembly`.

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

For most callers the recommended path is to construct an `Assembly`, call `assembly.recompute_ss()`, then read `assembly.ss_types(entity_id)`. `recompute_ss` runs H-bond detection and `classify` for every protein entity; until it is called, `ss_types` is empty.

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

## Solvent-accessible surface area (`analysis::sasa`)

Shrake-Rupley SASA over an assembly's protein atoms, using Bondi van der
Waals radii and a deterministic golden-spiral sampling of each atom's probe
sphere (no RNG). Water, ions, ligands, and nucleic acids are excluded,
matching the protein-scope convention of freesasa and biotite.

```rust,ignore
use molex::analysis::sasa::{DEFAULT_PROBE_RADIUS, DEFAULT_N_POINTS};

// Total protein SASA in Angstrom^2.
let area = assembly.sasa(DEFAULT_PROBE_RADIUS, DEFAULT_N_POINTS);
```

`DEFAULT_PROBE_RADIUS` is 1.4 (water); `DEFAULT_N_POINTS` is 960 test points
per atom (higher is finer and slower).

## Contacts (`analysis::contacts`)

`Assembly::contacts` enumerates atom pairs closer than a cutoff, reported at
the requested granularity. A single spatial grid keeps neighbor lookups
near-linear.

```rust,ignore
use molex::analysis::{Contact, ContactLevel};

// All atom-atom contacts within 4.0 A, deduped to residue pairs,
// keeping only inter-entity pairs.
let contacts: Vec<Contact> = assembly.contacts(4.0, ContactLevel::Residue, true);
```

`ContactLevel` is `Atom`, `Residue`, or `Chain`; `Residue` and `Chain` dedup
to unique residue/entity pairs, each keeping one representative atom pair.

## Transforms (`ops::transform`)

Kabsch alignment and superposition RMSD. `kabsch_alignment` is re-exported
from `molex::ops`; `rmsd` lives at `molex::ops::transform`.

```rust,ignore
use molex::ops::kabsch_alignment;
use molex::ops::transform::rmsd;

// Kabsch alignment (the rigid motion that minimizes RMSD between two point
// sets); None for mismatched lengths or fewer than 3 points.
let aligned: Option<(Mat3, Vec3)> = kabsch_alignment(&source, &target);

// Optimal-superposition RMSD; None for fewer than 3 points.
let deviation: Option<f32> = rmsd(&a, &b);
```

