# Entity System

The entity system lives in `molex::entity` (source: `src/entity/`).

## MoleculeEntity

The central type. An enum with four variants:

```rust,ignore
pub enum MoleculeEntity {
    Protein(ProteinEntity),
    NucleicAcid(NAEntity),
    SmallMolecule(SmallMoleculeEntity),
    Bulk(BulkEntity),
}
```

Common methods available on all variants (delegated to the inner type):

| Method | Returns | Description |
|---|---|---|
| `id()` | `EntityId` | Unique identifier |
| `molecule_type()` | `MoleculeType` | Classification (Protein, DNA, Ligand, etc.) |
| `atom_set()` | `&[Atom]` | All atoms |
| `atom_count()` | `usize` | Number of atoms |
| `positions()` | `Vec<Vec3>` | All atom positions |
| `label()` | `String` | Human-readable label (e.g. "Protein A", "Ligand (ATP)") |
| `residue_count()` | `usize` | Residues (polymer) or molecules (bulk) |

Variant-specific downcasting: `as_protein()`, `as_nucleic_acid()`, `as_small_molecule()`, `as_bulk()`.

## MoleculeType

```rust,ignore
pub enum MoleculeType {
    Protein, DNA, RNA, Ligand, Ion, Water, Lipid, Cofactor, Solvent,
}
```

Residue names are mapped to molecule types by `classify_residue()` using built-in lookup tables.

## Atom

```rust,ignore
pub struct Atom {
    pub position: Vec3,      // 3D position in angstroms
    pub occupancy: f32,      // 0.0 to 1.0
    pub b_factor: f32,       // temperature factor
    pub element: Element,    // chemical element
    pub name: [u8; 4],       // PDB atom name (e.g. b"CA  ")
    pub formal_charge: i8,   // signed formal charge; 0 = neutral
    pub observed: bool,      // true if parsed, false if fabricated by completion
}
```

`observed` is provenance metadata only (never part of atom identity); it backs
the Python `observed_mask` column. Atoms fabricated by completion carry
`observed: false`.

Residue name, residue number, and chain ID are stored on the entity/residue that owns the atom.

## Residue

```rust,ignore
pub struct Residue {
    pub name: [u8; 3],                  // structural-side residue name (e.g. b"ALA")
    pub label_seq_id: i32,              // mmCIF label_seq_id / PDB resSeq
    pub auth_seq_id: Option<i32>,       // mmCIF auth_seq_id; None falls back to label_seq_id
    pub auth_comp_id: Option<[u8; 3]>,  // mmCIF auth_comp_id; None falls back to name
    pub ins_code: Option<u8>,           // PDB iCode / mmCIF pdbx_PDB_ins_code; None = blank
    pub atom_range: Range<usize>,       // index range into the parent entity's atoms
}
```

Internal grouping uses the `label_*` fields; user-facing output uses the `auth_*` fields, falling back to the `label_*` values when the author-side identifiers are absent.

## ProteinEntity

A single protein chain. Implements both `Entity` and `Polymer` traits.

```rust,ignore
pub struct ProteinEntity {
    pub id: EntityId,
    pub atoms: Vec<Atom>,             // canonical order: N, CA, C, O, sidechain heavy..., H...
    pub residues: Vec<Residue>,       // name, number, atom_range
    pub segment_breaks: Vec<usize>,   // backbone gap indices
    pub bonds: Vec<CovalentBond>,     // intra-entity bonds (AtomId endpoints)
    pub pdb_chain_id: u8,             // structural-side chain byte (label_asym_id)
    pub auth_asym_id: Option<u8>,     // author-side chain byte; None = same as pdb_chain_id
}
```

Construction (`ProteinEntity::new`) reorders atoms into the canonical per-residue layout, drops residues missing any of N/CA/C/O (OXT counts as the C-terminal oxygen), computes segment breaks from C(i)->N(i+1) distance > 2.0 angstroms, and populates `bonds` from the `AminoAcid` chemistry tables plus universal backbone bonds (N-CA, CA-C, C=O) and inter-residue peptide bonds C(i)-N(i+1).

Derived views (computed on each call, not cached):

- `to_backbone() -> Vec<ResidueBackbone>` -- N, CA, C, O positions per residue
- `to_interleaved_segments() -> Vec<Vec<Vec3>>` -- N/CA/C positions per continuous segment (for spline rendering)

Bond filters:

- `backbone_bonds()` -- iterator of bonds whose endpoints both fall inside the canonical backbone region of any residue (offsets 0..4)
- `sidechain_bonds()` -- iterator of bonds with at least one heavy-atom sidechain endpoint

## NAEntity

A single DNA or RNA chain. Same overall shape as `ProteinEntity`: atoms, residues, segment breaks, intra-entity `bonds`, `pdb_chain_id`, and `auth_asym_id: Option<u8>`. The DNA-vs-RNA discriminator is carried in an extra `na_type: MoleculeType` field. Implements `Entity` and `Polymer`.

## SmallMoleculeEntity

A single non-polymer molecule (ligand, ion, cofactor, lipid). Implements `Entity` only.

## BulkEntity

A group of identical small molecules (water, solvent). All atoms stored in a single flat `Vec<Atom>`. Implements `Entity` only.

## EntityId

An opaque identifier allocated by `EntityIdAllocator`. Entities within a structure have unique IDs. The allocator is a simple incrementing counter.

## Entity and Polymer traits

```rust,ignore
pub trait Entity {
    fn id(&self) -> EntityId;
    fn molecule_type(&self) -> MoleculeType;
    fn atoms(&self) -> &[Atom];
    fn positions(&self) -> Vec<Vec3>;     // default impl
    fn atom_count(&self) -> usize;        // default impl
}

pub trait Polymer: Entity {
    fn residues(&self) -> &[Residue];
    fn residue_count(&self) -> usize;     // default impl
    fn segment_breaks(&self) -> &[usize];
    fn segment_count(&self) -> usize;     // default impl
    fn segment_range(&self, idx: usize) -> Range<usize>;  // default impl
    fn segment_residues(&self, idx: usize) -> &[Residue];  // default impl
}
```

## Surface types

The `entity::surface` module provides:

- **`VoxelGrid`** -- 3D `Array3<f32>` grid with unit cell metadata (dimensions, angles, origin, sampling intervals). Provides `voxel_size()` and `frac_to_cart_matrix()` for coordinate conversion.
- **`Density`** -- wraps `VoxelGrid` with density map metadata. Constructed by `mrc_file_to_density()` or `mrc_to_density()`.
