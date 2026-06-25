# Adapters

Format adapters live in `molex::adapters` (source: `src/adapters/`). Each adapter's primary API returns `Vec<MoleculeEntity>`.

Every structure parser feeds parser-emitted atom rows into the shared `EntityBuilder`, so chain/residue grouping, modified-residue merge logic, and entity classification are identical regardless of input format. Residues populate both their structural-side identifiers (`label_seq_id`, `name`) and the optional author-side identifiers from mmCIF / BinaryCIF (`auth_seq_id`, `auth_comp_id`, `ins_code`). `ProteinEntity` and `NAEntity` likewise carry an `auth_asym_id: Option<u8>` for the author-side chain label.

## PDB (`adapters::pdb`)

Hand-rolled column-positional scanner over wwPDB v3.3 section 9. Records other than `ATOM`, `HETATM`, `MODEL`, `ENDMDL`, `TER`, and `END` are skipped.

```rust,ignore
pdb_file_to_entities(path: &Path) -> Result<Vec<MoleculeEntity>, AdapterError>
pdb_str_to_entities(pdb_str: &str) -> Result<Vec<MoleculeEntity>, AdapterError>
structure_file_to_entities(path: &Path) -> Result<Vec<MoleculeEntity>, AdapterError>

// Per-MODEL entry points (NMR ensembles, multi-state trajectories)
pdb_str_to_all_models(pdb_str: &str)
    -> Result<Vec<Vec<MoleculeEntity>>, AdapterError>
pdb_file_to_all_models(path: &Path)
    -> Result<Vec<Vec<MoleculeEntity>>, AdapterError>

// Writers
assembly_to_pdb(assembly: &Assembly) -> Result<String, AdapterError>
entities_to_pdb<E: Borrow<MoleculeEntity>>(entities: &[E])
    -> Result<String, AdapterError>
```

`structure_file_to_entities` auto-detects PDB vs mmCIF by file extension (`.pdb`/`.ent` -> PDB, everything else -> mmCIF).

The writers refuse to emit when the assembly exceeds any legacy PDB limit (>99,999 atoms, >62 polymer chains, >9,999 residues per chain), returning `AdapterError::InvalidFormat` so callers can switch to the mmCIF writer.

## mmCIF (`adapters::cif`)

A streaming fast scanner handles the common case directly; a DOM-backed fallback covers structurally awkward files. Both paths emit `AtomRow` into `EntityBuilder`. The DOM types are also re-exported for typed extractors (`CoordinateData`, `ReflectionData`, `UnitCell`).

```rust,ignore
mmcif_file_to_entities(path: &Path) -> Result<Vec<MoleculeEntity>, AdapterError>
mmcif_str_to_entities(cif_str: &str) -> Result<Vec<MoleculeEntity>, AdapterError>

// One entity list per pdbx_PDB_model_num
mmcif_str_to_all_models(cif_str: &str)
    -> Result<Vec<Vec<MoleculeEntity>>, AdapterError>
mmcif_file_to_all_models(path: &Path)
    -> Result<Vec<Vec<MoleculeEntity>>, AdapterError>
```

## BinaryCIF (`adapters::bcif`)

Decodes BinaryCIF (MessagePack-encoded CIF) with column-level codecs.

```rust,ignore
bcif_file_to_entities(path: &Path) -> Result<Vec<MoleculeEntity>, AdapterError>
bcif_to_entities(data: &[u8]) -> Result<Vec<MoleculeEntity>, AdapterError>

// One entity list per pdbx_PDB_model_num
bcif_to_all_models(bytes: &[u8])
    -> Result<Vec<Vec<MoleculeEntity>>, AdapterError>
bcif_file_to_all_models(path: &Path)
    -> Result<Vec<Vec<MoleculeEntity>>, AdapterError>
```

## MRC/CCP4 density maps (`adapters::mrc`)

Parses MRC/CCP4 format electron density maps into `Density` (which wraps `VoxelGrid`).

```rust,ignore
mrc_file_to_density(path: &Path) -> Result<Density, DensityError>
mrc_to_density(data: &[u8]) -> Result<Density, DensityError>
```

## DCD trajectories (`adapters::dcd`)

Reads DCD binary trajectory files (CHARMM/NAMD format).

```rust,ignore
dcd_file_to_frames(path: &Path) -> Result<Vec<DcdFrame>, AdapterError>

pub struct DcdHeader { /* timestep, n_atoms, etc. */ }
pub struct DcdFrame { pub x: Vec<f32>, pub y: Vec<f32>, pub z: Vec<f32> }
pub struct DcdReader<R> { /* streaming reader */ }
```

## AtomWorks (`adapters::atomworks`, feature = `python`)

Bidirectional conversion between molex entities and Biotite `AtomArray` objects with AtomWorks annotations. Requires the `python` feature.

Entity-aware functions (preserve molecule type, entity ID, chain grouping):

```rust,ignore
entities_to_atom_array(assembly_bytes: Vec<u8>) -> PyResult<PyObject>
entities_to_atom_array_plus(assembly_bytes: Vec<u8>) -> PyResult<PyObject>
atom_array_to_entities(atom_array: PyObject) -> PyResult<Vec<u8>>
entities_to_atom_array_parsed(assembly_bytes: Vec<u8>, filename: &str) -> PyResult<PyObject>
parse_file_to_entities(path: &str) -> PyResult<Vec<u8>>
parse_file_full(path: &str) -> PyResult<PyObject>
```
