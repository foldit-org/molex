# Assembly

`Assembly` (source: `src/assembly.rs`) is the host-owned structural source of truth. It bundles a `Vec<MoleculeEntity>` with eagerly-computed derived data:

- per-entity DSSP secondary structure (`ss_types`)
- backbone H-bonds (Kabsch-Sander)
- cross-entity covalent bonds (currently disulfides)
- a monotonic `generation` counter

Every `&mut Assembly` mutation bumps `generation` and recomputes the full derived-data set before returning, so any consumer holding a snapshot sees consistent state.

```rust,ignore
use molex::{Assembly, CoordinateSnapshot};

let assembly = Assembly::new(entities); // runs DSSP secondary-structure classification
```

## Read accessors

| Method | Returns | Description |
|---|---|---|
| `entities()` | `&[Arc<MoleculeEntity>]` | All entities in declaration order (`Arc` so cloning the slice is O(entities) of refcount bumps) |
| `entity(id)` | `Option<&MoleculeEntity>` | Look up an entity by id |
| `generation()` | `u64` | Monotonic counter, bumps on mutation |
| `ss_types(id)` | `&[SSType]` | DSSP classification per residue for an entity |
| `connections()` | `&HashMap<ConnectionType, Vec<AtomLink>>` | Owner-populated rendering connections, keyed by category |
| `detect_fallback_connections()` | `HashMap<ConnectionType, Vec<AtomLink>>` | Viewer-only disulfide + backbone H-bond geometry computed on demand (pure) |

## Mutations

All mutations recompute derived data:

```rust,ignore
assembly.add_entity(new_entity);
assembly.remove_entity(entity_id);
assembly.update_positions(entity_id, &new_coords);   // skipped if atom counts mismatch
assembly.set_coordinate_snapshot(snapshot);          // bulk position update
```

## CoordinateSnapshot

`CoordinateSnapshot` captures per-entity positions for a "save / restore" workflow:

```rust,ignore
let saved = CoordinateSnapshot::from_assembly(&assembly);
// ... run mutations ...
assembly.set_coordinate_snapshot(saved); // restore
```

Entities present in the snapshot but missing from the assembly are ignored; entities in the assembly missing from the snapshot keep their current positions.
