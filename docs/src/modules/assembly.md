# Assembly

`Assembly` (source: `src/assembly.rs`) is the host-owned structural source of
truth. It holds a `Vec<Arc<MoleculeEntity>>` plus three pieces of side state:

- per-entity DSSP secondary structure (`ss_types`), populated only on demand
- owner-set rendering connections (`connections`), keyed by `ConnectionType`
- a monotonic `generation` counter that bumps on every mutation

Construction is secondary-structure-free. `Assembly::new` and
`Assembly::from_arcs` leave `ss_types` empty, because per-entity DSSP is
expensive and most callers do not need it. Secondary structure is opt-in:
call `recompute_ss()` when a consumer (a load, a render that reads SS) needs
it populated. Hydrogen bonds and disulfides are never stored on the
`Assembly`; they are either supplied by the owner via `set_connections` or
computed on demand from geometry via `detect_fallback_connections`.

Entities are stored behind `Arc`, so cloning an `Assembly` is O(entities) of
refcount bumps regardless of total atom count, and a mutation clones only the
touched entity (`Arc::make_mut`).

```rust,ignore
use molex::Assembly;

let mut assembly = Assembly::new(entities); // ss_types empty
assembly.recompute_ss();                    // opt in to DSSP secondary structure
```

## Read accessors

| Method | Returns | Description |
|---|---|---|
| `entities()` | `&[Arc<MoleculeEntity>]` | All entities in declaration order |
| `entity(id)` | `Option<&MoleculeEntity>` | Look up an entity by id |
| `into_entities()` | `Vec<MoleculeEntity>` | Consume the assembly, returning owned entities (side state discarded) |
| `generation()` | `u64` | Monotonic counter, bumps on mutation |
| `sequence()` | `Vec<String>` | One-letter sequence per polymer entity |
| `ss_types(id)` | `&[SSType]` | Per-backbone-residue SS for an entity (empty until `recompute_ss`) |
| `ss_per_residue(id)` | `Option<Vec<SSType>>` | SS spread across every residue of a protein (`Coil` where DSSP skipped) |
| `secondary_structure()` | `Vec<String>` | Per-entity Q3 string, aligned 1:1 with `entities()` |
| `connections()` | `&HashMap<ConnectionType, Vec<AtomLink>>` | Owner-set rendering connections |
| `detect_fallback_connections()` | `HashMap<ConnectionType, Vec<AtomLink>>` | Viewer-only disulfide + backbone H-bond geometry computed on demand (pure) |
| `covalent_bonds()` | `Vec<CovalentBond>` | Intra-entity bonds aggregated across entities |
| `contacts(cutoff, level, between_chains)` | `Vec<Contact>` | Atom/residue/chain contacts (see [Analysis](analysis.md)) |
| `sasa(probe, n_points)` | `f32` | Total protein solvent-accessible surface area (see [Analysis](analysis.md)) |

## Completion projections

`Assembly` re-completion is non-mutating: each method returns a fresh
assembly with its polymer entities rebuilt at a completion level (non-polymer
entities pass through unchanged). Atom indices and `AtomId`s are not stable
across these projections (a fabricated atom shifts every later atom), so
rendering connections are dropped rather than carried over; the generation
counter is preserved.

| Method | Level | Effect |
|---|---|---|
| `complete(level)` | `Completion` | Rebuild at the given level (the canonical projection) |
| `normalize()` | `Heavy` | Fabricate missing heavy atoms |
| `to_all_atom()` | `AllAtom` | Fabricate missing heavy atoms plus template hydrogens |

See the [Completion](completion.md) page for what each level fabricates.

## Mutation

Direct content mutators (`add_entity`, `remove_entity`, `entities_mut`) are
crate-internal. Cross-crate callers mutate through the typed edit path so the
host's broadcast routing has a single funnel:

```rust,ignore
use molex::ops::AssemblyEdit;

assembly.apply_edit(&AssemblyEdit::SetEntityCoords { entity, coords })?;
assembly.apply_edits(&edits)?; // apply a batch in order, stop at first error
```

Each successful apply bumps `generation`. Secondary structure is not
recomputed by the edit path; call `recompute_ss()` afterward if you need it.
See the [Edits](edit.md) page for the full `AssemblyEdit` vocabulary.

`carry_ss_from(&src)` copies per-entity SS from an earlier snapshot onto a
freshly built coordinate-only snapshot (the streaming render path), skipping
any entity whose residue count no longer matches so stale SS is never indexed
against the wrong residues.
