# Edits

`molex::ops::edit` (source: `src/ops/edit/`) is the typed cross-crate
mutation route for an `Assembly`. The direct content mutators on `Assembly`
are crate-internal; an external caller mutates by building `AssemblyEdit`
values and applying them, so the host's broadcast routing has a single typed
funnel. `AssemblyEdit`, `EditError`, and `BulkEditError` are re-exported from
`molex::ops`.

## AssemblyEdit

`AssemblyEdit` covers the steady-state mutation vocabulary. Residues are
addressed by index within their parent entity's residue list; `EntityId` is
the stable identifier across edits.

| Variant | Fields | Effect |
|---|---|---|
| `SetEntityCoords` | `entity`, `coords` | Replace every atom position in the entity (`coords.len()` must equal `atom_count`) |
| `SetResidueCoords` | `entity`, `residue_idx`, `coords` | Replace one residue's atom positions (count must match; shape unchanged) |
| `MutateResidue` | `entity`, `residue_idx`, `new_name`, `new_atoms`, `new_variants` | Replace a residue's identity and atoms (count may change; later residues' atom ranges shift) |
| `SetVariants` | `entity`, `residue_idx`, `variants` | Replace a residue's variant tag list |
| `AddEntity` | `entity` (boxed) | Append an entity (id must not collide) |
| `RemoveEntity` | `entity` | Remove an entity by id |

Per-residue topology (inserting or deleting a residue inside an entity) is not
enumerated; bridge consumers fall back to a full pose rebuild for that class
of change. The two topology variants (`AddEntity` / `RemoveEntity`) are
representable in memory but not on the delta wire.

## Apply path

```rust,ignore
use molex::ops::AssemblyEdit;

assembly.apply_edit(&edit)?;        // single edit
assembly.apply_edits(&edits)?;      // batch, in order
```

Each successful apply bumps the assembly's generation counter. Secondary
structure is not recomputed; call `recompute_ss()` afterward if a consumer
needs it. `apply_edits` stops at the first failing edit and returns a
`BulkEditError` carrying that edit's index plus the underlying `EditError`;
edits applied before it stay applied.

`EditError` distinguishes the failure modes: `UnknownEntity`,
`UnknownResidue`, `CountMismatch`, `NotPolymer` (a per-residue edit aimed at a
non-polymer entity), and `DuplicateEntity` (an `AddEntity` whose id already
exists).

The delta wire format ([Wire Format](wire.md#delta-format)) serializes a
`Vec<AssemblyEdit>` for transport; topology edits are excluded there.
