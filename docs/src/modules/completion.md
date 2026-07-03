# Completion

Parsed structures routinely omit atoms (hydrogens, distal sidechain atoms,
sometimes whole sidechains beyond the backbone). Completion brings each
standard residue's atom set up to its ideal-geometry template: for every
residue that resolves to a standard amino acid or DNA/RNA base, it rigid-fits
the template's present atoms onto the parsed residue (reusing Kabsch
alignment) and places each absent template atom at its transformed ideal
coordinate. The pass is purely additive: parsed atoms keep their coordinates,
only absent atoms are created. The machinery lives in `src/chemistry` and
`src/entity/molecule/complete.rs`.

## Completion levels

`Completion` selects what construction is allowed to fabricate and whether it
keeps parsed hydrogens:

| Level | Fabricates | Input hydrogens |
|---|---|---|
| `Raw` | nothing | kept |
| `Heavy` | absent heavy template atoms | dropped |
| `AllAtom` | absent heavy atoms plus template hydrogens | replaced by template H |
| `HeavyRaw` | nothing | dropped |

`Heavy` is the file-ingest default: the no-level PDB/mmCIF/BinaryCIF parse
entry points use it, so `Assembly::from_file` and friends fill missing heavy
atoms and drop input hydrogens. `Raw` is the pure construction level (wire
round-trip, array ingest, continuous rebuild) where inputs are already
complete and re-running the rigid fit would be waste. `AllAtom` is the opt-in
to a fully protonated model. `HeavyRaw` strips hydrogens without fabricating.

Completion runs on each chain's surviving residues; a residue dropped at
construction for lacking a complete backbone is not resurrected. Protein
C-termini gain their carboxylate `OXT` (and `HXT` under all-atom). N-terminal
extra protons and leaving atoms are never fabricated.

## Re-completion projections

`Assembly` exposes non-mutating re-completion: `complete(level)` returns a
fresh assembly with its polymer entities rebuilt at `level`, with
`normalize()` and `to_all_atom()` as the `Heavy` / `AllAtom` shortcuts.
Non-polymer entities pass through unchanged. Because a fabricated atom shifts
every later atom, atom indices and `AtomId`s are not stable across the
projection, and rendering connections are dropped rather than carried over.

```rust,ignore
use molex::entity::molecule::Completion;

let heavy = assembly.normalize();          // Completion::Heavy
let all_atom = assembly.to_all_atom();     // Completion::AllAtom
let custom = assembly.complete(Completion::Raw);
```

From Python the same levels are `Completion.Raw` / `Heavy` / `AllAtom`, passed
to `from_pdb` / `from_mmcif` / `from_bcif` (default `Heavy`) and to
`Assembly.complete`. The C API exposes `molex_Completion` and the
`*_with_completion` parse variants.
