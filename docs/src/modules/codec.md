# Errors

There is no `ops::codec` module. The crate's shared parser/serializer error
type is `AdapterError`, which lives in `molex::ops::error` and is re-exported
at the crate root as `molex::AdapterError` (source: `src/ops/error.rs`).

The assembly binary wire format lives in
[`molex::ops::wire`](wire.md); the structural-transform routines (Kabsch
alignment, RMSD) live in [`molex::ops::transform`](analysis.md#transforms-opstransform).

## Error type

`AdapterError` is the `Err` variant for the `Assembly` parse entry points
(`from_pdb` / `from_mmcif` / `from_bcif` / `from_file` and the `*_to_all_models`
multi-model functions), the PDB writers, and the assembly wire codec.

```rust,ignore
pub enum AdapterError {
    /// The input bytes/text do not conform to the expected format.
    InvalidFormat(String),
    /// A PDB file could not be parsed.
    PdbParseError(String),
    /// An error occurred during binary serialization or deserialization.
    SerializationError(String),
}
```

Density and trajectory adapters use their own error types (`DensityError`
from the MRC adapter, for instance); only the structure-and-wire paths funnel
through `AdapterError`.
