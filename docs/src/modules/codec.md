# Codec

`molex::ops::codec` holds the crate's shared adapter error type plus a
small entity-utility helper. The ASSEM02 binary wire format lives in
[`molex::ops::wire`](wire.md); the structural-transform routines live in
[`molex::ops::transform`](analysis.md#transforms).

## Public surface

```rust,ignore
use molex::ops::codec::{ca_positions, AdapterError};

// CA positions across every protein entity, residue-ordered per chain.
let ca: Vec<Vec3> = ca_positions(&entities);
```

## Error type

`AdapterError` is the `Err` variant for every adapter entry point
(`pdb_str_to_entities`, `mmcif_str_to_entities`, `bcif_to_entities`,
their `_file_*` and `_to_all_models` siblings), the PDB writers
(`assembly_to_pdb`, `entities_to_pdb`), and the ASSEM02 codec
(`ops::wire::serialize_assembly` / `deserialize_assembly`).

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
