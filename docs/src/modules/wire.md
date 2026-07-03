# Wire Format

`molex::ops::wire` holds two binary formats: the assembly format (a full
`Assembly` snapshot) and the delta format (an incremental edit list). Both
start with a fixed 8-byte magic followed by a `u8` version byte that selects
the payload layout; the magic never changes when the payload does. Only
version 1 exists today, and any other version is a clean decode rejection.

## Assembly format

The format is entity-aware: it preserves per-entity molecule-type metadata so
the decoder reconstructs `MoleculeEntity` variants without re-running residue
classification. It also carries each entity's `EntityId` and a trailing
per-residue variants section.

### Public surface

`assembly_bytes` is the only public free function; it serializes a raw entity
slice. `serialize_assembly` / `deserialize_assembly` are crate-internal. To
round-trip an `Assembly` use its `to_bytes` / `from_bytes` methods.

```rust,ignore
use molex::ops::wire::{assembly_bytes, ASSEMBLY_MAGIC};
use molex::Assembly;

// Serialize a raw entity slice.
let bytes: Vec<u8> = assembly_bytes(&entities)?;

// Round-trip an Assembly.
let bytes: Vec<u8> = assembly.to_bytes()?;
let decoded: Assembly = Assembly::from_bytes(&bytes)?; // ss_types empty

assert_eq!(&bytes[0..8], ASSEMBLY_MAGIC);
```

Decoding returns a secondary-structure-free assembly: `ss_types` is empty
until a caller invokes `recompute_ss()`. The decode path does not recompute
derived data.

### Byte layout

The chain id lives in the per-entity header as a length-prefixed string, so
chain ids are not capped to a single printable byte. Each atom row is 25
bytes.

```text
8 bytes:   magic "ASSEMBLY"
1 byte:    version (currently 1)
4 bytes:   entity_count (u32 BE)
Per entity header (variable):
  1 byte:  molecule_type wire byte
  4 bytes: atom_count (u32 BE), the body-row count for this entity
  4 bytes: entity_id (u32 BE), the originator's EntityId.raw()
  2 bytes: chain_len (u16 BE)
  chain_len bytes: chain id (UTF-8 label_asym_id; empty for non-polymers)
Per atom (25 bytes):
  12 bytes: x, y, z (f32 BE x 3)
  3 bytes:  res_name
  4 bytes:  res_num (i32 BE)
  4 bytes:  atom_name
  2 bytes:  element symbol (byte 0, byte 1 or 0)
Per-entity variants section (after all atoms, in entity-header order):
  per entity {
    4 bytes: variant_residue_count (u32 BE)
    per variant-bearing residue {
      4 bytes: label_seq_id (i32 BE)
      4 bytes: variant_count (u32 BE)
      per variant {
        1 byte:  tag
        tag-specific payload
      }
    }
  }
```

Each entity contributes a variant block even when it carries no variants: a
non-polymer entity or one with no variant-bearing residues emits a
`variant_residue_count` of 0 (4 bytes).

### Variant tags

```text
0x01  NTerminus    0 payload bytes
0x02  CTerminus    0 payload bytes
0x03  Disulfide    0 payload bytes
0x04  Protonation  1 sub-tag byte:
        0x01  HisDelta
        0x02  HisEpsilon
        0x03  HisDoubly
        0xFF  Custom (u16 BE length + UTF-8 bytes)
0xFE  Other        u16 BE length + UTF-8 bytes
```

Occupancy, b_factor, formal_charge, and the `observed` flag are **not**
preserved on the wire; deserialize resets occupancy / b_factor to 1.0 / 0.0.

## Delta format

`molex::ops::wire::delta` carries an incremental edit list (`Vec<AssemblyEdit>`)
rather than a full snapshot; it is the steady-state path between the host and
its plugins. Topology edits (`AddEntity` / `RemoveEntity`) are not
representable on the wire; callers broadcast a full assembly snapshot for that
class of change.

```rust,ignore
use molex::ops::wire::delta::{serialize_edits, deserialize_edits, DELTA_MAGIC};

let bytes = serialize_edits(&edits)?;
let edits = deserialize_edits(&bytes)?;
```

### Byte layout

```text
8 bytes:  magic b"DELTA\0\0\0"
1 byte:   version (currently 1)
4 bytes:  edit_count (u32 BE)
per edit:
  1 byte: tag
  tag-specific payload
```

Edit tags:

```text
0x01  SetEntityCoords   u32 entity_id, u32 coord_count, 12 bytes per coord (xyz f32 BE)
0x02  SetResidueCoords  u32 entity_id, u32 residue_idx, u32 coord_count, 12 bytes per coord
0x03  MutateResidue     u32 entity_id, u32 residue_idx, 3 bytes new_name, u32 atom_count,
                        25 bytes per atom (same row layout as the assembly format),
                        u32 variant_count, per-variant payload (same as the variants block)
0x04  SetVariants       u32 entity_id, u32 residue_idx, u32 variant_count, per-variant payload
```

See the [Edits](edit.md) page for the `AssemblyEdit` types these tags encode.
