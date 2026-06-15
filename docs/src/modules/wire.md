# Wire Format

`molex::ops::wire` is the home of the ASSEM02 binary wire format:
the on-disk / IPC format used to round-trip an `Assembly` through
bytes.

ASSEM02 is entity-aware: it preserves per-entity molecule-type
metadata so the decoder can reconstruct `MoleculeEntity` variants
without re-running residue classification. It also carries each
entity's `EntityId` and a trailing per-residue variants section.

The legacy ASSEM01 format (magic `ASSEM01\0`, atoms only, no variants
section) is still accepted by the deserializer and read as if every
residue carried empty variants. The serializer always emits ASSEM02.

## Public surface

```rust,ignore
use molex::ops::wire::{
    assembly_bytes, deserialize_assembly, serialize_assembly, ASSEMBLY_MAGIC,
};

// Serialize an Assembly
let bytes: Vec<u8> = serialize_assembly(&assembly)?;

// Serialize a raw entity slice (no Assembly wrapper)
let bytes: Vec<u8> = assembly_bytes(&entities)?;

// Deserialize back to an Assembly (recomputes derived data via Assembly::new)
let assembly: Assembly = deserialize_assembly(&bytes)?;

// Magic header for ASSEM02 detection
assert_eq!(&bytes[0..8], ASSEMBLY_MAGIC);
```

## Byte layout

```text
8 bytes:   magic "ASSEM02\0"
4 bytes:   entity_count (u32 BE)
Per entity header (9 bytes each):
  1 byte:  molecule_type wire byte
  4 bytes: atom_count (u32 BE)
  4 bytes: entity_id (u32 BE) — the originator's EntityId.raw()
Per atom (26 bytes):
  12 bytes: x, y, z (f32 BE x 3)
  1 byte:   chain_id
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

Each entity contributes a variant block even when it carries no
variants — a non-polymer entity or one with no variant-bearing
residues emits a `variant_residue_count` of 0 (4 bytes).

### Variant tags

```text
0x01  NTerminus    — 0 payload bytes
0x02  CTerminus    — 0 payload bytes
0x03  Disulfide    — 0 payload bytes
0x04  Protonation  — 1 sub-tag byte:
        0x01  HisDelta
        0x02  HisEpsilon
        0x03  HisDoubly
        0xFF  Custom — u16 BE length + UTF-8 bytes
0xFE  Other        — u16 BE length + UTF-8 bytes
```

Occupancy and b_factor are **not** preserved on the wire; deserialize
resets them to 1.0 and 0.0 respectively. Derived data (`ss_types`) is
recomputed by `Assembly::new` at deserialize time.
