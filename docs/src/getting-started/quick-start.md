# Quick Start

## Parse a structure file into an assembly

`Assembly::from_file` dispatches on the extension (`.pdb`/`.ent` -> PDB,
mmCIF otherwise) and defaults to `Completion::Heavy`:

```rust,ignore
use std::path::Path;
use molex::Assembly;

let assembly = Assembly::from_file(Path::new("1ubq.pdb"))?;
for e in assembly.entities() {
    println!("{:?}: {} atoms", e.molecule_type(), e.atom_count());
}
```

For string input use `Assembly::from_pdb(&str)` / `from_mmcif` / `from_bcif`,
each with a `_with(..., Completion)` variant. Reach the entities via
`assembly.entities()` and write back with `assembly.to_pdb()`.

## Work with entities

```rust,ignore
use molex::{MoleculeEntity, MoleculeType};

let entities = assembly.entities();

// Filter to protein chains
let proteins: Vec<_> = entities.iter()
    .filter(|e| e.molecule_type() == MoleculeType::Protein)
    .collect();

// Access protein-specific data
for entity in &proteins {
    let protein = entity.as_protein().unwrap();
    let backbone = protein.to_backbone();
    println!(
        "Chain {}: {} residues, {} segments",
        protein.pdb_chain_id as char,
        protein.residues.len(),
        protein.segment_count(),
    );
}
```

## Run DSSP secondary structure assignment

Secondary structure is opt-in: an `Assembly` starts with empty `ss_types`, so
call `recompute_ss()` before reading it. Backbone H-bonds are never stored;
they surface on demand through `detect_fallback_connections()`.

```rust,ignore
use molex::{Assembly, ConnectionType, SSType};

let mut assembly = Assembly::new(entities);
assembly.recompute_ss(); // populate ss_types (expensive; skipped at construction)

let protein_id = assembly.entities()[0].id();
for (i, ss) in assembly.ss_types(protein_id).iter().enumerate() {
    println!("Residue {}: {:?}", i, ss); // Helix, Sheet, or Coil
}
if let Some(hbonds) = assembly.detect_fallback_connections().get(&ConnectionType::HBond) {
    for link in hbonds {
        println!("hbond {:?} -> {:?}", link.a, link.b);
    }
}
```

## Serialize to assembly binary (for FFI/IPC)

```rust,ignore
use molex::ops::wire::assembly_bytes;

let bytes = assembly_bytes(&entities)?;
// Send `bytes` over FFI, IPC, or network
```

`assembly_bytes` is the only public wire entry point. To serialize an
`Assembly` rather than a raw entity slice, call `assembly.to_bytes()`; decode
with `Assembly::from_bytes(&bytes)`, which returns an assembly with empty
`ss_types`.

## Python usage

```python
import molex

# PDB round-trip via assembly bytes
assembly_bytes = molex.pdb_to_assembly_bytes(pdb_string)
pdb_back = molex.assembly_bytes_to_pdb(assembly_bytes)

# Biotite-agnostic columnar interchange (for ML pipelines)
table = molex.PyAtomTable.from_assembly_bytes(assembly_bytes)
coords = table.coords  # (N, 3) float32 numpy array
assembly_bytes = table.to_assembly_bytes()
```
