# Quick Start

## Parse a PDB file into entities

```rust,ignore
use molex::adapters::pdb::pdb_file_to_entities;
use std::path::Path;

let entities = pdb_file_to_entities(Path::new("1ubq.pdb"))?;
for e in &entities {
    println!("{}: {} atoms", e.label(), e.atom_count());
}
// Output:
//   Protein A: 660 atoms
//   Water (58 molecules): 58 atoms
```

## Auto-detect format by extension

`structure_file_to_entities` dispatches on `.pdb`/`.ent` vs mmCIF:

```rust,ignore
use molex::adapters::pdb::structure_file_to_entities;
use std::path::Path;

let entities = structure_file_to_entities(Path::new("3nez.cif"))?;
```

## Work with entities

```rust,ignore
use molex::{MoleculeEntity, MoleculeType};

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

DSSP and backbone H-bond detection are run automatically when you build an `Assembly`:

```rust,ignore
use molex::{Assembly, ConnectionType, SSType};

let assembly = Assembly::new(entities);
let protein_id = assembly.entities()[0].id();
for (i, ss) in assembly.ss_types(protein_id).iter().enumerate() {
    println!("Residue {}: {:?}", i, ss); // Helix, Sheet, or Coil
}
// Backbone H-bonds are computed on demand for rendering, not stored.
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

Pass an `Assembly` directly via `serialize_assembly(&assembly)` when the
derived-data pipeline has already been run.

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
