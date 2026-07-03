# Chemistry

`molex::chemistry` (source: `src/chemistry/`) owns the static residue
chemistry that entity construction and bond inference build on. The data is
pure: no entity references, no positions, no I/O.

## Tables

- **`amino_acids`** (`AminoAcid`): the standard amino acids plus
  modified-residue one-letter mapping. Provides the intra-residue bond
  topology used to populate `ProteinEntity::bonds`.
- **`nucleotides`** (`Nucleotide`): DNA/RNA bases and their bond topology.
- **`atom_name`** (`AtomName`, `is_protein_backbone_atom_name`): canonical
  4-character atom-name handling and the backbone-atom predicate.
- **`variant`** (`VariantTag`, `ProtonationState`): per-residue chemistry
  variants (terminus patches, disulfide, protonation states). Protonation is
  the load-bearing case: `HID` and `HIE` differ only in which nitrogen carries
  the hydrogen, which heavy-atom positions alone do not determine, so the tag
  is carried explicitly through the wire.

## Completion machinery

`completion` (`ResidueTemplate`, `TemplateAtom`) and the private
`completion_data` module hold the ideal-geometry templates that the
construction pass rigid-fits onto parsed residues to fabricate absent atoms.
See the [Completion](completion.md) page for how the levels use them.
