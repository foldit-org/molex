// foldit:allow-long-file: the AtomTable interchange plus the one
// into_entities/from_entities grouping core and the shared flat-atom walk.
//! Native, format-neutral, key-bearing flat atom interchange.
//!
//! `AtomTable` is the one columnar table the partition and concat cores run
//! over: the six `Atom` property columns plus the per-atom grouping keys
//! (`entity_id`/`chain_id`/`mol_type`/`res_id`/`res_name`) and the
//! `observed_mask` provenance bit. `into_entities` is the single partition
//! (flat rows -> entity hierarchy); `from_entities` is its inverse (entity
//! hierarchy -> flat rows).
//!
//! Trusted, already-structured columnar input (the wire decoder, the Python
//! columnar interchange) builds an `AtomTable` and calls `into_entities`.
//! Raw parser input does not go through here — it stays on `EntityBuilder`,
//! which owns altloc dedup, modified-residue merge, and completion.

use compact_str::CompactString;
use glam::Vec3;

use crate::element::Element;
use crate::entity::molecule::atom::{Atom, AtomColumns, AtomRef};
use crate::entity::molecule::bulk::BulkEntity;
use crate::entity::molecule::id::{EntityId, EntityIdAllocator};
use crate::entity::molecule::nucleic_acid::NAEntity;
use crate::entity::molecule::polymer::Residue;
use crate::entity::molecule::protein::ProteinEntity;
use crate::entity::molecule::small_molecule::SmallMoleculeEntity;
use crate::entity::molecule::{MoleculeEntity, MoleculeType};

/// A flat columnar atom table: payload columns plus the per-atom grouping
/// keys the partition needs.
///
/// Every column is one entry per atom and the same length ([`Self::len`]).
/// `position`/`occupancy`/`b_factor`/`element`/`name`/`formal_charge` mirror
/// [`Atom`]'s fields; `entity_id`/`chain_id`/`mol_type`/`res_id`/`res_name`
/// are the keys [`Self::into_entities`] splits on; `observed_mask` carries
/// each atom's parsed-vs-fabricated provenance into the rebuilt entity.
#[derive(Debug, Clone, Default)]
pub struct AtomTable {
    /// 3D positions in angstroms.
    pub position: Vec<Vec3>,
    /// Crystallographic occupancies (0.0 to 1.0).
    pub occupancy: Vec<f32>,
    /// Temperature factors (B-factor) in square angstroms.
    pub b_factor: Vec<f32>,
    /// Chemical elements.
    pub element: Vec<Element>,
    /// PDB-style 4-character atom names.
    pub name: Vec<[u8; 4]>,
    /// Formal charges (signed). 0 means neutral.
    pub formal_charge: Vec<i8>,
    /// Per-atom entity id. The entity run boundary.
    pub entity_id: Vec<u32>,
    /// Per-atom chain id (`label_asym_id`). Empty for non-polymer entities.
    pub chain_id: Vec<CompactString>,
    /// Per-atom molecule type. The dispatch axis.
    pub mol_type: Vec<MoleculeType>,
    /// Per-atom residue sequence id. The residue run boundary within an
    /// entity.
    pub res_id: Vec<i32>,
    /// Per-atom residue name, 3 chars, space-padded.
    pub res_name: Vec<[u8; 3]>,
    /// Per-atom provenance: `true` parsed, `false` fabricated by completion.
    pub observed_mask: Vec<bool>,
}

impl AtomTable {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A table with each column preallocated for `n` atoms.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            position: Vec::with_capacity(n),
            occupancy: Vec::with_capacity(n),
            b_factor: Vec::with_capacity(n),
            element: Vec::with_capacity(n),
            name: Vec::with_capacity(n),
            formal_charge: Vec::with_capacity(n),
            entity_id: Vec::with_capacity(n),
            chain_id: Vec::with_capacity(n),
            mol_type: Vec::with_capacity(n),
            res_id: Vec::with_capacity(n),
            res_name: Vec::with_capacity(n),
            observed_mask: Vec::with_capacity(n),
        }
    }

    /// Number of atoms (the shared column length).
    #[must_use]
    pub fn len(&self) -> usize {
        self.position.len()
    }

    /// Whether the table holds no atoms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.position.is_empty()
    }

    /// Push one atom row: its [`Atom`] payload plus the grouping keys it
    /// belongs to. Every column advances by one, preserving the shared
    /// length invariant.
    #[allow(
        clippy::too_many_arguments,
        reason = "one parameter per grouping key column; bundling into a \
                  struct just to satisfy the lint trades clarity for noise"
    )]
    pub fn push_row(
        &mut self,
        atom: &Atom,
        entity_id: u32,
        chain_id: CompactString,
        mol_type: MoleculeType,
        res_id: i32,
        res_name: [u8; 3],
    ) {
        self.position.push(atom.position);
        self.occupancy.push(atom.occupancy);
        self.b_factor.push(atom.b_factor);
        self.element.push(atom.element);
        self.name.push(atom.name);
        self.formal_charge.push(atom.formal_charge);
        self.entity_id.push(entity_id);
        self.chain_id.push(chain_id);
        self.mol_type.push(mol_type);
        self.res_id.push(res_id);
        self.res_name.push(res_name);
        self.observed_mask.push(atom.observed);
    }

    /// Gather row `i`'s payload back into a by-value [`Atom`], stamping
    /// `observed` from the mask.
    fn atom(&self, i: usize) -> Atom {
        Atom {
            position: self.position[i],
            occupancy: self.occupancy[i],
            b_factor: self.b_factor[i],
            element: self.element[i],
            name: self.name[i],
            formal_charge: self.formal_charge[i],
            observed: self.observed_mask[i],
        }
    }

    /// Partition the flat table into the entity hierarchy.
    ///
    /// Rows are grouped into entities on each `entity_id` change (rows of one
    /// entity must be contiguous), and each entity is dispatched by its first
    /// row's `mol_type` to the matching constructor. The constructors re-run
    /// their canonicalize pass (reorder + drop-backbone-incomplete-to-tail),
    /// so this never reorders or drops on its own; it only feeds the atoms in
    /// row order with `observed` carried on each [`Atom`].
    ///
    /// Polymer entities (`Protein`/`DNA`/`RNA`) split their run into residues
    /// on each `res_id` change, recording contiguous `atom_range`s, and take
    /// their `pdb_chain_id` from the first row. Non-polymer entities
    /// (`Ligand`/`Ion`/`Cofactor`/`Lipid` -> `SmallMolecule`,
    /// `Water`/`Solvent` -> `Bulk`) collect their atoms whole, taking the
    /// residue name from the first row; a `Bulk`'s `molecule_count` is the
    /// number of distinct `(chain_id, res_id)` pairs in its run.
    #[must_use]
    pub fn into_entities(self) -> Vec<MoleculeEntity> {
        let n = self.len();
        let mut entities = Vec::new();
        let mut allocator = EntityIdAllocator::new();

        let mut start = 0usize;
        while start < n {
            let eid = self.entity_id[start];
            let mut end = start + 1;
            while end < n && self.entity_id[end] == eid {
                end += 1;
            }
            entities
                .push(self.build_entity(allocator.from_raw(eid), start..end));
            start = end;
        }

        entities
    }

    /// Build one entity from the contiguous row range `[start, end)`, which
    /// must all share `entity_id`.
    fn build_entity(
        &self,
        id: EntityId,
        range: std::ops::Range<usize>,
    ) -> MoleculeEntity {
        let mol_type = self.mol_type[range.start];

        match mol_type {
            MoleculeType::Protein => {
                let (atoms, residues) = self.atoms_and_residues(range.clone());
                let chain_id = self.chain_id[range.start].to_string();
                MoleculeEntity::Protein(ProteinEntity::new(
                    id, atoms, residues, chain_id,
                ))
            }
            MoleculeType::DNA | MoleculeType::RNA => {
                let (atoms, residues) = self.atoms_and_residues(range.clone());
                let chain_id = self.chain_id[range.start].to_string();
                MoleculeEntity::NucleicAcid(NAEntity::new(
                    id, mol_type, atoms, residues, chain_id,
                ))
            }
            MoleculeType::Water | MoleculeType::Solvent => {
                let residue_name = self.res_name[range.start];
                let chain_id = self.chain_id[range.start].to_string();
                let molecule_count = self.distinct_chain_res(range.clone());
                let atoms = range.map(|i| self.atom(i)).collect();
                MoleculeEntity::Bulk(BulkEntity::new(
                    id,
                    mol_type,
                    atoms,
                    residue_name,
                    molecule_count,
                    chain_id,
                ))
            }
            MoleculeType::Ligand
            | MoleculeType::Ion
            | MoleculeType::Cofactor
            | MoleculeType::Lipid => {
                let residue_name = self.res_name[range.start];
                let chain_id = self.chain_id[range.start].to_string();
                let atoms = range.map(|i| self.atom(i)).collect();
                MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
                    id,
                    mol_type,
                    atoms,
                    residue_name,
                    chain_id,
                ))
            }
        }
    }

    /// Split a polymer entity's row range into a flat atom vec plus the
    /// residues indexing into it. A new residue starts on each `res_id`
    /// change; ranges are contiguous, gap-free, and cover every atom. Each
    /// `Atom` carries its `observed` bit so the constructor's canonicalize
    /// clone preserves it.
    fn atoms_and_residues(
        &self,
        range: std::ops::Range<usize>,
    ) -> (Vec<Atom>, Vec<Residue>) {
        let mut atoms = Vec::with_capacity(range.len());
        let mut residues = Vec::new();
        let mut current_res_id: Option<i32> = None;
        let mut current_res_name: [u8; 3] = [b' '; 3];
        let mut current_start = 0usize;

        for i in range {
            let new_residue =
                current_res_id.is_none_or(|r| r != self.res_id[i]);
            if new_residue {
                if let Some(r) = current_res_id {
                    residues.push(Residue {
                        name: current_res_name,
                        label_seq_id: r,
                        auth_seq_id: None,
                        auth_comp_id: None,
                        ins_code: None,
                        atom_range: current_start..atoms.len(),
                        variants: Vec::new(),
                    });
                }
                current_start = atoms.len();
                current_res_id = Some(self.res_id[i]);
                current_res_name = self.res_name[i];
            }
            atoms.push(self.atom(i));
        }
        if let Some(r) = current_res_id {
            residues.push(Residue {
                name: current_res_name,
                label_seq_id: r,
                auth_seq_id: None,
                auth_comp_id: None,
                ins_code: None,
                atom_range: current_start..atoms.len(),
                variants: Vec::new(),
            });
        }

        (atoms, residues)
    }

    /// Number of distinct `(chain_id, res_id)` pairs in `range` — a bulk
    /// entity's molecule count (one molecule per residue per chain).
    fn distinct_chain_res(&self, range: std::ops::Range<usize>) -> usize {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for i in range {
            let _ = seen.insert((self.chain_id[i].clone(), self.res_id[i]));
        }
        seen.len()
    }

    /// Concatenate the entity hierarchy back into a flat table — the inverse
    /// of [`Self::into_entities`].
    ///
    /// Walks entities in slice order via `walk_flat_atoms`, re-expanding the
    /// hierarchy keys to per-atom columns: polymers carry their real chain id
    /// and each residue's `(label_seq_id, name)`, walked in residue order over
    /// each `atom_range` (an atom referenced by no range is skipped); a
    /// `SmallMolecule` carries its chain id, `res_id` 1, and its residue
    /// name; a `Bulk` carries its chain id, a per-atom `res_id` of
    /// `raw_idx + 1`, and its residue name.
    #[must_use]
    pub fn from_entities<E: std::borrow::Borrow<MoleculeEntity>>(
        entities: &[E],
    ) -> Self {
        let total: usize =
            entities.iter().map(|e| e.borrow().atom_count()).sum();
        let mut table = Self::with_capacity(total);

        walk_flat_atoms(entities, |row| {
            table.push_row(
                &row.columns.gather(row.raw_idx as usize),
                row.entity_raw_id,
                CompactString::new(row.chain_id),
                row.mol_type,
                row.res_id,
                row.res_name,
            );
        });

        table
    }

    /// Visit every atom of `entities` in the canonical flat order, handing the
    /// consumer the borrowed [`AtomRef`] plus the residue keys
    /// (`res_name`, `res_id`) projected onto it — no column materialization.
    ///
    /// Driven by the same `walk_flat_atoms` walk as [`Self::from_entities`] and
    /// [`Self::flat_source_indices`], so a byte-writer over this and a rebuilt
    /// table never disagree on atom order or key synthesis.
    pub(crate) fn for_each_flat_row<E, F>(entities: &[E], mut f: F)
    where
        E: std::borrow::Borrow<MoleculeEntity>,
        F: FnMut(AtomRef<'_>, [u8; 3], i32),
    {
        walk_flat_atoms(entities, |row| {
            f(
                row.columns.atom_ref(row.raw_idx as usize),
                row.res_name,
                row.res_id,
            );
        });
    }

    /// Number of atom rows the canonical flat walk emits for one entity — the
    /// exact body-row count [`Self::for_each_flat_row`] yields, driven by the
    /// same `walk_flat_atoms`. For a polymer this is Σ residue `atom_range`
    /// lengths (atoms of dropped, backbone-incomplete residues sit in the
    /// storage columns but are referenced by no range, so they are not
    /// counted); for a non-polymer it is `columns().len()`.
    #[must_use]
    pub(crate) fn flat_atom_count(entity: &MoleculeEntity) -> usize {
        let mut count = 0usize;
        walk_flat_atoms(std::slice::from_ref(entity), |_row| count += 1);
        count
    }

    /// The per-atom source coordinates `(entity_raw_id, raw_idx)` in the same
    /// canonical flat order [`Self::from_entities`] lays atoms out, where
    /// `raw_idx` is the atom's index within its entity's storage columns.
    ///
    /// Consumers that need to land per-entity-addressed data (e.g. bond
    /// endpoints reported as `(entity, storage index)`) into flat atom index
    /// space use this; the flat position is the vec index.
    /// Driven by the same `walk_flat_atoms` walk as `from_entities`, so the
    /// two never disagree on ordering.
    #[must_use]
    pub fn flat_source_indices<E: std::borrow::Borrow<MoleculeEntity>>(
        entities: &[E],
    ) -> Vec<(u32, u32)> {
        let total: usize =
            entities.iter().map(|e| e.borrow().atom_count()).sum();
        let mut out = Vec::with_capacity(total);
        walk_flat_atoms(entities, |row| {
            out.push((row.entity_raw_id, row.raw_idx));
        });
        out
    }
}

/// Build an empty entity of `mol_type` — the zero-atom degenerate case of
/// [`AtomTable::into_entities`]'s partition, sharing the exact four-arm
/// `mol_type` -> variant dispatch [`AtomTable::build_entity`] uses, with empty
/// atoms and residues.
///
/// `into_entities` yields nothing for a zero-row run, so the trusted columnar
/// decoders (wire headers) that synthesize entities one at a time fall back to
/// this when a group decodes to no atoms — an
/// unreachable-in-practice path (groups always carry at least one atom) that
/// keeps the entity count aligned with the input group count. Polymers carry
/// `chain_id`; non-polymers take a blank residue name and (for `Bulk`) a
/// molecule count of 0.
pub(crate) fn empty_entity(
    id: EntityId,
    mol_type: MoleculeType,
    chain_id: String,
) -> MoleculeEntity {
    let res_name = [b' '; 3];
    match mol_type {
        MoleculeType::Protein => MoleculeEntity::Protein(ProteinEntity::new(
            id,
            Vec::new(),
            Vec::new(),
            chain_id,
        )),
        MoleculeType::DNA | MoleculeType::RNA => MoleculeEntity::NucleicAcid(
            NAEntity::new(id, mol_type, Vec::new(), Vec::new(), chain_id),
        ),
        MoleculeType::Water | MoleculeType::Solvent => MoleculeEntity::Bulk(
            BulkEntity::new(id, mol_type, Vec::new(), res_name, 0, chain_id),
        ),
        MoleculeType::Ligand
        | MoleculeType::Ion
        | MoleculeType::Cofactor
        | MoleculeType::Lipid => {
            MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
                id,
                mol_type,
                Vec::new(),
                res_name,
                chain_id,
            ))
        }
    }
}

/// One atom as visited by [`walk_flat_atoms`]: a borrow of its entity's
/// storage columns and its index within them, plus the residue/chain keys to
/// stamp onto it.
///
/// `entity_raw_id` + `raw_idx` identify the atom within its entity's storage
/// (the `raw_idx` is the storage-column index); the flat index is just the
/// visit order. `chain_id`/`res_name`/`res_id`/`mol_type` are the
/// hierarchy-projected keys (a non-polymer atom carries the entity's residue
/// name, the entity's chain, and a synthetic residue number).
struct FlatRow<'a> {
    entity_raw_id: u32,
    raw_idx: u32,
    columns: &'a AtomColumns,
    chain_id: &'a str,
    mol_type: MoleculeType,
    res_id: i32,
    res_name: [u8; 3],
}

/// Visit every atom of `entities` in the canonical flat / wire order:
/// entities in slice order; within a polymer, atoms in residue order via each
/// residue's `atom_range` (an atom referenced by no range is skipped); within
/// a non-polymer, raw storage order. The one place this ordering and the
/// non-polymer key synthesis live.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "atom counts fit in i32/u32 for valid structures"
)]
fn walk_flat_atoms<E: std::borrow::Borrow<MoleculeEntity>>(
    entities: &[E],
    mut f: impl FnMut(FlatRow<'_>),
) {
    for entity in entities {
        let entity = entity.borrow();
        let entity_raw_id = entity.id().raw();
        let mol_type = entity.molecule_type();
        let columns = entity.columns();

        match entity {
            MoleculeEntity::Protein(e) => walk_polymer(
                &mut f,
                entity_raw_id,
                mol_type,
                &e.residues,
                columns,
                &e.pdb_chain_id,
            ),
            MoleculeEntity::NucleicAcid(e) => walk_polymer(
                &mut f,
                entity_raw_id,
                mol_type,
                &e.residues,
                columns,
                &e.pdb_chain_id,
            ),
            MoleculeEntity::SmallMolecule(e) => {
                for raw_idx in 0..columns.len() {
                    f(FlatRow {
                        entity_raw_id,
                        raw_idx: raw_idx as u32,
                        columns,
                        chain_id: &e.pdb_chain_id,
                        mol_type,
                        res_id: 1,
                        res_name: e.residue_name,
                    });
                }
            }
            MoleculeEntity::Bulk(e) => {
                for raw_idx in 0..columns.len() {
                    f(FlatRow {
                        entity_raw_id,
                        raw_idx: raw_idx as u32,
                        columns,
                        chain_id: &e.pdb_chain_id,
                        mol_type,
                        res_id: (raw_idx as i32) + 1,
                        res_name: e.residue_name,
                    });
                }
            }
        }
    }
}

/// Walk one polymer entity's residue-ordered atoms, projecting each residue's
/// `(label_seq_id, name)` and the entity's chain id onto every atom.
#[allow(
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    reason = "atom indices fit in u32; the key set is the polymer projection"
)]
fn walk_polymer<'a>(
    f: &mut impl FnMut(FlatRow<'_>),
    entity_raw_id: u32,
    mol_type: MoleculeType,
    residues: &'a [Residue],
    columns: &'a AtomColumns,
    chain_id: &'a str,
) {
    for residue in residues {
        for raw_idx in residue.atom_range.clone() {
            f(FlatRow {
                entity_raw_id,
                raw_idx: raw_idx as u32,
                columns,
                chain_id,
                mol_type,
                res_id: residue.label_seq_id,
                res_name: residue.name,
            });
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "tests: coords are exact f32 copies, no arithmetic"
)]
mod tests {
    use super::*;
    use crate::entity::molecule::traits::Entity;

    fn mk_atom(name: [u8; 4], element: Element, pos: Vec3) -> Atom {
        Atom {
            position: pos,
            occupancy: 1.0,
            b_factor: 0.0,
            element,
            name,
            formal_charge: 0,
            observed: true,
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "test helper mirrors push_row's key set"
    )]
    fn push_protein_row(
        table: &mut AtomTable,
        eid: u32,
        chain: &str,
        res_name: [u8; 3],
        res_id: i32,
        atom: &Atom,
    ) {
        table.push_row(
            atom,
            eid,
            CompactString::new(chain),
            MoleculeType::Protein,
            res_id,
            res_name,
        );
    }

    #[test]
    fn into_entities_empty_yields_nothing() {
        assert!(AtomTable::new().into_entities().is_empty());
    }

    #[test]
    fn into_entities_groups_residues_by_res_id() {
        // Three atoms in residue 1 (ALA), two in residue 2 (GLY); the split
        // must land on the res_id change.
        let mut table = AtomTable::with_capacity(5);
        push_protein_row(
            &mut table,
            0,
            "A",
            *b"ALA",
            1,
            &mk_atom(*b"N   ", Element::N, Vec3::ZERO),
        );
        push_protein_row(
            &mut table,
            0,
            "A",
            *b"ALA",
            1,
            &mk_atom(*b"CA  ", Element::C, Vec3::X),
        );
        push_protein_row(
            &mut table,
            0,
            "A",
            *b"ALA",
            1,
            &mk_atom(*b"C   ", Element::C, Vec3::Y),
        );
        push_protein_row(
            &mut table,
            0,
            "A",
            *b"GLY",
            2,
            &mk_atom(*b"N   ", Element::N, Vec3::Z),
        );
        push_protein_row(
            &mut table,
            0,
            "A",
            *b"GLY",
            2,
            &mk_atom(*b"CA  ", Element::C, Vec3::ONE),
        );

        let (atoms, residues) = table.atoms_and_residues(0..5);
        assert_eq!(atoms.len(), 5);
        assert_eq!(residues.len(), 2);
        assert_eq!(residues[0].name, *b"ALA");
        assert_eq!(residues[0].label_seq_id, 1);
        assert_eq!(residues[0].atom_range, 0..3);
        assert_eq!(residues[1].name, *b"GLY");
        assert_eq!(residues[1].atom_range, 3..5);
    }

    #[test]
    fn into_entities_splits_on_entity_id_change() {
        let mut table = AtomTable::with_capacity(2);
        // Entity 0: a one-atom ligand. Entity 1: a one-atom ion.
        table.push_row(
            &mk_atom(*b"C1  ", Element::C, Vec3::ZERO),
            0,
            CompactString::default(),
            MoleculeType::Ligand,
            1,
            *b"LIG",
        );
        table.push_row(
            &mk_atom(*b"ZN  ", Element::Zn, Vec3::X),
            1,
            CompactString::default(),
            MoleculeType::Ion,
            1,
            *b"ZN ",
        );

        let entities = table.into_entities();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].molecule_type(), MoleculeType::Ligand);
        assert_eq!(entities[0].id(), EntityId::from_raw(0));
        assert_eq!(entities[1].molecule_type(), MoleculeType::Ion);
        assert_eq!(entities[1].id(), EntityId::from_raw(1));
    }

    #[test]
    fn water_molecule_count_uses_chain_res_pair() {
        // Two water atoms on chain "A" sharing res_id 1 are ONE molecule;
        // a third on chain "A" res_id 2 is a second. The pair-key counts 2.
        let mut table = AtomTable::with_capacity(3);
        for (res_id, x) in [(1, 0.0_f32), (1, 1.0), (2, 2.0)] {
            table.push_row(
                &mk_atom(*b"O   ", Element::O, Vec3::new(x, 0.0, 0.0)),
                0,
                CompactString::new("A"),
                MoleculeType::Water,
                res_id,
                *b"HOH",
            );
        }
        let entities = table.into_entities();
        assert_eq!(entities.len(), 1);
        let bulk = entities[0].as_bulk().unwrap();
        assert_eq!(bulk.molecule_count, 2);
        assert_eq!(bulk.columns().len(), 3);
    }

    #[test]
    fn observed_bit_flows_through_into_entities() {
        // A backbone-only residue whose CA was fabricated (observed = false).
        // The canonicalize clone must carry the bit onto the rebuilt entity.
        let mut table = AtomTable::with_capacity(4);
        let mut ca = mk_atom(*b"CA  ", Element::C, Vec3::X);
        ca.observed = false;
        push_protein_row(
            &mut table,
            0,
            "A",
            *b"ALA",
            1,
            &mk_atom(*b"N   ", Element::N, Vec3::ZERO),
        );
        push_protein_row(&mut table, 0, "A", *b"ALA", 1, &ca);
        push_protein_row(
            &mut table,
            0,
            "A",
            *b"ALA",
            1,
            &mk_atom(*b"C   ", Element::C, Vec3::Y),
        );
        push_protein_row(
            &mut table,
            0,
            "A",
            *b"ALA",
            1,
            &mk_atom(*b"O   ", Element::O, Vec3::Z),
        );

        let entities = table.into_entities();
        let columns = entities[0].columns();
        // Canonical order is N, CA, C, O; the CA (index 1) stays unobserved.
        let observed: Vec<bool> = columns.observed.clone();
        assert_eq!(observed, vec![true, false, true, true]);
    }

    /// The bijection contract: `into_entities(from_entities(&e)) == e` for
    /// already-canonical entities. Canonicalize is idempotent on canonical
    /// input, so the round-trip is an identity on the kept atoms.
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one fixture entity per variant plus per-column assertions"
    )]
    fn into_from_entities_bijection() {
        use crate::entity::molecule::id::EntityIdAllocator;

        let mut alloc = EntityIdAllocator::new();

        // A two-residue protein laid out in canonical N, CA, C, O order so
        // ProteinEntity::new neither reorders nor drops it.
        let protein_atoms = vec![
            mk_atom(*b"N   ", Element::N, Vec3::new(0.0, 0.0, 0.0)),
            mk_atom(*b"CA  ", Element::C, Vec3::new(1.0, 0.0, 0.0)),
            mk_atom(*b"C   ", Element::C, Vec3::new(2.0, 0.0, 0.0)),
            mk_atom(*b"O   ", Element::O, Vec3::new(2.0, 1.0, 0.0)),
            mk_atom(*b"N   ", Element::N, Vec3::new(3.2, 0.0, 0.0)),
            mk_atom(*b"CA  ", Element::C, Vec3::new(4.2, 0.0, 0.0)),
            mk_atom(*b"C   ", Element::C, Vec3::new(5.2, 0.0, 0.0)),
            mk_atom(*b"O   ", Element::O, Vec3::new(5.2, 1.0, 0.0)),
        ];
        let protein_residues = vec![
            Residue {
                name: *b"UNK",
                label_seq_id: 1,
                auth_seq_id: None,
                auth_comp_id: None,
                ins_code: None,
                atom_range: 0..4,
                variants: Vec::new(),
            },
            Residue {
                name: *b"UNK",
                label_seq_id: 2,
                auth_seq_id: None,
                auth_comp_id: None,
                ins_code: None,
                atom_range: 4..8,
                variants: Vec::new(),
            },
        ];
        let protein = MoleculeEntity::Protein(ProteinEntity::new(
            alloc.allocate(),
            protein_atoms,
            protein_residues,
            "A".to_owned(),
        ));

        let ligand = MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
            alloc.allocate(),
            MoleculeType::Ligand,
            vec![
                mk_atom(*b"C1  ", Element::C, Vec3::new(10.0, 0.0, 0.0)),
                mk_atom(*b"O1  ", Element::O, Vec3::new(11.0, 0.0, 0.0)),
            ],
            *b"LIG",
            String::from("B"),
        ));

        // A bulk water with molecule_count == atom_count, the round-trippable
        // case (from_entities synthesizes a distinct res_id per atom). Its
        // chain differs from the protein's so the round-trip exercises
        // non-polymer chain preservation.
        let water = MoleculeEntity::Bulk(BulkEntity::new(
            alloc.allocate(),
            MoleculeType::Water,
            vec![
                mk_atom(*b"O   ", Element::O, Vec3::new(20.0, 0.0, 0.0)),
                mk_atom(*b"O   ", Element::O, Vec3::new(21.0, 0.0, 0.0)),
            ],
            *b"HOH",
            2,
            String::from("W"),
        ));

        let entities = vec![protein, ligand, water];

        let rebuilt = AtomTable::from_entities(&entities).into_entities();

        assert_eq!(rebuilt.len(), entities.len());
        for (orig, rt) in entities.iter().zip(&rebuilt) {
            assert_eq!(orig.id(), rt.id());
            assert_eq!(orig.molecule_type(), rt.molecule_type());
            assert_eq!(orig.atom_count(), rt.atom_count());
            assert_eq!(orig.pdb_chain_id(), rt.pdb_chain_id());

            let oc = orig.columns();
            let rc = rt.columns();
            assert_eq!(oc.position, rc.position);
            assert_eq!(oc.element, rc.element);
            assert_eq!(oc.name, rc.name);
            assert_eq!(oc.occupancy, rc.occupancy);
            assert_eq!(oc.b_factor, rc.b_factor);
            assert_eq!(oc.formal_charge, rc.formal_charge);
            assert_eq!(oc.observed, rc.observed);

            assert_eq!(
                orig.residues().is_some(),
                rt.residues().is_some(),
                "residue presence must match"
            );
            if let (Some(or), Some(rr)) = (orig.residues(), rt.residues()) {
                assert_eq!(or.len(), rr.len());
                for (a, b) in or.iter().zip(rr) {
                    assert_eq!(a.name, b.name);
                    assert_eq!(a.label_seq_id, b.label_seq_id);
                    assert_eq!(a.atom_range, b.atom_range);
                }
            }
        }

        let water_orig = entities[2].as_bulk().unwrap();
        let water_rt = rebuilt[2].as_bulk().unwrap();
        assert_eq!(water_orig.molecule_count, water_rt.molecule_count);
    }
}
