//! Spatial contact enumeration over an assembly's atoms.
//!
//! Both queries flatten every entity's atoms into one position array and bin
//! them with a single `SpatialGrid`, so neighbor lookups stay near-linear
//! instead of all-pairs. The grid cell is sized to the query radius/cutoff so
//! every in-range partner falls within the 27-cell candidate stencil.

use std::collections::HashSet;

use glam::Vec3;

use crate::analysis::grid::SpatialGrid;
use crate::assembly::Assembly;
use crate::atom_id::AtomId;
use crate::entity::EntityId;

/// Smallest grid cell, guarding against a zero/negative radius collapsing the
/// bin size.
const MIN_CELL: f32 = 1e-3;

/// Granularity at which [`Assembly::contacts`] reports pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContactLevel {
    /// One pair per atom-atom contact.
    Atom,
    /// One pair per unique (entity, residue-index) pair in contact.
    Residue,
    /// One pair per unique entity-entity pair in contact.
    Chain,
}

/// A contact between two atoms. For [`ContactLevel::Residue`] and
/// [`ContactLevel::Chain`] the endpoints are one representative atom pair for
/// the deduplicated residue/chain pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Contact {
    /// First endpoint (the lower flat index of the pair).
    pub a: AtomId,
    /// Second endpoint (the higher flat index of the pair).
    pub b: AtomId,
}

/// Residue identity for dedup: the owning entity plus the residue index within
/// it (`None` for non-polymer atoms or atoms outside every `atom_range`).
type ResidueKey = (EntityId, Option<usize>);

/// All atoms of an assembly flattened into parallel arrays, walking entities in
/// declaration order. Index `i` is the "flat index" used to emit each unordered
/// pair once (`i < j`).
struct FlatAtoms {
    position: Vec<Vec3>,
    id: Vec<AtomId>,
}

impl FlatAtoms {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "atom indices are bounded by entity size (< u32::MAX)"
    )]
    fn build(assembly: &Assembly) -> Self {
        let mut flat = Self {
            position: Vec::new(),
            id: Vec::new(),
        };
        for entity in assembly.entities() {
            let entity_id = entity.id();
            for (index, &pos) in entity.positions().iter().enumerate() {
                flat.position.push(pos);
                flat.id.push(AtomId {
                    entity: entity_id,
                    index: index as u32,
                });
            }
        }
        flat
    }

    fn len(&self) -> usize {
        self.position.len()
    }

    fn is_empty(&self) -> bool {
        self.position.is_empty()
    }
}

impl Assembly {
    /// Every atom within `radius` of `center`, as stable [`AtomId`]s.
    ///
    /// Atoms are flattened across all entities and binned in one uniform grid;
    /// only the 27 cells around `center` are scanned, then filtered by exact
    /// distance. Order follows entity declaration then in-entity atom index.
    #[must_use]
    pub fn neighbor_search(&self, center: Vec3, radius: f32) -> Vec<AtomId> {
        let flat = FlatAtoms::build(self);
        if flat.is_empty() {
            return Vec::new();
        }
        let grid = SpatialGrid::build(&flat.position, radius.max(MIN_CELL));
        let radius_sq = radius * radius;
        let mut out = Vec::new();
        grid.for_each_candidate(center, |j| {
            if center.distance_squared(flat.position[j]) <= radius_sq {
                out.push(flat.id[j]);
            }
        });
        out
    }

    /// All atom pairs closer than `cutoff`, reported at the requested `level`.
    ///
    /// At [`ContactLevel::Atom`] each in-range atom-atom pair is emitted once
    /// (the lower flat index first). [`ContactLevel::Residue`] dedups those to
    /// unique (entity, residue-index) pairs and [`ContactLevel::Chain`] to
    /// unique entity-entity pairs, each keeping one representative atom pair.
    /// With `between_chains` set, only pairs whose endpoints live in different
    /// entities are kept.
    #[must_use]
    pub fn contacts(
        &self,
        cutoff: f32,
        level: ContactLevel,
        between_chains: bool,
    ) -> Vec<Contact> {
        let flat = FlatAtoms::build(self);
        if flat.is_empty() {
            return Vec::new();
        }
        let grid = SpatialGrid::build(&flat.position, cutoff.max(MIN_CELL));
        let cutoff_sq = cutoff * cutoff;
        let atom_pairs = atom_pairs(&flat, &grid, cutoff_sq, between_chains);
        match level {
            ContactLevel::Atom => atom_pairs,
            ContactLevel::Residue => self.dedup_residue(&atom_pairs),
            ContactLevel::Chain => dedup_chain(&atom_pairs),
        }
    }

    /// Collapse atom contacts to one representative per unique residue pair. An
    /// atom's residue is the residue whose `atom_range` contains its in-entity
    /// index; non-polymer atoms (and any outside every range) key on `None`,
    /// collapsing per entity.
    fn dedup_residue(&self, atom_pairs: &[Contact]) -> Vec<Contact> {
        let mut seen: HashSet<(ResidueKey, ResidueKey)> = HashSet::new();
        let mut out = Vec::new();
        for c in atom_pairs {
            let ra = self.residue_key(c.a);
            let rb = self.residue_key(c.b);
            if ra == rb {
                continue;
            }
            let key = if ra <= rb { (ra, rb) } else { (rb, ra) };
            if seen.insert(key) {
                out.push(*c);
            }
        }
        out
    }

    /// Residue key for `id`: its entity plus the residue index whose
    /// `atom_range` contains `id.index`, or `None` outside every range.
    fn residue_key(&self, id: AtomId) -> ResidueKey {
        let res = self.entity(id.entity).and_then(|e| {
            let idx = id.index as usize;
            e.residues()?
                .iter()
                .position(|r| r.atom_range.contains(&idx))
        });
        (id.entity, res)
    }
}

/// Core atom-atom contacts: each unordered in-range pair once, keyed on the
/// flat index (`i < j`) so no pair is double-counted.
fn atom_pairs(
    flat: &FlatAtoms,
    grid: &SpatialGrid,
    cutoff_sq: f32,
    between_chains: bool,
) -> Vec<Contact> {
    let mut out = Vec::new();
    for i in 0..flat.len() {
        grid.for_each_candidate(flat.position[i], |j| {
            if j <= i {
                return;
            }
            if between_chains && flat.id[i].entity == flat.id[j].entity {
                return;
            }
            if flat.position[i].distance_squared(flat.position[j]) < cutoff_sq {
                out.push(Contact {
                    a: flat.id[i],
                    b: flat.id[j],
                });
            }
        });
    }
    out
}

/// Collapse atom contacts to one representative per unique entity-entity pair.
fn dedup_chain(atom_pairs: &[Contact]) -> Vec<Contact> {
    let mut seen: HashSet<(EntityId, EntityId)> = HashSet::new();
    let mut out = Vec::new();
    for c in atom_pairs {
        let (ea, eb) = (c.a.entity, c.b.entity);
        if ea == eb {
            continue;
        }
        let key = if ea <= eb { (ea, eb) } else { (eb, ea) };
        if seen.insert(key) {
            out.push(*c);
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::element::Element;
    use crate::entity::molecule::atom::Atom;
    use crate::entity::molecule::id::EntityIdAllocator;
    use crate::entity::molecule::protein::ProteinEntity;
    use crate::entity::molecule::{MoleculeEntity, Residue};

    fn mk_atom(pos: Vec3) -> Atom {
        Atom {
            position: pos,
            occupancy: 1.0,
            b_factor: 0.0,
            element: Element::C,
            name: *b"C   ",
            formal_charge: 0,
            observed: true,
        }
    }

    /// One protein entity, `n` atoms strung along x at unit spacing, one
    /// residue spanning all of them.
    fn line_entity(
        alloc: &mut EntityIdAllocator,
        chain: &str,
        origin: Vec3,
        n: usize,
    ) -> MoleculeEntity {
        let atoms: Vec<Atom> = (0..n)
            .map(|i| mk_atom(origin + Vec3::new(i as f32, 0.0, 0.0)))
            .collect();
        let residue = Residue {
            name: *b"GLY",
            label_seq_id: 1,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: 0..atoms.len(),
            variants: Vec::new(),
        };
        let ent = ProteinEntity::new(
            alloc.allocate(),
            atoms,
            vec![residue],
            chain.to_owned(),
        );
        MoleculeEntity::Protein(ent)
    }

    #[test]
    fn neighbor_search_finds_close_atoms() {
        let mut alloc = EntityIdAllocator::new();
        let ent = line_entity(&mut alloc, "A", Vec3::ZERO, 5);
        let asm = Assembly::new(vec![ent]);

        // Centered on atom 2 (x=2) with radius 1.5 catches atoms 1, 2, 3.
        let hits = asm.neighbor_search(Vec3::new(2.0, 0.0, 0.0), 1.5);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn neighbor_search_empty_assembly() {
        let asm = Assembly::new(vec![]);
        assert!(asm.neighbor_search(Vec3::ZERO, 5.0).is_empty());
    }

    #[test]
    fn contacts_atom_pairs_unique() {
        let mut alloc = EntityIdAllocator::new();
        let ent = line_entity(&mut alloc, "A", Vec3::ZERO, 4);
        let asm = Assembly::new(vec![ent]);

        // Cutoff 1.5 links each adjacent pair (3 pairs) once.
        let pairs = asm.contacts(1.5, ContactLevel::Atom, false);
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn contacts_full_cover_is_n_choose_2() {
        let mut alloc = EntityIdAllocator::new();
        let n = 6;
        let ent = line_entity(&mut alloc, "A", Vec3::ZERO, n);
        let asm = Assembly::new(vec![ent]);

        // A cutoff past the whole span links every pair exactly once.
        let pairs = asm.contacts(1000.0, ContactLevel::Atom, false);
        assert_eq!(pairs.len(), n * (n - 1) / 2);
    }

    #[test]
    fn between_chains_keeps_only_cross_entity() {
        let mut alloc = EntityIdAllocator::new();
        // Two parallel lines 1 A apart in y; same x range so every A atom is
        // within cutoff of the matching B atom.
        let a = line_entity(&mut alloc, "A", Vec3::ZERO, 3);
        let b = line_entity(&mut alloc, "B", Vec3::new(0.0, 1.0, 0.0), 3);
        let asm = Assembly::new(vec![a, b]);

        let all = asm.contacts(1.5, ContactLevel::Atom, false);
        let cross = asm.contacts(1.5, ContactLevel::Atom, true);
        assert!(all.len() > cross.len());
        for c in &cross {
            assert_ne!(c.a.entity, c.b.entity);
        }
    }

    #[test]
    fn chain_level_dedups_to_entity_pair() {
        let mut alloc = EntityIdAllocator::new();
        let a = line_entity(&mut alloc, "A", Vec3::ZERO, 3);
        let b = line_entity(&mut alloc, "B", Vec3::new(0.0, 1.0, 0.0), 3);
        let asm = Assembly::new(vec![a, b]);

        let chains = asm.contacts(1.5, ContactLevel::Chain, true);
        assert_eq!(chains.len(), 1);
    }

    #[test]
    fn residue_level_dedups_to_residue_pair() {
        let mut alloc = EntityIdAllocator::new();
        // Two single-residue chains in contact dedup to one residue pair.
        let a = line_entity(&mut alloc, "A", Vec3::ZERO, 3);
        let b = line_entity(&mut alloc, "B", Vec3::new(0.0, 1.0, 0.0), 3);
        let asm = Assembly::new(vec![a, b]);

        let residues = asm.contacts(1.5, ContactLevel::Residue, true);
        assert_eq!(residues.len(), 1);
    }
}
