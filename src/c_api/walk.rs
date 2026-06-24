// foldit:allow-long-file: C-FFI tree accessors, one fn per ABI symbol.
//! C ABI walk accessors: assembly -> entity -> residue / atom.
//!
//! Split out of `c_api/mod.rs` for file-length reasons. All entries
//! here are `#[no_mangle] pub extern "C"` and so emit C symbols
//! identically to functions declared at the parent module scope.
//! cbindgen discovers them via the parent module's `pub mod walk;`.
//!
//! The per-atom walk (`molex_entity_atom` + the `molex_atom_*`
//! accessors) is the C-side equivalent of the Python binding's
//! columnar atom read (`Entity.to_arrays()` / `AtomArrays`): C consumers
//! iterate atom handles, Python consumers receive numpy columns. The two
//! atom-read shapes are intentionally different per ecosystem and are not
//! a symmetry gap to close. The entity-scalar and residue accessors below
//! mirror the Python `Entity` / `Residue` getters one-for-one, with one
//! intentional exception: `molex_entity_label` is not exposed. Every C
//! string accessor here returns a borrowed pointer into a stable byte
//! field, but `MoleculeEntity::label()` returns a computed display
//! `String` (for example "Ligand (ATP)"), which does not fit that
//! borrow-only convention. Python's `Entity.label` covers the need
//! natively; a C consumer that requires the label should add a
//! heap-allocated accessor freed via `molex_free_bytes` (the writer-entry
//! convention), not a borrowed one.

#![allow(
    clippy::missing_safety_doc,
    clippy::not_unsafe_ptr_arg_deref,
    reason = "Inherits the FFI-surface conventions documented at the top of \
              c_api/mod.rs."
)]

use std::sync::Arc;

use super::{
    assembly_inner, molex_Assembly, molex_Atom, molex_Entity, molex_EntityKind,
    molex_MoleculeType, molex_Residue,
};
use crate::assembly::Assembly;
use crate::element::Element;
use crate::entity::molecule::{Atom, MoleculeEntity, Residue};

// Assembly walk accessors

/// Monotonic generation counter; increments on every mutation.
///
/// Use this for cheap change detection on consumer side without
/// snapshotting the full atom set.
#[no_mangle]
pub extern "C" fn molex_assembly_generation(
    assembly: *const molex_Assembly,
) -> u64 {
    assembly_inner(assembly).map_or(0, Assembly::generation)
}

/// Number of entities in the assembly. Returns 0 if `assembly` is null.
#[no_mangle]
pub extern "C" fn molex_assembly_num_entities(
    assembly: *const molex_Assembly,
) -> usize {
    assembly_inner(assembly).map_or(0, |a| a.entities().len())
}

/// Borrow a non-owning view of the i-th entity. Returns null when
/// `assembly` is null or `i` is out of bounds.
#[no_mangle]
pub extern "C" fn molex_assembly_entity(
    assembly: *const molex_Assembly,
    i: usize,
) -> *const molex_Entity {
    let Some(a) = assembly_inner(assembly) else {
        return std::ptr::null();
    };
    a.entities().get(i).map_or(std::ptr::null(), |entity_arc| {
        let entity_ref: &MoleculeEntity = Arc::as_ref(entity_arc);
        std::ptr::from_ref(entity_ref).cast::<molex_Entity>()
    })
}

// Entity walk accessors

fn entity_inner<'a>(entity: *const molex_Entity) -> Option<&'a MoleculeEntity> {
    if entity.is_null() {
        return None;
    }
    Some(unsafe { &*entity.cast::<MoleculeEntity>() })
}

/// Raw entity id (`u32`). Returns 0 if `entity` is null.
#[no_mangle]
pub extern "C" fn molex_entity_id(entity: *const molex_Entity) -> u32 {
    entity_inner(entity).map_or(0, |e| e.id().raw())
}

/// Molecule type discriminant. Returns [`molex_MoleculeType::Solvent`]'s
/// integer value (8) as a placeholder if `entity` is null - callers
/// should null-check the handle first.
#[no_mangle]
pub extern "C" fn molex_entity_molecule_type(
    entity: *const molex_Entity,
) -> molex_MoleculeType {
    entity_inner(entity)
        .map_or(molex_MoleculeType::Solvent, |e| e.molecule_type().into())
}

/// Structural discriminant (the four-way variant taxonomy). Returns
/// [`molex_EntityKind::Bulk`]'s integer value (3) as a placeholder if
/// `entity` is null - callers should null-check the handle first.
#[no_mangle]
pub extern "C" fn molex_entity_kind(
    entity: *const molex_Entity,
) -> molex_EntityKind {
    entity_inner(entity)
        .map_or(molex_EntityKind::Bulk, |e| e.entity_kind().into())
}

/// First PDB chain-identifier byte for polymer entities.
///
/// Returns -1 when the entity has no chain id (small molecule / bulk),
/// when `entity` is null, or when the chain id is multi-character and so
/// cannot be represented as a single byte.
///
/// Chain ids are arbitrary strings (mmCIF `label_asym_id`); ribosome and
/// capsid assemblies use multi-character ids ("AA", "AB"). Use
/// [`molex_entity_chain_id`] for the full string. This single-byte
/// accessor is retained for ABI stability and returns the first byte only
/// when the id is exactly one byte.
#[no_mangle]
pub extern "C" fn molex_entity_pdb_chain_id(
    entity: *const molex_Entity,
) -> i32 {
    entity_inner(entity)
        .and_then(MoleculeEntity::pdb_chain_id)
        .and_then(|chain| {
            let bytes = chain.as_bytes();
            (bytes.len() == 1).then(|| i32::from(bytes[0]))
        })
        .unwrap_or(-1)
}

/// Pointer to this entity's full PDB chain-identifier UTF-8 bytes.
///
/// The chain id is the mmCIF `label_asym_id`. Writes the byte length to
/// `out_len` on success. Returns null and writes 0 for a non-polymer
/// entity (small molecule / bulk) or a null `entity`.
///
/// The pointer borrows the entity's owned chain string and is valid for
/// the entity's lifetime; the buffer is not NUL-terminated, so callers
/// must use `out_len`.
#[no_mangle]
pub extern "C" fn molex_entity_chain_id(
    entity: *const molex_Entity,
    out_len: *mut usize,
) -> *const u8 {
    let write_len = |len: usize| {
        if !out_len.is_null() {
            unsafe {
                *out_len = len;
            }
        }
    };
    let Some(chain) =
        entity_inner(entity).and_then(MoleculeEntity::pdb_chain_id)
    else {
        write_len(0);
        return std::ptr::null();
    };
    write_len(chain.len());
    chain.as_ptr()
}

/// Total atom count in this entity. Returns 0 if `entity` is null.
#[no_mangle]
pub extern "C" fn molex_entity_num_atoms(entity: *const molex_Entity) -> usize {
    entity_inner(entity).map_or(0, MoleculeEntity::atom_count)
}

/// Borrow a non-owning view of the i-th atom in this entity's flat atom
/// list. Returns null when `entity` is null or `i` is out of bounds.
#[no_mangle]
pub extern "C" fn molex_entity_atom(
    entity: *const molex_Entity,
    i: usize,
) -> *const molex_Atom {
    let Some(e) = entity_inner(entity) else {
        return std::ptr::null();
    };
    e.atom_set().get(i).map_or(std::ptr::null(), |atom: &Atom| {
        std::ptr::from_ref(atom).cast::<molex_Atom>()
    })
}

fn polymer_residues(entity: &MoleculeEntity) -> Option<&[Residue]> {
    match entity {
        MoleculeEntity::Protein(p) => Some(&p.residues),
        MoleculeEntity::NucleicAcid(n) => Some(&n.residues),
        MoleculeEntity::SmallMolecule(_) | MoleculeEntity::Bulk(_) => None,
    }
}

/// Pointer to the single 3-byte residue name carried by a non-polymer
/// entity (`SmallMolecule` / `Bulk`). Writes 3 to `out_len` on success;
/// returns null and writes 0 for polymers or a null `entity`.
///
/// The buffer is space-padded to 3 bytes; callers should strip trailing
/// ASCII spaces if needed.
#[no_mangle]
pub extern "C" fn molex_entity_residue_name_single(
    entity: *const molex_Entity,
    out_len: *mut usize,
) -> *const u8 {
    let write_len = |len: usize| {
        if !out_len.is_null() {
            unsafe {
                *out_len = len;
            }
        }
    };
    let Some(e) = entity_inner(entity) else {
        write_len(0);
        return std::ptr::null();
    };
    let name: &[u8; 3] = match e {
        MoleculeEntity::SmallMolecule(s) => &s.residue_name,
        MoleculeEntity::Bulk(b) => &b.residue_name,
        MoleculeEntity::Protein(_) | MoleculeEntity::NucleicAcid(_) => {
            write_len(0);
            return std::ptr::null();
        }
    };
    write_len(name.len());
    name.as_ptr()
}

/// Number of equal-sized molecule chunks the atom set should be split into.
///
/// For non-polymer entities only: returns 1 for `SmallMolecule`,
/// `BulkEntity::molecule_count` for `Bulk`, and 0 for polymers or a null
/// `entity`.
#[no_mangle]
pub extern "C" fn molex_entity_molecule_count(
    entity: *const molex_Entity,
) -> usize {
    let Some(e) = entity_inner(entity) else {
        return 0;
    };
    match e {
        MoleculeEntity::SmallMolecule(_) => 1,
        MoleculeEntity::Bulk(b) => b.molecule_count,
        MoleculeEntity::Protein(_) | MoleculeEntity::NucleicAcid(_) => 0,
    }
}

/// Number of indexable residues in this entity.
///
/// Returns the residue count for protein and nucleic acid entities; 0 for
/// small-molecule and bulk entities (which do not expose individual
/// residue records). Returns 0 if `entity` is null.
#[no_mangle]
pub extern "C" fn molex_entity_num_residues(
    entity: *const molex_Entity,
) -> usize {
    entity_inner(entity)
        .and_then(polymer_residues)
        .map_or(0, <[Residue]>::len)
}

/// Borrow a non-owning view of the i-th residue in this entity.
///
/// Returns null when `entity` is null, the entity is not a polymer, or
/// `i` is out of bounds.
#[no_mangle]
pub extern "C" fn molex_entity_residue(
    entity: *const molex_Entity,
    i: usize,
) -> *const molex_Residue {
    let Some(e) = entity_inner(entity) else {
        return std::ptr::null();
    };
    polymer_residues(e)
        .and_then(|residues| residues.get(i))
        .map_or(std::ptr::null(), |residue: &Residue| {
            std::ptr::from_ref(residue).cast::<molex_Residue>()
        })
}

/// Number of atoms in the i-th residue of this entity. Returns 0 if
/// `entity` is null, the entity has no residue records, or `i` is out
/// of bounds.
#[no_mangle]
pub extern "C" fn molex_entity_residue_num_atoms(
    entity: *const molex_Entity,
    i: usize,
) -> usize {
    let Some(e) = entity_inner(entity) else {
        return 0;
    };
    polymer_residues(e)
        .and_then(|residues| residues.get(i))
        .map_or(0, |residue| residue.atom_range.len())
}

/// Borrow a non-owning view of the j-th atom in the i-th residue of
/// this entity. Returns null on any out-of-bounds access or null
/// `entity` argument.
#[no_mangle]
pub extern "C" fn molex_entity_residue_atom(
    entity: *const molex_Entity,
    residue_idx: usize,
    atom_idx: usize,
) -> *const molex_Atom {
    let Some(e) = entity_inner(entity) else {
        return std::ptr::null();
    };
    let Some(residues) = polymer_residues(e) else {
        return std::ptr::null();
    };
    let Some(residue) = residues.get(residue_idx) else {
        return std::ptr::null();
    };
    let atoms = e.atom_set();
    let flat_idx = residue.atom_range.start + atom_idx;
    if flat_idx >= residue.atom_range.end {
        return std::ptr::null();
    }
    atoms.get(flat_idx).map_or(std::ptr::null(), |atom: &Atom| {
        std::ptr::from_ref(atom).cast::<molex_Atom>()
    })
}

// Residue accessors

fn residue_inner<'a>(residue: *const molex_Residue) -> Option<&'a Residue> {
    if residue.is_null() {
        return None;
    }
    Some(unsafe { &*residue.cast::<Residue>() })
}

/// Pointer to this residue's 3-byte name (e.g. b"ALA"). Writes 3 to
/// `out_len` on success. Returns null and writes 0 if `residue` is null.
///
/// The buffer is space-padded to 3 bytes; callers that want a trimmed
/// name should strip trailing ASCII spaces.
#[no_mangle]
pub extern "C" fn molex_residue_name(
    residue: *const molex_Residue,
    out_len: *mut usize,
) -> *const u8 {
    let Some(r) = residue_inner(residue) else {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null();
    };
    if !out_len.is_null() {
        unsafe {
            *out_len = r.name.len();
        }
    }
    r.name.as_ptr()
}

/// Author-side sequence id, falling back to the structural-side
/// (`label_seq_id`) when the author id is absent.
#[no_mangle]
pub extern "C" fn molex_residue_seq_id(residue: *const molex_Residue) -> i32 {
    residue_inner(residue).map_or(0, Residue::seq_id)
}

/// Structural-side sequence id (`label_seq_id`). Use this for stable
/// internal ordering rather than display.
#[no_mangle]
pub extern "C" fn molex_residue_label_seq_id(
    residue: *const molex_Residue,
) -> i32 {
    residue_inner(residue).map_or(0, |r| r.label_seq_id)
}

/// PDB insertion code byte (`iCode` / `pdbx_PDB_ins_code`). Returns 0
/// when the residue has no insertion code or `residue` is null.
#[no_mangle]
pub extern "C" fn molex_residue_ins_code(residue: *const molex_Residue) -> u8 {
    residue_inner(residue).and_then(|r| r.ins_code).unwrap_or(0)
}

// Atom accessors

fn atom_inner<'a>(atom: *const molex_Atom) -> Option<&'a Atom> {
    if atom.is_null() {
        return None;
    }
    Some(unsafe { &*atom.cast::<Atom>() })
}

/// Pointer to this atom's 4-byte PDB-style name (e.g. b"CA  "). Writes
/// 4 to `out_len` on success. Returns null and writes 0 if `atom` is null.
///
/// The buffer is space-padded; callers that want a trimmed atom name
/// should strip ASCII spaces.
#[no_mangle]
pub extern "C" fn molex_atom_name(
    atom: *const molex_Atom,
    out_len: *mut usize,
) -> *const u8 {
    let Some(a) = atom_inner(atom) else {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null();
    };
    if !out_len.is_null() {
        unsafe {
            *out_len = a.name.len();
        }
    }
    a.name.as_ptr()
}

/// Atomic number for this atom's element, or 0 for
/// [`Element::Unknown`] / a null `atom`.
#[no_mangle]
pub extern "C" fn molex_atom_atomic_number(atom: *const molex_Atom) -> u8 {
    atom_inner(atom).map_or(0, |a| Element::atomic_number(a.element))
}

/// Write this atom's `(x, y, z)` position into the 3-float output array.
/// No-op if either pointer is null.
#[no_mangle]
pub extern "C" fn molex_atom_position(
    atom: *const molex_Atom,
    out_xyz: *mut f32,
) {
    if out_xyz.is_null() {
        return;
    }
    let Some(a) = atom_inner(atom) else { return };
    unsafe {
        *out_xyz.add(0) = a.position.x;
        *out_xyz.add(1) = a.position.y;
        *out_xyz.add(2) = a.position.z;
    }
}

/// Crystallographic occupancy (0.0 to 1.0). Returns 0 if `atom` is null.
#[no_mangle]
pub extern "C" fn molex_atom_occupancy(atom: *const molex_Atom) -> f32 {
    atom_inner(atom).map_or(0.0, |a| a.occupancy)
}

/// Temperature factor (B-factor) in square angstroms. Returns 0 if
/// `atom` is null.
#[no_mangle]
pub extern "C" fn molex_atom_b_factor(atom: *const molex_Atom) -> f32 {
    atom_inner(atom).map_or(0.0, |a| a.b_factor)
}

/// Signed formal charge (0 means neutral). Returns 0 if `atom` is null.
#[no_mangle]
pub extern "C" fn molex_atom_formal_charge(atom: *const molex_Atom) -> i8 {
    atom_inner(atom).map_or(0, |a| a.formal_charge)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests assert against literal atom coordinates carried verbatim \
              through the handle; bit-exact equality is the correct check."
)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::c_api::{
        molex_Assembly, molex_EntityKind, molex_MoleculeType,
        molex_assembly_free,
    };
    use crate::entity::molecule::atom::Atom;
    use crate::entity::molecule::id::EntityIdAllocator;
    use crate::entity::molecule::protein::ProteinEntity;
    use crate::entity::molecule::small_molecule::SmallMoleculeEntity;
    use crate::entity::molecule::{MoleculeEntity, MoleculeType, Residue};

    fn mk_atom(name: [u8; 4], el: Element, pos: Vec3) -> Atom {
        Atom {
            position: pos,
            occupancy: 1.0,
            b_factor: 0.0,
            element: el,
            name,
            formal_charge: 0,
        }
    }

    /// Build an owned assembly handle holding one two-residue protein
    /// (ALA-GLY) and one small-molecule entity, mirroring the parser
    /// entry points' `Box::into_raw` ownership. Free with
    /// [`molex_assembly_free`].
    fn make_assembly_handle() -> *mut molex_Assembly {
        let mut alloc = EntityIdAllocator::new();

        let protein_atoms = vec![
            mk_atom(*b"N   ", Element::N, Vec3::ZERO),
            mk_atom(*b"CA  ", Element::C, Vec3::new(1.0, 0.0, 0.0)),
            mk_atom(*b"C   ", Element::C, Vec3::new(2.0, 0.0, 0.0)),
            mk_atom(*b"O   ", Element::O, Vec3::new(2.0, 1.0, 0.0)),
            mk_atom(*b"CB  ", Element::C, Vec3::new(1.0, -1.0, 0.0)),
            mk_atom(*b"N   ", Element::N, Vec3::new(3.2, 0.0, 0.0)),
            mk_atom(*b"CA  ", Element::C, Vec3::new(4.2, 0.0, 0.0)),
            mk_atom(*b"C   ", Element::C, Vec3::new(5.2, 0.0, 0.0)),
            mk_atom(*b"O   ", Element::O, Vec3::new(5.2, 1.0, 0.0)),
        ];
        // Residues named MSE so the missing-atom completion pass is a
        // no-op (MSE classifies as protein but resolves to no standard
        // template); these tests fix exact per-residue atom counts and
        // are about pointer-walking, not chemistry completion.
        let residues = vec![
            Residue {
                name: *b"MSE",
                label_seq_id: 1,
                auth_seq_id: None,
                auth_comp_id: None,
                ins_code: None,
                atom_range: 0..5,
                variants: Vec::new(),
            },
            Residue {
                name: *b"MSE",
                label_seq_id: 2,
                auth_seq_id: None,
                auth_comp_id: None,
                ins_code: None,
                atom_range: 5..9,
                variants: Vec::new(),
            },
        ];
        let protein = MoleculeEntity::Protein(ProteinEntity::new(
            alloc.allocate(),
            protein_atoms,
            residues,
            "A".to_owned(),
        ));

        let ligand = MoleculeEntity::SmallMolecule(SmallMoleculeEntity::new(
            alloc.allocate(),
            MoleculeType::Ligand,
            vec![mk_atom(*b"C1  ", Element::C, Vec3::new(10.0, 0.0, 0.0))],
            *b"LIG",
        ));

        let assembly = Assembly::new(vec![protein, ligand]);
        Box::into_raw(Box::new(assembly)).cast::<molex_Assembly>()
    }

    // -- Assembly accessors: null + out-of-range --

    #[test]
    fn assembly_accessors_handle_null_and_out_of_range() {
        let null = std::ptr::null::<molex_Assembly>();
        assert_eq!(molex_assembly_generation(null), 0);
        assert_eq!(molex_assembly_num_entities(null), 0);
        assert!(molex_assembly_entity(null, 0).is_null());

        let asm = make_assembly_handle();
        assert_eq!(molex_assembly_num_entities(asm), 2);
        assert!(!molex_assembly_entity(asm, 0).is_null());
        // Index one past the last entity must not deref out of bounds.
        assert!(molex_assembly_entity(asm, 2).is_null());
        assert!(molex_assembly_entity(asm, usize::MAX).is_null());
        molex_assembly_free(asm);
    }

    // -- Entity scalar accessors: null handle returns sentinel --

    #[test]
    fn entity_scalar_accessors_handle_null() {
        let null = std::ptr::null::<molex_Entity>();
        assert_eq!(molex_entity_id(null), 0);
        // Documented null sentinel: Solvent's integer value (8).
        assert_eq!(
            molex_entity_molecule_type(null),
            molex_MoleculeType::Solvent
        );
        // Documented null sentinel: Bulk's integer value (3).
        assert_eq!(molex_entity_kind(null), molex_EntityKind::Bulk);
        assert_eq!(molex_entity_pdb_chain_id(null), -1);
        assert_eq!(molex_entity_num_atoms(null), 0);
        assert_eq!(molex_entity_molecule_count(null), 0);
        assert_eq!(molex_entity_num_residues(null), 0);
    }

    #[test]
    fn entity_scalar_accessors_valid_baseline() {
        let asm = make_assembly_handle();
        let protein = molex_assembly_entity(asm, 0);
        assert!(!protein.is_null());
        assert_eq!(
            molex_entity_molecule_type(protein),
            molex_MoleculeType::Protein
        );
        assert_eq!(molex_entity_kind(protein), molex_EntityKind::Protein);
        assert_eq!(molex_entity_pdb_chain_id(protein), i32::from(b'A'));
        assert_eq!(molex_entity_num_atoms(protein), 9);
        assert_eq!(molex_entity_num_residues(protein), 2);
        molex_assembly_free(asm);
    }

    #[test]
    fn entity_chain_id_string_accessor() {
        let asm = make_assembly_handle();
        let protein = molex_assembly_entity(asm, 0);
        let mut len: usize = 0;
        let ptr = molex_entity_chain_id(protein, &raw mut len);
        assert!(!ptr.is_null());
        assert_eq!(len, 1);
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        assert_eq!(bytes, b"A");

        // Non-polymer entity (ligand at index 1) has no chain id.
        let ligand = molex_assembly_entity(asm, 1);
        let mut lig_len: usize = 7;
        let lig_ptr = molex_entity_chain_id(ligand, &raw mut lig_len);
        assert!(lig_ptr.is_null());
        assert_eq!(lig_len, 0);

        // Null entity writes 0 and returns null.
        let mut null_len: usize = 9;
        let null_ptr =
            molex_entity_chain_id(std::ptr::null(), &raw mut null_len);
        assert!(null_ptr.is_null());
        assert_eq!(null_len, 0);

        molex_assembly_free(asm);
    }

    // -- Entity indexed accessors: null + out-of-range --

    #[test]
    fn entity_atom_handles_null_and_out_of_range() {
        assert!(molex_entity_atom(std::ptr::null(), 0).is_null());

        let asm = make_assembly_handle();
        let entity = molex_assembly_entity(asm, 0);
        assert!(!molex_entity_atom(entity, 0).is_null());
        assert!(molex_entity_atom(entity, 9).is_null());
        assert!(molex_entity_atom(entity, usize::MAX).is_null());
        molex_assembly_free(asm);
    }

    #[test]
    fn entity_residue_handles_null_and_out_of_range() {
        assert!(molex_entity_residue(std::ptr::null(), 0).is_null());

        let asm = make_assembly_handle();
        let entity = molex_assembly_entity(asm, 0);
        assert!(!molex_entity_residue(entity, 0).is_null());
        assert!(molex_entity_residue(entity, 2).is_null());
        assert!(molex_entity_residue(entity, usize::MAX).is_null());
        molex_assembly_free(asm);
    }

    #[test]
    fn entity_residue_num_atoms_handles_null_and_out_of_range() {
        assert_eq!(molex_entity_residue_num_atoms(std::ptr::null(), 0), 0);

        let asm = make_assembly_handle();
        let entity = molex_assembly_entity(asm, 0);
        assert_eq!(molex_entity_residue_num_atoms(entity, 0), 5);
        assert_eq!(molex_entity_residue_num_atoms(entity, 2), 0);
        assert_eq!(molex_entity_residue_num_atoms(entity, usize::MAX), 0);
        molex_assembly_free(asm);
    }

    #[test]
    fn entity_residue_atom_handles_null_and_out_of_range() {
        assert!(molex_entity_residue_atom(std::ptr::null(), 0, 0).is_null());

        let asm = make_assembly_handle();
        let entity = molex_assembly_entity(asm, 0);
        // Valid (residue 0, atom 0) resolves.
        assert!(!molex_entity_residue_atom(entity, 0, 0).is_null());
        // Out-of-range residue index.
        assert!(molex_entity_residue_atom(entity, 2, 0).is_null());
        // In-range residue, out-of-range atom-within-residue. Residue 0
        // spans flat atoms 0..5, so local index 5 walks past the range
        // even though that flat slot still exists in the entity.
        assert!(molex_entity_residue_atom(entity, 0, 5).is_null());
        assert!(molex_entity_residue_atom(entity, 0, usize::MAX).is_null());
        molex_assembly_free(asm);
    }

    #[test]
    fn entity_residue_name_single_handles_null_and_polymer() {
        // Null entity: returns null and writes 0 to out_len.
        let mut out_len: usize = 99;
        let p = molex_entity_residue_name_single(
            std::ptr::null(),
            &raw mut out_len,
        );
        assert!(p.is_null());
        assert_eq!(out_len, 0);

        let asm = make_assembly_handle();
        // Polymer entity has no single residue name.
        let protein = molex_assembly_entity(asm, 0);
        out_len = 99;
        let p = molex_entity_residue_name_single(protein, &raw mut out_len);
        assert!(p.is_null());
        assert_eq!(out_len, 0);

        // Small-molecule entity exposes its 3-byte residue name.
        let ligand = molex_assembly_entity(asm, 1);
        out_len = 0;
        let p = molex_entity_residue_name_single(ligand, &raw mut out_len);
        assert!(!p.is_null());
        assert_eq!(out_len, 3);
        unsafe {
            let name = std::slice::from_raw_parts(p, out_len);
            assert_eq!(name, b"LIG");
        }
        molex_assembly_free(asm);
    }

    // -- Residue accessors: null handle returns sentinel --

    #[test]
    fn residue_accessors_handle_null() {
        let null = std::ptr::null::<molex_Residue>();

        let mut out_len: usize = 99;
        let p = molex_residue_name(null, &raw mut out_len);
        assert!(p.is_null());
        assert_eq!(out_len, 0);

        assert_eq!(molex_residue_seq_id(null), 0);
        assert_eq!(molex_residue_label_seq_id(null), 0);
        assert_eq!(molex_residue_ins_code(null), 0);
    }

    #[test]
    fn residue_accessors_valid_baseline() {
        let asm = make_assembly_handle();
        let entity = molex_assembly_entity(asm, 0);
        let residue = molex_entity_residue(entity, 0);
        assert!(!residue.is_null());

        let mut out_len: usize = 0;
        let p = molex_residue_name(residue, &raw mut out_len);
        assert!(!p.is_null());
        assert_eq!(out_len, 3);
        unsafe {
            let name = std::slice::from_raw_parts(p, out_len);
            assert_eq!(name, b"MSE");
        }
        assert_eq!(molex_residue_label_seq_id(residue), 1);
        molex_assembly_free(asm);
    }

    // -- Atom accessors: null handle returns sentinel --

    #[test]
    fn atom_accessors_handle_null() {
        let null = std::ptr::null::<molex_Atom>();

        let mut out_len: usize = 99;
        let p = molex_atom_name(null, &raw mut out_len);
        assert!(p.is_null());
        assert_eq!(out_len, 0);

        assert_eq!(molex_atom_atomic_number(null), 0);
        assert_eq!(molex_atom_occupancy(null), 0.0);
        assert_eq!(molex_atom_b_factor(null), 0.0);
        assert_eq!(molex_atom_formal_charge(null), 0);

        // Null atom (and null out buffer) must be a no-op, not a write
        // through a dangling pointer.
        let mut xyz = [-1.0f32; 3];
        molex_atom_position(null, xyz.as_mut_ptr());
        assert_eq!(xyz, [-1.0, -1.0, -1.0]);
        let valid_atom_unused = std::ptr::null::<molex_Atom>();
        molex_atom_position(valid_atom_unused, std::ptr::null_mut());
    }

    #[test]
    fn atom_accessors_valid_baseline() {
        let asm = make_assembly_handle();
        let entity = molex_assembly_entity(asm, 0);
        let atom = molex_entity_atom(entity, 0);
        assert!(!atom.is_null());

        let mut out_len: usize = 0;
        let p = molex_atom_name(atom, &raw mut out_len);
        assert!(!p.is_null());
        assert_eq!(out_len, 4);
        unsafe {
            let name = std::slice::from_raw_parts(p, out_len);
            assert_eq!(name, b"N   ");
        }
        assert_eq!(
            molex_atom_atomic_number(atom),
            Element::atomic_number(Element::N)
        );

        let mut xyz = [0.0f32; 3];
        molex_atom_position(atom, xyz.as_mut_ptr());
        assert_eq!(xyz, [0.0, 0.0, 0.0]);
        molex_assembly_free(asm);
    }
}
