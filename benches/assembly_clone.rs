//! Benchmark proving `Assembly::clone` scales with entity count, not
//! atom count. Each clone bumps `N` `Arc` refcounts (`N` = entity
//! count), so for a fixed entity count the time should be flat across
//! a sweep of chain lengths.
#![allow(
    missing_docs,
    unused_results,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap
)]

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use glam::Vec3;
use molex::entity::molecule::atom::Atom;
use molex::entity::molecule::id::EntityIdAllocator;
use molex::entity::molecule::protein::ProteinEntity;
use molex::entity::molecule::Residue;
use molex::{Assembly, Element, MoleculeEntity};

const CHAIN_COUNT: usize = 5;
const RESIDUES_PER_CHAIN_VARIANTS: &[usize] = &[10, 100, 1000];

fn mk_atom(name: [u8; 4], element: Element, position: Vec3) -> Atom {
    Atom {
        position,
        occupancy: 1.0,
        b_factor: 0.0,
        element,
        name,
        formal_charge: 0,
        observed: true,
    }
}

/// Synthetic protein chain with `n_residues` ALA residues laid out as
/// canonical N, CA, C, O backbone atoms.
fn make_chain(
    alloc: &mut EntityIdAllocator,
    chain: u8,
    n_residues: usize,
    origin: Vec3,
) -> MoleculeEntity {
    let id = alloc.allocate();
    let mut atoms = Vec::with_capacity(n_residues * 4);
    let mut residues = Vec::with_capacity(n_residues);
    for r in 0..n_residues {
        let base = origin + Vec3::new(3.8 * r as f32, 0.0, 0.0);
        let start = atoms.len();
        atoms.push(mk_atom(*b"N   ", Element::N, base));
        atoms.push(mk_atom(
            *b"CA  ",
            Element::C,
            base + Vec3::new(1.0, 0.0, 0.0),
        ));
        atoms.push(mk_atom(
            *b"C   ",
            Element::C,
            base + Vec3::new(2.0, 0.0, 0.0),
        ));
        atoms.push(mk_atom(
            *b"O   ",
            Element::O,
            base + Vec3::new(2.0, 1.0, 0.0),
        ));
        residues.push(Residue {
            name: *b"ALA",
            label_seq_id: (r as i32) + 1,
            auth_seq_id: None,
            auth_comp_id: None,
            ins_code: None,
            atom_range: start..atoms.len(),
            variants: Vec::new(),
        });
    }
    MoleculeEntity::Protein(ProteinEntity::new(
        id,
        atoms,
        residues,
        String::from(chain as char),
    ))
}

fn make_assembly(residues_per_chain: usize) -> Assembly {
    let mut alloc = EntityIdAllocator::new();
    let mut entities = Vec::with_capacity(CHAIN_COUNT);
    for c in 0..CHAIN_COUNT {
        // Spread chains apart so disulfide / H-bond detection doesn't
        // cross-link them.
        let origin = Vec3::new(0.0, 0.0, (c as f32) * 200.0);
        entities.push(make_chain(
            &mut alloc,
            b'A' + c as u8,
            residues_per_chain,
            origin,
        ));
    }
    Assembly::new(entities)
}

fn bench_assembly_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly_clone");
    for &res in RESIDUES_PER_CHAIN_VARIANTS {
        let assembly = make_assembly(res);
        let total_atoms: usize =
            assembly.entities().iter().map(|e| e.atom_count()).sum();
        group.bench_with_input(
            BenchmarkId::new(
                "5_chains",
                format!("{res}res_{total_atoms}atoms"),
            ),
            &assembly,
            |b, asm| b.iter(|| black_box(asm.clone())),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_assembly_clone);
criterion_main!(benches);
