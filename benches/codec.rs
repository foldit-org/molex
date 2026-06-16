//! Benchmarks for the assembly wire format round-trip.
#![allow(
    missing_docs,
    unused_results,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::suboptimal_flops,
    clippy::format_push_string,
    clippy::uninlined_format_args,
    unused_variables
)]

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use molex::adapters::pdb::pdb_str_to_entities;
use molex::ops::wire::{deserialize_assembly, serialize_assembly};
use molex::Assembly;

/// Build a synthetic PDB string with `n_residues` ALA residues (5 atoms each).
fn generate_pdb(n_residues: usize) -> String {
    let mut pdb = String::new();
    let mut serial = 1usize;
    for res_num in 1..=n_residues {
        for &(name, dx, dy, dz, elem) in &[
            ("N  ", 0.0f64, 0.0, 0.0, "N"),
            ("CA ", 1.47, 0.0, 0.0, "C"),
            ("C  ", 2.45, 1.0, 0.0, "C"),
            ("O  ", 2.45, 2.2, 0.0, "O"),
            ("CB ", 1.47, -1.5, 0.0, "C"),
        ] {
            let x = dx + (res_num as f64) * 3.8;
            pdb.push_str(&format!(
                "ATOM  {:>5} {:<4}ALA A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  \
                 0.00          {:>2}  \n",
                serial, name, res_num, x, dy, dz, elem
            ));
            serial += 1;
        }
    }
    pdb.push_str("END\n");
    pdb
}

fn build_assembly(n_atoms: usize) -> Assembly {
    let n_residues = (n_atoms / 5).max(1);
    let pdb = generate_pdb(n_residues);
    let entities = pdb_str_to_entities(&pdb).unwrap();
    Assembly::new(entities)
}

fn bench_assembly_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly_roundtrip");

    for n_atoms in [50, 500, 5000] {
        let assembly = build_assembly(n_atoms);
        let bytes = serialize_assembly(&assembly).unwrap();

        group.bench_with_input(
            BenchmarkId::new("serialize", format!("{n_atoms}_atoms")),
            &assembly,
            |b, assembly| {
                b.iter(|| serialize_assembly(black_box(assembly)).unwrap());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("deserialize", format!("{n_atoms}_atoms")),
            &bytes,
            |b, bytes| {
                b.iter(|| deserialize_assembly(black_box(bytes)).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_assembly_roundtrip);
criterion_main!(benches);
