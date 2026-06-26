//! Benchmark for the entities -> flat-arrays egress core.
//!
//! Times the pure-Rust egress both the numpy `to_arrays()` path and the Biotite
//! bridge route through: `AtomTable::from_entities` (the one flatten) plus the
//! shared de-vocab `columns` marshaling for every per-atom column (no Python
//! interpreter). Gated on the `python` feature because the atomworks adapter
//! that owns the marshaling layer is itself feature-gated.
#![allow(
    missing_docs,
    unused_results,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

#[cfg(feature = "python")]
mod egress_bench {
    use std::path::PathBuf;

    use criterion::{black_box, BenchmarkId, Criterion, Throughput};
    use molex::adapters::atomworks::columns;
    use molex::adapters::table::AtomTable;
    use molex::{Assembly, MoleculeEntity};

    /// The pure-Rust egress both numpy and Biotite route through: the one
    /// flatten plus the shared de-vocab marshaling for every per-atom column.
    fn egress<E: std::borrow::Borrow<MoleculeEntity>>(entities: &[E]) {
        let table = AtomTable::from_entities(entities);
        let n = table.len();
        black_box(columns::coords_flat(&table, 0..n));
        black_box(columns::chain_ids(&table, 0..n));
        black_box(columns::res_ids(&table, 0..n));
        black_box(columns::res_names(&table, 0..n));
        black_box(columns::atom_names(&table, 0..n));
        black_box(columns::elements(&table, 0..n));
        black_box(columns::occupancies(&table, 0..n));
        black_box(columns::b_factors(&table, 0..n));
        black_box(columns::entity_ids(&table, 0..n));
        black_box(columns::mol_types(&table, 0..n));
        black_box(columns::chain_types(&table, 0..n));
    }

    /// A real structure parsed into an `Assembly`.
    struct Fixture {
        name: &'static str,
        assembly: Assembly,
        /// Total atom count, for per-atom `Throughput` normalization.
        atoms: u64,
    }

    fn committed_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/data")
    }

    fn load_fixtures() -> Vec<Fixture> {
        let mut out = Vec::new();
        let dir = committed_dir();
        let mut names: Vec<(&'static str, PathBuf)> = vec![
            ("1ubq", dir.join("1ubq.cif")),
            ("4hhb", dir.join("4hhb.cif")),
        ];

        if let Some(large) = std::env::var_os("MOLEX_BENCH_LARGE_DIR") {
            let large = PathBuf::from(large);
            names.push(("6vxx", large.join("6vxx.cif")));
            names.push(("3j3q", large.join("3j3q.cif")));
        }

        for (name, path) in names {
            let Ok(cif) = std::fs::read_to_string(&path) else {
                continue;
            };
            let assembly = Assembly::from_mmcif(&cif).unwrap();
            let atoms = assembly
                .entities()
                .iter()
                .map(|e| e.atom_count())
                .sum::<usize>() as u64;
            out.push(Fixture {
                name,
                assembly,
                atoms,
            });
        }

        out
    }

    pub fn bench(c: &mut Criterion) {
        let fixtures = load_fixtures();
        let mut group = c.benchmark_group("egress");
        for fx in &fixtures {
            group.throughput(Throughput::Elements(fx.atoms));
            group.bench_with_input(
                BenchmarkId::new("egress", fx.name),
                &fx.assembly,
                |b, asm| {
                    b.iter(|| {
                        egress(black_box(asm.entities()));
                    });
                },
            );
        }
        group.finish();
    }
}

#[cfg(feature = "python")]
criterion::criterion_group!(benches, egress_bench::bench);
#[cfg(feature = "python")]
criterion::criterion_main!(benches);

#[cfg(not(feature = "python"))]
fn main() {
    eprintln!("egress bench requires --features python");
}
