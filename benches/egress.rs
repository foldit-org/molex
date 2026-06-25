//! Benchmark for the entities -> flat-arrays egress core.
//!
//! `collect_atom_data` is the shared pure-Rust column collector both the numpy
//! `to_arrays()` path and the Biotite bridge route through; this times it
//! directly (no Python interpreter). Gated on the `python` feature because the
//! atomworks adapter that owns the collector is itself feature-gated.
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
    use molex::adapters::atomworks::collect_atom_data;
    use molex::adapters::cif::mmcif_str_to_entities;
    use molex::{Assembly, MoleculeEntity};

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
            let entities = mmcif_str_to_entities(&cif).unwrap();
            let atoms = entities
                .iter()
                .map(MoleculeEntity::atom_count)
                .sum::<usize>() as u64;
            let assembly = Assembly::new(entities);
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
                BenchmarkId::new("collect_atom_data", fx.name),
                &fx.assembly,
                |b, asm| {
                    let total = fx.atoms as usize;
                    b.iter(|| {
                        black_box(collect_atom_data(
                            black_box(asm.entities()),
                            total,
                        ));
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
