//! Benchmarks for the assembly wire format round-trip over real structures.
#![allow(
    missing_docs,
    unused_results,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::suboptimal_flops,
    clippy::format_push_string,
    clippy::uninlined_format_args
)]

use std::path::PathBuf;

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
    Throughput,
};
use molex::Assembly;

fn assembly_atoms(asm: &Assembly) -> u64 {
    asm.entities().iter().map(|e| e.atom_count()).sum::<usize>() as u64
}

fn committed_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/data")
}

/// Parse a committed mmCIF fixture into an `Assembly`.
fn load_assembly(name: &str) -> Assembly {
    let cif =
        std::fs::read_to_string(committed_dir().join(format!("{name}.cif")))
            .unwrap();
    Assembly::from_mmcif(&cif).unwrap()
}

fn bench_assembly_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("assembly_roundtrip");

    for name in ["1ubq", "4hhb"] {
        let assembly = load_assembly(name);
        let bytes = assembly.to_bytes().unwrap();

        // Per-atom normalization: both directions move the same atom set.
        group.throughput(Throughput::Elements(assembly_atoms(&assembly)));

        group.bench_with_input(
            BenchmarkId::new("serialize", name),
            &assembly,
            |b, assembly| {
                b.iter(|| black_box(assembly).to_bytes().unwrap());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("deserialize", name),
            &bytes,
            |b, bytes| {
                b.iter(|| Assembly::from_bytes(black_box(bytes)).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_assembly_roundtrip);
criterion_main!(benches);
