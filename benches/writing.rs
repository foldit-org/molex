//! Write benchmarks (PDB / mmCIF) over real RCSB structures.
//!
//! Each structure is parsed once outside the timed closure; only the write is
//! timed. Committed fixtures (1UBQ, 4HHB) are always benched. When
//! `MOLEX_BENCH_LARGE_DIR` points at a directory holding
//! `6vxx.{pdb,cif,bcif}` and `3j3q.cif`, those are benched too. PDB write is
//! skipped for structures that exceed the legacy PDB format limits (3J3Q).
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

/// A structure parsed into an `Assembly`, with a flag for whether it fits the
/// legacy PDB format (gates the PDB-write bench).
struct Fixture {
    name: &'static str,
    assembly: Assembly,
    fits_pdb: bool,
}

fn committed_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/data")
}

/// Parse the always-committed fixtures plus, if `MOLEX_BENCH_LARGE_DIR` is set
/// and the files exist, the large perf targets. mmCIF is the parse source for
/// all structures.
fn load_fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    let dir = committed_dir();
    for name in ["1ubq", "4hhb"] {
        let cif =
            std::fs::read_to_string(dir.join(format!("{name}.cif"))).unwrap();
        out.push(Fixture {
            name,
            assembly: Assembly::from_mmcif(&cif).unwrap(),
            fits_pdb: true,
        });
    }

    if let Some(large) = std::env::var_os("MOLEX_BENCH_LARGE_DIR") {
        let large = PathBuf::from(large);
        if let Ok(cif) = std::fs::read_to_string(large.join("6vxx.cif")) {
            out.push(Fixture {
                name: "6vxx",
                assembly: Assembly::from_mmcif(&cif).unwrap(),
                fits_pdb: true,
            });
        }
        // 3J3Q is too large for the legacy PDB format.
        if let Ok(cif) = std::fs::read_to_string(large.join("3j3q.cif")) {
            out.push(Fixture {
                name: "3j3q",
                assembly: Assembly::from_mmcif(&cif).unwrap(),
                fits_pdb: false,
            });
        }
    }

    out
}

fn bench_write(c: &mut Criterion) {
    let fixtures = load_fixtures();
    let mut group = c.benchmark_group("write");
    for fx in &fixtures {
        // Per-atom normalization: both writers emit the same atom set.
        group.throughput(Throughput::Elements(assembly_atoms(&fx.assembly)));
        if fx.fits_pdb {
            group.bench_with_input(
                BenchmarkId::new("pdb", fx.name),
                &fx.assembly,
                |b, asm| {
                    b.iter(|| black_box(asm).to_pdb().unwrap());
                },
            );
        }
        group.bench_with_input(
            BenchmarkId::new("mmcif", fx.name),
            &fx.assembly,
            |b, asm| {
                b.iter(|| black_box(asm).to_mmcif());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_write);
criterion_main!(benches);
