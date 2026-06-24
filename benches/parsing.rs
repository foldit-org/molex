//! Parse benchmarks (PDB / mmCIF / BinaryCIF) over real RCSB structures.
//!
//! Committed fixtures (1UBQ, 4HHB) are always benched. When
//! `MOLEX_BENCH_LARGE_DIR` points at a directory holding
//! `6vxx.{pdb,cif,bcif}` and `3j3q.cif`, those are benched too so a dev can
//! exercise the real perf targets without committing them.
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
use molex::adapters::bcif::bcif_to_entities;
use molex::adapters::cif::mmcif_str_to_entities;
use molex::adapters::pdb::pdb_str_to_entities;
use molex::MoleculeEntity;

/// A structure's three serialized forms, as in-memory bytes/strings.
struct Fixture {
    name: &'static str,
    pdb: String,
    cif: String,
    bcif: Vec<u8>,
    /// `cif`-only structures (e.g. 3J3Q has no PDB/BCIF target) leave `pdb`
    /// and `bcif` empty; `has_pdb`/`has_bcif` gate the per-format benches.
    has_pdb: bool,
    has_bcif: bool,
    /// Atom count, parsed once at load time, to normalize each format's
    /// timing per-atom via `Throughput::Elements`.
    atoms: u64,
}

fn atom_count(cif: &str) -> u64 {
    mmcif_str_to_entities(cif)
        .unwrap()
        .iter()
        .map(MoleculeEntity::atom_count)
        .sum::<usize>() as u64
}

fn committed_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/data")
}

/// Load the always-committed fixtures plus, if `MOLEX_BENCH_LARGE_DIR` is set
/// and the files exist, the large perf targets.
fn load_fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    let dir = committed_dir();
    for name in ["1ubq", "4hhb"] {
        let cif =
            std::fs::read_to_string(dir.join(format!("{name}.cif"))).unwrap();
        out.push(Fixture {
            name,
            pdb: std::fs::read_to_string(dir.join(format!("{name}.pdb")))
                .unwrap(),
            atoms: atom_count(&cif),
            cif,
            bcif: std::fs::read(dir.join(format!("{name}.bcif"))).unwrap(),
            has_pdb: true,
            has_bcif: true,
        });
    }

    if let Some(large) = std::env::var_os("MOLEX_BENCH_LARGE_DIR") {
        let large = PathBuf::from(large);
        // 6VXX has all three serializations.
        if let (Ok(pdb), Ok(cif), Ok(bcif)) = (
            std::fs::read_to_string(large.join("6vxx.pdb")),
            std::fs::read_to_string(large.join("6vxx.cif")),
            std::fs::read(large.join("6vxx.bcif")),
        ) {
            out.push(Fixture {
                name: "6vxx",
                pdb,
                atoms: atom_count(&cif),
                cif,
                bcif,
                has_pdb: true,
                has_bcif: true,
            });
        }
        // 3J3Q is mmCIF-only (too large for the legacy PDB format).
        if let Ok(cif) = std::fs::read_to_string(large.join("3j3q.cif")) {
            out.push(Fixture {
                name: "3j3q",
                pdb: String::new(),
                atoms: atom_count(&cif),
                cif,
                bcif: Vec::new(),
                has_pdb: false,
                has_bcif: false,
            });
        }
    }

    out
}

fn bench_parse(c: &mut Criterion) {
    let fixtures = load_fixtures();
    let mut group = c.benchmark_group("parse");
    for fx in &fixtures {
        // Per-atom normalization: every format of a fixture decodes the same
        // atom set, so the count is shared across the three benches below.
        group.throughput(Throughput::Elements(fx.atoms));
        if fx.has_pdb {
            group.bench_with_input(
                BenchmarkId::new("pdb", fx.name),
                &fx.pdb,
                |b, pdb| {
                    b.iter(|| pdb_str_to_entities(black_box(pdb)).unwrap());
                },
            );
        }
        group.bench_with_input(
            BenchmarkId::new("mmcif", fx.name),
            &fx.cif,
            |b, cif| {
                b.iter(|| mmcif_str_to_entities(black_box(cif)).unwrap());
            },
        );
        if fx.has_bcif {
            group.bench_with_input(
                BenchmarkId::new("bcif", fx.name),
                &fx.bcif,
                |b, bcif| {
                    b.iter(|| bcif_to_entities(black_box(bcif)).unwrap());
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);
