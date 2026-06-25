//! Benchmarks for structural analysis operations.
//!
//! `kabsch_alignment` runs on synthetic point clouds (it's a pure geometry
//! kernel, independent of structure parsing). Everything else runs on real
//! RCSB fixtures (1UBQ, 4HHB), with `MOLEX_BENCH_LARGE_DIR` adding 6VXX and
//! 3J3Q when set.
#![allow(
    missing_docs,
    unused_results,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    clippy::format_push_string,
    clippy::uninlined_format_args
)]

use std::path::PathBuf;

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
    Throughput,
};
use glam::Vec3;
use molex::adapters::cif::mmcif_str_to_entities;
use molex::analysis::sasa::DEFAULT_N_POINTS;
use molex::analysis::{assembly_sasa, infer_bonds, ContactLevel};
use molex::ops::transform::kabsch_alignment;
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

/// Parse the always-committed fixtures plus, if `MOLEX_BENCH_LARGE_DIR` is set
/// and the files exist, the large perf targets.
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

fn bench_kabsch_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("kabsch_alignment");

    for n_points in [10, 50, 200, 1000] {
        // Synthetic, but point count is the natural element count here.
        group.throughput(Throughput::Elements(n_points as u64));
        let reference: Vec<Vec3> = (0..n_points)
            .map(|i| {
                let t = i as f32 * 0.1;
                Vec3::new(t.cos() * 10.0, t.sin() * 10.0, t * 0.5)
            })
            .collect();

        // Rotate + translate target
        let rotation =
            glam::Mat3::from_rotation_z(0.3) * glam::Mat3::from_rotation_x(0.2);
        let translation = Vec3::new(5.0, -3.0, 2.0);
        let target: Vec<Vec3> = reference
            .iter()
            .map(|p| rotation * *p + translation)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("kabsch", format!("{n_points}_points")),
            &(reference.clone(), target.clone()),
            |b, (reference, target)| {
                b.iter(|| {
                    kabsch_alignment(black_box(reference), black_box(target))
                });
            },
        );
    }

    group.finish();
}

fn bench_sasa(c: &mut Criterion, fixtures: &[Fixture]) {
    let mut group = c.benchmark_group("sasa");
    // SASA at full angular resolution is the slowest analysis; cap its sample
    // count so the group stays usable while still real.
    group.sample_size(20);
    for fx in fixtures {
        group.throughput(Throughput::Elements(fx.atoms));
        group.bench_with_input(
            BenchmarkId::new("assembly_sasa", fx.name),
            &fx.assembly,
            |b, asm| {
                b.iter(|| assembly_sasa(black_box(asm), 1.4, DEFAULT_N_POINTS));
            },
        );
    }
    group.finish();
}

fn bench_recompute_ss(c: &mut Criterion, fixtures: &[Fixture]) {
    let mut group = c.benchmark_group("recompute_ss");
    for fx in fixtures {
        group.throughput(Throughput::Elements(fx.atoms));
        group.bench_with_input(
            BenchmarkId::new("recompute_ss", fx.name),
            &fx.assembly,
            |b, asm| {
                b.iter(|| {
                    let mut a = asm.clone();
                    a.recompute_ss();
                    black_box(a);
                });
            },
        );
    }
    group.finish();
}

fn bench_phi_psi(c: &mut Criterion, fixtures: &[Fixture]) {
    let mut group = c.benchmark_group("phi_psi");
    for fx in fixtures {
        group.throughput(Throughput::Elements(fx.atoms));
        group.bench_with_input(
            BenchmarkId::new("phi_psi", fx.name),
            &fx.assembly,
            |b, asm| {
                b.iter(|| {
                    let proteins =
                        asm.entities().iter().filter_map(|e| e.as_protein());
                    for p in proteins {
                        black_box(p.phi_psi());
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_infer_bonds(c: &mut Criterion, fixtures: &[Fixture]) {
    let mut group = c.benchmark_group("infer_bonds");
    for fx in fixtures {
        group.throughput(Throughput::Elements(fx.atoms));
        group.bench_with_input(
            BenchmarkId::new("infer_bonds", fx.name),
            &fx.assembly,
            |b, asm| {
                b.iter(|| {
                    for entity in asm.entities() {
                        let atoms = entity.columns().to_atoms();
                        black_box(infer_bonds(black_box(&atoms), 0.45));
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_contacts(c: &mut Criterion, fixtures: &[Fixture]) {
    let mut group = c.benchmark_group("contacts");
    for fx in fixtures {
        group.throughput(Throughput::Elements(fx.atoms));
        group.bench_with_input(
            BenchmarkId::new("contacts", fx.name),
            &fx.assembly,
            |b, asm| {
                b.iter(|| {
                    black_box(asm.contacts(4.0, ContactLevel::Atom, false));
                });
            },
        );
    }
    group.finish();
}

fn bench_analysis(c: &mut Criterion) {
    bench_kabsch_alignment(c);
    let fixtures = load_fixtures();
    bench_sasa(c, &fixtures);
    bench_recompute_ss(c, &fixtures);
    bench_phi_psi(c, &fixtures);
    bench_infer_bonds(c, &fixtures);
    bench_contacts(c, &fixtures);
}

criterion_group!(benches, bench_analysis);
criterion_main!(benches);
