//! Synthetic B-factor benchmark isolating the atom-count cost knob.
//!
//! Holds the cell, grid spacing, and space group fixed while scaling `n_atoms`,
//! so the real-space gather (linear in atoms) grows while the FFT stays a fixed
//! offset. Times a single `b_factor_gradients` evaluation, the dominant
//! per-iteration cost, with `compute_map` hoisted to setup so sigma-A is ready.
//! Timing one fixed-work eval (rather than a full `refine`) removes the
//! convergence-iteration count as a confounder, so the curve reflects only the
//! per-knob cost.
#![allow(
    missing_docs,
    unused_results,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    // The no-feature `main` just tells the user the bench needs feature flags.
    clippy::print_stderr
)]

#[cfg(all(feature = "xtal", feature = "testutil"))]
mod xtal_atoms_bench {
    use criterion::{black_box, BenchmarkId, Criterion, Throughput};
    use molex::testutil::prepared_synthetic;

    /// Fixed cubic P1 cell, real-space grid spacing, and space group.
    const CELL: [f64; 6] = [60.0, 60.0, 60.0, 90.0, 90.0, 90.0];
    const SPACING: f64 = 1.5;
    const SG: i32 = 1;
    /// The B every atom is generated at (sets `f_obs`).
    const B_TRUE: f64 = 20.0;
    /// Flat B the map and gradient are evaluated at, offset from `B_TRUE` so
    /// the gradient is non-trivial.
    const B_START: f64 = 40.0;

    pub fn bench(c: &mut Criterion) {
        let mut group = c.benchmark_group("xtal_atoms");
        for &n in &[50_usize, 200, 800, 3200] {
            let case =
                prepared_synthetic(n, SG, CELL, SPACING, B_TRUE, B_START);

            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(n),
                &n,
                |b, _| {
                    b.iter(|| {
                        black_box(molex::testutil::gradient_once(black_box(
                            &case,
                        )));
                    });
                },
            );
        }
        group.finish();
    }
}

#[cfg(all(feature = "xtal", feature = "testutil"))]
criterion::criterion_group!(benches, xtal_atoms_bench::bench);
#[cfg(all(feature = "xtal", feature = "testutil"))]
criterion::criterion_main!(benches);

#[cfg(not(all(feature = "xtal", feature = "testutil")))]
fn main() {
    eprintln!("xtal_atoms bench requires --features xtal,testutil");
}
