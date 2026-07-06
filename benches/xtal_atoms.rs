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
    use criterion::measurement::WallTime;
    use criterion::{
        black_box, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
    };
    #[cfg(all(feature = "xtal", feature = "gpu"))]
    use molex::testutil::prepared_synthetic_native_orbit;
    use molex::testutil::{
        gradient_once_with, prepared_synthetic, PreparedCase,
    };
    use molex::xtal::StencilBackend;

    /// Fixed cubic P1 cell, real-space grid spacing, and space group.
    const CELL: [f64; 6] = [60.0, 60.0, 60.0, 90.0, 90.0, 90.0];
    const SPACING: f64 = 1.5;
    const SG: i32 = 1;
    /// The B every atom is generated at (sets `f_obs`).
    const B_TRUE: f64 = 20.0;
    /// Flat B the map and gradient are evaluated at, offset from `B_TRUE` so
    /// the gradient is non-trivial.
    const B_START: f64 = 40.0;

    /// Time one `gradient_once` arm under `backend`, labeled `label` at the
    /// current parameter point. Extracted so the per-point `#[cfg]` GPU arm
    /// stays flat rather than nesting another closure past the lint threshold.
    fn bench_arm(
        group: &mut BenchmarkGroup<'_, WallTime>,
        label: &str,
        n: usize,
        case: &mut PreparedCase,
        backend: StencilBackend,
    ) {
        group.bench_with_input(BenchmarkId::new(label, n), &n, |b, _| {
            b.iter(|| {
                black_box(gradient_once_with(&mut *case, backend));
            });
        });
    }

    pub fn bench(c: &mut Criterion) {
        let mut group = c.benchmark_group("xtal_atoms");
        for &n in &[50_usize, 200, 800, 3200] {
            let mut case =
                prepared_synthetic(n, SG, CELL, SPACING, B_TRUE, B_START);

            group.throughput(Throughput::Elements(n as u64));
            bench_arm(&mut group, "cpu", n, &mut case, StencilBackend::Cpu);

            #[cfg(all(feature = "xtal", feature = "gpu"))]
            {
                // Full resident GPU pipeline (GPU splat + mixed-radix FFT +
                // resident forward/inverse + GPU gather) on the same native
                // five-smooth grid as the cpu arm.
                let mut gpu = prepared_synthetic_native_orbit(
                    n, SG, CELL, SPACING, B_TRUE, B_START,
                );
                assert_eq!(
                    gpu.grid_dims, case.grid_dims,
                    "gpu arm must run the native five-smooth grid, not pow2",
                );
                // Warm up device init and shader compilation outside the timed
                // region so the arm reflects steady-state splat/FFT/gather, not
                // the one-time wgpu setup.
                let _ = gradient_once_with(&mut gpu, StencilBackend::Gpu);
                bench_arm(&mut group, "gpu", n, &mut gpu, StencilBackend::Gpu);
            }
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
