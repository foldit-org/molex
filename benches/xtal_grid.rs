//! Synthetic B-factor benchmark isolating the grid-size cost knob.
//!
//! Grows the unit cell at fixed grid spacing, fixed atom count, and fixed space
//! group, so the number of grid points (and hence the FFT) scales while voxels
//! per atom stay constant. Scaling resolution/spacing instead would also grow
//! voxels per atom and confound the FFT signal, so the cell is the knob.
//! Times a single `b_factor_gradients` evaluation (the dominant per-iteration
//! cost) with `compute_map` hoisted to setup so sigma-A is ready; timing one
//! fixed-work eval removes the convergence-iteration count as a confounder.
//! Throughput is reported in grid points; expect ~N log N.
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
mod xtal_grid_bench {
    use criterion::measurement::WallTime;
    use criterion::{
        black_box, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
    };
    #[cfg(feature = "xtal-gpu")]
    use molex::testutil::prepared_synthetic_native_orbit;
    use molex::testutil::{
        gradient_once_with, prepared_synthetic, PreparedCase,
    };
    use molex::xtal::StencilBackend;

    /// Fixed real-space grid spacing, space group, and atom count.
    const SPACING: f64 = 1.5;
    const SG: i32 = 1;
    const N_ATOMS: usize = 200;
    /// The B every atom is generated at (sets `f_obs`).
    const B_TRUE: f64 = 20.0;
    /// Flat B the map and gradient are evaluated at, offset from `B_TRUE` so
    /// the gradient is non-trivial.
    const B_START: f64 = 40.0;

    /// Time one `gradient_once` arm under `backend`, labeled `label` at the
    /// current grid-point count. Extracted so the per-point `#[cfg]` GPU arm
    /// stays flat rather than nesting another closure past the lint threshold.
    fn bench_arm(
        group: &mut BenchmarkGroup<'_, WallTime>,
        label: &str,
        grid_points: u64,
        case: &mut PreparedCase,
        backend: StencilBackend,
    ) {
        group.bench_with_input(
            BenchmarkId::new(label, grid_points),
            &grid_points,
            |b, _| {
                b.iter(|| {
                    black_box(gradient_once_with(&mut *case, backend));
                });
            },
        );
    }

    pub fn bench(c: &mut Criterion) {
        let mut group = c.benchmark_group("xtal_grid");
        for &edge in &[40.0_f64, 60.0, 80.0, 100.0] {
            let cell = [edge, edge, edge, 90.0, 90.0, 90.0];
            let mut case =
                prepared_synthetic(N_ATOMS, SG, cell, SPACING, B_TRUE, B_START);
            let [nu, nv, nw] = case.grid_dims;
            let grid_points = (nu * nv * nw) as u64;

            group.throughput(Throughput::Elements(grid_points));
            bench_arm(
                &mut group,
                "cpu",
                grid_points,
                &mut case,
                StencilBackend::Cpu,
            );

            #[cfg(feature = "xtal-gpu")]
            {
                // Full resident GPU pipeline (GPU splat + mixed-radix FFT +
                // resident forward/inverse + GPU gather) on the same native
                // five-smooth grid as the cpu arm.
                let mut gpu = prepared_synthetic_native_orbit(
                    N_ATOMS, SG, cell, SPACING, B_TRUE, B_START,
                );
                assert_eq!(
                    gpu.grid_dims, case.grid_dims,
                    "gpu arm must run the native five-smooth grid, not pow2",
                );
                // Warm up device init and shader compilation outside the timed
                // region so the arm reflects steady-state splat/FFT/gather, not
                // the one-time wgpu setup.
                let _ = gradient_once_with(&mut gpu, StencilBackend::Gpu);
                bench_arm(
                    &mut group,
                    "gpu",
                    grid_points,
                    &mut gpu,
                    StencilBackend::Gpu,
                );
            }
        }
        group.finish();
    }
}

#[cfg(all(feature = "xtal", feature = "testutil"))]
criterion::criterion_group!(benches, xtal_grid_bench::bench);
#[cfg(all(feature = "xtal", feature = "testutil"))]
criterion::criterion_main!(benches);

#[cfg(not(all(feature = "xtal", feature = "testutil")))]
fn main() {
    eprintln!("xtal_grid bench requires --features xtal,testutil");
}
