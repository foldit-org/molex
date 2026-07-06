//! Full-GPU-pipeline B-factor recovery on deposited structures.
//!
//! Runs recovery through the power-of-two splat-full-orbit forward model on the
//! GPU stencil backend, so the map, the forward Fc, and the gradient all run
//! GPU splat + GPU FFT + GPU gather. This is the acceptance gate for the GPU
//! splat's fixed-point quantization plus f32 arithmetic on the (pow2) grid: the
//! recovered B-factors must still correlate with the deposited ones above 0.3.
//!
//! On-device (Metal on macOS) and slow, so every case is `#[ignore]`d; run with
//! `cargo test --features xtal,gpu,minimization,testutil -- --ignored`.

#![cfg(all(
    feature = "xtal",
    feature = "gpu",
    feature = "minimization",
    feature = "testutil"
))]
#![allow(clippy::print_stdout, clippy::unwrap_used)]

use molex::testutil::recover_b_factors_full_gpu;

/// Recover, print the scale/quality row, and hold the recovered B-correlation
/// to the >= 0.3 bar.
fn check(pdb: &str) {
    let Some(r) = recover_b_factors_full_gpu(pdb) else {
        println!("{pdb} fixtures missing; skipping full-GPU recovery");
        return;
    };
    let [nu, nv, nw] = r.grid;
    println!(
        "GPU-RECOVERY {pdb}: atoms={} grid={nu}x{nv}x{nw} R-work {:.4} -> \
         {:.4} B-Pearson={:.4}",
        r.n_atoms, r.r_work_before, r.r_work_after, r.b_corr,
    );
    assert!(
        r.b_corr > 0.3,
        "{pdb}: full-GPU recovery Pearson {:.4} below the 0.3 bar",
        r.b_corr,
    );
}

/// Orthorhombic P2₁2₁2₁ (4 symops), the pipeline-validation structure.
#[test]
#[ignore = "on-device full-GPU recovery; run with --ignored"]
fn recovers_1aki_full_gpu() {
    check("1AKI");
}

/// Triclinic P1 on a larger cell with many atoms: the high-atom-count case
/// (orbit multiplicity 1, so the whole asymmetric unit splats directly).
#[test]
#[ignore = "on-device full-GPU recovery; run with --ignored"]
fn recovers_2oxy_full_gpu() {
    check("2OXY");
}

/// Monoclinic C121 (2 symops × 2 centering = orbit 4): a centered-lattice case
/// whose power-of-two grid (256 on the long axis) is larger than 1AKI's.
#[test]
#[ignore = "on-device full-GPU recovery; run with --ignored"]
fn recovers_6e6o_full_gpu() {
    check("6E6O");
}
