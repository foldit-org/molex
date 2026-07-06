//! Fixed-point encoding probe for the GPU density splat.
//!
//! Measures the empirical magnitudes of the real-space CPU Gaussian density
//! splat across the five recovery structures so a later fixed-point (u32
//! atomic) splat kernel can pick a scale `S` and decide signed vs unsigned
//! accumulation. WebGPU has no f32 atomics; the kernel will accumulate
//! `round(S * contribution)` into u32 atomics, so we need (a) the largest
//! accumulated voxel magnitude (sets `S`) and (b) whether any voxel or any
//! single per-atom contribution is negative (forces signed encoding).
//!
//! This is a measurement probe: it changes no production behavior and is
//! `#[ignore]`d. Run with:
//!   cargo test --release --features xtal,testutil \
//!     --test xtal_fixedpoint_probe -- --ignored --nocapture

#![cfg(all(feature = "xtal", feature = "testutil"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::print_stdout,
    clippy::too_many_lines
)]

use molex::element::Element;
use molex::testutil::refinement_from_cif_pair;
use molex::xtal::{
    compute_blur, cutoff_radius, form_factor, kernel_eval, splat_density,
    DensityGrid, FormFactor, SplatParams,
};

const STRUCTURES: [&str; 5] = ["1AKI", "4XDX", "6E6O", "7KOM", "2OXY"];

/// Peak (r = 0) value and the most-negative value over `[0, cutoff_sq]`, plus
/// the radius at which that minimum occurs.
fn kernel_extrema(ff: &FormFactor, b_atom: f64, blur: f64) -> (f64, f64, f64) {
    let cutoff = cutoff_radius(b_atom + blur);
    let cutoff_sq = cutoff * cutoff;
    let peak = kernel_eval(ff, b_atom, blur, 0.0);

    let steps = 200_000usize;
    let mut min_val = peak;
    let mut min_r = 0.0;
    for s in 0..=steps {
        let r_sq = cutoff_sq * (s as f64 / steps as f64);
        let v = kernel_eval(ff, b_atom, blur, r_sq);
        if v < min_val {
            min_val = v;
            min_r = r_sq.sqrt();
        }
    }
    (peak, min_val, min_r)
}

/// Replicate `XtalRefinement`'s d_min / blur derivation for a loaded case, then
/// run the raw ASU-only `splat_density` (no `symmetrize_sum`) into a fresh
/// grid. Returns `(grid, blur)`.
fn raw_splat(case: &molex::testutil::RealCase) -> (DensityGrid, f64) {
    let refinement = &case.refinement;
    let unit_cell = &refinement.unit_cell;

    // estimate_d_min: max d*^2 over the (already filtered) reflection set.
    let mut max_s2 = 0.0_f64;
    for r in &refinement.reflections {
        let s2 = unit_cell.d_star_sq(r.h, r.k, r.l);
        if s2 > max_s2 {
            max_s2 = s2;
        }
    }
    let d_min = if max_s2 > 0.0 {
        1.0 / max_s2.sqrt()
    } else {
        2.0
    };

    let b_min = case.b_factors.iter().copied().fold(f64::MAX, f64::min);
    let blur = compute_blur(d_min, 1.5, b_min);

    let ffs: Vec<&FormFactor> = case
        .elements
        .iter()
        .map(|&e| form_factor(e).expect("element has an IT92 form factor"))
        .collect();
    assert_eq!(
        ffs.len(),
        case.positions.len(),
        "every atom must resolve a form factor"
    );

    let [nu, nv, nw] = refinement.grid_dims;
    let mut grid = DensityGrid {
        data: vec![0.0; nu * nv * nw],
        nu,
        nv,
        nw,
    };
    let params = SplatParams {
        positions: &case.positions,
        form_factors: &ffs,
        b_factors: &case.b_factors,
        occupancies: &case.occupancies,
        unit_cell,
        blur,
    };
    splat_density(&mut grid, &params);
    (grid, blur)
}

/// Distinct elements in a case with the min B, max B, and atom count for each,
/// ordered by ascending atomic number.
fn element_summary(
    case: &molex::testutil::RealCase,
) -> Vec<(Element, f64, f64, usize)> {
    let mut out: Vec<(Element, f64, f64, usize)> = Vec::new();
    for (&e, &b) in case.elements.iter().zip(case.b_factors.iter()) {
        if let Some(entry) = out.iter_mut().find(|x| x.0 == e) {
            entry.1 = entry.1.min(b);
            entry.2 = entry.2.max(b);
            entry.3 += 1;
        } else {
            out.push((e, b, b, 1));
        }
    }
    out.sort_by_key(|x| x.0.atomic_number());
    out
}

#[test]
#[ignore = "loads all five recovery fixtures + full ASU splat; run with \
            --ignored"]
fn fixedpoint_probe() {
    println!("\n==== FIXED-POINT SPLAT PROBE ====\n");

    let mut global_max = f64::NEG_INFINITY;
    let mut global_min = f64::INFINITY;
    let mut global_peak_single = 0.0_f64;
    let mut any_kernel_negative = false;

    for pdb in STRUCTURES {
        let case = refinement_from_cif_pair(pdb)
            .unwrap_or_else(|| panic!("load {pdb} from tests/data"));

        let (grid, blur) = raw_splat(&case);

        let mut vmax = f32::NEG_INFINITY;
        let mut vmin = f32::INFINITY;
        for &v in &grid.data {
            vmax = vmax.max(v);
            vmin = vmin.min(v);
        }
        let [nu, nv, nw] = case.refinement.grid_dims;

        println!(
            "STRUCTURE {pdb}: atoms={} grid={nu}x{nv}x{nw} blur={blur:.3}",
            case.positions.len()
        );
        println!("  accumulated voxel: max={vmax:.6}  min={vmin:.6}");

        global_max = global_max.max(f64::from(vmax));
        global_min = global_min.min(f64::from(vmin));

        // Per-element kernel probe at each element's sharpest (min-B) atom.
        let elems = element_summary(&case);
        let heaviest_z =
            elems.iter().map(|e| e.0.atomic_number()).max().unwrap();
        for (e, b_min, b_max, count) in &elems {
            let is_heaviest = e.atomic_number() == heaviest_z;
            let flag = match e {
                Element::N => " [N]",
                Element::Cl => " [Cl]",
                _ if is_heaviest => " [heaviest]",
                _ => "",
            };
            let ff = form_factor(*e).expect("form factor");
            let (peak, min_val, min_r) = kernel_extrema(ff, *b_min, blur);
            let neg = min_val < 0.0;
            any_kernel_negative |= neg;
            global_peak_single = global_peak_single.max(peak);
            println!(
                "  element {:>2}{flag}: n={count} B=[{b_min:.2},{b_max:.2}] \
                 (probe@B={b_min:.2})  peak(r=0)={peak:.4}  \
                 kernel_min={min_val:.6} @r={min_r:.3}A  negative={neg}",
                e.symbol(),
            );
        }
        println!();
    }

    println!("==== GLOBALS ====");
    println!("  global max accumulated voxel : {global_max:.6}");
    println!("  global min accumulated voxel : {global_min:.6}");
    println!(
        "  global peak single contribution (r=0, occ=1): \
         {global_peak_single:.6}"
    );
    println!(
        "  any single-atom kernel goes negative in range: \
         {any_kernel_negative}"
    );

    // ── Fixed-point recommendation ─────────────────────────────────────
    //
    // The GPU kernel accumulates round(S * contribution) into 32-bit atomics.
    // Signed encoding interprets the u32 as two's-complement i32; wrapping
    // atomicAdd carries signed accumulation correctly as long as the running
    // sum stays within i32 range. Unsigned uses the full u32 but cannot
    // represent a negative partial sum.
    //
    // A negative accumulated voxel or a negative single contribution forces
    // signed, because atomic adds arrive in arbitrary order: an intermediate
    // partial sum can dip negative even where the final voxel is positive.
    let signed = global_min < 0.0 || any_kernel_negative;

    // Bound: keep S * worst-case magnitude under the range limit with a 2x
    // headroom margin. Worst-case magnitude is the larger of the max positive
    // accumulation and the max negative excursion we might see mid-accumulation
    // (approximated by |global_min|, plus the single-contribution magnitude).
    let worst_mag = global_max.max(global_min.abs()).max(global_peak_single);
    let range_limit = if signed {
        2.0_f64.powi(31)
    } else {
        2.0_f64.powi(32)
    };
    let margin = 2.0; // keep S*worst <= range_limit / margin
    let s_max = range_limit / (margin * worst_mag);

    // Round S down to a clean power-of-two-ish value for reproducibility.
    let s_pow2 = 2.0_f64.powf(s_max.log2().floor());

    println!("\n==== RECOMMENDATION ====");
    println!(
        "  verdict: {} encoding",
        if signed {
            "SIGNED (i32-in-u32)"
        } else {
            "UNSIGNED (u32)"
        }
    );
    println!(
        "  reason: global_min={global_min:.4} (<0 => signed: {}), \
         any_kernel_negative={any_kernel_negative}",
        global_min < 0.0
    );
    println!("  worst-case magnitude used for S: {worst_mag:.6}");
    println!(
        "  range limit ({}): 2^{}",
        if signed { "signed i32" } else { "unsigned u32" },
        if signed { 31 } else { 32 }
    );
    println!("  S upper bound (margin {margin:.0}x headroom): {s_max:.3e}");
    println!("  recommended S (power of two): {s_pow2:.3e}");
    println!(
        "  quantization floor 1/S: {:.3e} density units",
        1.0 / s_pow2
    );
    println!(
        "  smallest peak single contribution across elements is reported \
         per-structure above; compare 1/S against it"
    );
}
