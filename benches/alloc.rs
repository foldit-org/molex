//! Allocation-count metric over real RCSB structures.
//!
//! Not a criterion bench: a `#[global_allocator]` wrapping `System` counts
//! allocation COUNT and total BYTES per parse/write, and the metric is printed
//! as a table. Allocation churn is the profiled #1 parse cost, so
//! allocations-per-atom is the number the optimization work drives down. The
//! global allocator only affects this bench binary, which is correct.
//!
//! `MOLEX_BENCH_LARGE_DIR` adds 6VXX (all formats) and 3J3Q (mmCIF) when set.
#![allow(
    missing_docs,
    unused_results,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::suboptimal_flops,
    clippy::format_push_string,
    clippy::uninlined_format_args,
    clippy::print_stdout
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

use molex::Assembly;

/// `System` allocator that tallies live allocation count and total bytes.
/// Counting `alloc`/`alloc_zeroed`/`realloc` (not `dealloc`) gives the gross
/// allocation traffic for a single-threaded region between counter resets.
struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size(), Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size(), Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
        new_size: usize,
    ) -> *mut u8 {
        // A grow counts as a fresh allocation of the new size; the old block's
        // bytes were already tallied at its own alloc.
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(new_size, Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn reset() {
    ALLOCS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
}

fn snapshot() -> (usize, usize) {
    (ALLOCS.load(Relaxed), BYTES.load(Relaxed))
}

struct Fixture {
    name: &'static str,
    pdb: String,
    cif: String,
    bcif: Vec<u8>,
    has_pdb: bool,
    has_bcif: bool,
}

fn committed_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/data")
}

fn load_fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    let dir = committed_dir();
    for name in ["1ubq", "4hhb"] {
        out.push(Fixture {
            name,
            pdb: std::fs::read_to_string(dir.join(format!("{name}.pdb")))
                .unwrap(),
            cif: std::fs::read_to_string(dir.join(format!("{name}.cif")))
                .unwrap(),
            bcif: std::fs::read(dir.join(format!("{name}.bcif"))).unwrap(),
            has_pdb: true,
            has_bcif: true,
        });
    }

    if let Some(large) = std::env::var_os("MOLEX_BENCH_LARGE_DIR") {
        let large = PathBuf::from(large);
        if let (Ok(pdb), Ok(cif), Ok(bcif)) = (
            std::fs::read_to_string(large.join("6vxx.pdb")),
            std::fs::read_to_string(large.join("6vxx.cif")),
            std::fs::read(large.join("6vxx.bcif")),
        ) {
            out.push(Fixture {
                name: "6vxx",
                pdb,
                cif,
                bcif,
                has_pdb: true,
                has_bcif: true,
            });
        }
        if let Ok(cif) = std::fs::read_to_string(large.join("3j3q.cif")) {
            out.push(Fixture {
                name: "3j3q",
                pdb: String::new(),
                cif,
                bcif: Vec::new(),
                has_pdb: false,
                has_bcif: false,
            });
        }
    }

    out
}

fn atom_count(assembly: &Assembly) -> usize {
    assembly.entities().iter().map(|e| e.atom_count()).sum()
}

/// One measured row of the parse table.
struct Row {
    op: &'static str,
    format: &'static str,
    structure: &'static str,
    atoms: usize,
    allocs: usize,
    bytes: usize,
}

impl Row {
    fn per_atom(&self) -> f64 {
        if self.atoms == 0 {
            0.0
        } else {
            self.allocs as f64 / self.atoms as f64
        }
    }
}

/// Run `f` with the counters reset, returning its result plus the allocation
/// delta it produced.
fn measure<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    reset();
    let out = f();
    let (allocs, bytes) = snapshot();
    (out, allocs, bytes)
}

fn parse_rows(fixtures: &[Fixture]) -> Vec<Row> {
    let mut rows = Vec::new();
    for fx in fixtures {
        if fx.has_pdb {
            let (assembly, allocs, bytes) =
                measure(|| Assembly::from_pdb(&fx.pdb).unwrap());
            rows.push(Row {
                op: "parse",
                format: "pdb",
                structure: fx.name,
                atoms: atom_count(&assembly),
                allocs,
                bytes,
            });
        }
        let (assembly, allocs, bytes) =
            measure(|| Assembly::from_mmcif(&fx.cif).unwrap());
        rows.push(Row {
            op: "parse",
            format: "mmcif",
            structure: fx.name,
            atoms: atom_count(&assembly),
            allocs,
            bytes,
        });
        if fx.has_bcif {
            let (assembly, allocs, bytes) =
                measure(|| Assembly::from_bcif(&fx.bcif).unwrap());
            rows.push(Row {
                op: "parse",
                format: "bcif",
                structure: fx.name,
                atoms: atom_count(&assembly),
                allocs,
                bytes,
            });
        }
    }
    rows
}

fn write_rows(fixtures: &[Fixture]) -> Vec<Row> {
    let mut rows = Vec::new();
    for fx in fixtures {
        // Parse outside the measured region so only the write is counted.
        let assembly = Assembly::from_mmcif(&fx.cif).unwrap();
        let atoms = atom_count(&assembly);

        if fx.has_pdb {
            let (_, allocs, bytes) = measure(|| assembly.to_pdb().unwrap());
            rows.push(Row {
                op: "write",
                format: "pdb",
                structure: fx.name,
                atoms,
                allocs,
                bytes,
            });
        }
        let (_, allocs, bytes) = measure(|| assembly.to_mmcif());
        rows.push(Row {
            op: "write",
            format: "mmcif",
            structure: fx.name,
            atoms,
            allocs,
            bytes,
        });
    }
    rows
}

fn print_table(title: &str, rows: &[Row]) {
    println!("\n{title}");
    println!(
        "{:<6} {:<7} {:<10} {:>8} {:>12} {:>14} {:>12}",
        "op", "format", "structure", "atoms", "allocs", "bytes", "allocs/atom"
    );
    println!("{}", "-".repeat(74));
    for r in rows {
        println!(
            "{:<6} {:<7} {:<10} {:>8} {:>12} {:>14} {:>12.2}",
            r.op,
            r.format,
            r.structure,
            r.atoms,
            r.allocs,
            r.bytes,
            r.per_atom(),
        );
    }
}

/// Assert the known atom counts so a parse regression that changes the entity
/// graph surfaces here rather than silently skewing the per-atom metric.
fn check_counts(rows: &[Row]) {
    for r in rows {
        if r.op != "parse" {
            continue;
        }
        let expected = match r.structure {
            "1ubq" => Some(660),
            "4hhb" => Some(4779),
            _ => None,
        };
        if let Some(expected) = expected {
            assert!(
                r.atoms == expected,
                "{} {} parsed {} atoms, expected {}",
                r.format,
                r.structure,
                r.atoms,
                expected,
            );
        }
    }
}

fn main() {
    let fixtures = load_fixtures();

    let parses = parse_rows(&fixtures);
    check_counts(&parses);
    print_table("PARSE allocation metric", &parses);

    let writes = write_rows(&fixtures);
    print_table("WRITE allocation metric", &writes);

    println!(
        "\nAtom-count sanity: 1ubq=660, 4hhb=4779 (asserted above; parse \
         aborts on mismatch)."
    );
}
