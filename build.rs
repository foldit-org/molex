//! Build script: regenerates `include/molex.h` from the `c_api` module
//! whenever the `c-api` feature is active. Other build configurations
//! are a no-op.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "build scripts fail loud on misconfiguration; clean error \
              handling here would obscure the source of breakage."
)]

fn main() {
    println!("cargo:rerun-if-changed=src/c_api");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "c-api")]
    generate_c_header();
}

#[cfg(feature = "c-api")]
fn generate_c_header() {
    use std::path::PathBuf;

    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_path = PathBuf::from(&crate_dir).join("include").join("molex.h");

    std::fs::create_dir_all(out_path.parent().unwrap()).unwrap();

    let _ = cbindgen::generate(&crate_dir)
        .expect("cbindgen failed to generate molex.h")
        .write_to_file(&out_path);
}
