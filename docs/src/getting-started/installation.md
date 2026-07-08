# Installation

## As a Rust dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
molex = "0.3"
```

To enable Python bindings (PyO3 + NumPy, Biotite-free numpy-column interchange via `PyAtomTable`, with optional AtomWorks-style vocabulary columns):

```toml
[dependencies]
molex = { version = "0.3", features = ["python"] }
```

Other optional features:

- `c-api` -- C ABI, `libmolex.a`, and the generated `include/molex.h`
- `xtal` -- crystallographic refinement pipeline (density, scaling, sigma-A, FFT)
- `minimization` -- B-factor refinement (argmin), on top of `xtal`
- `gpu` -- GPU density/refinement backend (cubecl + wgpu), on top of `xtal`

## As a Python package

```bash
cd crates/molex
maturin develop --release --features python

python -c "import molex; print('OK')"
```

## Dependencies

molex pulls in:

- **glam** -- 3D math (`Vec3`, `Mat4`)
- **ndarray** -- 3D arrays for density grids
- **rmp** -- MessagePack for BinaryCIF decoding
- **flate2** -- gzip decompression
- **thiserror** -- error types

With the `python` feature:

- **pyo3** -- Python FFI
- **numpy** -- NumPy array interop
