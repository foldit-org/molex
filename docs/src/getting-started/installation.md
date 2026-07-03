# Installation

## As a Rust dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
molex = "0.3"
```

To enable Python bindings (PyO3 + NumPy + AtomWorks interop):

```toml
[dependencies]
molex = { version = "0.3", features = ["python"] }
```

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
