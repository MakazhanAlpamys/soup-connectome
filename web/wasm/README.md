# WASM validation boundary

This crate exposes the small validation boundary needed by a browser host:

- `validate_config` checks the fixed-point configuration limits;
- `validate_scb_block` validates the little-endian `.scb` header, CSR offsets,
  target indices, and positive delays;
- `runtime_version` reports the package version.

The WebGPU state transition and sparse scheduling remain in the shared WGSL
module used by the Python and browser hosts. This crate does not claim to be a
complete CPU connectome runtime in WASM.

Build the optional browser package from this directory:

```bash
wasm-pack build --target web --out-dir pkg --release
```

`pkg/` and Cargo build output are local build products and are intentionally not
committed. A browser-compatible wasm32 build is verified in CI/local toolchain
when the target is installed; browser execution and performance are `not tested`.
