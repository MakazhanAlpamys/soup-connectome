# WASM validation boundary

This crate exposes the WASM CPU streaming boundary needed by a browser host:

- `validate_config` checks the fixed-point configuration limits;
- `validate_scb_block` validates the little-endian `.scb` header, CSR offsets,
  target indices, and positive delays;
- `ConnectomeRuntime` executes the canonical fixed-point LIF transition and
  accepts one validated source block at a time through `schedule_block`;
- `runtime_version` reports the package version.

The WebGPU state transition and sparse scheduling remain in the shared WGSL
module used by the Python and browser hosts. The WASM runtime is the CPU
fallback; it does not duplicate the WebGPU kernels.

Build the optional browser package from this directory:

```bash
wasm-pack build --target web --out-dir pkg --release
```

`pkg/` and Cargo build output are local build products and are intentionally not
committed. A browser-compatible wasm32 build is verified in CI/local toolchain
when the target is installed; browser execution and performance are `not tested`.
