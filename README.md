# soup-connectome

Portable runtime for executing sparse biological connectomes with deterministic
fixed-point semantics.

The first dataset target is MaleCNS, the male fruit-fly connectome released by
Janelia. This project is infrastructure: a portable graph artifact, a
backend-independent integer simulation contract, and explicit residency and
device planning. It does not claim biological validation and it is not a game
demo.

## Phase 1 status

Implemented:

- CPU fixed-point LIF runtime;
- resident and CPU reference-streamed execution;
- mmap-friendly `.scx` graph artifacts with CSR `.scb` blocks;
- deterministic example graph and golden spike train;
- planner, backend allowlist, and Typer/Rich CLI;
- format, arithmetic, CLI, and resident/streamed parity tests.

Not implemented:

- CUDA kernels;
- WebGPU/WASM execution;
- automatic MaleCNS downloads or a networked data pipeline;
- biological or scientific validation of LIF parameters;
- morphology and EM-volume simulation.

Explicit `cuda` and `webgpu` selections fail with `not implemented`; they never
fall back silently to CPU. `auto` currently resolves to CPU because that is the
only implemented backend.

## Install

```bash
python -m pip install -e ".[dev]"
```

The core package does not depend on PyTorch or another accelerator runtime.

## Run the example

```bash
soup-connectome run --dataset example --device cpu
soup-connectome run --dataset example --device cpu --residency streamed
soup-connectome plan --dataset example --device cpu
soup-connectome benchmark --device auto
```

Run output is labeled `measured` because it comes from an actual execution.
Planner capacity is labeled `not tested` when no capacity is supplied; graph
byte requirements are design estimates, not a hardware benchmark.

## Canonical arithmetic

The cross-backend contract uses signed `int32` membrane potential with a design
scale of `2^16` voltage quanta per unit, signed `int16` synaptic impulses, and
positive integer timestep delays. These are representation choices, not
measured biological constants.

Leak uses arithmetic shifts rather than general multiplication:

```text
V := V - (V >> k)
```

The resulting leak factors are discrete by design. Signed `int32` overflow is
checked and rejected; wraparound is not part of the runtime semantics.

## Graph artifacts

An artifact is a local directory:

```text
graph.scx/
  manifest.json
  neurons.bin
  blocks/
    block_00000.scb
```

The manifest records source provenance, license, dtypes, quantization metadata,
scope, block ranges, and SHA-256 checksums. Blocks use source-indexed CSR so a
future backend can load outgoing connections on demand. The current example
writer accepts local in-memory graph data; it does not fetch remote files.

For MaleCNS, use the connection weights, body annotations, and
neurotransmitter files described by the [Janelia download page](https://male-cns.janelia.org/download/).
The handoff verified an approximately `1.15 GB` minimum input set; this is a
verified source fact recorded for UX planning, not a local transfer measurement.
The larger EM-related files are not required and must not be downloaded by this
project.

## Evidence policy

Quantitative claims are tagged in code and documentation:

- `measured` — produced by an actual local run or benchmark;
- `estimated` — a design calculation or planning assumption;
- `not tested` — no local evidence yet.

In particular, this repository currently has no measured MaleCNS conversion
size, accelerator throughput, real-time factor, or biological fidelity result.

## References

- [MaleCNS](https://male-cns.janelia.org/)
- [MaleCNS download](https://male-cns.janelia.org/download/)
- [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [Soup](https://github.com/MakazhanAlpamys/Soup)
- [Existing LIF implementation](https://github.com/eonsystemspbc/fly-brain)
