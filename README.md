# soup-connectome

Portable runtime for executing sparse biological connectomes with deterministic
fixed-point semantics.

The first dataset target is MaleCNS, the male fruit-fly connectome released by
Janelia. This project is infrastructure: a portable graph artifact, a
backend-independent integer simulation contract, and explicit residency and
device planning. It does not claim biological validation and it is not a game
demo.

## Current status

Implemented:

- CPU fixed-point LIF runtime;
- resident and disk-backed CPU streamed execution;
- optional resident and streamed CUDA backend with fixed-point tensor execution;
- mmap-friendly `.scx` graph artifacts with CSR `.scb` blocks;
- local MaleCNS Feather adapter with an explicit curated-node filter;
- deterministic example graph and golden spike train;
- planner, backend allowlist, and Typer/Rich CLI;
- format, arithmetic, CLI, and resident/streamed parity tests.

Not implemented:

- WebGPU/WASM execution;
- automatic MaleCNS downloads or a networked data pipeline;
- biological or scientific validation of LIF parameters;
- morphology and EM-volume simulation.

Explicit `webgpu` selection fails with `not implemented`; explicit `cuda`
selects the optional CUDA backend and reports a typed unavailable error when
the runtime or host GPU cannot execute it. Neither selection falls back
silently to CPU. `auto` currently resolves to CPU by design.

## Install

```bash
python -m pip install -e ".[dev,data]"
# Optional CUDA backend:
python -m pip install -e ".[dev,data,cuda]"
```

The core package does not depend on PyTorch or another accelerator runtime.
The CUDA extra uses PyTorch lazily; importing the planner, CPU backend, or graph
format does not import it.

## Run the example

```bash
soup-connectome run --dataset example --device cpu
soup-connectome run --dataset example --device cpu --residency streamed
soup-connectome run --dataset example --device cuda --residency streamed
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
scope, block ranges, and SHA-256 checksums. Blocks use source-indexed CSR.
Resident loading materializes all blocks, while `--residency streamed` opens
the artifact lazily and reads/validates one block at a time. This is a
correctness and memory-shape implementation, not a throughput claim; streamed
I/O performance is `not tested`. The CUDA streamed path transfers one source
block at a time and currently has no prefetch cache. The current example writer
accepts local in-memory graph data; it does not fetch remote files.

## Convert local MaleCNS files

The adapter requires three files supplied by the user: connection weights, body
annotations, and body neurotransmitters. It never downloads them. Inspect each
local Feather schema first:

```bash
soup-connectome inspect --file path/to/body-annotations.feather
soup-connectome inspect --file path/to/body-neurotransmitters.feather
```

For the verified MaleCNS v1.0 files, the measured column mapping is:

- weights: `body_pre`, `body_post`, `weight`;
- annotations: `bodyId`, `type`, `somaSide`;
- neurotransmitters: `body`, `consensus_nt`.

The official weights table is a segment-to-segment graph, while the annotation
table is a curated subset. For the first local artifact, use
`--node-filter annotations` so the graph contains only curated annotation IDs.
This is an explicit data-scope choice, not a claim that the raw segment graph is
biologically incomplete. `raw_endpoints` remains available, but full-scale raw
endpoint conversion is currently `not tested` for memory and runtime.

Use an explicit sign mapping and explicitly exclude labels for which this
fixed-point runtime has no globally verified static sign:

```bash
soup-connectome convert \
  --weights data/male-cns/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --annotations data/male-cns/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --neurotransmitters data/male-cns/body-neurotransmitters-male-cns-v1.0.feather \
  --output artifacts/male-cns-v1.0-annotated.scx \
  --annotation-id bodyId \
  --annotation-type type \
  --annotation-side somaSide \
  --neurotransmitter-id body \
  --neurotransmitter-name consensus_nt \
  --sign-mapping '{"acetylcholine": 1, "gaba": -1, "glutamate": 1}' \
  --exclude-neurotransmitter unclear \
  --exclude-neurotransmitter dopamine \
  --exclude-neurotransmitter histamine \
  --exclude-neurotransmitter octopamine \
  --exclude-neurotransmitter serotonin \
  --exclude-neurotransmitter unknown \
  --scope full \
  --node-filter annotations
```

Rows are scanned in batches, externally sorted by source, and emitted as source
blocks. Full `raw_endpoints` conversion builds a temporary disk-backed uint64
ID index instead of keeping the complete external-ID-to-dense-ID mapping in
RAM; the temporary SQLite file is removed after conversion. This bounds the
converter's ID-index memory pressure, but the full-scale RAM and runtime impact
is not tested. The default delay and chunk sizes are chosen representation
estimates, not measured biological or performance bounds. Weight overflow
rejects the conversion by default; `--overflow saturate` records the saturation
count in the manifest. `--exclude-neurotransmitter` is repeatable, and
excluded-edge counts are stored in the manifest.

For MaleCNS, use the connection weights, body annotations, and
neurotransmitter files described by the [Janelia download page](https://male-cns.janelia.org/download/).
The local download measured `151,856,684` weight rows, `211,577` annotation rows,
and `1,835,518` neurotransmitter rows. The three local Feather files measured
`1,109,008,094` bytes in total. These are local measurements for this exact
download, not universal hardware requirements.
The larger EM-related files are not required and must not be downloaded by this
project.

## Evidence policy

Quantitative claims are tagged in code and documentation:

- `measured` — produced by an actual local run or benchmark;
- `estimated` — a design calculation or planning assumption;
- `not tested` — no local evidence yet.

The local curated conversion is measured at `211,577` neurons,
`24,678,466` included edges, `1,349,920` excluded edges, and `0` saturated
weights. Checksum loading and a one-timestep CPU smoke-run passed. These are
artifact-validation measurements, not accelerator benchmarks or biological
fidelity results. CUDA parity is `not tested` on the current host because
minimal CUDA context allocation returned `CUDA_ERROR_OUT_OF_MEMORY` despite the
driver reporting a visible device.

## References

- [MaleCNS](https://male-cns.janelia.org/)
- [MaleCNS download](https://male-cns.janelia.org/download/)
- [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [Soup](https://github.com/MakazhanAlpamys/Soup)
- [Existing LIF implementation](https://github.com/eonsystemspbc/fly-brain)
