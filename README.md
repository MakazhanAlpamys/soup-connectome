# soup-connectome

> Portable runtime for executing sparse biological connectomes across CPU, CUDA, WebGPU, and WASM.

`soup-connectome` turns a sparse connectome into a portable `.scx` artifact and
executes it with deterministic fixed-point LIF semantics. The first dataset
target is [MaleCNS](https://male-cns.janelia.org/), the male fruit-fly
connectome released by Janelia.

This is infrastructure, not a game demo: the project focuses on graph storage,
streaming, backend parity, and explicit device planning. It does not claim
biological validation.

## Status at a glance

| Capability | Status | Evidence |
| --- | --- | --- |
| CPU resident + streamed runtime | `measured` | Full test suite and MaleCNS smoke |
| CUDA resident + streamed backend | implemented | Optional; full-scale CUDA performance `not tested` |
| Python WebGPU resident + streamed backend | `measured` | Example parity and MaleCNS smoke |
| WASM CPU streaming runtime | `measured` | Generated web/node bindings and parity fixture |
| MaleCNS `.scx` artifact | `measured` | `211,577` neurons, `24,678,466` edges |
| Biological/scientific validation | `not tested` | LIF parameters are runtime configuration |

## Validation snapshot

The local curated MaleCNS artifact contains `211,577` neurons and
`24,678,466` included edges across three streamed CSR blocks.

| Run | Result |
| --- | --- |
| Full-scale, one timestep, zero initial spikes, CPU | `2.245442 s`, `0` spikes — `measured` |
| Full-scale, one timestep, zero initial spikes, Python WebGPU | `2.695649 s`, `0` spikes — `measured` |
| Full-scale, two-timestep active synthetic stress, CPU | `130.364592 s`, spike counts `[1, 319]` — `measured` |
| Same active synthetic stress, Python WebGPU | `128.487143 s`, spike counts `[1, 319]` — `measured` |
| Standard config, four timesteps, seeded source, CPU | `221.537098 s`, spike counts `[1, 0, 0, 0]` — `measured` |
| Same standard benchmark, Python WebGPU | `266.778847 s`, spike counts `[1, 0, 0, 0]` — `measured` |

The active stress configuration uses `threshold=1`, `reset=0`,
`decay_shifts=[31]`, and `refractory_steps=0`; it is a propagation test, not
a biological calibration or representative throughput benchmark.

The reproducible standard benchmark uses `scripts/benchmark_malecns.py` with
`threshold=20000`, `reset=0`, `decay_shifts=[2]`, `refractory_steps=2`,
`--timesteps 4`, `--seed-neuron 0`, and `--seed-potential 30000`.

```bash
python scripts/benchmark_malecns.py --device cpu --residency streamed --timesteps 4
python scripts/benchmark_malecns.py --device webgpu --residency streamed --timesteps 4
```

## Why this runtime

Large sparse graphs create a different deployment problem from a conventional
dense neural model. The runtime keeps neuron state and the delay line resident
while streaming sparse source blocks, so graph residency is an explicit axis:

```text
MaleCNS Feather files
        │
        ▼
local adapter + sign mapping
        │
        ▼
portable .scx artifact
        │
        ├── CPU resident / streamed
        ├── CUDA resident / streamed
        ├── Python WebGPU resident / streamed
        └── browser WebGPU streamed / WASM CPU fallback
```

Streaming is a memory-shape and portability feature. It is not automatically a
throughput guarantee: the current streamed accelerator paths validate and
transfer one active source block at a time and do not use a prefetch cache.

## Quick start

Install the core package and development/data extras:

```bash
python -m pip install -e ".[dev,data]"
```

Optional backends:

```bash
python -m pip install -e ".[dev,data,cuda]"
python -m pip install -e ".[dev,data,webgpu]"
```

Run the deterministic fixture:

```bash
soup-connectome run --dataset example --device cpu
soup-connectome run --dataset example --device cpu --residency streamed
soup-connectome run --dataset example --device webgpu --residency streamed
soup-connectome plan --dataset example --device cpu
```

Explicit `cuda` and `webgpu` never silently fall back to CPU. `auto` resolves
to CPU by design.

## MaleCNS data workflow

The adapter is local-only and never downloads data automatically. It requires:

- connection weights;
- body annotations;
- body neurotransmitters.

For the verified MaleCNS v1.0 download, the measured columns are:

| File | Columns |
| --- | --- |
| weights | `body_pre`, `body_post`, `weight` |
| annotations | `bodyId`, `type`, `somaSide` |
| neurotransmitters | `body`, `consensus_nt` |

Inspect local schemas first:

```bash
soup-connectome inspect --file path/to/body-annotations.feather
soup-connectome inspect --file path/to/body-neurotransmitters.feather
```

The first artifact uses the curated annotation node filter and this explicit
sign mapping:

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

The local download measured `151,856,684` weight rows, `211,577` annotation
rows, `1,835,518` neurotransmitter rows, and `1,109,008,094` bytes across the
three Feather files. These are measurements for this exact download, not
universal hardware requirements. Raw data, generated artifacts, WASM build
outputs, and browser test results are intentionally ignored by Git.

## Runtime contract

The backend-independent simulation contract uses:

- signed `int32` membrane potentials;
- signed `int16` synaptic impulses;
- positive integer timestep delays;
- checked arithmetic with rejected overflow;
- arithmetic-shift leak: `V := V - (V >> k)`.

These are representation choices, not measured biological constants.

## Development and tests

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .

cd web
npm test
npm run test:e2e

cd wasm
cargo fmt --check
cargo test
cargo check --target wasm32-unknown-unknown
```

The repository uses three evidence labels:

- `measured` — produced by an actual local run or benchmark;
- `estimated` — a design calculation or planning assumption;
- `not tested` — no local evidence yet.

## Limitations

- No biological calibration or scientific fidelity claim is made.
- Morphology and EM-volume simulation are out of scope for the current phase.
- Automatic MaleCNS downloads and a networked data pipeline are not included.
- Full-scale active stress is measured for two timesteps with a synthetic
  configuration; longer runs on the Python reference path are `not tested`.
- Browser compatibility across GPU vendors and operating systems is `not tested`.

## Repository map

```text
src/soup_connectome/   runtime, graph format, adapters, backends, CLI
scripts/                reproducible full-scale benchmark entry point
tests/                 Python contract and parity tests
web/src/               browser WebGPU and WASM streaming host
web/wasm/              wasm-bindgen CPU runtime
web/test/              browser parity fixture
docs/                  project documentation
```

## References

- [MaleCNS](https://male-cns.janelia.org/)
- [MaleCNS download](https://male-cns.janelia.org/download/)
- [Soup](https://github.com/MakazhanAlpamys/Soup)
- [Google Research connectomics announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
