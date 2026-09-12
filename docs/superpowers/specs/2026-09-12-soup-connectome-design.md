# Soup Connectome Runtime Design Specification

- **Status:** proposed for review
- **Date:** 2026-09-12
- **Scope:** Phase 1 CPU runtime, example graph, CLI, tests, and README
- **Repository:** `MakazhanAlpamys/soup-connectome`

## Summary

`soup-connectome` is a standalone, portable runtime for executing a sparse
biological connectome. The first supported dataset is MaleCNS from Janelia.
The runtime is infrastructure: it provides a reproducible graph format,
fixed-point simulation semantics, backend contracts, and planning tools. It is
not a claim of biological validation and it is not a game or agent demo.

The architecture separates three concerns:

1. a mmap-friendly connectome artifact (`.scx` directory);
2. a backend-independent fixed-point LIF event loop; and
3. backend and residency implementations that must preserve the canonical
   integer semantics.

The initial implementation contains a CPU backend with both resident and
reference-streamed execution. CUDA and WebGPU are specified as contracts and
explicit failure modes so that later implementations cannot silently change
semantics.

## Decisions

### Canonical arithmetic

The canonical simulation representation is the weakest common arithmetic
contract: WebGPU-compatible signed integer operations.

| Quantity | Canonical representation | Status and constraints |
| --- | --- | --- |
| Membrane potential `V` | signed `int32`, with `2^16` voltage quanta per unit | design constant; not a measured biological calibration |
| Synaptic impulse `w` | signed `int16` voltage quanta | format constraint; quantization is dataset/configuration specific and not tested against MaleCNS until a local conversion is run |
| Neuron and edge indices | unsigned `uint32` dense indices | format constraint; the builder must reject a graph that cannot be represented |
| Delay | unsigned `uint16` timestep count | format constraint; every stored delay is positive |
| Refractory counter | unsigned `uint16` timestep count | format constraint |

Leak is expressed without a general multiplication:

```text
V := V - arithmetic_shift_right(V, k)
```

where `k` is a non-negative integer shift selected by the simulation
configuration. A composed leak may use several shifts:

```text
V := V - arithmetic_shift_right(V, k1) - arithmetic_shift_right(V, k2)
```

The shift form is a design choice for cross-backend parity, not a measured
biological law. It restricts leak factors to a discrete family; this trade-off
is intentional and documented. The implementation must define signed
right-shift as an arithmetic, two's-complement shift and test negative
potentials explicitly.

All state updates are checked for the signed `int32` range. The canonical
contract does not permit wraparound. If an operation would overflow, the CPU
backend raises a typed overflow error before committing the affected update.
Future backends must either provide the same observable error or reject the
configuration during planning. Exact atomic integer accumulation is valid only
under this no-overflow invariant.

### Runtime axes

The public configuration has three independent axes:

| Axis | Values | Meaning |
| --- | --- | --- |
| `device` | `auto`, `cpu`, `cuda`, `webgpu` | execution backend selection |
| `residency` | `resident`, `streamed` | whether graph blocks remain resident or are loaded by the scheduler |
| `scope` | `full`, `compact` | graph scope selected by the dataset artifact |

The old five names remain compatibility presets, not mutually exclusive
execution modes. Their mapping is explicit:

| Preset | Axis expansion |
| --- | --- |
| `portable` | `device=cpu`, `residency=resident`, `scope=compact` |
| `compact` | `device=auto`, `residency=resident`, `scope=compact` |
| `accelerated` | `device=cuda`, `residency=resident`, `scope=full` |
| `streamed` | `device=auto`, `residency=streamed`, `scope=full` |
| `full` | `device=auto`, `residency=resident`, `scope=full` |

Preset expansion must be shown in the CLI result. Explicit axis values take
precedence only when the user has not selected a conflicting preset; otherwise
the CLI returns a configuration error. `accelerated` therefore never means
"try CUDA and silently use CPU".

`auto` uses a closed backend allowlist. In Phase 1 the allowlist contains only
the CPU implementation, so `auto` may select CPU. An explicit backend that is
not implemented or unavailable fails with a clear `not implemented` or
`unavailable` error; it never falls back to CPU.

## Goals and non-goals

### Goals

- Execute a deterministic sparse graph on CPU.
- Make the simulation semantics precise enough for bit-for-bit backend tests.
- Store graph data in blocks that can be memory-mapped and streamed later.
- Keep core imports lightweight: no `torch` import at module scope or in the
  CPU-only installation.
- Provide a CLI that can inspect and run local artifacts without downloading
  data.
- Make all performance and capacity claims traceable to measured, estimated,
  or not-tested evidence.

### Non-goals for Phase 1

- CUDA or WebGPU execution.
- Automatic downloading of MaleCNS data.
- EM image volumes or morphology simulation.
- Scientific or biological validation of the LIF parameters.
- A complex UI, cluster scheduler, or integration into the Soup repository.
- A claim that the existing Soup runtime already simulates connectomes.

## Package boundaries

The package is intentionally independent of `soup-cli`:

```text
src/soup_connectome/
  graph/
    format.py       # manifest and binary block validation
    builder.py      # local source data -> .scx
    example.py      # deterministic small graph
  sim/
    lif.py          # canonical fixed-point state transitions
    planner.py      # residency and capacity planning, no torch import
  backends/
    base.py         # backend protocol and result types
    cpu.py          # Phase 1 implementation
    cuda.py         # explicit not-implemented boundary
    webgpu.py       # explicit not-implemented boundary
  cli.py            # Typer commands and Rich rendering
```

`pydantic v2` models are the single source of truth for user configuration and
the manifest. `typer` and `rich` provide the CLI. The supported Python range is
3.10 through 3.12; this is a compatibility target, not a performance claim.

## Connectome artifact format

The artifact is a directory with the `.scx` suffix:

```text
graph.scx/
  manifest.json
  neurons.bin
  blocks/
    block_00000.scb
    block_00001.scb
```

The exact block count is determined by the source graph and configuration; it
is not fixed by this specification.

### Manifest

`manifest.json` is UTF-8 JSON validated by a Pydantic model. It records at
least:

- format identifier and format version;
- dataset identifier, source URL, license, conversion timestamp, and source
  file checksums;
- neuron and edge counts;
- dense-index mapping metadata;
- voltage scale and weight quantization rule;
- dtype and endianness for every binary field;
- minimum and maximum delay;
- scope metadata, including the explicit selection rule for `compact`;
- ordered block descriptors with source range, edge count, byte size, and
  SHA-256 checksum.

Checksums cover the exact bytes of each binary file. The loader verifies the
manifest schema and can optionally verify checksums before simulation. A
checksum mismatch is an error, not a warning.

The manifest must preserve MaleCNS provenance and CC-BY attribution when a
local conversion is made. The converter accepts local files supplied by the
user; it must not fetch the source URLs automatically.

### Neuron table

`neurons.bin` contains a fixed-width little-endian record for each dense neuron
index. It preserves the source body identifier as metadata and stores the
simulation-relevant annotation fields: type/class code,
neurotransmitter/sign code, and side code. Unknown categorical values are
represented explicitly as unknown; they are never guessed from a name.

The source body identifier may be wider than the simulation index. The runtime
uses dense `uint32` indices for graph traversal and retains the source ID for
provenance and lookup.

### Sparse connection blocks

Each `block_*.scb` stores a contiguous range of source neurons in CSR form:

```text
header
row_offsets[source_count + 1]
target_indices[edge_count]
weights[edge_count]
delays[edge_count]
```

The header identifies the block format version, source range, row count, edge
count, and byte order. Every edge record is aligned by array position across
the three arrays. `target_indices` are dense neuron indices; `weights` are
signed fixed-point impulses; `delays` are positive timestep counts.

Blocks are ordered by source range. A loader validates monotonic row offsets,
array lengths, source-range containment, target-index bounds, delay bounds, and
the block checksum. A malformed block is rejected before it can mutate runtime
state.

The format is deliberately append-free and mmap-friendly. A later streaming
backend can load one block at a time without changing the canonical graph
meaning.

## Graph construction

The builder converts local Feather inputs into the artifact. The expected
source inputs are:

- MaleCNS connection weights;
- body annotations; and
- body neurotransmitter annotations.

The conversion joins source identifiers, assigns dense indices, applies the
declared deterministic weight/sign quantization, sorts outgoing edges by the
canonical key, and emits blocks in source order.

The builder must use chunked input and retain at most the current conversion
working block plus bounded metadata required for index assignment. The precise
working-set bound is not yet measured. The builder reports peak RSS in a
diagnostic field when the platform exposes it; it must not present that value as
a guarantee.

The converter never reads or downloads synapse-point, synapse-partner, or EM
volume files. Missing optional annotations produce an explicit error when the
requested graph requires them.

For reproducibility, the conversion configuration, source checksums, sorting
key, quantizer, and sign mapping are recorded in the manifest. A second
conversion with identical inputs and configuration must produce byte-identical
artifact files.

## Canonical simulation semantics

### State

For each dense neuron, the runtime stores:

- membrane potential `V: int32`;
- refractory counter: unsigned integer timestep count; and
- current spike bit.

The runtime also stores scheduled integer arrivals in a delay line. The delay
line capacity is derived from the maximum edge delay in the manifest; it is not
a hard-coded biological value.

### One timestep

The following order is normative:

1. Consume the arrival bucket for the current timestep and add its integer
   impulses to the target neurons with checked `int32` arithmetic.
2. For each neuron with a nonzero refractory counter, decrement the counter,
   keep the neuron at reset potential, and do not integrate the consumed input.
3. For each non-refractory neuron, apply the configured leak using the
   arithmetic shift rule and compare the resulting potential with threshold.
4. For every threshold crossing, emit exactly one spike bit, set potential to
   reset, and set the refractory counter to the configured value. Otherwise,
   clear the spike bit.
5. For each emitted source spike, traverse its CSR out-edges and add the
   signed weight to the delay-line bucket at `current_timestep + delay`.
6. Advance the timestep and expose the immutable spike vector for that step.

The order is a semantic contract, not an optimization hint. The example graph
and golden fixture must exercise positive and negative impulses, a delayed
arrival, leak on a negative value, thresholding, reset, and refractoriness.

The first Phase 1 implementation may use a straightforward full-neuron scan.
Event-driven edge traversal is still the graph contract: only sources whose
spike bit is set are traversed.

### Resident and streamed equivalence

Resident execution has all blocks available to the backend. Streamed execution
loads blocks through a scheduler. Both must feed the same canonical timestep
transition and produce identical spike vectors and final state for the same
initial state, graph, and configuration.

The delay line is essential to the streaming design: a spike emitted at one
timestep is not delivered before its stored positive delay. This allows a block
visit to contribute to future arrival buckets instead of implying that the
whole graph must be reread for every immediate update. The throughput impact is
not tested in Phase 1 and no real-time factor is promised.

## Backend contract

Backends implement a small protocol:

- report a stable backend name and availability reason;
- validate graph dtypes and simulation configuration;
- execute a fixed number of timesteps from a supplied initial state;
- return spike vectors, final state, and structured diagnostics;
- preserve canonical errors for malformed data and overflow.

The registry is a closed allowlist, with explicit entries for `cpu`, `cuda`,
and `webgpu`. Availability checks must be deterministic and side-effect free.
They may inspect installed runtime capabilities, but may not infer support from
an arbitrary device name or silently substitute another backend.

The CPU module must remain importable without optional accelerator packages.
CUDA and WebGPU modules may contain lazy imports inside their availability or
execution functions. Phase 1 exposes their absence as a typed not-implemented
result.

## Planning

`sim/planner.py` is a pure planning layer. It accepts graph metadata, runtime
configuration, and optional user-provided device capacity. It does not import
`torch` at module import time and must be usable on a machine with no GPU.

The plan reports:

- resolved axis values and selected backend;
- resident memory required by graph metadata and runtime buffers;
- whether the requested residency is feasible under the supplied capacity;
- the block order and cache policy that a future streamed backend would use;
- provenance for every capacity or bandwidth input.

No planner output may imply a measured throughput unless a benchmark supplied
that measurement. Constants used as safety margins are labeled as chosen
estimates, not measured bounds. If capacity is unknown, the plan says
`not tested` or `unknown` rather than inventing a default.

The MaleCNS source facts retained in project documentation are: the required
connection, body annotation, and neurotransmitter files total approximately
1.15 GB as a minimum download (verified source fact in the handoff); the
connection file is approximately 1.1 GB; and the larger EM-related files are
not required. The estimate of edge count and packed graph size in the handoff
is explicitly an estimate and must not be presented as a measurement until a
local conversion is performed.

## CLI

The CLI uses Typer and Rich and never uses bare `print()`.

Phase 1 must support:

```text
soup-connectome run --dataset example --device cpu
soup-connectome run --dataset example --device auto
soup-connectome benchmark --device cpu
soup-connectome plan --dataset example --device cpu
```

The final two commands are diagnostic tools: `benchmark` labels results as
measured only after executing a benchmark, while `plan` labels estimates and
unknowns. Dataset paths are local paths or registered built-in names. A name
that is not registered is an error; it does not trigger a download.

`run` accepts the three axes, the preset compatibility option, initial-state
configuration, timestep count, and an optional checksum-verification flag.
For Phase 1, `--device cpu --residency streamed` uses the CPU reference
streamed scheduler. Explicit `cuda` or `webgpu` return a clear not-implemented
result. `--device auto` resolves to CPU because that is the only implemented
backend in this phase. Streamed execution on a future accelerator remains
separate work.

The successful run output includes the resolved configuration, dataset
identity, number of executed timesteps, spike count, and a digest of the spike
train. These are execution results and must be labeled `measured` when shown.
No wall-clock or throughput value is printed unless it was actually measured.

## Testing strategy

Tests are organized around observable contracts:

- **Format round-trip:** build the example artifact, reload it, and compare
  manifest, neuron metadata, edges, weights, and delays.
- **Validation:** reject bad magic, bad checksum, non-monotonic CSR offsets,
  out-of-range indices, zero delays, and dtype/range violations.
- **Arithmetic:** verify signed shift behavior, checked overflow, threshold,
  reset, and refractory transitions.
- **Golden spikes:** compare the example graph's complete spike train and
  final state against a checked-in fixture.
- **Backend resolution:** verify closed-allowlist behavior, explicit
  unavailable-backend errors, and intentional `auto` resolution.
- **Resident/streamed equivalence:** run the same fixture through both
  execution paths and compare every spike vector and final state. The streamed
  path may initially be a test scheduler implementation, but the contract is
  established before accelerator work.
- **Import hygiene:** import planner and CPU modules in an environment where
  optional accelerator packages are absent.

Tests must not require a downloaded MaleCNS artifact. A separately marked
integration test may consume user-provided local data later; it is not part of
the default test suite.

## Error and safety rules

- Resolve paths with `os.path.realpath` and verify containment with
  `os.path.commonpath`; do not rely on `Path.resolve()` for containment.
- Refuse to overwrite an existing artifact unless an explicit future overwrite
  option is added and documented.
- Refuse malformed, checksum-mismatched, or semantically incompatible graph
  blocks before simulation.
- Never silently fall back from an explicitly selected backend.
- Never auto-download datasets or large source files.
- Do not include EM image volumes in any conversion or default data path.
- Keep source license and attribution in generated manifests and documentation.

## Phase 1 acceptance criteria

Phase 1 is complete when all of the following are true:

1. The example graph can be built, loaded, and executed through the CLI on CPU.
2. The fixed-point timestep contract is covered by unit tests and a golden
   spike fixture.
3. A malformed artifact and an arithmetic overflow produce typed, actionable
   errors.
4. The CLI reports resolved device, residency, and scope without claiming
   unsupported CUDA or WebGPU execution.
5. Planning works without importing `torch` or requiring a GPU.
6. The default test suite passes without network access or downloaded MaleCNS
   files.
7. README explains installation, example execution, artifact provenance,
   explicit limitations, and the measured/estimated/not-tested status of
   quantitative claims.

## Open items for implementation review

These items are intentionally deferred to the implementation plan rather than
silently decided here:

- exact Pydantic field names and serialization aliases;
- exact binary header layout and alignment details;
- the example graph topology and its golden spike fixture;
- the integer weight quantizer configuration for a future MaleCNS conversion;
- the cache replacement policy for a real streamed backend;
- CUDA and WebGPU kernel structure;
- a scientifically justified mapping from MaleCNS annotations to LIF
  parameters.

## References

- [MaleCNS](https://male-cns.janelia.org/)
- [MaleCNS download](https://male-cns.janelia.org/download/)
- [Google Research announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [MaleCNS article](https://doi.org/10.1016/j.cell.2026.08.015)
- [Soup](https://github.com/MakazhanAlpamys/Soup)
- [Existing LIF implementation](https://github.com/eonsystemspbc/fly-brain)
- [Browser simulation](https://huggingface.co/spaces/Xenova/fruit-fly-simulation)
- [MaleCNS experiment](https://github.com/dzhng/fly-escape)
