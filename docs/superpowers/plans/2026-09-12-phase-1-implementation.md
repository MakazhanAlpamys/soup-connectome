# Soup Connectome Runtime — Phase 1 Implementation Plan

## Goal

Deliver a runnable CPU-first package that makes the fixed-point simulation
contract executable and testable without downloading MaleCNS data. The example
graph is the first end-to-end artifact; the real-dataset converter is kept
local-input-only and is not part of the default test path.

## Constraints

- Preserve the approved `int32` fixed-point semantics and shift-only leak.
- Keep `torch` and accelerator packages out of core imports and dependencies.
- Use Pydantic for configuration and manifests, Typer/Rich for the CLI.
- Keep backend selection explicit and allowlist-based.
- Make every capacity or performance statement measured, estimated, or
  `not tested`.
- Use `apply_patch` for source edits and keep each implementation increment
  small enough to verify independently.

## Work sequence

### Add project scaffolding

Create `pyproject.toml` with the supported Python range, core dependencies,
optional development dependencies, the `soup-connectome` console script, and
Ruff configuration. Add the package and test directories with no import-time
optional backend dependencies.

Verification: install the project in editable mode if needed, import the public
package, and run the empty test collection.

### Specify configuration and errors through tests

Write tests for the three runtime axes, compatibility preset expansion, invalid
combinations, closed backend names, and typed errors for unsupported explicit
backends. Then implement Pydantic configuration models and small domain errors.

Verification: the tests prove that `auto` is intentional selection while an
explicit unavailable backend never falls back to CPU.

### Specify the graph model and binary format through tests

Write tests for an in-memory graph, deterministic block ordering, manifest
round-trip, binary block round-trip, checksums, path containment, and rejection
of malformed offsets, indices, delays, and integer ranges. Implement the
Pydantic manifest, immutable graph records, little-endian CSR block writer and
reader, and a builder that emits one block at a time from an iterable of source
rows.

Verification: a graph written twice from the same ordered input is byte-stable;
corruption fails before simulation.

### Build the deterministic example graph

Write the example fixture test before the implementation. The graph must
exercise an excitatory impulse, an inhibitory impulse, a delayed arrival,
negative-potential leak, threshold/reset, and refractoriness. Keep topology and
parameters intentionally small and record them as design-fixture constants,
not biological measurements. Implement a helper that returns the in-memory
graph and a helper that writes its `.scx` artifact.

Verification: the fixture test records the exact spike vectors and final state;
no network or external data is used.

### Specify and implement canonical LIF semantics

Write unit tests for signed arithmetic shift, checked `int32` addition,
arrival-before-leak order, threshold comparison, reset, refractory behavior,
and positive delay scheduling. Implement pure functions for checked arithmetic,
one-neuron state transition, and delay-line operations. Keep the simulator
state serializable so the same state can be passed to another backend.

Verification: all edge cases are asserted with exact integer values and the
golden example train passes.

### Implement CPU resident and reference streamed execution

Write parity tests first. Implement resident CSR traversal and a CPU reference
streamed scheduler that reads source blocks on demand while using the same
canonical transition function. The streamed scheduler is a correctness
reference, not a throughput claim. Its output must equal resident output for
every timestep and final state.

Verification: resident and streamed runs have identical spike vectors, state,
and digest on the example fixture.

### Add planner and backend registry

Write tests for planning without `torch`, known versus unknown capacity, and
the backend availability matrix. Implement pure planning data structures,
closed registry resolution, CPU backend, and typed CUDA/WebGPU not-implemented
boundaries with lazy optional imports.

Verification: importing planner and CPU modules succeeds with accelerator
packages absent; explicit unsupported selections produce actionable errors.

### Add CLI and README

Write CLI runner tests for `run`, `plan`, and `benchmark`, including Rich output
and nonzero failure paths. Implement Typer commands with resolved axis output,
measured run diagnostics, and no download behavior. Update README with install,
example usage, artifact format, provenance, limitations, and evidence labels.

Verification: invoke the installed console script and `python -m
soup_connectome` against the example dataset; run the full test suite offline.

## Expected files

```text
pyproject.toml
src/soup_connectome/
  __init__.py
  __main__.py
  cli.py
  config.py
  errors.py
  graph/{__init__,builder,example,format,malecns}.py
  sim/{__init__,lif,planner,runtime}.py
  backends/{__init__,base,cpu,cuda,webgpu}.py
tests/
  conftest.py
  test_config.py
  test_format.py
  test_example.py
  test_lif.py
  test_runtime.py
  test_planner.py
  test_cli.py
```

## Definition of done

- The full default test suite passes offline.
- The example can be run from the CLI on CPU.
- Format, arithmetic, backend selection, planner, and resident/streamed parity
  contracts are covered by tests.
- No optional accelerator dependency is imported at module scope.
- README and CLI distinguish measured output from estimates and untested
  behavior.
- The local MaleCNS adapter exposes schema inspection and deterministic
  Feather-to-`.scx` conversion without network access.
- Source changes are reviewed with `ruff check`, `ruff format --check`, and
  `git diff --check` before commit.
