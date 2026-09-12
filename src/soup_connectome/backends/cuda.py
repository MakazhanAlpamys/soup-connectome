"""Optional CUDA backend using lazy PyTorch imports.

The core package does not import PyTorch. This backend is intentionally
resident-only in its first implementation; streamed CUDA scheduling remains a
separate task because it needs a GPU block cache and prefetch policy.
"""

from __future__ import annotations

from typing import Any

from soup_connectome.config import Residency, SimulationConfig
from soup_connectome.errors import (
    BackendNotImplementedError,
    BackendUnavailableError,
    FixedPointOverflowError,
    GraphValidationError,
)
from soup_connectome.graph.format import ConnectomeGraph
from soup_connectome.sim.runtime import GraphSource, SimulationResult

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
UINT16_MAX = 2**16 - 1


def availability() -> tuple[bool, str]:
    """Report CUDA availability without importing PyTorch at module scope."""

    try:
        import torch
    except ImportError:
        return False, "CUDA backend requires the optional torch dependency"
    try:
        if not torch.cuda.is_available():
            return False, "CUDA runtime is unavailable on this host"
    except Exception as exc:  # pragma: no cover - depends on local driver state
        return False, f"CUDA availability check failed: {exc}"
    return True, "CUDA runtime is available"


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "CUDA backend requires the optional torch dependency"
        ) from exc
    return torch


def _checked_tensor(torch: Any, values: Any, label: str) -> Any:
    overflow = torch.any((values < INT32_MIN) | (values > INT32_MAX)).item()
    if overflow:
        raise FixedPointOverflowError(f"cuda {label} exceeds signed int32 range")
    return values


def _graph_tensors(torch: Any, graph: ConnectomeGraph) -> tuple[Any, Any, Any, Any]:
    offsets = [0]
    targets: list[int] = []
    weights: list[int] = []
    delays: list[int] = []
    for block in graph.blocks:
        for row in block.rows:
            for edge in row:
                targets.append(edge.target)
                weights.append(edge.weight)
                delays.append(edge.delay)
            offsets.append(len(targets))
    device = torch.device("cuda")
    return (
        torch.tensor(offsets, dtype=torch.int64, device=device),
        torch.tensor(targets, dtype=torch.int64, device=device),
        torch.tensor(weights, dtype=torch.int64, device=device),
        torch.tensor(delays, dtype=torch.int64, device=device),
    )


def _validate_initial_state(
    graph: ConnectomeGraph,
    initial_potentials: tuple[int, ...] | None,
    initial_refractory: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    potentials = initial_potentials or (0,) * graph.n_neurons
    refractory = initial_refractory or (0,) * graph.n_neurons
    if len(potentials) != graph.n_neurons or len(refractory) != graph.n_neurons:
        raise GraphValidationError("initial state length does not match graph")
    if any(not INT32_MIN <= value <= INT32_MAX for value in potentials):
        raise FixedPointOverflowError("initial potential exceeds signed int32 range")
    if any(value < 0 or value > UINT16_MAX for value in refractory):
        raise ValueError("initial refractory counter is outside uint16 range")
    return potentials, refractory


def _run_resident(
    torch: Any,
    graph: ConnectomeGraph,
    config: SimulationConfig,
    *,
    timesteps: int,
    initial_potentials: tuple[int, ...] | None,
    initial_refractory: tuple[int, ...] | None,
) -> SimulationResult:
    potentials, refractory = _validate_initial_state(graph, initial_potentials, initial_refractory)
    device = torch.device("cuda")
    potential_tensor = torch.tensor(potentials, dtype=torch.int32, device=device)
    refractory_tensor = torch.tensor(refractory, dtype=torch.int32, device=device)
    row_offsets, targets, weights, delays = _graph_tensors(torch, graph)
    bucket_count = graph.max_delay + 1
    delay_line = torch.zeros((bucket_count, graph.n_neurons), dtype=torch.int64, device=device)
    cursor = 0
    spike_history: list[tuple[bool, ...]] = []

    for _ in range(timesteps):
        arrivals = delay_line[cursor].clone()
        delay_line[cursor].zero_()
        _checked_tensor(torch, arrivals, "arrival bucket")

        refractory_mask = refractory_tensor != 0
        next_refractory = torch.where(
            refractory_mask, refractory_tensor - 1, torch.zeros_like(refractory_tensor)
        )
        raw_candidate = potential_tensor.to(torch.int64) + arrivals
        candidate = torch.where(
            refractory_mask,
            torch.full_like(raw_candidate, config.reset),
            raw_candidate,
        )
        _checked_tensor(torch, candidate, "potential update")
        leaked = candidate
        for shift in config.decay_shifts:
            leaked = leaked - (leaked >> shift)
            _checked_tensor(torch, leaked, "leak update")

        threshold = leaked >= config.threshold
        spikes = (~refractory_mask) & threshold
        next_potential = torch.where(
            refractory_mask | spikes,
            torch.full_like(potential_tensor, config.reset),
            leaked.to(torch.int32),
        )
        refractory_tensor = torch.where(
            spikes,
            torch.full_like(refractory_tensor, config.refractory_steps),
            next_refractory,
        )
        potential_tensor = next_potential
        spike_history.append(tuple(bool(value) for value in spikes.cpu().tolist()))

        for source in torch.nonzero(spikes, as_tuple=False).flatten().cpu().tolist():
            start = int(row_offsets[source].item())
            end = int(row_offsets[source + 1].item())
            if start == end:
                continue
            edge_delays = delays[start:end]
            bucket_indices = (cursor + edge_delays) % bucket_count
            delay_line.index_put_(
                (bucket_indices, targets[start:end]),
                weights[start:end],
                accumulate=True,
            )
        _checked_tensor(torch, delay_line, "delay-line accumulation")
        cursor = (cursor + 1) % bucket_count

    return SimulationResult(
        spikes=tuple(spike_history),
        final_potentials=tuple(int(value) for value in potential_tensor.cpu().tolist()),
        final_refractory=tuple(int(value) for value in refractory_tensor.cpu().tolist()),
    )


class CUDABackend:
    name = "cuda"

    def __init__(self) -> None:
        self.available, self.reason = availability()

    def run(
        self,
        graph: GraphSource,
        config: SimulationConfig,
        *,
        timesteps: int,
        initial_potentials: tuple[int, ...] | None = None,
        initial_refractory: tuple[int, ...] | None = None,
        residency: Residency | str = Residency.resident,
    ) -> SimulationResult:
        if not self.available:
            raise BackendUnavailableError(self.reason)
        if timesteps < 0:
            raise ValueError("timesteps cannot be negative")
        if Residency(residency) is Residency.streamed:
            raise BackendNotImplementedError(
                "cuda streamed residency is not implemented; use resident residency"
            )
        if not isinstance(graph, ConnectomeGraph):
            raise GraphValidationError("cuda resident execution requires a materialized graph")
        torch = _require_torch()
        try:
            return _run_resident(
                torch,
                graph,
                config,
                timesteps=timesteps,
                initial_potentials=initial_potentials,
                initial_refractory=initial_refractory,
            )
        except RuntimeError as exc:
            raise BackendUnavailableError(f"CUDA execution failed on this host: {exc}") from exc
