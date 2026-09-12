from __future__ import annotations

import hashlib
from dataclasses import dataclass

from soup_connectome.config import Residency, SimulationConfig
from soup_connectome.errors import GraphValidationError
from soup_connectome.graph.format import ConnectomeGraph
from soup_connectome.sim.lif import NeuronState, advance_neuron, checked_add


@dataclass(frozen=True, slots=True)
class SimulationResult:
    spikes: tuple[tuple[bool, ...], ...]
    final_potentials: tuple[int, ...]
    final_refractory: tuple[int, ...]

    @property
    def spike_count(self) -> int:
        return sum(spike for timestep in self.spikes for spike in timestep)

    @property
    def spike_digest(self) -> str:
        encoded = bytes(1 if spike else 0 for timestep in self.spikes for spike in timestep)
        return hashlib.sha256(encoded).hexdigest()


class _DelayLine:
    def __init__(self, max_delay: int) -> None:
        self._buckets: list[dict[int, int]] = [{} for _ in range(max_delay + 1)]
        self._cursor = 0

    def consume(self) -> dict[int, int]:
        bucket = self._buckets[self._cursor]
        self._buckets[self._cursor] = {}
        return bucket

    def schedule(self, target: int, delay: int, weight: int) -> None:
        bucket_index = (self._cursor + delay) % len(self._buckets)
        bucket = self._buckets[bucket_index]
        bucket[target] = checked_add(bucket.get(target, 0), weight)

    def advance(self) -> None:
        self._cursor = (self._cursor + 1) % len(self._buckets)


def _validate_initial_state(
    graph: ConnectomeGraph,
    initial_potentials: tuple[int, ...] | None,
    initial_refractory: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    potentials = initial_potentials or (0,) * graph.n_neurons
    refractory = initial_refractory or (0,) * graph.n_neurons
    if len(potentials) != graph.n_neurons or len(refractory) != graph.n_neurons:
        raise GraphValidationError("initial state length does not match graph")
    return potentials, refractory


def _schedule_spikes(
    graph: ConnectomeGraph,
    spikes: tuple[bool, ...],
    delay_line: _DelayLine,
    residency: Residency,
) -> None:
    if residency is Residency.resident:
        blocks = graph.blocks
    elif residency is Residency.streamed:
        # This iterator is the CPU reference for a future disk-backed loader.
        blocks = tuple(block for block in graph.blocks)
    else:
        raise ValueError(f"unsupported residency: {residency}")

    for block in blocks:
        for row_offset, row in enumerate(block.rows):
            source = block.source_start + row_offset
            if not spikes[source]:
                continue
            for edge in row:
                delay_line.schedule(edge.target, edge.delay, edge.weight)


def run_graph(
    graph: ConnectomeGraph,
    config: SimulationConfig,
    *,
    timesteps: int,
    initial_potentials: tuple[int, ...] | None = None,
    initial_refractory: tuple[int, ...] | None = None,
    residency: Residency | str = Residency.resident,
) -> SimulationResult:
    """Run the canonical CPU state transition for a fixed number of timesteps."""

    if timesteps < 0:
        raise ValueError("timesteps cannot be negative")
    selected_residency = Residency(residency)
    potentials, refractory = _validate_initial_state(graph, initial_potentials, initial_refractory)
    states = [NeuronState(potential, count) for potential, count in zip(potentials, refractory)]
    delay_line = _DelayLine(graph.max_delay)
    spike_history: list[tuple[bool, ...]] = []

    for _ in range(timesteps):
        arrivals = delay_line.consume()
        next_states = []
        for index, state in enumerate(states):
            next_states.append(advance_neuron(state, arrivals.get(index, 0), config))
        states = next_states
        spikes = tuple(state.spike for state in states)
        spike_history.append(spikes)
        _schedule_spikes(graph, spikes, delay_line, selected_residency)
        delay_line.advance()

    return SimulationResult(
        spikes=tuple(spike_history),
        final_potentials=tuple(state.potential for state in states),
        final_refractory=tuple(state.refractory for state in states),
    )
