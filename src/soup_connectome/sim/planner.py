from __future__ import annotations

from dataclasses import dataclass

from soup_connectome.config import Device, Residency, RuntimeAxes
from soup_connectome.graph.format import ConnectomeGraph, serialize_block


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    device: str
    residency: str
    scope: str
    required_bytes: int
    capacity_status: str
    evidence: str


def _resident_bytes(graph: ConnectomeGraph) -> int:
    neuron_bytes = graph.n_neurons * 16
    block_bytes = sum(len(serialize_block(block)) for block in graph.blocks)
    state_bytes = graph.n_neurons * 12
    return neuron_bytes + block_bytes + state_bytes


def _streamed_bytes(graph: ConnectomeGraph) -> int:
    neuron_bytes = graph.n_neurons * 16
    largest_block = max((len(serialize_block(block)) for block in graph.blocks), default=0)
    state_bytes = graph.n_neurons * 12
    return neuron_bytes + largest_block + state_bytes


def plan_graph(
    graph: ConnectomeGraph,
    axes: RuntimeAxes,
    *,
    capacity_bytes: int | None,
) -> ExecutionPlan:
    """Produce a capacity plan without importing an accelerator runtime."""

    device = Device(axes.device)
    residency = Residency(axes.residency)
    resolved_device = Device.cpu.value if device is Device.auto else device.value
    required_bytes = (
        _streamed_bytes(graph) if residency is Residency.streamed else _resident_bytes(graph)
    )
    if capacity_bytes is None:
        capacity_status = "not tested"
        evidence = "capacity was not supplied; required bytes are a design estimate"
    elif capacity_bytes < required_bytes:
        capacity_status = "insufficient"
        evidence = "comparison against user-supplied capacity"
    else:
        capacity_status = "feasible"
        evidence = "comparison against user-supplied capacity"
    return ExecutionPlan(
        device=resolved_device,
        residency=residency.value,
        scope=str(axes.scope),
        required_bytes=required_bytes,
        capacity_status=capacity_status,
        evidence=evidence,
    )
