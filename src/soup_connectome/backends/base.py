from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from soup_connectome.config import Device, Residency, SimulationConfig
from soup_connectome.errors import BackendNotImplementedError
from soup_connectome.graph.format import ConnectomeGraph
from soup_connectome.sim.runtime import SimulationResult


class Backend(Protocol):
    name: str
    available: bool

    def run(
        self,
        graph: ConnectomeGraph,
        config: SimulationConfig,
        *,
        timesteps: int,
        initial_potentials: tuple[int, ...] | None = None,
        residency: Residency | str = Residency.resident,
    ) -> SimulationResult: ...


@dataclass(frozen=True, slots=True)
class BackendInfo:
    name: str
    available: bool
    reason: str | None = None


def resolve_backend(device: Device | str) -> Backend:
    """Resolve only the closed backend allowlist; never silently downgrade."""

    selected = Device(device)
    if selected in (Device.auto, Device.cpu):
        from soup_connectome.backends.cpu import CPUBackend

        return CPUBackend()
    if selected is Device.cuda:
        raise BackendNotImplementedError("cuda backend is not implemented in Phase 1")
    if selected is Device.webgpu:
        raise BackendNotImplementedError("webgpu backend is not implemented in Phase 1")
    raise BackendNotImplementedError(f"backend is not implemented: {selected.value}")
