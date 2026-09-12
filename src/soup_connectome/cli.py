from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from soup_connectome.backends.base import resolve_backend
from soup_connectome.config import Device, Preset, Residency, Scope, SimulationConfig, resolve_axes
from soup_connectome.errors import ConnectomeError
from soup_connectome.graph.example import example_graph, example_simulation_config
from soup_connectome.graph.format import ConnectomeGraph, load_artifact
from soup_connectome.sim.planner import plan_graph

app = typer.Typer(help="Portable runtime for sparse biological connectomes.")
console = Console()


def _load_dataset(dataset: str) -> ConnectomeGraph:
    if dataset == "example":
        return example_graph()
    path = Path(dataset)
    if not path.is_dir():
        raise ValueError(f"dataset is not a registered local artifact: {dataset}")
    return load_artifact(path).graph


def _default_simulation_config(dataset: str) -> SimulationConfig:
    if dataset == "example":
        return example_simulation_config()
    return SimulationConfig(threshold=20000, reset=0, decay_shifts=(2,), refractory_steps=2)


def _initial_potentials(dataset: str, graph: ConnectomeGraph) -> tuple[int, ...]:
    if dataset == "example":
        return (32767, 0, 0, 0)
    return (0,) * graph.n_neurons


def _error(message: str) -> None:
    console.print(f"[red]error[/red] {message}")
    raise typer.Exit(code=1)


@app.command()
def run(
    dataset: str = typer.Option("example", help="Built-in name or local .scx artifact."),
    device: Optional[Device] = typer.Option(None, help="Execution backend."),
    residency: Optional[Residency] = typer.Option(None, help="Graph residency mode."),
    scope: Optional[Scope] = typer.Option(None, help="Graph scope."),
    preset: Optional[Preset] = typer.Option(None, help="Compatibility preset."),
    timesteps: int = typer.Option(
        4, min=0, help="Number of timesteps; fixture default is not a benchmark."
    ),
) -> None:
    try:
        axes = resolve_axes(preset=preset, device=device, residency=residency, scope=scope)
        graph = _load_dataset(dataset)
        backend = resolve_backend(axes.device)
        result = backend.run(
            graph,
            _default_simulation_config(dataset),
            timesteps=timesteps,
            initial_potentials=_initial_potentials(dataset, graph),
            residency=axes.residency,
        )
    except (ConnectomeError, ValueError) as exc:
        _error(str(exc))
    console.print(
        "measured "
        f"device={backend.name} requested_device={axes.device} "
        f"residency={axes.residency} scope={axes.scope} "
        f"timesteps={timesteps} spikes={result.spike_count} digest={result.spike_digest}"
    )


@app.command()
def plan(
    dataset: str = typer.Option("example", help="Built-in name or local .scx artifact."),
    device: Optional[Device] = typer.Option(None, help="Execution backend."),
    residency: Optional[Residency] = typer.Option(None, help="Graph residency mode."),
    scope: Optional[Scope] = typer.Option(None, help="Graph scope."),
    preset: Optional[Preset] = typer.Option(None, help="Compatibility preset."),
    capacity_bytes: Optional[int] = typer.Option(
        None, help="User-supplied capacity for comparison."
    ),
) -> None:
    try:
        axes = resolve_axes(preset=preset, device=device, residency=residency, scope=scope)
        graph = _load_dataset(dataset)
        execution_plan = plan_graph(graph, axes, capacity_bytes=capacity_bytes)
    except (ConnectomeError, ValueError) as exc:
        _error(str(exc))
    console.print(
        f"plan device={execution_plan.device} residency={execution_plan.residency} "
        f"scope={execution_plan.scope} required_bytes={execution_plan.required_bytes} "
        f"capacity={execution_plan.capacity_status} evidence={execution_plan.evidence}"
    )


@app.command()
def benchmark(
    dataset: str = typer.Option("example", help="Built-in name or local .scx artifact."),
    device: Optional[Device] = typer.Option(None, help="Execution backend."),
    residency: Optional[Residency] = typer.Option(None, help="Graph residency mode."),
    scope: Optional[Scope] = typer.Option(None, help="Graph scope."),
    preset: Optional[Preset] = typer.Option(None, help="Compatibility preset."),
    timesteps: int = typer.Option(4, min=0, help="Number of benchmark timesteps."),
) -> None:
    try:
        axes = resolve_axes(preset=preset, device=device, residency=residency, scope=scope)
        graph = _load_dataset(dataset)
        backend = resolve_backend(axes.device)
        started = time.perf_counter()
        result = backend.run(
            graph,
            _default_simulation_config(dataset),
            timesteps=timesteps,
            initial_potentials=_initial_potentials(dataset, graph),
            residency=axes.residency,
        )
        elapsed = time.perf_counter() - started
    except (ConnectomeError, ValueError) as exc:
        _error(str(exc))
    console.print(
        "measured "
        f"device={backend.name} requested_device={axes.device} "
        f"residency={axes.residency} scope={axes.scope} "
        f"timesteps={timesteps} wall_seconds={elapsed:.6f} spikes={result.spike_count}"
    )
