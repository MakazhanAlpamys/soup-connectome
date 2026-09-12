from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from soup_connectome.backends.base import resolve_backend
from soup_connectome.config import Device, Preset, Residency, Scope, SimulationConfig, resolve_axes
from soup_connectome.errors import ConnectomeError
from soup_connectome.graph.example import example_graph, example_simulation_config
from soup_connectome.graph.format import (
    ConnectomeGraph,
    DiskGraphArtifact,
    load_artifact,
    open_artifact,
)
from soup_connectome.graph.malecns import (
    MaleCNSColumns,
    WeightQuantizer,
    convert_male_cns,
    feather_columns,
)
from soup_connectome.sim.planner import plan_graph

app = typer.Typer(help="Portable runtime for sparse biological connectomes.")
console = Console()


def _load_dataset(dataset: str, residency: Residency) -> ConnectomeGraph | DiskGraphArtifact:
    if dataset == "example":
        return example_graph()
    path = Path(dataset)
    if not path.is_dir():
        raise ValueError(f"dataset is not a registered local artifact: {dataset}")
    if residency is Residency.streamed:
        return open_artifact(path)
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
        graph = _load_dataset(dataset, axes.residency)
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
        graph = _load_dataset(dataset, axes.residency)
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
        graph = _load_dataset(dataset, axes.residency)
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


@app.command("convert")
def convert(
    weights: Path = typer.Option(..., help="Local MaleCNS connection Feather file."),
    annotations: Path = typer.Option(..., help="Local MaleCNS body annotation Feather file."),
    neurotransmitters: Path = typer.Option(
        ..., help="Local MaleCNS body neurotransmitter Feather file."
    ),
    output: Path = typer.Option(..., help="New .scx artifact directory."),
    annotation_id: str = typer.Option(..., help="Body ID column in annotations."),
    neurotransmitter_id: str = typer.Option(..., help="Body ID column in neurotransmitters."),
    neurotransmitter_name: str = typer.Option(..., help="Neurotransmitter label column."),
    annotation_type: Optional[str] = typer.Option(None, help="Optional annotation type column."),
    annotation_side: Optional[str] = typer.Option(None, help="Optional annotation side column."),
    weight_pre: str = typer.Option("body_pre", help="Presynaptic body ID column."),
    weight_post: str = typer.Option("body_post", help="Postsynaptic body ID column."),
    weight_value: str = typer.Option("weight", help="Connection weight column."),
    sign_mapping: str = typer.Option(
        ..., help='JSON mapping, for example `{"acetylcholine": 1, "gaba": -1}`.'
    ),
    exclude_neurotransmitter: list[str] = typer.Option(
        [],
        "--exclude-neurotransmitter",
        help="Neurotransmitter label whose edges should be omitted; repeatable.",
    ),
    block_size: int = typer.Option(100000, min=1, help="Source block size; chosen estimate."),
    batch_size: int = typer.Option(65536, min=1, help="Feather scan batch size; chosen estimate."),
    sort_chunk_size: int = typer.Option(
        100000, min=1, help="External-sort chunk size; chosen estimate."
    ),
    weight_scale_num: int = typer.Option(1, min=1, help="Quantizer numerator."),
    weight_scale_den: int = typer.Option(1, min=1, help="Quantizer denominator."),
    overflow: str = typer.Option("reject", help="Quantizer overflow policy: reject or saturate."),
    scope: Scope = typer.Option(Scope.full, help="Artifact graph scope."),
    node_filter: str = typer.Option(
        "raw_endpoints",
        help="Node source: raw_endpoints or annotations.",
    ),
    delay_steps: int = typer.Option(
        1, min=1, help="Default edge delay; not a biological measurement."
    ),
) -> None:
    try:
        parsed_sign_mapping = json.loads(sign_mapping)
        if not isinstance(parsed_sign_mapping, dict):
            raise ValueError("sign mapping JSON must be an object")
        columns = MaleCNSColumns(
            weight_pre=weight_pre,
            weight_post=weight_post,
            weight_value=weight_value,
            annotation_id=annotation_id,
            annotation_type=annotation_type,
            annotation_side=annotation_side,
            neurotransmitter_id=neurotransmitter_id,
            neurotransmitter_name=neurotransmitter_name,
        )
        report = convert_male_cns(
            weights,
            annotations,
            neurotransmitters,
            output,
            columns=columns,
            sign_mapping=parsed_sign_mapping,
            block_size=block_size,
            excluded_neurotransmitters=exclude_neurotransmitter,
            quantizer=WeightQuantizer(
                numerator=weight_scale_num,
                denominator=weight_scale_den,
                overflow=overflow,
            ),
            scope=scope,
            node_filter=node_filter,
            delay_steps=delay_steps,
            batch_size=batch_size,
            sort_chunk_size=sort_chunk_size,
        )
    except (ConnectomeError, ValueError, TypeError) as exc:
        _error(str(exc))
    console.print(
        "measured "
        f"converted={report.artifact_path} neurons={report.n_neurons} "
        f"edges={report.n_edges} excluded_edges={report.excluded_edges} "
        f"saturated_weights={report.saturated_weights}"
    )


@app.command("inspect")
def inspect(
    file: Path = typer.Option(..., help="Local Feather file whose schema should be shown."),
) -> None:
    """Show local Feather columns without downloading or scanning table rows."""

    try:
        columns = feather_columns(file)
    except (ConnectomeError, ValueError) as exc:
        _error(str(exc))
    console.print(f"local schema file={file} columns={len(columns)}")
    for column in columns:
        console.print(f"- {column}")
