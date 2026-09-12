from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from soup_connectome.config import Scope
from soup_connectome.graph.format import (
    ConnectomeGraph,
    Edge,
    GraphBlock,
    NeuronRecord,
    write_artifact,
    write_artifact_from_blocks,
)


def graph_from_rows(
    n_neurons: int,
    rows: Iterable[tuple[int, Edge]],
    *,
    block_size: int,
    neurons: tuple[NeuronRecord, ...] | None = None,
) -> ConnectomeGraph:
    """Build a graph from locally supplied, source-ordered edge rows.

    The iterable must be sorted by source and each source may appear in any
    number of consecutive rows. This helper does not download or infer source
    data; it is the small deterministic core used by future Feather adapters.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if neurons is None:
        neurons = tuple(NeuronRecord(external_id=index) for index in range(n_neurons))
    return ConnectomeGraph(
        n_neurons=n_neurons,
        neurons=neurons,
        blocks=tuple(iter_blocks_from_rows(n_neurons, rows, block_size=block_size)),
    )


def iter_blocks_from_rows(
    n_neurons: int,
    rows: Iterable[tuple[int, Edge]],
    *,
    block_size: int,
) -> Iterable[GraphBlock]:
    """Yield source blocks while retaining only the current block in memory."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    row_iter = iter(rows)
    pending = next(row_iter, None)
    for source_start in range(0, n_neurons, block_size):
        source_end = min(source_start + block_size, n_neurons)
        block_rows = []
        for source in range(source_start, source_end):
            source_edges = []
            while pending is not None and pending[0] == source:
                source_edges.append(pending[1])
                pending = next(row_iter, None)
            if pending is not None and pending[0] < source:
                raise ValueError("rows must be ordered by non-decreasing source")
            block_rows.append(tuple(source_edges))
        yield GraphBlock(source_start=source_start, rows=tuple(block_rows))
    if pending is not None:
        raise ValueError(f"source index is outside graph: {pending[0]}")


def build_artifact(
    graph: ConnectomeGraph,
    destination: Path,
    *,
    dataset_id: str = "local",
    source_url: str | None = None,
    license: str | None = None,
    scope: Scope = Scope.full,
    conversion_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a local graph using the canonical artifact writer."""

    return write_artifact(
        graph,
        destination,
        dataset_id=dataset_id,
        source_url=source_url,
        license=license,
        scope=scope,
        conversion_metadata=conversion_metadata,
    )


def build_artifact_from_rows(
    n_neurons: int,
    rows: Iterable[tuple[int, Edge]],
    destination: Path,
    *,
    block_size: int,
    neurons: Iterable[NeuronRecord] | None = None,
    dataset_id: str = "local",
    source_url: str | None = None,
    license: str | None = None,
    scope: Scope = Scope.full,
    conversion_metadata: dict[str, Any] | None = None,
) -> Path:
    """Build a local artifact from source-ordered rows without whole-graph buffering."""

    if neurons is None:
        neurons = tuple(NeuronRecord(external_id=index) for index in range(n_neurons))
    return write_artifact_from_blocks(
        n_neurons,
        neurons,
        iter_blocks_from_rows(n_neurons, rows, block_size=block_size),
        destination,
        dataset_id=dataset_id,
        source_url=source_url,
        license=license,
        scope=scope,
        conversion_metadata=conversion_metadata,
    )
