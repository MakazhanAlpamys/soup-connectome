from pathlib import Path

import pytest

from soup_connectome.errors import GraphValidationError
from soup_connectome.graph.example import example_graph, example_simulation_config
from soup_connectome.graph.format import load_artifact, open_artifact, write_artifact
from soup_connectome.sim.runtime import run_graph


def test_streamed_reference_matches_resident_bit_for_bit() -> None:
    graph = example_graph()
    config = example_simulation_config()
    resident = run_graph(
        graph,
        config,
        timesteps=4,
        initial_potentials=(32767, 0, 0, 0),
        residency="resident",
    )
    streamed = run_graph(
        graph,
        config,
        timesteps=4,
        initial_potentials=(32767, 0, 0, 0),
        residency="streamed",
    )

    assert streamed == resident


def test_disk_streamed_artifact_matches_materialized_graph(tmp_path: Path) -> None:
    artifact_path = write_artifact(example_graph(), tmp_path / "example.scx")
    config = example_simulation_config()
    materialized = load_artifact(artifact_path).graph

    resident = run_graph(
        materialized,
        config,
        timesteps=4,
        initial_potentials=(32767, 0, 0, 0),
        residency="resident",
    )
    streamed = run_graph(
        open_artifact(artifact_path),
        config,
        timesteps=4,
        initial_potentials=(32767, 0, 0, 0),
        residency="streamed",
    )

    assert streamed == resident


def test_disk_artifact_cannot_claim_resident_execution(tmp_path: Path) -> None:
    artifact_path = write_artifact(example_graph(), tmp_path / "example.scx")

    with pytest.raises(GraphValidationError, match="materialized"):
        run_graph(
            open_artifact(artifact_path),
            example_simulation_config(),
            timesteps=1,
            residency="resident",
        )
