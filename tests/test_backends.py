from pathlib import Path

import pytest

from soup_connectome.backends.base import resolve_backend
from soup_connectome.backends.cuda import availability as cuda_availability
from soup_connectome.errors import BackendUnavailableError
from soup_connectome.graph.example import example_graph, example_simulation_config
from soup_connectome.graph.format import open_artifact, write_artifact
from soup_connectome.sim.runtime import run_graph


def test_cpu_backend_resolves() -> None:
    backend = resolve_backend("cpu")
    assert backend.name == "cpu"
    assert backend.available


def test_explicit_cuda_does_not_fall_back_to_cpu() -> None:
    available, reason = cuda_availability()
    if available:
        backend = resolve_backend("cuda")
        assert backend.name == "cuda"
    else:
        with pytest.raises(BackendUnavailableError, match="CUDA|cuda"):
            resolve_backend("cuda")
        assert reason


def test_cuda_example_matches_cpu_when_available() -> None:
    available, reason = cuda_availability()
    if not available:
        pytest.skip(reason)

    graph = example_graph()
    config = example_simulation_config()
    cpu = run_graph(
        graph,
        config,
        timesteps=4,
        initial_potentials=(32767, 0, 0, 0),
        residency="resident",
    )
    try:
        cuda = resolve_backend("cuda").run(
            graph,
            config,
            timesteps=4,
            initial_potentials=(32767, 0, 0, 0),
            residency="resident",
        )
    except BackendUnavailableError as exc:
        pytest.skip(str(exc))

    assert cuda == cpu


def test_cuda_refractory_state_matches_cpu_when_available() -> None:
    available, reason = cuda_availability()
    if not available:
        pytest.skip(reason)

    graph = example_graph()
    config = example_simulation_config()
    cpu = run_graph(
        graph,
        config,
        timesteps=3,
        initial_potentials=(0, 20000, 0, 0),
        initial_refractory=(1, 0, 2, 0),
        residency="resident",
    )
    try:
        cuda = resolve_backend("cuda").run(
            graph,
            config,
            timesteps=3,
            initial_potentials=(0, 20000, 0, 0),
            initial_refractory=(1, 0, 2, 0),
            residency="resident",
        )
    except BackendUnavailableError as exc:
        pytest.skip(str(exc))

    assert cuda == cpu


def test_cuda_streamed_artifact_matches_cpu_when_available(tmp_path: Path) -> None:
    available, reason = cuda_availability()
    if not available:
        pytest.skip(reason)

    artifact_path = write_artifact(example_graph(), tmp_path / "example.scx")
    graph = open_artifact(artifact_path)
    config = example_simulation_config()
    cpu = run_graph(
        graph,
        config,
        timesteps=4,
        initial_potentials=(32767, 0, 0, 0),
        residency="streamed",
    )
    try:
        cuda = resolve_backend("cuda").run(
            graph,
            config,
            timesteps=4,
            initial_potentials=(32767, 0, 0, 0),
            residency="streamed",
        )
    except BackendUnavailableError as exc:
        pytest.skip(str(exc))

    assert cuda == cpu
