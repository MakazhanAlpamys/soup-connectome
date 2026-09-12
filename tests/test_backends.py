from pathlib import Path

import pytest

from soup_connectome.backends.base import resolve_backend
from soup_connectome.backends.cuda import availability as cuda_availability
from soup_connectome.backends.webgpu import (
    WGSL_LIF_SHADER,
)
from soup_connectome.backends.webgpu import (
    availability as webgpu_availability,
)
from soup_connectome.errors import BackendNotImplementedError, BackendUnavailableError
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


def test_webgpu_shader_is_integer_only() -> None:
    assert "@compute" in WGSL_LIF_SHADER
    assert "atomicCompareExchangeWeak" in WGSL_LIF_SHADER
    assert "i64" not in WGSL_LIF_SHADER
    assert "f32" not in WGSL_LIF_SHADER


def test_webgpu_example_matches_cpu_when_available() -> None:
    available, reason = webgpu_availability()
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
        webgpu = resolve_backend("webgpu").run(
            graph,
            config,
            timesteps=4,
            initial_potentials=(32767, 0, 0, 0),
            residency="resident",
        )
    except BackendUnavailableError as exc:
        pytest.skip(str(exc))

    assert webgpu == cpu


def test_webgpu_refractory_state_matches_cpu_when_available() -> None:
    available, reason = webgpu_availability()
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
        webgpu = resolve_backend("webgpu").run(
            graph,
            config,
            timesteps=3,
            initial_potentials=(0, 20000, 0, 0),
            initial_refractory=(1, 0, 2, 0),
            residency="resident",
        )
    except BackendUnavailableError as exc:
        pytest.skip(str(exc))

    assert webgpu == cpu


def test_webgpu_streamed_residency_is_explicitly_unimplemented() -> None:
    available, reason = webgpu_availability()
    if not available:
        pytest.skip(reason)

    with pytest.raises(BackendNotImplementedError, match="streamed"):
        resolve_backend("webgpu").run(
            example_graph(),
            example_simulation_config(),
            timesteps=1,
            residency="streamed",
        )
