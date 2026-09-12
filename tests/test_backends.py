import pytest

from soup_connectome.backends.base import resolve_backend
from soup_connectome.errors import BackendNotImplementedError


def test_cpu_backend_resolves() -> None:
    backend = resolve_backend("cpu")
    assert backend.name == "cpu"
    assert backend.available


def test_explicit_cuda_does_not_fall_back_to_cpu() -> None:
    with pytest.raises(BackendNotImplementedError, match="cuda"):
        resolve_backend("cuda")
