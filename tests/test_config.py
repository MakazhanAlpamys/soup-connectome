import pytest
from pydantic import ValidationError

from soup_connectome.config import (
    Device,
    Residency,
    RuntimeAxes,
    SimulationConfig,
    resolve_axes,
)
from soup_connectome.errors import ConfigurationError


def test_portable_preset_expands_to_approved_axes() -> None:
    assert resolve_axes(preset="portable") == RuntimeAxes(
        device=Device.cpu,
        residency=Residency.resident,
        scope="compact",
    )


def test_explicit_preset_conflict_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="conflicts"):
        resolve_axes(preset="portable", device=Device.cuda)


def test_matching_explicit_preset_axis_is_allowed() -> None:
    assert resolve_axes(preset="portable", device=Device.cpu).device == "cpu"


def test_simulation_config_rejects_invalid_shift() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(
            threshold=1,
            reset=0,
            decay_shifts=(32,),
            refractory_steps=0,
        )
