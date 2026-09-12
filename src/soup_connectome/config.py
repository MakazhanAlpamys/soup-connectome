from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from soup_connectome.errors import ConfigurationError


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    webgpu = "webgpu"


class Residency(str, Enum):
    resident = "resident"
    streamed = "streamed"


class Scope(str, Enum):
    full = "full"
    compact = "compact"


class Preset(str, Enum):
    portable = "portable"
    compact = "compact"
    accelerated = "accelerated"
    streamed = "streamed"
    full = "full"


class RuntimeAxes(BaseModel):
    """The independent device, residency, and graph-scope selections."""

    model_config = ConfigDict(use_enum_values=True)

    device: Device = Device.auto
    residency: Residency = Residency.resident
    scope: Scope = Scope.full


class SimulationConfig(BaseModel):
    """Canonical fixed-point LIF parameters.

    Parameter values are simulation configuration, not biological calibration.
    The example graph supplies deliberately small fixture values; they are not
    measurements.
    """

    model_config = ConfigDict(extra="forbid")

    threshold: int = Field(ge=-(2**31), le=2**31 - 1)
    reset: int = Field(ge=-(2**31), le=2**31 - 1)
    decay_shifts: tuple[int, ...] = Field(min_length=1)
    refractory_steps: int = Field(ge=0, le=2**16 - 1)

    @field_validator("decay_shifts")
    @classmethod
    def validate_decay_shifts(cls, shifts: tuple[int, ...]) -> tuple[int, ...]:
        if any(shift < 0 or shift > 31 for shift in shifts):
            raise ValueError("decay shifts must be between zero and 31")
        return shifts


_PRESET_AXES: dict[Preset, RuntimeAxes] = {
    Preset.portable: RuntimeAxes(
        device=Device.cpu, residency=Residency.resident, scope=Scope.compact
    ),
    Preset.compact: RuntimeAxes(
        device=Device.auto, residency=Residency.resident, scope=Scope.compact
    ),
    Preset.accelerated: RuntimeAxes(
        device=Device.cuda, residency=Residency.resident, scope=Scope.full
    ),
    Preset.streamed: RuntimeAxes(
        device=Device.auto, residency=Residency.streamed, scope=Scope.full
    ),
    Preset.full: RuntimeAxes(device=Device.auto, residency=Residency.resident, scope=Scope.full),
}


def resolve_axes(
    *,
    preset: Preset | str | None = None,
    device: Device | str | None = None,
    residency: Residency | str | None = None,
    scope: Scope | Literal["full", "compact"] | None = None,
) -> RuntimeAxes:
    """Resolve a preset and optional explicit axes without silent overrides."""

    if preset is None:
        return RuntimeAxes(
            device=device or Device.auto,
            residency=residency or Residency.resident,
            scope=scope or Scope.full,
        )

    try:
        selected_preset = Preset(preset)
    except ValueError as exc:
        raise ConfigurationError(f"unknown preset: {preset}") from exc

    selected = _PRESET_AXES[selected_preset]
    explicit = {"device": device, "residency": residency, "scope": scope}
    for name, value in explicit.items():
        if value is None:
            continue
        enum_type = {"device": Device, "residency": Residency, "scope": Scope}[name]
        if enum_type(value) != enum_type(getattr(selected, name)):
            raise ConfigurationError(
                f"preset {selected_preset.value} conflicts with explicit {name}={value}"
            )
    return selected
