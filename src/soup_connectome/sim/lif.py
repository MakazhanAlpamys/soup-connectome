from __future__ import annotations

from dataclasses import dataclass

from soup_connectome.config import SimulationConfig
from soup_connectome.errors import FixedPointOverflowError

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def _checked_value(value: int) -> int:
    if not INT32_MIN <= value <= INT32_MAX:
        raise FixedPointOverflowError(f"value is outside signed int32 range: {value}")
    return value


def checked_add(left: int, right: int) -> int:
    """Add fixed-point values without permitting signed int32 wraparound."""

    _checked_value(left)
    _checked_value(right)
    return _checked_value(left + right)


def arithmetic_shift_right(value: int, shift: int) -> int:
    """Apply the canonical signed two's-complement arithmetic right shift."""

    _checked_value(value)
    if not 0 <= shift <= 31:
        raise ValueError("shift must be between zero and 31")
    return value >> shift


def apply_leak(value: int, shifts: tuple[int, ...]) -> int:
    """Apply one or more canonical shift-only leak components."""

    result = _checked_value(value)
    for shift in shifts:
        leak_component = arithmetic_shift_right(result, shift)
        result = _checked_value(result - leak_component)
    return result


@dataclass(frozen=True, slots=True)
class NeuronState:
    potential: int
    refractory: int
    spike: bool = False


def advance_neuron(
    state: NeuronState,
    input_current: int,
    config: SimulationConfig,
) -> NeuronState:
    """Advance one neuron according to the normative timestep ordering."""

    _checked_value(state.potential)
    _checked_value(input_current)
    if state.refractory < 0:
        raise ValueError("refractory counter cannot be negative")
    if state.refractory:
        return NeuronState(config.reset, state.refractory - 1, False)

    potential = checked_add(state.potential, input_current)
    potential = apply_leak(potential, config.decay_shifts)
    if potential >= config.threshold:
        return NeuronState(config.reset, config.refractory_steps, True)
    return NeuronState(potential, 0, False)
