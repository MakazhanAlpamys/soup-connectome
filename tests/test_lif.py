import pytest

from soup_connectome.errors import FixedPointOverflowError
from soup_connectome.sim.lif import (
    INT32_MAX,
    apply_leak,
    arithmetic_shift_right,
    checked_add,
)


def test_signed_arithmetic_shift_is_defined_for_negative_values() -> None:
    assert arithmetic_shift_right(-3, 1) == -2
    assert apply_leak(-3, (1,)) == -1


def test_checked_add_rejects_signed_int32_overflow() -> None:
    with pytest.raises(FixedPointOverflowError):
        checked_add(INT32_MAX, 1)
