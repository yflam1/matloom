"""Tests for matloom.utils.maths."""

from __future__ import annotations

import pytest

from matloom.utils.maths import (
    clamp,
    is_even,
    is_odd,
    minmax,
    next_multiple_of,
    rescale,
)


def test_is_even_odd():
    assert is_even(4)
    assert not is_even(3)
    assert is_odd(3)
    assert not is_odd(4)


def test_is_even_zero():
    assert is_even(0)
    assert not is_odd(0)


def test_rescale_basic():
    assert rescale(5, 0, 10, 0, 100) == pytest.approx(50)


def test_rescale_endpoints():
    assert rescale(0, 0, 10, 0, 1) == pytest.approx(0)
    assert rescale(10, 0, 10, 0, 1) == pytest.approx(1)


def test_rescale_out_of_range_asserts():
    with pytest.raises(AssertionError):
        rescale(20, 0, 10, 0, 100)


def test_rescale_inverted_target_asserts():
    with pytest.raises(AssertionError):
        rescale(5, 0, 10, 100, 0)


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10


def test_clamp_inverted_bounds_asserts():
    with pytest.raises(AssertionError):
        clamp(5, 10, 0)


def test_minmax():
    assert minmax(3, 1, 4, 1, 5, 9, 2, 6) == (1, 9)


def test_minmax_single():
    assert minmax(7) == (7, 7)


def test_next_multiple_of():
    assert next_multiple_of(7, 5) == 10
    assert next_multiple_of(10, 5) == 10
    assert next_multiple_of(11, 5) == 15


def test_next_multiple_of_rejects_non_positive():
    with pytest.raises(ValueError, match="greater than 0"):
        next_multiple_of(5, 0)


def test_validate_call_rejects_wrong_type():
    with pytest.raises(ValueError):
        is_even("not an int")  # type: ignore[arg-type]
