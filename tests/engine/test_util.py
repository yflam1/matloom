"""Tests for matloom.engine.util: color conversions and Threshold."""

from __future__ import annotations

import numpy as np
import pytest

from matloom.engine.expr import X
from matloom.engine.util import (
    Linear2sRGB,
    Threshold,
    linear_to_srgb,
    sRGB2Linear,
    srgb_to_linear,
)
from tests.conftest import assert_grid_matches_call

# |---------------------------|
# |   Scalar sRGB conversion  |
# |---------------------------|


def test_srgb_linear_endpoints():
    assert srgb_to_linear(0.0) == pytest.approx(0.0)
    assert srgb_to_linear(1.0) == pytest.approx(1.0)
    assert linear_to_srgb(0.0) == pytest.approx(0.0)
    assert linear_to_srgb(1.0) == pytest.approx(1.0)


def test_srgb_to_linear_uses_linear_segment_below_cutoff():
    # Below 0.04045 the transfer function is a plain divide by 12.92.
    assert srgb_to_linear(0.04) == pytest.approx(0.04 / 12.92)


def test_linear_to_srgb_uses_linear_segment_below_cutoff():
    assert linear_to_srgb(0.003) == pytest.approx(0.003 * 12.92)


def test_srgb_roundtrip_scalar():
    # The standard sRGB approximation has a tiny discontinuity at the 0.04045
    # cutoff (the linear and power segments don't meet exactly), so the
    # round-trip is only accurate to ~1e-4 right at the seam; elsewhere it is
    # essentially exact.
    for v in (0.0, 0.01, 0.2, 0.5, 0.9, 1.0):
        assert linear_to_srgb(srgb_to_linear(v)) == pytest.approx(v, abs=1e-12)
    assert linear_to_srgb(srgb_to_linear(0.04045)) == pytest.approx(0.04045, abs=1e-4)


def test_srgb_conversion_vectorized_matches_scalar():
    arr = np.linspace(0.0, 1.0, 17)
    lin = srgb_to_linear(arr)
    assert isinstance(lin, np.ndarray)
    for i, v in enumerate(arr):
        assert lin[i] == pytest.approx(srgb_to_linear(float(v)))
    back = linear_to_srgb(lin)
    np.testing.assert_allclose(back, arr, atol=1e-12)


def test_srgb_expression_nodes(axes):
    xs, ys = axes
    assert_grid_matches_call(sRGB2Linear(X()), xs, ys)
    assert_grid_matches_call(Linear2sRGB(X()), xs, ys)


# |---------------|
# |   Threshold   |
# |---------------|


def test_threshold_hard_above():
    t = Threshold(X(), above_at=0.5, above_to=1.0)
    assert t(0.4) == pytest.approx(0.4)
    assert t(0.5) == pytest.approx(1.0)
    assert t(0.9) == pytest.approx(1.0)


def test_threshold_hard_below():
    t = Threshold(X(), below_at=0.5, below_to=0.0)
    assert t(0.6) == pytest.approx(0.6)
    assert t(0.5) == pytest.approx(0.0)
    assert t(0.1) == pytest.approx(0.0)


def test_threshold_below_to_defaults_to_below_at():
    t = Threshold(X(), below_at=0.3)
    assert t(0.1) == pytest.approx(0.3)


def test_threshold_above_to_defaults_to_above_at():
    t = Threshold(X(), above_at=0.7)
    assert t(0.9) == pytest.approx(0.7)


def test_threshold_passthrough_without_cutoffs():
    t = Threshold(X())
    for v in (0.0, 0.25, 0.5, 1.0):
        assert t(v) == pytest.approx(v)


def test_threshold_clamp_both_ends():
    # Layer channels use this clamp-to-[0,1] pattern.
    t = Threshold(X(), below_at=0.0, above_at=1.0)
    assert t(-0.5) == pytest.approx(0.0)
    assert t(0.5) == pytest.approx(0.5)
    assert t(1.5) == pytest.approx(1.0)


def test_threshold_smoothstep_midpoint():
    # With a transition the value eases rather than snapping. At the exact
    # midpoint of the smoothstep window, smoothstep(0.5) == 0.5.
    t = Threshold(X(), above_at=1.0, above_to=2.0, transition_width=1.0)
    # window is [above_at - width, above_at] = [0, 1]; at v=0.5, t-param=0.5
    # out = (1 - s) * v + s * above_to, s = smoothstep(0.5) = 0.5
    assert t(0.5) == pytest.approx(0.5 * 0.5 + 0.5 * 2.0)


def test_threshold_smoothstep_endpoints_are_hard():
    t = Threshold(X(), above_at=1.0, above_to=2.0, transition_width=0.5)
    assert t(0.4) == pytest.approx(0.4)  # before window: unchanged
    assert t(1.0) == pytest.approx(2.0)  # at/after window end: fully replaced


def test_threshold_rejects_negative_transition_width():
    with pytest.raises(ValueError):
        Threshold(X(), above_at=1.0, transition_width=-0.1)


def test_threshold_grid_matches_scalar_hard(axes):
    xs, ys = axes
    assert_grid_matches_call(Threshold(X(), below_at=0.2, above_at=0.8), xs, ys)


def test_threshold_grid_matches_scalar_smooth(axes):
    xs, ys = axes
    assert_grid_matches_call(
        Threshold(X(), below_at=0.2, below_to=0.0, transition_width=0.3),
        xs,
        ys,
    )


def test_threshold_grid_matches_scalar_smooth_above(axes):
    xs, ys = axes
    assert_grid_matches_call(
        Threshold(X(), above_at=0.7, above_to=1.0, transition_width=0.3),
        xs,
        ys,
    )
