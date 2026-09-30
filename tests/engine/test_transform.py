"""Tests for matloom.engine.transform: Translate and Scale."""

from __future__ import annotations

import pytest

from matloom.engine.expr import Min, Sin, X, Y
from matloom.engine.noise import Worley, fBm
from matloom.engine.pattern import Bricks
from matloom.engine.transform import Rotate, Scale, Translate
from matloom.engine.util import Threshold
from tests.conftest import assert_grid_matches_call


def test_translate_shifts_x():
    # Translate(X(), 0.3, 0)(x, y) samples X() at x - 0.3.
    assert Translate(X(), 0.3, 0.0)(0.5, 0.0) == pytest.approx(0.2)


def test_translate_shifts_y():
    assert Translate(Y(), 0.0, 0.25)(0.0, 1.0) == pytest.approx(0.75)


def test_translate_wraps_bare_number():
    # A bare number coerces to Constant; translating it is still that constant.
    assert Translate(5, 1.0, 2.0)(0.0, 0.0) == 5.0


def test_scale_about_origin():
    # Scale(X(), 2, 1)(x, y) samples X() at x / 2.
    assert Scale(X(), 2.0, 1.0)(1.0, 0.0) == pytest.approx(0.5)


def test_scale_negative_mirrors():
    assert Scale(X(), -1.0, 1.0)(0.3, 0.0) == pytest.approx(-0.3)


def test_scale_zero_factor_rejected():
    with pytest.raises(ValueError):
        Scale(X(), 0.0, 1.0)
    with pytest.raises(ValueError):
        Scale(X(), 1.0, 0.0)


def test_grid_matches_call_translate(axes):
    xs, ys = axes
    assert_grid_matches_call(Translate(X() + Y(), 0.3, -0.2), xs, ys)


def test_grid_matches_call_scale(axes):
    xs, ys = axes
    assert_grid_matches_call(Scale(Sin(X() * Y()), 2.0, 0.5), xs, ys)


def test_nested_transforms_grid_matches_call(axes):
    xs, ys = axes
    e = Translate(Scale(X() + Y(), 2.0, 2.0), 0.1, 0.1)
    assert_grid_matches_call(e, xs, ys)


def test_rotate_90_turns_x_gradient_into_y():
    # Rotating the X-gradient 90 CCW makes it read the Y coordinate.
    e = Rotate(X(), 90)
    assert e(1.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert e(0.0, 1.0) == pytest.approx(1.0)


def test_rotate_zero_is_identity():
    assert Rotate(X() + Y(), 0)(0.3, 0.4) == pytest.approx(0.7)


def test_rotate_grid_matches_call_over_node_types(axes):
    # The grid path must equal the scalar path for Rotate wrapping any node —
    # this catches a forwarder that fails to thread the per-cell coords.
    xs, ys = axes
    nodes = [
        X() + Y(),
        Sin(X()),
        X() * Y(),
        Threshold(X(), below_at=0.3, above_at=0.7),
        Bricks(mortar=0.1),
        Min(X(), Y()),
        Translate(X() + Y(), 0.2, -0.1),
        Scale(Sin(X()), 2.0, 0.5),
    ]
    for node in nodes:
        assert_grid_matches_call(Rotate(node, 37), xs, ys)


def test_rotate_noise_grid_matches_call(axes):
    # Noise samples the warped coords per cell, so it rotates and stays
    # consistent between eval_grid and __call__.
    xs, ys = axes
    assert_grid_matches_call(Rotate(fBm(seed=5), 30), xs, ys)
    assert_grid_matches_call(Rotate(Worley(seed=3), 25), xs, ys)


def test_nested_rotation_composes(axes):
    xs, ys = axes
    assert_grid_matches_call(Rotate(Rotate(X() + Y(), 20), 25), xs, ys)
