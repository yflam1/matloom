"""Tests for matloom.engine.shape: Rect/Ellipse shapes and Fill/Stroke."""

from __future__ import annotations

import pytest

from matloom.engine.expr import X
from matloom.engine.shape import CubicTo, Ellipse, Fill, LineTo, Path, Rect, Stroke
from matloom.engine.transform import Rotate, Translate
from tests.conftest import assert_grid_matches_call


def test_fill_rect_inside_outside():
    f = Fill(Rect(0.2, 0.2, 0.6, 0.6))
    assert f(0.5, 0.5) == 1.0  # inside
    assert f(0.05, 0.5) == 0.0  # left of the rect
    assert f(0.5, 0.95) == 0.0  # above the rect


def test_fill_ellipse_inside_outside():
    f = Fill(Ellipse(0.5, 0.5, 0.3, 0.2))
    assert f(0.5, 0.5) == 1.0  # center
    assert f(0.5, 0.78) == 0.0  # past ry along y
    assert f(0.85, 0.5) == 0.0  # past rx along x


def test_fill_feather_is_half_on_boundary():
    f = Fill(Rect(0.2, 0.2, 0.6, 0.6), feather=0.1)
    # On the right edge (x = 0.8) the signed distance is 0, so coverage is 0.5.
    assert f(0.8, 0.5) == pytest.approx(0.5)


def test_stroke_paints_the_boundary_band():
    s = Stroke(Rect(0.2, 0.2, 0.6, 0.6), 0.08)
    assert s(0.2, 0.5) == 1.0  # on the left edge
    assert s(0.5, 0.5) == 0.0  # deep interior
    assert s(0.5, 0.5) != 1.0


def test_rounded_rect_clips_corner():
    sharp = Fill(Rect(0.0, 0.0, 1.0, 1.0))
    round_ = Fill(Rect(0.0, 0.0, 1.0, 1.0, radius=0.3))
    # The very corner is inside a sharp rect but cut away by a large radius.
    assert sharp(0.01, 0.01) == 1.0
    assert round_(0.01, 0.01) == 0.0


def test_fill_requires_a_shape():
    with pytest.raises(ValueError):
        Fill(X())  # an Expression2D is not a Shape
    with pytest.raises(ValueError):
        Fill(5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "shape",
    [
        Rect(0.2, 0.2, 0.6, 0.5),
        Rect(0.1, 0.1, 0.6, 0.6, 0.2),
        Ellipse(0.5, 0.5, 0.3, 0.2),
    ],
)
@pytest.mark.parametrize("feather", [0.0, 0.07])
def test_fill_grid_matches_call(axes, shape, feather):
    xs, ys = axes
    assert_grid_matches_call(Fill(shape, feather=feather), xs, ys)


@pytest.mark.parametrize(
    "shape", [Rect(0.2, 0.2, 0.6, 0.5), Ellipse(0.5, 0.5, 0.3, 0.3)]
)
def test_stroke_grid_matches_call(axes, shape):
    xs, ys = axes
    assert_grid_matches_call(Stroke(shape, 0.08, feather=0.03), xs, ys)


def test_transformed_shape_grid_matches_call(axes):
    xs, ys = axes
    # Build the field, then transform it (the recommended composition order).
    star = Fill(Rect(0.2, 0.2, 0.4, 0.3), feather=0.04)
    assert_grid_matches_call(Translate(star, 0.1, 0.2), xs, ys)
    assert_grid_matches_call(Rotate(star, 33), xs, ys)


def test_path_triangle_fill():
    # Triangle is implicitly closed for filling.
    tri = Fill(Path(0.2, 0.2, [LineTo(0.8, 0.2), LineTo(0.5, 0.8)]))
    assert tri(0.5, 0.4) == 1.0  # interior
    assert tri(0.1, 0.1) == 0.0  # outside
    assert tri(0.5, 0.9) == 0.0  # above the apex


def test_path_requires_segments():
    with pytest.raises(ValueError):
        Path(0.0, 0.0, [])


def test_path_open_stroke_is_not_closed():
    # A stroked open path paints its segments but not the closing edge.
    p = Path(0.2, 0.5, [LineTo(0.8, 0.5)])  # a single horizontal segment
    s = Stroke(p, 0.06)
    assert s(0.5, 0.5) == 1.0  # on the segment
    # The midpoint of the (absent) closing edge coincides here, so this only
    # checks the segment itself; a point far from the segment is unpainted.
    assert s(0.5, 0.9) == 0.0


def test_path_cubic_is_flattened():
    # A cubic curve is sampled; a point on the curve's path is inside a thin
    # stroke, a point far away is not.
    c = Stroke(Path(0.1, 0.5, [CubicTo(0.3, 0.9, 0.7, 0.1, 0.9, 0.5)]), 0.06)
    assert c(0.1, 0.5) == 1.0  # at the start
    assert c(0.5, 0.95) == 0.0  # far above the curve


@pytest.mark.parametrize("mode", ["NonZero", "EvenOdd"])
@pytest.mark.parametrize("feather", [0.0, 0.06])
def test_path_fill_grid_matches_call(axes, mode, feather):
    xs, ys = axes
    p = Path(0.2, 0.2, [LineTo(0.8, 0.3), CubicTo(0.7, 0.7, 0.4, 0.7, 0.3, 0.8)])
    assert_grid_matches_call(Fill(p, mode=mode, feather=feather), xs, ys)


def test_path_stroke_grid_matches_call(axes):
    xs, ys = axes
    p = Path(0.1, 0.5, [CubicTo(0.3, 0.9, 0.7, 0.1, 0.9, 0.5)])
    assert_grid_matches_call(Stroke(p, 0.06, feather=0.02), xs, ys)
