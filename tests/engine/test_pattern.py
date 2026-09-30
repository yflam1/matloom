"""Tests for matloom.engine.pattern: the Bricks lattice and the Weave pattern."""

from __future__ import annotations

import numpy as np
import pytest

from matloom.engine.pattern import Bricks, Weave
from tests.conftest import assert_grid_matches_call


def test_brick_center_is_one():
    # Center of the first cell (bw=1, bh=0.5) is well inside the brick body.
    b = Bricks()
    assert b(0.5, 0.25) == 1.0


def test_mortar_gap_is_zero():
    # A point inside the mortar gap (within mortar/2 of a cell edge).
    b = Bricks(brick_width=1.0, brick_height=0.5, mortar=0.1)
    assert b(0.01, 0.25) == 0.0  # near the x=0 cell boundary
    assert b(0.5, 0.49) == 0.0  # near the y=0.5 cell boundary


def test_running_bond_offsets_alternating_rows():
    # offset=0.5 shifts each successive row by half a brick. Row 0 has a brick
    # body around x=0.5; in row 1 (y in [0.5, 1)) the joints are shifted to
    # x=0.5, so the same x lands on a vertical mortar joint.
    b = Bricks(brick_width=1.0, brick_height=0.5, offset=0.5, mortar=0.1)
    assert b(0.5, 0.25) == 1.0  # row 0 brick
    assert b(0.5, 0.75) == 0.0  # row 1 joint sits at x=0.5 -> mortar


def test_stacked_bond_when_offset_zero():
    b = Bricks(offset=0.0, mortar=0.1)
    # Same x-position is a brick in every row when there is no offset.
    assert b(0.5, 0.25) == 1.0
    assert b(0.5, 0.75) == 1.0


def test_column_axis_offsets_columns():
    b = Bricks(
        brick_width=0.5,
        brick_height=1.0,
        offset=0.5,
        mortar=0.1,
        axis="column",
    )
    assert b(0.25, 0.5) == 1.0  # column 0 brick
    assert b(0.75, 0.5) == 0.0  # column 1 shifted by 0.5 -> joint at y=0.5


def test_feather_produces_smooth_edge():
    b = Bricks(brick_width=1.0, brick_height=0.5, mortar=0.1, feather=0.1)
    v = b(0.05, 0.25)  # exactly on the brick-body boundary (mortar/2 = 0.05)
    assert v == pytest.approx(0.5)
    assert 0.0 < b(0.07, 0.25) < 1.0  # within the feather band


def test_feather_zero_is_binary():
    b = Bricks(mortar=0.1)
    for x in np.linspace(0.0, 2.0, 17):
        for y in np.linspace(0.0, 2.0, 13):
            assert b(float(x), float(y)) in (0.0, 1.0)


@pytest.mark.parametrize("axis", ["row", "column"])
@pytest.mark.parametrize("feather", [0.0, 0.08])
def test_grid_matches_call(axes, axis, feather):
    xs, ys = axes
    b = Bricks(
        brick_width=0.7,
        brick_height=0.3,
        offset=0.5,
        mortar=0.06,
        axis=axis,
        feather=feather,
    )
    assert_grid_matches_call(b, xs, ys)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        Bricks(brick_width=0.0)
    with pytest.raises(ValueError):
        Bricks(mortar=-0.1)
    with pytest.raises(ValueError):
        Bricks(axis="diagonal")  # type: ignore[arg-type]


# |-----------|
# |   Weave   |
# |-----------|


def test_weave_thread_center_is_one():
    # Cell (0,0) of a plain weave has the warp on top; the thread runs through
    # the cell centre, so a texel at the centre of cell 0 is on the thread.
    w = Weave(base_freq=1.0, warp_width=0.5)
    assert w(0.5, 0.5) == 1.0


def test_weave_gap_between_threads_is_zero():
    # Away from the central thread band (warp_width=0.5 → band is [0.25, 0.75]),
    # the texel falls in the inter-thread gap.
    w = Weave(base_freq=1.0, warp_width=0.5)
    assert w(0.05, 0.5) == 0.0


def test_weave_plain_is_checkerboard_over_under():
    # Plain weave (1/1, shift 1): warp on top when (col - row) is even. With
    # full-width threads the on-top thread covers the whole cell, so the field
    # is 1 on warp-on-top cells and 0 (weft gap, here mid-cell in y) elsewhere.
    w = Weave(over=1, under=1, shift=1, base_freq=1.0, warp_width=1.0)
    # cell (0,0): warp on top → covered.
    assert w(0.5, 0.5) == 1.0


def test_weave_invalid_params_rejected():
    with pytest.raises(ValueError):
        Weave(over=0)
    with pytest.raises(ValueError):
        Weave(warp_width=0.0)
    with pytest.raises(ValueError):
        Weave(warp_width=1.5)
    with pytest.raises(ValueError):
        Weave(base_freq=0.0)


def test_weave_feather_zero_is_binary():
    w = Weave(over=2, under=1, shift=1, base_freq=6.0, warp_width=0.6)
    for x in np.linspace(0.0, 2.0, 15):
        for y in np.linspace(0.0, 2.0, 11):
            assert w(float(x), float(y)) in (0.0, 1.0)


@pytest.mark.parametrize("feather", [0.0, 0.15])
@pytest.mark.parametrize("shift", [0, 1, 2])
def test_weave_grid_matches_call(axes, feather, shift):
    xs, ys = axes
    w = Weave(
        over=2,
        under=1,
        shift=shift,
        base_freq_x=5.0,
        base_freq_y=7.0,
        warp_width=0.6,
        feather=feather,
    )
    assert_grid_matches_call(w, xs, ys)
