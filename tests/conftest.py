"""Shared pytest fixtures and helpers for the matloom core test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def axes():
    """A small, fixed pair of sample axes for grid-evaluation tests."""
    xs = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    return xs, ys


def assert_grid_matches_call(expr, xs, ys, *, rtol=1e-12, atol=1e-12):
    """
    Every Expression2D must agree between its scalar ``__call__`` path and its
    vectorized ``eval_grid`` path. This is the core invariant the engine relies
    on, so it is checked for nearly every node type.
    """
    grid = expr.eval_grid(xs, ys)
    assert grid.shape == (ys.size, xs.size)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            assert grid[i, j] == pytest.approx(expr(x, y), rel=rtol, abs=atol)
