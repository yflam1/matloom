"""
transform.py — coordinate transforms that wrap any :class:`Expression2D`.

A transform remaps the sample point before delegating to its child, so it works
on *any* expression (shapes, noise, gradients, …), not just masks.
:class:`Translate` and :class:`Scale` are axis-separable — they rewrite the 1-D
sample axes and stay fully vectorized. :class:`Rotate` couples x and y, so it
samples its child on explicit per-cell coordinate grids (the ``coords`` argument
of ``eval_grid``); noise nodes evaluate there through flat compiled coords
kernels that replicate their per-cell math, so they rotate with everything
else (slower than their separable fused path).

All transforms operate about the origin, matching the DSL's stated convention.
"""

import math

import numpy as np
from pydantic import validate_call

from matloom.engine.expr import Coords, Expression2D, coerce_args
from matloom.utils.dtypes import NonZeroFloat


class Translate(Expression2D):
    @coerce_args
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, expr: Expression2D, dx: float = 0.0, dy: float = 0.0) -> None:
        """
        Translate an expression by ``(dx, dy)`` in world space.

        Samples ``expr`` at ``(x - dx, y - dy)``, so a feature at the origin of
        ``expr`` appears at ``(dx, dy)``.

        Args:
            expr (Expression2D): The expression to translate.
            dx (float, optional): Shift along the x-axis. Defaults to 0.0.
            dy (float, optional): Shift along the y-axis. Defaults to 0.0.
        """
        self._expr = expr
        self._dx = dx
        self._dy = dy

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return self._expr(x - self._dx, y - self._dy)

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        xs2 = (xs - self._dx).astype(xs.dtype, copy=False)
        ys2 = (ys - self._dy).astype(ys.dtype, copy=False)
        coords2 = (
            (coords[0] - self._dx, coords[1] - self._dy) if coords is not None else None
        )
        return self._expr.eval_grid(xs2, ys2, coords2)


class Scale(Expression2D):
    @coerce_args
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        expr: Expression2D,
        sx: NonZeroFloat = 1.0,
        sy: NonZeroFloat = 1.0,
    ) -> None:
        """
        Scale an expression about the origin by ``(sx, sy)``.

        Samples ``expr`` at ``(x / sx, y / sy)``, so its features grow by a
        factor of ``sx``/``sy``. Negative factors mirror across the
        corresponding axis. Both factors must be non-zero.

        Args:
            expr (Expression2D): The expression to scale.
            sx (NonZeroFloat, optional): Scale factor along the x-axis.
                Defaults to 1.0.
            sy (NonZeroFloat, optional): Scale factor along the y-axis.
                Defaults to 1.0.
        """
        self._expr = expr
        self._sx = sx
        self._sy = sy

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return self._expr(x / self._sx, y / self._sy)

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        xs2 = (xs / self._sx).astype(xs.dtype, copy=False)
        ys2 = (ys / self._sy).astype(ys.dtype, copy=False)
        coords2 = (
            (coords[0] / self._sx, coords[1] / self._sy) if coords is not None else None
        )
        return self._expr.eval_grid(xs2, ys2, coords2)


class Rotate(Expression2D):
    @coerce_args
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(self, expr: Expression2D, degrees: float = 0.0) -> None:
        """
        Rotate an expression counter-clockwise about the origin.

        Samples ``expr`` at the inverse-rotated point, so its content turns CCW
        by ``degrees``. Because rotation couples the axes, the sample grid is no
        longer separable; noise nodes evaluate it per cell through a flat
        compiled coords kernel (slower than their separable fused path).

        Args:
            expr (Expression2D): The expression to rotate.
            degrees (float, optional): Counter-clockwise rotation in degrees.
                Defaults to 0.0.
        """
        self._expr = expr
        self._degrees = degrees
        rad = degrees * math.pi / 180.0
        self._cos = math.cos(rad)
        self._sin = math.sin(rad)

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        xr = self._cos * x + self._sin * y
        yr = -self._sin * x + self._cos * y
        return self._expr(xr, yr)

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        if coords is not None:
            bx, by = coords
        else:
            bx = np.broadcast_to(xs[np.newaxis, :], (ys.size, xs.size))
            by = np.broadcast_to(ys[:, np.newaxis], (ys.size, xs.size))
        xx = self._cos * bx + self._sin * by
        yy = -self._sin * bx + self._cos * by
        return self._expr.eval_grid(xs, ys, (xx, yy))
