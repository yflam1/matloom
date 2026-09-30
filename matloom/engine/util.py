from typing import Annotated, overload

import numpy as np
from numba import njit, prange
from pydantic import Field, validate_call

from matloom.engine.expr import Expression2D, coerce_args


@overload
def srgb_to_linear(x: float) -> float: ...
@overload
def srgb_to_linear(x: np.ndarray) -> np.ndarray: ...
def srgb_to_linear(x: float | np.ndarray) -> float | np.ndarray:
    if isinstance(x, np.ndarray):
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


@overload
def linear_to_srgb(x: float) -> float: ...
@overload
def linear_to_srgb(x: np.ndarray) -> np.ndarray: ...
def linear_to_srgb(x: float | np.ndarray) -> float | np.ndarray:
    if isinstance(x, np.ndarray):
        return np.where(x <= 0.0031308, x * 12.92, x ** (1 / 2.4) * 1.055 - 0.055)
    return x * 12.92 if x <= 0.0031308 else x ** (1 / 2.4) * 1.055 - 0.055


class sRGB2Linear(Expression2D):
    @coerce_args
    def __init__(self, expr: Expression2D) -> None:
        self._expr = expr

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return srgb_to_linear(self._expr(x, y))

    def eval_grid(self, xs: np.ndarray, ys: np.ndarray, coords=None) -> np.ndarray:
        return srgb_to_linear(self._expr.eval_grid(xs, ys, coords))


class Linear2sRGB(Expression2D):
    @coerce_args
    def __init__(self, expr: Expression2D) -> None:
        self._expr = expr

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return linear_to_srgb(self._expr(x, y))

    def eval_grid(self, xs: np.ndarray, ys: np.ndarray, coords=None) -> np.ndarray:
        return linear_to_srgb(self._expr.eval_grid(xs, ys, coords))


@njit(cache=True, parallel=True)
def _threshold_kernel(
    vals: np.ndarray,
    out: np.ndarray,
    has_above: bool,
    above_at: float,
    above_to: float,
    has_below: bool,
    below_at: float,
    below_to: float,
    transition_width: float,
) -> None:
    for i in prange(vals.size):
        v = vals.flat[i]
        r = v
        if has_above:
            if transition_width == 0.0:
                if v >= above_at:
                    r = above_to
            else:
                t = (v - (above_at - transition_width)) / transition_width
                if t >= 1.0:
                    r = above_to
                elif t > 0.0:
                    t = t * t * (3.0 - 2.0 * t)
                    r = (1.0 - t) * v + t * above_to
        if has_below:
            if transition_width == 0.0:
                if v <= below_at:
                    r = below_to
            else:
                t = (v - below_at) / transition_width
                if t <= 0.0:
                    r = below_to
                elif t < 1.0:
                    t = t * t * (3.0 - 2.0 * t)
                    r = (1.0 - t) * below_to + t * v
        out.flat[i] = r


class Threshold(Expression2D):
    @coerce_args
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        expr: Expression2D,
        *,
        below_at: float | None = None,
        below_to: float | None = None,
        above_at: float | None = None,
        above_to: float | None = None,
        transition_width: Annotated[float, Field(ge=0.0)] = 0.0,
    ) -> None:
        """
        Initialize the Threshold expression.

        Args:
            expr (Expression2D): The input expression to apply thresholding to.
            below_at (float | None, optional): The value at which to apply
                the below threshold. If None, no below threshold is applied.
                Defaults to None.
            below_to (float | None, optional): The value to replace with
                when the input is below the below threshold. If None,
                `below_to` is set to `below_at`. Defaults to None.
            above_at (float | None, optional): The value at which to apply
                the above threshold. If None, no above threshold is applied.
                Defaults to None.
            above_to (float | None, optional): The value to replace with
                when the input is above the above threshold. If None,
                `above_to` is set to `above_at`. Defaults to None.
            transition_width (float, optional): The width of the smooth
                transition (smoothstep) between the original value and the
                thresholded value. Defaults to 0.0 (hard threshold).
        """
        if below_at is None:
            below_to = None
        elif below_to is None:
            below_to = below_at
        if above_at is None:
            above_to = None
        elif above_to is None:
            above_to = above_at
        self._expr = expr
        self._below_at = below_at
        self._below_to = below_to
        self._above_at = above_at
        self._above_to = above_to
        self._transition_width = transition_width

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        val = self._expr(x, y)
        out = val
        if self._above_at is not None and self._above_to is not None:
            if self._transition_width == 0.0:
                if val >= self._above_at:
                    out = self._above_to
            else:
                edge0 = self._above_at - self._transition_width
                edge1 = self._above_at
                t = self._smoothstep(edge0, edge1, val)
                out = (1.0 - t) * val + t * self._above_to
        if self._below_at is not None and self._below_to is not None:
            if self._transition_width == 0.0:
                if val <= self._below_at:
                    out = self._below_to
            else:
                edge0 = self._below_at
                edge1 = self._below_at + self._transition_width
                t = self._smoothstep(edge0, edge1, val)
                out = (1.0 - t) * self._below_to + t * val
        return out

    def eval_grid(self, xs: np.ndarray, ys: np.ndarray, coords=None) -> np.ndarray:
        vals = self._expr.eval_grid(xs, ys, coords)
        out = np.empty_like(vals)
        _threshold_kernel(
            vals,
            out,
            self._above_at is not None,
            self._above_at if self._above_at is not None else 0.0,
            self._above_to if self._above_to is not None else 0.0,
            self._below_at is not None,
            self._below_at if self._below_at is not None else 0.0,
            self._below_to if self._below_to is not None else 0.0,
            self._transition_width,
        )
        return out

    def _smoothstep(self, edge0: float, edge1: float, x: float) -> float:
        """
        Standard GLSL smoothstep interpolation.
        """
        # Clamp x to the [0, 1] range
        t = max(0.0, min((x - edge0) / (edge1 - edge0), 1.0))
        # Smooth S-curve evaluation
        return t * t * (3.0 - 2.0 * t)
