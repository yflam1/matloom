"""
pattern.py — parametric tiling patterns that evaluate to a binary mask field.

Houses :class:`Bricks`, a running-bond brick lattice, and :class:`Weave`, an
over/under woven-thread lattice (plain / twill / satin). Like the noise nodes
each is a self-contained :class:`Expression2D` (a deterministic function of
``x``/``y`` with a handful of parameters), so it composes with the rest of the
engine and is driven from the DSL exactly like ``fBm``/``Worley``.
"""

import math
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, validate_call

from matloom.engine.expr import Expression2D, coerce_args


class Bricks(Expression2D):
    @coerce_args
    @validate_call
    def __init__(
        self,
        brick_width: Annotated[float, Field(gt=0.0)] = 1.0,
        brick_height: Annotated[float, Field(gt=0.0)] = 0.5,
        offset: float = 0.5,
        mortar: Annotated[float, Field(ge=0.0)] = 0.05,
        axis: Literal["row", "column"] = "row",
        feather: Annotated[float, Field(ge=0.0)] = 0.0,
    ) -> None:
        """
        Initialize the Bricks pattern.

        Evaluates to 1 on a brick and 0 in the mortar gap between bricks, with a
        smooth transition when ``feather`` > 0. The plane is tiled with cells of
        ``brick_width`` x ``brick_height``; ``mortar`` is the gap width (in world
        units) carved out between adjacent bricks; ``offset`` shifts each
        successive row (``axis="row"``) or column (``axis="column"``) by that
        fraction of a cell, so ``offset=0.5`` gives a classic running bond and
        ``offset=0`` a stacked bond.

        Args:
            brick_width (float, optional): Cell pitch along the x-axis (> 0).
                Defaults to 1.0.
            brick_height (float, optional): Cell pitch along the y-axis (> 0).
                Defaults to 0.5.
            offset (float, optional): Per-row/column shift as a fraction of a
                cell. Defaults to 0.5.
            mortar (float, optional): Gap width between bricks, in world units
                (>= 0). Defaults to 0.05.
            axis (str, optional): Whether successive rows or columns are offset.
                Defaults to "row".
            feather (float, optional): Width of the soft edge between brick and
                mortar, in world units (0 = hard edge). Defaults to 0.0.
        """
        self._brick_width = brick_width
        self._brick_height = brick_height
        self._offset = offset
        self._mortar = mortar
        self._axis = axis
        self._feather = feather

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        bw, bh = self._brick_width, self._brick_height
        if self._axis == "row":
            row = math.floor(y / bh)
            lx = x - row * self._offset * bw
            lx = lx - bw * math.floor(lx / bw)
            ly = y - bh * math.floor(y / bh)
        else:
            col = math.floor(x / bw)
            lx = x - bw * math.floor(x / bw)
            ly = y - col * self._offset * bh
            ly = ly - bh * math.floor(ly / bh)
        half_m = self._mortar / 2.0
        dx_in = min(lx - half_m, bw - half_m - lx)
        dy_in = min(ly - half_m, bh - half_m - ly)
        d = min(dx_in, dy_in)
        return self._coverage(d)

    def eval_grid(self, xs: np.ndarray, ys: np.ndarray, coords=None) -> np.ndarray:
        bw, bh = self._brick_width, self._brick_height
        if coords is not None:
            gx, gy = coords
            shape = gx.shape
        else:
            gx = xs[np.newaxis, :]
            gy = ys[:, np.newaxis]
            shape = (ys.size, xs.size)
        if self._axis == "row":
            row = np.floor(gy / bh)
            lx = gx - row * self._offset * bw
            lx = lx - bw * np.floor(lx / bw)
            ly = gy - bh * np.floor(gy / bh)
        else:
            col = np.floor(gx / bw)
            lx = gx - bw * np.floor(gx / bw)
            ly = gy - col * self._offset * bh
            ly = ly - bh * np.floor(ly / bh)
        half_m = self._mortar / 2.0
        dx_in = np.minimum(lx - half_m, bw - half_m - lx)
        dy_in = np.minimum(ly - half_m, bh - half_m - ly)
        d = np.broadcast_to(np.minimum(dx_in, dy_in), shape)
        return self._coverage_grid(d)

    def _coverage(self, d: float) -> float:
        f = self._feather
        if f == 0.0:
            return 1.0 if d >= 0.0 else 0.0
        t = (d + f / 2.0) / f
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return t * t * (3.0 - 2.0 * t)

    def _coverage_grid(self, d: np.ndarray) -> np.ndarray:
        f = self._feather
        if f == 0.0:
            return (d >= 0.0).astype(np.float64)
        t = np.clip((d + f / 2.0) / f, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)


class Weave(Expression2D):
    @coerce_args
    @validate_call
    def __init__(
        self,
        over: Annotated[int, Field(ge=1)] = 1,
        under: Annotated[int, Field(ge=1)] = 1,
        shift: int = 1,
        base_freq: Annotated[float, Field(gt=0.0)] | None = 8.0,
        base_freq_x: Annotated[float, Field(gt=0.0)] | None = None,
        base_freq_y: Annotated[float, Field(gt=0.0)] | None = None,
        warp_width: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5,
        feather: Annotated[float, Field(ge=0.0)] = 0.0,
    ) -> None:
        """
        Initialize the Weave pattern: an over/under interlacing of warp (lengthwise)
        and weft (crosswise) threads, the canonical structure of woven cloth.

        Evaluates to 1 on the thread that is *on top* at a texel and 0 in the gaps
        between threads, with a smooth transition when ``feather`` > 0. The plane
        is divided into a grid of ``base_freq`` cells per unit; in each cell either
        the warp or the weft is on top, decided by the standard weave draft
        ``((col - row * shift) mod (over + under)) < over``. ``over``/``under`` set
        the float length of the interlacing (``1/1`` = plain, ``2/1`` = twill,
        ``4/1`` with ``shift=2`` = satin); ``shift`` is the per-row diagonal step.
        ``warp_width`` is the fraction of each cell the thread occupies (the rest
        is the inter-thread gap).

        Args:
            over (int, optional): Cells a thread floats *over* before going under
                (>= 1). Defaults to 1.
            under (int, optional): Cells a thread floats *under* (>= 1). Defaults
                to 1.
            shift (int, optional): Per-row diagonal step of the draft. Defaults
                to 1.
            base_freq (float, optional): Cells (thread crossings) per unit on both
                axes (> 0). Defaults to 8.0.
            base_freq_x (float, optional): Per-axis override for x. If None,
                ``base_freq`` is used. Defaults to None.
            base_freq_y (float, optional): Per-axis override for y. If None,
                ``base_freq`` is used. Defaults to None.
            warp_width (float, optional): Thread width as a fraction of the cell,
                in (0, 1]. Defaults to 0.5.
            feather (float, optional): Width of the soft thread edge, in
                cell-fraction units (0 = hard edge). Defaults to 0.0.
        """
        if base_freq is None:
            if base_freq_x is None or base_freq_y is None:
                raise ValueError(
                    "If `base_freq` is None, both `base_freq_x` and "
                    "`base_freq_y` must be provided"
                )
            self._base_freq_x = base_freq_x
            self._base_freq_y = base_freq_y
        else:
            self._base_freq_x = base_freq if base_freq_x is None else base_freq_x
            self._base_freq_y = base_freq if base_freq_y is None else base_freq_y
        self._over = over
        self._under = under
        self._shift = shift
        self._warp_width = warp_width
        self._feather = feather

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        fx, fy = self._base_freq_x, self._base_freq_y
        period = self._over + self._under
        u, v = x * fx, y * fy
        col = math.floor(u)
        row = math.floor(v)
        m = (col - row * self._shift) % period
        warp_on_top = m < self._over
        half = self._warp_width / 2.0
        if warp_on_top:
            d = half - abs((u - col) - 0.5)
        else:
            d = half - abs((v - row) - 0.5)
        return self._coverage(d)

    def eval_grid(self, xs: np.ndarray, ys: np.ndarray, coords=None) -> np.ndarray:
        fx, fy = self._base_freq_x, self._base_freq_y
        if coords is not None:
            gx, gy = coords
        else:
            gx = xs[np.newaxis, :]
            gy = ys[:, np.newaxis]
        period = self._over + self._under
        u = gx * fx
        v = gy * fy
        col = np.floor(u)
        row = np.floor(v)
        m = np.mod(col - row * self._shift, period)
        warp_on_top = m < self._over
        half = self._warp_width / 2.0
        dx = half - np.abs((u - col) - 0.5)
        dy = half - np.abs((v - row) - 0.5)
        d = np.broadcast_to(np.where(warp_on_top, dx, dy), warp_on_top.shape)
        return self._coverage_grid(d)

    def _coverage(self, d: float) -> float:
        f = self._feather
        if f == 0.0:
            return 1.0 if d >= 0.0 else 0.0
        t = (d + f / 2.0) / f
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return t * t * (3.0 - 2.0 * t)

    def _coverage_grid(self, d: np.ndarray) -> np.ndarray:
        f = self._feather
        if f == 0.0:
            return (d >= 0.0).astype(np.float64)
        t = np.clip((d + f / 2.0) / f, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
