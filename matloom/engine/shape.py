"""
shape.py — SVG-like shapes and the Fill/Stroke operators that turn them into
binary mask fields.

A :class:`Shape` (``Rect``, ``Ellipse``, ``Path``) is *not* an
:class:`Expression2D` on its own — it is pure geometry exposing the distance to
its boundary. :class:`Fill` and :class:`Stroke` are the :class:`Expression2D`
nodes: ``Fill`` paints the shape's interior, ``Stroke`` paints a band along its
boundary. Both evaluate to 1 inside the painted region and 0 outside, with a
smooth transition when ``feather`` > 0.

Distances use only IEEE operations (``+ - * / sqrt abs min max``) and curves are
flattened with a fixed uniform step count, so the same formulas reproduce across
the Python and TypeScript engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, validate_call

from matloom.engine.expr import Coords, Expression2D, coerce_args

FillMode = Literal["NonZero", "EvenOdd"]

# Subdivisions per cubic Bézier when flattening to a polyline. Fixed (not
# adaptive) so both engines produce bit-identical vertices. Baked into the
# parity fixtures: changing it forces a fixture regen.
CUBIC_STEPS = 16


def _scalar(v) -> float:
    """Reduce an expression argument to a constant (shape coords are static)."""
    return float(v() if isinstance(v, Expression2D) else v)


def _coverage(s: np.ndarray, feather: float) -> np.ndarray:
    """
    Map a signed "insideness" ``s`` (> 0 inside the painted region) to coverage
    in [0, 1]. ``feather`` is the soft-edge width centered on the boundary
    (``s = 0``); 0 gives a hard edge. Works on scalars and arrays alike.
    """
    if feather == 0.0:
        return np.where(s >= 0.0, 1.0, 0.0)
    t = np.clip((s + feather / 2.0) / feather, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Shape(ABC):
    @abstractmethod
    def fill_distance(self, gx, gy, mode: str = "NonZero"):
        """
        Signed distance for filling: < 0 inside, > 0 outside. ``gx``/``gy`` may
        be scalars or numpy arrays. ``mode`` selects the fill rule for
        self-intersecting paths; convex primitives ignore it.
        """

    def stroke_distance(self, gx, gy):
        """Unsigned distance to the stroked boundary. Defaults to |fill|."""
        return np.abs(self.fill_distance(gx, gy))


class Rect(Shape):
    @coerce_args
    @validate_call
    def __init__(
        self,
        x: float,
        y: float,
        width: Annotated[float, Field(gt=0.0)],
        height: Annotated[float, Field(gt=0.0)],
        radius: Annotated[float, Field(ge=0.0)] = 0.0,
    ) -> None:
        """
        Axis-aligned rectangle with optional rounded corners.

        Args:
            x (float): X of the bottom-left corner.
            y (float): Y of the bottom-left corner.
            width (float): Width (> 0).
            height (float): Height (> 0).
            radius (float, optional): Corner rounding radius, clamped to
                ``min(width, height) / 2``. Defaults to 0.0.
        """
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._radius = min(radius, min(width, height) / 2.0)
        self._cx = x + width / 2.0
        self._cy = y + height / 2.0
        self._hx = width / 2.0
        self._hy = height / 2.0

    def fill_distance(self, gx, gy, mode: str = "NonZero"):
        r = self._radius
        qx = np.abs(gx - self._cx) - (self._hx - r)
        qy = np.abs(gy - self._cy) - (self._hy - r)
        ax = np.maximum(qx, 0.0)
        ay = np.maximum(qy, 0.0)
        outside = np.sqrt(ax * ax + ay * ay)
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return outside + inside - r


class Ellipse(Shape):
    @coerce_args
    @validate_call
    def __init__(
        self,
        cx: float,
        cy: float,
        rx: Annotated[float, Field(gt=0.0)],
        ry: Annotated[float, Field(gt=0.0)],
    ) -> None:
        """
        Axis-aligned ellipse.

        Args:
            cx (float): Center X.
            cy (float): Center Y.
            rx (float): Radius along X (> 0).
            ry (float): Radius along Y (> 0).
        """
        self._cx = cx
        self._cy = cy
        self._rx = rx
        self._ry = ry

    def fill_distance(self, gx, gy, mode: str = "NonZero"):
        # Gradient-corrected approximate ellipse SDF (exact for a circle).
        px = gx - self._cx
        py = gy - self._cy
        rx, ry = self._rx, self._ry
        ex = px / rx
        ey = py / ry
        k1 = np.sqrt(ex * ex + ey * ey)
        fx = px / (rx * rx)
        fy = py / (ry * ry)
        k2 = np.sqrt(fx * fx + fy * fy)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = k1 * (k1 - 1.0) / k2
        # At the exact center k2 == 0; the distance there is -min(rx, ry).
        return np.where(k2 == 0.0, -min(rx, ry), d)


@dataclass(frozen=True)
class Segment:
    """A path segment produced by :func:`LineTo` / :func:`CubicTo`."""

    kind: str  # "line" | "cubic"
    points: tuple[float, ...]  # line: (x, y); cubic: (c1x, c1y, c2x, c2y, x, y)


def LineTo(x, y) -> Segment:
    """A straight segment to ``(x, y)``."""
    return Segment("line", (_scalar(x), _scalar(y)))


def CubicTo(c1x, c1y, c2x, c2y, x, y) -> Segment:
    """A cubic Bézier to ``(x, y)`` with control points ``(c1*, c2*)``."""
    return Segment("cubic", tuple(_scalar(v) for v in (c1x, c1y, c2x, c2y, x, y)))


class Path(Shape):
    def __init__(self, start_x, start_y, segments) -> None:
        """
        A path of straight and cubic-Bézier segments, flattened to a polyline.

        Args:
            start_x (float): Starting point X.
            start_y (float): Starting point Y.
            segments (list[Segment]): One or more LineTo/CubicTo segments.
        """
        self._start = (_scalar(start_x), _scalar(start_y))
        self._segments = list(segments)
        if not self._segments:
            raise ValueError("Path needs at least one segment")
        for s in self._segments:
            if not isinstance(s, Segment):
                raise TypeError("Path segments must be LineTo(...) or CubicTo(...)")
        poly = self._flatten()
        self._px = np.array([p[0] for p in poly], dtype=float)
        self._py = np.array([p[1] for p in poly], dtype=float)

    def _flatten(self) -> list[tuple[float, float]]:
        pts = [self._start]
        cx, cy = self._start
        for seg in self._segments:
            if seg.kind == "line":
                cx, cy = seg.points
                pts.append((cx, cy))
            else:
                c1x, c1y, c2x, c2y, ex, ey = seg.points
                for k in range(1, CUBIC_STEPS + 1):
                    t = k / CUBIC_STEPS
                    mt = 1.0 - t
                    a = mt * mt * mt
                    b = 3.0 * mt * mt * t
                    c = 3.0 * mt * t * t
                    d = t * t * t
                    pts.append(
                        (
                            a * cx + b * c1x + c * c2x + d * ex,
                            a * cy + b * c1y + c * c2y + d * ey,
                        )
                    )
                cx, cy = ex, ey
        return pts

    def _poly_distance(self, gx, gy, closed: bool):
        px, py = self._px, self._py
        n = px.size
        dist = np.full_like(np.asarray(gx, dtype=float), np.inf)
        last = n if closed else n - 1
        for i in range(last):
            j = (i + 1) % n
            ax, ay, bx, by = px[i], py[i], px[j], py[j]
            abx, aby = bx - ax, by - ay
            apx, apy = gx - ax, gy - ay
            denom = abx * abx + aby * aby
            if denom == 0.0:
                d = np.sqrt(apx * apx + apy * apy)
            else:
                t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
                dx = apx - t * abx
                dy = apy - t * aby
                d = np.sqrt(dx * dx + dy * dy)
            dist = np.minimum(dist, d)
        return dist

    def _inside(self, gx, gy, mode: str):
        px, py = self._px, self._py
        n = px.size
        if mode == "EvenOdd":
            cn = np.zeros_like(np.asarray(gx, dtype=float))
            for i in range(n):
                j = (i + 1) % n
                ax, ay, bx, by = px[i], py[i], px[j], py[j]
                cond = (ay <= gy) != (by <= gy)
                with np.errstate(divide="ignore", invalid="ignore"):
                    xint = ax + (gy - ay) * (bx - ax) / (by - ay)
                cn = cn + np.where(cond & (gx < xint), 1.0, 0.0)
            return np.mod(cn, 2.0) >= 1.0
        wn = np.zeros_like(np.asarray(gx, dtype=float))
        for i in range(n):
            j = (i + 1) % n
            ax, ay, bx, by = px[i], py[i], px[j], py[j]
            cr = (bx - ax) * (gy - ay) - (gx - ax) * (by - ay)
            up = (ay <= gy) & (by > gy) & (cr > 0.0)
            down = (ay > gy) & (by <= gy) & (cr < 0.0)
            wn = wn + np.where(up, 1.0, 0.0) - np.where(down, 1.0, 0.0)
        return wn != 0.0

    def fill_distance(self, gx, gy, mode: str = "NonZero"):
        # Implicitly closed for filling.
        dist = self._poly_distance(gx, gy, closed=True)
        inside = self._inside(gx, gy, mode)
        return np.where(inside, -dist, dist)

    def stroke_distance(self, gx, gy):
        # Open for stroking (the path is not implicitly closed).
        return self._poly_distance(gx, gy, closed=False)


class Fill(Expression2D):
    @coerce_args
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        shape: Shape,
        mode: FillMode = "NonZero",
        feather: Annotated[float, Field(ge=0.0)] = 0.0,
    ) -> None:
        """
        Paint the interior of ``shape``.

        Args:
            shape (Shape): The shape (Rect/Ellipse/Path) to fill.
            mode (FillMode, optional): Fill rule for self-intersecting paths.
                Defaults to "NonZero".
            feather (float, optional): Soft-edge width in world units (0 =
                hard). Defaults to 0.0.
        """
        self._shape = shape
        self._mode = mode
        self._feather = feather

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        d = self._shape.fill_distance(x, y, self._mode)
        return float(_coverage(-d, self._feather))

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        gx, gy = coords if coords is not None else np.meshgrid(xs, ys)
        d = self._shape.fill_distance(gx, gy, self._mode)
        return _coverage(-d, self._feather)


class Stroke(Expression2D):
    @coerce_args
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        shape: Shape,
        width: Annotated[float, Field(gt=0.0)],
        feather: Annotated[float, Field(ge=0.0)] = 0.0,
    ) -> None:
        """
        Paint a band of the given ``width`` centered on ``shape``'s boundary.

        Args:
            shape (Shape): The shape (Rect/Ellipse/Path) to outline.
            width (float): Stroke width in world units (> 0).
            feather (float, optional): Soft-edge width in world units (0 =
                hard). Defaults to 0.0.
        """
        self._shape = shape
        self._width = width
        self._feather = feather

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        d = self._shape.stroke_distance(x, y)
        return float(_coverage(self._width / 2.0 - d, self._feather))

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        gx, gy = coords if coords is not None else np.meshgrid(xs, ys)
        d = self._shape.stroke_distance(gx, gy)
        return _coverage(self._width / 2.0 - d, self._feather)
