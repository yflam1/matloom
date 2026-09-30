import math
import random
from typing import Annotated, Any, Literal

import numpy as np
from numba import njit, prange
from opensimplex.internals import _noise2
from pydantic import Field, validate_call

from matloom import logger
from matloom.engine.expr import Coords, Expression2D, coerce_args
from matloom.utils.dtypes import NonZeroFloat

# Module-level twin of fBm._FACTOR: the numba kernels below cannot resolve the
# class attribute, so they reference this constant directly. Keep the
# expression byte-identical to the class one (same double, same bits).
_FBM_FACTOR = 2 / (3**0.5)

# Axis dtypes the fused separable fBm kernel is compiled and 0-ulp-verified
# for; any other dtype falls back to the legacy per-octave numpy loop.
_GRID_AXIS_DTYPES = (np.float32, np.float64)


def seed(seed: int | None = None) -> None:
    if seed is None:
        logger.warning("seed is None, not seeding")
        return
    random.seed(seed)
    np.random.seed(seed)


def get_random_seed() -> int:
    return random.randint(0, 2**31 - 1)


class _Noise(Expression2D):
    @coerce_args
    @validate_call
    def __init__(
        self,
        *,
        base_freq: NonZeroFloat | None = 1.0,
        base_freq_x: NonZeroFloat | None = None,
        base_freq_y: NonZeroFloat | None = None,
        to_01: bool = False,
        seed: Annotated[int, Field(ge=0)] | None = None,
        set_args: frozenset[str] | None = None,
    ) -> None:
        """
        ABC for noise generators.

        Args:
            base_freq (NonZeroFloat | None, optional): Base frequency for
                the noise. Defaults to 1.0.
            base_freq_x (NonZeroFloat | None, optional): Base frequency
                along the x-axis. If None, `base_freq` is used. Defaults to
                None.
            base_freq_y (NonZeroFloat | None, optional): Base frequency
                along the y-axis. If None, `base_freq` is used. Defaults to
                None.
            to_01 (bool, optional): If True, scale output to [0, 1]. Defaults
                to False.
            seed (Annotated[int, Field(ge=0)] | None, optional): Random seed
                for noise generation. If None, a random seed is used. Defaults
                to None.
            set_args (frozenset[str] | None, optional): The names of the
                keyword/positional args the author explicitly wrote, threaded in
                by the parser so :func:`matloom.engine.serialize.serialize_expr`
                emits only those (preserving explicit defaults like
                ``octaves=6``). ``None`` (code-constructed noise) falls back to
                value-comparison emission. ``seed`` is always emitted regardless.
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
        self._to_01 = to_01
        self._seed = get_random_seed() if seed is None else seed
        self._set_args = set_args


class fBm(_Noise):
    _FACTOR = _FBM_FACTOR

    @coerce_args
    @validate_call
    def __init__(
        self,
        octaves: Annotated[int, Field(ge=1)] = 6,
        lacunarity: Annotated[float, Field(gt=0.0)] = 2.0,
        gain: Annotated[float, Field(gt=0.0)] = 0.5,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the fBm noise generator.

        Args:
            octaves (int, optional): Number of simplex noise layers to combine.
                Set to 1 if you only want raw simplex noise. Defaults to 6.
            lacunarity (float, optional): Frequency multiplier per octave.
                Defaults to 2.0.
            gain (float, optional): Amplitude multiplier per octave. Defaults
                to 0.5.
            **kwargs: Additional keyword arguments passed to the :class:`Noise`
                base class.
        """
        from opensimplex import OpenSimplex

        super().__init__(**kwargs)
        self._octaves = octaves
        self._lacunarity = lacunarity
        self._gain = gain
        self._simplex = OpenSimplex(self._seed)

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        freq_x, freq_y = self._base_freq_x, self._base_freq_y
        total, amp, norm = 0.0, 1.0, 0.0
        for _ in range(self._octaves):
            total += self._raw(x * freq_x, y * freq_y) * amp
            norm += amp
            freq_x *= self._lacunarity
            freq_y *= self._lacunarity
            amp *= self._gain
        out = total / norm
        if self._to_01:
            out = (out + 1.0) / 2.0
        out = max(0.0 if self._to_01 else -1.0, min(out, 1.0))
        return out

    def eval_grid(self, xs: np.ndarray, ys: np.ndarray, coords=None) -> np.ndarray:
        if coords is not None:
            # Non-separable coords (e.g. under Rotate): flat numba kernel that
            # replicates __call__ per cell (the old np.vectorize fallback was
            # per-pixel Python), with the output dtype vectorize would have
            # inferred (see _coords_out_dtype).
            xx, yy = coords
            if xx.shape != yy.shape:
                # The flat kernel indexes both grids element-for-element; the
                # np.vectorize path it replaced broadcast the pair instead.
                raise ValueError(
                    f"coords grids must have the same shape, got {xx.shape} "
                    f"and {yy.shape}"
                )
            out = np.empty(xx.size, dtype=_coords_out_dtype(self, coords))
            _fbm_coords_kernel(
                xx.ravel(),
                yy.ravel(),
                out,
                self._simplex._perm,
                self._octaves,
                self._lacunarity,
                self._gain,
                self._base_freq_x,
                self._base_freq_y,
                self._to_01,
            )
            return out.reshape(xx.shape)
        if xs.dtype not in _GRID_AXIS_DTYPES or ys.dtype not in _GRID_AXIS_DTYPES:
            # Escape hatch: an axis dtype the fused kernel is not verified
            # for keeps the legacy numpy loop, bit-identical to the old path.
            return self._eval_grid_numpy_loop(xs, ys)
        # Fused separable path: the per-octave 1-D axis scaling stays in numpy
        # (np.multiply with an out buffer preserves the per-octave float32
        # rounding on float32 axes; computing it in-kernel would round in
        # float64 and change bits), and one kernel fuses the octave loop.
        octs = self._octaves
        xs_scaled = np.empty((octs, xs.size), dtype=xs.dtype)
        ys_scaled = np.empty((octs, ys.size), dtype=ys.dtype)
        fx, fy = self._base_freq_x, self._base_freq_y
        for k in range(octs):
            np.multiply(xs, fx, out=xs_scaled[k])
            np.multiply(ys, fy, out=ys_scaled[k])
            fx *= self._lacunarity
            fy *= self._lacunarity
        out = np.empty((ys.size, xs.size))
        _fbm_grid_kernel(xs_scaled, ys_scaled, out, self._simplex._perm, self._gain)
        if self._to_01:
            out += 1.0
            out /= 2.0
        np.clip(out, 0.0 if self._to_01 else -1.0, 1.0, out=out)
        return out

    def _eval_grid_numpy_loop(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Legacy per-octave noise2array loop, kept verbatim as the fallback
        for axis dtypes ``_fbm_grid_kernel`` does not support."""
        out = np.zeros((ys.size, xs.size))
        freq_x, freq_y = self._base_freq_x, self._base_freq_y
        amp, norm = 1.0, 0.0
        xs_scaled = np.empty_like(xs)
        ys_scaled = np.empty_like(ys)
        for _ in range(self._octaves):
            np.multiply(xs, freq_x, out=xs_scaled)
            np.multiply(ys, freq_y, out=ys_scaled)
            layer = self._raw_grid(xs_scaled, ys_scaled)
            np.multiply(layer, amp, out=layer)
            out += layer
            norm += amp
            freq_x *= self._lacunarity
            freq_y *= self._lacunarity
            amp *= self._gain
        out /= norm
        if self._to_01:
            out += 1.0
            out /= 2.0
        np.clip(out, 0.0 if self._to_01 else -1.0, 1.0, out=out)
        return out

    def _raw(self, x: float, y: float) -> float:
        return self._simplex.noise2(x, y) * self._FACTOR

    def _raw_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self._simplex.noise2array(xs, ys) * self._FACTOR


@njit(cache=True, parallel=True)
def _fbm_coords_kernel(
    xs: np.ndarray,
    ys: np.ndarray,
    out: np.ndarray,
    perm: np.ndarray,
    octaves: int,
    lacunarity: float,
    gain: float,
    base_freq_x: float,
    base_freq_y: float,
    to_01: bool,
) -> None:
    """Per-cell fBm over non-separable coords, replicating ``fBm.__call__``
    op-for-op in float64 (np.vectorize called ``__call__`` per point with
    Python floats), so the coords path stays bit-identical to the old one."""
    for i in prange(xs.size):
        freq_x = base_freq_x
        freq_y = base_freq_y
        total = 0.0
        amp = 1.0
        norm = 0.0
        for _ in range(octaves):
            # Left-associated like __call__: (noise2 * _FACTOR) * amp.
            total += _noise2(xs[i] * freq_x, ys[i] * freq_y, perm) * _FBM_FACTOR * amp
            norm += amp
            freq_x = freq_x * lacunarity
            freq_y = freq_y * lacunarity
            amp = amp * gain
        out[i] = total / norm
    if to_01:
        for i in range(xs.size):
            v = (out[i] + 1.0) / 2.0
            out[i] = max(0.0, min(v, 1.0))
    else:
        for i in range(xs.size):
            out[i] = max(-1.0, min(out[i], 1.0))


@njit(cache=True, parallel=True)
def _fbm_grid_kernel(
    xs_s: np.ndarray,
    ys_s: np.ndarray,
    out: np.ndarray,
    perm: np.ndarray,
    gain: float,
) -> None:
    """Fused separable fBm: xs_s/ys_s hold the per-octave scaled 1-D axes
    (stacked by eval_grid with numpy multiplies so float32 axes keep their
    per-octave rounding), and this kernel folds the per-octave noise2array
    dispatches plus the numpy accumulation into one pass. noise2array is
    opensimplex's ``_noise2a``, which evaluates ``_noise2(x[x_i], y[y_i])``
    over the same axis values, and the term grouping ``(n * _FACTOR) * amp``
    with ``norm += amp`` before ``amp *= gain`` matches the old
    ``out = zeros; out += layer`` sequence exactly."""
    octaves = xs_s.shape[0]
    nx = xs_s.shape[1]
    ny = ys_s.shape[1]
    for i in prange(ny):
        for j in range(nx):
            total = 0.0
            amp = 1.0
            norm = 0.0
            for o in range(octaves):
                total += _noise2(xs_s[o, j], ys_s[o, i], perm) * _FBM_FACTOR * amp
                norm += amp
                amp = amp * gain
            out[i, j] = total / norm


@njit(cache=True)
def _hash_to_01(cx: int, cy: int, seed: int, which: int) -> float:
    """
    Deterministic per-cell float in [0, 1) via PCG-style integer hash.
    """
    H = 0xFFFFFFFFFFFFFFFF
    h = (cx * 1664525 + cy * 22695477 + seed * 6364136223846793005 + which) & H
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & H
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & H
    h ^= h >> 33
    return (h & 0xFFFFFFFF) / 0x100000000


@njit(cache=True, parallel=True)
def _worley_kernel(
    xs: np.ndarray,
    ys: np.ndarray,
    out: np.ndarray,
    base_freq_x: float,
    base_freq_y: float,
    seed: int,
    distance: int,
    combination: int,
    norm: float,
    to_01: bool,
) -> None:
    for i in prange(ys.size):
        y = ys[i] * base_freq_y
        iy = int(math.floor(y))
        for j in range(xs.size):
            x = xs[j] * base_freq_x
            ix = int(math.floor(x))
            f1 = math.inf
            f2 = math.inf
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cx = ix + dx
                    cy = iy + dy
                    px = _hash_to_01(cx, cy, seed, 0) + cx
                    py = _hash_to_01(cx, cy, seed, 1) + cy
                    ddx = x - px
                    ddy = y - py
                    if distance == 0:
                        d = math.sqrt(ddx * ddx + ddy * ddy)
                    elif distance == 1:
                        d = abs(ddx) + abs(ddy)
                    else:
                        d = max(abs(ddx), abs(ddy))
                    if d < f1:
                        f2 = f1
                        f1 = d
                    elif d < f2:
                        f2 = d
            if combination == 0:
                v = f1
            elif combination == 1:
                v = f2
            elif combination == 2:
                v = f2 - f1
            else:
                v = f2 + f1
            if to_01:
                v = min(v / norm, 1.0)
            out[i, j] = max(0.0, v)


@njit(cache=True, parallel=True)
def _worley_coords_kernel(
    xs: np.ndarray,
    ys: np.ndarray,
    out: np.ndarray,
    base_freq_x: float,
    base_freq_y: float,
    seed: int,
    distance: int,
    combination: int,
    norm: float,
    to_01: bool,
) -> None:
    """Per-cell Worley over non-separable coords: the inner body of
    ``_worley_kernel`` evaluated over flat coordinate arrays, replicating
    ``Worley.__call__`` op-for-op in float64 (np.vectorize called ``__call__``
    per point with Python floats). ``out`` carries the dtype vectorize would
    have inferred (see ``_coords_out_dtype``), so a float32 output is the same
    float64 value rounded once on store."""
    for i in prange(xs.size):
        x = xs[i] * base_freq_x
        y = ys[i] * base_freq_y
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        f1 = math.inf
        f2 = math.inf
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                cx = ix + dx
                cy = iy + dy
                px = _hash_to_01(cx, cy, seed, 0) + cx
                py = _hash_to_01(cx, cy, seed, 1) + cy
                ddx = x - px
                ddy = y - py
                if distance == 0:
                    d = math.sqrt(ddx * ddx + ddy * ddy)
                elif distance == 1:
                    d = abs(ddx) + abs(ddy)
                else:
                    d = max(abs(ddx), abs(ddy))
                if d < f1:
                    f2 = f1
                    f1 = d
                elif d < f2:
                    f2 = d
        if combination == 0:
            v = f1
        elif combination == 1:
            v = f2
        elif combination == 2:
            v = f2 - f1
        else:
            v = f2 + f1
        if to_01:
            v = min(v / norm, 1.0)
        out[i] = max(0.0, v)


def _coords_out_dtype(node: "Worley | fBm", coords: Coords) -> type:
    """Predict the output dtype the old ``np.vectorize(node.__call__)`` coords
    path produced, so the flat kernels stay byte-compatible with it.

    np.vectorize infers its output type from one probe call on the first
    elements of the input arrays, boxed as numpy scalars (numpy's
    ``_get_ufunc_and_otypes`` uses ``arg.flat[0]``), while its per-element
    calls receive Python floats. On float32 grids the probe computes with
    float32 intermediates, and when its final min/max clamps do not fire it
    returns ``np.float32``, so vectorize emitted a float32 array: every float64
    per-point value rounded once on store. Float32 output iff the probe
    returns ``np.float32``.
    """
    if coords[0].size == 0 or coords[1].size == 0:
        # Mirror the replaced np.vectorize path, which refused size-0 inputs
        # because it cannot run its otype probe.
        raise ValueError(
            "cannot call `vectorize` on size 0 inputs unless `otypes` is set"
        )
    probe = node.__call__(coords[0].flat[0], coords[1].flat[0])
    return np.float32 if isinstance(probe, np.float32) else np.float64


class Worley(_Noise):
    # Tight upper bounds verified by global optimization. Empirical sampling
    # always falls short because the maximal configurations are degenerate
    # (measure zero under random placement), so these values cannot be
    # confirmed by grid sampling alone.
    _NORM = {"euclidean": math.sqrt(2), "manhattan": 2.0, "chebyshev": 1.0}
    _NORM_F2 = {
        "euclidean": math.sqrt(2.5),
        "manhattan": 2.0,
        "chebyshev": 1.5,
    }
    _NORM_F2_MINUS_F1 = {
        "euclidean": math.sqrt(2.5),
        "manhattan": 2.0,
        "chebyshev": 1.5,
    }
    _NORM_F2_PLUS_F1 = {
        "euclidean": 2 * math.sqrt(2),
        "manhattan": 4.0,
        "chebyshev": 2.0,
    }

    _DISTANCE_CODE = {"euclidean": 0, "manhattan": 1, "chebyshev": 2}
    _COMBINATION_CODE = {"F1": 0, "F2": 1, "F2-F1": 2, "F2+F1": 3}

    @validate_call
    def __init__(
        self,
        distance: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
        combination: Literal["F1", "F2", "F2-F1", "F2+F1"] = "F1",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Worley noise generator.

        Args:
            distance (str, optional): Distance metric used to measure proximity
                to feature points. "euclidean" produces round cells,
                "manhattan" produces diamond-shaped cells, and "chebyshev"
                produces square cells. Defaults to "euclidean".
            combination (str, optional): How F1 and F2 distances are combined.
                "F1" uses only the closest point distance (classic cellular
                look); "F2" uses the second-closest (larger, rounder regions);
                "F2-F1" subtracts them to produce sharp ridges along cell
                boundaries; "F2+F1" adds them for a blobby appearance. Defaults
                to "F1".
            **kwargs: Additional keyword arguments passed to the :class:`Noise`
                base class.
        """
        super().__init__(**kwargs)
        self._distance = distance
        self._combination = combination
        match combination:
            case "F1":
                self._norm = self._NORM[distance]
            case "F2":
                self._norm = self._NORM_F2[distance]
            case "F2-F1":
                self._norm = self._NORM_F2_MINUS_F1[distance]
            case "F2+F1":
                self._norm = self._NORM_F2_PLUS_F1[distance]

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        x, y = x * self._base_freq_x, y * self._base_freq_y
        ix, iy = math.floor(x), math.floor(y)
        f1, f2 = math.inf, math.inf
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                cx, cy = ix + dx, iy + dy
                px = _hash_to_01(cx, cy, self._seed, 0) + cx
                py = _hash_to_01(cx, cy, self._seed, 1) + cy
                d = self._dist(x - px, y - py)
                if d < f1:
                    f1, f2 = d, f1
                elif d < f2:
                    f2 = d
        match self._combination:
            case "F1":
                out = f1
            case "F2":
                out = f2
            case "F2-F1":
                out = f2 - f1
            case "F2+F1":
                out = f2 + f1
            case _:
                raise ValueError(f"Invalid combination: {self._combination}")
        if self._to_01:
            out = out / self._norm
        return max(0.0, min(out, 1.0 if self._to_01 else math.inf))

    def eval_grid(self, xs: np.ndarray, ys: np.ndarray, coords=None) -> np.ndarray:
        if coords is not None:
            # Non-separable coords (e.g. under Rotate): flat numba kernel that
            # replicates __call__ per cell (the old np.vectorize fallback was
            # per-pixel Python), with the output dtype vectorize would have
            # inferred (see _coords_out_dtype).
            xx, yy = coords
            if xx.shape != yy.shape:
                # The flat kernel indexes both grids element-for-element; the
                # np.vectorize path it replaced broadcast the pair instead.
                raise ValueError(
                    f"coords grids must have the same shape, got {xx.shape} "
                    f"and {yy.shape}"
                )
            out = np.empty(xx.size, dtype=_coords_out_dtype(self, coords))
            _worley_coords_kernel(
                xx.ravel(),
                yy.ravel(),
                out,
                self._base_freq_x,
                self._base_freq_y,
                self._seed,
                self._DISTANCE_CODE[self._distance],
                self._COMBINATION_CODE[self._combination],
                self._norm,
                self._to_01,
            )
            return out.reshape(xx.shape)
        out = np.empty((ys.size, xs.size))
        _worley_kernel(
            xs,
            ys,
            out,
            self._base_freq_x,
            self._base_freq_y,
            self._seed,
            self._DISTANCE_CODE[self._distance],
            self._COMBINATION_CODE[self._combination],
            self._norm,
            self._to_01,
        )
        return out

    def _dist(self, dx: float, dy: float) -> float:
        match self._distance:
            case "euclidean":
                return math.sqrt(dx * dx + dy * dy)
            case "manhattan":
                return abs(dx) + abs(dy)
            case "chebyshev":
                return max(abs(dx), abs(dy))
            case _:
                raise ValueError(f"Invalid distance: {self._distance}")
