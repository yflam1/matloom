"""Tests for matloom.engine.noise: fBm and Worley noise generators."""

from __future__ import annotations

import math

import numpy as np
import pytest
from opensimplex import OpenSimplex
from opensimplex.internals import _noise2

from matloom.engine.expr import Constant, Sqrt
from matloom.engine.noise import Worley, _coords_out_dtype, fBm, get_random_seed, seed

# |-------------------|
# |   Seed helpers    |
# |-------------------|


def test_seed_makes_get_random_seed_deterministic():
    seed(123)
    a = get_random_seed()
    seed(123)
    b = get_random_seed()
    assert a == b


def test_get_random_seed_in_range():
    seed(0)
    for _ in range(50):
        s = get_random_seed()
        assert 0 <= s < 2**31


# |---------|
# |   fBm   |
# |---------|


def test_fbm_is_deterministic_for_fixed_seed():
    a = fBm(seed=42)
    b = fBm(seed=42)
    assert a(0.1, 0.2) == b(0.1, 0.2)


def test_fbm_different_seeds_differ():
    a = fBm(seed=1)(0.1, 0.2)
    b = fBm(seed=2)(0.1, 0.2)
    assert a != b


def test_fbm_default_range_is_signed():
    f = fBm(seed=7)
    vals = [f(x * 0.123, y * 0.321) for x in range(20) for y in range(20)]
    assert min(vals) >= -1.0
    assert max(vals) <= 1.0
    assert min(vals) < 0.0  # signed output should dip negative somewhere


def test_fbm_to_01_range():
    f = fBm(seed=7, to_01=True)
    vals = [f(x * 0.123, y * 0.321) for x in range(20) for y in range(20)]
    assert min(vals) >= 0.0
    assert max(vals) <= 1.0


def test_fbm_grid_matches_scalar():
    f = fBm(seed=99)
    xs = np.linspace(0.0, 2.0, 7)
    ys = np.linspace(0.0, 2.0, 6)
    grid = f.eval_grid(xs, ys)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            assert grid[i, j] == pytest.approx(f(x, y), abs=1e-9)


def test_fbm_single_octave_is_raw_simplex_scaled():
    # With octaves=1, output is one simplex sample * _FACTOR, clamped.
    f = fBm(octaves=1, seed=5)
    raw = f._raw(0.3 * f._base_freq_x, 0.7 * f._base_freq_y)
    assert f(0.3, 0.7) == pytest.approx(max(-1.0, min(raw, 1.0)))


def test_fbm_rejects_zero_base_freq():
    with pytest.raises(ValueError):
        fBm(base_freq=0.0, seed=1)


def test_fbm_rejects_octaves_below_one():
    with pytest.raises(ValueError):
        fBm(octaves=0, seed=1)


def test_fbm_separate_axis_frequencies():
    f = fBm(base_freq_x=2.0, base_freq_y=3.0, seed=1)
    assert f._base_freq_x == 2.0
    assert f._base_freq_y == 3.0


def test_fbm_base_freq_none_requires_both_axes():
    with pytest.raises(ValueError, match="base_freq"):
        fBm(base_freq=None, base_freq_x=2.0, seed=1)


def test_fbm_octaves_accepts_coerced_expression():
    # An Expression2D in an int param is evaluated to a scalar and rounded.
    f = fBm(octaves=Sqrt(Constant(9.0)), seed=1)  # -> 3
    assert f._octaves == 3


# |------------|
# |   Worley   |
# |------------|


def test_worley_deterministic_for_fixed_seed():
    a = Worley(seed=42)
    b = Worley(seed=42)
    assert a(0.1, 0.2) == b(0.1, 0.2)


def test_worley_grid_matches_scalar():
    w = Worley(seed=11)
    xs = np.linspace(0.0, 3.0, 7)
    ys = np.linspace(0.0, 3.0, 6)
    grid = w.eval_grid(xs, ys)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            assert grid[i, j] == pytest.approx(w(x, y), abs=1e-9)


@pytest.mark.parametrize("distance", ["euclidean", "manhattan", "chebyshev"])
@pytest.mark.parametrize("combination", ["F1", "F2", "F2-F1", "F2+F1"])
def test_worley_all_modes_grid_matches_scalar(distance, combination):
    w = Worley(distance=distance, combination=combination, seed=3)
    xs = np.linspace(0.0, 2.0, 5)
    ys = np.linspace(0.0, 2.0, 5)
    grid = w.eval_grid(xs, ys)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            assert grid[i, j] == pytest.approx(w(x, y), abs=1e-9)


def test_worley_to_01_is_bounded():
    w = Worley(seed=5, to_01=True)
    vals = [w(x * 0.21, y * 0.21) for x in range(20) for y in range(20)]
    assert min(vals) >= 0.0
    assert max(vals) <= 1.0


def test_worley_f1_non_negative():
    w = Worley(combination="F1", seed=5)
    vals = [w(x * 0.21, y * 0.21) for x in range(15) for y in range(15)]
    assert min(vals) >= 0.0


def test_worley_f2_minus_f1_non_negative():
    # F2 >= F1 always, so F2-F1 >= 0.
    w = Worley(combination="F2-F1", seed=5)
    vals = [w(x * 0.21, y * 0.21) for x in range(15) for y in range(15)]
    assert min(vals) >= 0.0


def test_worley_rejects_invalid_distance():
    with pytest.raises(ValueError):
        Worley(distance="taxicab", seed=1)


def test_worley_rejects_invalid_combination():
    with pytest.raises(ValueError):
        Worley(combination="F3", seed=1)


def test_worley_euclidean_matches_manual_distance():
    # Verify the euclidean metric against a direct two-point computation.
    w = Worley(distance="euclidean", combination="F1", seed=8)
    # _dist is the private metric; sanity-check it.
    assert w._dist(3.0, 4.0) == pytest.approx(5.0)


def test_worley_manhattan_metric():
    w = Worley(distance="manhattan", seed=8)
    assert w._dist(-3.0, 4.0) == pytest.approx(7.0)


def test_worley_chebyshev_metric():
    w = Worley(distance="chebyshev", seed=8)
    assert w._dist(-3.0, 4.0) == pytest.approx(4.0)


# |----------------------------------|
# |   Coords-path + fused kernels    |
# |----------------------------------|


def _rotate_shaped_coords(xs, ys, degrees=37.0):
    """Coords pair built exactly like Rotate.eval_grid's (cos*bx + sin*by),
    so the coords path sees the grid shapes and dtypes Rotate produces."""
    rad = degrees * math.pi / 180.0
    cos_, sin_ = math.cos(rad), math.sin(rad)
    bx = np.broadcast_to(xs[np.newaxis, :], (ys.size, xs.size))
    by = np.broadcast_to(ys[:, np.newaxis], (ys.size, xs.size))
    return cos_ * bx + sin_ * by, -sin_ * bx + cos_ * by


def _assert_bit_identical(got, want):
    """0-ulp oracle: same shape, same dtype, same bits via an integer view
    (never approx; the view also catches NaN and signed-zero drift)."""
    assert got.shape == want.shape
    assert got.dtype == want.dtype
    view = np.uint64 if np.dtype(got.dtype).itemsize == 8 else np.uint32
    assert np.array_equal(
        np.ascontiguousarray(got).view(view),
        np.ascontiguousarray(want).view(view),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("to_01", [True, False])
@pytest.mark.parametrize("combination", ["F1", "F2", "F2-F1", "F2+F1"])
@pytest.mark.parametrize("distance", ["euclidean", "manhattan", "chebyshev"])
def test_worley_coords_kernel_bit_matches_vectorize(
    distance, combination, to_01, dtype
):
    w = Worley(
        distance=distance,
        combination=combination,
        to_01=to_01,
        base_freq=7.3,
        seed=31,
    )
    xs = np.linspace(-1.4, 2.2, 25, dtype=dtype)
    ys = np.linspace(-0.8, 1.9, 20, dtype=dtype)
    xx, yy = _rotate_shaped_coords(xs, ys)
    grid = w.eval_grid(xs, ys, coords=(xx, yy))
    # The replaced path, run as-is: its otype-probe dtype inference included.
    vec = np.vectorize(w.__call__)(xx, yy)
    assert grid.dtype == vec.dtype
    _assert_bit_identical(grid, vec)
    # Scalar oracle: vectorize called __call__ per point with Python floats,
    # then cast each float64 result to the probe-inferred dtype on store.
    scalar = np.array(
        [w(float(a), float(b)) for a, b in zip(xx.ravel(), yy.ravel())],
        dtype=grid.dtype,
    ).reshape(xx.shape)
    _assert_bit_identical(grid, scalar)


def test_worley_coords_probe_predicts_float32_case():
    # On float32 coords the vectorize probe computes with float32
    # intermediates and, when its final clamps do not fire, returns
    # np.float32, so the old coords path emitted float32 arrays (27 of 57
    # real corpus coords calls are this case). Pin that the probe rule
    # reproduces it and the kernel output matches the old path bit-for-bit.
    rng = np.random.default_rng(0)
    xx = rng.uniform(-1.0, 1.0, (3, 4)).astype(np.float32)
    yy = rng.uniform(-1.0, 1.0, (3, 4)).astype(np.float32)
    w = Worley(
        distance="chebyshev",
        combination="F2-F1",
        base_freq_x=53,
        base_freq_y=41,
        to_01=True,
        seed=104,
    )
    assert _coords_out_dtype(w, (xx, yy)) is np.float32
    vec = np.vectorize(w.__call__)(xx, yy)
    assert vec.dtype == np.float32
    grid = w.eval_grid(np.empty(1), np.empty(1), coords=(xx, yy))
    assert grid.dtype == np.float32
    _assert_bit_identical(grid, vec)


_WORLEY_STRESS_CONFIGS = [
    dict(seed=12),
    dict(
        base_freq_x=800, base_freq_y=640, to_01=True, seed=55
    ),  # huge freq: clamps fire
    dict(base_freq=0.01, seed=8),  # tiny freq: v >> 1, clamps never fire
    dict(base_freq=-14, to_01=True, seed=144),  # negative freq
    dict(
        distance="chebyshev",
        combination="F2-F1",
        base_freq_x=53,
        base_freq_y=41,
        to_01=True,
        seed=104,
    ),
    dict(distance="manhattan", combination="F2+F1", seed=9),
]


def test_worley_coords_kernel_stress_battery_bit_matches_scalar():
    # >=1M points across configs and coord dtypes against the Python-float
    # __call__ oracle, on adversarial domains: cell boundaries (integer coords
    # under integer freqs, +-1e-6 straddles), huge/tiny/negative frequencies.
    total = 0
    for dtype in (np.float32, np.float64):
        rng = np.random.default_rng(4 if dtype == np.float32 else 5)
        n = 84_000
        for cfg in _WORLEY_STRESS_CONFIGS:
            w = Worley(**cfg)
            xr = rng.uniform(-3.0, 3.0, n // 2)
            lattice = rng.choice(np.arange(-3.0, 4.0), size=n - n // 2)
            eps = rng.choice([0.0, 1e-6, -1e-6, 0.5], size=lattice.size)
            xx = np.concatenate([xr, lattice + eps]).astype(dtype)
            yr = rng.uniform(-3.0, 3.0, n // 2)
            lattice_y = rng.choice(np.arange(-3.0, 4.0), size=n - n // 2)
            eps_y = rng.choice([0.0, 1e-6, -1e-6, 0.5], size=lattice_y.size)
            yy = np.concatenate([yr, lattice_y + eps_y]).astype(dtype)
            grid = w.eval_grid(np.empty(1), np.empty(1), coords=(xx, yy))
            assert grid.dtype == _coords_out_dtype(w, (xx, yy))
            # Same flat[0] as the big grid, so vectorize infers the same otype.
            probe_grid_xx = xx.ravel()[:4].reshape(2, 2)
            probe_grid_yy = yy.ravel()[:4].reshape(2, 2)
            assert (
                grid.dtype
                == np.vectorize(w.__call__)(probe_grid_xx, probe_grid_yy).dtype
            )
            scalar = np.array(
                [w(float(a), float(b)) for a, b in zip(xx, yy)],
                dtype=grid.dtype,
            )
            _assert_bit_identical(grid, scalar)
            total += xx.size
    assert total >= 1_000_000


_FBM_COORDS_CONFIGS = [
    dict(octaves=1, seed=1),
    dict(octaves=4, seed=2),
    dict(octaves=6, seed=3),
    dict(octaves=6, lacunarity=2.1, gain=0.56, seed=46),  # corpus 'corrosion'
    dict(octaves=4, base_freq=210, to_01=True, seed=83),  # corpus 'grain'
    dict(octaves=1, base_freq_x=43, base_freq_y=56, to_01=True, seed=172),  # aniso
    dict(octaves=3, base_freq=0.05, seed=7),  # huge period
    dict(octaves=6, lacunarity=3.7, gain=0.75, seed=9),
    dict(octaves=2, base_freq=-14, to_01=True, seed=144),  # negative freq
]


def test_fbm_coords_kernel_bit_matches_vectorize():
    # >=1M points across configs and coord dtypes: the kernel must equal both
    # the old np.vectorize path and the Python-float __call__ oracle,
    # bit-for-bit, on adversarial domains (simplex cell-boundary lattices
    # with +-1e-7 straddles, huge/tiny/negative frequencies).
    total = 0
    for dtype in (np.float32, np.float64):
        rng = np.random.default_rng(7 if dtype == np.float32 else 8)
        n = 56_000
        for cfg in _FBM_COORDS_CONFIGS:
            f = fBm(**cfg)
            xr = rng.uniform(-4.0, 4.0, n // 2)
            lattice = rng.choice(np.arange(-4.0, 5.0), size=n - n // 2)
            eps = rng.choice([0.0, 1e-7, -1e-7, 0.25], size=lattice.size)
            xx = np.concatenate([xr, lattice + eps]).astype(dtype)
            yr = rng.uniform(-4.0, 4.0, n - n // 2)
            lattice_y = rng.choice(np.arange(-4.0, 5.0), size=n // 2)
            eps_y = rng.choice([0.0, 1e-7, -1e-7, 0.25], size=lattice_y.size)
            yy = np.concatenate([yr, lattice_y + eps_y]).astype(dtype)
            grid = f.eval_grid(np.empty(1), np.empty(1), coords=(xx, yy))
            vec = np.vectorize(f.__call__)(xx, yy)
            assert grid.dtype == vec.dtype
            _assert_bit_identical(grid, vec)
            scalar = np.array(
                [f(float(a), float(b)) for a, b in zip(xx, yy)],
                dtype=grid.dtype,
            )
            _assert_bit_identical(grid, scalar)
            total += xx.size
    assert total >= 1_000_000


def _legacy_noise2array_eval(node, xs, ys):
    """The pre-kernel separable branch of fBm.eval_grid, verbatim, as the
    pinned bit-exactness oracle for the fused grid kernel: per-octave
    noise2array dispatch plus the numpy accumulation."""
    out = np.zeros((ys.size, xs.size))
    freq_x, freq_y = node._base_freq_x, node._base_freq_y
    amp, norm = 1.0, 0.0
    xs_scaled = np.empty_like(xs)
    ys_scaled = np.empty_like(ys)
    for _ in range(node._octaves):
        np.multiply(xs, freq_x, out=xs_scaled)
        np.multiply(ys, freq_y, out=ys_scaled)
        layer = node._raw_grid(xs_scaled, ys_scaled)
        np.multiply(layer, amp, out=layer)
        out += layer
        norm += amp
        freq_x *= node._lacunarity
        freq_y *= node._lacunarity
        amp *= node._gain
    out /= norm
    if node._to_01:
        out += 1.0
        out /= 2.0
    np.clip(out, 0.0 if node._to_01 else -1.0, 1.0, out=out)
    return out


_FBM_GRID_CONFIGS = [
    dict(octaves=1, seed=1),
    dict(octaves=2, seed=2),
    dict(octaves=4, seed=3),
    dict(octaves=6, seed=4),
    dict(octaves=8, seed=5),
    dict(octaves=6, lacunarity=2.1, gain=0.56, seed=46),
    dict(octaves=4, base_freq=210, to_01=True, seed=83),
    dict(octaves=1, base_freq_x=43, base_freq_y=56, to_01=True, seed=172),
    dict(octaves=6, base_freq=0.05, seed=7),
    dict(octaves=6, lacunarity=3.7, gain=0.75, seed=9),
    dict(octaves=2, base_freq=-14, to_01=True, seed=144),
    dict(
        octaves=5,
        lacunarity=1.13,
        gain=0.71,
        base_freq=3.8,
        to_01=True,
        seed=46,
    ),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fbm_grid_fused_bit_matches_noise2array(dtype):
    # The fused kernel must reproduce the legacy per-octave noise2array loop
    # exactly, including the per-octave float32 rounding of the axis scaling
    # on float32 axes (the reason the scaling stays in numpy).
    rng = np.random.default_rng(6)
    for cfg in _FBM_GRID_CONFIGS:
        f = fBm(**cfg)
        xs = rng.uniform(0.0, 1.0, 96).astype(dtype)
        ys = rng.uniform(0.0, 1.0, 80).astype(dtype)
        got = f.eval_grid(xs, ys)
        want = _legacy_noise2array_eval(f, xs, ys)
        assert got.dtype == np.float64
        _assert_bit_identical(got, want)


def test_fbm_grid_unsupported_axis_dtype_keeps_legacy_behavior():
    # Escape hatch: axis dtypes outside float32/float64 route to the legacy
    # numpy loop. No such dtype completes on this stack (float16 dies inside
    # noise2array's own numba kernel, ints in np.multiply's out-cast) — the
    # old path raised the same errors, so pin identical behavior, plus real
    # value coverage on mixed float32/float64 axes (both kernel-supported).
    f = fBm(octaves=2, seed=11)
    xs16 = np.linspace(0.0, 1.0, 8, dtype=np.float16)
    ys16 = np.linspace(0.0, 1.0, 6, dtype=np.float16)
    with pytest.raises(NotImplementedError):
        f.eval_grid(xs16, ys16)
    with pytest.raises(NotImplementedError):
        _legacy_noise2array_eval(f, xs16, ys16)
    g = fBm(octaves=5, seed=13)
    xm = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    ym = np.linspace(0.0, 1.0, 7, dtype=np.float64)
    _assert_bit_identical(g.eval_grid(xm, ym), _legacy_noise2array_eval(g, xm, ym))


def test_opensimplex_internals_contract():
    # The coords/grid kernels import opensimplex.internals._noise2 and pass
    # OpenSimplex._perm straight to it (opensimplex is pinned to 0.4.5.1 in
    # pyproject.toml), so an upgrade that changes these internals must fail
    # here, loudly, instead of drifting the noise bits.
    s = OpenSimplex(7)
    assert isinstance(s._perm, np.ndarray)
    assert s._perm.dtype == np.int64
    assert s._perm.shape == (256,)
    assert OpenSimplex.noise2(s, 0.1, 0.2) == _noise2(0.1, 0.2, s._perm)
    assert type(_noise2(0.1, 0.2, s._perm)) is float


@pytest.mark.parametrize("factory", [lambda: Worley(seed=1), lambda: fBm(seed=1)])
def test_noise_coords_empty_grids_match_vectorize(factory):
    # np.vectorize refused size-0 inputs because it cannot run its otype
    # probe; the kernel path mirrors that ValueError instead of the
    # IndexError coords[0].flat[0] would raise.
    node = factory()
    xx = np.empty((0, 4), dtype=np.float32)
    yy = np.empty((0, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="size 0 inputs"):
        node.eval_grid(np.empty(0), np.empty(0), coords=(xx, yy))
