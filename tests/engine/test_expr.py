"""Tests for matloom.engine.expr: the Expression2D tree and its operators."""

from __future__ import annotations

import gc
import math
import weakref

import numpy as np
import pytest

from matloom.engine.expr import (
    Abs,
    Add,
    AdditionExpression2D,
    Ceil,
    Constant,
    Cos,
    Div,
    Floor,
    Log,
    Max,
    Min,
    Mul,
    Pow,
    Ref,
    Sin,
    Sqrt,
    Sub,
    X,
    Y,
    _grid_cache,
    _GridCache,
    grid_cache_scope,
)
from matloom.engine.layer import Layer
from matloom.engine.main import LayeredMaterial
from matloom.engine.noise import fBm
from matloom.engine.preview import fixed_light_preview
from tests.conftest import assert_grid_matches_call

# |-----------------|
# |   Leaf nodes    |
# |-----------------|


def test_constant_returns_value_everywhere():
    c = Constant(0.7)
    assert c() == 0.7
    assert c(3.0, -2.0) == 0.7


def test_constant_default_is_zero():
    assert Constant()() == 0.0


def test_constant_integer_literal_grid_is_float():
    # A bare integer constant (e.g. ``Layer(1)``) must still evaluate to a
    # floating grid; an int64 grid would raise a cast error in downstream
    # in-place float math (e.g. relief.composite_height).
    grid = Constant(1).eval_grid(np.array([0.0, 1.0]), np.array([0.0]))
    assert np.issubdtype(grid.dtype, np.floating)
    np.testing.assert_array_equal(grid, [[1.0, 1.0]])


def test_x_and_y_select_coordinates():
    assert X()(0.3, 0.9) == 0.3
    assert Y()(0.3, 0.9) == 0.9


def test_x_y_default_coordinates_are_zero():
    assert X()() == 0.0
    assert Y()() == 0.0


def test_leaf_grids(axes):
    xs, ys = axes
    assert_grid_matches_call(Constant(0.25), xs, ys)
    assert_grid_matches_call(X(), xs, ys)
    assert_grid_matches_call(Y(), xs, ys)


# |---------------------|
# |   Binary operators  |
# |---------------------|


@pytest.mark.parametrize(
    "build, expected",
    [
        (lambda: X() + Y(), 0.3 + 0.4),
        (lambda: X() - Y(), 0.3 - 0.4),
        (lambda: X() * Y(), 0.3 * 0.4),
        (lambda: X() / Y(), 0.3 / 0.4),
        (lambda: X() ** Y(), 0.3**0.4),
    ],
)
def test_operator_overloads(build, expected):
    assert build()(0.3, 0.4) == pytest.approx(expected)


def test_reflected_operators_wrap_scalars():
    # 2 + X() must coerce the bare int into a Constant via __radd__.
    assert (2 + X())(0.5) == 2.5
    assert (2 - X())(0.5) == 1.5
    assert (2 * X())(0.5) == 1.0
    assert (2 / X())(0.5) == 4.0
    assert (2 ** X())(3.0) == 8.0


def test_negation():
    assert (-X())(0.6) == -0.6


def test_functional_constructors_match_operators():
    assert Add(X(), Y())(0.1, 0.2) == pytest.approx((X() + Y())(0.1, 0.2))
    assert Sub(X(), Y())(0.5, 0.2) == pytest.approx(0.3)
    assert Mul(X(), Y())(0.5, 0.4) == pytest.approx(0.2)
    assert Div(X(), Y())(0.5, 0.4) == pytest.approx(1.25)
    assert Pow(X(), Y())(2.0, 3.0) == pytest.approx(8.0)


def test_binary_grids(axes):
    xs, ys = axes
    assert_grid_matches_call(X() + Y(), xs, ys)
    assert_grid_matches_call(X() * Y(), xs, ys)
    assert_grid_matches_call(X() ** (Y() + 1), xs, ys)


# |-------------------|
# |   Division guard  |
# |-------------------|


def test_division_by_zero_scalar_warns_and_returns_zero():
    expr = Constant(1.0) / Constant(0.0)
    with pytest.warns(RuntimeWarning, match="Division by zero"):
        assert expr(0.0, 0.0) == 0.0


def test_division_by_zero_grid_warns_and_zeroes(axes):
    xs, ys = axes
    expr = Constant(1.0) / X()  # x == 0 in the first column
    with pytest.warns(RuntimeWarning, match="Division by zero"):
        grid = expr.eval_grid(xs, ys)
    assert np.all(grid[:, 0] == 0.0)
    assert np.all(np.isfinite(grid))


# |-------------------|
# |   Unary functions |
# |-------------------|


@pytest.mark.parametrize(
    "cls, fn, arg",
    [
        (Sin, math.sin, 0.7),
        (Cos, math.cos, 0.7),
        (Abs, abs, -0.7),
        (Sqrt, math.sqrt, 0.49),
        (Floor, math.floor, 2.7),
        (Ceil, math.ceil, 2.1),
    ],
)
def test_unary_functions(cls, fn, arg):
    assert cls(Constant(arg))() == pytest.approx(float(fn(arg)))


def test_unary_grids(axes):
    xs, ys = axes
    for cls in (Sin, Cos, Abs, Sqrt, Floor, Ceil):
        assert_grid_matches_call(cls(X() + 0.1), xs, ys)


def test_min_max():
    assert Min(Constant(0.2), Constant(0.8))() == 0.2
    assert Max(Constant(0.2), Constant(0.8))() == 0.8


def test_min_max_grids(axes):
    xs, ys = axes
    assert_grid_matches_call(Min(X(), Y()), xs, ys)
    assert_grid_matches_call(Max(X(), Y()), xs, ys)


# |---------|
# |   Log   |
# |---------|


def test_log_natural_default_base():
    assert Log(Constant(math.e))() == pytest.approx(1.0)


@pytest.mark.parametrize("base, value, expected", [(10, 1000, 3.0), (2, 8, 3.0)])
def test_log_special_bases(base, value, expected):
    assert Log(Constant(value), base)() == pytest.approx(expected)


def test_log_arbitrary_base():
    assert Log(Constant(27), 3)() == pytest.approx(3.0)


def test_log_grid_special_bases(axes):
    xs, ys = axes
    assert_grid_matches_call(Log(X() + 1.0, 10), xs, ys)
    assert_grid_matches_call(Log(X() + 1.0, 2), xs, ys)
    assert_grid_matches_call(Log(X() + 1.0, 5), xs, ys)


# |-------------------------|
# |   coerce_args behavior  |
# |-------------------------|


def test_bare_number_coerced_to_constant_in_expression_param():
    # Add's params are Expression2D; passing ints must wrap them in Constant.
    expr = Add(1, 2)
    assert isinstance(expr, AdditionExpression2D)
    assert expr() == 3.0


def test_expression_coerced_to_scalar_in_numeric_param():
    # Log's `base` is a float param; an Expression2D there is evaluated via
    # call() at the origin. Log(1000, base=Log(... )) style: use Sqrt(2) ~ 1.414.
    base_expr = Sqrt(Constant(4.0))  # -> 2.0 scalar
    log = Log(Constant(8.0), base_expr)
    assert log() == pytest.approx(3.0)


def test_repr_roundtrips_class_name():
    assert repr(Constant(1.5)) == "Constant(1.5)"
    assert repr(X()) == "X()"
    assert "+" in repr(X() + Y())


def test_default_call_uses_origin():
    # Sin with no args evaluates at x=y=0.
    assert Sin(X())() == pytest.approx(0.0)


# |--------------------------|
# |   Ref grid-cache memo    |
# |--------------------------|

# A Define-heavy program in the corpus style: `tone` feeds five channel
# consumers, `grain` one, so the shared fBm behind `tone` is the dedupe target.
_TONE_GRAIN_PROGRAM = """\
View(0, 0, 6, 6)
Define(tone, fBm(octaves=4, base_freq=3, to_01=True, seed=7))
Define(grain, fBm(octaves=5, base_freq=25, to_01=True, seed=11))
Material(
  Layer(1)
    .basecolor((190 + (tone * 50)), (45 + (tone * 25)), (35 + (tone * 18)))
    .roughness((0.4 + (grain * 0.2)))
    .metallic(tone)
    .height((tone * 0.04))
)
"""


def test_ref_grid_cache_dedupes_shared_defines(monkeypatch):
    # Every use of `tone` routes through a Ref to the SAME fBm object (the
    # parser shares one expr per Define), so inside a render scope its
    # eval_grid runs exactly once no matter how many channels consume it.
    material = LayeredMaterial.deserialize(_TONE_GRAIN_PROGRAM)
    tone = dict(material._defs)["tone"]
    calls = []
    original = fBm.eval_grid

    def counting(self, xs, ys, coords=None):
        if self is tone:
            calls.append(1)
        return original(self, xs, ys, coords)

    monkeypatch.setattr(fBm, "eval_grid", counting)
    fixed_light_preview(material, 64, 64)
    assert sum(calls) == 1  # five consumers, one evaluation

    calls.clear()
    fixed_light_preview(material, 64, 64, memo=False)
    assert sum(calls) > 1  # escape hatch: every consumer re-evaluates


def test_ref_grid_cache_ignores_non_ref_sharing(monkeypatch):
    # Code-built sharing that never routes through a Ref is NOT deduped (the
    # hook is the Ref boundary by design), it just evaluates per consumer.
    calls = []
    original = fBm.eval_grid

    def counting(self, xs, ys, coords=None):
        calls.append(1)
        return original(self, xs, ys, coords)

    shared = fBm(octaves=3, base_freq=8, to_01=True, seed=5)
    material = LayeredMaterial()
    material.add_layer(
        Layer(
            alpha=Constant(1.0),
            roughness=shared,
            metallic=shared,
            height=shared * 0.05,
        )
    )
    monkeypatch.setattr(fBm, "eval_grid", counting)
    fixed_light_preview(material, 64, 64)
    assert sum(calls) == 3  # roughness + metallic + height, no memo shortcut


def test_ref_grid_cache_no_false_sharing():
    # Two structurally identical but DISTINCT seed-less fBm constructions
    # realize different seeds, so they must never merge (an id-keyed cache
    # keeps them apart; a structural/repr key would falsely share them — do
    # not switch to one).
    fbm_a = fBm(octaves=4, base_freq=7, to_01=True)
    fbm_b = fBm(octaves=4, base_freq=7, to_01=True)
    xs = np.linspace(0.0, 6.0, 16, dtype=np.float32)
    ys = np.linspace(0.0, 6.0, 16, dtype=np.float32)
    with grid_cache_scope():
        a = Ref("a", fbm_a).eval_grid(xs, ys)
        b = Ref("b", fbm_b).eval_grid(xs, ys)
        assert Ref("a", fbm_a).eval_grid(xs, ys) is a  # repeat lookup hits
    assert a is not b
    assert not np.array_equal(a, b)


def test_grid_cache_noop_without_ref():
    # Plain programs (no Ref) never touch the cache: the memo is inert by
    # construction outside the Ref delegate path.
    xs = np.linspace(0.0, 6.0, 8, dtype=np.float32)
    ys = np.linspace(0.0, 6.0, 8, dtype=np.float32)
    expr = fBm(octaves=3, base_freq=9, to_01=True, seed=4)
    with grid_cache_scope():
        cache = _grid_cache.get()
        g1 = expr.eval_grid(xs, ys)
        g2 = expr.eval_grid(xs, ys)
        assert len(cache._d) == 0
    np.testing.assert_array_equal(g1, g2)


def test_grid_cache_lru_byte_cap():
    cap = 64  # exactly two 4-element float64 grids (32 bytes each)
    cache = _GridCache(max_bytes=cap)

    def key(tag):
        return ("expr", tag)

    a, b, c = (np.full(4, v, dtype=np.float64) for v in (0.0, 1.0, 2.0))
    cache.put(key("k1"), a, ())
    cache.put(key("k2"), b, ())
    assert cache._bytes == 64
    assert cache.get(key("k1")) is a  # hit refreshes k1 as most recently used
    cache.put(key("k3"), c, ())  # over cap -> evict the LRU (k2), keep k1
    assert cache._bytes == 64
    assert cache.get(key("k2")) is None
    assert cache.get(key("k1")) is a
    assert cache.get(key("k3")) is c
    assert cache.get("unknown") is None
    # An entry larger than the cap is never inserted (and evicts nothing).
    cache.put(key("big"), np.zeros(64), ())
    assert cache.get(key("big")) is None
    assert cache._bytes == 64
    # Under sustained pressure the cache stays at/below the cap and keeps the
    # most recent entries.
    pressure = _GridCache(max_bytes=cap)
    for j in range(20):
        pressure.put(("k", j), np.full(4, float(j)), ())
        assert pressure._bytes <= cap
    assert pressure.get(("k", 19)) is not None
    assert pressure.get(("k", 0)) is None


def test_grid_cache_pins_key_objects_until_scope_exit():
    # The delegate expression and the axes its key was derived from are pinned
    # inside the entry: they survive gc while the scope is active (so a freed
    # id can never be recycled into a false hit) and are released after.
    xs = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    expr = fBm(octaves=2, base_freq=4, to_01=True, seed=3)
    expr_ref, xs_ref = weakref.ref(expr), weakref.ref(xs)
    with grid_cache_scope():
        Ref("a", expr).eval_grid(xs, ys)
        cache = _grid_cache.get()
        del expr, xs, ys
        gc.collect()
        assert expr_ref() is not None
        assert xs_ref() is not None
    del cache
    gc.collect()
    assert expr_ref() is None
    assert xs_ref() is None


def test_grid_cache_nested_scopes():
    # A nested scope gets a fresh independent cache; exiting it restores the
    # outer one with its entries intact.
    xs = np.linspace(0.0, 6.0, 8, dtype=np.float32)
    ys = np.linspace(0.0, 6.0, 8, dtype=np.float32)
    shared = fBm(octaves=3, base_freq=5, to_01=True, seed=6)
    with grid_cache_scope():
        outer = _grid_cache.get()
        g1 = Ref("a", shared).eval_grid(xs, ys)
        with grid_cache_scope():
            inner = _grid_cache.get()
            assert inner is not outer
            g2 = Ref("b", shared).eval_grid(xs, ys)
            assert g2 is not g1  # fresh cache: re-evaluated, not shared
            assert len(inner._d) == 1
            assert len(outer._d) == 1  # the inner put did not touch the outer
        assert _grid_cache.get() is outer
        g3 = Ref("c", shared).eval_grid(xs, ys)
        assert g3 is g1  # outer cache still hits
    assert _grid_cache.get() is None


def test_grid_cache_scope_survives_mid_render_exception(monkeypatch):
    # An eval_grid raising mid-render must not poison the contextvar: the
    # scope's finally resets it, and the next render is unaffected.
    material = LayeredMaterial.deserialize(_TONE_GRAIN_PROGRAM)
    original = fBm.eval_grid
    reference = fixed_light_preview(material, 64, 64, memo=False)
    calls = []

    def flaky(self, xs, ys, coords=None):
        calls.append(1)
        if len(calls) == 2:  # grain evaluates after tone is already cached
            raise RuntimeError("mid-render failure")
        return original(self, xs, ys, coords)

    monkeypatch.setattr(fBm, "eval_grid", flaky)
    with pytest.raises(RuntimeError, match="mid-render failure"):
        fixed_light_preview(material, 64, 64)
    assert _grid_cache.get() is None  # the scope exited cleanly
    monkeypatch.setattr(fBm, "eval_grid", original)
    img = fixed_light_preview(material, 64, 64)
    np.testing.assert_array_equal(img, reference)
