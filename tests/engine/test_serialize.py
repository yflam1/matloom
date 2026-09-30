"""Tests for matloom.engine.serialize: Expression2D tree -> canonical DSL."""

from __future__ import annotations

import pytest

from matloom.engine.expr import (
    Abs,
    Ceil,
    Constant,
    Cos,
    Floor,
    Log,
    Max,
    Min,
    Sin,
    Sqrt,
    X,
    Y,
)
from matloom.engine.noise import Worley, fBm
from matloom.engine.parser import parse_expr
from matloom.engine.pattern import Bricks
from matloom.engine.serialize import serialize_expr
from matloom.engine.util import Threshold

# |----------------|
# |   Primitives   |
# |----------------|


def test_serialize_integer_constant():
    assert serialize_expr(Constant(5.0)) == "5"


def test_serialize_float_constant():
    assert serialize_expr(Constant(0.25)) == "0.25"


def test_serialize_x_y():
    assert serialize_expr(X()) == "X()"
    assert serialize_expr(Y()) == "Y()"


def test_serialize_negative_via_unary_minus():
    # -X() parses to (-1 * X()); serializer renders it back as -X().
    assert serialize_expr(parse_expr("-X()")) == "-X()"


def test_serialize_binary_ops():
    assert serialize_expr(X() + Y()) == "(X() + Y())"
    assert serialize_expr(X() - Y()) == "(X() - Y())"
    assert serialize_expr(X() * Y()) == "(X() * Y())"
    assert serialize_expr(X() / Y()) == "(X() / Y())"
    assert serialize_expr(X() ** Y()) == "(X() ** Y())"


def test_serialize_unary_functions():
    assert serialize_expr(Sin(X())) == "Sin(X())"
    assert serialize_expr(Cos(X())) == "Cos(X())"
    assert serialize_expr(Abs(X())) == "Abs(X())"
    assert serialize_expr(Sqrt(X())) == "Sqrt(X())"
    assert serialize_expr(Floor(X())) == "Floor(X())"
    assert serialize_expr(Ceil(X())) == "Ceil(X())"


def test_serialize_min_max():
    assert serialize_expr(Min(X(), Y())) == "Min(X(), Y())"
    assert serialize_expr(Max(X(), Y())) == "Max(X(), Y())"


def test_serialize_log_natural_omits_base():
    assert serialize_expr(Log(X())) == "Log(X())"


def test_serialize_log_with_base():
    assert serialize_expr(Log(X(), 10)) == "Log(X(), 10)"


# |----------------|
# |   Threshold    |
# |----------------|


def test_serialize_threshold_minimal():
    assert (
        serialize_expr(Threshold(X(), below_at=0.2)) == "Threshold(X(), below_at=0.2)"
    )


def test_serialize_threshold_full():
    t = Threshold(
        X(),
        below_at=0.1,
        below_to=0.0,
        above_at=0.9,
        above_to=1.0,
        transition_width=0.05,
    )
    s = serialize_expr(t)
    assert s.startswith("Threshold(X()")
    assert "below_at=0.1" in s
    assert "below_to=0" in s
    assert "above_at=0.9" in s
    assert "above_to=1" in s
    assert "transition_width=0.05" in s


def test_serialize_threshold_omits_redundant_to():
    # below_to == below_at, so below_to is not emitted.
    assert (
        serialize_expr(Threshold(X(), below_at=0.3)) == "Threshold(X(), below_at=0.3)"
    )


# |---------------|
# |   Noise       |
# |---------------|


def test_serialize_fbm_bakes_seed():
    s = serialize_expr(fBm(seed=7))
    assert s == "fBm(seed=7)"


def test_serialize_fbm_omits_defaults():
    # octaves=6, lacunarity=2, gain=0.5, base_freq=1 are all defaults.
    assert serialize_expr(fBm(seed=1)) == "fBm(seed=1)"


def test_serialize_fbm_includes_non_defaults():
    s = serialize_expr(fBm(octaves=4, base_freq=2.0, to_01=True, seed=9))
    assert "octaves=4" in s
    assert "base_freq=2" in s
    assert "to_01=True" in s
    assert "seed=9" in s


def test_serialize_fbm_separate_frequencies():
    s = serialize_expr(fBm(base_freq_x=2.0, base_freq_y=3.0, seed=1))
    assert "base_freq_x=2" in s
    assert "base_freq_y=3" in s


def test_serialize_worley_defaults():
    assert serialize_expr(Worley(seed=2)) == "Worley(seed=2)"


def test_serialize_worley_non_defaults():
    s = serialize_expr(Worley(distance="manhattan", combination="F2", seed=4))
    assert 'distance="manhattan"' in s
    assert 'combination="F2"' in s


# |---------------------|
# |   Noise set-tracking (parse path)   |
# |---------------------|
# A parsed noise node carries the arg names the author wrote (set_args), so the
# serializer emits only those, preserving explicit defaults like `octaves=6` or
# `to_01=False` that value-comparison would drop. Code-constructed noise (no
# set_args) keeps the value-comparison behavior covered by the tests above.


def test_serialize_fbm_keeps_explicit_default_octaves():
    # octaves=6 is the default, but the author wrote it explicitly -> kept.
    assert serialize_expr(parse_expr("fBm(octaves=6, base_freq=2, seed=9)")) == (
        "fBm(octaves=6, base_freq=2, seed=9)"
    )


def test_serialize_fbm_keeps_explicit_default_to_01_false():
    assert serialize_expr(parse_expr("fBm(base_freq=2, to_01=False, seed=9)")) == (
        "fBm(base_freq=2, to_01=False, seed=9)"
    )


def test_serialize_fbm_omits_unset_defaults():
    # Nothing beyond base_freq+seed was authored -> octaves/lacunarity/gain/to_01 omitted.
    assert (
        serialize_expr(parse_expr("fBm(base_freq=2, seed=9)"))
        == "fBm(base_freq=2, seed=9)"
    )


def test_serialize_fbm_one_axis_freq():
    # Only base_freq_x authored; base_freq_y defaults to base_freq(1.0) and is omitted.
    assert serialize_expr(parse_expr("fBm(base_freq_x=2, seed=9)")) == (
        "fBm(base_freq_x=2, seed=9)"
    )


def test_serialize_worley_keeps_explicit_defaults():
    assert (
        serialize_expr(
            parse_expr('Worley(distance="euclidean", combination="F1", seed=4)')
        )
        == 'Worley(distance="euclidean", combination="F1", seed=4)'
    )


def test_serialize_fbm_explicit_defaults_roundtrip_stable():
    once = serialize_expr(
        parse_expr("fBm(octaves=6, base_freq=2, to_01=False, seed=9)")
    )
    twice = serialize_expr(parse_expr(once))
    assert once == twice


# |---------------|
# |   Bricks      |
# |---------------|


def test_serialize_bricks_defaults():
    assert serialize_expr(Bricks()) == "Bricks()"


def test_serialize_bricks_non_defaults():
    s = serialize_expr(
        Bricks(brick_width=2.0, offset=0.25, axis="column", feather=0.02)
    )
    assert "brick_width=2" in s
    assert "offset=0.25" in s
    assert 'axis="column"' in s
    assert "feather=0.02" in s
    # Defaults are omitted.
    assert "brick_height" not in s
    assert "mortar" not in s


def test_serialize_ref_emits_bare_name():
    from matloom.engine.expr import Ref

    expr = parse_expr("Fill(Rect(0.2, 0.2, 0.6, 0.6))")
    assert serialize_expr(Ref("star", expr)) == "star"
    # A reference inside a transform serializes to the name, not the inlined expr.
    s = serialize_expr(parse_expr("Rotate(star, 30)", {"star": expr}))
    assert s == "Rotate(star, 30)"


def test_serialize_unsupported_raises():
    class Bogus:
        pass

    with pytest.raises(TypeError, match="cannot serialize"):
        serialize_expr(Bogus())  # type: ignore[arg-type]


# |-----------------------|
# |   Round-trip parity   |
# |-----------------------|


@pytest.mark.parametrize(
    "src",
    [
        "5",
        "0.25",
        "X()",
        "Y()",
        "(X() + Y())",
        "(X() * 2)",
        "Sin(X())",
        "Min(X(), Y())",
        "Log(X(), 10)",
        "Threshold(X(), below_at=0.2, above_at=0.8)",
        "fBm(octaves=4, seed=9)",
        "fBm(octaves=6, base_freq=2, seed=9)",
        "fBm(base_freq_x=2, base_freq_y=3, seed=9)",
        'Worley(distance="euclidean", combination="F1", seed=4)',
        'Worley(distance="manhattan", seed=4)',
        "Bricks()",
        'Bricks(brick_width=2, offset=0.25, axis="column", feather=0.02)',
        "Translate(X(), 0.3, 0)",
        "Scale(X(), 2, 0.5)",
        "Rotate(X(), 45)",
        "Fill(Rect(0.2, 0.2, 0.6, 0.6))",
        "Fill(Rect(0, 0, 1, 1, 0.1), feather=0.02)",
        'Fill(Ellipse(0.5, 0.5, 0.3, 0.2), mode="EvenOdd")',
        "Stroke(Rect(0.2, 0.2, 0.6, 0.6), 0.04, feather=0.01)",
        "Fill(Path(0.2, 0.2, LineTo(0.8, 0.2), LineTo(0.5, 0.8)))",
        "Stroke(Path(0.1, 0.5, CubicTo(0.3, 0.9, 0.7, 0.1, 0.9, 0.5)), 0.05)",
    ],
)
def test_parse_serialize_roundtrip_is_stable(src):
    # Serializing a parsed tree, then re-parsing and re-serializing, must be a
    # fixed point (canonical form is idempotent).
    once = serialize_expr(parse_expr(src))
    twice = serialize_expr(parse_expr(once))
    assert once == twice
