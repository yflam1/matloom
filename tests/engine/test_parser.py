"""Tests for matloom.engine.parser: parsing the channel expression DSL."""

from __future__ import annotations

import math

import pytest

from matloom.engine.expr import (
    AdditionExpression2D,
    Constant,
    DivisionExpression2D,
    MultiplicationExpression2D,
    PowerExpression2D,
    SubtractionExpression2D,
)
from matloom.engine.noise import Worley, fBm
from matloom.engine.parser import parse_expr
from matloom.engine.pattern import Bricks
from matloom.engine.util import Threshold

# |-------------|
# |   Literals  |
# |-------------|


def test_parse_integer():
    e = parse_expr("42")
    assert isinstance(e, Constant)
    assert e() == 42


def test_parse_float():
    assert parse_expr("3.5")() == 3.5


def test_parse_scientific_notation():
    assert parse_expr("1e3")() == 1000.0
    assert parse_expr("1.5e-2")() == pytest.approx(0.015)


def test_parse_constants_pi_e():
    assert parse_expr("pi")() == pytest.approx(math.pi)
    assert parse_expr("e")() == pytest.approx(math.e)


def test_empty_string_raises():
    # The parser itself does not special-case empty input; callers (e.g. the
    # editor) decide what an empty channel means before parsing.
    with pytest.raises(ValueError, match="Expected expression"):
        parse_expr("")


# |---------------|
# |   Operators   |
# |---------------|


def test_addition():
    e = parse_expr("1 + 2")
    assert isinstance(e, AdditionExpression2D)
    assert e() == 3


def test_operator_precedence():
    # 1 + 2 * 3 == 7, not 9
    assert parse_expr("1 + 2 * 3")() == 7


def test_parentheses_override_precedence():
    assert parse_expr("(1 + 2) * 3")() == 9


def test_power_is_right_associative():
    # 2 ** 3 ** 2 == 2 ** 9 == 512
    assert parse_expr("2 ** 3 ** 2")() == 512


def test_unary_minus():
    assert parse_expr("-5")() == -5
    assert parse_expr("-X()")(0.3) == pytest.approx(-0.3)


def test_unary_plus():
    assert parse_expr("+5")() == 5


def test_division():
    e = parse_expr("6 / 2")
    assert isinstance(e, DivisionExpression2D)
    assert e() == 3


def test_subtraction_and_multiplication_types():
    assert isinstance(parse_expr("5 - 2"), SubtractionExpression2D)
    assert isinstance(parse_expr("5 * 2"), MultiplicationExpression2D)
    assert isinstance(parse_expr("5 ** 2"), PowerExpression2D)


def test_coordinate_expression():
    e = parse_expr("X() * 2 + 1")
    assert e(0.5) == pytest.approx(2.0)


# |---------------|
# |   Functions   |
# |---------------|


def test_parse_fbm_keyword_args():
    e = parse_expr("fBm(base_freq=2, seed=5)")
    assert isinstance(e, fBm)
    assert e._base_freq_x == 2.0
    assert e._seed == 5


def test_parse_fbm_positional_args():
    # positional order: octaves, lacunarity, gain, base_freq, ...
    e = parse_expr("fBm(3, 2.0, 0.5, seed=1)")
    assert e._octaves == 3


def test_parse_worley_string_arg():
    e = parse_expr('Worley(combination="F2-F1", seed=3)')
    assert isinstance(e, Worley)
    assert e._combination == "F2-F1"


def test_parse_threshold():
    e = parse_expr("Threshold(X(), below_at=0.2, above_at=0.8)")
    assert isinstance(e, Threshold)
    assert e._below_at == 0.2
    assert e._above_at == 0.8


def test_nested_function_calls():
    e = parse_expr("Sin(Mul(X(), 2))")
    assert e(math.pi / 4) == pytest.approx(math.sin(math.pi / 2))


def test_functional_form_mul():
    e = parse_expr("Mul(X(), 2)")
    assert e(3.0) == 6.0


def test_parse_bricks_keyword_and_positional():
    e = parse_expr('Bricks(0.4, 0.2, offset=0.25, axis="column")')
    assert isinstance(e, Bricks)
    assert e._brick_width == 0.4
    assert e._brick_height == 0.2
    assert e._offset == 0.25
    assert e._axis == "column"


def test_parse_path_collects_trailing_segments():
    from matloom.engine.shape import Fill, Path

    e = parse_expr("Fill(Path(0, 0, LineTo(1, 0), LineTo(1, 1)))")
    assert isinstance(e, Fill)
    assert isinstance(e._shape, Path)
    assert len(e._shape._segments) == 2


def test_bare_shape_must_be_wrapped():
    with pytest.raises(TypeError, match="Fill"):
        parse_expr("Rect(0, 0, 1, 1)")


def test_bare_segment_must_be_in_path():
    with pytest.raises(TypeError, match="Path"):
        parse_expr("LineTo(1, 2)")


def test_env_resolves_name_to_ref():
    from matloom.engine.expr import Ref

    env = {"box": parse_expr("Fill(Rect(0.2, 0.2, 0.6, 0.6))")}
    e = parse_expr("Translate(box, 0.1, 0.0)", env)
    ref = parse_expr("box", env)
    assert isinstance(ref, Ref)
    # The reference delegates to the stored expression.
    assert ref(0.5, 0.5) == env["box"](0.5, 0.5)
    # A reference is usable anywhere an expression is.
    assert e(0.45, 0.5) == env["box"](0.35, 0.5)


def test_unknown_name_without_env_still_raises():
    with pytest.raises(ValueError, match="Unknown name"):
        parse_expr("box")


# |--------------|
# |   Errors     |
# |--------------|


def test_unknown_function_raises():
    with pytest.raises(ValueError, match="Unknown function"):
        parse_expr("Bogus(1)")


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown name"):
        parse_expr("foobar")


def test_function_referenced_without_call_raises():
    with pytest.raises(ValueError, match="call it"):
        parse_expr("fBm")


def test_too_many_positional_args_raises():
    with pytest.raises(ValueError, match="Too many positional"):
        parse_expr("X(1)")


def test_unknown_parameter_raises():
    with pytest.raises(ValueError, match="Unknown parameter"):
        parse_expr("fBm(bogus=1)")


def test_duplicate_parameter_raises():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_expr("fBm(seed=1, seed=2)")


def test_trailing_comma_raises():
    with pytest.raises(ValueError, match="Trailing comma"):
        parse_expr("Min(X(), )")


def test_unterminated_string_raises():
    with pytest.raises(ValueError, match="Unterminated string"):
        parse_expr('Worley(combination="F1)')


def test_unexpected_character_raises():
    with pytest.raises(ValueError, match="Unexpected character"):
        parse_expr("X() @ 2")


def test_trailing_input_raises():
    with pytest.raises(ValueError, match="trailing"):
        parse_expr("1 2")


def test_bare_bool_literal_rejected():
    with pytest.raises(TypeError):
        parse_expr("True")


def test_whitespace_is_ignored():
    assert parse_expr("  1   +    2  ")() == 3
