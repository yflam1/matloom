"""
serialize.py — turn an :class:`Expression2D` tree back into canonical DSL
source. The Python counterpart of `frontend/src/expr-lang/serialize.ts`.

The output is parseable by :func:`matloom.engine.parser.parse_expr` (and by the
frontend), so materials round-trip through the textual format. As in the
frontend, `Constant`s become bare numbers and noise nodes have their seed
baked in (`fBm(..., seed=N)`) so re-import reproduces the exact same noise.
"""

from matloom.engine.const import e
from matloom.engine.expr import (
    Abs,
    AdditionExpression2D,
    Ceil,
    Constant,
    Cos,
    DivisionExpression2D,
    Expression2D,
    Floor,
    Log,
    Max,
    Min,
    MultiplicationExpression2D,
    PowerExpression2D,
    Ref,
    Sin,
    Sqrt,
    SubtractionExpression2D,
    X,
    Y,
)
from matloom.engine.noise import Worley, fBm
from matloom.engine.pattern import Bricks, Weave
from matloom.engine.shape import Ellipse, Fill, Path, Rect, Shape, Stroke
from matloom.engine.transform import Rotate, Scale, Translate
from matloom.engine.util import Linear2sRGB, Threshold, sRGB2Linear


def _fmt_num(v: float) -> str:
    """Match the frontend's String(v): integral values stay integral."""
    f = float(v)
    if f.is_integer():
        return str(int(f))
    return repr(f)


def _freq_parts(s: frozenset[str] | None, bfx: float, bfy: float) -> list[str]:
    """Emit the frequency args. ``s`` is the set of explicitly-authored arg
    names (``base_freq``/``base_freq_x``/``base_freq_y``); ``None`` falls back
    to value-comparison (the pre-set-tracking behavior, used when a noise node
    was constructed in code without ``set_args``)."""
    if s is None:
        if bfx == bfy:
            return [] if bfx == 1.0 else [f"base_freq={_fmt_num(bfx)}"]
        return [f"base_freq_x={_fmt_num(bfx)}", f"base_freq_y={_fmt_num(bfy)}"]
    parts: list[str] = []
    if "base_freq" in s:
        # `base_freq` sets both axes' default; emit it from whichever axis was
        # NOT overridden by an explicit base_freq_x/y (that axis still equals
        # base_freq). When both axes are explicit, base_freq has no effect.
        if "base_freq_x" not in s:
            parts.append(f"base_freq={_fmt_num(bfx)}")
        elif "base_freq_y" not in s:
            parts.append(f"base_freq={_fmt_num(bfy)}")
    if "base_freq_x" in s:
        parts.append(f"base_freq_x={_fmt_num(bfx)}")
    if "base_freq_y" in s:
        parts.append(f"base_freq_y={_fmt_num(bfy)}")
    return parts


def _opt(s: frozenset[str] | None, name: str, value: object, default: object) -> bool:
    """Whether to emit a noise scalar arg: set-tracked when ``s`` is given,
    else value-comparison against ``default`` (pre-set-tracking behavior)."""
    if s is not None:
        return name in s
    return value != default


def _serialize_shape(shape: Shape) -> str:
    """Serialize a :class:`Shape` (the geometry inside Fill/Stroke)."""
    if isinstance(shape, Rect):
        parts = [
            _fmt_num(shape._x),
            _fmt_num(shape._y),
            _fmt_num(shape._width),
            _fmt_num(shape._height),
        ]
        if shape._radius != 0.0:
            parts.append(_fmt_num(shape._radius))
        return f"Rect({', '.join(parts)})"
    if isinstance(shape, Ellipse):
        return (
            f"Ellipse({_fmt_num(shape._cx)}, {_fmt_num(shape._cy)}, "
            f"{_fmt_num(shape._rx)}, {_fmt_num(shape._ry)})"
        )
    if isinstance(shape, Path):
        parts = [_fmt_num(shape._start[0]), _fmt_num(shape._start[1])]
        for seg in shape._segments:
            nums = ", ".join(_fmt_num(v) for v in seg.points)
            parts.append(f"{'LineTo' if seg.kind == 'line' else 'CubicTo'}({nums})")
        return f"Path({', '.join(parts)})"
    raise TypeError(f"cannot serialize shape: {type(shape).__name__}")


def serialize_expr(expr: Expression2D) -> str:
    """Serialize an :class:`Expression2D` tree to canonical DSL source."""
    if isinstance(expr, Ref):
        return expr._name
    if isinstance(expr, Constant):
        return _fmt_num(expr._value)
    if isinstance(expr, X):
        return "X()"
    if isinstance(expr, Y):
        return "Y()"

    # Unary minus is parsed as `(-1 * operand)`; render it back as `-operand`.
    if isinstance(expr, MultiplicationExpression2D):
        left, right = expr._left, expr._right
        if isinstance(left, Constant) and left._value == -1:
            inner = serialize_expr(right)
            return f"-({inner})" if inner.startswith("-") else f"-{inner}"
        return f"({serialize_expr(left)} * {serialize_expr(right)})"
    if isinstance(expr, AdditionExpression2D):
        return f"({serialize_expr(expr._left)} + {serialize_expr(expr._right)})"
    if isinstance(expr, SubtractionExpression2D):
        return f"({serialize_expr(expr._minuend)} - {serialize_expr(expr._subtrahend)})"
    if isinstance(expr, DivisionExpression2D):
        return f"({serialize_expr(expr._dividend)} / {serialize_expr(expr._divisor)})"
    if isinstance(expr, PowerExpression2D):
        # The parser binds `**` tighter than a leading unary minus, so a base
        # that renders with a leading '-' (a negative Constant, or the `-x`
        # unary-minus form) would re-parse as `-(base ** exp)` — a different
        # value. Parenthesize it so `(-3 ** e)` round-trips as `((-3) ** e)`.
        base = serialize_expr(expr._base)
        if base.startswith("-"):
            base = f"({base})"
        return f"({base} ** {serialize_expr(expr._exponent)})"
    if isinstance(expr, Min):
        return f"Min({serialize_expr(expr._expr1)}, {serialize_expr(expr._expr2)})"
    if isinstance(expr, Max):
        return f"Max({serialize_expr(expr._expr1)}, {serialize_expr(expr._expr2)})"

    if isinstance(expr, Sin):
        return f"Sin({serialize_expr(expr._expr)})"
    if isinstance(expr, Cos):
        return f"Cos({serialize_expr(expr._expr)})"
    if isinstance(expr, Abs):
        return f"Abs({serialize_expr(expr._expr)})"
    if isinstance(expr, Sqrt):
        return f"Sqrt({serialize_expr(expr._expr)})"
    if isinstance(expr, Floor):
        return f"Floor({serialize_expr(expr._expr)})"
    if isinstance(expr, Ceil):
        return f"Ceil({serialize_expr(expr._expr)})"
    if isinstance(expr, sRGB2Linear):
        return f"sRGB2Linear({serialize_expr(expr._expr)})"
    if isinstance(expr, Linear2sRGB):
        return f"Linear2sRGB({serialize_expr(expr._expr)})"
    if isinstance(expr, Log):
        arg = serialize_expr(expr._expr)
        return (
            f"Log({arg})" if expr._base == e else f"Log({arg}, {_fmt_num(expr._base)})"
        )

    if isinstance(expr, Threshold):
        parts = [serialize_expr(expr._expr)]
        if expr._below_at is not None:
            parts.append(f"below_at={_fmt_num(expr._below_at)}")
            if expr._below_to is not None and expr._below_to != expr._below_at:
                parts.append(f"below_to={_fmt_num(expr._below_to)}")
        if expr._above_at is not None:
            parts.append(f"above_at={_fmt_num(expr._above_at)}")
            if expr._above_to is not None and expr._above_to != expr._above_at:
                parts.append(f"above_to={_fmt_num(expr._above_to)}")
        if expr._transition_width != 0:
            parts.append(f"transition_width={_fmt_num(expr._transition_width)}")
        return f"Threshold({', '.join(parts)})"

    if isinstance(expr, fBm):
        s = expr._set_args
        parts: list[str] = []
        if _opt(s, "octaves", expr._octaves, 6):
            parts.append(f"octaves={_fmt_num(expr._octaves)}")
        if _opt(s, "lacunarity", expr._lacunarity, 2.0):
            parts.append(f"lacunarity={_fmt_num(expr._lacunarity)}")
        if _opt(s, "gain", expr._gain, 0.5):
            parts.append(f"gain={_fmt_num(expr._gain)}")
        parts.extend(_freq_parts(s, expr._base_freq_x, expr._base_freq_y))
        if _opt(s, "to_01", expr._to_01, False):
            parts.append(f"to_01={'True' if expr._to_01 else 'False'}")
        parts.append(f"seed={_fmt_num(expr._seed)}")
        return f"fBm({', '.join(parts)})"

    if isinstance(expr, Worley):
        s = expr._set_args
        parts = []
        if _opt(s, "distance", expr._distance, "euclidean"):
            parts.append(f'distance="{expr._distance}"')
        if _opt(s, "combination", expr._combination, "F1"):
            parts.append(f'combination="{expr._combination}"')
        parts.extend(_freq_parts(s, expr._base_freq_x, expr._base_freq_y))
        if _opt(s, "to_01", expr._to_01, False):
            parts.append(f"to_01={'True' if expr._to_01 else 'False'}")
        parts.append(f"seed={_fmt_num(expr._seed)}")
        return f"Worley({', '.join(parts)})"

    if isinstance(expr, Translate):
        child = serialize_expr(expr._expr)
        return f"Translate({child}, {_fmt_num(expr._dx)}, {_fmt_num(expr._dy)})"

    if isinstance(expr, Scale):
        child = serialize_expr(expr._expr)
        return f"Scale({child}, {_fmt_num(expr._sx)}, {_fmt_num(expr._sy)})"

    if isinstance(expr, Rotate):
        child = serialize_expr(expr._expr)
        return f"Rotate({child}, {_fmt_num(expr._degrees)})"

    if isinstance(expr, Bricks):
        parts = []
        if expr._brick_width != 1.0:
            parts.append(f"brick_width={_fmt_num(expr._brick_width)}")
        if expr._brick_height != 0.5:
            parts.append(f"brick_height={_fmt_num(expr._brick_height)}")
        if expr._offset != 0.5:
            parts.append(f"offset={_fmt_num(expr._offset)}")
        if expr._mortar != 0.05:
            parts.append(f"mortar={_fmt_num(expr._mortar)}")
        if expr._axis != "row":
            parts.append(f'axis="{expr._axis}"')
        if expr._feather != 0.0:
            parts.append(f"feather={_fmt_num(expr._feather)}")
        return f"Bricks({', '.join(parts)})"

    if isinstance(expr, Weave):
        parts = []
        if expr._over != 1:
            parts.append(f"over={_fmt_num(expr._over)}")
        if expr._under != 1:
            parts.append(f"under={_fmt_num(expr._under)}")
        if expr._shift != 1:
            parts.append(f"shift={_fmt_num(expr._shift)}")
        bfx, bfy = expr._base_freq_x, expr._base_freq_y
        if bfx == bfy:
            if bfx != 8.0:
                parts.append(f"base_freq={_fmt_num(bfx)}")
        else:
            parts.append(f"base_freq_x={_fmt_num(bfx)}")
            parts.append(f"base_freq_y={_fmt_num(bfy)}")
        if expr._warp_width != 0.5:
            parts.append(f"warp_width={_fmt_num(expr._warp_width)}")
        if expr._feather != 0.0:
            parts.append(f"feather={_fmt_num(expr._feather)}")
        return f"Weave({', '.join(parts)})"

    if isinstance(expr, Fill):
        parts = [_serialize_shape(expr._shape)]
        if expr._mode != "NonZero":
            parts.append(f'mode="{expr._mode}"')
        if expr._feather != 0.0:
            parts.append(f"feather={_fmt_num(expr._feather)}")
        return f"Fill({', '.join(parts)})"

    if isinstance(expr, Stroke):
        parts = [_serialize_shape(expr._shape), _fmt_num(expr._width)]
        if expr._feather != 0.0:
            parts.append(f"feather={_fmt_num(expr._feather)}")
        return f"Stroke({', '.join(parts)})"

    raise TypeError(f"cannot serialize expression: {type(expr).__name__}")
