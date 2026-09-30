"""
parser.py — parse the channel expression DSL into an :class:`Expression2D`.

This is the Python counterpart of `frontend/src/expr-lang/parser.ts` and
accepts the same grammar:

    fBm(base_freq=2, seed=5)        Worley(combination="F2-F1")
    Sin(X())                        Mul(X(), 2)              X()*2 + 1

Positional and keyword arguments are both supported (positional binding follows
the same parameter order the frontend registry uses, so the two engines accept
exactly the same strings). `parse_expr` returns an :class:`Expression2D`;
bare numbers become :class:`Constant`.
"""

from collections.abc import Callable, Mapping
from typing import Any

from matloom.engine.const import e, pi
from matloom.engine.expr import (
    Abs,
    Add,
    Ceil,
    Constant,
    Cos,
    Div,
    Expression2D,
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
)
from matloom.engine.noise import Worley, fBm
from matloom.engine.pattern import Bricks, Weave
from matloom.engine.shape import (
    CubicTo,
    Ellipse,
    Fill,
    LineTo,
    Path,
    Rect,
    Segment,
    Shape,
    Stroke,
)
from matloom.engine.transform import Rotate, Scale, Translate
from matloom.engine.util import Threshold

# name -> (callable, ordered parameter names). The parameter order matches the
# frontend registry so positional arguments bind identically on both sides.
_REGISTRY: dict[str, tuple[Callable[..., Any], list[str]]] = {
    "fBm": (
        fBm,
        [
            "octaves",
            "lacunarity",
            "gain",
            "base_freq",
            "base_freq_x",
            "base_freq_y",
            "to_01",
            "seed",
        ],
    ),
    "Worley": (
        Worley,
        [
            "distance",
            "combination",
            "base_freq",
            "base_freq_x",
            "base_freq_y",
            "to_01",
            "seed",
        ],
    ),
    "Threshold": (
        Threshold,
        [
            "expr",
            "below_at",
            "below_to",
            "above_at",
            "above_to",
            "transition_width",
        ],
    ),
    "Bricks": (
        Bricks,
        ["brick_width", "brick_height", "offset", "mortar", "axis", "feather"],
    ),
    "Weave": (
        Weave,
        [
            "over",
            "under",
            "shift",
            "base_freq",
            "base_freq_x",
            "base_freq_y",
            "warp_width",
            "feather",
        ],
    ),
    "Translate": (Translate, ["expr", "dx", "dy"]),
    "Scale": (Scale, ["expr", "sx", "sy"]),
    "Rotate": (Rotate, ["expr", "degrees"]),
    "Rect": (Rect, ["x", "y", "width", "height", "radius"]),
    "Ellipse": (Ellipse, ["cx", "cy", "rx", "ry"]),
    "Fill": (Fill, ["shape", "mode", "feather"]),
    "Stroke": (Stroke, ["shape", "width", "feather"]),
    "Path": (Path, ["start_x", "start_y"]),
    "LineTo": (LineTo, ["x", "y"]),
    "CubicTo": (CubicTo, ["c1x", "c1y", "c2x", "c2y", "x", "y"]),
    "Constant": (Constant, ["value"]),
    "X": (X, []),
    "Y": (Y, []),
    "Abs": (Abs, ["expr"]),
    "Sqrt": (Sqrt, ["expr"]),
    "Floor": (Floor, ["expr"]),
    "Ceil": (Ceil, ["expr"]),
    "Log": (Log, ["expr", "base"]),
    "Sin": (Sin, ["expr"]),
    "Cos": (Cos, ["expr"]),
    "Min": (Min, ["expr1", "expr2"]),
    "Max": (Max, ["expr1", "expr2"]),
    "Add": (Add, ["left", "right"]),
    "Sub": (Sub, ["minuend", "subtrahend"]),
    "Mul": (Mul, ["left", "right"]),
    "Div": (Div, ["dividend", "divisor"]),
    "Pow": (Pow, ["base", "exponent"]),
}

# Functions that track which args the author explicitly wrote (so the
# serializer can emit only those, preserving explicit defaults like
# `octaves=6`). The parser threads `set_args=frozenset(<appeared arg names>)`
# into their constructors.
_TRACKS_SET: frozenset[str] = frozenset({"fBm", "Worley"})

# Functions taking a variable number of trailing positional arguments after
# their fixed params; the extras are collected into a list bound to this name.
_REST_PARAMS: dict[str, str] = {"Path": "segments"}

_CONSTS: dict[str, float] = {"pi": pi, "e": e}

_LITERALS: dict[str, Any] = {
    "True": True,
    "False": False,
    "None": None,
    "true": True,
    "false": False,
    "null": None,
}

_OPERATOR_CHARS = "+-*/()=,"


# |---------------|
# |   Tokenizer   |
# |---------------|


class _Token:
    __slots__ = ("type", "value")

    def __init__(self, type_: str, value: Any) -> None:
        self.type = type_
        self.value = value


def _tokenize(src: str) -> list[_Token]:
    tokens: list[_Token] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue

        # String literal (single or double quote).
        if c in "'\"":
            j = i + 1
            while j < n and src[j] != c:
                j += 1
            if j >= n:
                raise ValueError("Unterminated string literal")
            tokens.append(_Token("STRING", src[i + 1 : j]))
            i = j + 1
            continue

        # Number (integers stay int so pydantic int fields validate cleanly).
        if c.isdigit() or c == ".":
            j = i
            is_float = False
            while j < n and src[j].isdigit():
                j += 1
            if j < n and src[j] == ".":
                is_float = True
                j += 1
                while j < n and src[j].isdigit():
                    j += 1
            if j < n and src[j] in "eE":
                is_float = True
                j += 1
                if j < n and src[j] in "+-":
                    j += 1
                while j < n and src[j].isdigit():
                    j += 1
            text = src[i:j]
            tokens.append(_Token("NUMBER", float(text) if is_float else int(text)))
            i = j
            continue

        # Identifier / keyword.
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            tokens.append(_Token("IDENT", src[i:j]))
            i = j
            continue

        # Operators / punctuation (match ** before *).
        if c == "*" and i + 1 < n and src[i + 1] == "*":
            tokens.append(_Token("OP", "**"))
            i += 2
        elif c in _OPERATOR_CHARS:
            tokens.append(_Token("OP", c))
            i += 1
        else:
            raise ValueError(f"Unexpected character {c!r}")
    return tokens


# |-------------------------------|
# |   Recursive-descent parser    |
# |-------------------------------|


class _Parser:
    def __init__(
        self,
        tokens: list[_Token],
        env: Mapping[str, Expression2D] | None = None,
    ) -> None:
        self._tokens = tokens
        self._pos = 0
        self._env = env or {}

    def _peek(self) -> _Token | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> _Token | None:
        tok = self._peek()
        if tok is not None:
            self._pos += 1
        return tok

    def _expect(self, op: str) -> None:
        tok = self._advance()
        if tok is None or tok.value != op:
            found = tok.value if tok else "end of input"
            raise ValueError(f"Expected {op!r} but found {found!r}")

    def parse(self) -> Any:
        result = self._expr()
        if self._pos < len(self._tokens):
            rest = " ".join(str(t.value) for t in self._tokens[self._pos :])
            raise ValueError(f"Unexpected trailing input: {rest}")
        return result

    def _expr(self) -> Any:
        return self._addsub()

    def _addsub(self) -> Any:
        left = self._muldiv()
        while (tok := self._peek()) is not None:
            if tok.value == "+":
                self._advance()
                left = Add(left, self._muldiv())
            elif tok.value == "-":
                self._advance()
                left = Sub(left, self._muldiv())
            else:
                break
        return left

    def _muldiv(self) -> Any:
        left = self._unary()
        while (tok := self._peek()) is not None:
            if tok.value == "*":
                self._advance()
                left = Mul(left, self._unary())
            elif tok.value == "/":
                self._advance()
                left = Div(left, self._unary())
            else:
                break
        return left

    def _unary(self) -> Any:
        tok = self._peek()
        if tok is not None and tok.value in ("-", "+"):
            self._advance()
            operand = self._power()
            return Mul(-1, operand) if tok.value == "-" else operand
        return self._power()

    def _power(self) -> Any:
        left = self._atom()
        tok = self._peek()
        if tok is not None and tok.value == "**":
            self._advance()
            return Pow(left, self._unary())  # right-associative
        return left

    def _atom(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise ValueError("Expected expression but found end of input")

        if tok.type in ("NUMBER", "STRING"):
            self._advance()
            return tok.value

        if tok.type == "IDENT":
            self._advance()
            name = tok.value
            nxt = self._peek()
            if nxt is not None and nxt.value == "(":
                self._advance()
                args = self._arglist()
                self._expect(")")
                return self._call(name, args)
            if name in _LITERALS:
                return _LITERALS[name]
            if name in _CONSTS:
                return _CONSTS[name]
            if name in _REGISTRY:
                raise ValueError(f"{name!r} is a function — call it as {name}()")
            if name in self._env:
                return Ref(name, self._env[name])
            raise ValueError(f"Unknown name: {name}")

        if tok.value == "(":
            self._advance()
            inner = self._expr()
            self._expect(")")
            return inner

        raise ValueError(f"Unexpected token {tok.value!r}")

    def _arglist(self) -> list[tuple[str | None, Any]]:
        head = self._peek()
        if head is None or head.value == ")":
            return []
        args = [self._arg()]
        while (tok := self._peek()) is not None and tok.value == ",":
            self._advance()
            nxt = self._peek()
            if nxt is not None and nxt.value == ")":
                raise ValueError("Trailing comma in argument list")
            args.append(self._arg())
        return args

    def _arg(self) -> tuple[str | None, Any]:
        tok = self._peek()
        if tok is not None and tok.type == "IDENT":
            nxt = (
                self._tokens[self._pos + 1]
                if self._pos + 1 < len(self._tokens)
                else None
            )
            if nxt is not None and nxt.value == "=":
                key = self._advance().value  # type: ignore[union-attr]
                self._advance()  # consume '='
                return key, self._expr()
        return None, self._expr()

    def _call(self, name: str, raw_args: list[tuple[str | None, Any]]) -> Any:
        entry = _REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"Unknown function: {name}")
        func, params = entry
        rest = _REST_PARAMS.get(name)

        resolved: dict[str, Any] = {}
        positional = [v for k, v in raw_args if k is None]
        keyword = [(k, v) for k, v in raw_args if k is not None]

        rest_values: list[Any] = []
        for idx, value in enumerate(positional):
            if idx < len(params):
                resolved[params[idx]] = value
            elif rest is not None:
                rest_values.append(value)
            else:
                raise ValueError(f"Too many positional arguments for {name}")
        for key, value in keyword:
            if key not in params:
                raise ValueError(f"Unknown parameter {key!r} for {name}")
            if key in resolved:
                raise ValueError(f"Duplicate value for parameter {key!r} in {name}")
            resolved[key] = value
        if rest is not None:
            resolved[rest] = rest_values

        # Thread the set of explicitly-authored arg names to noise constructors
        # so the serializer emits only those (preserving explicit defaults).
        # `resolved` holds exactly the args that appeared in the source
        # (positional args already mapped to their names), captured before this
        # injection so `set_args` does not include itself.
        if name in _TRACKS_SET:
            resolved["set_args"] = frozenset(resolved.keys())

        return func(**resolved)


def parse_expr(src: str, env: Mapping[str, Expression2D] | None = None) -> Expression2D:
    """
    Parse a channel expression; bare numbers become :class:`Constant`. ``env``
    maps definition names to expressions, so a bare name referencing a
    ``Define`` resolves to a :class:`Ref`.
    """
    result = _Parser(_tokenize((src or "").strip()), env).parse()
    if isinstance(result, bool):
        raise TypeError("Expected an expression, got bool")
    if isinstance(result, (int, float)):
        return Constant(result)
    if isinstance(result, Expression2D):
        return result
    if isinstance(result, Shape):
        raise TypeError(
            "A shape is only valid inline inside Fill(...) or Stroke(...); "
            "it cannot be bound by Define or referenced by name"
        )
    if isinstance(result, Segment):
        raise TypeError("LineTo(...)/CubicTo(...) are only valid inside Path(...)")
    raise TypeError(f"Expected an expression, got {type(result).__name__}")
