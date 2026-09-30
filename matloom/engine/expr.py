"""The ``Expression2D`` tree: scalar ``__call__`` + vectorized ``eval_grid``.

Purity contract: ``eval_grid`` implementations and their consumers must never
mutate the arrays returned by child expressions in place. Every node allocates
its own output today, which is what makes the render-scoped memo (see
:func:`grid_cache_scope`) sound — a memo hit hands the SAME array object to
every consumer of a shared subtree, so an in-place consumer would corrupt all
later consumers; the memo-on/off bit-identity test fails loudly in that case.
"""

import functools
import inspect
import math
import types
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Union, get_args, get_origin, get_type_hints

import numpy as np
from typing_extensions import Self

from matloom.engine.const import e

# Per-cell coordinate grids (xx, yy), each of shape (h, w). When supplied to
# ``eval_grid`` they override the separable 1-D axes for the X/Y leaves, letting
# a non-separable transform (Rotate) sample its child subtree. Noise nodes
# ignore them and keep sampling the original axes, so they stay axis-aligned.
Coords = tuple[np.ndarray, np.ndarray]


def _is_expression_type(tp: Any) -> bool:
    # Reject parameterized generics such as ``frozenset[str]`` before the
    # ``issubclass`` call. On Python 3.10 a ``types.GenericAlias`` is considered
    # an instance of ``type`` (fixed in 3.11+), so the ``isinstance`` guard
    # alone lets it through; ``issubclass`` then dispatches to
    # ``ABCMeta.__subclasscheck__`` and raises "issubclass() arg 1 must be a
    # class". ``get_origin`` is non-None only for parameterized generics, so it
    # filters them on every version.
    if not isinstance(tp, type) or get_origin(tp) is not None:
        return False
    return issubclass(tp, Expression2D)


def _accepts_expression(annotation: Any) -> bool:
    """
    Whether `annotation` is :class:`Expression2D` (or a subclass), on its own
    or as a member of an `Optional`/`Union`.
    """
    if _is_expression_type(annotation):
        return True
    if get_origin(annotation) in (Union, types.UnionType):
        return any(_is_expression_type(arg) for arg in get_args(annotation))
    return False


def _numeric_kind(annotation: Any) -> str | None:
    """
    Classify `annotation` as the plain scalar type it expects: `"int"` or
    `"float"` (directly, or as a member of an `Optional`/`Union`), else
    `None`. `bool` is intentionally not treated as numeric even though it is a
    subclass of `int`.
    """

    def kind_of(tp: Any) -> str | None:
        if tp is int:
            return "int"
        if tp is float:
            return "float"
        return None

    direct = kind_of(annotation)
    if direct is not None:
        return direct
    if get_origin(annotation) in (Union, types.UnionType):
        for arg in get_args(annotation):
            kind = kind_of(arg)
            if kind is not None:
                return kind
    return None


def coerce_args(init: Callable[..., None]) -> Callable[..., None]:
    """
    Decorator for an `__init__` that reconciles bare scalars and expressions:

    - For every parameter annotated as :class:`Expression2D` (or
      `Expression2D | None`), wrap an `int`/`float` argument in
      :class:`Constant`, so callers can pass bare numbers wherever an
      expression is expected.
    - For every parameter annotated as a plain `int`/`float` (directly or
      within an `Optional`/`Union`), evaluate an :class:`Expression2D`
      argument to a scalar by calling it (`x`/`y` default to `0.0`).
      `float` parameters keep the value as-is; `int` parameters round it.
      This lets expression classes such as :class:`Log`/:class:`Sqrt` be
      reused wherever a scalar is expected, e.g. `fBm(octaves=Log(1000, 10))`.
    """
    sig = inspect.signature(init)
    # Resolve annotations lazily on first use: forward references such as
    # `Constant` and the expression subclasses are not yet defined when this
    # decorator runs at class-definition time.
    expr_params: set[str] = set()
    numeric_params: dict[str, str] = {}
    resolved = False

    @functools.wraps(init)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        nonlocal resolved
        if not resolved:
            hints = get_type_hints(inspect.unwrap(init))
            for name in sig.parameters:
                annotation = hints.get(name)
                if _accepts_expression(annotation):
                    expr_params.add(name)
                else:
                    kind = _numeric_kind(annotation)
                    if kind is not None:
                        numeric_params[name] = kind
            resolved = True
        bound = sig.bind(*args, **kwargs)
        for name in expr_params:
            value = bound.arguments.get(name)
            if isinstance(value, (int, float)):
                bound.arguments[name] = Constant(value)
        for name, kind in numeric_params.items():
            value = bound.arguments.get(name)
            if isinstance(value, Expression2D):
                scalar = value()
                bound.arguments[name] = (
                    round(scalar) if kind == "int" else float(scalar)
                )
        return init(*bound.args, **bound.kwargs)

    return wrapper


class Expression2D(ABC):
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        instance = super().__new__(cls)
        _args = [repr(a) for a in args]
        _args.extend(f"{k}={v!r}" for k, v in kwargs.items())
        instance._init_args_str = ", ".join(_args)
        return instance

    @abstractmethod
    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        pass

    def __add__(self, other: Self):
        return AdditionExpression2D(self, other)

    def __radd__(self, other: Self):
        return AdditionExpression2D(other, self)

    def __sub__(self, other: Self):
        return SubtractionExpression2D(self, other)

    def __rsub__(self, other: Self):
        return SubtractionExpression2D(other, self)

    def __mul__(self, other: Self):
        return MultiplicationExpression2D(self, other)

    def __rmul__(self, other: Self):
        return MultiplicationExpression2D(other, self)

    def __truediv__(self, other: Self):
        return DivisionExpression2D(self, other)

    def __rtruediv__(self, other: Self):
        return DivisionExpression2D(other, self)

    def __pow__(self, other: Self):
        return PowerExpression2D(self, other)

    def __rpow__(self, other: Self):
        return PowerExpression2D(other, self)

    def __neg__(self):
        return MultiplicationExpression2D(-1.0, self)

    def __repr__(self) -> str:
        args = getattr(self, "_init_args_str", "")
        return f"{self.__class__.__name__}({args})"

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        """
        Evaluate on a meshgrid (fallback). xs/ys are 1-D coordinate arrays;
        when ``coords`` is given it supplies explicit per-cell (xx, yy) grids
        instead (used by :class:`~matloom.engine.transform.Rotate`).
        """
        xx, yy = coords if coords is not None else np.meshgrid(xs, ys)
        return np.vectorize(self.__call__)(xx, yy)


class AdditionExpression2D(Expression2D):
    @coerce_args
    def __init__(self, left: Expression2D, right: Expression2D) -> None:
        self._left = left
        self._right = right

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        a = self._left(x, y)
        b = self._right(x, y)
        return a + b

    def __repr__(self) -> str:
        return f"({self._left} + {self._right})"

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        a = self._left.eval_grid(xs, ys, coords)
        b = self._right.eval_grid(xs, ys, coords)
        return a + b


class SubtractionExpression2D(Expression2D):
    @coerce_args
    def __init__(self, minuend: Expression2D, subtrahend: Expression2D) -> None:
        self._minuend = minuend
        self._subtrahend = subtrahend

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        a = self._minuend(x, y)
        b = self._subtrahend(x, y)
        return a - b

    def __repr__(self) -> str:
        return f"({self._minuend} - {self._subtrahend})"

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        a = self._minuend.eval_grid(xs, ys, coords)
        b = self._subtrahend.eval_grid(xs, ys, coords)
        return a - b


class MultiplicationExpression2D(Expression2D):
    @coerce_args
    def __init__(self, left: Expression2D, right: Expression2D) -> None:
        self._left = left
        self._right = right

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return self._left(x, y) * self._right(x, y)

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        return self._left.eval_grid(xs, ys, coords) * self._right.eval_grid(
            xs, ys, coords
        )

    def __repr__(self) -> str:
        return f"({self._left} * {self._right})"


class DivisionExpression2D(Expression2D):
    @coerce_args
    def __init__(self, dividend: Expression2D, divisor: Expression2D) -> None:
        self._dividend = dividend
        self._divisor = divisor

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        b = self._divisor(x, y)
        if b == 0.0:
            warnings.warn(
                f"Division by zero in {self!r} at (x={x}, y={y}); returning 0",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0
        a = self._dividend(x, y)
        return a / b

    def __repr__(self) -> str:
        return f"({self._dividend} / {self._divisor})"

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        a = self._dividend.eval_grid(xs, ys, coords)
        b = self._divisor.eval_grid(xs, ys, coords)
        zero = b == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a / b
        if zero.any():
            warnings.warn(
                f"Division by zero in {self!r} at {int(zero.sum())} grid "
                f"point(s); returning 0 there",
                RuntimeWarning,
                stacklevel=2,
            )
            out[zero] = 0.0
        return out


class PowerExpression2D(Expression2D):
    @coerce_args
    def __init__(self, base: Expression2D, exponent: Expression2D) -> None:
        self._base = base
        self._exponent = exponent

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        a = self._base(x, y)
        b = self._exponent(x, y)
        return a**b

    def __repr__(self) -> str:
        return f"({self._base} ** {self._exponent})"

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        a = self._base.eval_grid(xs, ys, coords)
        b = self._exponent.eval_grid(xs, ys, coords)
        return a**b


def Add(left: Expression2D, right: Expression2D) -> Expression2D:
    return AdditionExpression2D(left, right)


def Sub(minuend: Expression2D, subtrahend: Expression2D) -> Expression2D:
    return SubtractionExpression2D(minuend, subtrahend)


def Mul(left: Expression2D, right: Expression2D) -> Expression2D:
    return MultiplicationExpression2D(left, right)


def Div(dividend: Expression2D, divisor: Expression2D) -> Expression2D:
    return DivisionExpression2D(dividend, divisor)


def Pow(base: Expression2D, exponent: Expression2D) -> Expression2D:
    return PowerExpression2D(base, exponent)


# Transient per-render cap: ~128 f64 grids at 256x256 (32 at 512), bounding the
# memo's RSS while keeping the few hot Define'd grids that carry the speedup.
_GRID_CACHE_MAX_BYTES = 64 << 20


class _GridCache:
    """Per-render memo of ``eval_grid`` results: a byte-capped LRU keyed by
    object ids. Every entry PINS the objects its key is derived from (the
    delegate expression, ``xs``/``ys``, the ``coords`` grids), so a freed
    object's id cannot be recycled and collide with a live entry. Entries
    larger than the cap are skipped, never inserted-and-evicted. Confined to
    one thread's context — no locking needed (one render per thread)."""

    def __init__(self, max_bytes: int = _GRID_CACHE_MAX_BYTES) -> None:
        self._d: OrderedDict[tuple, tuple[np.ndarray, tuple]] = OrderedDict()
        self._bytes = 0
        self._max_bytes = max_bytes

    def get(self, key: tuple) -> np.ndarray | None:
        """Return the memoized grid for ``key`` (marking it most recently
        used), or ``None`` on a miss."""
        entry = self._d.get(key)
        if entry is None:
            return None
        self._d.move_to_end(key)
        return entry[0]

    def put(self, key: tuple, out: np.ndarray, pins: tuple) -> None:
        """Memoize ``out`` under ``key``, holding ``pins`` (the objects the
        key's ids came from) so they outlive the entry. Skips entries larger
        than the cap, then evicts least-recently-used entries until the total
        is back under it."""
        if out.nbytes > self._max_bytes:
            return
        if key in self._d:
            # A re-put replaces the entry: drop the old bytes first so they
            # are not double-counted (Ref.eval_grid only puts after a miss, so
            # this is future-proofing for direct users of the cache).
            self._bytes -= self._d[key][0].nbytes
        self._d[key] = (out, pins)
        self._d.move_to_end(key)
        self._bytes += out.nbytes
        while self._bytes > self._max_bytes:
            _, (dropped, _) = self._d.popitem(last=False)
            self._bytes -= dropped.nbytes


_grid_cache: ContextVar["_GridCache | None"] = ContextVar(
    "matloom_grid_cache", default=None
)


@contextmanager
def grid_cache_scope() -> Iterator[None]:
    """Activate the per-render ``eval_grid`` memo for the enclosed block
    (engine-internal; used by :mod:`matloom.engine.preview`). Only
    :class:`Ref` consults it: the parser routes every use of a ``Define``'d
    name through ``Ref(name, shared_expr)``, so an identity-keyed hit is
    exactly the same computation and skips the entire shared subtree.
    Outside a scope (export/parity/diagnostics) the memo is inert. The scope
    is exception-safe: the cache is dropped even if the render raises, and a
    nested scope gets a fresh independent cache. Consumers must not mutate
    returned child outputs in place — a hit hands out the SAME array object
    to every consumer of the shared subtree (see the module docstring's
    purity contract)."""
    token = _grid_cache.set(_GridCache())
    try:
        yield
    finally:
        _grid_cache.reset(token)


class Ref(Expression2D):
    """
    A named reference to a stored expression (a ``Define``). It delegates every
    evaluation to the bound expression but serializes back to the bare name, so
    a definition round-trips as a single ``Define`` rather than being inlined.
    Tree walks (e.g. noise collection) stop at a Ref — the referenced
    expression is owned by its definition, not by each use site.

    Under an active :func:`grid_cache_scope`, ``eval_grid`` memoizes the
    delegate's grid by object identity, so one ``Define`` referenced from many
    channels evaluates once per render.
    """

    def __init__(self, name: str, expr: "Expression2D") -> None:
        self._name = name
        self._expr = expr

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return self._expr(x, y)

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        cache = _grid_cache.get()
        if cache is None:
            return self._expr.eval_grid(xs, ys, coords)
        coords_key = None if coords is None else (id(coords[0]), id(coords[1]))
        key = (id(self._expr), id(xs), id(ys), coords_key)
        hit = cache.get(key)
        if hit is not None:
            return hit
        out = self._expr.eval_grid(xs, ys, coords)
        # Pin every object the key was derived from so a recycled id can never
        # alias a live entry (pinning the coords tuple pins both its grids).
        pins = (self._expr, xs, ys) if coords is None else (self._expr, xs, ys, coords)
        cache.put(key, out, pins)
        return out


class Constant(Expression2D):
    @coerce_args
    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return self._value

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        shape = coords[0].shape if coords is not None else (ys.size, xs.size)
        # Always emit a floating grid. A bare integer literal (e.g. ``Layer(1)``
        # or ``.height(5)``) would otherwise yield an int64 array that poisons
        # downstream float math (in-place ``out=`` ufuncs / numba kernels raise a
        # cast error on it). float64 matches the int64→float64 promotion these
        # constants already triggered in mixed expressions, so values are
        # unchanged; ``self._value`` (and thus serialization) stays an int.
        return np.full(shape, self._value, dtype=np.float64)


class X(Expression2D):
    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return x

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        if coords is not None:
            return coords[0]
        return np.broadcast_to(xs[np.newaxis, :], (ys.size, xs.size)).copy()


class Y(Expression2D):
    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return y

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        if coords is not None:
            return coords[1]
        return np.broadcast_to(ys[:, np.newaxis], (ys.size, xs.size)).copy()


class Abs(Expression2D):
    @coerce_args
    def __init__(self, expr: Expression2D) -> None:
        self._expr = expr

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return abs(self._expr(x, y))

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        return np.abs(self._expr.eval_grid(xs, ys, coords))


class Sqrt(Expression2D):
    @coerce_args
    def __init__(self, expr: Expression2D) -> None:
        self._expr = expr

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return math.sqrt(self._expr(x, y))

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        return np.sqrt(self._expr.eval_grid(xs, ys, coords))


class Floor(Expression2D):
    @coerce_args
    def __init__(self, expr: Expression2D) -> None:
        self._expr = expr

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return float(math.floor(self._expr(x, y)))

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        return np.floor(self._expr.eval_grid(xs, ys, coords))


class Ceil(Expression2D):
    @coerce_args
    def __init__(self, expr: Expression2D) -> None:
        self._expr = expr

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return float(math.ceil(self._expr(x, y)))

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        return np.ceil(self._expr.eval_grid(xs, ys, coords))


class Log(Expression2D):
    @coerce_args
    def __init__(self, expr: Expression2D, base: float = e) -> None:
        self._expr = expr
        self._base = base

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return math.log(self._expr(x, y), self._base)

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        val = self._expr.eval_grid(xs, ys, coords)
        if self._base == 10:
            return np.log10(val)
        if self._base == 2:
            return np.log2(val)
        out = np.log(val)
        if self._base != e:
            out /= np.log(self._base)
        return out


class Sin(Expression2D):
    @coerce_args
    def __init__(self, expr: Expression2D) -> None:
        self._expr = expr

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return math.sin(self._expr(x, y))

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        return np.sin(self._expr.eval_grid(xs, ys, coords))


class Cos(Expression2D):
    @coerce_args
    def __init__(self, expr: Expression2D) -> None:
        self._expr = expr

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return math.cos(self._expr(x, y))

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        return np.cos(self._expr.eval_grid(xs, ys, coords))


class Min(Expression2D):
    @coerce_args
    def __init__(self, expr1: Expression2D, expr2: Expression2D) -> None:
        self._expr1 = expr1
        self._expr2 = expr2

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return min(self._expr1(x, y), self._expr2(x, y))

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        return np.minimum(
            self._expr1.eval_grid(xs, ys, coords),
            self._expr2.eval_grid(xs, ys, coords),
        )


class Max(Expression2D):
    @coerce_args
    def __init__(self, expr1: Expression2D, expr2: Expression2D) -> None:
        self._expr1 = expr1
        self._expr2 = expr2

    def __call__(self, x: float = 0.0, y: float = 0.0) -> float:
        return max(self._expr1(x, y), self._expr2(x, y))

    def eval_grid(
        self, xs: np.ndarray, ys: np.ndarray, coords: Coords | None = None
    ) -> np.ndarray:
        return np.maximum(
            self._expr1.eval_grid(xs, ys, coords),
            self._expr2.eval_grid(xs, ys, coords),
        )
