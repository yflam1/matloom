"""
generate.py — a grammar-free random generator for layered-material programs.

Where :mod:`matloom.bnf` samples *strings* from a grammar, this module samples
*objects*: it builds a real :class:`~matloom.engine.main.LayeredMaterial` out of
the same engine constructors the parser uses, then leans on
:meth:`LayeredMaterial.serialize` to emit canonical DSL. Building objects (rather
than concatenating text) means every sampled program is valid by construction —
all pydantic validation, clamping, and serialization run exactly as they do for
hand-written materials, so the output always round-trips through
:meth:`LayeredMaterial.deserialize` and through the frontend importer.

What a generated program looks like (mirrors the canonical format)::

    View(<x1>, <y1>, <x2>, <y2>)
    Define(def1, <expr>)            # zero or more, def1, def2, …
    Define(def2, <expr>)            # later defs / layers may reference earlier
    Material(
      Layer(<alpha>)               # bottom-most first
        .basecolor(<r>, <g>, <b>)
        .metallic(<m>) .roughness(<r>) .emissive(<r>, <g>, <b>, <s>)
        .height(<h>),
      …
    )

Generation order matches the spec: (1) the number of layers, (2) the ``View``
region, (3) the ``Define`` preamble, (4) the ``Material`` layer stack. Every
channel slot is an independently sampled expression, drawn recursively from the
**full** pool of engine nodes — leaves (``Constant``/``X``/``Y``, plus any
``Ref`` to an already-defined ``Define``), the noise/pattern/shape generators
(``fBm``/``Worley``/``Bricks``/``Fill``/``Stroke``), the unary/binary math
nodes, the coordinate transforms (``Translate``/``Scale``/``Rotate``) and
``Threshold``.

Numeric arguments are sampled type-faithfully (see :class:`GeneratorConfig`):
``real`` over a bounded uniform range, ``pos_real`` strictly positive,
``non_neg_real`` ``>= 0``, positive integers (``octaves``) ``>= 1``, and color
bytes in ``[0, 255]``. The true ``(-inf, inf)`` / ``(0, inf)`` domains are
approximated by configurable bounded ranges, since an unbounded uniform is not
sampleable.

Reproducibility: pass ``seed`` (to the function or ``--seed`` on the CLI) and a
run is fully deterministic; all randomness flows through a single
:class:`random.Random` instance, so noise seeds are drawn from it too rather
than from the global RNG.
"""

import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass

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
from matloom.engine.layer import Color, Emissive, Layer
from matloom.engine.main import LayeredMaterial
from matloom.engine.noise import Worley, fBm
from matloom.engine.pattern import Bricks, Weave
from matloom.engine.shape import (
    CubicTo,
    Ellipse,
    Fill,
    LineTo,
    Path,
    Rect,
    Shape,
    Stroke,
)
from matloom.engine.transform import Rotate, Scale, Translate
from matloom.engine.util import Threshold

# Largest noise seed the engine accepts cleanly (matches noise.get_random_seed).
_MAX_SEED = 2**31 - 1

_WORLEY_DISTANCES = ("euclidean", "manhattan", "chebyshev")
_WORLEY_COMBINATIONS = ("F1", "F2", "F2-F1", "F2+F1")
_BRICK_AXES = ("row", "column")
_FILL_MODES = ("NonZero", "EvenOdd")


@dataclass
class GeneratorConfig:
    """
    Knobs controlling the random material generator. All ranges are inclusive of
    the documented endpoints unless noted; ``*_max`` bounds approximate the
    otherwise-unbounded numeric domains.

    Layers:
        min_layers / max_layers: the layer count is drawn uniformly from
            ``[min_layers, max_layers]`` (both positive, ``min <= max``).
            Defaults 1 and 3.
        num_layers: if set, use exactly this many layers (overrides the range).

    Defines:
        min_defs / max_defs: the number of ``Define``s is drawn from
            ``[min_defs, max_defs]`` (``0 <= min <= max``). Defaults 0 and 2.
        num_defs: if set, use exactly this many defines.

    Expressions:
        max_depth: maximum nesting depth of a generated expression tree
            (``>= 0``). At depth 0 only non-recursive nodes are produced.
        leaf_bias: probability in ``[0, 1]`` of stopping at a non-recursive node
            before ``max_depth`` is reached (higher → shallower trees).

    Layer 0:
        opaque_base: when True the bottom-most layer's alpha is the constant 1
            (a fully opaque substrate); when False it is a random expression.

    View:
        random_view: when True (default) the ``View`` region is a random
            non-degenerate rectangle; when False it is pinned to the unit square
            ``(0, 0, 1, 1)`` (the engine default).

    Numeric sampling:
        real_lo / real_hi: bounds for ``real`` parameters.
        pos_real_min / pos_real_max: bounds for ``pos_real`` (strictly > 0).
        non_neg_real_max: upper bound for ``non_neg_real`` (lower bound 0).
        max_octaves: upper bound for ``fBm`` octaves (lower bound 1).
        max_path_segments: upper bound for the number of ``Path`` segments
            (lower bound 1).
    """

    min_layers: int = 1
    max_layers: int = 3
    num_layers: int | None = None
    min_defs: int = 0
    max_defs: int = 2
    num_defs: int | None = None
    max_depth: int = 6
    leaf_bias: float = 0.5
    opaque_base: bool = True
    random_view: bool = True
    real_lo: float = -10.0
    real_hi: float = 10.0
    pos_real_min: float = 1e-3
    pos_real_max: float = 10.0
    non_neg_real_max: float = 10.0
    max_octaves: int = 8
    max_path_segments: int = 4

    def validate(self) -> None:
        if self.num_layers is not None:
            if self.num_layers < 1:
                raise ValueError("num_layers must be a positive integer")
        else:
            if self.min_layers < 1 or self.max_layers < 1:
                raise ValueError("min_layers and max_layers must be positive")
            if self.min_layers > self.max_layers:
                raise ValueError("min_layers must be <= max_layers")
        if self.num_defs is not None:
            if self.num_defs < 0:
                raise ValueError("num_defs must be >= 0")
        else:
            if self.min_defs < 0 or self.max_defs < 0:
                raise ValueError("min_defs and max_defs must be >= 0")
            if self.min_defs > self.max_defs:
                raise ValueError("min_defs must be <= max_defs")
        if self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if not 0.0 <= self.leaf_bias <= 1.0:
            raise ValueError("leaf_bias must be in [0, 1]")
        if self.real_lo > self.real_hi:
            raise ValueError("real_lo must be <= real_hi")
        if not 0.0 < self.pos_real_min <= self.pos_real_max:
            raise ValueError("require 0 < pos_real_min <= pos_real_max")
        if self.non_neg_real_max < 0.0:
            raise ValueError("non_neg_real_max must be >= 0")
        if self.max_octaves < 1:
            raise ValueError("max_octaves must be >= 1")
        if self.max_path_segments < 1:
            raise ValueError("max_path_segments must be >= 1")


class MaterialGenerator:
    """
    Stateful random generator. One instance owns a single :class:`random.Random`
    (seeded for reproducibility) and the running table of ``Define``s so that
    references resolve only to already-defined names.
    """

    def __init__(
        self, config: GeneratorConfig | None = None, *, seed: int | None = None
    ) -> None:
        self._cfg = config or GeneratorConfig()
        self._cfg.validate()
        self._rng = random.Random(seed)
        # (name, expression) for each Define produced so far, in order.
        self._env: list[tuple[str, Expression2D]] = []

    # |---------------------------|
    # |   Numeric sampling helpers |
    # |---------------------------|

    def _real(self) -> float:
        return self._rng.uniform(self._cfg.real_lo, self._cfg.real_hi)

    def _pos_real(self) -> float:
        return self._rng.uniform(self._cfg.pos_real_min, self._cfg.pos_real_max)

    def _non_neg_real(self) -> float:
        return self._rng.uniform(0.0, self._cfg.non_neg_real_max)

    def _nonzero_real(self) -> float:
        # NonZeroFloat: a real that is not exactly 0. Resampling away an exact 0
        # is essentially never needed but keeps the contract exact.
        v = self._real()
        while v == 0.0:
            v = self._real()
        return v

    def _zero_to_one(self) -> float:
        return self._rng.random()

    def _degrees(self) -> float:
        return self._rng.uniform(0.0, 360.0)

    def _byte(self) -> int:
        return self._rng.randint(0, 255)

    def _seed(self) -> int:
        return self._rng.randint(0, _MAX_SEED)

    def _chance(self, p: float) -> bool:
        return self._rng.random() < p

    # |--------------------|
    # |   Expression tree  |
    # |--------------------|

    def expression(self, depth: int | None = None) -> Expression2D:
        """Sample a random :class:`Expression2D` up to ``depth`` (default
        ``config.max_depth``)."""
        if depth is None:
            depth = self._cfg.max_depth
        if depth <= 0 or self._chance(self._cfg.leaf_bias):
            return self._terminal()
        return self._rng.choice(self._recursive)(depth)

    def _terminal(self) -> Expression2D:
        choices = list(self._terminals)
        if self._env:
            choices.append(self._gen_ref)
        return self._rng.choice(choices)()

    # -- non-recursive (terminal) producers --

    def _gen_constant(self) -> Expression2D:
        return Constant(self._real())

    def _gen_x(self) -> Expression2D:
        return X()

    def _gen_y(self) -> Expression2D:
        return Y()

    def _gen_ref(self) -> Expression2D:
        name, expr = self._rng.choice(self._env)
        return Ref(name, expr)

    def _gen_bricks(self) -> Expression2D:
        return Bricks(
            brick_width=self._pos_real(),
            brick_height=self._pos_real(),
            offset=self._real(),
            mortar=self._non_neg_real(),
            axis=self._rng.choice(_BRICK_AXES),
            feather=self._non_neg_real() if self._chance(0.5) else 0.0,
        )

    def _gen_weave(self) -> Expression2D:
        kwargs: dict = {
            "over": self._rng.randint(1, 4),
            "under": self._rng.randint(1, 4),
            "shift": self._rng.randint(0, 3),
            "warp_width": self._rng.uniform(0.05, 1.0),
            "feather": self._non_neg_real() if self._chance(0.5) else 0.0,
        }
        # Weave frequencies must be strictly positive (unlike noise, which
        # accepts a sign), so sample from the positive range.
        if self._chance(0.5):
            kwargs["base_freq"] = self._pos_real()
        else:
            kwargs["base_freq_x"] = self._pos_real()
            kwargs["base_freq_y"] = self._pos_real()
        return Weave(**kwargs)

    def _gen_fbm(self) -> Expression2D:
        kwargs: dict = {
            "octaves": self._rng.randint(1, self._cfg.max_octaves),
            "lacunarity": self._pos_real(),
            "gain": self._pos_real(),
            "to_01": self._chance(0.5),
            "seed": self._seed(),
        }
        self._add_freq(kwargs)
        return fBm(**kwargs)

    def _gen_worley(self) -> Expression2D:
        kwargs: dict = {
            "distance": self._rng.choice(_WORLEY_DISTANCES),
            "combination": self._rng.choice(_WORLEY_COMBINATIONS),
            "to_01": self._chance(0.5),
            "seed": self._seed(),
        }
        self._add_freq(kwargs)
        return Worley(**kwargs)

    def _add_freq(self, kwargs: dict) -> None:
        """Populate either a shared ``base_freq`` or per-axis frequencies."""
        if self._chance(0.5):
            kwargs["base_freq"] = self._nonzero_real()
        else:
            kwargs["base_freq_x"] = self._nonzero_real()
            kwargs["base_freq_y"] = self._nonzero_real()

    def _gen_fill(self) -> Expression2D:
        return Fill(
            self._shape(),
            mode=self._rng.choice(_FILL_MODES),
            feather=self._non_neg_real() if self._chance(0.5) else 0.0,
        )

    def _gen_stroke(self) -> Expression2D:
        return Stroke(
            self._shape(),
            width=self._pos_real(),
            feather=self._non_neg_real() if self._chance(0.5) else 0.0,
        )

    # -- recursive producers (take the current depth, recurse at depth - 1) --

    def _gen_unary(self, depth: int) -> Expression2D:
        node = self._rng.choice((Abs, Sqrt, Floor, Ceil, Sin, Cos))
        return node(self.expression(depth - 1))

    def _gen_log(self, depth: int) -> Expression2D:
        base = self._pos_real()
        if abs(base - 1.0) < 1e-9:  # log base must be > 0 and != 1
            base += 0.5
        return Log(self.expression(depth - 1), base)

    def _gen_binary(self, depth: int) -> Expression2D:
        node = self._rng.choice((Add, Sub, Mul, Div, Pow, Min, Max))
        return node(self.expression(depth - 1), self.expression(depth - 1))

    def _gen_translate(self, depth: int) -> Expression2D:
        return Translate(self.expression(depth - 1), self._real(), self._real())

    def _gen_scale(self, depth: int) -> Expression2D:
        return Scale(
            self.expression(depth - 1),
            self._nonzero_real(),
            self._nonzero_real(),
        )

    def _gen_rotate(self, depth: int) -> Expression2D:
        return Rotate(self.expression(depth - 1), self._degrees())

    def _gen_threshold(self, depth: int) -> Expression2D:
        kwargs: dict = {}
        if self._chance(0.6):
            kwargs["below_at"] = self._real()
            if self._chance(0.5):
                kwargs["below_to"] = self._real()
        if self._chance(0.6):
            kwargs["above_at"] = self._real()
            if self._chance(0.5):
                kwargs["above_to"] = self._real()
        if self._chance(0.5):
            kwargs["transition_width"] = self._non_neg_real()
        return Threshold(self.expression(depth - 1), **kwargs)

    # -- shapes (geometry inside Fill/Stroke; not expressions on their own) --

    def _shape(self) -> Shape:
        return self._rng.choice((self._rect, self._ellipse, self._path))()

    def _rect(self) -> Shape:
        radius = self._non_neg_real() if self._chance(0.5) else 0.0
        return Rect(
            self._real(),
            self._real(),
            self._pos_real(),
            self._pos_real(),
            radius,
        )

    def _ellipse(self) -> Shape:
        return Ellipse(self._real(), self._real(), self._pos_real(), self._pos_real())

    def _path(self) -> Shape:
        n = self._rng.randint(1, self._cfg.max_path_segments)
        segments = [self._segment() for _ in range(n)]
        return Path(self._real(), self._real(), segments)

    def _segment(self):
        if self._chance(0.5):
            return LineTo(self._real(), self._real())
        return CubicTo(*(self._real() for _ in range(6)))

    # Producer pools, assembled once. Terminals are depth-independent; recursive
    # producers take the remaining depth budget.
    @property
    def _terminals(self) -> tuple[Callable[[], Expression2D], ...]:
        return (
            self._gen_constant,
            self._gen_x,
            self._gen_y,
            self._gen_bricks,
            self._gen_weave,
            self._gen_fbm,
            self._gen_worley,
            self._gen_fill,
            self._gen_stroke,
        )

    @property
    def _recursive(self) -> tuple[Callable[[int], Expression2D], ...]:
        return (
            self._gen_unary,
            self._gen_log,
            self._gen_binary,
            self._gen_translate,
            self._gen_scale,
            self._gen_rotate,
            self._gen_threshold,
        )

    # |------------------|
    # |   Material parts |
    # |------------------|

    def _channel(self) -> Expression2D:
        return self.expression(self._cfg.max_depth)

    def _color(self) -> Color:
        return Color(r=self._channel(), g=self._channel(), b=self._channel())

    def _emissive(self) -> Emissive:
        return Emissive(
            r=self._channel(),
            g=self._channel(),
            b=self._channel(),
            strength=self._channel(),
        )

    def _layer(self, *, opaque: bool) -> Layer:
        alpha: Expression2D = Constant(1.0) if opaque else self._channel()
        return Layer(
            alpha=alpha,
            basecolor=self._color(),
            metallic=self._channel(),
            roughness=self._channel(),
            sheen=self._channel(),
            coat=self._channel(),
            transmission=self._channel(),
            ior=self._channel(),
            subsurface=self._channel(),
            anisotropy=self._channel(),
            emissive=self._emissive(),
            height=self._channel(),
        )

    def _resolve_count(self, exact: int | None, lo: int, hi: int) -> int:
        return exact if exact is not None else self._rng.randint(lo, hi)

    def generate(self) -> LayeredMaterial:
        """Build and return a random :class:`LayeredMaterial`."""
        cfg = self._cfg
        self._env = []
        material = LayeredMaterial()

        # (1) number of layers.
        n_layers = self._resolve_count(cfg.num_layers, cfg.min_layers, cfg.max_layers)

        # (2) View region: a non-degenerate rectangle (x1 < x2, y1 < y2), or the
        # unit square (0, 0, 1, 1) when random_view is disabled.
        if cfg.random_view:
            x1 = self._real()
            y1 = self._real()
            material.set_view(x1, y1, x1 + self._pos_real(), y1 + self._pos_real())
        else:
            material.set_view(0.0, 0.0, 1.0, 1.0)

        # (3) Define preamble; each may reference earlier defines.
        n_defs = self._resolve_count(cfg.num_defs, cfg.min_defs, cfg.max_defs)
        for i in range(1, n_defs + 1):
            name = f"def{i}"
            expr = self.expression(cfg.max_depth)
            material.add_def(name, expr)
            self._env.append((name, expr))

        # (4) Material layer stack; index 0 is the bottom-most layer.
        for idx in range(n_layers):
            opaque = cfg.opaque_base and idx == 0
            material.add_layer(self._layer(opaque=opaque))

        return material


def generate_material(
    config: GeneratorConfig | None = None,
    *,
    seed: int | None = None,
    as_string: bool = False,
) -> LayeredMaterial | str:
    """
    Generate a random material program.

    Args:
        config: generation knobs (see :class:`GeneratorConfig`); defaults used
            when omitted.
        seed: optional seed for reproducible output.
        as_string: when True, return the serialized canonical DSL string;
            otherwise return the :class:`LayeredMaterial` object (the default).

    Returns:
        A :class:`LayeredMaterial`, or its serialized string when ``as_string``.
    """
    material = MaterialGenerator(config, seed=seed).generate()
    return material.serialize() if as_string else material


# |---------|
# |   CLI   |
# |---------|


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matloom-generate",
        description="Randomly generate a layered-material DSL program.",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="seed for reproducible output"
    )
    p.add_argument("--min-layers", type=int, default=1)
    p.add_argument("--max-layers", type=int, default=3)
    p.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="exact layer count (overrides range)",
    )
    p.add_argument("--min-defs", type=int, default=0)
    p.add_argument("--max-defs", type=int, default=2)
    p.add_argument(
        "--num-defs",
        type=int,
        default=None,
        help="exact Define count (overrides range)",
    )
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--leaf-bias", type=float, default=0.5)
    p.add_argument(
        "--opaque-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="force the bottom-most layer's alpha to 1 (use --no-opaque-base to disable)",
    )
    p.add_argument(
        "--random-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="randomize the View region (use --no-random-view to pin it to 0,0,1,1)",
    )
    p.add_argument("--real-lo", type=float, default=-10.0)
    p.add_argument("--real-hi", type=float, default=10.0)
    p.add_argument("--pos-real-min", type=float, default=1e-3)
    p.add_argument("--pos-real-max", type=float, default=10.0)
    p.add_argument("--non-neg-real-max", type=float, default=10.0)
    p.add_argument("--max-octaves", type=int, default=8)
    p.add_argument("--max-path-segments", type=int, default=4)
    p.add_argument(
        "--as-string",
        action="store_true",
        help="(API parity) request the serialized string; the CLI prints DSL either way",
    )
    return p


def _cli(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    config = GeneratorConfig(
        min_layers=args.min_layers,
        max_layers=args.max_layers,
        num_layers=args.num_layers,
        min_defs=args.min_defs,
        max_defs=args.max_defs,
        num_defs=args.num_defs,
        max_depth=args.max_depth,
        leaf_bias=args.leaf_bias,
        opaque_base=args.opaque_base,
        random_view=args.random_view,
        real_lo=args.real_lo,
        real_hi=args.real_hi,
        pos_real_min=args.pos_real_min,
        pos_real_max=args.pos_real_max,
        non_neg_real_max=args.non_neg_real_max,
        max_octaves=args.max_octaves,
        max_path_segments=args.max_path_segments,
    )
    result = generate_material(config, seed=args.seed, as_string=args.as_string)
    print(result if isinstance(result, str) else result.serialize())


if __name__ == "__main__":
    _cli()
