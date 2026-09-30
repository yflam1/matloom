"""Seed-sweep + CMA-ES polish for layered-material programs.

The LLM proposes a program's *structure* (layers, masks, which channels, which
functions); this module disposes its *realization* — first the noise **seeds**
(a cheap discrete sweep), then, optionally, the *continuous parameters* (the
numeric literals) via CMA-ES — against a held-out verifier. "FunSearch-for-
materials": LLM structure, evolution parameters. The seed sweep is primary;
CMA-ES is a secondary, per-variant continuous polish (``max_cma_evals`` toggles
it).

Design:

- **Tunable set is engine-schema-driven.** A literal is tunable iff it is a
  numeric literal in the LLM-authored program text that is NOT a ``View``
  argument and NOT a named argument the engine treats as structural (``seed``,
  ``octaves``, ``to_01``, ``over``/``under``/``shift``, ``axis``,
  ``distance``/``combination``, ``mode``). Frequencies (``base_freq`` etc.) are
  searched in log space; other named args by sensible linear ranges; positional
  literals (channel values, multipliers, coordinates) by a wide range — the
  engine auto-clamps channel outputs, so wide bounds are safe and CMA-ES adapts
  its step size to fine-tune within them. Operating on the **authored text**
  (not the canonical AST) avoids tuning engine artifacts: the auto-clamp
  ``Threshold`` the parser wraps around every channel value, and the ``255``
  sRGB-conversion divisor the canonical form inserts.

- **Apply edits the text in place** (right-to-left so spans stay valid) and
  re-parses for every fitness evaluation, so precomputed node values
  (``Rotate._cos``, ``Rect._cx``) are recomputed correctly via the round-trip.

- **Fitness is a held-out verifier** (MobileCLIP2-S2) on a fast **no-Blender
  fixed-light preview** (albedo shaded by a fixed Lambertian light + the engine's
  normal/AO, with roughness/metallic/emissive/transmission approximated). It is
  CLIP-family, so headline gains are claimed on the non-CLIP-family eval metrics.

- **Accept-if-better**: the returned program is the highest-scoring one across
  the seed sweep and all CMA-ES polishes (the ``max``), so the polish never does
  worse on the fitness than the as-authored program.

- **Opt-in render pool**: ``polish_program(parallel_renders=n)`` renders the
  seed sweep's variants across ``n`` spawned worker processes. Off by default
  (the library path renders sequentially, byte-for-byte); renders are pure
  functions of the program text, so enabling it changes only wall time, never
  the returned program or the logged scores.

The CMA-ES optimizer is the vetted ``cma`` package (lazy-imported), so the
covariance / step-size math is not hand-rolled. The module is import-safe
without ``cma``; :func:`polish_program` only imports it when ``max_cma_evals >
0`` (the seed sweep, tunable extraction, and preview are ``cma``-free).
"""

import math
import multiprocessing
import os
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from matloom import logger
from matloom.engine.main import LayeredMaterial
from matloom.engine.preview import fixed_light_preview
from matloom.utils.dtypes import ImgLike, SDict
from matloom.utils.progress import progress_bar

# Named args that are structural (not continuous-tunable).
_EXCLUDED_ARGS = frozenset(
    {
        "seed",
        "octaves",
        "to_01",
        "over",
        "under",
        "shift",
        "axis",
        "distance",
        "combination",
        "mode",
    }
)
# All args of View(...) are structural (the sampling window, not the material).
_EXCLUDED_CALLS = frozenset({"View"})

# Bounds for named args: name -> (lo, hi, scale). "log" searches log10(v) in
# [lo, hi] (so lo/hi are log10 bounds); "lin" searches v in [lo, hi]. Bounds are
# the SEARCH SPACE, not a hint: an over-wide range (e.g. base_freq over 6
# decades, mortar up to 20x a brick) lets CMA-ES spend its whole budget scoring
# degenerate programs (a "wall" that is one brick, or all mortar). Keep each
# range within the world-scale band where the material still reads as itself;
# the engine only requires gt=0/ge=0, so the sane band is far narrower.
_FREQ = (-1.0, 1.7, "log")  # log10 bounds -> v in [0.1, 50]
_NAMED_BOUNDS: SDict[tuple[float, float, str]] = {
    "base_freq": _FREQ,
    "base_freq_x": _FREQ,
    "base_freq_y": _FREQ,
    "lacunarity": (1.0, 4.0, "lin"),
    "gain": (0.0, 1.0, "lin"),
    "mortar": (0.0, 1.0, "lin"),
    "feather": (0.0, 0.2, "lin"),
    "brick_width": (0.05, 10.0, "lin"),
    "brick_height": (0.05, 10.0, "lin"),
    "offset": (0.0, 1.0, "lin"),
    "warp_width": (0.01, 1.0, "lin"),
    "width": (0.001, 10.0, "lin"),
    "radius": (0.0, 10.0, "lin"),
    "transition_width": (0.0, 5.0, "lin"),
    "below_at": (0.0, 1.0, "lin"),
    "below_to": (0.0, 1.0, "lin"),
    "above_at": (0.0, 1.0, "lin"),
    "above_to": (0.0, 1.0, "lin"),
    "dx": (-100.0, 100.0, "lin"),
    "dy": (-100.0, 100.0, "lin"),
    "sx": (0.01, 100.0, "lin"),
    "sy": (0.01, 100.0, "lin"),
    "degrees": (-360.0, 360.0, "lin"),
    "x": (-100.0, 100.0, "lin"),
    "y": (-100.0, 100.0, "lin"),
    "cx": (-100.0, 100.0, "lin"),
    "cy": (-100.0, 100.0, "lin"),
    "rx": (0.01, 100.0, "lin"),
    "ry": (0.01, 100.0, "lin"),
    "height": (-0.5, 0.5, "lin"),  # named-arg form (rare); usually positional
    "base": (1.1, 100.0, "lin"),
}
# Bounds for POSITIONAL literals by enclosing call name (channel values, shape
# coords, etc.). These are the valid ranges so the stored program stays sane
# (the engine would clamp a render anyway, but the serialized literals should
# not be degenerate). Arithmetic groups inherit the enclosing channel's call
# name, so an offset/amplitude inside `(150 + tone*50)` in `.basecolor(...)`
# picks up the basecolor range.
_CALL_BOUNDS: SDict[tuple[float, float, str]] = {
    "basecolor": (0.0, 255.0, "lin"),
    "roughness": (0.0, 1.0, "lin"),
    "metallic": (0.0, 1.0, "lin"),
    "sheen": (0.0, 1.0, "lin"),
    "coat": (0.0, 1.0, "lin"),
    "transmission": (0.0, 1.0, "lin"),
    "subsurface": (0.0, 1.0, "lin"),
    "anisotropy": (0.0, 1.0, "lin"),
    "ior": (1.0, 3.0, "lin"),
    # Relief is per-pixel-steep (dx ~ view/pixels ~ 0.02 for a 6-unit window at
    # 256px), so |height| > ~0.5 is an ~89deg+ cliff that reads as "popped out of
    # screen". Authored mortar-recess relief is |h| <= 0.25; allow 2x headroom.
    "height": (-0.5, 0.5, "lin"),
    "emissive": (0.0, 255.0, "lin"),
    "Layer": (0.0, 1.0, "lin"),  # the alpha mask argument
    "Threshold": (0.0, 1.0, "lin"),
    "Fill": (0.0, 5.0, "lin"),
    "Stroke": (0.001, 10.0, "lin"),  # width (>0) is the one positional numeric
}
# Per-POSITION bounds for calls whose positional args are heterogeneous (some
# constrained > 0, some not). Keyed by (call_name, zero-based position); a hit
# here takes precedence over the call-wide ``_CALL_BOUNDS`` entry so a
# width/height that the parser requires > 0 is never driven to 0 (which would
# make the polished program unparseable). Position counts only top-level
# positional args of the call (literals inside an arithmetic group belong to
# the group, not a new position).
_POS_BOUNDS: dict[tuple[str, int], tuple[float, float, str]] = {
    # Rect(x, y, width>0, height>0, radius>=0)
    ("Rect", 0): (-100.0, 100.0, "lin"),
    ("Rect", 1): (-100.0, 100.0, "lin"),
    ("Rect", 2): (0.001, 50.0, "lin"),
    ("Rect", 3): (0.001, 50.0, "lin"),
    ("Rect", 4): (0.0, 10.0, "lin"),
    # Ellipse(cx, cy, rx>0, ry>0)
    ("Ellipse", 0): (-100.0, 100.0, "lin"),
    ("Ellipse", 1): (-100.0, 100.0, "lin"),
    ("Ellipse", 2): (0.001, 50.0, "lin"),
    ("Ellipse", 3): (0.001, 50.0, "lin"),
}
_DEFAULT_POSITIONAL = (-1000.0, 1000.0, "lin")

_NUM_RE = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+")


@dataclass
class Tunable:
    """One numeric literal in the program text."""

    start: int
    end: int
    lo: float
    hi: float
    scale: str  # "lin" | "log"
    origin: str  # human-readable source, for the log

    def to_unit(self, v: float) -> float:
        """Map a real value to the [0, 1] search coordinate."""
        if self.scale == "log":
            return (math.log10(max(v, 1e-12)) - self.lo) / (self.hi - self.lo)
        return (v - self.lo) / (self.hi - self.lo)

    def from_unit(self, t: float) -> float:
        """Map a [0, 1] search coordinate back to a real value, clamped."""
        t = min(1.0, max(0.0, t))
        if self.scale == "log":
            return 10.0 ** (self.lo + t * (self.hi - self.lo))
        return self.lo + t * (self.hi - self.lo)


def _default_score_fn() -> Callable[[Sequence[ImgLike], str], list[float]]:
    """Build the MobileCLIP2-S2 verifier callable (lazy). Mirrors the selector in
    :mod:`matloom.engine.text2dsl` so selection and the CMA-ES fitness share one
    verifier load and one prompt template."""
    from matloom.metrics.clipscore import ClipScore

    clip = ClipScore.MobileClip2()
    return lambda imgs, txt: clip(imgs, [f"A photo of {txt}"]).scores[0].tolist()


_SEED_RE = re.compile(r"seed\s*=\s*(-?\d+)")


def _seed_variants(base_program: str, k: int, rng: np.random.Generator) -> list[str]:
    """``k`` variants of ``base_program`` with each distinct ``seed=N`` value
    redrawn to a fresh non-negative int (the first variant is the original).

    Expressions that share a seed in ``base_program`` (the LLM deliberately
    correlated two noises) stay correlated: they are remapped together to one
    new shared value. Expressions with distinct seeds are redrawn
    independently, so a canonical program — canonicalization bakes a distinct
    seed per fBm/Worley — gets a fully fresh, decorrelated realization per
    variant. The sweep thus samples the program's real N-D seed space rather
    than collapsing every noise to one shared seed (which would reintroduce
    the correlation the seed-pinning canonicalization removed).
    """
    if k <= 0 or not _SEED_RE.search(base_program):
        return [base_program]
    # Group existing seed values so authored correlation is preserved:
    # expressions sharing a value are remapped together to one new value.
    old_values = sorted({int(m.group(1)) for m in _SEED_RE.finditer(base_program)})
    variants = [base_program]
    for _ in range(k):
        remap = {o: int(rng.integers(0, 2**31)) for o in old_values}
        variants.append(
            _SEED_RE.sub(lambda m, r=remap: f"seed={r[int(m.group(1))]}", base_program)
        )
    return variants


def _bounds_for(
    call_name: str | None,
    arg_name: str | None,
    position: int | None = None,
    *,
    is_base_layer: bool = False,
) -> tuple[float, float, str] | None:
    """Bounds for a literal, or None to exclude it. ``call_name`` is the
    enclosing call (e.g. ``basecolor``, ``fBm``, ``Rect``); ``arg_name`` is the
    named arg if it was written as ``name=`` (else None for positional);
    ``position`` is the zero-based top-level positional index inside the call
    (None when the literal is nested inside an arithmetic group, so it inherits
    the call-wide bounds rather than a per-position one). ``is_base_layer``
    marks the bottom-most ``Layer(...)``: its alpha mask argument is structural
    (the opaque substrate every realistic material builds up from), so it is
    never tuned — see ``_extract_tunables``."""
    if call_name in _EXCLUDED_CALLS:
        return None
    # The bottom layer's alpha is the opaque substrate. The fixed-light preview
    # divides it out (it never reaches the verifier), so CMA-ES can only degrade
    # it with no fitness signal — and the see-through surface appears downstream.
    if is_base_layer and call_name == "Layer" and arg_name is None:
        return None
    if arg_name is not None:
        if arg_name in _EXCLUDED_ARGS:
            return None
        return _NAMED_BOUNDS.get(arg_name, _DEFAULT_POSITIONAL)
    if position is not None and (call_name, position) in _POS_BOUNDS:
        return _POS_BOUNDS[(call_name, position)]
    return _CALL_BOUNDS.get(call_name or "", _DEFAULT_POSITIONAL)


def _extract_tunables(program: str) -> list[Tunable]:
    """Find the tunable numeric literals in ``program`` (the LLM-authored text)
    with their text spans and bounds, in source order."""
    tunables: list[Tunable] = []
    # Each frame: [call_name, arg_name, pos_index, group_depth]. ``pos_index``
    # counts top-level positional args of the CALL (group_depth 0); a bare
    # arithmetic group inherits the enclosing call's name but sits at
    # group_depth 1, so literals inside it do not advance the call's position.
    call_stack: list[list[Any]] = []
    pending_call: str | None = None
    layer_count = 0  # how many Layer(...) calls seen so far; the FIRST is the base
    pending_is_base_layer = False
    i, n = 0, len(program)

    def top() -> list[Any]:
        return call_stack[-1] if call_stack else [None, None, 0, 0, False]

    while i < n:
        c = program[i]
        if c.isspace():
            i += 1
            continue
        if c == "(":
            # A function/method call uses the pending identifier; a bare
            # arithmetic group inherits the enclosing call name so an
            # offset/amplitude inside a channel expression picks up that
            # channel's bounds. A group sits one level deeper than its call, so
            # its literals do not advance the call's positional index.
            parent = top()
            parent_call = parent[0]
            parent_depth = parent[3] if call_stack else 0
            is_base = pending_is_base_layer if pending_call is not None else False
            call_stack.append(
                [
                    pending_call if pending_call is not None else parent_call,
                    None,
                    0,
                    0 if pending_call is not None else parent_depth + 1,
                    is_base,  # [4] is_base_layer: only the FIRST Layer(...) marks it
                ]
            )
            pending_call = None
            pending_is_base_layer = False
            i += 1
            continue
        if c == ")":
            if call_stack:
                call_stack.pop()
            i += 1
            continue
        if c == ",":
            # An arg name does not carry past a comma at the call's own level.
            if call_stack and call_stack[-1][3] == 0:
                call_stack[-1][1] = None
                call_stack[-1][2] += 1  # next top-level positional arg
            i += 1
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (program[j].isalnum() or program[j] == "_"):
                j += 1
            k = j
            while k < n and program[k].isspace():
                k += 1
            if k < n and program[k] == "(":
                pending_call = program[i:j]  # a call: its name applies at the next '('
                if pending_call == "Layer":
                    pending_is_base_layer = layer_count == 0
                    layer_count += 1
            elif k < n and program[k] == "=" and call_stack and call_stack[-1][3] == 0:
                # A named arg applies only at the call's own level (a `=` inside
                # an arithmetic group is a comparison, not a kwarg).
                call_stack[-1][1] = program[i:j]  # named arg for the upcoming value
            # else: a bare variable reference (e.g. `tone`) — not a literal.
            i = j
            continue
        if c in "=.+-*/":
            i += 1
            continue
        # A number, possibly with a unary leading '-' (the '-' was consumed above
        # as an operator char, so handle a negative literal by checking context).
        m = _NUM_RE.match(program, i)
        if not m:
            i += 1
            continue
        start, end = m.start(), m.end()
        # Negative literal: a '-' immediately before the number, preceded by an
        # open-paren / comma / '=' / start (i.e. unary, not binary subtraction).
        p = start - 1
        while p >= 0 and program[p].isspace():
            p -= 1
        if p >= 0 and program[p] == "-":
            q = p - 1
            while q >= 0 and program[q].isspace():
                q -= 1
            if q < 0 or program[q] in "(,=":
                start = p  # include the '-' in the span
        call_name, arg_name, pos, depth, is_base = top()
        # Positional bounds apply only to a literal at the call's own level
        # (group_depth 0); a literal inside an arithmetic group inherits the
        # call-wide bounds via _CALL_BOUNDS.
        position = pos if depth == 0 and arg_name is None else None
        bounds = _bounds_for(
            call_name, arg_name, position, is_base_layer=is_base and depth == 0
        )
        if bounds is not None:
            lo, hi, scale = bounds
            # Skip a literal whose authored value is outside its bounds: it is a
            # structural constant the LLM chose (e.g. a hash multiplier like
            # 43758.5453) rather than a tunable parameter, and clamping it would
            # corrupt the program. Keeping it fixed preserves idempotency at the
            # original theta (_apply_values leaves out-of-scope spans untouched).
            try:
                val = float(program[start:end])
            except ValueError:
                val = 0.0
            # Compare in REAL space: for a log-scale tunable lo/hi are log10
            # bounds, so the real-space extent is [10**lo, 10**hi]. Comparing the
            # raw literal against lo/hi directly would wrongly skip every
            # positive frequency (2.5 > max(-1, 1.7)) and leave it untunable.
            rlo, rhi = (10.0**lo, 10.0**hi) if scale == "log" else (lo, hi)
            if val < min(rlo, rhi) or val > max(rlo, rhi):
                i = end
                continue
            origin = f"{call_name or 'positional'}" + (
                f".{arg_name}" if arg_name else ""
            )
            tunables.append(Tunable(start, end, lo, hi, scale, origin))
        i = end
    return tunables


def _fmt_num(v: float) -> str:
    """Compact numeric formatting for reinsertion into the program text."""
    if v == 0.0:
        return "0"
    av = abs(v)
    if av >= 1e6 or av < 1e-4:
        return f"{v:.6e}"
    if v == int(v) and av < 1e6:
        return f"{int(v)}"
    return f"{v:.6g}"


def _apply_values(base_program: str, theta: Sequence[float]) -> str:
    """Materialize a candidate: replace each tunable literal in
    ``base_program`` with the value mapped from ``theta`` (aligned with
    :func:`_extract_tunables` order). Spans are replaced right-to-left so earlier
    positions stay valid. The result is re-parsed by the caller for rendering."""
    tunables = _extract_tunables(base_program)
    if len(tunables) != len(theta):
        raise ValueError(f"theta length {len(theta)} != tunables {len(tunables)}")
    out = base_program
    for t, x in sorted(zip(tunables, theta), key=lambda px: px[0].start, reverse=True):
        v = t.from_unit(float(x))
        out = out[: t.start] + _fmt_num(v) + out[t.end :]
    return out


def _cma_one_program(
    base_program: str,
    objective: Callable[[Sequence[float]], float],
    *,
    max_evals: int,
    sigma0: float,
    seed: int,
) -> tuple[str, SDict[Any]]:
    import cma

    log: SDict[Any] = {}
    tunables = _extract_tunables(base_program)
    log["n_tunables"] = len(tunables)
    if len(tunables) == 0:
        return base_program, log

    x0: list[float] = [
        min(1.0, max(0.0, t.to_unit(float(base_program[t.start : t.end]))))
        for t in tunables
    ]
    opts = {
        "maxfevals": max_evals,
        "bounds": [[0.0] * len(tunables), [1.0] * len(tunables)],
        "verbose": -9,
        "seed": seed,
    }
    try:
        # eval_initial_x is a fmin2 keyword, NOT an opts entry (opts rejects it).
        # It scores x0 (the as-authored params) and seeds es.best with it, so the
        # returned program / logged fbest never silently drop below the original:
        # without it CMA-ES only ever samples offspring, and a start point that is
        # already a local peak (a well-authored program) yields a "best" below the
        # original even though that original was handed in as x0.
        xopt, es = cma.fmin2(objective, x0, sigma0, opts, eval_initial_x=True)
    except Exception as e:  # noqa: BLE001 - CMA-ES failure -> keep the variant
        logger.error(f"CMA-ES failed: {e}")
        return base_program, log

    log["n_evals"] = es.countevals
    log["score"] = -es.result.fbest
    polished = _apply_values(base_program, xopt)
    return polished, log


# Fitness floor for a candidate that fails to render/parse: far below any real
# CLIP score (>= 0 after the *100 scaling), and finite so record.json stays
# standard JSON and the CMA-ES objective stays finite (cma is rank-based, but a
# +inf objective disables its ftol termination).
_BROKEN_SCORE = -1e6

# Process-pool defaults for the seed sweep. The 4-worker count measured
# 2.4-2.6x end-to-end on the sweep before the preview-kernel rewrite; renders
# are far faster now, so spawn + first-worker numba cache warmup (~1-1.7s)
# dominates short sweeps and the end-to-end benefit shrinks accordingly.
# Per-worker numba threads are left at their default (pinning them measured
# no better). Batches smaller than _POOL_MIN_BATCH stay in-process: the CMA-ES
# objective is evaluated one solution at a time, so a pool would only add
# round-trip latency there.
_DEFAULT_POOL_WORKERS = 4
_POOL_MIN_BATCH = 2
# Operator override consulted only when polish_program's parallel_renders is
# None (explicit param > env; the library default stays off either way).
_POOL_ENV = "MATLOOM_POLISH_WORKERS"


def _render_program(
    program_text: str, width: int | None = None, height: int | None = None
) -> tuple[np.ndarray | None, str | None]:
    """Render one variant in a pool worker. Must stay module-level (pickled by
    reference) and engine-only (spawned workers re-import this module; no
    torch/Blender). ``width``/``height`` default to the preview's own defaults,
    which is exactly the call the sequential path makes, so the bytes are
    identical; production passes only program texts, so pooled renders always
    run at those defaults (the size parameters exist for the tests). Returns
    ``(img, None)`` or ``(None, str(e))`` so a degenerate candidate maps to the
    fitness floor with the same error text as the sequential path's logged
    ``{e}``."""
    try:
        material = LayeredMaterial.deserialize(program_text)
        if width is None or height is None:
            return fixed_light_preview(material), None
        return fixed_light_preview(material, width, height), None
    except Exception as e:  # noqa: BLE001 - keep the batch alive
        return None, str(e)


def _resolve_pool_workers(parallel_renders: int | None, env: str | None = None) -> int:
    """Resolve the render-pool worker count for a polish run (0 = sequential).

    Pure precedence rule (unit-tested): an explicit ``parallel_renders`` beats
    the ``MATLOOM_POLISH_WORKERS`` env var, which beats "off". ``env`` injects
    the env value for tests; ``None`` reads the real environment. The env
    accepts a worker count, ``auto`` (the measured default, clamped to the CPU
    count), or ``0``/``off``/``false``/``no`` (sequential); anything unparsable
    is off. The library default is OFF — no caller spawns processes unless it
    opts in, so parent-side monkeypatched renders (the tests') are always the
    ones used.
    """
    if parallel_renders is not None:
        return max(0, int(parallel_renders))
    raw = (os.environ.get(_POOL_ENV, "") if env is None else env).strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return 0
    if raw in ("auto", "on", "true", "yes"):
        return _default_pool_workers()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _default_pool_workers() -> int:
    """Worker count for call sites that enable the pool by default: the
    measured 4-worker optimum, clamped to the CPU count (a 1-core box gets a
    1-worker pool; such callers expose their own 0-opt-out)."""
    return min(_DEFAULT_POOL_WORKERS, os.cpu_count() or 1)


def polish_program(
    prompt: str,
    program: str,
    *,
    seed_sweep: int = 1000,
    max_cma_evals: int = 0,
    cma_sigma0: float = 0.25,
    seed: int | None = None,
    score_fn: Callable[[Sequence[ImgLike], str], list[float]] | None = None,
    parallel_renders: int | None = 0,
) -> tuple[str, SDict[Any]]:
    """Polish ``program`` against a verifier and return ``(best_program, log)``.

    - **Seed sweep (primary):** ``seed_sweep`` re-seeded variants of ``program``
      are scored at the as-authored params (cheap, no CMA-ES). Shared seeds stay
      correlated; distinct seeds redraw independently (see :func:`_seed_variants`).
    - **CMA-ES (secondary):** when ``max_cma_evals > 0``, the continuous literals
      of **every** seed variant are optimized against the verifier.

    The returned program is the highest-scoring one across the sweep and all
    CMA-ES polishes (accept-if-better via the ``max``), so the polish never does
    worse on the fitness than the as-authored program. ``score_fn(images, prompt)
    -> list[float]`` is the batched verifier (one score per ``(H, W, 3)`` uint8
    image); when omitted, MobileCLIP2-S2 is loaded lazily. ``max_cma_evals == 0``
    runs the seed sweep only; ``seed_sweep == 0`` (or no ``seed=`` literals)
    collapses the sweep to the original. ``cma`` is lazy-imported; a CMA-ES
    failure or a variant with no tunable literals leaves that variant in place
    (logged under each candidate's ``cma``).

    ``parallel_renders`` opts the seed sweep's renders into a spawned
    ``ProcessPoolExecutor``: ``0`` (default) is today's sequential path
    byte-for-byte, ``n > 0`` renders the sweep across ``n`` spawn workers, and
    ``None`` defers to ``MATLOOM_POLISH_WORKERS`` (param > env). The resolved
    count is logged under ``config["parallel_renders"]``. Renders are
    pure functions of the program text and ``executor.map`` yields in submission
    order, so the images, the scores, and the accept-if-better tie-breaks are
    identical to sequential; scoring, the sweep RNG, and selection never leave
    the parent. A batch smaller than 2 (the CMA-ES per-solution calls) always
    renders in-process. An executor-level failure (``BrokenProcessPool``: worker
    segfault/OOM kill) logs a warning and falls back to sequential rendering for
    the rest of the run, so a dying worker cannot abort the sweep.

    Spawn-safety: the ``spawn`` start method re-imports the calling module as
    ``__mp_main__``, so an external caller enabling the pool from a script must
    guard its entry point with ``if __name__ == "__main__":`` (the repo's
    console scripts already do). Spawned workers also JIT the numba kernels from
    the on-disk cache on first use — a cold cache costs roughly 1-2s per worker,
    amortized over the sweep. When ``parallel_renders`` is ``None``, the
    ``MATLOOM_POLISH_WORKERS`` env var (a worker count, ``auto``, or ``0``) is
    consulted instead (param > env).
    """
    rng = np.random.default_rng(seed)
    fit = score_fn or _default_score_fn()
    workers = _resolve_pool_workers(parallel_renders)
    log: SDict[Any] = {
        "config": {
            "seed_sweep": seed_sweep,
            "max_cma_evals": max_cma_evals,
            "cma_sigma0": cma_sigma0,
            "seed": seed,
            "parallel_renders": workers,
        },
        "evals": 0,  # Every fitness call
    }
    # The render pool is created lazily on the first batch big enough to profit
    # and shut down before returning; once an executor-level failure disables
    # it, pool_live stays False for the rest of the run.
    executor: ProcessPoolExecutor | None = None
    pool_live = workers > 0

    def fitness(programs: Sequence[str]) -> list[float]:
        nonlocal executor, pool_live
        log["evals"] += len(programs)
        # A single degenerate candidate (a CMA-ES theta that breaks the parse, or
        # a preview edge case) must not abort the batch: the seed-sweep batch has
        # no surrounding try/except, and a CMA-ES eval failure would otherwise
        # kill that variant's whole run. Score it at a finite floor well below
        # any real CLIP score so it always loses, and skip the verifier call on
        # it. A finite sentinel (not -inf) keeps record.json standard JSON and the
        # CMA-ES objective finite (cma is rank-based, but +inf disables its ftol
        # termination).
        scores: list[float] = [_BROKEN_SCORE for _ in programs]
        imgs: list[np.ndarray] = []
        valid: list[int] = []
        pooled = False
        if pool_live and len(programs) >= _POOL_MIN_BATCH:
            if executor is None:
                # One-time: the spawn start method re-imports the calling
                # module in every worker, so an unguarded external entry point
                # would re-run its top-level side effects per worker.
                logger.info(
                    f"Spawning {workers} render workers (spawn); external "
                    "entry points must be __main__-guarded"
                )
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            try:
                with progress_bar() as pbar:
                    pbar_id = pbar.add_task("Rendering", total=len(programs))
                    # executor.map yields in submission order, so imgs/valid and
                    # the scores fit() receives are element-for-element what the
                    # sequential loop produces. A per-candidate failure is an
                    # error payload from the worker, never an exception here.
                    for i, (img, err) in enumerate(
                        executor.map(_render_program, programs)
                    ):
                        if img is None:
                            logger.error(f"Degenerate candidate scored at floor: {err}")
                        else:
                            imgs.append(img)
                            valid.append(i)
                        pbar.advance(pbar_id)
                pooled = True
            except Exception as e:  # noqa: BLE001 - executor-level, not per-candidate
                # BrokenProcessPool (worker segfault / OOM kill) surfaces here,
                # not inside the worker. The sequential path structurally cannot
                # abort the sweep, so neither may the pooled one: disable the
                # pool for the rest of the run and re-render this batch
                # in-process (renders are pure functions of the program text, so
                # the redone results are identical).
                logger.warning(f"Render pool failed; rendering sequentially: {e}")
                pool_live = False
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception as e2:  # noqa: BLE001 - best-effort teardown
                    logger.debug(f"Failed pool also failed to shut down: {e2}")
                executor = None
                imgs.clear()
                valid.clear()
        if not pooled:
            with progress_bar() as pbar:
                pbar_id = pbar.add_task("Rendering", total=len(programs))
                for i, prog in enumerate(programs):
                    try:
                        imgs.append(
                            fixed_light_preview(LayeredMaterial.deserialize(prog))
                        )
                        valid.append(i)
                    except Exception as e:  # noqa: BLE001 - keep the batch alive
                        logger.error(f"Degenerate candidate scored at floor: {e}")
                    pbar.advance(pbar_id)
        if len(imgs) > 0:
            for i, s in zip(valid, fit(imgs, prompt)):
                scores[i] = s
        return scores

    try:
        logger.info(f"Sweeping {seed_sweep} seed variants")
        variants = _seed_variants(program, seed_sweep, rng)
        candidates = [
            {"trajectory": [v], "scores": [s]}
            for v, s in zip(variants, fitness(variants))
        ]
        if max_cma_evals > 0:
            logger.info("Polishing candidates with CMA-ES")
            with progress_bar() as pbar:
                pbar_id = pbar.add_task("CMA-ES", total=len(candidates))
                for cand in candidates:
                    base_program = cand["trajectory"][0]
                    cma_polished, cma_log = _cma_one_program(
                        base_program,
                        lambda theta, bp=base_program: (
                            -fitness([_apply_values(bp, theta)])[0]
                        ),
                        max_evals=max_cma_evals,
                        sigma0=cma_sigma0,
                        seed=int(rng.integers(0, 2**31)),
                    )
                    cand["trajectory"].append(cma_polished)
                    cand["scores"].append(cma_log.get("score", 0))
                    cand["cma"] = cma_log
                    pbar.advance(pbar_id)
                pbar.remove_task(pbar_id)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    final_program, final_score = max(
        [
            (prog, score)
            for cand in candidates
            for prog, score in zip(cand["trajectory"], cand["scores"])
        ],
        key=lambda x: x[1],
    )
    log["prompt"] = prompt
    log["initial_score"] = candidates[0]["scores"][0]
    log["initial_program"] = program
    log["final_score"] = final_score
    log["final_program"] = final_program
    log["candidates"] = candidates
    return final_program, log
