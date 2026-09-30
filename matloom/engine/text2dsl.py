"""
text2dsl.py — an LLM-powered natural-language → layered-material DSL pipeline.

Given a textual prompt (e.g. "weathered red brick wall"), this module asks an
LLM to author a program in the layered-material DSL (the same string format
:class:`~matloom.engine.main.LayeredMaterial` round-trips), validates that it
parses, and returns it. Two optional, independently-toggleable stages make it
useful for both production and ablation studies:

* **Few-shot** (``few_shot=True``, default): a curated set of prompt→program
  examples is prepended to the system prompt. Disable it for a zero-shot
  baseline.
* **Critique / refinement** (``refine > 0`` with at least one of
  ``critic_render``, ``critic_code``, or ``critic_diagnostics``): a critic
  reviews the generated program and its feedback is fed back into the same
  generation conversation, which revises the program. Repeats ``refine`` times.
  Three independent signals select what the critic sees: ``critic_render``
  renders the program with the no-Blender :func:`matloom.engine.preview.fixed_light_preview`
  for a vision model; ``critic_code`` shows the critic the DSL source so its
  suggestions can be line-targeted edits (no render needed);
  ``critic_diagnostics`` appends objective appearance scalars from the unlit
  channel maps. The three are orthogonal, so all seven non-empty combinations
  are selectable (e.g. ``critic_code`` + ``critic_diagnostics`` is a grounded
  code reviewer with no image render or vision model).

The LLM client is :class:`matloom.utils.llm.Llm` (an OpenAI-compatible Chat
Completions wrapper that works against any ``base_url``).
Generation emits the program inside ``<material>...</material>`` tags;
the critique uses a :class:`JsonResponseModel` for structured output.
A parse failure is sent back to the model — with the
parser's error message — for up to ``parse_retries`` corrective attempts.

CLI: ``matloom-text2dsl "<prompt>"`` (console script) /
``python -m matloom.engine.text2dsl``. It prints the program and, with
``--output-dir``, also saves a timestamped YAML record of the prompt + program.
"""

import argparse
import math
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, validate_call

from matloom import logger
from matloom.utils.dtypes import ImgLike, PathLike, SDict
from matloom.utils.llm import Conversation, CoStar, JsonResponseModel, Llm
from matloom.utils.progress import progress_bar

DEFAULT_MODEL = "openai/gpt-5.6-luna"
# Local retries for a *transport* hiccup (an empty "response is None" from the
# provider, common under high concurrency). These retry the same request in
# place with backoff; they are distinct from the conversation-level program
# parse-retry loop (``parse_retries``), which re-prompts with the parser error.
_TRANSPORT_RETRIES = 3
# The program is returned wrapped in these tags so it is trivially extractable
# from any surrounding prose the model emits.
_MATERIAL_RE = re.compile(r"<material>\s*(.*?)\s*</material>", re.DOTALL)


class Text2DslError(RuntimeError):
    """Raised when the pipeline cannot obtain a parseable program."""


# |==========================================================================|
# |   System prompt (the DSL specification handed to the generation model)   |
# |==========================================================================|

# The DSL reference lives in the SYSTEM prompt: it is stable across a whole
# conversation (and across refinement turns), so a provider can cache it, and it
# frames the model's entire task. Kept deliberately concise — the full grammar
# and math are in README.md; this is the operational subset a model needs.
_DSL_ROLE = """\
You are an expert technical artist.
You write **layered-material DSL** programs: a compact, declarative language that compiles deterministically into physically based material maps.
Given a natural-language description of a material, output a program that reproduces it.
"""

# The DSL spec (everything after the role intro). Shared with the informed
# critic, which pairs it with a CRITIC role (not this generator role) so it is
# not told to "write programs" while judging them.
_DSL_SPEC = """\
# Program structure
A program is an optional `View`, then zero or more `Define`s, then exactly one `Material`, in that order:

<structure>
View(x1, y1, x2, y2)            # the sampled window (optional, default 0, 0, 1, 1; see "Coordinates & View" below)
Define(name, <expr>)            # reusable named expression (optional, repeatable; see "Define" below)
Material(
  Layer(<alpha-mask>)           # bottom-most layer FIRST; alpha 0..1 (default 1 = fully covering)
    .basecolor(<r>, <g>, <b>)   # each 0..255      (default 0, 0, 0)   base color (diffuse albedo of the layer)
    .metallic(<m>)              # 0..1             (default 0)         0 = dielectric, 1 = metal
    .roughness(<r>)             # 0..1             (default 0)         0 = mirror, 1 = fully matte
    .sheen(<s>)                 # 0..1             (default 0)         cloth/velvet/fabric fuzz
    .coat(<c>)                  # 0..1             (default 0)         clearcoat (lacquer, car paint, glaze)
    .transmission(<t>)          # 0..1             (default 0)         glass / ice / translucency
    .ior(<n>)                   # >= 1             (default 1.5)       refractive index
    .subsurface(<ss>)           # 0..1             (default 0)         skin/wax/jade/marble (scatter color = basecolor)
    .anisotropy(<a>)            # 0..1             (default 0)         brushed metal / satin
    .emissive(<r>, <g>, <b>, <strength>)   # RGB 0..255, strength >= 0  (default 0, 0, 0, 0)  glow
    .height(<h>),               # coordinate units (default 0); any real value incl. negative; drives relief
  ...                           # more layers, each composited OVER the ones below
)
</structure>

Every channel (the alpha mask included) is a scalar EXPRESSION over the sample point — a literal number, a coordinate term, a `Define`d name, or any nesting of the functions below.
RGB args are expressions too, so `.basecolor((150 + noise * 50), 60, 45)` is valid.
Each layer method is optional and chained with a leading dot; an omitted method takes the default shown above.
Channels are auto-clamped to the ranges above (e.g. metallic is clamped to [0, 1], ior to [>= 1]); height is NOT clamped.
Layers composite with the over-operator: an upper layer with alpha 1 fully hides those below it in color/metallic/roughness/sheen/coat/transmission/ior/subsurface/anisotropy; partial alpha blends them.
Height is the exception — see point 2 below.

# Coordinates & View
The material is a 2D field defined on the entire XY plane (origin at (0, 0), x rightwards, y upwards).
Every expression is evaluated at sample points on that plane; positions, sizes, noise frequencies, and heights all share one unit — the coordinate unit.
`View(x1, y1, x2, y2)` (default 0, 0, 1, 1) declares the rectangle of that plane that is sampled into the output maps: (x1, y1) is the bottom-left corner, (x2, y2) the top-right.
It sets the spatial extent — hence the apparent scale — of the rendered swatch; it does not change the material, only which region is sampled.
Choose the window to fit the pattern: a brick wall wants several bricks across (e.g. View(0, 0, 6, 6) with 1-unit-wide bricks); a single tile wants View(0, 0, 1, 1).
The window maps onto the whole output image regardless of aspect, so with a square render a non-square window stretches the pattern.

# Define
`Define(name, <expr>)` binds `name` to an expression.
Reference it later by writing the BARE name (e.g. `tone`, NOT `Ref(tone)`) anywhere an expression is expected — in a channel, a layer mask, or a later `Define`.
A `Define` may reference earlier `Define`s but not itself or later ones.
A `Define` may not bind a bare shape (`Path`/`Rect`/`Ellipse`): a shape constructor may only appear inline inside `Fill(...)`/`Stroke(...)`.
To stroke one outline at several widths, repeat the constructor inside each `Stroke(...)`.
Use it to author a pattern/noise once and reuse it across the mask, color, and height of a layer (the canonical idiom).

# Expressions
Every channel is one expression.
The full vocabulary (function name — params with `=default`; required params have no default):

## Coordinates & constants
- `X()`, `Y()` — the sample point's coordinates on the plane; across the output maps, X() spans [x1, x2] left-to-right and Y() spans [y1, y2] bottom-to-top (the View window).
- `pi` (3.14159...), `e` (2.71828...) — bare value constants (no parentheses).

## Operators (write these directly; lowest→highest precedence: `+ -`, `* /`, unary `-`, `**`)
- `a + b`, `a - b`, `a * b`, `a / b`, `a ** b` (power, right-associative), unary `-a`.
- Function spellings also exist: `Add(left, right)`, `Sub(minuend, subtrahend)`, `Mul(left, right)`, `Div(dividend, divisor)`, `Pow(base, exponent)` — prefer the operators.

## Math (each takes one expression unless noted)
- `Abs(expr)`, `Sqrt(expr)`, `Floor(expr)` (round down), `Ceil(expr)` (round up).
- `Sin(expr)`, `Cos(expr)` — argument in radians.
- `Log(expr, base=e)` — logarithm; omit `base` for natural log.
- `Min(expr1, expr2)`, `Max(expr1, expr2)` — element-wise min / max.

## Threshold (remap / clamp a value, with an optional smoothstep ramp)
- `Threshold(expr, below_at=None, below_to=None, above_at=None, above_to=None, transition_width=0)`
  - `below_at`: clamp inputs below this value (None disables the lower clamp); `below_to`: the value they take (defaults to `below_at`).
  - `above_at`: clamp inputs above this value (None disables the upper clamp); `above_to`: the value they take (defaults to `above_at`).
  - `transition_width`: 0 = hard step; > 0 = smoothstep ramp of that width around each cutoff.
  - Idioms: clamp to [0,1] → `Threshold(x, below_at=0, above_at=1)`; binarize a noise → `below_at=0.5, below_to=0, above_at=0.5, above_to=1`.

## Noise (organic variation; output is [-1, 1] unless `to_01=True`)
- `fBm(octaves=6, lacunarity=2.0, gain=0.5, base_freq=1.0, base_freq_x=None, base_freq_y=None, to_01=False, seed=None)` — fractal Brownian motion (layered simplex).
  `octaves` = detail layers; `lacunarity` = freq mult/octave; `gain` = amplitude mult/octave.
- `Worley(distance="euclidean", combination="F1", base_freq=1.0, base_freq_x=None, base_freq_y=None, to_01=False, seed=None)` — cellular noise.
  `distance`: "euclidean" (round) / "manhattan" (diamond) / "chebyshev" (square); `combination`: "F1" (cells) / "F2" / "F2-F1" (sharp ridges/cracks) / "F2+F1" (blobby).
- For both: `base_freq` is the isotropic frequency (feature size scales as 1/base_freq in coordinate units); `base_freq_x`/`base_freq_y` override one axis to stretch the pattern.
  `to_01=True` maps the output into [0,1] (use it whenever the noise feeds a color, weight, or mask). `seed` is random when omitted.

## Patterns (deterministic lattices; output 1 on the feature, 0 between — i.e. ready-made masks)
- `Bricks(brick_width=1.0, brick_height=0.5, offset=0.5, mortar=0.05, axis="row", feather=0)` — running-bond brick lattice (1 = brick, 0 = mortar).
  `brick_width`/`brick_height` = cell pitch (> 0); `offset` = per-row/col shift as a cell fraction (0.5 = running bond, 0 = stacked); `mortar` = gap width in coordinate units (>= 0); `axis`: "row" or "column" (which direction is offset); `feather` = soft edge in coordinate units.
- `Weave(over=1, under=1, shift=1, base_freq=8.0, base_freq_x=None, base_freq_y=None, warp_width=0.5, feather=0)` — woven over/under thread lattice (1 = on-top thread, 0 = gap).
  `over`/`under` = cells a thread floats over/under (1/1 plain, 2/1 twill); `shift` = per-row diagonal step (e.g. shift=2 → satin); `base_freq`/`base_freq_x`/`base_freq_y` = thread crossings per unit (per axis); `warp_width` = thread width as a cell fraction in (0, 1]; `feather` = soft thread edge.

## Shapes (an outline; NOT a mask by itself — wrap in `Fill` or `Stroke` to get a 0/1 field)
- `Rect(x, y, width, height, radius=0)` — bottom-left corner (x, y); `width`/`height` > 0; `radius` rounds corners (clamped to min(width,height)/2).
- `Ellipse(cx, cy, rx, ry)` — centre (cx, cy); radii `rx`, `ry` > 0.
- `Path(startX, startY, <segment>, <segment>, ...)` — polyline/curve from (startX, startY); one or more trailing segments:
  `LineTo(x, y)` (straight) or `CubicTo(c1x, c1y, c2x, c2y, x, y)` (cubic Bézier). Used inside Fill/Stroke.
- `Fill(shape, mode="NonZero", feather=0)` — paint the interior (1 inside, 0 outside).
  `mode`: "NonZero" or "EvenOdd" (the fill rule for self-intersecting paths); `feather` = soft edge in coordinate units.
- `Stroke(shape, width, feather=0)` — paint a band of `width` (> 0, coordinate units) centered on the boundary; `feather` = soft edge.

## Transforms (reposition any expression about the origin)
- `Translate(expr, dx=0, dy=0)` — shift by (dx, dy).
- `Scale(expr, sx=1, sy=1)` — scale about the origin (factors non-zero; > 1 enlarges the sampled feature).
- `Rotate(expr, degrees=0)` — rotate counter-clockwise about the origin.

## Combining masks (they are ordinary [0,1] fields, so compose with arithmetic)
- `Max(a, b)` = union, `Min(a, b)` = intersection, `1 - s` = invert, `a * (1 - b)` = subtract `b` from `a`.

# Five things that surprise generators (read before emitting)
1. A multi-tone surface (e.g. gray bricks, red mortar) must encode the visible pattern in the **alpha MASK**, not in height: put the pattern on an upper layer's mask and let a substrate layer show through the gaps.
   Driving only the height with a pattern leaves the color flat.
2. Height composites by `max` over *covered* layers, independent of layer order; alpha only gates whether a layer contributes its height.
   Color (order-dependent) and relief (order-independent) can therefore disagree.
3. Only the *variation* of height is visible (normal + AO are derived from it).
   A constant or purely linear height is flat — drive height from a pattern/noise to get real relief.
   Type absolute differences in coordinate units.
4. A bare shape (`Rect`/`Ellipse`/`Path`) is NOT a usable field — its constructor must be written inline inside `Fill(...)` or `Stroke(...)` before it can drive a mask or channel (a shape cannot be `Define`d and referenced by name).
   Everything else (noise, patterns, the filled/stroked result) is already an expression.
5. Mind the *scale*: noise frequencies and shape/pattern sizes are in the View's coordinate units, so they must match the `View(...)` window (a `base_freq` of 1 over a `View(0,0,6,6)` gives only one feature across the whole surface).
   Ranges are enforced for you (channels are auto-clamped to the table above; height is the only unclamped one), but author values within range anyway — e.g. height in absolute coordinate-unit differences, not 0..1.
"""

# The full reference (generator role + DSL spec), used by the generator.
_DSL_REFERENCE = _DSL_ROLE + "\n\n" + _DSL_SPEC

_OUTPUT_INSTRUCTIONS = """\
# Output format
Think artistically and thoroughly, then output the complete program wrapped in a single pair of tags:

<material>
View(...)
Define(...)
Material(
  ...
)
</material>

Output nothing after the closing </material> tag.
The program must parse and round-trip; prefer a small number of well-chosen layers over many redundant ones.
"""

_ORGANIC_PLAYBOOK = """\
# Organic texture playbook
Natural organic surfaces are mostly continuous variation (grain, tone, relief) with occasional discrete features. Author them with the idioms below, all in the existing vocabulary.

- Multi-octave layering. Stack two `fBm` Defines at different scales: a low-`octaves`, low-`base_freq` field for broad form (e.g. `fBm(octaves=3, base_freq=2, to_01=True)`) and a high-`octaves`, high-`base_freq` field for fine grain. Reuse each wherever its scale is needed.
- Anisotropic striation. For directional grain (tree bark, wood, brushed metal), split `fBm`/`Worley` with `base_freq_x`/`base_freq_y`: a high frequency on the across-grain axis and a low one on the along-grain axis elongates the features. Vertical bark ridges over a `View(0, 0, 2, 2)` read as `fBm(octaves=5, base_freq_x=18, base_freq_y=2, to_01=True)` (thin across X, long along Y).
- Coordinate-domain warping. `fBm` and `Worley` sample the plane internally and take no coordinate argument, so they cannot be warped directly; instead, add a noise to a coordinate term before a pattern function (`Sin`/`Cos`) to distort it. `Define(warp, fBm(octaves=4, base_freq=1.5, to_01=True, seed=5))` then `Define(band, Sin(((Y() + (warp * 0.7)) * 12)))` turns regular banding into turbulent stratification. Drive `.height(...)` from the same warped pattern so relief follows the distortion.
- Micro-gloss grain on roughness. Add a high-frequency, low-amplitude `fBm` only to `.roughness(...)`, e.g. `.roughness((0.6 + (grain * 0.25)))` with `grain = fBm(octaves=5, base_freq=45, to_01=True)`. Color is untouched; only gloss varies, giving skin/bark/stone microstructure without flattening the tone.
- Substrate-matrix-first for discrete-on-continuous (pebbles on sand, rust flakes on metal, leaves on soil). This is "surprise #1" applied to organic textures: put the continuous material in a bottom `Layer(1)` (alpha 1, fBm-driven color/roughness/height), then add the coarse discrete features as upper layers whose masks let the substrate show through between them. Build those masks with `Threshold(Worley(..., to_01=True), below_at=..., below_to=1, above_at=..., above_to=0)` for round islands, or a thresholded `fBm` for amorphous blobs. Never make one layer do both substrate and features.
"""


# Six curated prompt → program pairs chosen to teach the *essence* of the DSL,
# not just its syntax. They deliberately span 1→4 layers and a broad slice of the
# expression vocabulary: nested transcendental veining; the pattern-in-the-mask
# idiom; set-operator mask algebra (`1 - rust`, `scratch * paint`, `smallMask *
# gap`); expression-driven color (RGB args are expressions, not just constants);
# shapes + transforms (`Fill`/`Stroke`, `Ellipse`/`Path`/`CubicTo`, `Translate`/
# `Scale`/`Rotate`); fBm/Worley relief plus the `Max(0, r - dist)` Worley
# distance-field dome; the Floor-cell idiom for a plane-filling pattern that
# continues under any View; and the OpenPBR channels in context (transmission/
# ior, anisotropy, coat, subsurface, emissive). Every program is asserted to
# parse + round-trip by tests/engine/test_text2dsl.py, so they track the live DSL.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    # 1 layer, deeply nested expressions: turbulent veins drive color, a hard
    # threshold, AND relief on one layer; subsurface + coat + low roughness.
    (
        "Polished green marble",
        """\
View(0, 0, 2, 2)
Define(turb, fBm(octaves=5, base_freq=3, to_01=True, seed=4))
Define(veins, Threshold(Abs(Sin(((X() + (turb * 2)) * pi))), below_at=0.82, below_to=1, above_at=0.97, above_to=0))
Material(
  Layer(1)
    .basecolor((45 + (veins * 35)), (95 + (veins * 55)), (65 + (veins * 35)))
    .roughness(0.15)
    .coat(0.8)
    .subsurface(0.6)
    .height((veins * 0.012))
)
""",
    ),
    # 2 layers: a transmissive dielectric substrate (transmission + ior) with a
    # Stroke(Rect) frame masked on top (anisotropic brushed metal).
    (
        "Frosted glass panel in a brushed steel frame",
        """\
Define(frame, Stroke(Rect(0.08, 0.08, 0.84, 0.84, 0.04), 0.06, feather=0.005))
Material(
  Layer(1)
    .basecolor(220, 232, 238)
    .roughness(0.4)
    .transmission(0.92)
    .ior(1.5),
  Layer(frame)
    .basecolor(180, 182, 188)
    .metallic(1)
    .roughness(0.35)
    .anisotropy(0.8)
    .height((frame * 0.02))
)
""",
    ),
    # 3 layers, the canonical pattern-in-the-MASK idiom: mortar substrate shows
    # through a Bricks mask; a noise Define varies brick color/roughness/height;
    # a partial-alpha grime layer weathers the top.
    (
        "Weathered red brick wall with pale mortar",
        """\
View(0, 0, 6, 6)
Define(brick, Bricks(brick_width=1, brick_height=0.45, mortar=0.06, feather=0.005))
Define(tone, fBm(octaves=4, base_freq=2.5, to_01=True, seed=7))
Define(grime, Threshold(fBm(octaves=5, base_freq=1.5, to_01=True, seed=15), below_at=0.55, below_to=0, above_at=0.8, above_to=0.6))
Material(
  Layer(1)
    .basecolor(175, 170, 160)
    .roughness(0.95)
    .height((fBm(octaves=3, base_freq=10, to_01=True, seed=2) * 0.02)),
  Layer(brick)
    .basecolor((150 + (tone * 50)), (55 + (tone * 25)), (45 + (tone * 15)))
    .roughness((0.7 + (tone * 0.2)))
    .height((0.04 + (tone * 0.05))),
  Layer(grime)
    .basecolor(60, 55, 50)
    .roughness(1)
)
""",
    ),
    # 4 layers, set-operator mask algebra: bare steel, paint where there is NO
    # rust (1 - rust), rust patches, and scratches only on painted area
    # (scratch * paint) that re-expose the metal.
    (
        "Chipped blue paint over rusted steel",
        """\
View(0, 0, 3, 3)
Define(rust, Threshold(fBm(octaves=5, base_freq=4, to_01=True, seed=9), below_at=0.63, below_to=0, above_at=0.7, above_to=1))
Define(paint, (1 - rust))
Define(scratch, Threshold(Worley(base_freq=6, to_01=True, seed=3), below_at=0.02, below_to=1, above_at=0.06, above_to=0))
Material(
  Layer(1)
    .basecolor(120, 122, 128)
    .metallic(1)
    .roughness(0.4),
  Layer(paint)
    .basecolor(40, 90, 140)
    .roughness(0.35)
    .coat(0.6)
    .height((paint * 0.01)),
  Layer(rust)
    .basecolor((110 + (rust * 40)), (55 + (rust * 20)), (32 + (rust * 12)))
    .roughness(0.95)
    .height((rust * 0.03)),
  Layer((scratch * paint))
    .basecolor(150, 152, 158)
    .metallic(1)
    .roughness(0.3)
)
""",
    ),
    # 3 layers, substrate-matrix-first: a fine fBm sand substrate (alpha 1) with
    # two scales of Worley pebbles as upper layers whose thresholded masks let the
    # sand show between them (small pebbles sit only in the gaps via
    # `smallMask * gap`). Pebble relief is a `Max(0, r - Worley)` distance-field
    # dome — not the flat mask — so stones round off; expression-driven channels
    # throughout.
    (
        "Beach sand with pebbles",
        """\
View(0, 0, 3, 3)
Define(sand, fBm(octaves=5, base_freq=22, to_01=True, seed=2))
Define(grain, fBm(octaves=4, base_freq=70, to_01=True, seed=6))
Define(big, Worley(base_freq=2.5, to_01=True, seed=9))
Define(small, Worley(base_freq=6, to_01=True, seed=21))
Define(bigMask, Threshold(big, below_at=0.2, below_to=1, above_at=0.28, above_to=0, transition_width=0.03))
Define(smallMask, Threshold(small, below_at=0.14, below_to=1, above_at=0.22, above_to=0, transition_width=0.02))
Define(gap, (1 - bigMask))
Define(bigDome, Max(0, (0.26 - big)))
Define(smallDome, Max(0, (0.2 - small)))
Define(tone, fBm(octaves=4, base_freq=3, to_01=True, seed=17))
Material(
  Layer(1)
    .basecolor((205 + (sand * 25)), (180 + (sand * 22)), (145 + (sand * 20)))
    .roughness((0.85 + (grain * 0.1)))
    .height((grain * 0.01)),
  Layer(bigMask)
    .basecolor((70 + (tone * 70)), (68 + (tone * 62)), (62 + (tone * 52)))
    .roughness((0.55 + (tone * 0.25)))
    .height((bigDome * 0.08)),
  Layer((smallMask * gap))
    .basecolor((60 + (tone * 55)), (56 + (tone * 48)), (50 + (tone * 40)))
    .roughness((0.6 + (tone * 0.2)))
    .height((smallDome * 0.05))
)
""",
    ),
    # 4 layers, a plane-filling pattern that continues under any View: the
    # Floor-cell idiom (per-cell hash → random position/size/rotation/brightness)
    # scatters glow-in-the-dark 4-point stars across the infinite plane, while a
    # crescent moon (Ellipse/Fill subtracted by a translated Ellipse) and a comet
    # (CubicTo Stroke + Ellipse head, Scale/Rotate/Translate) are placed accents.
    # Emissive drives the glow at per-star varied strength.
    (
        "Glow-in-the-dark stars with a crescent moon and a comet",
        """\
View(0, 0, 3, 3)
Define(cloud, fBm(octaves=4, base_freq=1.5, to_01=True, seed=8))
Define(gx, (X() * 4))
Define(gy, (Y() * 4))
Define(cx, Floor(gx))
Define(cy, Floor(gy))
Define(fx, (gx - cx))
Define(fy, (gy - cy))
Define(r0, (Sin((cx * 12.989) + (cy * 78.233)) * 437.545))
Define(hpres, (r0 - Floor(r0)))
Define(r1, (Sin((cx * 39.347) + (cy * 11.135)) * 246.635))
Define(hjx, (r1 - Floor(r1)))
Define(r2, (Sin((cx * 73.156) + (cy * 52.235)) * 132.123))
Define(hjy, (r2 - Floor(r2)))
Define(r3, (Sin((cx * 26.651) + (cy * 91.774)) * 567.988))
Define(hsize, (r3 - Floor(r3)))
Define(r4, (Sin((cx * 63.726) + (cy * 10.873)) * 321.753))
Define(hrot, (r4 - Floor(r4)))
Define(r5, (Sin((cx * 45.164) + (cy * 67.982)) * 987.432))
Define(hbri, (r5 - Floor(r5)))
Define(present, Threshold(hpres, below_at=0.45, below_to=0, above_at=0.45, above_to=1))
Define(lx, (fx - (0.3 + (hjx * 0.4))))
Define(ly, (fy - (0.3 + (hjy * 0.4))))
Define(ang, (hrot * pi))
Define(ca, Cos(ang))
Define(sa, Sin(ang))
Define(lxr, ((lx * ca) + (ly * sa)))
Define(lyr, ((ly * ca) - (lx * sa)))
Define(slen, (0.1 + (hsize * 0.1)))
Define(swid, (0.02 + (hsize * 0.015)))
Define(rayX, Max(0, (1 - Sqrt((((lxr) / slen)**2) + (((lyr) / swid)**2)))))
Define(rayY, Max(0, (1 - Sqrt((((lxr) / swid)**2) + (((lyr) / slen)**2)))))
Define(spark, (Max(rayX, rayY) * present))
Define(disc, Fill(Ellipse(0, 0, 0.38, 0.38)))
Define(bite, Fill(Ellipse(0, 0, 0.34, 0.34)))
Define(crescent, (disc * (1 - Translate(bite, dx=-0.16, dy=0.1))))
Define(moon, Translate(crescent, dx=0.65, dy=2.3))
Define(tail, Stroke(Path(0, 0, CubicTo(0.25, 0.1, 0.5, 0.08, 0.72, 0.01)), 0.012, feather=0.008))
Define(head, Fill(Ellipse(0.74, 0.01, 0.05, 0.05)))
Define(comet0, Max(tail, head))
Define(comet, Translate(Rotate(Scale(comet0, sx=0.9, sy=0.9), 20), dx=1.9, dy=2.5))
Material(
  Layer(1)
    .basecolor((18 + (cloud * 20)), (24 + (cloud * 24)), (55 + (cloud * 40)))
    .roughness(0.9),
  Layer(spark)
    .basecolor(120, 220, 150)
    .roughness(0.7)
    .emissive(120, 255, 170, (spark * (0.5 + (hbri * 1.5)))),
  Layer(moon)
    .basecolor(230, 225, 200)
    .roughness(0.8)
    .emissive(255, 245, 210, (moon * 1.3)),
  Layer(comet)
    .basecolor(220, 245, 220)
    .roughness(0.6)
    .emissive(200, 255, 210, (comet * 1.6))
)
""",
    ),
]


def _build_system_prompt(*, few_shot: bool = True, playbook: bool = True) -> str:
    """Assemble the generation system prompt: the DSL reference, the optional
    organic-texture playbook, an optional block of few-shot examples, and the
    output format. ``few_shot`` and ``playbook`` are independent ablation axes:
    a minimal baseline sets both False."""
    parts = [_DSL_REFERENCE]
    if playbook:
        parts.append(_ORGANIC_PLAYBOOK)
    if few_shot:
        ex = [
            """\
# Examples
These are illustrative samples of well-formed programs only.
They teach the vocabulary and idioms of the DSL, not a template to reproduce.
Treat each new description literally: output exactly what it asks for and nothing more.
Do not add layers, weathering, stains, grime, or any feature the description does not mention, even if a similar example below contains them.
A new description that merely resembles an example is not an invitation to copy that example's extra detail.""",
        ]
        for desc, program in FEW_SHOT_EXAMPLES:
            program = program.strip()
            ex.append(f"Description: {desc}\n<material>\n{program}\n</material>")
        parts.append("\n\n".join(ex))
    parts.append(_OUTPUT_INSTRUCTIONS)
    return "\n\n".join(parts)


# |==========================================================================|
# |   Critique (render / code / diagnostics signals; optional descriptors)   |
# |==========================================================================|

_CRITIC_ROLE = "You are a meticulous material-appearance critic."


def _build_critic_system_prompt(render: bool, code: bool, diagnostics: bool) -> str:
    """Assemble the critic system prompt for a (render, code, diagnostics)
    triple.

    All configs share ``_CRITIC_ROLE``; what the critic is shown and how it
    should judge are composed from the three independent signals. ``render`` shows
    a rendered image; ``code`` shows the DSL source (and appends the DSL spec, no
    generator role, so edits can be line-targeted); ``diagnostics`` appends
    objective appearance scalars. For a visual critic (``render``) the scalars
    supplement the image, so a line tells it to use them to ground its visual
    judgment; a text-only critic (no ``render``) is told so by its judge clause,
    which is why the scalars need no "use me" lead-in in the user message.
    The render is always a flat head-on swatch (no scene staging).
    """
    shown = ["a target text description"]
    if render:
        shown.append(
            "a single rendered image of the material on a flat plane viewed head-on"
        )
    if code:
        shown.append("the DSL program source that produced it")
    if diagnostics:
        caveat = "; no render is shown" if not render else ""
        shown.append(
            f"a set of objective appearance scalars (each with a one-line definition, from the unlit material maps{caveat})"
        )
    shown_str = "You are shown " + ", ".join(shown) + "."

    # The layout note a visual critic needs: which render it is looking at. The
    # preview is a flat head-on swatch, so little relief/reflectance/transmission
    # is visible. Placed in the judge clause so every visual config gets it.
    layout_note = """\
The image shows the material on a flat plane viewed head-on; little relief, reflectance, or transmission is visible, so judge those qualities only from what is visible.
"""

    if render and code:
        judge = f"""\
Judge how well the image matches the description, then give the smallest set of line-targeted, parameter-specific edits the author can apply directly: reference layer indices in source order (0 = the first Layer listed, i.e. the bottom-most substrate), channel names, and exact value changes (e.g. "Layer 2 .roughness: 0.35 -> 0.6", "Layer 3's mask is inverted; use (1 - mask)", "raise pebble base_freq from 2.5 to 4").
{layout_note}Do not rewrite the whole program.
"""
    elif render:  # render only, no code (uninformed)
        judge = f"""\
Judge how well the image matches the description.
You do not see (and must not assume) any code or parameters — reason from the visible appearance: color, glossiness, pattern/structure, scale, relief, and any special quality named (metallic, fabric, glass, glowing, etc.).
{layout_note}Be specific and actionable.
"""
    elif code:  # code only, no render (text-only static analysis)
        if diagnostics:
            judge = """\
Judge how well the DSL program and its appearance scalars match the description, then give the smallest set of line-targeted, parameter-specific edits the author can apply directly: reference layer indices in source order (0 = the bottom-most substrate), channel names, and exact value changes.
You do not see a render of this program — reason from the source and the objective scalars. Cross-check the scalars against the code (e.g. height_range is 0 while .height() is non-constant means the noise is flat; raise its base_freq).
Do not rewrite the whole program.
"""
        else:
            judge = """\
Judge how well the DSL program matches the description by reading the source, then give the smallest set of line-targeted, parameter-specific edits the author can apply directly: reference layer indices in source order (0 = the bottom-most substrate), channel names, and exact value changes.
You do not see a render of this program — reason from the source and the description only; flag values that contradict the description (e.g. metallic 1 for a dielectric, a constant height where relief is expected, a View that makes the pattern too large or too small).
Do not rewrite the whole program.
"""
    elif diagnostics:  # diagnostics only (no render, no code)
        judge = """\
Judge whether the scalars are appropriate for the description, and suggest concrete changes that would improve the match (e.g. "raise roughness toward matte", "add relief; height_range is 0", "make the pattern finer or more directional", "add a fully covering substrate; the background may be showing through")
Reason from the scalars and the description only.
"""
    else:  # no signal — _critique guards this out; fail loudly if ever reached
        raise ValueError("At least one of render/code/diagnostics is needed")

    parts = [_CRITIC_ROLE, shown_str, judge]
    if diagnostics and render:
        parts.append(
            "The appearance scalars are objective measurements (not code); use them to ground or check your visual judgment."
        )
    if code:
        parts.append("\n" + _DSL_SPEC)
    return "\n".join(parts)


# Grid resolution for the diagnostics descriptors (independent of the render
# resolution; small is enough since the scalars are statistics, not pixels).
_DESCRIPTOR_GRID = 128


class Critique(JsonResponseModel):
    """Structured critique of a render against its prompt."""

    match_score: Annotated[
        int,
        Field(
            ge=0,
            le=100,
            description="0-100, how well the image matches the description",
        ),
    ]
    differences: Annotated[
        list[str],
        Field(description="Concrete visual mismatches between image and description"),
    ]
    suggestions: Annotated[
        list[str],
        Field(description="Concrete appearance changes that would improve the match"),
    ]


# |==========================================================================|
# |   Client construction                                                    |
# |==========================================================================|


def build_llm(
    model: str, *, base_url: str | None = None, api_key: str | None = None
) -> Llm:
    """Construct an :class:`Llm` for ``model``."""
    return Llm(
        model,
        max_completion_tokens=65536,
        timeout=600.0,
        max_retries=5,
        api_key=api_key,
        base_url=base_url,
    )


# |==========================================================================|
# |   Generation                                                             |
# |==========================================================================|


def extract_program(text: str) -> str | None:
    """Return the program inside ``<material>...</material>`` (or None)."""
    m = _MATERIAL_RE.search(text)
    return m.group(1).strip() if m else None


def _validate(program: str) -> tuple[bool, str]:
    """Try to deserialize ``program``; return (ok, error-message)."""
    from matloom.engine.main import LayeredMaterial

    try:
        LayeredMaterial.deserialize(program)
        return True, ""
    except Exception as e:  # noqa: BLE001 - surface any parse/validation failure
        logger.error(f"Program failed to parse: {e}")
        return False, str(e)


def _request_program(
    convo: Conversation,
    *,
    temperature: float | None,
    reasoning_effort: str | None,
    parse_retries: int,
) -> str:
    """Read the latest assistant turn for a parseable program; on failure, send
    the parser's error back into the SAME conversation and retry, up to
    ``parse_retries`` corrective turns. The caller must have already sent the
    initiating user turn (the prompt, or the critique)."""
    for attempt in range(parse_retries + 1):
        last = convo.history[-1]
        content = last.content or ""
        program = extract_program(content)
        if program is None:
            feedback = """\
I could not find a program.
Reply with ONLY the program wrapped in a single <material>...</material> tag pair.
"""
        else:
            ok, err = _validate(program)
            if ok:
                return program
            feedback = f"""\
The program failed to parse with this error:"
<error>
{err}
</error>
Here is your output:
<material>
{program}
</material>
Fix it and reply with ONLY the corrected <material>...</material>.
"""
        if attempt == parse_retries:
            break
        convo.send(
            feedback,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_parse_retries=_TRANSPORT_RETRIES,
        )
    raise Text2DslError(f"No parseable program after {parse_retries + 1} attempt(s)")


def extract_preamble(text: str) -> str:
    """Return any text the LLM wrote before the first ``<material>`` tag.

    ``_OUTPUT_INSTRUCTIONS`` invites the model to "think artistically and
    thoroughly, then output" the program, so a reply often carries design
    rationale ahead of the tag block. This recovers that preamble (stripped) so
    it can be re-injected when the assistant turn is rewritten; empty when the
    reply is just the tag block. Trailing text after ``</material>`` is not
    recovered — the instructions ask for none and it is usually noise.
    """
    m = _MATERIAL_RE.search(text)
    return text[: m.start()].strip() if m else ""


def _canonicalize_program(program: str) -> str:
    """Round-trip ``program`` through deserialize→serialize.

    This bakes every auto-seeded ``fBm``/``Worley`` into an explicit
    ``seed=N`` literal (the serializer emits ``seed=`` unconditionally), so
    re-parsing the program — by a critique render, the selector render, or
    CMA-ES fitness evals — reproduces the same noise instead of re-rolling it.
    It also normalizes the text (canonical arithmetic/number formatting); with
    set-tracked serialization the tunable set is preserved (no default-channel
    bloat, no ``255`` divisors), so this is safe for the polish.

    Falls back to the authored text if canonicalization fails — a parse edge
    case during the round-trip must never abort the run. The caller has already
    validated parseability via :func:`_validate`, so this normally succeeds.
    """
    from matloom.engine.main import LayeredMaterial

    try:
        return LayeredMaterial.deserialize(program).serialize()
    except Exception:  # noqa: BLE001 - never abort the run
        return program


def _pin_seeds(convo: Conversation, program: str) -> str:
    """Canonicalize ``program`` and rewrite the LLM's last assistant turn in place.

    Canonicalization bakes an explicit ``seed=N`` into every fBm/Worley so
    re-parsing the program (a critique render, the selector, CMA-ES fitness
    evals) reproduces the same noise instead of re-rolling it. The rewritten
    turn also lets the next revision see the baked seeds and copy them forward
    for unchanged layers, keeping an untouched layer's noise stable across refine
    depths.

    The LLM's preamble before ``<material>`` — the "think artistically" reasoning
    invited by ``_OUTPUT_INSTRUCTIONS`` — is preserved ahead of the canonicalized
    block so the model keeps its own design rationale as context on later refine
    turns. The raw reply stays in ``convo.history`` for provenance either way.
    Returns the canonicalized program.
    """
    raw = convo.history[-1].content or ""
    preamble = extract_preamble(raw)
    program = _canonicalize_program(program)
    body = f"<material>\n{program}\n</material>"
    convo.overwrite_last_assistant(f"{preamble}\n\n{body}" if preamble else body)
    return program


def _critique(
    critic: Llm,
    prompt: str,
    program: str,
    *,
    render: bool = False,
    code: bool = False,
    diagnostics: bool = False,
    width: int,
    height: int,
    temperature: float | None,
    reasoning_effort: str | None,
    extra_body: SDict[Any] | None = None,
    verbose: bool = True,
) -> Critique | None:
    """Critique ``program`` against ``prompt`` and return a :class:`Critique`, or
    None to skip refinement this round.

    The critic is a stateless one-shot call (no dialogue history) so its
    ``match_score`` stays comparable across refinement depths. Three independent
    signals select what it sees:

    - ``render`` — render the program with the no-Blender
      :func:`matloom.engine.preview.fixed_light_preview` and feed the image to a
      vision model (the appearance-only critic when ``code`` is False).
    - ``code`` — also show the DSL source, so suggestions can be line-targeted,
      parameter-specific edits (works with or without ``render``).
    - ``diagnostics`` — append the objective appearance scalars (with
      one-line definitions) to the critic's user message for any config.

    At least one signal must be on, else None is returned (nothing to critique).
    In a visual config (``render``) the scalars supplement the image and are
    self-defining (each carries its definition), so the diagnostics-only system
    prompt is NOT injected there — it is the role prompt only for the text-only
    no-``render`` configs. The render is always a flat head-on swatch (no scene
    staging). Returns None when a diagnostics-only round has nothing to critique
    (descriptors unavailable)."""
    # 1. Objective descriptors are computed from the program itself (the unlit
    #    composited channel maps) — no Blender, no lighting. The "why/use them"
    #    framing lives in the system prompt (_build_critic_system_prompt), so the
    #    user message just presents the scalars with a plain label.
    descriptor_block = ""
    if diagnostics:
        from matloom.engine.diagnostics import describe_program

        descriptors = describe_program(
            program, width=_DESCRIPTOR_GRID, height=_DESCRIPTOR_GRID
        )
        bullets = "\n".join(
            f"- {d.label}: {d.formatted} — {d.definition}" for d in descriptors.values()
        )
        if bullets:
            descriptor_block = f"Appearance scalars (computed on a {_DESCRIPTOR_GRID}x{_DESCRIPTOR_GRID} grid):\n{bullets}"

    # 2. Nothing to critique unless there is a render, the code, or descriptors.
    if not (render or code) and not descriptor_block:
        return None

    # 3. Render only when a vision critic needs an image. The no-Blender
    #    fixed_light_preview is pure numpy (no hang risk), so the SIGALRM
    #    render-timeout the old Blender path needed is gone. It returns RGB;
    #    the vision-LLM client (matloom.utils.llm.msg._encode_image) encodes an
    #    ndarray as BGR, so convert here or the critic sees R/B swapped.
    img = None
    if render:
        import cv2

        from matloom.engine.main import LayeredMaterial
        from matloom.engine.preview import fixed_light_preview

        img = cv2.cvtColor(
            fixed_light_preview(LayeredMaterial.deserialize(program), width, height),
            cv2.COLOR_RGB2BGR,
        )

    # 4. System prompt: the role + what the critic is shown + how to judge, all
    #    composed from the (render, code, diagnostics) triple.
    sys_prompt = _build_critic_system_prompt(render, code, diagnostics)

    # 5. User message: the description, plus the program (code) and the
    #    descriptor block (diagnostics), worded for visual vs text-only.
    if render:
        context = "You are comparing a single rendered image of a material against a target text description, judging only the visible appearance."
        objective = f"""\
Compare the attached render against this target description and report how well they match:
<description>
{prompt}
</description>
"""
    else:
        if code and diagnostics:
            context = "You are judging a material from its DSL program source and objective appearance scalars against a target text description."
        elif code:
            context = "You are judging a material from its DSL program source against a target text description."
        else:  # diagnostics only
            context = "You are judging a material from its objective appearance scalars against a target text description."
        objective = f"""\
Judge how well the material suits this target description and report the match:
<description>
{prompt}
</description>
"""
    if code:
        objective += f"""
The DSL program:
<program>
{program}
</program>
Give the smallest set of line-targeted edits that would improve the match.
"""
    if descriptor_block:
        objective += f"\n{descriptor_block}"

    user = CoStar.Json(context=context, objective=objective, response=Critique.to_str())
    # The critic is a one-shot completion (no dialogue history): its own
    # config-specific system prompt + the prompt/render/program/descriptors. It
    # gets a fresh call rather than the generation conversation.
    out = critic(
        user,
        Critique,
        sys_prompt=sys_prompt,
        images=[img] if img is not None else None,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body,
        max_parse_retries=_TRANSPORT_RETRIES,
        verbose=verbose,
    )
    return out.response


def _argmax_shallowest(scores: dict[int, float]) -> int | None:
    """Index of the highest score, ties broken toward the shallowest (smallest
    index) so an earlier candidate wins on a tie. None if empty."""
    if not scores:
        return None
    return min(scores, key=lambda d: (-scores[d], d))


def _select_by_critic(critiques: list[dict]) -> int | None:
    """Best trajectory index by the critic ``match_score`` (one critique per
    depth, in order). None if there are no critiques."""
    return _argmax_shallowest({i: c["match_score"] for i, c in enumerate(critiques)})


def _load_mobileclip2() -> Callable[[Sequence[ImgLike], str], list[float]]:
    from matloom.metrics.clipscore import ClipScore

    clip = ClipScore.MobileClip2()
    return lambda imgs, txt: clip(imgs, [f"A photo of {txt}"]).scores[0].tolist()


def _select_by_mobileclip2(
    trajectory: Sequence[str],
    prompt: str,
    clip: Callable[[Sequence[ImgLike], str], list[float]] | None,
    *,
    width: int,
    height: int,
) -> int | None:
    """Best trajectory index by the MobileCLIP2-S2 image/text score on a flat
    swatch render of each program (the matched-protocol selector). Best-effort:
    returns None if the verifier is unavailable. Uses the no-Blender
    :func:`matloom.engine.preview.fixed_light_preview` (which returns RGB, what
    CLIP expects, so no BGR<->RGB conversion is needed); torch
    (matloom.metrics) is lazy-imported only when ``select_best`` is on."""
    if clip is None:
        return None
    from matloom.engine.main import LayeredMaterial
    from matloom.engine.preview import fixed_light_preview

    logger.info("Selecting by MobileCLIP2")
    imgs = [
        fixed_light_preview(LayeredMaterial.deserialize(prog), width, height)
        for prog in trajectory
    ]
    scores = {i: score for i, score in enumerate(clip(imgs, prompt))}
    return _argmax_shallowest(scores)


@validate_call
def text_to_material(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    playbook: bool = True,
    few_shot: bool = True,
    parse_retries: int = 3,
    refine: int = 0,
    critic_model: str | None = None,
    critic_base_url: str | None = None,
    critic_api_key: str | None = None,
    critic_render: bool = False,
    critic_code: bool = False,
    critic_diagnostics: bool = False,
    select_best: bool = False,
    polish: bool = False,
    polish_seed_sweep: int = 1000,
    polish_cma_evals: int = 0,
    polish_parallel_renders: int | None = None,
    width: int = 512,
    height: int = 512,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[list[str], SDict[Any]]:
    """Generate a trajectory of layered-material DSL programs from a
    natural-language ``prompt``.

    Args:
        prompt: the material description.
        model: generation model name (a literal model id).
        base_url / api_key: override the config for the generation model.
        temperature / reasoning_effort: per-call LLM controls (None → provider
            default).
        few_shot: include the curated few-shot examples in the system prompt.
        playbook: include the organic-texture playbook in the system prompt. The
            two are independent ablation axes; a minimal baseline sets both False.
        parse_retries: corrective turns allowed when a reply fails to parse
            (0 disables retrying; default 3).
        refine: number of critique→revise rounds (0 disables refinement). Needs
            at least one of ``critic_render``, ``critic_code``, or
            ``critic_diagnostics``.
        critic_model: model for the critic (defaults to ``model``).
        critic_base_url / critic_api_key: override the config for the critic
            model (defaults to the generation model's ``base_url``/``api_key``).
        critic_render: render the program with the no-Blender
            :func:`matloom.engine.preview.fixed_light_preview` and feed the image
            to a vision critic. The three refinement signals — ``critic_render``,
            ``critic_code``, ``critic_diagnostics`` — are independent, so all
            seven non-empty combinations are selectable.
        critic_code: show the critic the DSL source so its suggestions can be
            line-targeted, parameter-specific edits. Works with or without
            ``critic_render`` (without it, a text-only static-analysis critic).
        critic_diagnostics: append objective appearance scalars (with one-line
            definitions, computed from the unlit channel maps) to the critic's
            user message. With neither ``critic_render`` nor ``critic_code`` this
            makes refinement diagnostics-only (a text-only critic; no image
            render).
        select_best: pick the best program in the trajectory and record its index
            in the record under ``selected`` (see Returns). Two selectors
            are computed: the critic ``match_score`` (free; needs a critic) and
            the MobileCLIP2-S2 score on a flat swatch render (the matched-protocol
            verifier; lazy-loads torch). Each is None when its signal
            is unavailable.
        polish: after selection, polish up to three trajectory programs (p0,
            the chosen program (see ``selected``), and the last) and append the
            best result to the trajectory. Each is first run through a cheap
            discrete seed sweep (``polish_seed_sweep`` variants), then, when
            ``polish_cma_evals > 0``, CMA-ES polishes the continuous literals of
            every seed variant; the highest-scoring program across the sweep and
            all CMA-ES polishes is kept (accept-if-better on the MobileCLIP2-S2
            verifier over a fast no-Blender fixed-light preview). Lazy-loads the
            ``cma`` package. The per-program polish logs are recorded under
            ``record["polish"]`` and the best polished program is appended to
            the trajectory written by :func:`save_record`.
        polish_seed_sweep: per program, how many re-seeded variants to try
            before CMA-ES (the primary, cheap discrete sweep; default 1000). 0
            disables the sweep, so only CMA-ES (on the as-authored program) runs.
        polish_cma_evals: CMA-ES evaluation budget per seed variant (the
            secondary, continuous polish; default 0 = CMA-ES off, seed sweep
            only).
        polish_parallel_renders: worker processes for the polish seed-sweep
            renders (None = the measured default, min(4, CPUs), so the pool is
            on by default wherever ``polish`` runs; 0 = render sequentially).
            Renders are pure functions of the program text, so this changes
            only wall time, never the polished program. The sweep runs inside
            ``python -m matloom.engine.text2dsl`` / the console script, whose
            entry points are spawn-guarded; an embedding script must guard its
            own entry point (see ``polish_program``).
        width / height: render resolution used for the visual critique and the
            MobileCLIP2 selection.
        seed: when set, seeds Python's ``random`` and ``numpy.random`` so the run
            is reproducible from this seed — notably the auto-seeds baked into
            each program by :func:`_canonicalize_program` and the CMA-ES polish.
            For single-call (CLI) reproducibility only: ``text_to_material`` is
            run concurrently in the eval pipeline, and seeding the global RNG
            from concurrent calls would race, so leave this ``None`` there.
        verbose: print the conversation panels.

    Returns:
        A ``(trajectory, record)`` tuple. ``trajectory`` is the list of programs
        after 0, 1, ..., refines (all parseable by ``LayeredMaterial.deserialize``).
        ``record`` is a dict ``{"prompt", "timestamp", "model", "settings",
        "critiques", "selected"}`` (the same shape :func:`save_record` writes to
        ``record.json``).
        ``record["critiques"]`` is a list of ``Critique`` dicts
        (``match_score``, ``differences``, ``suggestions``), one per scored
        program: ``critiques[i]`` scores ``trajectory[i]`` and, for all but a
        trailing entry, drove the revision producing ``trajectory[i + 1]``. When
        the refine loop completes with no skipped round a final critique of the
        last program is appended that produces no revision. The list is empty
        without refinement or when every round was skipped.
        ``record["selected"]`` is ``{"critic": int|None, "mobileclip2":
        int|None}`` (set when ``select_best`` is on; both None otherwise).
    """
    from matloom.engine import noise

    noise.seed(seed)
    stamp = datetime.now(timezone.utc).strftime(r"%Y%m%d-%H%M%S-%f")
    need_refine = refine > 0 and (critic_render or critic_code or critic_diagnostics)
    duration = {
        "total": None,
        "generation": [],
        "critique": [],
        "select": None,
        "polish": [],
    }
    model_extra_body = (
        {"reasoning": {"enabled": True}}
        if "gemma-4" in model or "qwen3.6" in model
        else {}
    )
    critic_extra_body = (
        {"reasoning": {"enabled": True}}
        if critic_model and ("gemma-4" in critic_model or "qwen3.6" in critic_model)
        else {}
    )
    start_time_total = time.perf_counter()
    with progress_bar(transient=False) as pbar:
        pbar_id = pbar.add_task("Generating material", total=need_refine * refine + 1)
        gen = build_llm(model, base_url=base_url, api_key=api_key)
        convo = gen.conversation(
            sys_prompt=_build_system_prompt(few_shot=few_shot, playbook=playbook),
            verbose=verbose,
        )
        start_time_g = time.perf_counter()
        convo.send(
            f"Create a material for this description:\n<description>\n{prompt}\n</description>",
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            extra_body=model_extra_body,
            max_parse_retries=_TRANSPORT_RETRIES,
        )
        program = _request_program(
            convo,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            parse_retries=parse_retries,
        )
        # Pin noise seeds (and preserve the LLM's preamble) — see _pin_seeds.
        program = _pin_seeds(convo, program)
        duration["generation"].append(time.perf_counter() - start_time_g)
        pbar.advance(pbar_id)
        # The trajectory records the program after 0, 1, ... refines so a single
        # refine=N run answers every refine depth 1..N.
        trajectory: list[str] = [program]
        critiques: list[SDict[Any]] = []

        if need_refine:
            critic_llm = build_llm(
                critic_model or model,
                base_url=critic_base_url or base_url,
                api_key=critic_api_key or api_key,
            )
            for _ in range(refine):
                pbar.update(pbar_id, description="Critiquing material")
                start_time_c = time.perf_counter()
                verdict = _critique(
                    critic_llm,
                    prompt,
                    program,
                    render=critic_render,
                    code=critic_code,
                    diagnostics=critic_diagnostics,
                    width=width,
                    height=height,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    extra_body=critic_extra_body,
                    verbose=verbose,
                )
                if verdict is None:
                    break  # Render/descriptors unavailable — stop refining
                duration["critique"].append(time.perf_counter() - start_time_c)
                critiques.append(verdict.model_dump())
                pbar.update(pbar_id, description="Refining material")
                notes = "\n".join(
                    f"- {d}" for d in (verdict.differences + verdict.suggestions)
                )
                start_time_g = time.perf_counter()
                convo.send(
                    f"""\
A critic reviewed your material against the original description and scored the match {verdict.match_score}/100.
Their notes:

{notes}

Revise the material to address these and reply with ONLY the updated <material>...</material>.
When keeping a layer unchanged, copy it exactly, including any seed= values, so its noise stays the same.
    """,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    extra_body=model_extra_body,
                    max_parse_retries=_TRANSPORT_RETRIES,
                )
                program = _request_program(
                    convo,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    parse_retries=parse_retries,
                )
                # Pin seeds on the revised program (see the depth-0 note above).
                program = _pin_seeds(convo, program)
                duration["generation"].append(time.perf_counter() - start_time_g)
                pbar.advance(pbar_id)
                trajectory.append(program)
            else:
                # The loop completed with no `break` (no round was skipped), so
                # score the final revised program too: every depth's program then has
                # a critique, including the last. This trailing entry produces no
                # revision (see the Returns section of `text_to_material`).
                pbar.update(pbar_id, description="Critiquing material")
                start_time_c = time.perf_counter()
                verdict = _critique(
                    critic_llm,
                    prompt,
                    program,
                    render=critic_render,
                    code=critic_code,
                    diagnostics=critic_diagnostics,
                    width=width,
                    height=height,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    extra_body=critic_extra_body,
                    verbose=verbose,
                )
                if verdict is not None:
                    duration["critique"].append(time.perf_counter() - start_time_c)
                    critiques.append(verdict.model_dump())

        pbar.remove_task(pbar_id)

    # Load MobileCLIP2 lazily and best-effort. torch, Blender, and the model
    # weights are optional (torch-free CI, headless boxes), and a single-program
    # trajectory with ``select_best`` has nothing to select among, so the load is
    # both deferred past that short-circuit and guarded: a missing dependency
    # yields ``clip = None`` (the selector and polish fall back to the critic
    # signal) instead of aborting the run.
    clip = None
    if (select_best and len(trajectory) > 1) or polish:
        try:
            clip = _load_mobileclip2()
        except Exception as e:  # noqa: BLE001 - best-effort optional load; any failure degrades to clip=None
            logger.error(f"MobileCLIP2 unavailable: {e}")

    # Selection: the best trajectory index by each signal (None where the signal
    # is absent). The MobileCLIP2 selector is the matched-protocol verifier
    # (selects on a flat swatch); the critic selector is free (reuses the
    # refinement scores) and is clean w.r.t. all eval metrics when the critic
    # model differs from the judge model.
    selected: SDict[int | None] = {"critic": None, "mobileclip2": None}
    if select_best:
        if len(trajectory) > 1:
            selected["critic"] = _select_by_critic(critiques)
            start_time_s = time.perf_counter()
            selected["mobileclip2"] = _select_by_mobileclip2(
                trajectory, prompt, clip, width=width, height=height
            )
            duration["select"] = time.perf_counter() - start_time_s
        else:
            # A single-program trajectory (no refinement, or it broke at depth 0)
            # has nothing to select among; the only candidate is trivially index 0,
            # so skip the verifier load and render entirely.
            selected["critic"] = 0
            selected["mobileclip2"] = 0

    chosen_before_polish = selected["mobileclip2"]
    if chosen_before_polish is None:
        chosen_before_polish = selected["critic"]
    if chosen_before_polish is None:
        chosen_before_polish = len(trajectory) - 1

    polish_logs: list[SDict[Any]] | None = None
    # The resolved worker count (not the raw parameter), so the run record
    # reflects the pool that actually ran (the raw None would read as "no pool").
    pool_workers: int | None = None
    if polish:
        from matloom.engine.polish import _default_pool_workers, polish_program

        cand_idxs = sorted({0, chosen_before_polish, len(trajectory) - 1})  # Dedup
        polish_logs = []
        # Pool on by default (renders are pure functions of the text, so only
        # wall time changes); 0 via the kwarg/--polish-parallel-renders opts out.
        pool_workers = (
            _default_pool_workers()
            if polish_parallel_renders is None
            else polish_parallel_renders
        )
        for idx in cand_idxs:
            logger.info(f"Polishing trajectory[{idx}]")
            start_time_p = time.perf_counter()
            _, polish_log = polish_program(
                prompt,
                trajectory[idx],
                seed_sweep=polish_seed_sweep,
                max_cma_evals=polish_cma_evals,
                seed=seed,
                score_fn=clip,
                parallel_renders=pool_workers,
            )
            duration["polish"].append(time.perf_counter() - start_time_p)
            polish_logs.append({"index": idx, **polish_log})
        best_polish = max(
            polish_logs, key=lambda log: log.get("final_score", -math.inf)
        )
        trajectory.append(
            best_polish.get("final_program", trajectory[chosen_before_polish])
        )

    duration["total"] = time.perf_counter() - start_time_total
    record = {
        "prompt": prompt,
        "timestamp": stamp,
        "settings": {
            "model": model,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "few_shot": few_shot,
            "playbook": playbook,
            "parse_retries": parse_retries,
            "refine": refine,
            "critic_model": critic_model,
            "critic_render": critic_render,
            "critic_code": critic_code,
            "critic_diagnostics": critic_diagnostics,
            "select_best": select_best,
            "polish": polish,
            "polish_seed_sweep": polish_seed_sweep,
            "polish_cma_evals": polish_cma_evals,
            "polish_parallel_renders": pool_workers,
            "width": width,
            "height": height,
            "seed": seed,
        },
        "duration_seconds": duration,
        "critiques": critiques,
        "selected": selected,
        "chosen_before_polish": chosen_before_polish,
        "polish": polish_logs,
    }
    return trajectory, record


# |==========================================================================|
# |   Persistence + CLI                                                      |
# |==========================================================================|


def save_record(
    output_dir: PathLike, trajectory: list[str], record: SDict[Any]
) -> Path:
    """Write ``trajectory`` to ``trajectory.xml`` and ``record`` to ``record.json``
    under a timestamped subdirectory of ``output_dir``. ``record`` is the dict
    returned by :func:`text_to_material`; its ``timestamp`` names the subdirectory."""
    from lxml import etree

    from matloom.utils.file import save_json

    stamp = record["timestamp"]
    out = Path(output_dir) / f"slate_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    root = etree.Element("trajectory")
    for prog in trajectory:
        # Include leading newline and trailing newline + 2 spaces for closing tag alignment
        etree.SubElement(root, "program").text = etree.CDATA(f"\n{prog}\n  ")
    tree = etree.ElementTree(root)
    # pretty_print=True indents outer tags while keeping CDATA content strictly formatted
    tree.write(
        out / "trajectory.xml",
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True,
    )
    save_json(record, out / "record.json", indent=2)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matloom-text2dsl",
        description=(
            "Generate a layered-material DSL program from a natural-language "
            "prompt using an LLM. Prints the program."
        ),
    )
    p.add_argument(
        "--prompt", required=True, help="natural-language material description"
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="generation model")
    p.add_argument("--base-url", default=None, help="override the model's base URL")
    p.add_argument("--api-key", default=None, help="override the model's API key")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument(
        "--reasoning-effort",
        default=None,
        help="override the model's reasoning effort (None → model's default)",
    )
    p.add_argument(
        "--playbook",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include the organic-texture playbook (independent of --few-shot)",
    )
    p.add_argument(
        "--few-shot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include few-shot examples (independent of --playbook; --no-playbook --no-few-shot is the minimal baseline)",
    )
    p.add_argument(
        "--parse-retries",
        type=int,
        default=3,
        help="corrective turns when a reply fails to parse (0 disables)",
    )
    p.add_argument(
        "--refine",
        type=int,
        default=0,
        help=(
            "number of critique→revise rounds (needs at least one of "
            "--critic-render, --critic-code, or --critic-diagnostics)"
        ),
    )
    p.add_argument(
        "--critic-model",
        default=None,
        help="model for the critic (default: same as --model)",
    )
    p.add_argument(
        "--critic-base-url",
        default=None,
        help="override the critic model's base URL (default: same as --base-url)",
    )
    p.add_argument(
        "--critic-api-key",
        default=None,
        help="override the critic model's API key (default: same as --api-key)",
    )
    p.add_argument(
        "--critic-render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "render the program (Blender) and feed the image to a vision critic. "
            "Independent of --critic-code and --critic-diagnostics; combine for any of "
            "the 7 refinement configs"
        ),
    )
    p.add_argument(
        "--critic-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "show the critic the DSL source for line-targeted edits (works with "
            "or without --critic-render; without it, a text-only static-analysis "
            "critic needing no Blender/vision model)"
        ),
    )
    p.add_argument(
        "--critic-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "append objective appearance scalars (with definitions, from the unlit "
            "channel maps) to the critic's message; with neither --critic-render "
            "nor --critic-code this makes refinement diagnostics-only (a text-only "
            "critic, no Blender render)"
        ),
    )
    p.add_argument(
        "--select-best",
        action="store_true",
        help=(
            "after refinement, select the best program in the trajectory by two "
            "signals and record the indices in the record: the critic match_score "
            "(needs at least one of --critic-render, --critic-code, or --critic-diagnostics) "
            "and the MobileCLIP2-S2 score on a flat swatch render (lazy-loads Blender "
            "+ torch). The rendered/printed program is the mobileclip2-chosen one "
            "(falls back to the critic-score pick, then the last)"
        ),
    )
    p.add_argument(
        "--polish",
        action="store_true",
        help=(
            "after selection, polish p0, the chosen program, and the last: a "
            "cheap discrete seed sweep (see --polish-seed-sweep) then, with "
            "--polish-cma-evals > 0, CMA-ES on every seed variant. The best "
            "program across the sweep and all CMA-ES polishes is appended to the "
            "trajectory (accept-if-better on the MobileCLIP2-S2 verifier over a "
            "fast no-Blender fixed-light preview). Lazy-loads the 'cma' package"
        ),
    )
    p.add_argument(
        "--polish-seed-sweep",
        type=int,
        default=1000,
        help=(
            "per program, how many re-seeded variants to try before CMA-ES (the "
            "primary, cheap discrete sweep; default 1000; 0 disables, leaving "
            "only CMA-ES on the as-authored program)"
        ),
    )
    p.add_argument(
        "--polish-cma-evals",
        type=int,
        default=0,
        help=(
            "CMA-ES evaluation budget per seed variant (the secondary, "
            "continuous polish; default 0 = CMA-ES off, seed sweep only)"
        ),
    )
    p.add_argument(
        "--polish-parallel-renders",
        type=int,
        default=None,
        help=(
            "worker processes for the polish seed-sweep renders (default: "
            "min(4, CPUs), so the pool is on wherever --polish runs; 0 renders "
            "sequentially). Purely a wall-time knob: renders are deterministic "
            "functions of the program text"
        ),
    )
    p.add_argument("--width", type=int, default=512, help="critique render width")
    p.add_argument("--height", type=int, default=512, help="critique render height")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "seed Python/numpy RNG for a reproducible single run: pins the "
            "auto-seeds baked into each program's noise and the CMA-ES polish. "
            "Single-call only (the eval pipeline runs concurrently; leave unset "
            "there)"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "if set, save the trajectory and record to a timestamped subdirectory "
            "under here"
        ),
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="render the final program using Blender",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="if --output-dir and --render are set, show the rendered image",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the conversation panels",
    )
    return p


def _cli(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    trajectory, record = text_to_material(
        args.prompt,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        playbook=args.playbook,
        few_shot=args.few_shot,
        parse_retries=args.parse_retries,
        refine=args.refine,
        critic_model=args.critic_model,
        critic_base_url=args.critic_base_url,
        critic_api_key=args.critic_api_key,
        critic_render=args.critic_render,
        critic_code=args.critic_code,
        critic_diagnostics=args.critic_diagnostics,
        select_best=args.select_best,
        polish=args.polish,
        polish_seed_sweep=args.polish_seed_sweep,
        polish_cma_evals=args.polish_cma_evals,
        polish_parallel_renders=args.polish_parallel_renders,
        width=args.width,
        height=args.height,
        seed=args.seed,
        verbose=not args.quiet,
    )
    program = trajectory[-1]
    logger.info(f"""\
Material program generated by {args.model} for "{args.prompt}":
{"-" * 40}
{program}
{"-" * 40}
""")

    out_dir = None
    if args.output_dir is not None:
        out_dir = save_record(args.output_dir, trajectory, record)
        logger.info(f"Saved to {out_dir}")
    if args.render:
        from matloom.render import _show_images, program_to_material

        # The optional final display render stays Blender-backed (a local-only
        # preview of the finished material, not part of the generation method).
        # The critic's --critic-scene flag is gone, so the display is the flat
        # swatch layout.
        if out_dir is not None:
            renders = program_to_material(program, out_dir, scene=False)
            if args.show:
                _show_images(renders)
        else:
            import tempfile

            with tempfile.TemporaryDirectory(prefix="matloom-render-") as tmp:
                _show_images(program_to_material(program, tmp, scene=False))


if __name__ == "__main__":
    _cli()
