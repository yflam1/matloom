"""
matloom.engine — the layered-material DSL: expression tree, noise, layer data
classes, the material evaluator, and textual (de)serialization of a material.

This is the reference implementation. A byte-compatible TypeScript port lives in
``frontend/src/engine`` and the two MUST stay in sync (see
``tests/parity/generate_fixtures.py`` for the parity contract and the
cross-engine change protocol).

Modules
-------
- :mod:`matloom.engine.expr` — the ``Expression2D`` tree: leaves (``Constant``,
  ``X``, ``Y``), unary (``Sin``/``Cos``/``Log``/``Abs``/``Sqrt``/``Floor``/
  ``Ceil``), binary (``Add``/``Sub``/``Mul``/``Div``/``Pow``/``Min``/``Max``),
  and ``Ref`` (``expr.py:413``, a named reference to a ``Define``).
  ``coerce_args`` (``expr.py:86``) lets bare numbers stand in for expressions
  and vice-versa. Each node has a scalar ``__call__`` and a vectorized
  ``eval_grid``; the optional ``coords`` supplies per-cell coordinate grids so a
  non-separable transform (``Rotate``) can sample its child. A render-scoped
  ``eval_grid`` memo (``grid_cache_scope``, ``expr.py:394``; byte-capped LRU,
  inert outside a scope) is consulted only inside ``Ref.eval_grid``, so a
  ``Define`` referenced from many channels evaluates once per render. Purity
  contract: ``eval_grid`` consumers must never mutate returned child outputs
  in place — a memo hit hands out the SAME array object to every consumer of
  a shared subtree, so an in-place op corrupts later consumers (the
  memo-on/off bit-identity test catches it).
- :mod:`matloom.engine.noise` — ``fBm`` (layered OpenSimplex, ``noise.py:89``) and
  ``Worley`` (cellular, ``noise.py:430``) noise, plus ``seed``/``get_random_seed``.
  Every grid path evaluates through flat numba kernels that replicate the scalar
  ``__call__`` bit-for-bit: ``_worley_kernel`` (``noise.py:297``) for Worley's
  separable axes, ``_fbm_grid_kernel`` (``noise.py:255``) for fBm's (the per-octave
  1-D axis scaling stays in numpy so float32 axes keep their per-octave rounding),
  and ``_worley_coords_kernel`` (``noise.py:350``) / ``_fbm_coords_kernel``
  (``noise.py:216``) for non-separable ``coords`` (e.g. inside ``Rotate``); the
  per-cell hash is ``_hash_to_01`` (``noise.py:282``). The coords kernels pick
  their output dtype via ``_coords_out_dtype`` (``noise.py:407``), which replicates
  ``np.vectorize``'s otype-probe rule, and reach into opensimplex's pinned
  internal ``_noise2`` (``opensimplex==0.4.5.1``), so any opensimplex/numba/numpy
  bump requires re-running the 0-ulp and fingerprint gates.
- :mod:`matloom.engine.util` — sRGB↔linear conversions and the ``Threshold``
  expression (clamp/smoothstep), with the numba kernel ``_threshold_kernel``
  (``util.py:55``).
- :mod:`matloom.engine.pattern` — ``Bricks`` (``pattern.py:21``): a running-bond
  brick lattice (1 = brick, 0 = mortar), pure ``Floor``/arithmetic. ``Weave``
  (``pattern.py:128``): an over/under woven-thread lattice (plain/twill/satin),
  the fabric counterpart of ``Bricks``, pure ``Floor``/mod arithmetic so exact
  across engines.
- :mod:`matloom.engine.transform` — coordinate transforms about the origin:
  ``Translate``/``Scale`` (``transform.py:26``, separable axis rewrites) and
  ``Rotate`` (``transform.py:97``, builds the per-cell ``coords``).
- :mod:`matloom.engine.shape` — SVG-like shapes: ``Rect``/``Ellipse`` (analytic
  SDF) and ``Path`` (``shape.py:169``, cubic-Bézier polyline, point-in-polygon
  fill + distance-to-polyline stroke), plus ``Fill``/``Stroke`` (``shape.py:270``),
  the ``Expression2D`` nodes that turn a ``Shape`` into a binary mask field.
- :mod:`matloom.engine.layer` — ``Color``/``Emissive``/``Layer`` pydantic models;
  channels exposed as threshold-clamped expressions. ``Color.from_int``
  (``layer.py:53``) builds a 0–255 color. Besides ``metallic``/``roughness``,
  ``Layer`` carries the OpenPBR-aligned scalar channels ``sheen``/``coat``/
  ``transmission``/``subsurface``/``anisotropy`` (each clamped ``[0,1]``) and
  ``ior`` (clamped ``>=1``, default 1.5), plus a ``height`` channel (absolute
  coordinate units, left unclamped). New scalar channels are added here as a
  ``raw_*`` field + a clamped ``@computed_field`` accessor; the chain order
  (``…roughness → sheen → coat → transmission → ior → subsurface → anisotropy →
  emissive → height``) must stay byte-identical across both engines + README.
- :mod:`matloom.engine.relief` — derives surface relief from the per-layer
  height channel: ``composite_height`` (the ``maxᵢ { hᵢ : αᵢ > 0 }`` operator;
  alpha is a *coverage mask* gating where each layer contributes, not a
  multiplier, and not the alpha-weighted blend the other channels use; heights
  are in coordinate units, may be any real value incl. negative, with no
  substrate floor), ``height_to_normal`` (central-difference OpenGL/+Y-up
  tangent normal) and ``height_to_ao`` (box-blur cavity AO; the blur radius is
  per-axis so a caller can hold the blur's *world* size fixed as the texel size
  changes, which makes AO crop-invariant). Byte-for-byte mirror of
  ``frontend/src/engine/relief.ts``.
- :mod:`matloom.engine.parser` — string → ``Expression2D``. Tokenizer +
  recursive-descent parser; the ``_REGISTRY`` (``parser.py:60``) fixes the
  positional-arg order shared with the frontend, ``_REST_PARAMS`` handles
  ``Path``'s trailing segments, and an optional ``env`` resolves ``Define``'d
  names to ``Ref``s.
- :mod:`matloom.engine.serialize` — ``Expression2D`` → canonical DSL string
  (``serialize_expr``, ``serialize.py:82``), including shapes/transforms/
  ``Bricks``/``Ref``. Inverse of :mod:`matloom.engine.parser`; bakes noise seeds
  in. A ``Pow`` base that renders with a leading ``-`` (negative ``Constant``/
  unary minus) is parenthesized so ``(-3 ** e)`` round-trips as ``((-3) ** e)``
  (the parser binds ``**`` tighter than unary minus).
- :mod:`matloom.engine.generate` — grammar-free *random* generator
  (``MaterialGenerator``/``generate_material``): samples a whole
  ``LayeredMaterial`` by calling the same engine constructors the parser uses
  (not by string concatenation), so every program is valid by construction and
  round-trips. Recursively builds channel expressions from the full node pool
  (leaves incl. ``Ref``s to earlier ``Define``s, noise/pattern/shape generators,
  unary/binary math, transforms, ``Threshold``); type-faithful numeric sampling
  over configurable bounded ranges (``GeneratorConfig``); deterministic under
  ``seed``. CLI: ``matloom-generate``. Mirror of ``frontend/src/generate.ts``.
- :mod:`matloom.engine.main` — ``LayeredMaterial`` (``main.py:40``): holds a layer
  stack, an ordered ``Define`` table, and a ``View`` (region ``x1,y1,x2,y2``);
  ``serialize``/``deserialize`` to the ``[View(…)] [Define(…)…]
  Material(Layer(…)…)`` text format. ``export`` composites and writes basecolor
  + the ``[0,1]`` scalar maps (metallic/roughness/sheen/coat/transmission/
  subsurface/anisotropy as ``.png``), the float ``ior.{exr,hdr}``,
  ``emissive.{exr,hdr}``, ``height.{exr,hdr}`` and the derived ``normal.png``/
  ``ao.png`` (via :mod:`matloom.engine.relief`). The height map is a float
  ``.exr`` when ``OPENCV_IO_ENABLE_OPENEXR=1``, else a Radiance ``.hdr`` fallback
  that clamps negative heights to 0 (it warns when that would lose
  below-substrate values). CLI: ``matloom-export`` — ``--program`` (required, the
  DSL string ``matloom-generate`` prints) plus optional ``--output-dir``/
  ``--width``/``--height``/``--y-up``.
- :mod:`matloom.engine.text2dsl` — the LLM text→DSL pipeline; see its own
  docstring. CLI: ``matloom-text2dsl``.
- :mod:`matloom.engine.diagnostics` — objective *appearance descriptors*
  (luminance, contrast, saturation, edge density, dominant frequency,
  anisotropy, relief range/tilt/AO, and the PBR channel means) computed from
  the unlit composited channel maps on a small grid; handed to a critic with
  one-line definitions so it can judge appropriateness against the prompt.
- :mod:`matloom.engine.polish` — seed-sweep (primary) + CMA-ES (secondary) polish
  of an LLM-authored program against the MobileCLIP2-S2 verifier on the
  no-Blender :mod:`matloom.engine.preview` (LLM structure, evolution parameters).
  The seed sweep redraws the program's noise seeds; CMA-ES, when enabled
  (``max_cma_evals > 0``), tunes the continuous literals of every seed variant.
  The returned program is the highest-scoring one across the sweep and all
  CMA-ES polishes (accept-if-better via the max). ``cma`` is lazy-imported (seed
  sweep + extraction are cma-free). The sweep's renders can fan out over a
  spawned ``ProcessPoolExecutor`` (``parallel_renders`` / env
  ``MATLOOM_POLISH_WORKERS``): renders are pure functions of the program text and
  ``executor.map`` preserves submission order, so the pooled results are
  byte-identical to the sequential ones.
- :mod:`matloom.engine.preview` — the no-Blender fixed-light RGB preview renderer
  (``fixed_light_preview``): pure-numpy composite of the unlit channel maps (the
  same over-operator + height relief as :mod:`matloom.engine.main`'s export)
  shaded under one fixed directional light, head-on camera. Approximates
  roughness/metallic/clearcoat/subsurface/sheen/transmission/emissive; every
  Layer channel is composited and consumed (alpha coverage composites over a
  mid-gray background; emissive is soft-knee tone-mapped to preserve the HDR
  strength axis). The renderer the text2dsl vision critic and the polish
  verifier see, so the text→DSL method is Blender-free and portable. Each
  composite runs inside the per-render ``Ref``-boundary ``eval_grid`` memo
  (:func:`matloom.engine.expr.grid_cache_scope`); ``memo=False`` on
  ``fixed_light_preview``/``_composite`` is the bit-identical bisection escape
  hatch. Returns ``uint8`` ``(H, W, 3)`` RGB (vision-LLM callers must
  BGR-convert; CLIP callers consume it directly).
- :mod:`matloom.engine.const` — ``pi``, ``e``.

Accelerated evaluation
----------------------
The engine stays numpy+numba only; this note preserves the measured verdict
from a full GPU diagnosis so it survives outside /tmp. The byte-exact render
contract cannot run on this machine's GPU: MPS hard-rejects float64 tensors,
and float32 pipelines diverge from the CPU bytes on ~6.6% of real corpus
programs (the ``Sin(coord*43758.5453)`` hash idiom and sub-texel feather
edges are not float32-safe). Per-node GPU dispatch is launch-bound (~17-21 us
per launch, 1000+ per render) and measured slower than the numba kernels, so
GPU evaluation must never be wired per-node in-engine; the only paying design
is a batched float32 compiled-graph pre-screen with an exact CPU re-rank, and
it belongs at the whole-render orchestration seam (batch preview generation).
A torch-CPU float64 compiled simplex measured bit-identical and 2.4x faster
at 1M points, but its ~6.7s/shape compile warmup and Inductor FMA risk buy
only ~1.2-1.3x end-to-end after the numba stages, so it stays out too.

Cross-engine change protocol
----------------------------
- **Engine math (either side):** update the mirror in the other engine, add or
  adjust unit tests, regenerate parity fixtures, and run both suites.
- **DSL grammar / functions:** update ``parser.py`` + ``serialize.py`` AND
  ``frontend/src/expr-lang/{parser,registry,serialize}.ts``, the EBNF in
  ``README.md``, and tests on both sides. The README EBNF must stay one single,
  self-contained ``ebnf`` block: it is the paper-ready spec and must parse +
  validate real programs via ``matloom.bnf.ExtendedBNF`` with start symbol
  ``program``.
- **Layer channels** (e.g. ``height``): these are layer *methods*, not
  expression functions — update ``layer.py`` + ``main.py``'s
  ``_serialize_layer``/``_parse_layer_chunk``/``_apply_method``/``_build_layer``,
  and the frontend mirror in ``state.ts``/``channels.ts``/``eval.ts``/
  ``material-io.ts``, plus the ``layer`` EBNF rule in ``README.md``. Keep the
  canonical ``Material(Layer(…)…)`` string identical across engines (the
  ``material-io`` round-trip tests enforce this).
"""
