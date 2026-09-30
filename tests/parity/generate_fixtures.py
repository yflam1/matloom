"""
generate_fixtures.py — emit cross-engine parity fixtures from the canonical
Python engine.

The Python engine in ``matloom/engine`` is the reference implementation; the
TypeScript engine in ``frontend/src/engine`` is meant to reproduce it. This
script samples the Python engine and writes the results to
``frontend/test/parity/fixtures.json``, which the vitest parity spec
(``frontend/test/parity/parity.test.ts``) then checks the TS engine against.

The fixtures cover these guarantees:

1. ``scalar`` — expression values at sample points. ``exact`` entries must match
   bit-for-bit (arithmetic; a raw OpenSimplex sample uses only arithmetic, so a
   single sample is reproducible exactly); ``approx`` entries (Sin/Cos/Log/…)
   may differ in the last ULP because libm and V8 round transcendentals
   differently.
2. ``serialize`` — canonical DSL strings. Both engines must emit identical text.
3. ``shape_grid`` — feathered shape/pattern fields over a small grid (the smooth
   coverage the scalar samples miss), at single-precision tolerance.
4. ``fbm_grid`` — a small fBm field, to confirm the gridded noise path matches.
5. ``relief`` — composited height + derived normal/AO maps.
6. ``pbr`` — the six OpenPBR channels (sheen/coat/transmission/ior/subsurface/
   anisotropy) composited across a multi-layer material and packed into the two
   RGBA8 maps the engines emit; asserted byte-for-byte exact.

Run:  python -m tests.parity.generate_fixtures

Parity contract (verified, with deliberate exceptions)
-------------------------------------------------------
- **Exact (bit-for-bit):** arithmetic, leaves, ``Threshold``,
  hard-edged ``Bricks``/``Weave`` (pure ``Floor``/mod arithmetic; the feathered
  ``Weave`` lattice compares with the feathered shape coverage below), ``Translate``/``Scale``,
  and a single OpenSimplex sample (``OpenSimplex(seed).noise2`` is identical
  because the 64-bit seed LCG is reproduced with ``BigInt`` in
  ``frontend/src/engine/opensimplex.ts``). Shape **fill decisions**
  (``Fill``/``Path`` winding, ``Rect``/``Ellipse`` interior) are also exact: the
  SDF/winding use only correctly-rounded IEEE ops, so the 0/1 result agrees
  bit-for-bit away from the boundary. The **height** composite
  (``maxᵢ { hᵢ : αᵢ > 0 }``) is exact in principle; the relief fixture uses
  single-precision tolerance because it mixes f32 sample axes with f64
  constants. The **OpenPBR channel packing** (the ``pbr`` fixture: alpha-blended
  across a multi-layer material and quantized into the two RGBA8 ``pbr1``/``pbr2``
  maps, ior normalized over ``[1, IOR_PACK_MAX]``) is asserted byte-for-byte exact,
  the quantization is integer-valued so the bytes agree despite the f32 blend.
- **Approx (~1e-11, last ULP):** transcendentals (``Sin``/``Cos``/``Log``/
  ``Sqrt``) differ between libm and V8; multi-octave ``fBm`` differs similarly
  (Python ``2/3**0.5`` vs TS ``2/Math.sqrt(3)``, plus per-octave summation
  rounding); ``Rotate`` (transcendental rotation matrix); and the smooth
  (feathered) coverage of shapes. Feathered **shape grid fixtures** compare at
  single-precision tolerance (~1e-6), like the fBm grid fixture, because the TS
  path stores into a ``Float32Array``. The **relief fixtures** (composited
  height, derived normal, derived AO) compare at ~1e-5 for the same reason (the
  TS relief math runs in ``Float32``). Cubic flattening is bit-identical across
  engines (uniform ``CUBIC_STEPS``, Horner form in float64).
- **NOT equal — known divergence:** **Worley noise**. The Python kernel hashes
  cells with a 64-bit integer mix (``noise.py:151`` ``_hash_to_01``); the TS port
  uses a 32-bit mix (``frontend/src/engine/noise.ts`` ``hash01``). The fields are
  statistically equivalent but not equal, so parity is intentionally not asserted
  for Worley. If you unify the hashes, add a Worley exact-parity fixture.
- **f32/f64 boundary caveat:** TS samples on ``Float32`` axes while Python uses
  ``float64``, so a hard shape edge can flip a single boundary pixel. Keep exact
  fixtures sampled away from edges; use feathered grid fixtures for the rest.
- **``Ref``/``Define``:** the Python ↔ TS material string is checked by a fixed
  canonical round-trip (``tests/engine/test_main.py`` +
  ``frontend/test/app/material-io.test.ts``); the expression-level ``serialize``
  parity covers the rest.
- **Evaluation backends (within-TS parity):** the off-thread CPU tiers
  (``worker``, tile ``pool``) run the identical ``eval-core`` engine, so they are
  byte-for-byte identical to the synchronous path, including tiled evaluation
  reassembled from row stripes (``frontend/test/eval/`` asserts this). They
  therefore inherit the cross-engine parity above unchanged. The WebGPU ``gpu``
  tier is intentionally NOT parity-faithful (GPU transcendentals/fma diverge
  beyond tolerance); it is a preview-only backend with a CPU fallback, never the
  authoritative frame, so no parity fixture is asserted for it.

Cross-engine change protocol
-----------------------------
After any engine-math change on either side, regenerate the committed JSON and
run both suites::

    python -m tests.parity.generate_fixtures   # then commit the JSON

When changing the DSL grammar or a layer channel, update both engines + the EBNF
in ``README.md``; see the cross-engine change protocol in
``matloom/engine/__init__.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from matloom.engine.main import LayeredMaterial
from matloom.engine.noise import fBm
from matloom.engine.parser import parse_expr
from matloom.engine.relief import composite_height, height_to_ao, height_to_normal
from matloom.engine.serialize import serialize_expr

# IOR is packed into a byte over the [1, IOR_PACK_MAX] range, mirroring the
# frontend's `IOR_PACK_MAX` in frontend/src/engine/material.ts.
IOR_PACK_MAX = 3.0

# Sample points shared by every scalar fixture.
_POINTS = [(0.0, 0.0), (0.1, 0.2), (0.5, 0.5), (0.75, 0.25), (1.0, 1.0)]

# (source, parity-mode). "exact" must match bit-for-bit; "approx" allows the
# last-ULP differences inherent to transcendental functions.
_SCALAR_CASES: list[tuple[str, str]] = [
    # Arithmetic & leaves — exact.
    ("5", "exact"),
    ("0.25", "exact"),
    ("X()", "exact"),
    ("Y()", "exact"),
    ("(X() + Y())", "exact"),
    ("(X() - Y())", "exact"),
    ("(X() * 2)", "exact"),
    ("(X() * Y())", "exact"),
    ("((X() + 1) / (Y() + 1))", "exact"),
    ("(X() ** 2)", "exact"),
    ("Min(X(), Y())", "exact"),
    ("Max(X(), Y())", "exact"),
    ("Abs((X() - Y()))", "exact"),
    ("Floor((X() * 10))", "exact"),
    ("Ceil((X() * 10))", "exact"),
    # threshold — exact (piecewise polynomial, same constants).
    ("Threshold(X(), below_at=0.2, above_at=0.8)", "exact"),
    (
        "Threshold(X(), below_at=0.2, below_to=0, transition_width=0.3)",
        "exact",
    ),
    # Bricks — pure Floor/arithmetic (hard edge exact; feathered uses a
    # polynomial smoothstep, marked approx out of caution).
    ("Bricks()", "exact"),
    ("Bricks(brick_width=0.4, brick_height=0.2, mortar=0.05)", "exact"),
    ('Bricks(offset=0.25, axis="column", feather=0.04)', "approx"),
    # Weave — pure Floor/mod arithmetic. The sample points fall on exact cell
    # boundaries or clear thread/gap interiors, so the hard-edge field is exact.
    ("Weave(over=2, under=1, base_freq=4, warp_width=0.5)", "exact"),
    # Transforms — separable axis rewrites; exact for an exact child, approx
    # when the child uses a transcendental.
    ("Translate(X(), 0.3, 0)", "exact"),
    ("Scale(X(), 2, 0.5)", "exact"),
    ("Translate(Bricks(), 0.1, 0.2)", "exact"),
    ("Scale(Sin(X()), 2, 1)", "approx"),
    # Rotate uses a transcendental rotation matrix → approx.
    ("Rotate(X(), 90)", "approx"),
    ("Rotate((X() + Y()), 30)", "approx"),
    ("Rotate(Bricks(mortar=0.1), 37)", "approx"),
    # Shapes — a feather=0 Fill is a hard 0/1 decided by the SDF sign, which is
    # computed with correctly-rounded IEEE ops, so it matches bit-for-bit (the
    # smooth feathered values are checked by the shape grid fixtures below).
    ("Fill(Rect(0.2, 0.2, 0.6, 0.6))", "exact"),
    ("Fill(Ellipse(0.5, 0.5, 0.3, 0.2))", "exact"),
    ("Fill(Rect(0, 0, 1, 1, 0.2))", "exact"),
    # Path fill is decided by an integer winding test (no sqrt) → exact.
    ("Fill(Path(0.2, 0.2, LineTo(0.8, 0.2), LineTo(0.5, 0.8)))", "exact"),
    # Noise — approx. The single OpenSimplex sample is bit-identical across
    # engines, but multi-octave fBm diverges in the last ULP: Python's
    # _FACTOR = 2 / (3 ** 0.5) uses pow() while the TS port uses Math.sqrt(3),
    # and the per-octave summation rounds at the 16th digit. The tolerance is
    # still ~1e-12, far tighter than any real bug would survive.
    ("fBm(seed=5)", "approx"),
    ("fBm(octaves=3, base_freq=4, seed=11)", "approx"),
    ("fBm(to_01=True, seed=7)", "approx"),
    # Transcendentals — approx (libm vs V8 last-ULP).
    ("Sin(X())", "approx"),
    ("Cos((X() * 3.14159))", "approx"),
    ("Sqrt((X() + 1))", "approx"),
    ("Log((X() + 1), 10)", "approx"),
    ("Log((X() + 2))", "approx"),
]

# Sources whose canonical serialization both engines must reproduce identically.
_SERIALIZE_CASES = [
    "5",
    "0.25",
    "X()",
    "-X()",
    "(X() + Y())",
    "(X() * 2)",
    "Sin(X())",
    "Min(X(), Y())",
    "Log(X(), 10)",
    "Threshold(X(), below_at=0.2, above_at=0.8)",
    "fBm(octaves=4, base_freq=2, seed=9)",
    # Set-tracking: explicit defaults the author wrote are preserved, and
    # per-axis frequencies emit by name.
    "fBm(octaves=6, base_freq=2, seed=9)",
    "fBm(base_freq=2, to_01=False, seed=9)",
    "fBm(base_freq_x=2, seed=9)",
    "fBm(base_freq_x=2, base_freq_y=3, seed=9)",
    'Worley(distance="euclidean", combination="F1", seed=4)',
    'Worley(distance="manhattan", combination="F2", seed=4)',
    "Bricks()",
    'Bricks(brick_width=2, brick_height=1, offset=0.25, mortar=0.1, axis="column", feather=0.02)',
    "Weave()",
    "Weave(over=2, under=1, shift=2, base_freq=12, warp_width=0.6, feather=0.05)",
    "Translate(X(), 0.3, 0)",
    "Scale(X(), 2, 0.5)",
    "Rotate(X(), 45)",
    "Fill(Rect(0.2, 0.2, 0.6, 0.6))",
    "Fill(Rect(0, 0, 1, 1, 0.1), feather=0.02)",
    'Fill(Ellipse(0.5, 0.5, 0.3, 0.2), mode="EvenOdd")',
    "Stroke(Rect(0.2, 0.2, 0.6, 0.6), 0.04, feather=0.01)",
    "Stroke(Ellipse(0.5, 0.5, 0.3, 0.2), 0.05)",
    "Fill(Path(0.2, 0.2, LineTo(0.8, 0.2), LineTo(0.5, 0.8)))",
    "Stroke(Path(0.1, 0.5, CubicTo(0.3, 0.9, 0.7, 0.1, 0.9, 0.5)), 0.05)",
]


# Feathered shape / pattern fields over a small grid — these exercise the smooth
# coverage values (and the f32/f64 boundary behavior) that the 5-point scalar
# samples above miss. Compared at single-precision tolerance like the fBm grid.
_SHAPE_GRID_CASES = [
    "Fill(Rect(0.2, 0.2, 0.6, 0.5), feather=0.06)",
    "Fill(Ellipse(0.5, 0.5, 0.3, 0.2), feather=0.06)",
    "Stroke(Rect(0.2, 0.2, 0.6, 0.5), 0.06, feather=0.03)",
    "Fill(Path(0.2, 0.2, LineTo(0.8, 0.2), LineTo(0.5, 0.8)), feather=0.05)",
    "Stroke(Path(0.1, 0.5, CubicTo(0.3, 0.9, 0.7, 0.1, 0.9, 0.5)), 0.06, feather=0.03)",
    "Weave(over=2, under=1, shift=1, base_freq=5, warp_width=0.6, feather=0.15)",
]


def _scalar_fixtures() -> list[dict]:
    out = []
    for src, mode in _SCALAR_CASES:
        expr = parse_expr(src)
        values = [expr(x, y) for x, y in _POINTS]
        out.append({"src": src, "mode": mode, "values": values})
    return out


def _serialize_fixtures() -> list[dict]:
    out = []
    for src in _SERIALIZE_CASES:
        out.append({"src": src, "serialized": serialize_expr(parse_expr(src))})
    return out


def _shape_grid_fixtures() -> list[dict]:
    # float32 axes fix the sample points to exactly what the TS engine uses; the
    # reference is then computed in float64 (an exact widening of those points),
    # so the only cross-engine gap is the TS Float32Array store (~1e-7).
    xs = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    out = []
    for src in _SHAPE_GRID_CASES:
        grid = parse_expr(src).eval_grid(xs.astype(np.float64), ys.astype(np.float64))
        out.append(
            {
                "src": src,
                "xs": xs.tolist(),
                "ys": ys.tolist(),
                "grid": grid.flatten().tolist(),
            }
        )
    return out


def _fbm_grid_fixture() -> dict:
    f = fBm(seed=5)
    # Use float32 axes to mirror the real engine (LayeredMaterial uses float32
    # sample axes) and the TS engine (which always samples on Float32 axes).
    xs = np.linspace(0.0, 2.0, 8, dtype=np.float32)
    ys = np.linspace(0.0, 2.0, 6, dtype=np.float32)
    grid = f.eval_grid(xs, ys)
    return {
        "seed": 5,
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "grid": grid.flatten().tolist(),  # row-major (ys.size * xs.size)
    }


def _relief_fixtures() -> dict:
    """Height compositing (max) and the derived normal / AO maps. The
    height-composite is compared at single precision (smooth alphas, so no hard
    f32/f64 edge flips); normal/AO use a fixed synthetic field with exactly
    representable values so the only gap is float32-vs-float64 derivation."""
    xs = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    # Two layers exercising the coverage-mask composite H = maxᵢ{ hᵢ : αᵢ > 0 }.
    # Layer 0 (α=1) is present everywhere with height 8X-4 (spans -4 → +4). Layer 1
    # has α=(1-X), which fades to 0 by X=1, and a constant height -1: where present
    # it contributes its FULL -1 (opacity does not scale it), and at X=1 it is
    # excluded entirely (α=0). This locks in both the negative-height path (a carve
    # below the substrate) and the new coverage-gate semantics — a positive-only or
    # α-scaled fixture would not distinguish them.
    layers = [("1", "(X() * 8 - 4)"), ("(1 - X())", "(0 - 1)")]
    # Mirror the real pipeline (main.py casts every grid to its float dtype before
    # compositing); a bare `Constant` expression like "1" otherwise yields int64.
    alpha_grids = [
        parse_expr(a).eval_grid(xs, ys).astype(np.float64) for a, _ in layers
    ]
    height_grids = [
        parse_expr(h).eval_grid(xs, ys).astype(np.float64) for _, h in layers
    ]
    h_comp = composite_height(alpha_grids, height_grids)

    # A fixed synthetic height field (exactly representable values) for the
    # normal/AO derivation: a +4 plateau and a -2 pit (below-substrate terrain),
    # so the derivation is exercised on both signs. Plus lateral spacing + AO
    # radius.
    hf = np.zeros((6, 8), dtype=np.float64)
    hf[2:4, 3:6] = 4.0
    hf[4:6, 1:3] = -2.0
    dx, dy, radius = 10.0, 10.0, 2
    nrm = height_to_normal(hf, dx, dy, y_up=True)
    ao = height_to_ao(hf, dx, radius=radius, strength=1.0)
    # Anisotropic AO (radius_x ≠ radius_y) locks the per-axis box blur the
    # crop-invariant path relies on: under a crop the AO radius scales with the
    # zoom, and a non-uniform crop scales the two axes differently.
    radius_x_aniso, radius_y_aniso = 4, 2
    ao_aniso = height_to_ao(
        hf, dx, radius=radius_x_aniso, strength=1.0, radius_y=radius_y_aniso
    )
    # Fractional AO radii: the crop zoom is continuous, so the AO radius
    # (AO_RADIUS · zoom) is generally non-integer. These lock the fractional
    # box-blur lerp path (per-axis blend of the enclosing integer blurs) that
    # keeps cavity darkening continuous under a crop — no 1-texel `floor` step.
    radius_frac = 2.5
    ao_fractional = height_to_ao(hf, dx, radius=radius_frac, strength=1.0)
    radius_x_frac, radius_y_frac = 3.5, 1.5
    ao_fractional_aniso = height_to_ao(
        hf, dx, radius=radius_x_frac, strength=1.0, radius_y=radius_y_frac
    )

    return {
        "height_composite": {
            "xs": xs.tolist(),
            "ys": ys.tolist(),
            "layers": [{"alpha": a, "height": h} for a, h in layers],
            "grid": h_comp.flatten().tolist(),
        },
        "normal": {
            "w": 8,
            "h": 6,
            "dx": dx,
            "dy": dy,
            "y_up": True,
            "h_grid": hf.flatten().tolist(),
            "nx": nrm[..., 0].flatten().tolist(),
            "ny": nrm[..., 1].flatten().tolist(),
            "nz": nrm[..., 2].flatten().tolist(),
        },
        "ao": {
            "w": 8,
            "h": 6,
            "dx": dx,
            "radius": radius,
            "strength": 1.0,
            "h_grid": hf.flatten().tolist(),
            "ao": ao.flatten().tolist(),
        },
        "ao_aniso": {
            "w": 8,
            "h": 6,
            "dx": dx,
            "radiusX": radius_x_aniso,
            "radiusY": radius_y_aniso,
            "strength": 1.0,
            "h_grid": hf.flatten().tolist(),
            "ao": ao_aniso.flatten().tolist(),
        },
        "ao_fractional": {
            "w": 8,
            "h": 6,
            "dx": dx,
            "radius": radius_frac,
            "strength": 1.0,
            "h_grid": hf.flatten().tolist(),
            "ao": ao_fractional.flatten().tolist(),
        },
        "ao_fractional_aniso": {
            "w": 8,
            "h": 6,
            "dx": dx,
            "radiusX": radius_x_frac,
            "radiusY": radius_y_frac,
            "strength": 1.0,
            "h_grid": hf.flatten().tolist(),
            "ao": ao_fractional_aniso.flatten().tolist(),
        },
    }


def _to_byte_grid(v: np.ndarray) -> list[int]:
    """Mirror the frontend's ``toByte`` (frontend/src/engine/grid.ts):
    ``Math.min(255, Math.max(0, (x * 255 + 0.5) | 0))`` — clamp to [0,1] then
    truncate toward zero. ``| 0`` floors the already-non-negative product, so
    ``np.floor`` reproduces it exactly."""
    q = np.floor(np.clip(v, 0.0, 1.0) * 255.0 + 0.5)
    return np.clip(q, 0, 255).astype(np.uint8).flatten().tolist()


def _pbr_fixtures() -> dict:
    """The OpenPBR reflectance channels (sheen/coat/transmission/ior/subsurface/
    anisotropy) packed into the two RGBA8 maps the frontend emits — pbr1 =
    (sheen, coat, transmission, subsurface), pbr2 = (anisotropy, ior-normalized,
    0, 255). A multi-layer material with constant AND expression-driven channels
    exercises the alpha-weighted over-blend across layers and the ior
    normalization, so the bytes must match the TS engine's evaluateMaterial
    exactly. (No earlier fixture covered these six channels at all.)

    The byte quantization is integer-valued; the alpha here is a smooth partial
    coverage (Threshold of X(), in [0.4, 0.6], so BOTH layers always contribute
    to the over-blend), and the float32 accumulation matches the TS Float32Array
    path, so this is asserted EXACT in the TS spec."""
    program = (
        "View(0, 0, 1, 1)\n"
        "Material(\n"
        "  Layer(1)\n"
        "    .basecolor(120, 120, 120).metallic(0.2).roughness(0.5)\n"
        "    .sheen(0.1).coat(0.2).transmission(0.3).ior(1.4)"
        ".subsurface(0.4).anisotropy(0.5),\n"
        "  Layer(Threshold(X(), below_at=0.4, above_at=0.6))\n"
        "    .basecolor(200, 50, 50).roughness(Y())\n"
        "    .sheen(X()).coat((X() * Y())).transmission(0.7).ior(2.0)\n"
        "    .subsurface(Y()).anisotropy((1 - X()))\n"
        ")"
    )
    m = LayeredMaterial.deserialize(program)
    w = h = 6
    # Mirror the export/TS sampling: xs left→right, ys reversed for y-up.
    xs = m._generate_axis_coordinates(w, 0.0, 1.0)
    ys = m._generate_axis_coordinates(h, 0.0, 1.0, reverse=True)
    alpha_grids = [
        layer.alpha.eval_grid(xs, ys).astype(m._dtype) for layer in m._layers
    ]
    alpha, Vs = m._resolve_alpha(alpha_grids)

    def blend(attr: str) -> np.ndarray:
        return m._weighted_blend(
            xs, ys, alpha, Vs, [getattr(layer, attr) for layer in m._layers]
        )

    sheen = _to_byte_grid(blend("sheen"))
    coat = _to_byte_grid(blend("coat"))
    transmission = _to_byte_grid(blend("transmission"))
    subsurface = _to_byte_grid(blend("subsurface"))
    anisotropy = _to_byte_grid(blend("anisotropy"))
    ior_norm = _to_byte_grid((blend("ior") - 1.0) / (IOR_PACK_MAX - 1.0))

    n = w * h
    pbr1 = [0] * (n * 4)
    pbr2 = [0] * (n * 4)
    for i in range(n):
        pbr1[i * 4] = sheen[i]
        pbr1[i * 4 + 1] = coat[i]
        pbr1[i * 4 + 2] = transmission[i]
        pbr1[i * 4 + 3] = subsurface[i]
        pbr2[i * 4] = anisotropy[i]
        pbr2[i * 4 + 1] = ior_norm[i]
        pbr2[i * 4 + 2] = 0
        pbr2[i * 4 + 3] = 255

    return {
        "program": program,
        "w": w,
        "h": h,
        "y_up": True,
        "pbr1": pbr1,
        "pbr2": pbr2,
    }


def main() -> Path:
    fixtures = {
        "points": _POINTS,
        "scalar": _scalar_fixtures(),
        "serialize": _serialize_fixtures(),
        "shape_grid": _shape_grid_fixtures(),
        "fbm_grid": _fbm_grid_fixture(),
        "relief": _relief_fixtures(),
        "pbr": _pbr_fixtures(),
    }
    out_path = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "test"
        / "parity"
        / "fixtures.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixtures, indent=2))
    return out_path


if __name__ == "__main__":
    path = main()
    print(f"wrote parity fixtures to {path}")
