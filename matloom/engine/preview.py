"""No-Blender fixed-light preview renderer.

A fast, software-agnostic RGB preview of a :class:`LayeredMaterial` — pure numpy,
no Blender, no torch, no file I/O. It composites the unlit channel maps in
memory (the same over-operator + height relief the authoritative
:mod:`matloom.engine.main` export uses) and shades them under a single fixed
directional light with a straight-on camera, approximating roughness, metallic,
clearcoat, subsurface, sheen, transmission, and emissive.

Purpose: it is the renderer the generation pipeline's vision critic and the
polish verifier see, so the text→DSL **method** does not depend on Blender and
can run anywhere (including a future browser port). It is a fast fitness/
critique surface, not a faithful lit render: there is no global illumination,
no Fresnel at grazing angles, no real refraction, and the specular lobe is a
Blinn-Phong proxy. Every Layer channel is composited and consumed (alpha
coverage composites over a mid-gray background, emissive is soft-knee tone-mapped
to preserve the HDR strength axis).

Returns ``uint8`` ``(H, W, 3)`` **RGB** (note: the LLM vision client in
:mod:`matloom.utils.llm.msg` expects BGR, so callers feeding a vision model must
``cv2.cvtColor(..., cv2.COLOR_RGB2BGR)``; the CLIP/MobileCLIP2 path wants RGB
and consumes the output directly).

Each composite runs inside a per-render ``Ref``-boundary ``eval_grid`` memo
(:func:`matloom.engine.expr.grid_cache_scope`), so a ``Define``'d subtree
referenced from several channels evaluates once per render; pass ``memo=False``
to :func:`fixed_light_preview`/:func:`_composite` to render without it (for
bisection — the result is bit-identical either way).
"""

from contextlib import nullcontext

import numpy as np

from matloom.engine.expr import grid_cache_scope
from matloom.engine.main import LayeredMaterial
from matloom.utils.dtypes import SDict

# A single fixed directional light + straight-on camera, so there are no
# scene/lighting confounds a verifier cannot attribute. Roughness, metallic,
# AO, emissive, and transmission are approximated (disclosed): this is a fast
# fitness surface, not a faithful lit render.
_LIGHT = np.array([0.5, 0.5, 1.0])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)
_VIEW = np.array([0.0, 0.0, 1.0])
_HALF = _LIGHT + _VIEW
_HALF = _HALF / np.linalg.norm(_HALF)
_BG = 0.5  # transmission + alpha-coverage background (mid-gray)


def _composite(
    material: LayeredMaterial, width: int, height: int, memo: bool = True
) -> SDict[np.ndarray]:
    """Composite the unlit channel maps in memory (no Blender, no file I/O).
    Runs inside the per-render ``Ref``-boundary ``eval_grid`` memo unless
    ``memo=False`` (the bisection escape hatch; the result is bit-identical
    either way)."""
    from matloom.engine.relief import (
        composite_height,
        height_to_ao,
        height_to_normal,
    )
    from matloom.engine.util import linear_to_srgb

    with grid_cache_scope() if memo else nullcontext():
        (x1, y1, x2, y2) = material._view
        xs = material._generate_axis_coordinates(width, x1, x2)
        ys = material._generate_axis_coordinates(height, y1, y2, reverse=True)
        layers = material._layers
        alpha_grids = [
            layer.alpha.eval_grid(xs, ys).astype(np.float64) for layer in layers
        ]
        alpha, vs = material._resolve_alpha(
            [a.astype(material._dtype) for a in alpha_grids]
        )
        alpha = alpha.astype(np.float64)
        vs = vs.astype(np.float64)

        def blend(samplers):
            b = np.zeros((ys.size, xs.size), dtype=np.float64)
            for i, s in enumerate(samplers):
                b += vs[i] * s.eval_grid(xs, ys).astype(np.float64)
            mask = alpha > 0
            b[mask] /= alpha[mask]
            return b

        rgb_lin = np.stack(
            [
                blend([getattr(layer.basecolor, ch) for layer in layers])
                for ch in ("r", "g", "b")
            ],
            axis=-1,
        )
        albedo = linear_to_srgb(rgb_lin)  # 0-1 sRGB
        rough = blend([layer.roughness for layer in layers])
        metal = blend([layer.metallic for layer in layers])
        trans = blend([layer.transmission for layer in layers])
        emiss_lin = np.stack(
            [
                blend(
                    [
                        getattr(layer.emissive, ch) * layer.emissive.strength
                        for layer in layers
                    ]
                )
                for ch in ("r", "g", "b")
            ],
            axis=-1,
        )
        # Emissive is a linear HDR quantity (main.py _export_emissive writes it
        # unclipped for Blender to tonemap). Hard-clipping here would flatten the
        # >1 strength axis (strength=5 reading identical to strength=1) and give
        # CMA-ES no gradient on bright emitters. Soft-knee tonemap [0, inf) -> [0,
        # 1) preserves the strength ordering, then sRGB-encode to match albedo.
        emiss = linear_to_srgb(emiss_lin / (1.0 + emiss_lin))
        sheen = blend([layer.sheen for layer in layers])
        coat = blend([layer.coat for layer in layers])
        subsurface = blend([layer.subsurface for layer in layers])
        anisotropy = blend([layer.anisotropy for layer in layers])
        ior = blend([layer.ior for layer in layers])
        dx = abs(x2 - x1) / width
        dy = abs(y2 - y1) / height
        h = composite_height(
            [a.astype(np.float64) for a in alpha_grids],
            [layer.height.eval_grid(xs, ys).astype(np.float64) for layer in layers],
        )
        normal = height_to_normal(
            h.astype(np.float32), dx, dy, y_up=True
        )  # (H, W, 3) (nx, ny, nz)
        ao = height_to_ao(h.astype(np.float32), dx)
        return {
            "albedo": albedo,
            "rough": rough,
            "metal": metal,
            "trans": trans,
            "emiss": emiss,
            "sheen": sheen,
            "coat": coat,
            "subsurface": subsurface,
            "anisotropy": anisotropy,
            "ior": ior,
            "normal": normal,
            "ao": ao,
            "alpha": alpha,
        }


def fixed_light_preview(
    material: LayeredMaterial,
    width: int = 256,
    height: int = 256,
    memo: bool = True,
) -> np.ndarray:
    """Render a fast fixed-light RGB preview (uint8 ``(H, W, 3)``) of the
    material. No Blender. Returns RGB; see the module docstring for the
    BGR-vs-RGB caller contract. ``memo=False`` disables the per-render
    ``eval_grid`` memo (bisection escape hatch; bit-identical result)."""
    c = _composite(material, width, height, memo=memo)
    albedo = c["albedo"]
    # Coverage alpha from the over-operator (main.py writes it as basecolor's
    # 4th channel so the Blender render shows the scene background through
    # gaps). Composite the shaded material over the same background by it, so
    # partial-coverage texels read as background rather than black/over-opaque.
    alpha = np.clip(c["alpha"], 0.0, 1.0)
    rough = np.clip(c["rough"], 0.0, 1.0)
    metal = np.clip(c["metal"], 0.0, 1.0)
    trans = np.clip(c["trans"], 0.0, 1.0)
    sheen = np.clip(c["sheen"], 0.0, 1.0)
    coat = np.clip(c["coat"], 0.0, 1.0)
    subsurface = np.clip(c["subsurface"], 0.0, 1.0)
    anisotropy = np.clip(c["anisotropy"], 0.0, 1.0)
    ior = np.clip(c["ior"], 1.0, 3.0)
    emiss = c["emiss"]
    n = c["normal"]
    nz = np.clip(np.where(np.isfinite(n[..., 2]), n[..., 2], 1.0), -1.0, 1.0)
    nx = np.where(np.isfinite(n[..., 0]), n[..., 0], 0.0)
    ny = np.where(np.isfinite(n[..., 1]), n[..., 1], 0.0)
    ndotl = np.clip(nx * _LIGHT[0] + ny * _LIGHT[1] + nz * _LIGHT[2], 0.0, 1.0)
    ndoth = np.clip(nx * _HALF[0] + ny * _HALF[1] + nz * _HALF[2], 0.0, 1.0)
    ndotv = nz  # camera straight-on
    # Roughness -> Blinn-Phong shininess (mirror at roughness 0, broad at 1). The
    # range is mild so a specular highlight is visible on a near-flat surface
    # (a point light + perfectly flat mirror would otherwise be sub-pixel).
    shininess = (1.0 - rough) ** 2 * 100.0 + 2.0
    # Anisotropy (brushed/satin) softens the specular lobe. A true anisotropic
    # stretch needs a tangent frame the engine does not expose here; broadening
    # is a faithful-enough preview that "satin reads softer than mirror".
    shininess = shininess / (1.0 + anisotropy * 4.0)
    spec = ndoth**shininess
    # Schlick F0 from IOR for dielectrics: ((1-ior)/(1+ior))^2; metals tint by albedo.
    f0_dielectric = ((1.0 - ior) / (1.0 + ior)) ** 2
    f0 = f0_dielectric[..., None] * (1.0 - metal)[..., None] + albedo * metal[..., None]
    specular = f0 * spec[..., None]
    # Clearcoat: a dielectric reflection on top. A sharp Blinn-Phong lobe is
    # sub-pixel on a flat surface, so the visible effect is the normal-incidence
    # reflection (F0 * coat, uniform) with a mild lobe for relief.
    specular = specular + (0.04 * coat * (0.5 + 0.5 * ndoth**50.0))[..., None]
    # Subsurface: wrap lighting softens the diffuse terminator (light bleeds
    # around the edge); at subsurface=0 this is plain ndotl.
    wrap = np.clip((ndotl + subsurface) / (1.0 + subsurface), 0.0, 1.0)
    diffuse = albedo * (1.0 - metal)[..., None] * (wrap * c["ao"])[..., None]
    # Sheen: a grazing rim (fabric fuzz), strongest where the normal turns away
    # from the camera, plus a faint flat brightening so it reads on a swatch.
    sheen_rim = sheen * (1.0 - ndotv) ** 5
    diffuse = diffuse + albedo * sheen[..., None] * 0.1 + sheen_rim[..., None] * 0.5
    # A small ambient fill so metals (which have no diffuse) still show their
    # albedo to the verifier; without it a flat metal renders black.
    ambient = albedo * 0.2
    shaded = ambient + diffuse + specular + emiss
    # Transmission: mix the material toward a fixed background (rough
    # refraction proxy), then composite that material color over the same
    # background by the coverage alpha (gaps show background, not black).
    mat = shaded * (1.0 - trans[..., None]) + _BG * trans[..., None]
    out = mat * alpha[..., None] + _BG * (1.0 - alpha[..., None])
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)
