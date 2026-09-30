"""Objective *appearance descriptors* for layered-material programs.

The grammar check and the parse-repair loop guarantee a *valid* program, not a
*sensible* one. Rather than classify a material as "bad" with hard thresholds
(which false-positive on prompts that legitimately want something flat,
transparent, or low-contrast), this module computes **objective, interpretable
scalars** from the engine's own **unlit composited channel maps** and lets the
prompt-aware critic decide whether they are appropriate.

Everything is computed from the composited albedo, the scalar PBR channels
(roughness, metallic, sheen, coat, transmission, subsurface, anisotropy), IOR,
emissive, and the relief-derived normal/AO — **no scene lighting, no Blender**.
A lit render introduces confounds a critic cannot attribute (a bright spot from
a specular highlight or scene geometry, not the material); the unlit channel
maps describe the material itself, objectively, and are fair across methods
(any method's output can be described this way, not just a DSL program).

Scalars are reported with one-line definitions (see :data:`DESCRIPTOR_DEFINITIONS`
and :class:`Descriptor`) so a critic — text-only or vision — can read them
without any out-of-band knowledge. The module is torch-free, Blender-free,
cv2-free, and deterministic; message formatting lives in the caller
(:mod:`matloom.engine.text2dsl`).
"""

import numpy as np
from pydantic import BaseModel

from matloom import logger
from matloom.engine.main import LayeredMaterial
from matloom.engine.relief import (
    composite_height,
    height_to_ao,
    height_to_normal,
)
from matloom.engine.util import linear_to_srgb
from matloom.utils.dtypes import SDict

# Channels that composite as alpha-weighted [0, 1] scalars (same set as
# ``LayeredMaterial._export_scalar``).
_SCALAR_CHANNELS: tuple[str, ...] = (
    "metallic",
    "roughness",
    "sheen",
    "coat",
    "transmission",
    "subsurface",
    "anisotropy",
)

# One-line definition for every emitted scalar, in the order they are reported.
# A critic sees the caller-built line: ``- {label}: {formatted} — {definition}``.
DESCRIPTOR_DEFINITIONS: SDict[str] = {
    # --- albedo (the composited base color, sRGB) --------------------------- #
    "albedo_luminance_mean": "mean luminance 0-255 (brightness)",
    "albedo_luminance_std": "luminance std 0-255 (tonal variation; 0 = one tone)",
    "albedo_rms_contrast": "luminance std/mean (overall contrast; higher = more contrast)",
    "albedo_saturation": "mean saturation 0-1 (0 = gray; 1 = a channel is fully off, vivid)",
    "albedo_edge_density": "mean luminance gradient (fine-detail level; higher = more detail, unbounded)",
    "albedo_dominant_freq": "FFT peak in cycles/image (pattern scale; 0 = no pattern; higher = finer features)",
    "albedo_freq_bandwidth": "FFT radial spread 0-1 (narrow = single scale, wide = multi-scale)",
    "albedo_anisotropy": "1 - minor/major eigenvalue ratio 0-1 (0 = isotropic, 1 = directional)",
    "albedo_color_span": "max per-channel range 0-255 (dynamic-range usage; 0 = one color)",
    "albedo_tonal_entropy": "luminance histogram entropy 0-1 (0 = one tone, 1 = flat distribution)",
    # --- roughness / metallic ------------------------------------------------ #
    "roughness_mean": "mean roughness 0-1 (0 = mirror, 1 = matte)",
    "roughness_std": "roughness std 0-1 (gloss variation across the surface)",
    "metallic_mean": "mean metallic 0-1 (0 = dielectric, 1 = metal)",
    "metallic_coverage": "fraction of texels with metallic > 0.5",
    # --- relief (height, normal, AO) ----------------------------------------- #
    "height_range": "relief range in coordinate units (0 = flat; drives normal/AO)",
    "height_std": "relief std in coordinate units",
    "height_dominant_freq": "relief FFT peak in cycles/image (0 = no relief pattern; higher = finer)",
    "height_anisotropy": "1 - minor/major eigenvalue ratio 0-1 (0 = isotropic bumps, 1 = directional ridges)",
    "normal_mean_tilt_deg": "mean surface tilt angle from flat in degrees (0 = flat)",
    "ao_mean": "mean ambient occlusion 0-1 (1 = no crevices, lower = darker crevices)",
    "ao_std": "ambient-occlusion std 0-1 (crease variation)",
    # --- the other PBR channels (mean only; 0/1.5 = defaults) ---------------- #
    "sheen_mean": "mean sheen 0-1 (cloth/velvet fuzz; 0 = none)",
    "coat_mean": "mean clearcoat 0-1 (lacquer/glaze; 0 = none)",
    "transmission_mean": "mean transmission 0-1 (0 = opaque, 1 = clear glass)",
    "subsurface_mean": "mean subsurface 0-1 (0 = none, 1 = full scatter)",
    "anisotropy_mean": "mean anisotropy 0-1 (brushed/satin directionality; 0 = none)",
    "ior_mean": "mean index of refraction (>= 1; 1.5 = typical dielectric)",
    "emissive_mean": "mean emissive luminance, linear (0 = not glowing; higher = brighter; can exceed 1 for HDR glow; not directly comparable to the sRGB albedo luminance)",
}


# Initialisms in descriptor keys that are uppercased in the human label.
_ACRONYMS = {"ao", "ior", "rms"}


def _label(name: str) -> str:
    """Human-readable label for a descriptor key: underscores -> spaces,
    initialisms uppercased, first letter capitalized."""
    words = [w.upper() if w in _ACRONYMS else w for w in name.split("_")]
    words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


def _fmt(v: float) -> str:
    """Compact, readable scalar formatting: 3 sig figs, no trailing noise."""
    if v == 0.0:
        return "0"
    if abs(v) >= 100.0:
        return f"{v:.1f}"
    if abs(v) >= 10.0:
        return f"{v:.2f}"
    return f"{v:.3f}"


class Descriptor(BaseModel):
    """One objective appearance scalar with its human label and definition.

    ``name`` is the machine key (e.g. ``albedo_rms_contrast``; for ``record.json``
    and grep); ``label`` and ``formatted`` derive from it for the critic message.
    """

    name: str
    value: float
    definition: str

    @property
    def label(self) -> str:
        return _label(self.name)

    @property
    def formatted(self) -> str:
        return _fmt(self.value)


# --------------------------------------------------------------------------- #
# Scalar computations (pure numpy; no cv2, no torch, no Blender)
# --------------------------------------------------------------------------- #
def _luminance_srgb(rgb01: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of sRGB values in [0, 1], returned in [0, 1]."""
    return 0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2]


def _saturation(rgb01: np.ndarray) -> np.ndarray:
    """Per-pixel saturation in [0, 1]: 1 - min/max per pixel (0 on gray/black)."""
    mx = rgb01.max(axis=-1)
    mn = rgb01.min(axis=-1)
    out = np.zeros_like(mx)
    nz = mx > 0
    out[nz] = 1.0 - (mn[nz] / mx[nz])
    return out


def _edge_density(lum01: np.ndarray, mask: np.ndarray) -> float:
    """Mean gradient magnitude of the luminance over the mask INTERIOR (a
    unitless fine-detail level). The mask is eroded by one texel so the
    material/background boundary is not counted as a fake edge."""
    if not mask.any():
        return 0.0
    interior = mask.copy()
    interior[1:] &= mask[:-1]
    interior[:-1] &= mask[1:]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    if not interior.any():
        return 0.0
    gy, gx = np.gradient(lum01)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag[interior].mean())


def _tonal_entropy(lum01: np.ndarray, mask: np.ndarray, bins: int = 32) -> float:
    """Normalized luminance histogram entropy over the material region, 0-1."""
    if not mask.any():
        return 0.0
    hist, _ = np.histogram(lum01[mask], bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-(p * np.log2(p)).sum() / np.log2(bins))


def _fft_features(arr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """``(dominant_freq, bandwidth, anisotropy)`` of a 2D field over ``mask``.

    ``dominant_freq`` is the FFT peak radial bin in cycles/image (0 = no pattern;
    excludes DC); ``bandwidth`` is the magnitude-weighted radial std, normalized
    by the max radius (0-1, 0 = single scale); ``anisotropy`` is 1 - lambda2/lambda1
    of the energy covariance about the DC (0 = isotropic, 1 = directional).

    The mean-removed field is multiplied by the mask before the FFT, so texels
    outside the material are 0 rather than a constant ``-mean`` step (a constant
    over a rectangle would inject a box-shaped spectrum). This is an honest
    improvement, not a claim of correctness: masked spectral analysis is hard,
    so for partial-coverage materials the FFT features are approximate and the
    boundary still contributes some leakage.
    """
    if not mask.any() or min(arr.shape) < 4:
        return 0.0, 0.0, 0.0
    a = arr.astype(np.float64)
    a = (a - float(a[mask].mean())) * mask  # zero outside the material region
    spec = np.fft.fftshift(np.abs(np.fft.fft2(a)))
    ny, nx = spec.shape
    cy, cx = ny // 2, nx // 2
    rmax = int(min(cy, cx))
    if rmax < 2:
        return 0.0, 0.0, 0.0
    yy, xx = np.indices((ny, nx))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rint = r.astype(np.int64)
    acc = np.bincount(rint.ravel(), weights=spec.ravel(), minlength=rmax + 1)
    cnt = np.bincount(rint.ravel(), minlength=rmax + 1).astype(np.float64)
    cnt[cnt == 0] = 1.0
    radial = acc / cnt
    radial[0] = 0.0  # exclude DC
    band = radial[1 : rmax + 1]
    # A flat (or all-zero) field has no pattern: report 0, not the argmax bin.
    dom = 0 if float(band.max()) <= 1e-12 else int(np.argmax(band)) + 1
    wsum = float(band.sum())
    if wsum > 0:
        pos = np.arange(1, rmax + 1, dtype=np.float64)
        mean_r = float((band * pos).sum() / wsum)
        var_r = float((band * (pos - mean_r) ** 2).sum() / wsum)
        bw = float(np.sqrt(var_r)) / rmax
    else:
        bw = 0.0
    energy = spec * spec
    total = float(energy.sum())
    if total <= 0.0:
        return float(dom), bw, 0.0
    dxx = xx - cx
    dyy = yy - cy
    mxx = float((energy * dxx * dxx).sum() / total)
    myy = float((energy * dyy * dyy).sum() / total)
    mxy = float((energy * dxx * dyy).sum() / total)
    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = np.sqrt(max(tr * tr / 4.0 - det, 0.0))
    l1 = tr / 2.0 + disc
    l2 = tr / 2.0 - disc
    aniso = 0.0 if l1 <= 0 else float(1.0 - l2 / l1)
    return float(dom), bw, aniso


# --------------------------------------------------------------------------- #
# Descriptor computation
# --------------------------------------------------------------------------- #
def describe(
    material: LayeredMaterial, *, width: int = 128, height: int = 128
) -> SDict[float]:
    """Evaluate ``material`` on a ``width`` x ``height`` grid over its own
    ``View`` and return the objective appearance scalars (keys in
    :data:`DESCRIPTOR_DEFINITIONS`). All scalars are computed over the
    material region (texels where the total alpha is > 0)."""
    layers = material._layers
    if not layers:
        return {k: 0.0 for k in DESCRIPTOR_DEFINITIONS}

    (x1, y1, x2, y2) = material._view
    xs = material._generate_axis_coordinates(width, x1, x2)
    ys = material._generate_axis_coordinates(height, y1, y2, reverse=True)
    alpha_grids = [layer.alpha.eval_grid(xs, ys).astype(np.float64) for layer in layers]
    alpha, vs = material._resolve_alpha(
        [a.astype(material._dtype) for a in alpha_grids]
    )
    alpha = alpha.astype(np.float64)
    vs = vs.astype(np.float64)
    mask = alpha > 1e-3
    if not mask.any():
        return {k: 0.0 for k in DESCRIPTOR_DEFINITIONS}

    def blend(samplers) -> np.ndarray:
        # A degenerate channel expression (Sqrt(-1) -> NaN, 1/0 -> inf, Log(0)
        # -> -inf) survives the layer's np.clip (clip passes NaN through) and
        # would otherwise print as "nan" in the critic message. Treat such
        # texels as absent/black (0), matching how composite_height drops
        # non-finite heights.
        out = material._weighted_blend(xs, ys, alpha, vs, samplers).astype(np.float64)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # --- albedo (sRGB, 0-1 and 0-255) ---------------------------------------- #
    rgb_lin = np.stack(
        [
            blend([getattr(layer.basecolor, ch) for layer in layers])
            for ch in ("r", "g", "b")
        ],
        axis=-1,
    )
    rgb01 = linear_to_srgb(rgb_lin)  # displayed albedo, 0-1
    rgb255 = rgb01 * 255.0
    lum = _luminance_srgb(rgb01)
    lum_mean = float(lum[mask].mean())
    lum_std = float(lum[mask].std())
    rms_contrast = float(lum_std / lum_mean) if lum_mean > 1e-6 else 0.0
    sat = _saturation(rgb01)
    dom_f, bw, aniso = _fft_features(lum, mask)
    spans = rgb255[mask].max(axis=0) - rgb255[mask].min(axis=0)

    # --- roughness / metallic ------------------------------------------------ #
    rough = blend([layer.roughness for layer in layers])
    metal = blend([layer.metallic for layer in layers])

    # --- relief (height -> normal/AO) --------------------------------------- #
    dx = abs(x2 - x1) / width
    dy = abs(y2 - y1) / height
    height_grids = [
        layer.height.eval_grid(xs, ys).astype(np.float64) for layer in layers
    ]
    h = composite_height(alpha_grids, height_grids)
    h_masked = h[mask]
    h_range = float(h_masked.max() - h_masked.min()) if h_masked.size else 0.0
    h_std = float(h_masked.std()) if h_masked.size else 0.0
    h_dom, _, h_aniso = _fft_features(h, mask)
    normal = height_to_normal(h.astype(np.float32), dx, dy, y_up=True)
    nz = normal[..., 2]
    tilt = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))
    ao = height_to_ao(h.astype(np.float32), dx)

    # --- the other PBR channels (means; defaults are 0, ior default 1.5) ---- #
    chan_means: SDict[float] = {}
    for name in _SCALAR_CHANNELS:
        if name in ("metallic", "roughness"):
            continue
        chan_means[name] = float(
            blend([getattr(layer, name) for layer in layers])[mask].mean()
        )
    ior = blend([layer.ior for layer in layers])
    # Emissive channels are 0-1 (normalized from 0-255) times strength, so the
    # blended luminance can exceed 1 for HDR glow; do not clip or rescale it.
    emissive_lin = np.stack(
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
    emissive_lum = np.clip(_luminance_srgb(np.clip(emissive_lin, 0.0, None)), 0.0, None)

    return {
        "albedo_luminance_mean": lum_mean * 255.0,
        "albedo_luminance_std": lum_std * 255.0,
        "albedo_rms_contrast": rms_contrast,
        "albedo_saturation": float(sat[mask].mean()),
        "albedo_edge_density": _edge_density(lum, mask),
        "albedo_dominant_freq": dom_f,
        "albedo_freq_bandwidth": bw,
        "albedo_anisotropy": aniso,
        "albedo_color_span": float(spans.max()),
        "albedo_tonal_entropy": _tonal_entropy(lum, mask),
        "roughness_mean": float(rough[mask].mean()),
        "roughness_std": float(rough[mask].std()),
        "metallic_mean": float(metal[mask].mean()),
        "metallic_coverage": float((metal[mask] > 0.5).mean()),
        "height_range": h_range,
        "height_std": h_std,
        "height_dominant_freq": h_dom,
        "height_anisotropy": h_aniso,
        "normal_mean_tilt_deg": float(tilt[mask].mean()),
        "ao_mean": float(ao[mask].mean()),
        "ao_std": float(ao[mask].std()),
        "sheen_mean": chan_means.get("sheen", 0.0),
        "coat_mean": chan_means.get("coat", 0.0),
        "transmission_mean": chan_means.get("transmission", 0.0),
        "subsurface_mean": chan_means.get("subsurface", 0.0),
        "anisotropy_mean": chan_means.get("anisotropy", 0.0),
        "ior_mean": float(ior[mask].mean()),
        "emissive_mean": float(emissive_lum[mask].mean()),
    }


def describe_program(
    program: str, *, width: int = 128, height: int = 128
) -> SDict[Descriptor]:
    """Deserialize ``program`` and run :func:`describe` on it, wrapping each
    scalar in a :class:`Descriptor` (machine ``name`` + ``value`` +
    ``definition``; ``label``/``formatted`` are derived). Order follows
    :data:`DESCRIPTOR_DEFINITIONS` so a critic reads the scalars in a stable
    sequence. Missing scalars are logged and skipped."""
    scalars = describe(LayeredMaterial.deserialize(program), width=width, height=height)
    diagnoses: SDict[Descriptor] = {}
    for name, definition in DESCRIPTOR_DEFINITIONS.items():
        if name not in scalars:
            logger.warning(f"Descriptor {name} missing")
            continue
        diagnoses[name] = Descriptor(
            name=name, value=scalars[name], definition=definition
        )
    return diagnoses
