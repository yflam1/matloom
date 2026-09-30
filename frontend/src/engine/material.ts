/**
 * material.ts — the grid axis helper and evaluate_material(), which composites
 * a layer stack into the basecolor / ORM / emissive texture maps the viewer
 * uploads. Mirrors LayeredMaterial.export() in matloom/engine.
 *
 * Evaluation is split into two phases so it can be parallelized by row band
 * (see eval/pool.ts) without changing results:
 *
 *   - {@link evalChannels} — the per-cell work (alpha compositing, basecolor,
 *     roughness/metallic, raw emissive, composited height). Every output cell is
 *     independent of every other, so a row stripe computes the same bytes the
 *     full grid would for those rows.
 *   - {@link finishMaps} — the whole-grid work that must see the assembled
 *     field: relief (AO box blur + normal central differences) and the global
 *     emissive normalization. Run once, on the main thread.
 *
 * {@link evaluateMaterial} composes the two over the full grid; its byte output
 * is identical to the previous single-pass implementation.
 */
import {
  gridMul,
  linearToSrgbScalar,
  toByte,
  type Axis,
  type Grid,
} from "./grid";
import { processAlphas, weightedBlend } from "./compositing";
import {
  AO_RADIUS,
  AO_STRENGTH,
  compositeHeight,
  heightToAo,
  heightToNormal,
} from "./relief";
import type { Layer } from "./layer";
import type { Expression2D } from "./expression";

export interface EvalRegion {
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  yUp?: boolean;
  /**
   * The span of the *authored* region, used only to keep relief (AO) at a fixed
   * world scale regardless of how the rendered region is cropped. When a crop
   * narrows `x1..x2`/`y1..y2` to a sub-rectangle (at the same resolution), these
   * stay at the full authored span, so the AO blur covers a constant world area
   * and the cavity darkening does not drift as you zoom. Default: the rendered
   * span (→ no scaling; identical to the pre-crop behavior).
   */
  reliefSpanX?: number;
  reliefSpanY?: number;
}

/** A region with every field resolved to a concrete value. */
export type ResolvedRegion = Required<EvalRegion>;

export interface MaterialMaps {
  /** RGBA8, sRGB-encoded base color with coverage in alpha. */
  basecolor: Uint8Array;
  /** RGBA8 packed occlusion(AO) / roughness / metallic / 255. */
  orm: Uint8Array;
  /** RGBA8 packed OpenPBR weights: R=sheen, G=coat, B=transmission, A=subsurface. */
  pbr1: Uint8Array;
  /** RGBA8 packed: R=anisotropy, G=ior (normalized over [1,3] → [0,1]), B=0, A=255. */
  pbr2: Uint8Array;
  /** RGB8 emissive, normalized by `emissiveIntensity`. */
  emissive: Uint8Array;
  /** Scale that maps the emissive bytes back to their scene-linear values (a relative multiplier, not a physical unit). */
  emissiveIntensity: number;
  /** RGBA8 tangent-space normal (OpenGL +Y); alpha is unused. */
  normal: Uint8Array;
  /** Per-texel surface displacement in coordinate units (the composited height), row-major w*h; for real vertex displacement of the viewer mesh. */
  displacement: Float32Array;
}

// IOR is packed into a byte over the [1, IOR_PACK_MAX] range; the viewer decodes
// it back with the same constant. Covers the common dielectric range (1.0–2.5)
// with headroom; values above clamp to the max when packed.
export const IOR_PACK_MAX = 3.0;

/**
 * The per-cell evaluation output, before whole-grid finishing. Carries the
 * final basecolor, the ORM with its red (AO) channel still zero, the raw
 * (un-normalized) emissive and the composited height field. A band of rows
 * produces a stripe of this; {@link assembleChannels} concatenates stripes.
 */
export interface ChannelData {
  /** RGBA8 basecolor — final. */
  basecolor: Uint8Array;
  /** RGBA8 ORM with G=roughness, B=metallic, A=255; R (AO) filled by finishMaps. */
  orm: Uint8Array;
  /** RGBA8 OpenPBR weights: R=sheen, G=coat, B=transmission, A=subsurface — final. */
  pbr1: Uint8Array;
  /** RGBA8 OpenPBR: R=anisotropy, G=ior-normalized, B=0, A=255 — final. */
  pbr2: Uint8Array;
  /** Raw emissive RGB floats (pre-normalization), length cells*3. */
  emf: Float32Array;
  /** Composited height field (max operator), length cells. */
  height: Grid;
}

// Cell-center coordinates across [v1, v2]; reversed (top = max) when `reverse`.
// Float32 to match the original engine exactly — the precision is observable
// near noise cell boundaries (see grid.ts Axis).
export function axisCoords(
  length: number,
  v1: number,
  v2: number,
  reverse: boolean,
): Axis {
  if (reverse) {
    const t = v1;
    v1 = v2;
    v2 = t;
  }
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++)
    out[i] = v1 + ((i + 0.5) * (v2 - v1)) / length;
  return out;
}

/** Fill an EvalRegion's defaults (the viewer's standard unit window). */
export function resolveRegion({
  x1 = 0,
  y1 = 0,
  x2 = 1,
  y2 = 1,
  yUp = true,
  reliefSpanX,
  reliefSpanY,
}: EvalRegion = {}): ResolvedRegion {
  // The relief reference span defaults to the rendered span, so an uncropped
  // render (and every caller that does not set it) scales relief by 1× — i.e.
  // byte-identical to the pre-crop behavior.
  return {
    x1,
    y1,
    x2,
    y2,
    yUp,
    reliefSpanX: reliefSpanX ?? Math.abs(x2 - x1),
    reliefSpanY: reliefSpanY ?? Math.abs(y2 - y1),
  };
}

const RGB_CHANNELS = [
  [0, "r"],
  [1, "g"],
  [2, "b"],
] as const;

// Whether every cell of a grid holds the same value. Used to short-circuit the
// relief derivation for a height field with no spatial variation (the common
// "no height" case). Early-exits on the first differing cell, so it is O(1) for
// the overwhelmingly more frequent non-constant fields and O(n) only when the
// field really is uniform (still far cheaper than the box blur it replaces).
function isConstantGrid(g: Grid): boolean {
  const v0 = g[0];
  for (let i = 1; i < g.length; i++) if (g[i] !== v0) return false;
  return true;
}

/**
 * Per-cell evaluation over the sample axes `xs`/`ys` (which may be a row stripe
 * of the full grid). Independent per output cell, so the result for a stripe is
 * byte-identical to the same rows of the full-grid evaluation.
 */
export function evalChannels(
  layers: Layer[],
  xs: Axis,
  ys: Axis,
  w: number,
  h: number,
): ChannelData {
  const n = w * h;
  const alphaGrids = layers.map((l) => l.alpha.evalGrid(xs, ys, w, h));
  const { alpha, Vs } = processAlphas(alphaGrids);

  const basecolor = new Uint8Array(n * 4);
  for (const [ci, attr] of RGB_CHANNELS) {
    // Fuse the linear→sRGB conversion into the quantization pass to skip an
    // intermediate Float32Array per channel. `Math.fround` reproduces the
    // store-into-Float32Array rounding the previous `linearToSrgb(...)` step
    // applied, so the bytes are identical.
    const blended = weightedBlend(
      layers.map((l) => l.basecolor[attr].evalGrid(xs, ys, w, h)),
      alpha,
      Vs,
    );
    for (let i = 0; i < n; i++)
      basecolor[i * 4 + ci] = toByte(
        Math.fround(linearToSrgbScalar(blended[i])),
      );
  }
  for (let i = 0; i < n; i++) basecolor[i * 4 + 3] = toByte(alpha[i]);

  // ORM: red (AO) is filled by finishMaps from the assembled height field; G/B
  // carry roughness/metallic, which are per-cell.
  const orm = new Uint8Array(n * 4);
  const rough = weightedBlend(
    layers.map((l) => l.roughness.evalGrid(xs, ys, w, h)),
    alpha,
    Vs,
  );
  const metal = weightedBlend(
    layers.map((l) => l.metallic.evalGrid(xs, ys, w, h)),
    alpha,
    Vs,
  );
  for (let i = 0; i < n; i++) {
    orm[i * 4 + 1] = toByte(rough[i]);
    orm[i * 4 + 2] = toByte(metal[i]);
    orm[i * 4 + 3] = 255;
  }

  // OpenPBR reflectance weights, alpha-weighted-blended like roughness/metallic
  // and packed into two RGBA8 maps. ior is normalized over [1, IOR_PACK_MAX].
  const blendCh = (pick: (l: Layer) => Expression2D) =>
    weightedBlend(
      layers.map((l) => pick(l).evalGrid(xs, ys, w, h)),
      alpha,
      Vs,
    );
  const sheen = blendCh((l) => l.sheen);
  const coat = blendCh((l) => l.coat);
  const transmission = blendCh((l) => l.transmission);
  const subsurface = blendCh((l) => l.subsurface);
  const anisotropy = blendCh((l) => l.anisotropy);
  const ior = blendCh((l) => l.ior);
  const pbr1 = new Uint8Array(n * 4);
  const pbr2 = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    pbr1[i * 4] = toByte(sheen[i]);
    pbr1[i * 4 + 1] = toByte(coat[i]);
    pbr1[i * 4 + 2] = toByte(transmission[i]);
    pbr1[i * 4 + 3] = toByte(subsurface[i]);
    pbr2[i * 4] = toByte(anisotropy[i]);
    pbr2[i * 4 + 1] = toByte((ior[i] - 1) / (IOR_PACK_MAX - 1));
    pbr2[i * 4 + 2] = 0;
    pbr2[i * 4 + 3] = 255;
  }

  // Raw emissive (strength shared across R/G/B, so evaluate it once per layer).
  const strengths = layers.map((l) =>
    l.emissive.strength.evalGrid(xs, ys, w, h),
  );
  const emf = new Float32Array(n * 3);
  for (const [ci, attr] of RGB_CHANNELS) {
    const bl = weightedBlend(
      layers.map((l, li) =>
        gridMul(l.emissive[attr].evalGrid(xs, ys, w, h), strengths[li]),
      ),
      alpha,
      Vs,
    );
    for (let i = 0; i < n; i++) emf[i * 3 + ci] = bl[i];
  }

  // Per-layer heights composited with the max operator.
  const height = compositeHeight(
    alphaGrids,
    layers.map((l) => l.height.evalGrid(xs, ys, w, h)),
  );

  return { basecolor, orm, pbr1, pbr2, emf, height };
}

/**
 * Concatenate per-band channel stripes (in any order) into the full-grid
 * {@link ChannelData}. Each part covers `data.height` rows starting at `y0`.
 */
export function assembleChannels(
  parts: { y0: number; data: ChannelData }[],
  w: number,
  h: number,
): ChannelData {
  const n = w * h;
  const out: ChannelData = {
    basecolor: new Uint8Array(n * 4),
    orm: new Uint8Array(n * 4),
    pbr1: new Uint8Array(n * 4),
    pbr2: new Uint8Array(n * 4),
    emf: new Float32Array(n * 3),
    height: new Float32Array(n),
  };
  for (const { y0, data } of parts) {
    out.basecolor.set(data.basecolor, y0 * w * 4);
    out.orm.set(data.orm, y0 * w * 4);
    out.pbr1.set(data.pbr1, y0 * w * 4);
    out.pbr2.set(data.pbr2, y0 * w * 4);
    out.emf.set(data.emf, y0 * w * 3);
    out.height.set(data.height, y0 * w);
  }
  return out;
}

/**
 * Whole-grid finishing: derive relief (AO + normal + displacement) from the
 * assembled height field and normalize emissive against its global maximum.
 * Mutates `ch.orm` to fill the AO (red) channel; consumes `ch.basecolor`/`orm`.
 */
export function finishMaps(
  ch: ChannelData,
  w: number,
  h: number,
  region: ResolvedRegion,
): MaterialMaps {
  const { x1, y1, x2, y2, yUp, reliefSpanX, reliefSpanY } = region;
  const n = w * h;
  const { basecolor, orm, pbr1, pbr2, emf, height: heightField } = ch;

  // Relief. Fast path: a spatially-uniform height field has no slope and no
  // cavities, so its AO is 1 everywhere, its normal is flat (0,0,1), and the
  // box-blur / central-difference passes are provably redundant. `toByte`
  // collapses the ~1e-9 f32 drift a full box blur would introduce back to the
  // same bytes (255 / 128,128,255), so the fast path is byte-identical — it
  // just skips the O(n·radius) work. Most materials have no height variation,
  // making this the common case.
  const dx = Math.abs(x2 - x1) / w;
  const dy = Math.abs(y2 - y1) / h;
  // Crop zoom: how much larger the authored region is than the rendered one. 1
  // when uncropped. The AO blur radius scales with it so the blur always covers
  // the same *world* area (≈ AO_RADIUS · authoredSpan/resolution), keeping the
  // cavity darkening invariant as the crop narrows. The radius is a continuous
  // real (not rounded to integer texels) so the cavity darkening changes
  // smoothly with the crop, never in the 1-texel steps a `floor` would produce.
  // The normal map is intentionally left as the true per-texel gradient, so it
  // genuinely sharpens as a crop reveals finer detail.
  const renderSpanX = Math.abs(x2 - x1);
  const renderSpanY = Math.abs(y2 - y1);
  const zx = renderSpanX > 0 ? reliefSpanX / renderSpanX : 1;
  const zy = renderSpanY > 0 ? reliefSpanY / renderSpanY : 1;
  const aoRadiusX = Math.max(1, AO_RADIUS * zx);
  const aoRadiusY = Math.max(1, AO_RADIUS * zy);
  const flatRelief = isConstantGrid(heightField);
  const ao = flatRelief
    ? null
    : heightToAo(heightField, w, h, dx, aoRadiusX, AO_STRENGTH, aoRadiusY);
  const nrm = flatRelief
    ? null
    : heightToNormal(heightField, w, h, dx, dy, yUp);

  // Surface displacement equals the composited height directly (coordinate units).
  const displacement = new Float32Array(n);
  for (let i = 0; i < n; i++) displacement[i] = heightField[i];

  // Fill the AO (red) slot of the ORM map.
  for (let i = 0; i < n; i++) orm[i * 4] = ao === null ? 255 : toByte(ao[i]);

  // Normal RGBA: rgb = OpenGL-encoded normal (alpha unused, kept opaque). A flat
  // relief encodes (0,0,1) → bytes (128,128,255).
  const normal = new Uint8Array(n * 4);
  if (nrm === null) {
    for (let i = 0; i < n; i++) {
      normal[i * 4] = 128;
      normal[i * 4 + 1] = 128;
      normal[i * 4 + 2] = 255;
      normal[i * 4 + 3] = 255;
    }
  } else {
    const { nx, ny, nz } = nrm;
    for (let i = 0; i < n; i++) {
      normal[i * 4] = toByte(nx[i] * 0.5 + 0.5);
      normal[i * 4 + 1] = toByte(ny[i] * 0.5 + 0.5);
      normal[i * 4 + 2] = toByte(nz[i] * 0.5 + 0.5);
      normal[i * 4 + 3] = 255;
    }
  }

  // Emissive: normalize against the global maximum (so the brightest texel maps
  // to 255) and record that scale as emissiveIntensity. `max` is associative, so
  // a global max assembled from per-stripe maxima is identical.
  let maxE = 1;
  for (let i = 0; i < emf.length; i++) if (emf[i] > maxE) maxE = emf[i];
  const emissive = new Uint8Array(emf.length);
  for (let i = 0; i < emf.length; i++) emissive[i] = toByte(emf[i] / maxE);

  return {
    basecolor,
    orm,
    pbr1,
    pbr2,
    emissive,
    emissiveIntensity: maxE,
    normal,
    displacement,
  };
}

export function evaluateMaterial(
  layers: Layer[],
  width: number,
  height: number,
  region: EvalRegion = {},
): MaterialMaps {
  // Mirrors LayeredMaterial.export(): xs spans [x1, x2] left→right, ys spans
  // [y1, y2] and is reversed when yUp so row 0 is the top (max Y).
  const r = resolveRegion(region);
  const xs = axisCoords(width, r.x1, r.x2, false);
  const ys = axisCoords(height, r.y1, r.y2, r.yUp);
  const ch = evalChannels(layers, xs, ys, width, height);
  return finishMaps(ch, width, height, r);
}
