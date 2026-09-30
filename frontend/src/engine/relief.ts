/**
 * relief.ts — derive surface relief maps from a composited per-layer height
 * field. A byte-for-byte port of matloom/engine/relief.py (verified by the
 * cross-engine parity tests).
 *
 * A layer's `height` channel is an absolute displacement in **coordinate units**
 * (the same units as the X/Y plane). The layers combine with a *max* operator
 * (not the alpha-weighted blend the other channels use), and alpha is a pure
 * *coverage mask* — it gates where a layer contributes height but does not scale
 * the magnitude:
 *
 *     H(x, y) = maxᵢ { hᵢ(x, y) : αᵢ(x, y) > 0 }
 *
 * Heights may be any real value, including negative (carving below the
 * substrate), so there is no substrate floor; where no layer is present the
 * surface is 0.
 */
import { type Grid } from "./grid";

// AO tuning, shared with the Python engine (see relief.py). `radius` is the
// box-blur half width in texels; `strength` scales the darkening.
export const AO_RADIUS = 8;
export const AO_STRENGTH = 1.0;

/** Per-cell normal components (each `[-1, 1]`); a flat surface is (0, 0, 1). */
export interface Normal {
  nx: Grid;
  ny: Grid;
  nz: Grid;
}

/**
 * Combine per-layer height fields: `maxᵢ { hᵢ : αᵢ > 0 }`. Alpha is a coverage
 * mask, not a multiplier — a layer contributes its full height where it is
 * present (`α > 0`) and nothing where absent. Heights may be negative, so there
 * is no substrate floor; where no layer contributes the surface is 0.
 */
export function compositeHeight(alphaGrids: Grid[], heightGrids: Grid[]): Grid {
  const size = alphaGrids[0].length;
  const out = new Float32Array(size);
  // Start as "no contribution" (−∞); replaced wherever a layer is present.
  out.fill(-Infinity);
  for (let li = 0; li < alphaGrids.length; li++) {
    const a = alphaGrids[li];
    const hgt = heightGrids[li];
    for (let k = 0; k < size; k++) {
      const v = hgt[k];
      // Present where `α > 0`; opacity no longer scales height. A height
      // expression can be non-finite (e.g. `Log(0)` → −∞, `Sqrt(-1)` → NaN);
      // treat that texel as absent so a poisoned layer never wins. (`NaN > x` is
      // already false; the `isFinite` guard also rejects ±∞.)
      if (a[k] > 0 && Number.isFinite(v) && v > out[k]) out[k] = v;
    }
  }
  for (let k = 0; k < size; k++) {
    const v = out[k];
    // −Infinity (no layer present) → substrate 0; also normalize −0 → +0 so a
    // zero-contribution texel compares equal to the substrate.
    out[k] = v === -Infinity || v === 0 ? 0 : v;
  }
  return out;
}

/**
 * Derive an OpenGL (+Y-up) tangent-space normal from a height field. Central
 * differences with clamped edges; row 0 is max Y when `yUp`, so the row-direction
 * derivative is negated to recover ∂H/∂Y in world space.
 */
export function heightToNormal(
  height: Grid,
  w: number,
  h: number,
  dx: number,
  dy: number,
  yUp: boolean,
): Normal {
  const dHdx = centralDiffX(height, w, h, dx);
  const dHdyRows = centralDiffY(height, w, h, dy);
  const size = w * h;
  const nx = new Float32Array(size);
  const ny = new Float32Array(size);
  const nz = new Float32Array(size);
  for (let k = 0; k < size; k++) {
    const dHdy = yUp ? -dHdyRows[k] : dHdyRows[k];
    const x = -dHdx[k];
    const y = -dHdy;
    const invLen = 1.0 / Math.sqrt(x * x + y * y + 1.0);
    nx[k] = x * invLen;
    ny[k] = y * invLen;
    nz[k] = invLen; // z component is 1 · invLen
  }
  return { nx, ny, nz };
}

/**
 * Approximate ambient occlusion (cavity darkening). A texel below the local mean
 * of its neighborhood is darkened; slopes and peaks stay bright. The depth below
 * the mean is normalized by the world radius of the blur (`radiusX·dx`).
 *
 * The blur radius is given separately per axis (`radiusX` columns, `radiusY`
 * rows) so a caller can hold the *world* size of the neighborhood fixed while the
 * texel size changes — e.g. under a crop, where the rendered region shrinks but
 * the resolution does not. Keeping the world radius constant makes AO
 * crop-invariant: the cavity darkening depends only on the surface geometry, not
 * on how finely it happens to be sampled. `radiusY` defaults to `radiusX`, so an
 * isotropic caller is unchanged.
 */
export function heightToAo(
  height: Grid,
  w: number,
  h: number,
  dx: number,
  radiusX: number = AO_RADIUS,
  strength: number = AO_STRENGTH,
  radiusY: number = radiusX,
): Grid {
  const mean = boxBlur(height, w, h, radiusX, radiusY);
  const norm = Math.max(radiusX, 1) * dx;
  const out = new Float32Array(w * h);
  for (let k = 0; k < out.length; k++) {
    let cavity = (mean[k] - height[k]) / norm;
    cavity = cavity < 0 ? 0 : cavity > 1 ? 1 : cavity;
    const ao = 1.0 - strength * cavity;
    out[k] = ao < 0 ? 0 : ao > 1 ? 1 : ao;
  }
  return out;
}

// Clamped central difference along columns (X). Interior: (a[j+1]−a[j−1])/(2·s);
// edges: one-sided /s. A width-1 grid yields zeros. Mirrors relief.py axis=1.
function centralDiffX(a: Grid, w: number, h: number, spacing: number): Grid {
  const out = new Float32Array(w * h);
  if (w < 2) return out;
  for (let i = 0; i < h; i++) {
    const row = i * w;
    for (let j = 0; j < w; j++) {
      const idx = row + j;
      if (j === 0) out[idx] = (a[idx + 1] - a[idx]) / spacing;
      else if (j === w - 1) out[idx] = (a[idx] - a[idx - 1]) / spacing;
      else out[idx] = (a[idx + 1] - a[idx - 1]) / (2.0 * spacing);
    }
  }
  return out;
}

// Clamped central difference along rows. Mirrors relief.py axis=0.
function centralDiffY(a: Grid, w: number, h: number, spacing: number): Grid {
  const out = new Float32Array(w * h);
  if (h < 2) return out;
  for (let i = 0; i < h; i++) {
    const row = i * w;
    for (let j = 0; j < w; j++) {
      const idx = row + j;
      if (i === 0) out[idx] = (a[idx + w] - a[idx]) / spacing;
      else if (i === h - 1) out[idx] = (a[idx] - a[idx - w]) / spacing;
      else out[idx] = (a[idx + w] - a[idx - w]) / (2.0 * spacing);
    }
  }
  return out;
}

// Separable, edge-clamped box blur with a `2·radiusX+1` (columns) by
// `2·radiusY+1` (rows) window. Taps summed in ascending order (k = −radius …
// +radius) to match relief.py's summation. `radiusY` defaults to `radiusX`.
//
// The radius may be fractional: a non-integer radius is the linear blend of the
// two enclosing integer-radius blurs (separable: lerp each axis independently).
// At an integer radius the blend collapses to the integer blur exactly
// (`1.0·x + 0.0·y = x` in IEEE 754), so integer-radius behavior is bit-identical to
// the direct path — the fractional support is a pure superset, never a
// behavior change for existing callers.
function boxBlur(
  a: Grid,
  w: number,
  h: number,
  radiusX: number,
  radiusY: number = radiusX,
): Grid {
  if (radiusX <= 0 && radiusY <= 0) return a.slice();
  const rXlo = Math.floor(radiusX);
  const rYlo = Math.floor(radiusY);
  const fracX = radiusX - rXlo;
  const fracY = radiusY - rYlo;
  // X axis: lerp the floor/ceil integer blurs.
  const xLo = boxBlurAxisX(a, w, h, rXlo);
  const xHi = fracX > 0 ? boxBlurAxisX(a, w, h, rXlo + 1) : xLo;
  const blurX = lerpGrid(xLo, xHi, fracX);
  // Y axis on the X-blurred field.
  const yLo = boxBlurAxisY(blurX, w, h, rYlo);
  const yHi = fracY > 0 ? boxBlurAxisY(blurX, w, h, rYlo + 1) : yLo;
  return lerpGrid(yLo, yHi, fracY);
}

// Linear blend of two grids by `t` ∈ [0, 1]. At t = 0 returns `a` (the floor
// blur) untouched so an integer radius is bit-exact; at t = 1 returns `b`.
function lerpGrid(a: Grid, b: Grid, t: number): Grid {
  if (t === 0) return a;
  if (t === 1) return b;
  const out = new Float32Array(a.length);
  const u = 1 - t;
  for (let k = 0; k < out.length; k++) out[k] = u * a[k] + t * b[k];
  return out;
}

function boxBlurAxisX(a: Grid, w: number, h: number, radius: number): Grid {
  const out = new Float32Array(w * h);
  const denom = 2 * radius + 1;
  for (let i = 0; i < h; i++) {
    const row = i * w;
    for (let j = 0; j < w; j++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++) {
        let jj = j + k;
        if (jj < 0) jj = 0;
        else if (jj > w - 1) jj = w - 1;
        sum += a[row + jj];
      }
      out[row + j] = sum / denom;
    }
  }
  return out;
}

function boxBlurAxisY(a: Grid, w: number, h: number, radius: number): Grid {
  const out = new Float32Array(w * h);
  const denom = 2 * radius + 1;
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++) {
        let ii = i + k;
        if (ii < 0) ii = 0;
        else if (ii > h - 1) ii = h - 1;
        sum += a[ii * w + j];
      }
      out[i * w + j] = sum / denom;
    }
  }
  return out;
}
