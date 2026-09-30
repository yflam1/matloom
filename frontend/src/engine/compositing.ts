/**
 * compositing.ts — over-operator alpha resolution and weighted channel blend.
 * Mirrors matloom/engine/compositing.py.
 */
import { type Grid } from "./grid";

export interface AlphaWeights {
  /** Final coverage per cell: 1 − Π(1 − αᵢ). */
  alpha: Grid;
  /** Per-layer visibility weights Vᵢ = αᵢ · Π_{j>i}(1 − αⱼ). */
  Vs: Grid[];
}

/**
 * Resolve per-layer alpha grids (bottom-most first) into the final coverage
 * and the per-layer visibility weights used to blend every other channel.
 */
export function processAlphas(alphaGrids: Grid[]): AlphaWeights {
  const n = alphaGrids.length;
  const size = alphaGrids[0].length;
  const Vs = alphaGrids.map(() => new Float32Array(size));
  const run = new Float32Array(size).fill(1);
  for (let i = n - 1; i >= 0; i--) {
    const a = alphaGrids[i];
    for (let k = 0; k < size; k++) {
      Vs[i][k] = a[k] * run[k];
      run[k] *= 1 - a[k];
    }
  }
  const alpha = new Float32Array(size);
  for (let k = 0; k < size; k++) alpha[k] = 1 - run[k];
  return { alpha, Vs };
}

/**
 * Alpha-weighted average of per-layer channel grids: Σ Vᵢ·gᵢ / α, with cells
 * of zero coverage left at 0.
 */
export function weightedBlend(grids: Grid[], alpha: Grid, Vs: Grid[]): Grid {
  const size = alpha.length;
  const out = new Float32Array(size);
  for (let i = 0; i < grids.length; i++) {
    const g = grids[i];
    const v = Vs[i];
    for (let k = 0; k < size; k++) out[k] += v[k] * g[k];
  }
  for (let k = 0; k < size; k++) out[k] = alpha[k] > 0 ? out[k] / alpha[k] : 0;
  return out;
}
