/**
 * grid.ts — Float32Array grid primitives (w*h flat, row-major) and the scalar
 * color-space conversions used throughout the engine. Mirrors the array helpers
 * of matloom/engine/utils.py.
 */

/** A material channel evaluated over the sample grid, row-major (w*h). */
export type Grid = Float32Array;

/**
 * Sample-axis coordinates (per-column xs / per-row ys). Float32, matching the
 * original engine: the reduced precision is observable — it shifts values near
 * noise cell boundaries — so it must be preserved exactly.
 */
export type Axis = Float32Array;

/**
 * Build a grid of length `n`, filling each cell from its index.
 *
 * The hot per-pixel kernels below (binary ops, the unary nodes in
 * expression.ts) deliberately do NOT route through this helper: a closure
 * dispatched per cell is megamorphic across call sites and never inlined by V8,
 * so each kernel inlines its own `for` loop instead. `makeGrid` remains for the
 * cold builders and as a tested primitive mirroring numpy's `fromfunction`.
 */
export function makeGrid(n: number, fn: (i: number) => number): Grid {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = fn(i);
  return a;
}

/** Constant grid: every cell set to `v`. */
export function gridFull(n: number, v: number): Grid {
  return new Float32Array(n).fill(v);
}

/** Broadcast a per-column x-axis vector across all `h` rows. */
export function gridFromX(xs: Axis, w: number, h: number): Grid {
  const out = new Float32Array(w * h);
  for (let r = 0; r < h; r++) {
    const base = r * w;
    for (let c = 0; c < w; c++) out[base + c] = xs[c];
  }
  return out;
}

/** Broadcast a per-row y-axis vector across all `w` columns. */
export function gridFromY(ys: Axis, w: number, h: number): Grid {
  const out = new Float32Array(w * h);
  for (let r = 0; r < h; r++) {
    const v = ys[r];
    const base = r * w;
    for (let c = 0; c < w; c++) out[base + c] = v;
  }
  return out;
}

/** Map a unary function over a grid. */
export function gridMap(a: Grid, fn: (v: number) => number): Grid {
  const n = a.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = fn(a[i]);
  return out;
}

// Binary grid ops accept either another grid or a scalar (broadcast). Each
// inlines two `for` loops (scalar / grid operand) so the arithmetic is a
// monomorphic, JIT-inlinable body rather than a per-cell closure call. Results
// are bit-identical to the previous `makeGrid` form (same IEEE f32 stores).
type GridOperand = Grid | number;

export function gridAdd(a: Grid, b: GridOperand): Grid {
  const n = a.length;
  const out = new Float32Array(n);
  if (typeof b === "number") for (let i = 0; i < n; i++) out[i] = a[i] + b;
  else for (let i = 0; i < n; i++) out[i] = a[i] + b[i];
  return out;
}
export function gridSub(a: Grid, b: GridOperand): Grid {
  const n = a.length;
  const out = new Float32Array(n);
  if (typeof b === "number") for (let i = 0; i < n; i++) out[i] = a[i] - b;
  else for (let i = 0; i < n; i++) out[i] = a[i] - b[i];
  return out;
}
export function gridMul(a: Grid, b: GridOperand): Grid {
  const n = a.length;
  const out = new Float32Array(n);
  if (typeof b === "number") for (let i = 0; i < n; i++) out[i] = a[i] * b;
  else for (let i = 0; i < n; i++) out[i] = a[i] * b[i];
  return out;
}
export function gridDiv(a: Grid, b: GridOperand): Grid {
  const n = a.length;
  const out = new Float32Array(n);
  if (typeof b === "number") for (let i = 0; i < n; i++) out[i] = a[i] / b;
  else for (let i = 0; i < n; i++) out[i] = a[i] / b[i];
  return out;
}
export function gridPow(a: Grid, b: GridOperand): Grid {
  const n = a.length;
  const out = new Float32Array(n);
  if (typeof b === "number") for (let i = 0; i < n; i++) out[i] = a[i] ** b;
  else for (let i = 0; i < n; i++) out[i] = a[i] ** b[i];
  return out;
}
export function gridMin(a: Grid, b: GridOperand): Grid {
  const n = a.length;
  const out = new Float32Array(n);
  if (typeof b === "number")
    for (let i = 0; i < n; i++) out[i] = Math.min(a[i], b);
  else for (let i = 0; i < n; i++) out[i] = Math.min(a[i], b[i]);
  return out;
}
export function gridMax(a: Grid, b: GridOperand): Grid {
  const n = a.length;
  const out = new Float32Array(n);
  if (typeof b === "number")
    for (let i = 0; i < n; i++) out[i] = Math.max(a[i], b);
  else for (let i = 0; i < n; i++) out[i] = Math.max(a[i], b[i]);
  return out;
}

export function linearToSrgb(a: Grid): Grid {
  const n = a.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = linearToSrgbScalar(a[i]);
  return out;
}
export function srgbToLinear(a: Grid): Grid {
  const n = a.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = srgbToLinearScalar(a[i]);
  return out;
}

// Scalar color-space conversions (same formulas as the grid versions above),
// for the single-point Expression2D.call() path. Mirror utils.py's overloads.
export function linearToSrgbScalar(v: number): number {
  return v <= 0.0031308 ? v * 12.92 : v ** (1 / 2.4) * 1.055 - 0.055;
}
export function srgbToLinearScalar(v: number): number {
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

/** Quantize a [0,1] float to a clamped, rounded uint8 (0–255). */
export function toByte(v: number): number {
  return Math.min(255, Math.max(0, (v * 255 + 0.5) | 0));
}
