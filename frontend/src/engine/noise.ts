/**
 * noise.ts — fBm (layered simplex) and Worley/cellular noise expressions.
 * Mirrors matloom/engine/noise.py; the sampling math is unchanged.
 */
import {
  Expression2D,
  coerceFloat,
  coerceInt,
  type ExprLike,
  type CoordCtx,
} from "./expression";
import type { Grid, Axis } from "./grid";
import { openSimplex, type SimplexNoise } from "./opensimplex";

/** Pick a random 31-bit seed when the caller passes none. */
function randomSeed(): number {
  return (Math.random() * 2 ** 31) | 0;
}

// A linspace axis is fully determined by its length and endpoints, so the dims
// plus the four axis extremes uniquely identify the sample grid. Used to skip
// recomputing a noise field when it is re-evaluated over the same grid (e.g.
// after editing an unrelated channel — see app/eval.ts's expression reuse).
function gridKey(xs: Axis, ys: Axis, w: number, h: number): string {
  return `${w}x${h}|${xs[0]},${xs[w - 1]}|${ys[0]},${ys[h - 1]}`;
}

// Single-entry memo for a noise node: the returned grid is only ever read by
// downstream ops (every grid.ts helper allocates a fresh array), so handing
// back the same instance on a cache hit is safe.
interface GridMemo {
  key: string;
  grid: Grid;
}

/**
 * Base for seedable noise (fBm, Worley, …). Owns the seed, whether it was
 * auto-generated (vs. pinned by the user), and the result memo.
 *
 * `autoSeeded` distinguishes `fBm()` — whose seed the system rolled and may
 * roll again on request — from `fBm(seed=5)`, which the user fixed and must
 * never change. `reseed()` only acts on the former, which is what lets the UI
 * offer a "regenerate noise" control without ever disturbing a pinned seed.
 */
export abstract class NoiseExpression2D extends Expression2D {
  /** Human-readable node name (e.g. "fBm"), for labelling in the UI. */
  readonly noiseName: string;
  /** The active seed. */
  protected seed: number;
  /** True when the seed was system-generated (no explicit `seed=` given). */
  readonly autoSeeded: boolean;
  protected memo: GridMemo | null = null;

  // Source span of the originating call within its field's text, tagged by the
  // parser so the UI can echo each noise's exact call and locate it. Empty/-1
  // when the node is built programmatically rather than parsed.
  sourceText = "";
  sourceStart = -1;
  sourceEnd = -1;
  /**
   * The arg names the author explicitly wrote (captured from the opts object's
   * own keys, so it works for parsed and programmatically-built nodes alike).
   * The serializer emits only these, preserving explicit defaults like
   * `octaves=6` or `to_01=False`. `seed` is always emitted regardless. Mirrors
   * the Python engine's `_set_args`.
   */
  readonly setArgs: Set<string>;

  constructor(
    noiseName: string,
    seed: number | null | undefined,
    setArgs: Set<string> = new Set(),
  ) {
    super();
    this.noiseName = noiseName;
    this.autoSeeded = seed == null;
    this.seed = seed ?? randomSeed();
    this.setArgs = setArgs;
  }

  /** The active seed — lets the UI reveal an otherwise-hidden auto seed. */
  get currentSeed(): number {
    return this.seed;
  }

  /**
   * Roll a fresh seed, but only for auto-seeded nodes; pinned seeds are left
   * untouched. Returns whether the seed actually changed (so callers can tell
   * if a re-render is warranted). Invalidates the memo and lets subclasses
   * rebuild any seed-derived state.
   */
  reseed(): boolean {
    if (!this.autoSeeded) return false;
    this.seed = randomSeed();
    this.memo = null;
    this.onSeedChanged();
    return true;
  }

  /**
   * Adopt a previously remembered auto seed. Only auto-seeded nodes are
   * affected — a pinned `seed=` is authoritative and left untouched. This lets
   * the app re-attach the seed a node was given on its first appearance after
   * the expression is re-parsed (so editing around a noise, or pinning then
   * un-pinning it, does not re-roll it). Like reseed() it drops the memo and
   * rebuilds any seed-derived state.
   */
  restoreSeed(seed: number): void {
    if (!this.autoSeeded || this.seed === seed) return;
    this.seed = seed;
    this.memo = null;
    this.onSeedChanged();
  }

  /**
   * A stable key for this node's arguments *excluding* the seed. Two calls with
   * the same signature are "the same noise" for the purpose of remembering an
   * auto-generated seed: editing elsewhere in the expression, or adding then
   * removing a `seed=`, leaves the signature unchanged so the seed is reused.
   */
  abstract argSignature(): string;

  /** Hook for subclasses to rebuild state derived from the seed (e.g. tables). */
  protected onSeedChanged(): void {}
}

/** Collect every seedable noise node in an expression tree (depth-first). */
export function collectNoise(expr: Expression2D): NoiseExpression2D[] {
  const out: NoiseExpression2D[] = [];
  const visit = (e: Expression2D): void => {
    if (e instanceof NoiseExpression2D) out.push(e);
    for (const c of e.children()) visit(c);
  };
  visit(expr);
  return out;
}

// ---------------------------------------------------------------------------
// fBm  (fractal Brownian motion)
// ---------------------------------------------------------------------------
export interface FbmOptions {
  octaves?: ExprLike;
  lacunarity?: ExprLike;
  gain?: ExprLike;
  base_freq?: ExprLike;
  base_freq_x?: ExprLike | null;
  base_freq_y?: ExprLike | null;
  to_01?: boolean;
  seed?: ExprLike | null;
}

export class fBm extends NoiseExpression2D {
  private readonly oct: number;
  private readonly lac: number;
  private readonly gain: number;
  private readonly bfx: number;
  private readonly bfy: number;
  private readonly to01: boolean;
  private simplex: SimplexNoise;
  private readonly F = 2 / Math.sqrt(3);

  constructor(opts: FbmOptions = {}) {
    const setArgs = new Set(Object.keys(opts));
    const {
      octaves = 6,
      lacunarity = 2.0,
      gain = 0.5,
      base_freq = 1.0,
      base_freq_x = null,
      base_freq_y = null,
      to_01 = false,
      seed = null,
    } = opts;
    // seed is int-typed; coerce an expression (and round) before super(), which
    // must run before `this` is touched.
    super("fBm", seed == null ? null : coerceInt(seed), setArgs);
    this.oct = coerceInt(octaves);
    this.lac = coerceFloat(lacunarity);
    this.gain = coerceFloat(gain);
    this.bfx =
      base_freq_x == null ? coerceFloat(base_freq) : coerceFloat(base_freq_x);
    this.bfy =
      base_freq_y == null ? coerceFloat(base_freq) : coerceFloat(base_freq_y);
    this.to01 = to_01;
    this.simplex = openSimplex(this.seed);
  }

  // The simplex permutation table is derived from the seed, so rebuild it.
  protected onSeedChanged(): void {
    this.simplex = openSimplex(this.seed);
  }

  argSignature(): string {
    return `fBm|${this.oct}|${this.lac}|${this.gain}|${this.bfx}|${this.bfy}|${this.to01}`;
  }

  /** Resolved arguments (excluding seed), for serialization. */
  get exportParams(): {
    octaves: number;
    lacunarity: number;
    gain: number;
    base_freq_x: number;
    base_freq_y: number;
    to_01: boolean;
  } {
    return {
      octaves: this.oct,
      lacunarity: this.lac,
      gain: this.gain,
      base_freq_x: this.bfx,
      base_freq_y: this.bfy,
      to_01: this.to01,
    };
  }

  call(x = 0.0, y = 0.0): number {
    let fx = this.bfx;
    let fy = this.bfy;
    let total = 0.0;
    let amp = 1.0;
    let norm = 0.0;
    for (let o = 0; o < this.oct; o++) {
      total += this.simplex.noise2(x * fx, y * fy) * this.F * amp;
      norm += amp;
      fx *= this.lac;
      fy *= this.lac;
      amp *= this.gain;
    }
    let out = total / norm;
    if (this.to01) out = (out + 1.0) / 2.0;
    return Math.max(this.to01 ? 0.0 : -1.0, Math.min(out, 1.0));
  }

  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    if (ctx) {
      // Non-separable coords (e.g. under Rotate): sample per cell so the grid
      // path matches call(). Slower, and the memo is bypassed.
      const out = new Float32Array(w * h);
      for (let i = 0; i < out.length; i++)
        out[i] = this.call(ctx.xg[i], ctx.yg[i]);
      return out;
    }
    const key = gridKey(xs, ys, w, h);
    if (this.memo && this.memo.key === key) return this.memo.grid;
    const n = w * h;
    const out = new Float32Array(n);
    let fx = this.bfx;
    let fy = this.bfy;
    let amp = 1.0;
    let norm = 0.0;
    const sx = new Float64Array(xs.length);
    const sy = new Float64Array(ys.length);
    for (let o = 0; o < this.oct; o++) {
      for (let i = 0; i < xs.length; i++) sx[i] = xs[i] * fx;
      for (let i = 0; i < ys.length; i++) sy[i] = ys[i] * fy;
      const raw = this.simplex.noise2array(sx, sy);
      const sc = amp * this.F;
      for (let i = 0; i < n; i++) out[i] += raw[i] * sc;
      norm += amp;
      fx *= this.lac;
      fy *= this.lac;
      amp *= this.gain;
    }
    const inv = 1 / norm;
    const lo = this.to01 ? 0 : -1;
    for (let i = 0; i < n; i++) {
      let v = out[i] * inv;
      if (this.to01) v = (v + 1) * 0.5;
      out[i] = Math.min(1, Math.max(lo, v));
    }
    this.memo = { key, grid: out };
    return out;
  }
}

// ---------------------------------------------------------------------------
// Worley  (cellular noise)
// ---------------------------------------------------------------------------
export type WorleyDistance = "euclidean" | "manhattan" | "chebyshev";
export type WorleyCombination = "F1" | "F2" | "F2-F1" | "F2+F1";

// Per-cell hash → a deterministic float in [0, 1). `w` distinguishes the x
// (0) and y (1) feature-point offsets for the same cell.
function hash01(cx: number, cy: number, seed: number, w: number): number {
  let h =
    (Math.imul(cx, 1664525) +
      Math.imul(cy, 22695477) +
      Math.imul(seed, 1013904223) +
      w * 1664525) &
    0xffffffff;
  h ^= h >>> 16;
  h = Math.imul(h, 0x45d9f3b) & 0xffffffff;
  h ^= h >>> 16;
  return (h >>> 0) / 0x100000000;
}

// Normalization constants per (distance, combination) so to_01 maps the
// typical max distance to ~1.
const WORLEY_NORM: Record<WorleyDistance, Record<WorleyCombination, number>> = {
  euclidean: {
    F1: Math.SQRT2,
    F2: Math.sqrt(2.5),
    "F2-F1": Math.sqrt(2.5),
    "F2+F1": 2 * Math.SQRT2,
  },
  manhattan: { F1: 2, F2: 2, "F2-F1": 2, "F2+F1": 4 },
  chebyshev: { F1: 1, F2: 1.5, "F2-F1": 1.5, "F2+F1": 2 },
};

export interface WorleyOptions {
  distance?: WorleyDistance;
  combination?: WorleyCombination;
  base_freq?: ExprLike;
  base_freq_x?: ExprLike | null;
  base_freq_y?: ExprLike | null;
  to_01?: boolean;
  seed?: ExprLike | null;
}

export class Worley extends NoiseExpression2D {
  private readonly dist: WorleyDistance;
  private readonly comb: WorleyCombination;
  private readonly bfx: number;
  private readonly bfy: number;
  private readonly to01: boolean;
  private readonly norm: number;

  constructor(opts: WorleyOptions = {}) {
    const setArgs = new Set(Object.keys(opts));
    const {
      distance = "euclidean",
      combination = "F1",
      base_freq = 1.0,
      base_freq_x = null,
      base_freq_y = null,
      to_01 = false,
      seed = null,
    } = opts;
    super("Worley", seed == null ? null : coerceInt(seed), setArgs);
    this.dist = distance;
    this.comb = combination;
    this.bfx =
      base_freq_x == null ? coerceFloat(base_freq) : coerceFloat(base_freq_x);
    this.bfy =
      base_freq_y == null ? coerceFloat(base_freq) : coerceFloat(base_freq_y);
    this.to01 = to_01;
    this.norm = WORLEY_NORM[distance][combination];
  }

  argSignature(): string {
    return `Worley|${this.dist}|${this.comb}|${this.bfx}|${this.bfy}|${this.to01}`;
  }

  /** Resolved arguments (excluding seed), for serialization. */
  get exportParams(): {
    distance: WorleyDistance;
    combination: WorleyCombination;
    base_freq_x: number;
    base_freq_y: number;
    to_01: boolean;
  } {
    return {
      distance: this.dist,
      combination: this.comb,
      base_freq_x: this.bfx,
      base_freq_y: this.bfy,
      to_01: this.to01,
    };
  }

  call(x = 0.0, y = 0.0): number {
    const { dist, comb, norm, to01, bfx, bfy, seed } = this;
    const sx = x * bfx;
    const sy = y * bfy;
    const ix = Math.floor(sx);
    const iy = Math.floor(sy);
    let f1 = Infinity;
    let f2 = Infinity;
    for (let dx = -2; dx <= 2; dx++)
      for (let dy = -2; dy <= 2; dy++) {
        const cx = ix + dx;
        const cy = iy + dy;
        const px = hash01(cx, cy, seed, 0) + cx;
        const py = hash01(cx, cy, seed, 1) + cy;
        const ddx = sx - px;
        const ddy = sy - py;
        const d =
          dist === "euclidean"
            ? Math.sqrt(ddx * ddx + ddy * ddy)
            : dist === "manhattan"
              ? Math.abs(ddx) + Math.abs(ddy)
              : Math.max(Math.abs(ddx), Math.abs(ddy));
        if (d < f1) {
          f2 = f1;
          f1 = d;
        } else if (d < f2) f2 = d;
      }
    const v =
      comb === "F1"
        ? f1
        : comb === "F2"
          ? f2
          : comb === "F2-F1"
            ? f2 - f1
            : f2 + f1;
    return Math.max(0, to01 ? Math.min(v / norm, 1) : v);
  }

  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    if (ctx) {
      // Non-separable coords (e.g. under Rotate): sample per cell so the grid
      // path matches call().
      const out = new Float32Array(w * h);
      for (let i = 0; i < out.length; i++)
        out[i] = this.call(ctx.xg[i], ctx.yg[i]);
      return out;
    }
    const key = gridKey(xs, ys, w, h);
    if (this.memo && this.memo.key === key) return this.memo.grid;
    const out = new Float32Array(w * h);
    const { dist, comb, norm, to01, bfx, bfy, seed } = this;
    for (let yi = 0; yi < h; yi++) {
      const y = ys[yi] * bfy;
      const iy = Math.floor(y);
      for (let xi = 0; xi < w; xi++) {
        const x = xs[xi] * bfx;
        const ix = Math.floor(x);
        let f1 = Infinity;
        let f2 = Infinity;
        for (let dx = -2; dx <= 2; dx++)
          for (let dy = -2; dy <= 2; dy++) {
            const cx = ix + dx;
            const cy = iy + dy;
            const px = hash01(cx, cy, seed, 0) + cx;
            const py = hash01(cx, cy, seed, 1) + cy;
            const ddx = x - px;
            const ddy = y - py;
            const d =
              dist === "euclidean"
                ? Math.sqrt(ddx * ddx + ddy * ddy)
                : dist === "manhattan"
                  ? Math.abs(ddx) + Math.abs(ddy)
                  : Math.max(Math.abs(ddx), Math.abs(ddy));
            if (d < f1) {
              f2 = f1;
              f1 = d;
            } else if (d < f2) f2 = d;
          }
        const v =
          comb === "F1"
            ? f1
            : comb === "F2"
              ? f2
              : comb === "F2-F1"
                ? f2 - f1
                : f2 + f1;
        out[yi * w + xi] = Math.max(0, to01 ? Math.min(v / norm, 1) : v);
      }
    }
    this.memo = { key, grid: out };
    return out;
  }
}
