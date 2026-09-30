/**
 * pattern.ts — parametric tiling patterns that evaluate to a binary mask field.
 * Houses Bricks (a running-bond brick lattice) and Weave (an over/under woven-
 * thread lattice). Mirrors matloom/engine/pattern.py; the per-cell math is
 * unchanged.
 */
import {
  Expression2D,
  coerceFloat,
  type ExprLike,
  type CoordCtx,
} from "./expression";
import { makeGrid, type Grid, type Axis } from "./grid";

export type BricksAxis = "row" | "column";

export interface BricksOptions {
  brick_width?: ExprLike;
  brick_height?: ExprLike;
  offset?: ExprLike;
  mortar?: ExprLike;
  axis?: BricksAxis;
  feather?: ExprLike;
}

export class Bricks extends Expression2D {
  private readonly bw: number;
  private readonly bh: number;
  private readonly offset: number;
  private readonly mortar: number;
  private readonly axis: BricksAxis;
  private readonly feather: number;

  constructor(opts: BricksOptions = {}) {
    super();
    const {
      brick_width = 1.0,
      brick_height = 0.5,
      offset = 0.5,
      mortar = 0.05,
      axis = "row",
      feather = 0.0,
    } = opts;
    this.bw = coerceFloat(brick_width);
    this.bh = coerceFloat(brick_height);
    this.offset = coerceFloat(offset);
    this.mortar = coerceFloat(mortar);
    this.axis = axis;
    this.feather = coerceFloat(feather);
  }

  /** Resolved arguments, for serialization. */
  get exportParams(): {
    brick_width: number;
    brick_height: number;
    offset: number;
    mortar: number;
    axis: BricksAxis;
    feather: number;
  } {
    return {
      brick_width: this.bw,
      brick_height: this.bh,
      offset: this.offset,
      mortar: this.mortar,
      axis: this.axis,
      feather: this.feather,
    };
  }

  private coverage(d: number): number {
    const f = this.feather;
    if (f === 0) return d >= 0 ? 1 : 0;
    const t = (d + f / 2) / f;
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    return t * t * (3 - 2 * t);
  }

  call(x = 0.0, y = 0.0): number {
    const { bw, bh } = this;
    let lx: number;
    let ly: number;
    if (this.axis === "row") {
      const row = Math.floor(y / bh);
      lx = x - row * this.offset * bw;
      lx = lx - bw * Math.floor(lx / bw);
      ly = y - bh * Math.floor(y / bh);
    } else {
      const col = Math.floor(x / bw);
      lx = x - bw * Math.floor(x / bw);
      ly = y - col * this.offset * bh;
      ly = ly - bh * Math.floor(ly / bh);
    }
    const halfM = this.mortar / 2;
    const dxIn = Math.min(lx - halfM, bw - halfM - lx);
    const dyIn = Math.min(ly - halfM, bh - halfM - ly);
    return this.coverage(Math.min(dxIn, dyIn));
  }

  // The pattern is cheap and purely pointwise, so evaluating cell-by-cell
  // through call() keeps the grid path byte-identical to the scalar path.
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    if (ctx) return makeGrid(w * h, (i) => this.call(ctx.xg[i], ctx.yg[i]));
    return makeGrid(w * h, (i) => this.call(xs[i % w], ys[(i / w) | 0]));
  }
}

export interface WeaveOptions {
  over?: ExprLike;
  under?: ExprLike;
  shift?: ExprLike;
  base_freq?: ExprLike;
  base_freq_x?: ExprLike | null;
  base_freq_y?: ExprLike | null;
  warp_width?: ExprLike;
  feather?: ExprLike;
}

/**
 * Weave — an over/under interlacing of warp (lengthwise) and weft (crosswise)
 * threads, the canonical structure of woven cloth. Evaluates to 1 on the thread
 * that is on top at a texel and 0 in the inter-thread gaps. Mirrors
 * matloom/engine/pattern.py Weave.
 */
export class Weave extends Expression2D {
  private readonly over: number;
  private readonly under: number;
  private readonly shift: number;
  private readonly bfx: number;
  private readonly bfy: number;
  private readonly warpWidth: number;
  private readonly feather: number;

  constructor(opts: WeaveOptions = {}) {
    super();
    const {
      over = 1,
      under = 1,
      shift = 1,
      base_freq = 8.0,
      base_freq_x = null,
      base_freq_y = null,
      warp_width = 0.5,
      feather = 0.0,
    } = opts;
    this.over = coerceFloat(over);
    this.under = coerceFloat(under);
    this.shift = coerceFloat(shift);
    this.bfx =
      base_freq_x == null ? coerceFloat(base_freq) : coerceFloat(base_freq_x);
    this.bfy =
      base_freq_y == null ? coerceFloat(base_freq) : coerceFloat(base_freq_y);
    this.warpWidth = coerceFloat(warp_width);
    this.feather = coerceFloat(feather);
  }

  /** Resolved arguments, for serialization. */
  get exportParams(): {
    over: number;
    under: number;
    shift: number;
    base_freq_x: number;
    base_freq_y: number;
    warp_width: number;
    feather: number;
  } {
    return {
      over: this.over,
      under: this.under,
      shift: this.shift,
      base_freq_x: this.bfx,
      base_freq_y: this.bfy,
      warp_width: this.warpWidth,
      feather: this.feather,
    };
  }

  private coverage(d: number): number {
    const f = this.feather;
    if (f === 0) return d >= 0 ? 1 : 0;
    const t = (d + f / 2) / f;
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    return t * t * (3 - 2 * t);
  }

  call(x = 0.0, y = 0.0): number {
    const period = this.over + this.under;
    const u = x * this.bfx;
    const v = y * this.bfy;
    const col = Math.floor(u);
    const row = Math.floor(v);
    // Match Python/numpy mod: always non-negative for a positive divisor.
    let m = (col - row * this.shift) % period;
    if (m < 0) m += period;
    const warpOnTop = m < this.over;
    const half = this.warpWidth / 2;
    const d = warpOnTop
      ? half - Math.abs(u - col - 0.5)
      : half - Math.abs(v - row - 0.5);
    return this.coverage(d);
  }

  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    if (ctx) return makeGrid(w * h, (i) => this.call(ctx.xg[i], ctx.yg[i]));
    return makeGrid(w * h, (i) => this.call(xs[i % w], ys[(i / w) | 0]));
  }
}
