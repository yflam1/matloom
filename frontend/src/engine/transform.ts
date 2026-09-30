/**
 * transform.ts — coordinate transforms that wrap any Expression2D. A transform
 * remaps the sample point before delegating to its child, so it works on any
 * expression (shapes, noise, gradients, …). Translate and Scale are
 * axis-separable (they rewrite the 1-D sample axes); Rotate couples x and y, so
 * it samples its child on explicit per-cell coordinate grids (the `ctx`
 * argument) — noise nodes fall back to per-cell sampling there, so they rotate
 * too (slower than their separable path). All transforms operate about the
 * origin. Mirrors matloom/engine/transform.py.
 */
import {
  Expression2D,
  coerceConstant,
  coerceFloat,
  type ExprLike,
  type CoordCtx,
} from "./expression";
import { type Grid, type Axis } from "./grid";

export class Translate extends Expression2D {
  private readonly e: Expression2D;
  private readonly dx: number;
  private readonly dy: number;

  constructor(expr: ExprLike, dx: ExprLike = 0, dy: ExprLike = 0) {
    super();
    this.e = coerceConstant(expr);
    this.dx = coerceFloat(dx);
    this.dy = coerceFloat(dy);
  }

  /** Resolved shifts, for serialization. */
  get exportParams(): { dx: number; dy: number } {
    return { dx: this.dx, dy: this.dy };
  }

  call(x = 0.0, y = 0.0): number {
    return this.e.call(x - this.dx, y - this.dy);
  }

  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const xs2 = new Float32Array(w);
    const ys2 = new Float32Array(h);
    for (let i = 0; i < w; i++) xs2[i] = xs[i] - this.dx;
    for (let i = 0; i < h; i++) ys2[i] = ys[i] - this.dy;
    let ctx2: CoordCtx | undefined;
    if (ctx) {
      const xg = new Float32Array(ctx.xg.length);
      const yg = new Float32Array(ctx.yg.length);
      for (let i = 0; i < xg.length; i++) {
        xg[i] = ctx.xg[i] - this.dx;
        yg[i] = ctx.yg[i] - this.dy;
      }
      ctx2 = { xg, yg };
    }
    return this.e.evalGrid(xs2, ys2, w, h, ctx2);
  }

  children(): Expression2D[] {
    return [this.e];
  }
}

export class Scale extends Expression2D {
  private readonly e: Expression2D;
  private readonly sx: number;
  private readonly sy: number;

  constructor(expr: ExprLike, sx: ExprLike = 1, sy: ExprLike = 1) {
    super();
    this.e = coerceConstant(expr);
    this.sx = coerceFloat(sx);
    this.sy = coerceFloat(sy);
    if (this.sx === 0 || this.sy === 0)
      throw new Error("Scale factors must be non-zero");
  }

  /** Resolved factors, for serialization. */
  get exportParams(): { sx: number; sy: number } {
    return { sx: this.sx, sy: this.sy };
  }

  call(x = 0.0, y = 0.0): number {
    return this.e.call(x / this.sx, y / this.sy);
  }

  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const xs2 = new Float32Array(w);
    const ys2 = new Float32Array(h);
    for (let i = 0; i < w; i++) xs2[i] = xs[i] / this.sx;
    for (let i = 0; i < h; i++) ys2[i] = ys[i] / this.sy;
    let ctx2: CoordCtx | undefined;
    if (ctx) {
      const xg = new Float32Array(ctx.xg.length);
      const yg = new Float32Array(ctx.yg.length);
      for (let i = 0; i < xg.length; i++) {
        xg[i] = ctx.xg[i] / this.sx;
        yg[i] = ctx.yg[i] / this.sy;
      }
      ctx2 = { xg, yg };
    }
    return this.e.evalGrid(xs2, ys2, w, h, ctx2);
  }

  children(): Expression2D[] {
    return [this.e];
  }
}

export class Rotate extends Expression2D {
  private readonly e: Expression2D;
  private readonly degrees: number;
  private readonly cos: number;
  private readonly sin: number;

  constructor(expr: ExprLike, degrees: ExprLike = 0) {
    super();
    this.e = coerceConstant(expr);
    this.degrees = coerceFloat(degrees);
    const rad = (this.degrees * Math.PI) / 180;
    this.cos = Math.cos(rad);
    this.sin = Math.sin(rad);
  }

  /** Resolved angle, for serialization. */
  get exportParams(): { degrees: number } {
    return { degrees: this.degrees };
  }

  call(x = 0.0, y = 0.0): number {
    const xr = this.cos * x + this.sin * y;
    const yr = -this.sin * x + this.cos * y;
    return this.e.call(xr, yr);
  }

  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const n = w * h;
    const xg = new Float32Array(n);
    const yg = new Float32Array(n);
    const c = this.cos;
    const s = this.sin;
    if (ctx) {
      for (let i = 0; i < n; i++) {
        const bx = ctx.xg[i];
        const by = ctx.yg[i];
        xg[i] = c * bx + s * by;
        yg[i] = -s * bx + c * by;
      }
    } else {
      for (let yi = 0; yi < h; yi++)
        for (let xi = 0; xi < w; xi++) {
          const i = yi * w + xi;
          const bx = xs[xi];
          const by = ys[yi];
          xg[i] = c * bx + s * by;
          yg[i] = -s * bx + c * by;
        }
    }
    // xs/ys are still forwarded for any separable child path below.
    return this.e.evalGrid(xs, ys, w, h, { xg, yg });
  }

  children(): Expression2D[] {
    return [this.e];
  }
}
