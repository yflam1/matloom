/**
 * expression.ts — the Expression2D tree: leaves, unary/binary nodes, the
 * functional Add/Sub/Mul/Div/Pow constructors, and Threshold. Mirrors
 * matloom/engine/{expr,utils}.py exactly; the per-cell arithmetic is unchanged.
 */
import {
  type Grid,
  type Axis,
  gridFull,
  gridFromX,
  gridFromY,
  gridAdd,
  gridSub,
  gridMul,
  gridDiv,
  gridPow,
  gridMin,
  gridMax,
  linearToSrgb,
  srgbToLinear,
  linearToSrgbScalar,
  srgbToLinearScalar,
} from "./grid";

/**
 * Per-cell coordinate grids (each length w*h) that override the X/Y leaves,
 * letting a non-separable transform (Rotate) sample its child subtree. Noise
 * nodes ignore them and keep sampling the separable axes, so they stay
 * axis-aligned. Mirrors the `coords` argument in matloom/engine/expr.py.
 */
export interface CoordCtx {
  xg: Grid;
  yg: Grid;
}

/** A scalar-valued function of the sample grid. */
export abstract class Expression2D {
  abstract evalGrid(
    xs: Axis,
    ys: Axis,
    w: number,
    h: number,
    ctx?: CoordCtx,
  ): Grid;
  /**
   * Evaluate at a single point. Mirrors expr.py's `__call__` (x/y default to
   * 0), and is the scalar path coerceFloat/coerceInt use to reduce an
   * expression passed where a plain number is expected.
   */
  abstract call(x?: number, y?: number): number;
  /**
   * Direct sub-expressions, for walking the tree (e.g. to find noise nodes, or
   * one day to transpile it). Leaves return none; composites override this.
   */
  children(): Expression2D[] {
    return [];
  }
}

/** Anything an expression argument may be before coercion. */
export type ExprLike = Expression2D | number;

// Wrap a bare number as a Constant so callers can pass plain numbers wherever
// an Expression2D is expected. Mirrors expr.py's @coerce_args decorator (the
// expression-typed direction).
export function coerceConstant(v: ExprLike): Expression2D {
  return typeof v === "number" ? new Constant(v) : v;
}

// Python's round(): round half to even (banker's rounding), so coerceInt
// matches CPython for the .5 case (round(2.5) == 2, round(1.5) == 2).
function pyRound(x: number): number {
  const fl = Math.floor(x);
  const diff = x - fl;
  if (diff < 0.5) return fl;
  if (diff > 0.5) return fl + 1;
  return fl % 2 === 0 ? fl : fl + 1;
}

// Reduce an ExprLike to a plain float where a scalar is expected: an
// Expression2D is evaluated at the origin (x=y=0). Mirrors @coerce_args's
// numeric-parameter direction, letting Log/Sqrt/… stand in for bare numbers,
// e.g. fBm({ base_freq: new Sqrt(2) }).
export function coerceFloat(v: ExprLike): number {
  return typeof v === "number" ? v : v.call();
}

// As coerceFloat, but for int-typed parameters: a coerced expression value is
// rounded (matching Python's round()), so e.g. fBm({ octaves: new Log(1000, 10) })
// resolves to 3. Plain numbers pass through untouched.
export function coerceInt(v: ExprLike): number {
  return typeof v === "number" ? v : pyRound(v.call());
}

// ---------------------------------------------------------------------------
// Leaf & unary expressions  (expr.py + utils.py)
// ---------------------------------------------------------------------------
export class Constant extends Expression2D {
  private readonly value: number;
  constructor(value: ExprLike = 0.0) {
    super();
    this.value = coerceFloat(value);
  }
  /** The literal scalar, for serialization. */
  get literal(): number {
    return this.value;
  }
  call(): number {
    return this.value;
  }
  evalGrid(_xs: Axis, _ys: Axis, w: number, h: number): Grid {
    return gridFull(w * h, this.value);
  }
}

export class X extends Expression2D {
  call(x = 0.0): number {
    return x;
  }
  evalGrid(xs: Axis, _ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    return ctx ? ctx.xg : gridFromX(xs, w, h);
  }
}

export class Y extends Expression2D {
  call(_x = 0.0, y = 0.0): number {
    return y;
  }
  evalGrid(_xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    return ctx ? ctx.yg : gridFromY(ys, w, h);
  }
}

/**
 * A named reference to a stored expression (a `Define`). Delegates evaluation
 * to its target but serializes back to the bare name, and — by keeping the
 * default empty children() — stops tree walks (collectNoise) at the boundary,
 * so the definition's noise is owned by the definition, not each use site.
 * Mirrors Ref in matloom/engine/expr.py.
 */
export class Ref extends Expression2D {
  constructor(
    readonly name: string,
    private readonly target: Expression2D,
  ) {
    super();
  }
  /** The resolved definition this name points to (for codegen/transpilation). */
  get refTarget(): Expression2D {
    return this.target;
  }
  call(x = 0.0, y = 0.0): number {
    return this.target.call(x, y);
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    return this.target.evalGrid(xs, ys, w, h, ctx);
  }
}

/** Base for the single-operand expressions (Sin, Cos, sRGB/linear). */
abstract class UnaryExpression2D extends Expression2D {
  protected readonly e: Expression2D;
  constructor(e: ExprLike) {
    super();
    this.e = coerceConstant(e);
  }
  children(): Expression2D[] {
    return [this.e];
  }
}

export class Sin extends UnaryExpression2D {
  call(x = 0.0, y = 0.0): number {
    return Math.sin(this.e.call(x, y));
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const src = this.e.evalGrid(xs, ys, w, h, ctx);
    const n = src.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.sin(src[i]);
    return out;
  }
}
export class Cos extends UnaryExpression2D {
  call(x = 0.0, y = 0.0): number {
    return Math.cos(this.e.call(x, y));
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const src = this.e.evalGrid(xs, ys, w, h, ctx);
    const n = src.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.cos(src[i]);
    return out;
  }
}
export class Log extends Expression2D {
  private readonly e: Expression2D;
  private readonly base: number;
  constructor(expr: ExprLike, base: ExprLike = Math.E) {
    super();
    this.e = coerceConstant(expr);
    this.base = coerceFloat(base);
  }
  children(): Expression2D[] {
    return [this.e];
  }
  /** Logarithm base, for serialization. */
  get logBase(): number {
    return this.base;
  }
  // Scalar path mirrors math.log(v, base): natural log divided by log(base),
  // NOT Math.log10/Math.log2, so the value matches Python exactly (those
  // special cases differ in the last ULP — e.g. log10(1000) is 3 vs. 2.999…).
  call(x = 0.0, y = 0.0): number {
    const v = this.e.call(x, y);
    return this.base === Math.E
      ? Math.log(v)
      : Math.log(v) / Math.log(this.base);
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const src = this.e.evalGrid(xs, ys, w, h, ctx);
    const n = src.length;
    const out = new Float32Array(n);
    // The grid path keeps the base-10/2 special cases (as before) because
    // Math.log10/Math.log2 differ from log/log(base) in the last ULP and the
    // parity fixtures were generated against them.
    if (this.base === 10)
      for (let i = 0; i < n; i++) out[i] = Math.log10(src[i]);
    else if (this.base === 2)
      for (let i = 0; i < n; i++) out[i] = Math.log2(src[i]);
    else if (this.base === Math.E)
      for (let i = 0; i < n; i++) out[i] = Math.log(src[i]);
    else {
      const denom = Math.log(this.base);
      for (let i = 0; i < n; i++) out[i] = Math.log(src[i]) / denom;
    }
    return out;
  }
}
export class Abs extends UnaryExpression2D {
  call(x = 0.0, y = 0.0): number {
    return Math.abs(this.e.call(x, y));
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const src = this.e.evalGrid(xs, ys, w, h, ctx);
    const n = src.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.abs(src[i]);
    return out;
  }
}
export class Sqrt extends UnaryExpression2D {
  call(x = 0.0, y = 0.0): number {
    return Math.sqrt(this.e.call(x, y));
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const src = this.e.evalGrid(xs, ys, w, h, ctx);
    const n = src.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.sqrt(src[i]);
    return out;
  }
}
export class Floor extends UnaryExpression2D {
  call(x = 0.0, y = 0.0): number {
    return Math.floor(this.e.call(x, y));
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const src = this.e.evalGrid(xs, ys, w, h, ctx);
    const n = src.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.floor(src[i]);
    return out;
  }
}
export class Ceil extends UnaryExpression2D {
  call(x = 0.0, y = 0.0): number {
    return Math.ceil(this.e.call(x, y));
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const src = this.e.evalGrid(xs, ys, w, h, ctx);
    const n = src.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.ceil(src[i]);
    return out;
  }
}
export class sRGB2Linear extends UnaryExpression2D {
  call(x = 0.0, y = 0.0): number {
    return srgbToLinearScalar(this.e.call(x, y));
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    return srgbToLinear(this.e.evalGrid(xs, ys, w, h, ctx));
  }
}
export class Linear2sRGB extends UnaryExpression2D {
  call(x = 0.0, y = 0.0): number {
    return linearToSrgbScalar(this.e.call(x, y));
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    return linearToSrgb(this.e.evalGrid(xs, ys, w, h, ctx));
  }
}

// ---------------------------------------------------------------------------
// Binary composites  (expr.py)
// ---------------------------------------------------------------------------
type GridBinOp = (a: Grid, b: Grid) => Grid;
type ScalarBinOp = (a: number, b: number) => number;

/** Base for the two-operand expressions; subclasses supply the grid op. */
abstract class BinaryExpression2D extends Expression2D {
  protected readonly l: Expression2D;
  protected readonly r: Expression2D;
  constructor(
    l: ExprLike,
    r: ExprLike,
    private readonly op: GridBinOp,
    private readonly scalarOp: ScalarBinOp,
  ) {
    super();
    this.l = coerceConstant(l);
    this.r = coerceConstant(r);
  }
  call(x = 0.0, y = 0.0): number {
    return this.scalarOp(this.l.call(x, y), this.r.call(x, y));
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    return this.op(
      this.l.evalGrid(xs, ys, w, h, ctx),
      this.r.evalGrid(xs, ys, w, h, ctx),
    );
  }
  children(): Expression2D[] {
    return [this.l, this.r];
  }
}

export class AdditionExpression2D extends BinaryExpression2D {
  constructor(l: ExprLike, r: ExprLike) {
    super(l, r, gridAdd, (a, b) => a + b);
  }
}
export class SubtractionExpression2D extends BinaryExpression2D {
  constructor(l: ExprLike, r: ExprLike) {
    super(l, r, gridSub, (a, b) => a - b);
  }
}
export class MultiplicationExpression2D extends BinaryExpression2D {
  constructor(l: ExprLike, r: ExprLike) {
    super(l, r, gridMul, (a, b) => a * b);
  }
}
export class DivisionExpression2D extends BinaryExpression2D {
  constructor(l: ExprLike, r: ExprLike) {
    super(l, r, gridDiv, (a, b) => a / b);
  }
  // Guard division by zero: return 0 (with a warning) wherever the divisor is
  // 0, instead of producing Infinity/NaN. Mirrors DivisionExpression2D in
  // matloom/engine/expr.py.
  call(x = 0.0, y = 0.0): number {
    const b = this.r.call(x, y);
    if (b === 0) {
      console.warn(`Division by zero at (x=${x}, y=${y}); returning 0`);
      return 0;
    }
    const a = this.l.call(x, y);
    return a / b;
  }
  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const a = this.l.evalGrid(xs, ys, w, h, ctx);
    const b = this.r.evalGrid(xs, ys, w, h, ctx);
    const out = new Float32Array(a.length);
    let zeros = 0;
    for (let i = 0; i < a.length; i++) {
      if (b[i] === 0) {
        zeros++;
        out[i] = 0;
      } else {
        out[i] = a[i] / b[i];
      }
    }
    if (zeros > 0) {
      console.warn(
        `Division by zero at ${zeros} grid point(s); returning 0 there`,
      );
    }
    return out;
  }
}
export class PowerExpression2D extends BinaryExpression2D {
  constructor(l: ExprLike, r: ExprLike) {
    super(l, r, gridPow, (a, b) => a ** b);
  }
}
export class Min extends BinaryExpression2D {
  constructor(l: ExprLike, r: ExprLike) {
    super(l, r, gridMin, Math.min);
  }
}
export class Max extends BinaryExpression2D {
  constructor(l: ExprLike, r: ExprLike) {
    super(l, r, gridMax, Math.max);
  }
}

// Functional constructors mirroring expr.py's Add/Sub/Mul/Div/Pow helpers, so
// the same `Div(a, b)` form works in both engines.
export const Add = (left: ExprLike, right: ExprLike) =>
  new AdditionExpression2D(left, right);
export const Sub = (minuend: ExprLike, subtrahend: ExprLike) =>
  new SubtractionExpression2D(minuend, subtrahend);
export const Mul = (left: ExprLike, right: ExprLike) =>
  new MultiplicationExpression2D(left, right);
export const Div = (dividend: ExprLike, divisor: ExprLike) =>
  new DivisionExpression2D(dividend, divisor);
export const Pow = (base: ExprLike, exponent: ExprLike) =>
  new PowerExpression2D(base, exponent);

// ---------------------------------------------------------------------------
// Threshold  (utils.py)
// ---------------------------------------------------------------------------
export interface ThresholdOptions {
  below_at?: ExprLike | null;
  below_to?: ExprLike | null;
  above_at?: ExprLike | null;
  above_to?: ExprLike | null;
  transition_width?: ExprLike;
}

// Coerce an optional scalar option to a number (or null), so callers may pass
// an expression — e.g. Threshold(x, { above_at: new Log(1000, 10) }) — exactly
// as @coerce_args allows on the Python side.
function coerceFloatOrNull(v: ExprLike | null | undefined): number | null {
  return v == null ? null : coerceFloat(v);
}

// Standard GLSL smoothstep, matching utils.py's Threshold._smoothstep — used by
// the scalar call() path (the grid path inlines the same S-curve).
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min((x - edge0) / (edge1 - edge0), 1));
  return t * t * (3 - 2 * t);
}

export class Threshold extends Expression2D {
  private readonly e: Expression2D;
  private readonly ba: number | null;
  private readonly bt: number | null;
  private readonly aa: number | null;
  private readonly at: number | null;
  private readonly tw: number;

  constructor(expr: ExprLike, opts: ThresholdOptions = {}) {
    super();
    const {
      below_at = null,
      below_to = null,
      above_at = null,
      above_to = null,
      transition_width = 0,
    } = opts;
    const ba = coerceFloatOrNull(below_at);
    const bt = coerceFloatOrNull(below_to);
    const aa = coerceFloatOrNull(above_at);
    const at = coerceFloatOrNull(above_to);
    this.e = coerceConstant(expr);
    this.ba = ba;
    this.bt = ba !== null && bt === null ? ba : bt;
    this.aa = aa;
    this.at = aa !== null && at === null ? aa : at;
    this.tw = coerceFloat(transition_width);
  }

  /** Resolved threshold cutoffs, for serialization. */
  get exportParams(): {
    below_at: number | null;
    below_to: number | null;
    above_at: number | null;
    above_to: number | null;
    transition_width: number;
  } {
    return {
      below_at: this.ba,
      below_to: this.bt,
      above_at: this.aa,
      above_to: this.at,
      transition_width: this.tw,
    };
  }

  call(x = 0.0, y = 0.0): number {
    const v = this.e.call(x, y);
    let out = v;
    const { aa, at, ba, bt, tw } = this;
    if (aa !== null && at !== null) {
      if (tw === 0) {
        if (v >= aa) out = at;
      } else {
        const t = smoothstep(aa - tw, aa, v);
        out = (1 - t) * v + t * at;
      }
    }
    if (ba !== null && bt !== null) {
      if (tw === 0) {
        if (v <= ba) out = bt;
      } else {
        const t = smoothstep(ba, ba + tw, v);
        out = (1 - t) * bt + t * v;
      }
    }
    return out;
  }

  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    const src = this.e.evalGrid(xs, ys, w, h, ctx);
    const out = new Float32Array(src.length);
    const { aa, at, ba, bt, tw } = this;
    const hasAbove = aa !== null;
    const hasBelow = ba !== null;
    for (let i = 0; i < src.length; i++) {
      const v = src[i];
      let r = v;
      if (hasAbove) {
        if (tw === 0) {
          if (v >= aa) r = at as number;
        } else {
          const t = (v - (aa - tw)) / tw;
          if (t >= 1) r = at as number;
          else if (t > 0) {
            const s = t * t * (3 - 2 * t);
            r = (1 - s) * v + s * (at as number);
          }
        }
      }
      if (hasBelow) {
        if (tw === 0) {
          if (v <= ba) r = bt as number;
        } else {
          const t = (v - ba) / tw;
          if (t <= 0) r = bt as number;
          else if (t < 1) {
            const s = t * t * (3 - 2 * t);
            r = (1 - s) * (bt as number) + s * v;
          }
        }
      }
      out[i] = r;
    }
    return out;
  }

  children(): Expression2D[] {
    return [this.e];
  }
}
