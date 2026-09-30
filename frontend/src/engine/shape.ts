/**
 * shape.ts — SVG-like shapes and the Fill/Stroke operators. A Shape (Rect,
 * Ellipse, Path) is pure geometry exposing the distance to its boundary;
 * Fill/Stroke are the Expression2D nodes that turn it into a binary mask field
 * (1 painted, 0 elsewhere, feathered between). Mirrors matloom/engine/shape.py;
 * distances use only IEEE ops and curves flatten with a fixed step count, so it
 * reproduces across engines.
 */
import {
  Expression2D,
  coerceFloat,
  type ExprLike,
  type CoordCtx,
} from "./expression";
import { makeGrid, type Grid, type Axis } from "./grid";

export type FillMode = "NonZero" | "EvenOdd";

// Subdivisions per cubic Bézier when flattening (fixed, not adaptive, so both
// engines produce identical vertices). Mirrors matloom/engine/shape.py.
const CUBIC_STEPS = 16;

/** Map a signed insideness `s` (> 0 inside) to coverage in [0, 1]. */
function coverage(s: number, feather: number): number {
  if (feather === 0) return s >= 0 ? 1 : 0;
  const t = Math.min(1, Math.max(0, (s + feather / 2) / feather));
  return t * t * (3 - 2 * t);
}

export abstract class Shape {
  /** Signed distance for filling: < 0 inside, > 0 outside. */
  abstract fillDistance(x: number, y: number, mode?: FillMode): number;
  /** Unsigned distance to the stroked boundary. Defaults to |fill|. */
  strokeDistance(x: number, y: number): number {
    return Math.abs(this.fillDistance(x, y, "NonZero"));
  }
}

export class Rect extends Shape {
  private readonly cx: number;
  private readonly cy: number;
  private readonly hx: number;
  private readonly hy: number;
  private readonly r: number;

  constructor(
    readonly x: ExprLike,
    readonly y: ExprLike,
    readonly width: ExprLike,
    readonly height: ExprLike,
    readonly radius: ExprLike = 0,
  ) {
    super();
    const px = coerceFloat(x);
    const py = coerceFloat(y);
    const w = coerceFloat(width);
    const h = coerceFloat(height);
    this.cx = px + w / 2;
    this.cy = py + h / 2;
    this.hx = w / 2;
    this.hy = h / 2;
    this.r = Math.min(coerceFloat(radius), Math.min(w, h) / 2);
  }

  /** Resolved geometry, for serialization. */
  get exportParams(): {
    x: number;
    y: number;
    width: number;
    height: number;
    radius: number;
  } {
    return {
      x: this.cx - this.hx,
      y: this.cy - this.hy,
      width: this.hx * 2,
      height: this.hy * 2,
      radius: this.r,
    };
  }

  fillDistance(x: number, y: number): number {
    const r = this.r;
    const qx = Math.abs(x - this.cx) - (this.hx - r);
    const qy = Math.abs(y - this.cy) - (this.hy - r);
    const ax = Math.max(qx, 0);
    const ay = Math.max(qy, 0);
    const outside = Math.sqrt(ax * ax + ay * ay);
    const inside = Math.min(Math.max(qx, qy), 0);
    return outside + inside - r;
  }
}

export class Ellipse extends Shape {
  private readonly cxv: number;
  private readonly cyv: number;
  private readonly rx: number;
  private readonly ry: number;

  constructor(
    readonly cx: ExprLike,
    readonly cy: ExprLike,
    readonly radiusX: ExprLike,
    readonly radiusY: ExprLike,
  ) {
    super();
    this.cxv = coerceFloat(cx);
    this.cyv = coerceFloat(cy);
    this.rx = coerceFloat(radiusX);
    this.ry = coerceFloat(radiusY);
  }

  /** Resolved geometry, for serialization. */
  get exportParams(): { cx: number; cy: number; rx: number; ry: number } {
    return { cx: this.cxv, cy: this.cyv, rx: this.rx, ry: this.ry };
  }

  fillDistance(x: number, y: number): number {
    const px = x - this.cxv;
    const py = y - this.cyv;
    const { rx, ry } = this;
    const ex = px / rx;
    const ey = py / ry;
    const k1 = Math.sqrt(ex * ex + ey * ey);
    const fx = px / (rx * rx);
    const fy = py / (ry * ry);
    const k2 = Math.sqrt(fx * fx + fy * fy);
    if (k2 === 0) return -Math.min(rx, ry);
    return (k1 * (k1 - 1)) / k2;
  }
}

export type Segment =
  | { kind: "line"; x: number; y: number }
  | {
      kind: "cubic";
      c1x: number;
      c1y: number;
      c2x: number;
      c2y: number;
      x: number;
      y: number;
    };

/** A straight segment to (x, y). */
export function lineTo(x: ExprLike, y: ExprLike): Segment {
  return { kind: "line", x: coerceFloat(x), y: coerceFloat(y) };
}

/** A cubic Bézier to (x, y) with control points (c1*, c2*). */
export function cubicTo(
  c1x: ExprLike,
  c1y: ExprLike,
  c2x: ExprLike,
  c2y: ExprLike,
  x: ExprLike,
  y: ExprLike,
): Segment {
  return {
    kind: "cubic",
    c1x: coerceFloat(c1x),
    c1y: coerceFloat(c1y),
    c2x: coerceFloat(c2x),
    c2y: coerceFloat(c2y),
    x: coerceFloat(x),
    y: coerceFloat(y),
  };
}

export class Path extends Shape {
  private readonly sx: number;
  private readonly sy: number;
  private readonly segs: Segment[];
  private readonly px: number[];
  private readonly py: number[];

  constructor(startX: ExprLike, startY: ExprLike, segments: Segment[]) {
    super();
    if (!segments.length) throw new Error("Path needs at least one segment");
    this.sx = coerceFloat(startX);
    this.sy = coerceFloat(startY);
    this.segs = segments;
    const pts = this.flatten();
    this.px = pts.map((p) => p[0]);
    this.py = pts.map((p) => p[1]);
  }

  /** Resolved start point and segments, for serialization. */
  get exportParams(): { startX: number; startY: number; segments: Segment[] } {
    return { startX: this.sx, startY: this.sy, segments: this.segs };
  }

  private flatten(): [number, number][] {
    const pts: [number, number][] = [[this.sx, this.sy]];
    let cx = this.sx;
    let cy = this.sy;
    for (const seg of this.segs) {
      if (seg.kind === "line") {
        cx = seg.x;
        cy = seg.y;
        pts.push([cx, cy]);
      } else {
        for (let k = 1; k <= CUBIC_STEPS; k++) {
          const t = k / CUBIC_STEPS;
          const mt = 1 - t;
          const a = mt * mt * mt;
          const b = 3 * mt * mt * t;
          const c = 3 * mt * t * t;
          const d = t * t * t;
          pts.push([
            a * cx + b * seg.c1x + c * seg.c2x + d * seg.x,
            a * cy + b * seg.c1y + c * seg.c2y + d * seg.y,
          ]);
        }
        cx = seg.x;
        cy = seg.y;
      }
    }
    return pts;
  }

  private polyDistance(x: number, y: number, closed: boolean): number {
    const { px, py } = this;
    const n = px.length;
    const last = closed ? n : n - 1;
    let dist = Infinity;
    for (let i = 0; i < last; i++) {
      const j = (i + 1) % n;
      const ax = px[i];
      const ay = py[i];
      const abx = px[j] - ax;
      const aby = py[j] - ay;
      const apx = x - ax;
      const apy = y - ay;
      const denom = abx * abx + aby * aby;
      let d: number;
      if (denom === 0) {
        d = Math.sqrt(apx * apx + apy * apy);
      } else {
        const t = Math.min(1, Math.max(0, (apx * abx + apy * aby) / denom));
        const dx = apx - t * abx;
        const dy = apy - t * aby;
        d = Math.sqrt(dx * dx + dy * dy);
      }
      if (d < dist) dist = d;
    }
    return dist;
  }

  private inside(x: number, y: number, mode: FillMode): boolean {
    const { px, py } = this;
    const n = px.length;
    if (mode === "EvenOdd") {
      let cn = 0;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        const ay = py[i];
        const by = py[j];
        if (ay <= y !== by <= y) {
          const xint = px[i] + ((y - ay) * (px[j] - px[i])) / (by - ay);
          if (x < xint) cn ^= 1;
        }
      }
      return cn === 1;
    }
    let wn = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const ax = px[i];
      const ay = py[i];
      const by = py[j];
      const cr = (px[j] - ax) * (y - ay) - (x - ax) * (by - ay);
      if (ay <= y) {
        if (by > y && cr > 0) wn++;
      } else if (by <= y && cr < 0) wn--;
    }
    return wn !== 0;
  }

  fillDistance(x: number, y: number, mode: FillMode = "NonZero"): number {
    const d = this.polyDistance(x, y, true);
    return this.inside(x, y, mode) ? -d : d;
  }

  strokeDistance(x: number, y: number): number {
    return this.polyDistance(x, y, false);
  }
}

export class Fill extends Expression2D {
  private readonly shape: Shape;
  private readonly mode: FillMode;
  private readonly feather: number;

  constructor(shape: Shape, mode: FillMode = "NonZero", feather: ExprLike = 0) {
    super();
    this.shape = shape;
    this.mode = mode;
    this.feather = coerceFloat(feather);
  }

  get shapeRef(): Shape {
    return this.shape;
  }
  get exportParams(): { mode: FillMode; feather: number } {
    return { mode: this.mode, feather: this.feather };
  }

  call(x = 0.0, y = 0.0): number {
    return coverage(-this.shape.fillDistance(x, y, this.mode), this.feather);
  }

  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    if (ctx) return makeGrid(w * h, (i) => this.call(ctx.xg[i], ctx.yg[i]));
    return makeGrid(w * h, (i) => this.call(xs[i % w], ys[(i / w) | 0]));
  }
}

export class Stroke extends Expression2D {
  private readonly shape: Shape;
  private readonly width: number;
  private readonly feather: number;

  constructor(shape: Shape, width: ExprLike, feather: ExprLike = 0) {
    super();
    this.shape = shape;
    this.width = coerceFloat(width);
    this.feather = coerceFloat(feather);
  }

  get shapeRef(): Shape {
    return this.shape;
  }
  get exportParams(): { width: number; feather: number } {
    return { width: this.width, feather: this.feather };
  }

  call(x = 0.0, y = 0.0): number {
    const d = this.shape.strokeDistance(x, y);
    return coverage(this.width / 2 - d, this.feather);
  }

  evalGrid(xs: Axis, ys: Axis, w: number, h: number, ctx?: CoordCtx): Grid {
    if (ctx) return makeGrid(w * h, (i) => this.call(ctx.xg[i], ctx.yg[i]));
    return makeGrid(w * h, (i) => this.call(xs[i % w], ys[(i / w) | 0]));
  }
}
