/**
 * wgsl-codegen.ts — translate an Expression2D tree into a WGSL expression for
 * the WebGPU preview tier.
 *
 * This is NOT a parity-faithful path: GPU transcendentals (sin/cos/log/…) and
 * fused multiply-adds are spec'd only to a few ULP and differ from the CPU/libm
 * engine beyond the cross-engine tolerance. The result is a fast *approximate*
 * preview; the authoritative frame always comes from a CPU tier.
 *
 * Coverage is the analytic core: constants, X/Y, the arithmetic and unary math
 * nodes, sRGB conversions, Threshold, named refs, and the coordinate transforms
 * (Translate/Scale/Rotate, expressed as coordinate substitutions — the GPU
 * evaluates per cell, so even the non-separable Rotate is just a substitution).
 * Noise (fBm/Worley), Bricks and the SVG shapes are NOT codegen'd; hitting one
 * throws {@link GpuUnsupportedError}, and the backend falls back to the CPU tier
 * for that material. The emitted helpers `s2l`/`l2s`/`thr` live in
 * {@link WGSL_PRELUDE}.
 */
import {
  Expression2D,
  Constant,
  X,
  Y,
  Ref,
  Sin,
  Cos,
  Log,
  Abs,
  Sqrt,
  Floor,
  Ceil,
  sRGB2Linear,
  Linear2sRGB,
  AdditionExpression2D,
  SubtractionExpression2D,
  MultiplicationExpression2D,
  DivisionExpression2D,
  PowerExpression2D,
  Min,
  Max,
  Threshold,
  Translate,
  Scale,
  Rotate,
} from "../engine";

/** Thrown when a node has no WGSL translation; signals "fall back to the CPU". */
export class GpuUnsupportedError extends Error {
  constructor(node: string) {
    super(`WebGPU codegen does not support: ${node}`);
    this.name = "GpuUnsupportedError";
  }
}

// Format a finite JS number as a WGSL f32 literal. `String(n)` is round-trip
// exact; appending `f` makes it an f32 literal in every form WGSL accepts
// (`2f`, `0.5f`, `1e-7f`). Non-finite values have no literal → unsupported.
export function f32lit(n: number): string {
  if (!Number.isFinite(n))
    throw new GpuUnsupportedError(`non-finite constant ${n}`);
  return `${String(n)}f`;
}

/** Shared WGSL helpers the generated channel expressions call into. */
export const WGSL_PRELUDE = `
fn s2l(v: f32) -> f32 {
  if (v <= 0.04045) { return v / 12.92; }
  return pow((v + 0.055) / 1.055, 2.4);
}
fn l2s(v: f32) -> f32 {
  if (v <= 0.0031308) { return v * 12.92; }
  return pow(v, 1.0 / 2.4) * 1.055 - 0.055;
}
fn smooth01(e0: f32, e1: f32, x: f32) -> f32 {
  let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}
// Mirrors utils Threshold: optional below/above clamps with a smooth transition.
fn thr(v: f32, hasB: bool, ba: f32, bt: f32, hasA: bool, aa: f32, at: f32, tw: f32) -> f32 {
  var out = v;
  if (hasA) {
    if (tw == 0.0) {
      if (v >= aa) { out = at; }
    } else {
      let t = smooth01(aa - tw, aa, v);
      out = (1.0 - t) * v + t * at;
    }
  }
  if (hasB) {
    if (tw == 0.0) {
      if (v <= ba) { out = bt; }
    } else {
      let t = smooth01(ba, ba + tw, v);
      out = (1.0 - t) * bt + t * v;
    }
  }
  return out;
}
`;

/**
 * Compile an expression to a WGSL f32 expression in terms of the coordinate
 * expressions `x`/`y` (themselves WGSL strings, so transforms substitute
 * coordinates by recompiling their child against rewritten ones). Named refs
 * resolve through their stored target. Throws {@link GpuUnsupportedError} for
 * any node outside the supported subset.
 */
export function compileExpr(e: Expression2D, x: string, y: string): string {
  const c = (child: Expression2D, xx = x, yy = y): string =>
    compileExpr(child, xx, yy);

  if (e instanceof Constant) return f32lit(e.literal);
  if (e instanceof X) return x;
  if (e instanceof Y) return y;
  if (e instanceof Ref) return c(e.refTarget);

  if (e instanceof AdditionExpression2D) {
    const [l, r] = e.children();
    return `(${c(l)} + ${c(r)})`;
  }
  if (e instanceof SubtractionExpression2D) {
    const [l, r] = e.children();
    return `(${c(l)} - ${c(r)})`;
  }
  if (e instanceof MultiplicationExpression2D) {
    const [l, r] = e.children();
    return `(${c(l)} * ${c(r)})`;
  }
  if (e instanceof DivisionExpression2D) {
    const [l, r] = e.children();
    return `(${c(l)} / ${c(r)})`;
  }
  if (e instanceof PowerExpression2D) {
    const [l, r] = e.children();
    return `pow(${c(l)}, ${c(r)})`;
  }
  if (e instanceof Min) {
    const [l, r] = e.children();
    return `min(${c(l)}, ${c(r)})`;
  }
  if (e instanceof Max) {
    const [l, r] = e.children();
    return `max(${c(l)}, ${c(r)})`;
  }

  if (e instanceof Sin) return `sin(${c(e.children()[0])})`;
  if (e instanceof Cos) return `cos(${c(e.children()[0])})`;
  if (e instanceof Abs) return `abs(${c(e.children()[0])})`;
  if (e instanceof Sqrt) return `sqrt(${c(e.children()[0])})`;
  if (e instanceof Floor) return `floor(${c(e.children()[0])})`;
  if (e instanceof Ceil) return `ceil(${c(e.children()[0])})`;
  if (e instanceof Log) {
    const a = c(e.children()[0]);
    if (e.logBase === Math.E) return `log(${a})`;
    if (e.logBase === 2) return `log2(${a})`;
    return `(log(${a}) / ${f32lit(Math.log(e.logBase))})`;
  }
  if (e instanceof sRGB2Linear) return `s2l(${c(e.children()[0])})`;
  if (e instanceof Linear2sRGB) return `l2s(${c(e.children()[0])})`;

  if (e instanceof Threshold) {
    const p = e.exportParams;
    const v = c(e.children()[0]);
    const hasB = p.below_at !== null;
    const hasA = p.above_at !== null;
    return (
      `thr(${v}, ${hasB}, ${f32lit(p.below_at ?? 0)}, ${f32lit(p.below_to ?? 0)}, ` +
      `${hasA}, ${f32lit(p.above_at ?? 0)}, ${f32lit(p.above_to ?? 0)}, ` +
      `${f32lit(p.transition_width)})`
    );
  }

  // Coordinate transforms: substitute the child's sample point. The GPU runs
  // per cell, so the non-separable Rotate is just another substitution.
  if (e instanceof Translate) {
    const p = e.exportParams;
    return c(
      e.children()[0],
      `(${x} - ${f32lit(p.dx)})`,
      `(${y} - ${f32lit(p.dy)})`,
    );
  }
  if (e instanceof Scale) {
    const p = e.exportParams;
    return c(
      e.children()[0],
      `(${x} / ${f32lit(p.sx)})`,
      `(${y} / ${f32lit(p.sy)})`,
    );
  }
  if (e instanceof Rotate) {
    const rad = (e.exportParams.degrees * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const xr = `(${f32lit(cos)} * ${x} + ${f32lit(sin)} * ${y})`;
    const yr = `(${f32lit(-sin)} * ${x} + ${f32lit(cos)} * ${y})`;
    return c(e.children()[0], xr, yr);
  }

  throw new GpuUnsupportedError(e.constructor.name);
}

/** Whether an expression tree compiles to WGSL (no unsupported nodes). */
export function exprSupported(e: Expression2D): boolean {
  try {
    compileExpr(e, "x", "y");
    return true;
  } catch (err) {
    if (err instanceof GpuUnsupportedError) return false;
    throw err;
  }
}
