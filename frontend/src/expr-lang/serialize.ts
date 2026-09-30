/**
 * serialize.ts — the inverse of parser.ts: turns an Expression2D tree back into
 * canonical DSL source. Used by the material import/export feature.
 *
 * The output is parseable by parseExpr() (and by the Python engine's parser),
 * so a material round-trips through the string format. Two deliberate choices:
 *   - `Constant`s become bare numbers (`127`, `0.5`), never `Constant(127)`.
 *   - auto-seeded noise has its current seed baked in (`fBm(..., seed=N)`), so a
 *     re-import reproduces the exact same noise rather than re-rolling.
 * Because it serializes the parsed tree, formatting is canonicalized: `X()*2`
 * comes back as `(X() * 2)`.
 */
import {
  Expression2D,
  Constant,
  X,
  Y,
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
  fBm,
  Worley,
  Bricks,
  Weave,
  Translate,
  Scale,
  Rotate,
  Rect,
  Ellipse,
  Fill,
  Stroke,
  Shape,
  Path,
  Ref,
} from "../engine";

/** Format a number as the parser would read it back: integers stay integral. */
function fmtNum(v: number): string {
  return String(v);
}

/** Append `kw=val` to `parts` only when `val` differs from the parser default. */
function pushIfSet(
  parts: string[],
  name: string,
  val: number,
  dflt: number,
): void {
  if (val !== dflt) parts.push(`${name}=${fmtNum(val)}`);
}

// Set-driven base_freq handling for noise: emit each frequency arg by name iff
// the author wrote it. `base_freq` (the both-axes default) is emitted from the
// axis NOT overridden by an explicit base_freq_x/y; when both axes are
// explicit, base_freq has no effect and is omitted. Mirrors the Python engine's
// `_freq_parts` set-driven path.
function freqParts(set: Set<string>, bfx: number, bfy: number): string[] {
  const parts: string[] = [];
  if (set.has("base_freq")) {
    if (!set.has("base_freq_x")) parts.push(`base_freq=${fmtNum(bfx)}`);
    else if (!set.has("base_freq_y")) parts.push(`base_freq=${fmtNum(bfy)}`);
  }
  if (set.has("base_freq_x")) parts.push(`base_freq_x=${fmtNum(bfx)}`);
  if (set.has("base_freq_y")) parts.push(`base_freq_y=${fmtNum(bfy)}`);
  return parts;
}

/** Serialize a Shape (the geometry inside Fill/Stroke). */
function serializeShape(s: Shape): string {
  if (s instanceof Rect) {
    const p = s.exportParams;
    const parts = [fmtNum(p.x), fmtNum(p.y), fmtNum(p.width), fmtNum(p.height)];
    if (p.radius !== 0) parts.push(fmtNum(p.radius));
    return `Rect(${parts.join(", ")})`;
  }
  if (s instanceof Ellipse) {
    const p = s.exportParams;
    return `Ellipse(${fmtNum(p.cx)}, ${fmtNum(p.cy)}, ${fmtNum(p.rx)}, ${fmtNum(p.ry)})`;
  }
  if (s instanceof Path) {
    const p = s.exportParams;
    const parts = [fmtNum(p.startX), fmtNum(p.startY)];
    for (const seg of p.segments) {
      if (seg.kind === "line") {
        parts.push(`LineTo(${fmtNum(seg.x)}, ${fmtNum(seg.y)})`);
      } else {
        parts.push(
          `CubicTo(${fmtNum(seg.c1x)}, ${fmtNum(seg.c1y)}, ${fmtNum(seg.c2x)}, ${fmtNum(seg.c2y)}, ${fmtNum(seg.x)}, ${fmtNum(seg.y)})`,
        );
      }
    }
    return `Path(${parts.join(", ")})`;
  }
  throw new Error(`cannot serialize shape: ${s.constructor.name}`);
}

/** Serialize an Expression2D tree to canonical DSL source. */
export function serializeExpr(e: Expression2D): string {
  if (e instanceof Ref) return e.name;
  if (e instanceof Constant) return fmtNum(e.literal);
  if (e instanceof X) return "X()";
  if (e instanceof Y) return "Y()";

  // Unary minus is parsed as `(-1 * operand)`; render it back as `-operand`
  // (parenthesizing an already-negative operand so `--x` never appears).
  if (e instanceof MultiplicationExpression2D) {
    const [l, r] = e.children();
    if (l instanceof Constant && l.literal === -1) {
      const inner = serializeExpr(r);
      return inner.startsWith("-") ? `-(${inner})` : `-${inner}`;
    }
    return `(${serializeExpr(l)} * ${serializeExpr(r)})`;
  }
  if (e instanceof AdditionExpression2D) {
    const [l, r] = e.children();
    return `(${serializeExpr(l)} + ${serializeExpr(r)})`;
  }
  if (e instanceof SubtractionExpression2D) {
    const [l, r] = e.children();
    return `(${serializeExpr(l)} - ${serializeExpr(r)})`;
  }
  if (e instanceof DivisionExpression2D) {
    const [l, r] = e.children();
    return `(${serializeExpr(l)} / ${serializeExpr(r)})`;
  }
  if (e instanceof PowerExpression2D) {
    const [l, r] = e.children();
    // The parser binds `**` tighter than a leading unary minus, so a base that
    // renders with a leading '-' (a negative Constant, or the `-x` unary-minus
    // form) would re-parse as `-(base ** exp)` — a different value. Parenthesize
    // it so `(-3 ** e)` round-trips as `((-3) ** e)`.
    let base = serializeExpr(l);
    if (base.startsWith("-")) base = `(${base})`;
    return `(${base} ** ${serializeExpr(r)})`;
  }
  if (e instanceof Min) {
    const [l, r] = e.children();
    return `Min(${serializeExpr(l)}, ${serializeExpr(r)})`;
  }
  if (e instanceof Max) {
    const [l, r] = e.children();
    return `Max(${serializeExpr(l)}, ${serializeExpr(r)})`;
  }

  if (e instanceof Sin) return `Sin(${serializeExpr(e.children()[0])})`;
  if (e instanceof Cos) return `Cos(${serializeExpr(e.children()[0])})`;
  if (e instanceof Abs) return `Abs(${serializeExpr(e.children()[0])})`;
  if (e instanceof Sqrt) return `Sqrt(${serializeExpr(e.children()[0])})`;
  if (e instanceof Floor) return `Floor(${serializeExpr(e.children()[0])})`;
  if (e instanceof Ceil) return `Ceil(${serializeExpr(e.children()[0])})`;
  if (e instanceof sRGB2Linear)
    return `sRGB2Linear(${serializeExpr(e.children()[0])})`;
  if (e instanceof Linear2sRGB)
    return `Linear2sRGB(${serializeExpr(e.children()[0])})`;
  if (e instanceof Log) {
    const arg = serializeExpr(e.children()[0]);
    return e.logBase === Math.E
      ? `Log(${arg})`
      : `Log(${arg}, ${fmtNum(e.logBase)})`;
  }

  if (e instanceof Threshold) {
    const p = e.exportParams;
    const parts = [serializeExpr(e.children()[0])];
    if (p.below_at !== null) {
      parts.push(`below_at=${fmtNum(p.below_at)}`);
      if (p.below_to !== null && p.below_to !== p.below_at)
        parts.push(`below_to=${fmtNum(p.below_to)}`);
    }
    if (p.above_at !== null) {
      parts.push(`above_at=${fmtNum(p.above_at)}`);
      if (p.above_to !== null && p.above_to !== p.above_at)
        parts.push(`above_to=${fmtNum(p.above_to)}`);
    }
    pushIfSet(parts, "transition_width", p.transition_width, 0);
    return `Threshold(${parts.join(", ")})`;
  }

  if (e instanceof fBm) {
    const p = e.exportParams;
    const s = e.setArgs;
    const parts: string[] = [];
    if (s.has("octaves")) parts.push(`octaves=${fmtNum(p.octaves)}`);
    if (s.has("lacunarity")) parts.push(`lacunarity=${fmtNum(p.lacunarity)}`);
    if (s.has("gain")) parts.push(`gain=${fmtNum(p.gain)}`);
    parts.push(...freqParts(s, p.base_freq_x, p.base_freq_y));
    if (s.has("to_01")) parts.push(`to_01=${p.to_01 ? "True" : "False"}`);
    parts.push(`seed=${fmtNum(e.currentSeed)}`);
    return `fBm(${parts.join(", ")})`;
  }

  if (e instanceof Worley) {
    const p = e.exportParams;
    const s = e.setArgs;
    const parts: string[] = [];
    if (s.has("distance")) parts.push(`distance="${p.distance}"`);
    if (s.has("combination")) parts.push(`combination="${p.combination}"`);
    parts.push(...freqParts(s, p.base_freq_x, p.base_freq_y));
    if (s.has("to_01")) parts.push(`to_01=${p.to_01 ? "True" : "False"}`);
    parts.push(`seed=${fmtNum(e.currentSeed)}`);
    return `Worley(${parts.join(", ")})`;
  }

  if (e instanceof Translate) {
    const [child] = e.children();
    const p = e.exportParams;
    return `Translate(${serializeExpr(child)}, ${fmtNum(p.dx)}, ${fmtNum(p.dy)})`;
  }

  if (e instanceof Scale) {
    const [child] = e.children();
    const p = e.exportParams;
    return `Scale(${serializeExpr(child)}, ${fmtNum(p.sx)}, ${fmtNum(p.sy)})`;
  }

  if (e instanceof Rotate) {
    const [child] = e.children();
    const p = e.exportParams;
    return `Rotate(${serializeExpr(child)}, ${fmtNum(p.degrees)})`;
  }

  if (e instanceof Bricks) {
    const p = e.exportParams;
    const parts: string[] = [];
    pushIfSet(parts, "brick_width", p.brick_width, 1.0);
    pushIfSet(parts, "brick_height", p.brick_height, 0.5);
    pushIfSet(parts, "offset", p.offset, 0.5);
    pushIfSet(parts, "mortar", p.mortar, 0.05);
    if (p.axis !== "row") parts.push(`axis="${p.axis}"`);
    pushIfSet(parts, "feather", p.feather, 0.0);
    return `Bricks(${parts.join(", ")})`;
  }

  if (e instanceof Weave) {
    const p = e.exportParams;
    const parts: string[] = [];
    pushIfSet(parts, "over", p.over, 1);
    pushIfSet(parts, "under", p.under, 1);
    pushIfSet(parts, "shift", p.shift, 1);
    // Weave's base frequency defaults to 8 (not 1 like the noises), so it has
    // its own omit-when-default handling rather than reusing freqParts.
    if (p.base_freq_x === p.base_freq_y) {
      if (p.base_freq_x !== 8.0)
        parts.push(`base_freq=${fmtNum(p.base_freq_x)}`);
    } else {
      parts.push(`base_freq_x=${fmtNum(p.base_freq_x)}`);
      parts.push(`base_freq_y=${fmtNum(p.base_freq_y)}`);
    }
    pushIfSet(parts, "warp_width", p.warp_width, 0.5);
    pushIfSet(parts, "feather", p.feather, 0.0);
    return `Weave(${parts.join(", ")})`;
  }

  if (e instanceof Fill) {
    const p = e.exportParams;
    const parts = [serializeShape(e.shapeRef)];
    if (p.mode !== "NonZero") parts.push(`mode="${p.mode}"`);
    if (p.feather !== 0) parts.push(`feather=${fmtNum(p.feather)}`);
    return `Fill(${parts.join(", ")})`;
  }

  if (e instanceof Stroke) {
    const p = e.exportParams;
    const parts = [serializeShape(e.shapeRef), fmtNum(p.width)];
    if (p.feather !== 0) parts.push(`feather=${fmtNum(p.feather)}`);
    return `Stroke(${parts.join(", ")})`;
  }

  throw new Error(`cannot serialize expression: ${e.constructor.name}`);
}
