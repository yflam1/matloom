/**
 * generate.ts — a grammar-free random generator for layered-material programs.
 *
 * The TypeScript mirror of `matloom/engine/generate.py`. It samples whole
 * materials by building real engine objects (the same constructors the parser
 * targets) and serializing them with `serializeExpr`, so every generated program
 * is valid by construction and round-trips through `parseImportedLayers` /
 * `serializeMaterial`.
 *
 * Output format (identical to the canonical material string):
 *
 *   View(<x1>, <y1>, <x2>, <y2>)
 *   Define(def1, <expr>)            // zero or more, def1, def2, …
 *   Material(
 *     Layer(<alpha>)               // bottom-most first
 *       .basecolor(<r>, <g>, <b>)
 *       .metallic(<m>) .roughness(<r>) .emissive(<r>, <g>, <b>, <s>)
 *       .height(<h>),
 *     …
 *   )
 *
 * Generation order matches the spec: (1) layer count, (2) View region, (3) the
 * Define preamble (each may reference earlier defines), (4) the layer stack.
 * Every channel slot is an independently sampled expression drawn recursively
 * from the full node pool — leaves (Constant/X/Y, plus Refs to earlier defines),
 * the noise/pattern/shape generators (fBm/Worley/Bricks/Fill/Stroke), the
 * unary/binary math nodes, the transforms (Translate/Scale/Rotate) and
 * Threshold. Numeric arguments are sampled type-faithfully over configurable
 * bounded ranges (the true (-inf, inf) / (0, inf) domains are approximated, as
 * an unbounded uniform is not sampleable).
 *
 * Reproducibility: pass `seed` and a run is fully deterministic. All randomness
 * flows through a single seeded PRNG (mulberry32), so noise seeds are drawn from
 * it too rather than from Math.random.
 */
import {
  Abs,
  Bricks,
  Weave,
  Ceil,
  Constant,
  Cos,
  Div,
  DivisionExpression2D,
  Ellipse,
  type Expression2D,
  Fill,
  Floor,
  Log,
  Max,
  Min,
  Path,
  Pow,
  Rect,
  Ref,
  Rotate,
  Scale,
  type Segment,
  type Shape,
  Sin,
  Sqrt,
  Stroke,
  Threshold,
  Translate,
  Worley,
  X,
  Y,
  fBm,
  Add,
  Sub,
  Mul,
  cubicTo,
  lineTo,
} from "./engine";
import { serializeExpr } from "./expr-lang";

const MAX_SEED = 2 ** 31 - 1;

const WORLEY_DISTANCES = ["euclidean", "manhattan", "chebyshev"] as const;
const WORLEY_COMBINATIONS = ["F1", "F2", "F2-F1", "F2+F1"] as const;
const BRICK_AXES = ["row", "column"] as const;
const FILL_MODES = ["NonZero", "EvenOdd"] as const;

/** Knobs controlling the random material generator (see Python GeneratorConfig). */
export interface GeneratorConfig {
  /** Layer count is drawn from [minLayers, maxLayers] (both positive). */
  minLayers: number;
  maxLayers: number;
  /** Exact layer count (overrides the range). */
  numLayers: number | null;
  /** Define count is drawn from [minDefs, maxDefs] (>= 0). */
  minDefs: number;
  maxDefs: number;
  /** Exact Define count (overrides the range). */
  numDefs: number | null;
  /** Maximum expression nesting depth (>= 0). */
  maxDepth: number;
  /** Probability in [0, 1] of stopping at a non-recursive node early. */
  leafBias: number;
  /** Force the bottom-most layer's alpha to the constant 1. */
  opaqueBase: boolean;
  /** Randomize the View region; when false it is pinned to (0, 0, 1, 1). */
  randomView: boolean;
  /** Bounds for `real` parameters. */
  realLo: number;
  realHi: number;
  /** Bounds for `pos_real` (strictly > 0). */
  posRealMin: number;
  posRealMax: number;
  /** Upper bound for `non_neg_real` (lower bound 0). */
  nonNegRealMax: number;
  /** Upper bound for fBm octaves (lower bound 1). */
  maxOctaves: number;
  /** Upper bound for the number of Path segments (lower bound 1). */
  maxPathSegments: number;
}

export const DEFAULT_CONFIG: GeneratorConfig = {
  minLayers: 1,
  maxLayers: 3,
  numLayers: null,
  minDefs: 0,
  maxDefs: 2,
  numDefs: null,
  maxDepth: 6,
  leafBias: 0.5,
  opaqueBase: true,
  randomView: true,
  realLo: -10.0,
  realHi: 10.0,
  posRealMin: 1e-3,
  posRealMax: 10.0,
  nonNegRealMax: 10.0,
  maxOctaves: 8,
  maxPathSegments: 4,
};

function validateConfig(c: GeneratorConfig): void {
  if (c.numLayers !== null) {
    if (c.numLayers < 1)
      throw new Error("numLayers must be a positive integer");
  } else {
    if (c.minLayers < 1 || c.maxLayers < 1)
      throw new Error("minLayers and maxLayers must be positive");
    if (c.minLayers > c.maxLayers)
      throw new Error("minLayers must be <= maxLayers");
  }
  if (c.numDefs !== null) {
    if (c.numDefs < 0) throw new Error("numDefs must be >= 0");
  } else {
    if (c.minDefs < 0 || c.maxDefs < 0)
      throw new Error("minDefs and maxDefs must be >= 0");
    if (c.minDefs > c.maxDefs) throw new Error("minDefs must be <= maxDefs");
  }
  if (c.maxDepth < 0) throw new Error("maxDepth must be >= 0");
  if (!(c.leafBias >= 0 && c.leafBias <= 1))
    throw new Error("leafBias must be in [0, 1]");
  if (c.realLo > c.realHi) throw new Error("realLo must be <= realHi");
  if (!(c.posRealMin > 0 && c.posRealMin <= c.posRealMax))
    throw new Error("require 0 < posRealMin <= posRealMax");
  if (c.nonNegRealMax < 0) throw new Error("nonNegRealMax must be >= 0");
  if (c.maxOctaves < 1) throw new Error("maxOctaves must be >= 1");
  if (c.maxPathSegments < 1) throw new Error("maxPathSegments must be >= 1");
}

/** A deterministic, seedable PRNG (mulberry32) returning floats in [0, 1). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** One generated layer: every channel slot is an independent expression. */
export interface GeneratedLayer {
  alpha: Expression2D;
  basecolor: [Expression2D, Expression2D, Expression2D];
  metallic: Expression2D;
  roughness: Expression2D;
  sheen: Expression2D;
  coat: Expression2D;
  transmission: Expression2D;
  ior: Expression2D;
  subsurface: Expression2D;
  anisotropy: Expression2D;
  emissive: [Expression2D, Expression2D, Expression2D, Expression2D];
  height: Expression2D;
}

export interface GeneratedDef {
  name: string;
  expr: Expression2D;
}

/** Match the parser's number formatting (integers stay integral). */
function fmtNum(v: number): string {
  return String(v);
}

/**
 * Serialize a color channel into the 0–255 convention, mirroring the Python
 * engine's `_color_to_format`: a `<expr> / 255` shape unwraps to `<expr>`,
 * anything else is scaled up by 255.
 */
function colorToFormat(raw: Expression2D): string {
  if (raw instanceof DivisionExpression2D) {
    const [dividend, divisor] = raw.children();
    if (divisor instanceof Constant && divisor.literal === 255)
      return serializeExpr(dividend);
  }
  return `(${serializeExpr(raw)} * 255)`;
}

function indentLines(text: string, pad: string): string {
  return text
    .split("\n")
    .map((line) => pad + line)
    .join("\n");
}

function serializeLayer(layer: GeneratedLayer): string {
  const [r, g, b] = layer.basecolor;
  const [er, eg, eb, strength] = layer.emissive;
  return [
    `Layer(${serializeExpr(layer.alpha)})`,
    `  .basecolor(${colorToFormat(r)}, ${colorToFormat(g)}, ${colorToFormat(b)})`,
    `  .metallic(${serializeExpr(layer.metallic)})`,
    `  .roughness(${serializeExpr(layer.roughness)})`,
    `  .sheen(${serializeExpr(layer.sheen)})`,
    `  .coat(${serializeExpr(layer.coat)})`,
    `  .transmission(${serializeExpr(layer.transmission)})`,
    `  .ior(${serializeExpr(layer.ior)})`,
    `  .subsurface(${serializeExpr(layer.subsurface)})`,
    `  .anisotropy(${serializeExpr(layer.anisotropy)})`,
    `  .emissive(${colorToFormat(er)}, ${colorToFormat(eg)}, ${colorToFormat(eb)}, ${serializeExpr(strength)})`,
    `  .height(${serializeExpr(layer.height)})`,
  ].join("\n");
}

/**
 * A generated material: the View region, the ordered Define table, and the
 * layer stack (index 0 is the bottom-most layer). `serialize()` emits the
 * canonical DSL string.
 */
export class GeneratedMaterial {
  constructor(
    readonly view: [number, number, number, number],
    readonly defs: GeneratedDef[],
    readonly layers: GeneratedLayer[],
  ) {}

  serialize(): string {
    const [x1, y1, x2, y2] = this.view;
    const lines: string[] = [
      `View(${fmtNum(x1)}, ${fmtNum(y1)}, ${fmtNum(x2)}, ${fmtNum(y2)})`,
    ];
    if (this.defs.length)
      lines.push(
        this.defs
          .map((d) => `Define(${d.name}, ${serializeExpr(d.expr)})`)
          .join("\n"),
      );
    const body =
      this.layers.length === 0
        ? "Material()"
        : `Material(\n${this.layers
            .map((l) => indentLines(serializeLayer(l), "  "))
            .join(",\n")}\n)`;
    lines.push(body);
    return lines.join("\n");
  }
}

/**
 * Stateful random generator. One instance owns a single seeded PRNG and the
 * running table of Defines so that references resolve only to earlier names.
 */
export class MaterialGenerator {
  private readonly cfg: GeneratorConfig;
  private readonly rng: () => number;
  private env: GeneratedDef[] = [];

  constructor(config: Partial<GeneratorConfig> = {}, seed?: number) {
    this.cfg = { ...DEFAULT_CONFIG, ...config };
    validateConfig(this.cfg);
    const s = seed ?? Math.floor(Math.random() * 2 ** 31);
    this.rng = mulberry32(s);
  }

  // --- numeric sampling helpers (type-faithful, bounded) ---

  real(): number {
    return this.cfg.realLo + this.rng() * (this.cfg.realHi - this.cfg.realLo);
  }

  posReal(): number {
    return (
      this.cfg.posRealMin +
      this.rng() * (this.cfg.posRealMax - this.cfg.posRealMin)
    );
  }

  nonNegReal(): number {
    return this.rng() * this.cfg.nonNegRealMax;
  }

  nonzeroReal(): number {
    let v = this.real();
    while (v === 0) v = this.real();
    return v;
  }

  zeroToOne(): number {
    return this.rng();
  }

  degrees(): number {
    return this.rng() * 360;
  }

  byte(): number {
    return Math.floor(this.rng() * 256);
  }

  seed(): number {
    return Math.floor(this.rng() * (MAX_SEED + 1));
  }

  private chance(p: number): boolean {
    return this.rng() < p;
  }

  private pick<T>(items: readonly T[]): T {
    return items[Math.floor(this.rng() * items.length)];
  }

  private randint(lo: number, hi: number): number {
    return lo + Math.floor(this.rng() * (hi - lo + 1));
  }

  // --- expression tree ---

  expression(depth: number = this.cfg.maxDepth): Expression2D {
    if (depth <= 0 || this.chance(this.cfg.leafBias)) return this.terminal();
    return this.pick(this.recursive)(depth);
  }

  private terminal(): Expression2D {
    const choices = [...this.terminals];
    if (this.env.length) choices.push(() => this.genRef());
    return this.pick(choices)();
  }

  private get terminals(): Array<() => Expression2D> {
    return [
      () => new Constant(this.real()),
      () => new X(),
      () => new Y(),
      () => this.genBricks(),
      () => this.genWeave(),
      () => this.genFbm(),
      () => this.genWorley(),
      () => this.genFill(),
      () => this.genStroke(),
    ];
  }

  private get recursive(): Array<(depth: number) => Expression2D> {
    return [
      (d) => this.genUnary(d),
      (d) => this.genLog(d),
      (d) => this.genBinary(d),
      (d) => new Translate(this.expression(d - 1), this.real(), this.real()),
      (d) =>
        new Scale(
          this.expression(d - 1),
          this.nonzeroReal(),
          this.nonzeroReal(),
        ),
      (d) => new Rotate(this.expression(d - 1), this.degrees()),
      (d) => this.genThreshold(d),
    ];
  }

  private genRef(): Expression2D {
    const { name, expr } = this.pick(this.env);
    return new Ref(name, expr);
  }

  private genBricks(): Expression2D {
    return new Bricks({
      brick_width: this.posReal(),
      brick_height: this.posReal(),
      offset: this.real(),
      mortar: this.nonNegReal(),
      axis: this.pick(BRICK_AXES),
      feather: this.chance(0.5) ? this.nonNegReal() : 0,
    });
  }

  private genWeave(): Expression2D {
    // Weave frequencies must be strictly positive (unlike noise), so sample
    // from the positive range rather than the signed `freq()` helper.
    const freq = this.chance(0.5)
      ? { base_freq: this.posReal() }
      : { base_freq_x: this.posReal(), base_freq_y: this.posReal() };
    return new Weave({
      over: this.randint(1, 4),
      under: this.randint(1, 4),
      shift: this.randint(0, 3),
      warp_width: 0.05 + this.rng() * (1.0 - 0.05),
      feather: this.chance(0.5) ? this.nonNegReal() : 0,
      ...freq,
    });
  }

  private genFbm(): Expression2D {
    return new fBm({
      octaves: this.randint(1, this.cfg.maxOctaves),
      lacunarity: this.posReal(),
      gain: this.posReal(),
      to_01: this.chance(0.5),
      seed: this.seed(),
      ...this.freq(),
    });
  }

  private genWorley(): Expression2D {
    return new Worley({
      distance: this.pick(WORLEY_DISTANCES),
      combination: this.pick(WORLEY_COMBINATIONS),
      to_01: this.chance(0.5),
      seed: this.seed(),
      ...this.freq(),
    });
  }

  /** Either a shared base_freq or per-axis frequencies. */
  private freq(): {
    base_freq?: number;
    base_freq_x?: number;
    base_freq_y?: number;
  } {
    return this.chance(0.5)
      ? { base_freq: this.nonzeroReal() }
      : { base_freq_x: this.nonzeroReal(), base_freq_y: this.nonzeroReal() };
  }

  private genFill(): Expression2D {
    return new Fill(
      this.shape(),
      this.pick(FILL_MODES),
      this.chance(0.5) ? this.nonNegReal() : 0,
    );
  }

  private genStroke(): Expression2D {
    return new Stroke(
      this.shape(),
      this.posReal(),
      this.chance(0.5) ? this.nonNegReal() : 0,
    );
  }

  private genUnary(depth: number): Expression2D {
    const node = this.pick([Abs, Sqrt, Floor, Ceil, Sin, Cos]);
    return new node(this.expression(depth - 1));
  }

  private genLog(depth: number): Expression2D {
    let base = this.posReal();
    if (Math.abs(base - 1) < 1e-9) base += 0.5; // log base must be > 0 and != 1
    return new Log(this.expression(depth - 1), base);
  }

  private genBinary(depth: number): Expression2D {
    const factories: Array<(a: Expression2D, b: Expression2D) => Expression2D> =
      [
        (a, b) => Add(a, b),
        (a, b) => Sub(a, b),
        (a, b) => Mul(a, b),
        (a, b) => Div(a, b),
        (a, b) => Pow(a, b),
        (a, b) => new Min(a, b),
        (a, b) => new Max(a, b),
      ];
    return this.pick(factories)(
      this.expression(depth - 1),
      this.expression(depth - 1),
    );
  }

  private genThreshold(depth: number): Expression2D {
    const opts: {
      below_at?: number;
      below_to?: number;
      above_at?: number;
      above_to?: number;
      transition_width?: number;
    } = {};
    if (this.chance(0.6)) {
      opts.below_at = this.real();
      if (this.chance(0.5)) opts.below_to = this.real();
    }
    if (this.chance(0.6)) {
      opts.above_at = this.real();
      if (this.chance(0.5)) opts.above_to = this.real();
    }
    if (this.chance(0.5)) opts.transition_width = this.nonNegReal();
    return new Threshold(this.expression(depth - 1), opts);
  }

  // --- shapes (geometry inside Fill/Stroke) ---

  private shape(): Shape {
    return this.pick([
      () => this.rect(),
      () => this.ellipse(),
      () => this.path(),
    ])();
  }

  private rect(): Shape {
    const radius = this.chance(0.5) ? this.nonNegReal() : 0;
    return new Rect(
      this.real(),
      this.real(),
      this.posReal(),
      this.posReal(),
      radius,
    );
  }

  private ellipse(): Shape {
    return new Ellipse(
      this.real(),
      this.real(),
      this.posReal(),
      this.posReal(),
    );
  }

  private path(): Shape {
    const n = this.randint(1, this.cfg.maxPathSegments);
    const segments: Segment[] = [];
    for (let i = 0; i < n; i++) segments.push(this.segment());
    return new Path(this.real(), this.real(), segments);
  }

  private segment(): Segment {
    if (this.chance(0.5)) return lineTo(this.real(), this.real());
    return cubicTo(
      this.real(),
      this.real(),
      this.real(),
      this.real(),
      this.real(),
      this.real(),
    );
  }

  // --- material assembly ---

  private channel(): Expression2D {
    return this.expression(this.cfg.maxDepth);
  }

  private layer(opaque: boolean): GeneratedLayer {
    return {
      alpha: opaque ? new Constant(1) : this.channel(),
      basecolor: [this.channel(), this.channel(), this.channel()],
      metallic: this.channel(),
      roughness: this.channel(),
      sheen: this.channel(),
      coat: this.channel(),
      transmission: this.channel(),
      ior: this.channel(),
      subsurface: this.channel(),
      anisotropy: this.channel(),
      emissive: [
        this.channel(),
        this.channel(),
        this.channel(),
        this.channel(),
      ],
      height: this.channel(),
    };
  }

  private resolveCount(exact: number | null, lo: number, hi: number): number {
    return exact ?? this.randint(lo, hi);
  }

  generate(): GeneratedMaterial {
    const cfg = this.cfg;
    this.env = [];

    // (1) number of layers.
    const nLayers = this.resolveCount(
      cfg.numLayers,
      cfg.minLayers,
      cfg.maxLayers,
    );

    // (2) View region: a non-degenerate rectangle (x1 < x2, y1 < y2), or the
    // unit square (0, 0, 1, 1) when randomView is disabled.
    let view: [number, number, number, number];
    if (cfg.randomView) {
      const x1 = this.real();
      const y1 = this.real();
      view = [x1, y1, x1 + this.posReal(), y1 + this.posReal()];
    } else {
      view = [0, 0, 1, 1];
    }

    // (3) Define preamble; each may reference earlier defines.
    const nDefs = this.resolveCount(cfg.numDefs, cfg.minDefs, cfg.maxDefs);
    const defs: GeneratedDef[] = [];
    for (let i = 1; i <= nDefs; i++) {
      const name = `def${i}`;
      const expr = this.expression(cfg.maxDepth);
      defs.push({ name, expr });
      this.env.push({ name, expr });
    }

    // (4) layer stack; index 0 is the bottom-most layer.
    const layers: GeneratedLayer[] = [];
    for (let idx = 0; idx < nLayers; idx++) {
      layers.push(this.layer(cfg.opaqueBase && idx === 0));
    }

    return new GeneratedMaterial(view, defs, layers);
  }
}

/**
 * Generate a random material program.
 *
 * @param config generation knobs (merged over DEFAULT_CONFIG).
 * @param opts.seed optional seed for reproducible output.
 * @param opts.asString when true, return the serialized canonical DSL string;
 *   otherwise return the GeneratedMaterial object (the default).
 */
export function generateMaterial(
  config: Partial<GeneratorConfig> = {},
  opts: { seed?: number; asString?: boolean } = {},
): GeneratedMaterial | string {
  const material = new MaterialGenerator(config, opts.seed).generate();
  return opts.asString ? material.serialize() : material;
}
