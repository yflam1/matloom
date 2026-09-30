/**
 * registry.ts — the expression registry: a single ordered list of callable
 * functions and value constants that drives BOTH parsing (expr-lang/parser)
 * and the editor UI (completion dropdown + signature popup). Adding an entry
 * here is all it takes to expose a new function in the language and the UI.
 */
import {
  fBm,
  Worley,
  Bricks,
  Weave,
  Threshold,
  Constant,
  X,
  Y,
  Abs,
  Sqrt,
  Floor,
  Ceil,
  Log,
  Sin,
  Cos,
  Min,
  Max,
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Translate,
  Scale,
  Rotate,
  Rect,
  Ellipse,
  Fill,
  Stroke,
  Shape,
  Path,
  lineTo,
  cubicTo,
  type Segment,
  type Expression2D,
  type ThresholdOptions,
} from "../engine";

/** Any value an expression argument may hold after evaluation. */
export type ExprValue =
  | Expression2D
  | number
  | string
  | boolean
  | null
  | Shape
  | Segment;

/** One parameter of a registry function (drives the signature popup). */
export interface ParamSpec {
  name: string;
  type: string;
  required: boolean;
  default?: ExprValue;
  doc?: string;
  variadic?: boolean;
}

export interface RegistryEntry {
  name: string;
  kind: "expr" | "const";
  /** "call" → invoked as name(...); "value" → a bare constant like pi. */
  callForm: "call" | "value";
  summary: string;
  params?: ParamSpec[];
  /** Variadic entries (min/max) take positional numbers and carry no params. */
  variadic?: boolean;
  /**
   * A trailing rest parameter: positional args beyond `params` are collected
   * into an array bound to this name (used by Path's segment list).
   */
  restParam?: ParamSpec;
  // Builders are intentionally heterogeneous (expr nodes, numbers, …); the
  // parser passes a resolved-args object, or a positional array when variadic.
  build: (args: any) => ExprValue;
}

export const EXPR_REGISTRY: RegistryEntry[] = [
  // --- expression functions (kind: "expr", callForm: "call") ---
  {
    name: "fBm",
    kind: "expr",
    callForm: "call",
    summary: "Fractal Brownian motion (layered simplex noise)",
    params: [
      {
        name: "octaves",
        type: "int",
        required: false,
        default: 6,
        doc: "Number of simplex-noise layers to combine (1 = raw simplex)",
      },
      {
        name: "lacunarity",
        type: "float",
        required: false,
        default: 2.0,
        doc: "Frequency multiplier per octave",
      },
      {
        name: "gain",
        type: "float",
        required: false,
        default: 0.5,
        doc: "Amplitude multiplier per octave",
      },
      {
        name: "base_freq",
        type: "float",
        required: false,
        default: 1.0,
        doc: "Base frequency of the noise",
      },
      {
        name: "base_freq_x",
        type: "float?",
        required: false,
        default: null,
        doc: "Base frequency along X (falls back to base_freq)",
      },
      {
        name: "base_freq_y",
        type: "float?",
        required: false,
        default: null,
        doc: "Base frequency along Y (falls back to base_freq)",
      },
      {
        name: "to_01",
        type: "bool",
        required: false,
        default: false,
        doc: "Scale output to [0, 1] instead of [-1, 1]",
      },
      {
        name: "seed",
        type: "int?",
        required: false,
        default: null,
        doc: "Random seed (random when omitted)",
      },
    ],
    build(args) {
      return new fBm(args);
    },
  },
  {
    name: "Worley",
    kind: "expr",
    callForm: "call",
    summary: "Worley/cellular noise",
    params: [
      {
        name: "distance",
        type: 'union<"euclidean"|"manhattan"|"chebyshev">',
        required: false,
        default: "euclidean",
        doc: "Distance metric: euclidean→round cells, manhattan→diamonds, chebyshev→squares",
      },
      {
        name: "combination",
        type: 'union<"F1"|"F2"|"F2-F1"|"F2+F1">',
        required: false,
        default: "F1",
        doc: "How F1/F2 distances combine: F1 classic cells, F2 rounder regions, F2-F1 sharp ridges, F2+F1 blobby",
      },
      {
        name: "base_freq",
        type: "float",
        required: false,
        default: 1.0,
        doc: "Base frequency of the noise",
      },
      {
        name: "base_freq_x",
        type: "float?",
        required: false,
        default: null,
        doc: "Base frequency along X (falls back to base_freq)",
      },
      {
        name: "base_freq_y",
        type: "float?",
        required: false,
        default: null,
        doc: "Base frequency along Y (falls back to base_freq)",
      },
      {
        name: "to_01",
        type: "bool",
        required: false,
        default: false,
        doc: "Scale output to [0, 1] instead of [-1, 1]",
      },
      {
        name: "seed",
        type: "int?",
        required: false,
        default: null,
        doc: "Random seed (random when omitted)",
      },
    ],
    build(args) {
      return new Worley(args);
    },
  },
  {
    name: "Bricks",
    kind: "expr",
    callForm: "call",
    summary: "Brick lattice (1 = brick, 0 = mortar)",
    params: [
      {
        name: "brick_width",
        type: "float",
        required: false,
        default: 1.0,
        doc: "Cell pitch along X (> 0)",
      },
      {
        name: "brick_height",
        type: "float",
        required: false,
        default: 0.5,
        doc: "Cell pitch along Y (> 0)",
      },
      {
        name: "offset",
        type: "float",
        required: false,
        default: 0.5,
        doc: "Per-row/column shift as a fraction of a cell (0.5 = running bond, 0 = stacked)",
      },
      {
        name: "mortar",
        type: "float",
        required: false,
        default: 0.05,
        doc: "Gap width between bricks, in world units (>= 0)",
      },
      {
        name: "axis",
        type: 'union<"row"|"column">',
        required: false,
        default: "row",
        doc: "Whether successive rows or columns are offset",
      },
      {
        name: "feather",
        type: "float",
        required: false,
        default: 0.0,
        doc: "Soft-edge width in world units (0 = hard edge)",
      },
    ],
    build(args) {
      return new Bricks(args);
    },
  },
  {
    name: "Weave",
    kind: "expr",
    callForm: "call",
    summary: "Woven over/under thread lattice (1 = on-top thread, 0 = gap)",
    params: [
      {
        name: "over",
        type: "int",
        required: false,
        default: 1,
        doc: "Cells a thread floats over before going under (1/1 plain, 2/1 twill)",
      },
      {
        name: "under",
        type: "int",
        required: false,
        default: 1,
        doc: "Cells a thread floats under (>= 1)",
      },
      {
        name: "shift",
        type: "int",
        required: false,
        default: 1,
        doc: "Per-row diagonal step of the weave draft",
      },
      {
        name: "base_freq",
        type: "float",
        required: false,
        default: 8.0,
        doc: "Cells (thread crossings) per unit on both axes (> 0)",
      },
      {
        name: "base_freq_x",
        type: "float?",
        required: false,
        default: null,
        doc: "Cells per unit along X (falls back to base_freq)",
      },
      {
        name: "base_freq_y",
        type: "float?",
        required: false,
        default: null,
        doc: "Cells per unit along Y (falls back to base_freq)",
      },
      {
        name: "warp_width",
        type: "float",
        required: false,
        default: 0.5,
        doc: "Thread width as a fraction of the cell, in (0, 1]",
      },
      {
        name: "feather",
        type: "float",
        required: false,
        default: 0.0,
        doc: "Soft thread edge in cell-fraction units (0 = hard edge)",
      },
    ],
    build(args) {
      return new Weave(args);
    },
  },
  {
    name: "Threshold",
    kind: "expr",
    callForm: "call",
    summary: "Clamp/smoothstep values above/below cutoffs",
    params: [
      {
        name: "expr",
        type: "expr",
        required: true,
        doc: "Input expression to threshold",
      },
      {
        name: "below_at",
        type: "float?",
        required: false,
        default: null,
        doc: "Apply the lower threshold at this value (None disables it)",
      },
      {
        name: "below_to",
        type: "float?",
        required: false,
        default: null,
        doc: "Replacement value below below_at (defaults to below_at)",
      },
      {
        name: "above_at",
        type: "float?",
        required: false,
        default: null,
        doc: "Apply the upper threshold at this value (None disables it)",
      },
      {
        name: "above_to",
        type: "float?",
        required: false,
        default: null,
        doc: "Replacement value above above_at (defaults to above_at)",
      },
      {
        name: "transition_width",
        type: "float",
        required: false,
        default: 0.0,
        doc: "Smoothstep transition width (0 = hard threshold)",
      },
    ],
    build(args) {
      const { expr, ...opts } = args as {
        expr: Expression2D;
      } & ThresholdOptions;
      return new Threshold(expr, opts);
    },
  },
  {
    name: "Constant",
    kind: "expr",
    callForm: "call",
    summary: "Constant value",
    params: [
      {
        name: "value",
        type: "float",
        required: false,
        default: 0.0,
        doc: "The constant value",
      },
    ],
    build(args) {
      return "value" in args ? new Constant(args.value) : new Constant();
    },
  },
  {
    name: "X",
    kind: "expr",
    callForm: "call",
    summary: "X coordinate of the sample point",
    params: [],
    build() {
      return new X();
    },
  },
  {
    name: "Y",
    kind: "expr",
    callForm: "call",
    summary: "Y coordinate of the sample point",
    params: [],
    build() {
      return new Y();
    },
  },
  {
    name: "Abs",
    kind: "expr",
    callForm: "call",
    summary: "Absolute value of an expression",
    params: [
      { name: "expr", type: "expr", required: true, doc: "Input expression" },
    ],
    build(args) {
      return new Abs(args.expr);
    },
  },
  {
    name: "Sqrt",
    kind: "expr",
    callForm: "call",
    summary: "Square root of an expression",
    params: [
      { name: "expr", type: "expr", required: true, doc: "Input expression" },
    ],
    build(args) {
      return new Sqrt(args.expr);
    },
  },
  {
    name: "Floor",
    kind: "expr",
    callForm: "call",
    summary: "Largest integer ≤ an expression",
    params: [
      { name: "expr", type: "expr", required: true, doc: "Input expression" },
    ],
    build(args) {
      return new Floor(args.expr);
    },
  },
  {
    name: "Ceil",
    kind: "expr",
    callForm: "call",
    summary: "Smallest integer ≥ an expression",
    params: [
      { name: "expr", type: "expr", required: true, doc: "Input expression" },
    ],
    build(args) {
      return new Ceil(args.expr);
    },
  },
  {
    name: "Log",
    kind: "expr",
    callForm: "call",
    summary: "Logarithm of an expression with optional base",
    params: [
      {
        name: "expr",
        type: "expr",
        required: true,
        doc: "Input expression",
      },
      {
        name: "base",
        type: "float",
        required: false,
        default: Math.E,
        doc: "Logarithm base (e = natural log)",
      },
    ],
    build(args) {
      return new Log(args.expr, "base" in args ? args.base : Math.E);
    },
  },
  {
    name: "Sin",
    kind: "expr",
    callForm: "call",
    summary: "Sine of an expression (radians)",
    params: [
      { name: "expr", type: "expr", required: true, doc: "Input expression" },
    ],
    build(args) {
      return new Sin(args.expr);
    },
  },
  {
    name: "Cos",
    kind: "expr",
    callForm: "call",
    summary: "Cosine of an expression (radians)",
    params: [
      { name: "expr", type: "expr", required: true, doc: "Input expression" },
    ],
    build(args) {
      return new Cos(args.expr);
    },
  },
  {
    name: "Min",
    kind: "expr",
    callForm: "call",
    summary: "Element-wise minimum of two expressions",
    params: [
      { name: "expr1", type: "expr", required: true, doc: "First expression" },
      {
        name: "expr2",
        type: "expr",
        required: true,
        doc: "Second expression",
      },
    ],
    build(args) {
      return new Min(args.expr1, args.expr2);
    },
  },
  {
    name: "Max",
    kind: "expr",
    callForm: "call",
    summary: "Element-wise maximum of two expressions",
    params: [
      { name: "expr1", type: "expr", required: true, doc: "First expression" },
      {
        name: "expr2",
        type: "expr",
        required: true,
        doc: "Second expression",
      },
    ],
    build(args) {
      return new Max(args.expr1, args.expr2);
    },
  },
  {
    name: "Add",
    kind: "expr",
    callForm: "call",
    summary: "a + b",
    params: [
      { name: "left", type: "expr", required: true, doc: "Left operand" },
      { name: "right", type: "expr", required: true, doc: "Right operand" },
    ],
    build(args) {
      return Add(args.left, args.right);
    },
  },
  {
    name: "Sub",
    kind: "expr",
    callForm: "call",
    summary: "a − b",
    params: [
      { name: "minuend", type: "expr", required: true, doc: "Left operand" },
      {
        name: "subtrahend",
        type: "expr",
        required: true,
        doc: "Right operand",
      },
    ],
    build(args) {
      return Sub(args.minuend, args.subtrahend);
    },
  },
  {
    name: "Mul",
    kind: "expr",
    callForm: "call",
    summary: "a × b",
    params: [
      { name: "left", type: "expr", required: true, doc: "Left operand" },
      { name: "right", type: "expr", required: true, doc: "Right operand" },
    ],
    build(args) {
      return Mul(args.left, args.right);
    },
  },
  {
    name: "Div",
    kind: "expr",
    callForm: "call",
    summary: "a ÷ b",
    params: [
      { name: "dividend", type: "expr", required: true, doc: "Left operand" },
      { name: "divisor", type: "expr", required: true, doc: "Right operand" },
    ],
    build(args) {
      return Div(args.dividend, args.divisor);
    },
  },
  {
    name: "Pow",
    kind: "expr",
    callForm: "call",
    summary: "a ** b",
    params: [
      { name: "base", type: "expr", required: true, doc: "Base" },
      { name: "exponent", type: "expr", required: true, doc: "Exponent" },
    ],
    build(args) {
      return Pow(args.base, args.exponent);
    },
  },
  {
    name: "Translate",
    kind: "expr",
    callForm: "call",
    summary: "Translate an expression by (dx, dy)",
    params: [
      {
        name: "expr",
        type: "expr",
        required: true,
        doc: "Expression to translate",
      },
      {
        name: "dx",
        type: "float",
        required: false,
        default: 0.0,
        doc: "Shift along X",
      },
      {
        name: "dy",
        type: "float",
        required: false,
        default: 0.0,
        doc: "Shift along Y",
      },
    ],
    build(args) {
      return new Translate(args.expr, args.dx, args.dy);
    },
  },
  {
    name: "Scale",
    kind: "expr",
    callForm: "call",
    summary: "Scale an expression about the origin by (sx, sy)",
    params: [
      {
        name: "expr",
        type: "expr",
        required: true,
        doc: "Expression to scale",
      },
      {
        name: "sx",
        type: "float",
        required: false,
        default: 1.0,
        doc: "Scale factor along X (non-zero)",
      },
      {
        name: "sy",
        type: "float",
        required: false,
        default: 1.0,
        doc: "Scale factor along Y (non-zero)",
      },
    ],
    build(args) {
      return new Scale(args.expr, args.sx, args.sy);
    },
  },
  {
    name: "Rotate",
    kind: "expr",
    callForm: "call",
    summary: "Rotate an expression counter-clockwise about the origin",
    params: [
      {
        name: "expr",
        type: "expr",
        required: true,
        doc: "Expression to rotate",
      },
      {
        name: "degrees",
        type: "float",
        required: false,
        default: 0.0,
        doc: "Counter-clockwise rotation in degrees",
      },
    ],
    build(args) {
      return new Rotate(args.expr, args.degrees);
    },
  },
  {
    name: "Rect",
    kind: "expr",
    callForm: "call",
    summary: "Rectangle shape (use inside Fill/Stroke)",
    params: [
      { name: "x", type: "float", required: true, doc: "Bottom-left corner X" },
      { name: "y", type: "float", required: true, doc: "Bottom-left corner Y" },
      { name: "width", type: "float", required: true, doc: "Width (> 0)" },
      { name: "height", type: "float", required: true, doc: "Height (> 0)" },
      {
        name: "radius",
        type: "float",
        required: false,
        default: 0.0,
        doc: "Corner rounding radius (clamped to min(width, height) / 2)",
      },
    ],
    build(args) {
      return new Rect(args.x, args.y, args.width, args.height, args.radius);
    },
  },
  {
    name: "Ellipse",
    kind: "expr",
    callForm: "call",
    summary: "Ellipse shape (use inside Fill/Stroke)",
    params: [
      { name: "cx", type: "float", required: true, doc: "Center X" },
      { name: "cy", type: "float", required: true, doc: "Center Y" },
      {
        name: "rx",
        type: "float",
        required: true,
        doc: "Radius along X (> 0)",
      },
      {
        name: "ry",
        type: "float",
        required: true,
        doc: "Radius along Y (> 0)",
      },
    ],
    build(args) {
      return new Ellipse(args.cx, args.cy, args.rx, args.ry);
    },
  },
  {
    name: "Fill",
    kind: "expr",
    callForm: "call",
    summary: "Paint the interior of a shape (1 inside, 0 outside)",
    params: [
      { name: "shape", type: "shape", required: true, doc: "Shape to fill" },
      {
        name: "mode",
        type: 'union<"NonZero"|"EvenOdd">',
        required: false,
        default: "NonZero",
        doc: "Fill rule for self-intersecting paths",
      },
      {
        name: "feather",
        type: "float",
        required: false,
        default: 0.0,
        doc: "Soft-edge width in world units (0 = hard)",
      },
    ],
    build(args) {
      if (!(args.shape instanceof Shape))
        throw new SyntaxError(
          "Fill expects a shape (Rect/Ellipse/Path) as its first argument",
        );
      return new Fill(args.shape, args.mode, args.feather);
    },
  },
  {
    name: "Stroke",
    kind: "expr",
    callForm: "call",
    summary: "Paint a band along a shape's boundary",
    params: [
      { name: "shape", type: "shape", required: true, doc: "Shape to outline" },
      {
        name: "width",
        type: "float",
        required: true,
        doc: "Stroke width in world units (> 0)",
      },
      {
        name: "feather",
        type: "float",
        required: false,
        default: 0.0,
        doc: "Soft-edge width in world units (0 = hard)",
      },
    ],
    build(args) {
      if (!(args.shape instanceof Shape))
        throw new SyntaxError(
          "Stroke expects a shape (Rect/Ellipse/Path) as its first argument",
        );
      return new Stroke(args.shape, args.width, args.feather);
    },
  },
  {
    name: "Path",
    kind: "expr",
    callForm: "call",
    summary: "Path(startX, startY, LineTo/CubicTo, …) — wrap in Fill/Stroke",
    params: [
      { name: "startX", type: "float", required: true, doc: "Start point X" },
      { name: "startY", type: "float", required: true, doc: "Start point Y" },
    ],
    restParam: {
      name: "segments",
      type: "LineTo|CubicTo",
      required: true,
      doc: "One or more path segments — LineTo(x, y) or CubicTo(c1x, c1y, c2x, c2y, x, y)",
    },
    build(args) {
      const segments = (args.segments ?? []) as unknown[];
      for (const s of segments) {
        const seg = s as { kind?: string } | null;
        if (!seg || (seg.kind !== "line" && seg.kind !== "cubic"))
          throw new SyntaxError(
            "Path segments must be LineTo(...) or CubicTo(...)",
          );
      }
      return new Path(args.startX, args.startY, segments as Segment[]);
    },
  },
  {
    name: "LineTo",
    kind: "expr",
    callForm: "call",
    summary: "Straight path segment to (x, y)",
    params: [
      { name: "x", type: "float", required: true, doc: "End X" },
      { name: "y", type: "float", required: true, doc: "End Y" },
    ],
    build(args) {
      return lineTo(args.x, args.y);
    },
  },
  {
    name: "CubicTo",
    kind: "expr",
    callForm: "call",
    summary: "Cubic Bézier path segment",
    params: [
      { name: "c1x", type: "float", required: true, doc: "First control X" },
      { name: "c1y", type: "float", required: true, doc: "First control Y" },
      { name: "c2x", type: "float", required: true, doc: "Second control X" },
      { name: "c2y", type: "float", required: true, doc: "Second control Y" },
      { name: "x", type: "float", required: true, doc: "End X" },
      { name: "y", type: "float", required: true, doc: "End Y" },
    ],
    build(args) {
      return cubicTo(args.c1x, args.c1y, args.c2x, args.c2y, args.x, args.y);
    },
  },
  // --- const (kind: "const", callForm: "value") ---
  {
    name: "pi",
    kind: "const",
    callForm: "value",
    summary: "π (3.14159…)",
    build() {
      return Math.PI;
    },
  },
  {
    name: "e",
    kind: "const",
    callForm: "value",
    summary: "e (2.71828…)",
    build() {
      return Math.E;
    },
  },
];

/** name → entry lookup for fast resolution during parsing and caret analysis. */
export const REGISTRY_MAP = new Map(EXPR_REGISTRY.map((e) => [e.name, e]));
