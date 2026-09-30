/**
 * material-io.ts — import/export of the layer stack to the textual material
 * format (pretty-printed with a 2-space indent):
 *
 *   Material(
 *     Layer(<alpha>)
 *       .basecolor(<r>, <g>, <b>)
 *       .metallic(<m>)
 *       .roughness(<rough>)
 *       .emissive(<r>, <g>, <b>, <strength>),
 *     …
 *   )
 *
 * The first Layer in the string is the bottom-most ("Layer 1"); the store keeps
 * layers top-most first, so the order is reversed on the way in and out. Color
 * channels (basecolor/emissive RGB) are expressed in the 0–255 convention;
 * alpha/metallic/roughness/strength are in the engine's native domain. Only
 * visible layers are exported, and auto-noise seeds are baked in so a re-import
 * reproduces the exact same noise.
 *
 * This module owns BOTH directions and mirrors matloom/engine's serialize()/
 * deserialize(): a string produced here is parseable by the Python engine and
 * vice versa.
 */
import { collectNoise, type Expression2D } from "../engine";
import { parseExpr, serializeExpr } from "../expr-lang";
import {
  defaultChannels,
  uid,
  type ChannelKey,
  type ColorMode,
  type LayerState,
  type MaterialDef,
  type Store,
} from "./state";

/** The viewing window carried by a `View(...)` statement. */
export interface ImportedView {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/** Serialize the visible layers (and View / Define preamble) to the material string. */
export function serializeMaterial(store: Store): string {
  // Resolve definitions first, in order, so later defs may reference earlier
  // ones and channels can reference any of them.
  const env = new Map<string, Expression2D>();
  const defLines: string[] = [];
  for (const d of store.defs) {
    const expr = parseExpr(d.src, env);
    defLines.push(`Define(${d.name}, ${serializeExpr(expr)})`);
    env.set(d.name, expr);
  }
  // Store is top-most first; the format is bottom-most first.
  const bottomFirst = store.layers.filter((l) => l.enabled).reverse();
  const body =
    bottomFirst.length === 0
      ? "Material()"
      : `Material(\n${bottomFirst
          .map((l) => indentLines(serializeLayer(l, env), "  "))
          .join(",\n")}\n)`;
  const lines: string[] = [];
  const viewLine = serializeView(store);
  if (viewLine) lines.push(viewLine);
  if (defLines.length) lines.push(defLines.join("\n"));
  lines.push(body);
  return lines.join("\n");
}

// Emit a `View(...)` line only when the region was explicitly authored (by
// import of a `View(...)` line or by editing the region in the panel). An
// unauthored default region is omitted, mirroring the Python engine's
// `_view_set`. `String(v)` matches the Python engine's number formatting.
function serializeView(store: Store): string {
  const s = store.settings;
  if (!s.viewSet) return "";
  return `View(${s.x1}, ${s.y1}, ${s.x2}, ${s.y2})`;
}

// One layer as a `Layer(...)` header plus one indented `.channel(...)` line each.
// The caller indents the whole block to nest it under `Material(`.
function serializeLayer(
  layer: LayerState,
  env: Map<string, Expression2D>,
): string {
  const ch = layer.channels;
  const bc = layer.colorMode.basecolor;
  const em = layer.colorMode.emissive;
  const set = layer.setChannels;
  // Emit only channels the author explicitly set; an untouched alpha yields
  // `Layer()`. Mirrors the Python engine's `_serialize_layer`.
  const lines: string[] = [
    set.has("alpha")
      ? `Layer(${field(layer, "alpha", ch.alpha, env)})`
      : "Layer()",
  ];
  if (set.has("basecolor"))
    lines.push(
      `  .basecolor(${field(layer, "basecolor.r", ch.basecolor.r, env, bc)}, ` +
        `${field(layer, "basecolor.g", ch.basecolor.g, env, bc)}, ` +
        `${field(layer, "basecolor.b", ch.basecolor.b, env, bc)})`,
    );
  if (set.has("metallic"))
    lines.push(`  .metallic(${field(layer, "metallic", ch.metallic, env)})`);
  if (set.has("roughness"))
    lines.push(`  .roughness(${field(layer, "roughness", ch.roughness, env)})`);
  if (set.has("sheen"))
    lines.push(`  .sheen(${field(layer, "sheen", ch.sheen, env)})`);
  if (set.has("coat"))
    lines.push(`  .coat(${field(layer, "coat", ch.coat, env)})`);
  if (set.has("transmission"))
    lines.push(
      `  .transmission(${field(layer, "transmission", ch.transmission, env)})`,
    );
  if (set.has("ior")) lines.push(`  .ior(${field(layer, "ior", ch.ior, env)})`);
  if (set.has("subsurface"))
    lines.push(
      `  .subsurface(${field(layer, "subsurface", ch.subsurface, env)})`,
    );
  if (set.has("anisotropy"))
    lines.push(
      `  .anisotropy(${field(layer, "anisotropy", ch.anisotropy, env)})`,
    );
  if (set.has("emissive"))
    lines.push(
      `  .emissive(${field(layer, "emissive.r", ch.emissive.r, env, em)}, ` +
        `${field(layer, "emissive.g", ch.emissive.g, env, em)}, ` +
        `${field(layer, "emissive.b", ch.emissive.b, env, em)}, ` +
        `${field(layer, "emissive.strength", ch.emissive.strength, env)})`,
    );
  if (set.has("height"))
    lines.push(`  .height(${field(layer, "height", ch.height, env)})`);
  return lines.join("\n");
}

function indentLines(text: string, pad: string): string {
  return text
    .split("\n")
    .map((line) => pad + line)
    .join("\n");
}

// Serialize one channel: parse its source, re-attach the seed the UI is showing
// for any auto-seeded noise, then emit canonical source. Color channels are
// scaled to the 0–255 convention when the UI shows them in 0–1 mode.
function field(
  layer: LayerState,
  fieldKey: string,
  src: string,
  env: Map<string, Expression2D>,
  colorMode?: ColorMode,
): string {
  const expr = parseExpr(src, env);
  bakeSeeds(layer, fieldKey, expr);
  const out = serializeExpr(expr);
  return colorMode === "01" ? `(${out} * 255)` : out;
}

// Restore the seeds the evaluator remembered for this field's auto-seeded noise
// onto the freshly parsed tree, so the export bakes the seed currently rendered
// rather than a throwaway one. Mirrors the restore half of eval.ts's
// reconcileSeeds (queue per argument signature for repeated identical noises).
function bakeSeeds(
  layer: LayerState,
  fieldKey: string,
  expr: Expression2D,
): void {
  const mem = layer.seedMemory[fieldKey];
  if (!mem) return;
  const avail: Record<string, number[]> = {};
  for (const sig of Object.keys(mem)) avail[sig] = mem[sig].slice();
  for (const node of collectNoise(expr)) {
    if (!node.autoSeeded) continue;
    const q = avail[node.argSignature()];
    const seed = q && q.length ? q.shift() : undefined;
    if (seed !== undefined) node.restoreSeed(seed);
  }
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

interface ParsedLayer {
  alpha?: string;
  basecolor?: [string, string, string];
  metallic?: string;
  roughness?: string;
  sheen?: string;
  coat?: string;
  transmission?: string;
  ior?: string;
  subsurface?: string;
  anisotropy?: string;
  emissive?: [string, string, string, string];
  height?: string;
}

/** Single-scalar OpenPBR channels handled uniformly on import. */
const SCALAR_OPENPBR_CHANNELS = [
  "sheen",
  "coat",
  "transmission",
  "ior",
  "subsurface",
  "anisotropy",
] as const;

/**
 * Parse a material string into fresh top-most-first LayerState objects and the
 * Define preamble. Throws (without mutating anything) if the string is
 * malformed or any channel/definition fails to parse, so a bad import never
 * clobbers the current stack.
 */
export function parseImportedLayers(text: string): {
  layers: LayerState[];
  defs: MaterialDef[];
  view: ImportedView | null;
} {
  const { view, rest: afterView } = splitView(text);
  const { defs, materialSrc } = splitDefs(afterView);
  // Resolve definitions in order so later ones may reference earlier ones.
  const env = new Map<string, Expression2D>();
  for (const d of defs) {
    try {
      env.set(d.name, parseExpr(d.src, env));
    } catch (e) {
      throw new SyntaxError(
        `invalid definition '${d.name}': ${(e as Error).message}`,
      );
    }
  }
  // Format is bottom-most first; the store wants top-most first.
  const layers = parseMaterial(materialSrc)
    .reverse()
    .map((p) => toLayerState(p, env));
  return { layers, defs, view };
}

// Split a leading `View(x1, y1, x2, y2)` statement (if any) from the rest.
function splitView(text: string): { view: ImportedView | null; rest: string } {
  const src = text.trim();
  if (!/^View\s*\(/.test(src)) return { view: null, rest: src };
  const open = src.indexOf("(");
  const close = matchParen(src, open);
  const parts = splitTopLevel(src.slice(open + 1, close));
  if (parts.length !== 4)
    throw new SyntaxError(
      "View(...) takes exactly 4 arguments (x1, y1, x2, y2)",
    );
  const n = parts.map((p) => {
    const v = /\S/.test(p) ? Number(p) : NaN;
    if (!Number.isFinite(v))
      throw new SyntaxError(`invalid View argument: '${p}'`);
    return v;
  });
  return {
    view: { x1: n[0], y1: n[1], x2: n[2], y2: n[3] },
    rest: src.slice(close + 1).trim(),
  };
}

function toLayerState(
  p: ParsedLayer,
  env: Map<string, Expression2D>,
): LayerState {
  const ch = defaultChannels();
  // Set-tracking: only channels present in the imported text are "set", so a
  // re-export preserves the author's exact channel set instead of re-emitting
  // every default.
  const setChannels = new Set<ChannelKey>();
  if (p.alpha !== undefined) {
    ch.alpha = p.alpha;
    setChannels.add("alpha");
  }
  if (p.basecolor) {
    [ch.basecolor.r, ch.basecolor.g, ch.basecolor.b] = p.basecolor;
    setChannels.add("basecolor");
  }
  if (p.metallic !== undefined) {
    ch.metallic = p.metallic;
    setChannels.add("metallic");
  }
  if (p.roughness !== undefined) {
    ch.roughness = p.roughness;
    setChannels.add("roughness");
  }
  for (const key of SCALAR_OPENPBR_CHANNELS) {
    const v = p[key];
    if (v !== undefined) {
      ch[key] = v;
      setChannels.add(key);
    }
  }
  if (p.emissive) {
    [ch.emissive.r, ch.emissive.g, ch.emissive.b, ch.emissive.strength] =
      p.emissive;
    setChannels.add("emissive");
  }
  if (p.height !== undefined) {
    ch.height = p.height;
    setChannels.add("height");
  }
  // Validate every channel parses before we commit to building the layer.
  for (const [key, val] of Object.entries({
    alpha: ch.alpha,
    "basecolor.r": ch.basecolor.r,
    "basecolor.g": ch.basecolor.g,
    "basecolor.b": ch.basecolor.b,
    metallic: ch.metallic,
    roughness: ch.roughness,
    sheen: ch.sheen,
    coat: ch.coat,
    transmission: ch.transmission,
    ior: ch.ior,
    subsurface: ch.subsurface,
    anisotropy: ch.anisotropy,
    "emissive.r": ch.emissive.r,
    "emissive.g": ch.emissive.g,
    "emissive.b": ch.emissive.b,
    "emissive.strength": ch.emissive.strength,
    height: ch.height,
  })) {
    try {
      parseExpr(val, env);
    } catch (e) {
      throw new SyntaxError(
        `invalid expression for ${key}: ${(e as Error).message}`,
      );
    }
  }
  return {
    id: uid(),
    enabled: true,
    collapsed: false,
    channels: ch,
    setChannels,
    // The format expresses colors in 0–255, so imported colors use 0–255 mode.
    colorMode: { basecolor: "255", emissive: "255" },
    errors: {},
    editorRefs: {},
    seedMemory: {},
  };
}

// ---------------------------------------------------------------------------
// Material-string parser (bracket- and quote-aware)
// ---------------------------------------------------------------------------

// Split leading `Define(name, expr)` statements from the `Material(...)` body.
function splitDefs(text: string): {
  defs: MaterialDef[];
  materialSrc: string;
} {
  let src = text.trim();
  const defs: MaterialDef[] = [];
  while (/^Define\s*\(/.test(src)) {
    const open = src.indexOf("(");
    const close = matchParen(src, open);
    const parts = splitTopLevel(src.slice(open + 1, close));
    if (parts.length !== 2)
      throw new SyntaxError("Define(name, expr) takes exactly two arguments");
    if (!/^[A-Za-z_]\w*$/.test(parts[0]))
      throw new SyntaxError(`invalid definition name: ${parts[0]}`);
    defs.push({ name: parts[0], src: parts[1] });
    src = src.slice(close + 1).trim();
  }
  return { defs, materialSrc: src };
}

function parseMaterial(text: string): ParsedLayer[] {
  const src = text.trim();
  if (!/^Material\s*\(/.test(src))
    throw new SyntaxError("expected the material to start with 'Material('");
  const open = src.indexOf("(");
  const close = matchParen(src, open);
  if (src.slice(close + 1).trim() !== "")
    throw new SyntaxError("unexpected text after the closing ')'");
  return splitTopLevel(src.slice(open + 1, close)).map(parseLayerChunk);
}

function parseLayerChunk(chunk: string): ParsedLayer {
  const text = chunk.trim();
  if (!/^Layer\s*\(/.test(text))
    throw new SyntaxError(`expected 'Layer(' in: ${text}`);
  const result: ParsedLayer = {};

  let pos = text.indexOf("(");
  let close = matchParen(text, pos);
  const layerArgs = splitTopLevel(text.slice(pos + 1, close));
  if (layerArgs.length > 1)
    throw new SyntaxError("Layer() takes at most one argument (alpha)");
  if (layerArgs.length === 1) result.alpha = layerArgs[0];
  pos = close + 1;

  while (pos < text.length) {
    while (pos < text.length && /\s/.test(text[pos])) pos++;
    if (pos >= text.length) break;
    if (text[pos] !== ".")
      throw new SyntaxError(`expected '.' but found '${text[pos]}'`);
    pos++;
    const nameStart = pos;
    while (pos < text.length && /[A-Za-z_]/.test(text[pos])) pos++;
    const method = text.slice(nameStart, pos);
    while (pos < text.length && /\s/.test(text[pos])) pos++;
    if (text[pos] !== "(")
      throw new SyntaxError(`expected '(' after .${method}`);
    close = matchParen(text, pos);
    applyMethod(result, method, splitTopLevel(text.slice(pos + 1, close)));
    pos = close + 1;
  }
  return result;
}

function applyMethod(r: ParsedLayer, method: string, args: string[]): void {
  const expect = (n: number) => {
    if (args.length !== n)
      throw new SyntaxError(
        `${method}() expects ${n} argument(s), got ${args.length}`,
      );
  };
  switch (method) {
    case "alpha":
      expect(1);
      r.alpha = args[0];
      break;
    case "basecolor":
      expect(3);
      r.basecolor = [args[0], args[1], args[2]];
      break;
    case "metallic":
      expect(1);
      r.metallic = args[0];
      break;
    case "roughness":
      expect(1);
      r.roughness = args[0];
      break;
    case "sheen":
    case "coat":
    case "transmission":
    case "ior":
    case "subsurface":
    case "anisotropy":
      expect(1);
      r[method] = args[0];
      break;
    case "emissive":
      expect(4);
      r.emissive = [args[0], args[1], args[2], args[3]];
      break;
    case "height":
      expect(1);
      r.height = args[0];
      break;
    default:
      throw new SyntaxError(`unknown layer property: ${method}`);
  }
}

// Index of the ')' matching the '(' at `open`, ignoring parens inside strings.
function matchParen(text: string, open: number): number {
  let depth = 0;
  let quote = "";
  for (let i = open; i < text.length; i++) {
    const c = text[i];
    if (quote) {
      if (c === quote) quote = "";
      continue;
    }
    if (c === "'" || c === '"') quote = c;
    else if (c === "(") depth++;
    else if (c === ")" && --depth === 0) return i;
  }
  throw new SyntaxError("unbalanced parentheses");
}

// Split at top-level commas (depth 0, outside strings). Whitespace-only input
// yields an empty list; every returned fragment is trimmed.
function splitTopLevel(s: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let quote = "";
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (quote) {
      if (c === quote) quote = "";
      continue;
    }
    if (c === "'" || c === '"') quote = c;
    else if (c === "(") depth++;
    else if (c === ")") depth--;
    else if (c === "," && depth === 0) {
      out.push(s.slice(start, i).trim());
      start = i + 1;
    }
  }
  const tail = s.slice(start).trim();
  if (out.length === 0 && tail === "") return [];
  out.push(tail);
  return out;
}
