/**
 * eval-core.ts — the DOM-free heart of material evaluation, shared by the main
 * thread, the single eval worker, and the tile-parallel worker pool.
 *
 * The viewer's app layer (`app/eval.ts`) owns parsing, the noise-seed lifecycle
 * and the UI; once it has resolved a layer stack to concrete `Expression2D`
 * trees it serializes them — via the canonical DSL (`serializeExpr`), with every
 * auto-seed baked in — into a {@link MaterialSpec}. A spec is a plain,
 * structured-cloneable value: it crosses a `postMessage` boundary unchanged and
 * re-evaluates to a byte-identical result on the far side, because
 * `serializeExpr` → `parseExpr` is loss-free within one JS engine (numbers
 * round-trip through `String`/`Number`) and the engine math is deterministic.
 *
 * `evalSpec()` rebuilds the layer stack from a spec and runs the exact same
 * `evaluateMaterial()` the synchronous path always used, so moving evaluation
 * onto a worker changes *where* the work runs, never *what* it computes. A
 * {@link SpecCache} preserves the parsed trees (and therefore the noise grid
 * memos) across calls, so editing one channel does not recompute the others —
 * the same memoization the single-threaded path relied on, now per worker.
 */
import {
  Color,
  Emissive,
  Layer,
  evaluateMaterial,
  evalChannels,
  axisCoords,
  resolveRegion,
  type ChannelData,
  type EvalRegion,
  type Expression2D,
  type MaterialMaps,
} from "./engine";
import { parseExpr, serializeExpr } from "./expr-lang";

/** Half-open row range `[y0, y1)` of the full grid (used by the tile pool). */
export interface RowBand {
  y0: number;
  y1: number;
}

/**
 * The seventeen channel fields of a layer, in the canonical order. A spec layer
 * is a record keyed by these; engine-domain (already scaled to [0,1] where the
 * UI used a 0–255 color convention), so the worker assembles a `Layer` directly
 * without needing the layer's color mode.
 */
export const FIELD_KEYS = [
  "alpha",
  "basecolor.r",
  "basecolor.g",
  "basecolor.b",
  "metallic",
  "roughness",
  "sheen",
  "coat",
  "transmission",
  "ior",
  "subsurface",
  "anisotropy",
  "emissive.r",
  "emissive.g",
  "emissive.b",
  "emissive.strength",
  "height",
] as const;

export type FieldKey = (typeof FIELD_KEYS)[number];

/** A single `Define(name, expr)`, serialized (seeds baked in). */
export interface SpecDef {
  name: string;
  src: string;
}

/**
 * A fully self-contained, transferable description of a material to evaluate.
 * Layers are bottom-most first (the order `evaluateMaterial` composites). Field
 * strings are canonical DSL in the engine's native [0,1] domain.
 */
export interface MaterialSpec {
  /** Ordered `Define`s; later ones may reference earlier ones. */
  defs: SpecDef[];
  /** Bottom-most-first layers; each maps every {@link FIELD_KEYS} to DSL. */
  layers: Record<string, string>[];
  region: EvalRegion;
  width: number;
  height: number;
}

/**
 * Serialize resolved per-field expression trees (engine-domain) and the
 * definition environment into a {@link MaterialSpec}. The defs are serialized
 * from their resolved expressions so any auto-seeded noise inside a definition
 * is baked at the seed the main thread is rendering, exactly as the channel
 * fields are.
 */
export function buildSpec(
  layers: Record<string, Expression2D>[],
  defsEnv: Map<string, Expression2D>,
  region: EvalRegion,
  width: number,
  height: number,
): MaterialSpec {
  const defs: SpecDef[] = [];
  for (const [name, expr] of defsEnv)
    defs.push({ name, src: serializeExpr(expr) });
  const specLayers = layers.map((fields) => {
    const out: Record<string, string> = {};
    for (const key of FIELD_KEYS) out[key] = serializeExpr(fields[key]);
    return out;
  });
  return { defs, layers: specLayers, region, width, height };
}

/**
 * Persistent parse + noise-memo cache for repeated {@link evalSpec} calls. One
 * instance lives for the lifetime of a backend (the main thread, or a worker),
 * so a field whose DSL is unchanged between evaluations reuses its parsed tree
 * — and that tree's memoized noise grids — instead of recomputing. Editing one
 * channel therefore only recomputes that channel, matching the single-threaded
 * engine's behavior.
 */
export class SpecCache {
  private defsVersion = "";
  private defsEnv = new Map<string, Expression2D>();
  // Parsed channel-field trees, keyed by their DSL source. Cleared whenever the
  // definitions change (a def's expression objects, and their Refs, go stale).
  private exprCache = new Map<string, Expression2D>();

  /** Build (or reuse) the definition environment for `defs`. */
  private getDefsEnv(defs: SpecDef[]): Map<string, Expression2D> {
    const version = JSON.stringify(defs);
    if (version === this.defsVersion) return this.defsEnv;
    const env = new Map<string, Expression2D>();
    for (const d of defs) {
      try {
        env.set(d.name, parseExpr(d.src, env));
      } catch {
        // A malformed def is dropped; fields referencing it then fail to parse
        // below, which the app surfaces as a channel error — same as the
        // main-thread path.
      }
    }
    this.defsVersion = version;
    this.defsEnv = env;
    this.exprCache.clear(); // Refs into the old env are now invalid.
    return env;
  }

  /** Parse a field's DSL, reusing the cached tree when the source is unchanged. */
  private parse(src: string, env: Map<string, Expression2D>): Expression2D {
    const hit = this.exprCache.get(src);
    if (hit) return hit;
    const expr = parseExpr(src, env);
    this.exprCache.set(src, expr);
    return expr;
  }

  // Build the layer stack for a spec, reusing cached trees/memos and pruning
  // entries no longer referenced. Shared by the full and stripe eval paths.
  private buildLayers(spec: MaterialSpec): Layer[] {
    const env = this.getDefsEnv(spec.defs);
    const seen = new Set<string>();
    const layers = spec.layers.map((fields) => {
      for (const key of FIELD_KEYS) seen.add(fields[key]);
      return buildLayerFromFields(fields, env, (src) => this.parse(src, env));
    });
    for (const src of this.exprCache.keys())
      if (!seen.has(src)) this.exprCache.delete(src);
    return layers;
  }

  /** Evaluate a full spec, reusing cached trees/memos where the DSL is unchanged. */
  eval(spec: MaterialSpec): MaterialMaps {
    return evaluateMaterial(
      this.buildLayers(spec),
      spec.width,
      spec.height,
      spec.region,
    );
  }

  /**
   * Evaluate only the rows in `band`, returning a per-cell channel stripe (no
   * relief / emissive normalization — those need the whole grid and are done by
   * the pool once the stripes are assembled). The stripe is byte-identical to
   * the same rows of a full {@link eval}, because every output cell is computed
   * independently from its own coordinates.
   */
  evalStripe(spec: MaterialSpec, band: RowBand): ChannelData {
    const layers = this.buildLayers(spec);
    const r = resolveRegion(spec.region);
    const xs = axisCoords(spec.width, r.x1, r.x2, false);
    // The stripe samples the SAME y-axis values as the full grid (a view into
    // it), so noise/X/Y land on identical coordinates row-for-row.
    const ys = axisCoords(spec.height, r.y1, r.y2, r.yUp).subarray(
      band.y0,
      band.y1,
    );
    return evalChannels(layers, xs, ys, spec.width, band.y1 - band.y0);
  }
}

// Assemble an engine `Layer` from a spec layer's channel DSL fields. Mirrors
// app/eval.ts's buildLayer, minus the color-mode scaling (already baked into the
// field strings): the worker is given engine-domain expressions directly.
function buildLayerFromFields(
  fields: Record<string, string>,
  _env: Map<string, Expression2D>,
  parse: (src: string) => Expression2D,
): Layer {
  return new Layer({
    alpha: parse(fields["alpha"]),
    basecolor: new Color({
      r: parse(fields["basecolor.r"]),
      g: parse(fields["basecolor.g"]),
      b: parse(fields["basecolor.b"]),
    }),
    metallic: parse(fields["metallic"]),
    roughness: parse(fields["roughness"]),
    sheen: parse(fields["sheen"]),
    coat: parse(fields["coat"]),
    transmission: parse(fields["transmission"]),
    ior: parse(fields["ior"]),
    subsurface: parse(fields["subsurface"]),
    anisotropy: parse(fields["anisotropy"]),
    emissive: new Emissive({
      r: parse(fields["emissive.r"]),
      g: parse(fields["emissive.g"]),
      b: parse(fields["emissive.b"]),
      strength: parse(fields["emissive.strength"]),
    }),
    height: parse(fields["height"]),
  });
}

/**
 * Parse a spec into its engine `Layer[]` (bottom-most first), without caching.
 * Used by the WebGPU tier, which codegens shader expressions from the resolved
 * layer trees rather than evaluating them on the CPU.
 */
export function specToLayers(spec: MaterialSpec): Layer[] {
  const env = new Map<string, Expression2D>();
  for (const d of spec.defs) {
    try {
      env.set(d.name, parseExpr(d.src, env));
    } catch {
      // A malformed def is dropped; referencing fields then fail to parse below.
    }
  }
  return spec.layers.map((fields) =>
    buildLayerFromFields(fields, env, (src) => parseExpr(src, env)),
  );
}

/**
 * Evaluate a spec with a fresh, throwaway cache. Convenience for one-shot
 * evaluation (tests, the sync fallback's cold path); long-lived callers should
 * keep a {@link SpecCache} and call `cache.eval(spec)` to retain memoization.
 */
export function evalSpec(spec: MaterialSpec): MaterialMaps {
  return new SpecCache().eval(spec);
}

/** The five backing buffers of a {@link MaterialMaps}, for zero-copy transfer. */
export function mapsTransferList(maps: MaterialMaps): Transferable[] {
  return [
    maps.basecolor.buffer,
    maps.orm.buffer,
    maps.emissive.buffer,
    maps.normal.buffer,
    maps.displacement.buffer,
  ];
}

/** The four backing buffers of a channel stripe, for zero-copy transfer. */
export function channelTransferList(ch: ChannelData): Transferable[] {
  return [ch.basecolor.buffer, ch.orm.buffer, ch.emf.buffer, ch.height.buffer];
}
