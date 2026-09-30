/**
 * eval.ts — turns the layer stack into a rendered frame. Editing a channel
 * debounces → parse → build a MaterialSpec → evaluate it on a backend →
 * viewer.applyFrame(). A low-res preview is rendered first on rapid edits, with
 * the full-res frame following once typing settles.
 *
 * Parsing, the noise-seed lifecycle and the noise chips stay on the main thread
 * (they own UI + RNG state); the spec they produce — canonical DSL with seeds
 * baked in — is handed to a pluggable {@link EvalBackend} that runs the
 * byte-identical `eval-core` engine on a worker (default), the main thread
 * (fallback), a worker pool, or a WebGPU preview tier.
 */
import {
  Constant,
  DivisionExpression2D,
  collectNoise,
  type Expression2D,
  type NoiseExpression2D,
} from "../engine";
import { parseExpr } from "../expr-lang";
import { buildSpec } from "../eval-core";
import {
  SyncBackend,
  workersSupported,
  type EvalBackend,
} from "../eval/backend";
import { createPoolBackend } from "../eval/pool";
import type { ViewerHandle } from "../viewer";
import { byId } from "./dom";
import { flatChannels } from "./channels";
import type { LayerState, Store } from "./state";

const DEBOUNCE_MS = 180;
const IDLE_MS = 350;

type Status = "idle" | "busy" | "ok" | "error";

/**
 * Crop-mode region override: the region to render *instead of* the authored
 * View, set by the crop controller as the camera moves; null restores the
 * authored region. The render *resolution* is never overridden — it is always
 * the user's manual width/height. Setting it does not itself trigger an
 * evaluation — the caller schedules one.
 */
export interface CropRegion {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface ChannelError {
  layer: string;
  channel: string;
  message: string;
}

export interface Evaluator {
  /** Schedule an evaluation; `lowRes` renders a quick preview first. */
  scheduleEval(lowRes?: boolean): void;
  /**
   * Roll a fresh seed for every auto-seeded noise node in the current material
   * and re-render. No-op when there are none. Pinned (`seed=`) noise is never
   * touched.
   */
  reseed(): void;
  /**
   * Register a listener for how many auto-seeded noise nodes the last render
   * contained, so the UI can show/hide the reseed control. Called on every
   * evaluation; fires immediately with the current count.
   */
  onReseedableChange(cb: (count: number) => void): void;
  /**
   * Replace the authoritative ("settle") backend — the full-resolution frame
   * that is the source of truth. Disposes the previous one and re-renders. The
   * eval-core CPU tiers (sync / worker / pool) are all byte-identical.
   */
  setSettleBackend(next: EvalBackend): void;
  /**
   * Set (or clear, with null) the optional fast "preview" backend used only for
   * the low-res pass during rapid edits — e.g. the WebGPU compute tier. It may
   * be an approximate (not parity-faithful) renderer; the settle backend always
   * produces the committed frame. Disposes any previous preview backend.
   */
  setPreviewBackend(next: EvalBackend | null): void;
  /** The active backend kinds, for the UI/status line. */
  backendKinds(): {
    settle: EvalBackend["kind"];
    preview: EvalBackend["kind"] | null;
  };
  /**
   * Set (or clear, with null) the crop-mode region — the region rendered in
   * place of the authored View. The resolution is unaffected (always the manual
   * width/height). Does not trigger an evaluation; call {@link scheduleEval}
   * afterwards.
   */
  setCropRegion(region: CropRegion | null): void;
  /**
   * Evaluate immediately at the current effective region/resolution, bypassing
   * the debounce. Used by the crop controller, which has already coalesced
   * camera motion, so the extra debounce would only add lag.
   */
  renderNow(): void;
}

// The best parity-faithful CPU backend available: the tile pool (multi-core)
// when workers exist, else the synchronous main-thread fallback. The pool's
// worker construction can throw under a strict CSP, so degrade gracefully.
function createDefaultSettleBackend(): EvalBackend {
  if (workersSupported()) {
    try {
      return createPoolBackend();
    } catch {
      // fall through to the synchronous backend
    }
  }
  return new SyncBackend();
}

export function createEvaluator(
  store: Store,
  getViewer: () => ViewerHandle,
): Evaluator {
  let debTimer: ReturnType<typeof setTimeout> | undefined;
  let idleTimer: ReturnType<typeof setTimeout> | undefined;

  // The authoritative ("settle") backend: the full-resolution frame that is the
  // source of truth. Defaults to the best available parity-faithful CPU tier
  // (tile pool → single worker → synchronous fallback).
  let settleBackend: EvalBackend = createDefaultSettleBackend();
  // Optional fast "preview" backend for the low-res pass only (e.g. the WebGPU
  // compute tier, which is approximate). Null = use the settle backend for both.
  let previewBackend: EvalBackend | null = null;
  // Monotonic evaluation generation. A backend result (which arrives
  // asynchronously) is applied only when no newer evaluation has started since
  // it was dispatched, so a slow full-res frame never clobbers a newer preview.
  let evalGen = 0;

  // Crop-mode region override: when set, the rendered region comes from here
  // instead of store.settings (the authored View). Set by the crop controller
  // as the camera moves. The resolution always stays the user's manual setting.
  let cropRegion: CropRegion | null = null;

  // The region to actually render: the crop override when active, else the
  // authored settings. yUp always tracks the authored setting. The relief
  // reference span is *always* the authored View span (never the crop), so the
  // AO blur covers a constant world area and the cavity shading does not drift
  // as a crop narrows the rendered region.
  function effectiveRegion(): {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    yUp: boolean;
    reliefSpanX: number;
    reliefSpanY: number;
  } {
    const r = cropRegion ?? store.settings;
    const s = store.settings;
    return {
      x1: r.x1,
      y1: r.y1,
      x2: r.x2,
      y2: r.y2,
      yUp: s.yUp,
      reliefSpanX: Math.abs(s.x2 - s.x1),
      reliefSpanY: Math.abs(s.y2 - s.y1),
    };
  }
  function effectiveDims(): [number, number] {
    // Resolution is always the user's manual setting — crop mode only changes
    // the rendered region, never the texel count.
    return [store.settings.width, store.settings.height];
  }

  // Auto-seeded noise nodes in the most recent render, and the listener that
  // mirrors their count into the reseed button. Captured each runEval so the
  // button tracks whatever noise the current material actually uses.
  let reseedable: NoiseExpression2D[] = [];
  let reseedListener: ((count: number) => void) | null = null;

  // Parsed-expression cache, keyed by `${layerId}::${fieldKey}`. Reusing the
  // expression object whenever a field's source is unchanged means editing one
  // channel no longer re-parses (and re-seeds unseeded noise in) the others;
  // the stable objects also let the engine's noise nodes hit their memoized
  // grids, so untouched channels skip recomputation entirely.
  const parseCache = new Map<string, { src: string; expr: Expression2D }>();

  // Resolved material-level definitions, cached by a version token (the
  // serialized defs). Rebuilt only when the defs change, so a definition's
  // expression objects — and any auto-seeded noise inside them — stay stable
  // across edits to unrelated channels. A def that fails to parse is skipped;
  // channels referencing it then surface their own parse error.
  let defsCache: { version: string; env: Map<string, Expression2D> } | null =
    null;
  function buildDefsEnv(): { version: string; env: Map<string, Expression2D> } {
    const version = JSON.stringify(store.defs);
    if (defsCache && defsCache.version === version) return defsCache;
    const env = new Map<string, Expression2D>();
    for (const d of store.defs) {
      try {
        env.set(d.name, parseExpr(d.src, env));
      } catch {
        // Broken definition: leave it out of the env.
      }
    }
    defsCache = { version, env };
    return defsCache;
  }

  // Drop entries for layers that no longer exist so removed layers don't pin
  // their (potentially large) memoized noise grids in memory.
  function pruneParseCache(): void {
    const live = new Set(store.layers.map((l) => l.id));
    for (const k of parseCache.keys()) {
      if (!live.has(k.slice(0, k.indexOf("::")))) parseCache.delete(k);
    }
  }

  // Parse a field, reusing the cached expression when its source is unchanged.
  // Returns `fresh: true` only when the source actually re-parsed (new nodes),
  // which is the moment reconcileSeeds must re-attach remembered seeds. A parse
  // error leaves the stale entry in place but the source-mismatch guard prevents
  // it from ever being reused for the new (broken) source.
  function parseField(
    layerId: string,
    key: string,
    src: string,
    env: Map<string, Expression2D>,
    version: string,
  ): { expr: Expression2D; fresh: boolean } {
    const cacheKey = `${layerId}::${key}::${version}`;
    const hit = parseCache.get(cacheKey);
    if (hit && hit.src === src) return { expr: hit.expr, fresh: false };
    const expr = parseExpr(src, env);
    parseCache.set(cacheKey, { src, expr });
    return { expr, fresh: true };
  }

  // Keep an unseeded noise's auto-generated seed stable across edits. On a fresh
  // parse the new node arrives with a throwaway random seed; if the field
  // remembers a seed for that node's argument signature (from a previous render
  // or a duplicated layer), re-attach it. Either way the field's memory is
  // rebuilt from the live nodes so reseeds are captured and a pinned `seed=`
  // node still reserves its slot — so removing the `seed=` later restores the
  // original generated seed. Signatures no longer present are dropped.
  function reconcileSeeds(
    layer: LayerState,
    key: string,
    expr: Expression2D,
    fresh: boolean,
  ): void {
    const nodes = collectNoise(expr);
    if (!nodes.length) {
      delete layer.seedMemory[key];
      return;
    }
    const prev = layer.seedMemory[key] ?? {};
    const avail: Record<string, number[]> = {};
    for (const sig of Object.keys(prev)) avail[sig] = prev[sig].slice();
    const next: Record<string, number[]> = {};
    const take = (sig: string): number | undefined => {
      const q = avail[sig];
      return q && q.length ? q.shift() : undefined;
    };
    const put = (sig: string, seed: number): void => {
      (next[sig] ??= []).push(seed);
    };
    for (const node of nodes) {
      const sig = node.argSignature();
      const remembered = take(sig);
      if (node.autoSeeded) {
        if (fresh && remembered !== undefined) node.restoreSeed(remembered);
        put(sig, node.currentSeed);
      } else if (remembered !== undefined) {
        // Pinned now, but it had an auto seed before — keep it reserved so
        // un-pinning (removing `seed=`) brings the original seed back.
        put(sig, remembered);
      }
    }
    layer.seedMemory[key] = next;
  }

  // Reduced-resolution preview that keeps the aspect ratio (cap longest side).
  function previewDims(): [number, number] {
    const [w, h] = effectiveDims();
    const longest = Math.max(w, h);
    if (longest <= 256) return [w, h];
    const f = 256 / longest;
    return [Math.max(1, Math.round(w * f)), Math.max(1, Math.round(h * f))];
  }

  function scheduleEval(lowRes = false): void {
    clearTimeout(debTimer);
    clearTimeout(idleTimer);
    if (lowRes) {
      // The quick low-res pass may use the (approximate) preview backend; the
      // follow-up full-res pass always uses the authoritative settle backend.
      // Dimensions are read at fire time so the latest crop region applies.
      debTimer = setTimeout(() => {
        const [lw, lh] = previewDims();
        runEval(lw, lh, true);
      }, DEBOUNCE_MS);
      idleTimer = setTimeout(() => {
        const [w, h] = effectiveDims();
        runEval(w, h, false);
      }, DEBOUNCE_MS + IDLE_MS);
    } else {
      debTimer = setTimeout(() => {
        const [w, h] = effectiveDims();
        runEval(w, h, false);
      }, DEBOUNCE_MS);
    }
  }

  function renderNow(): void {
    clearTimeout(debTimer);
    clearTimeout(idleTimer);
    const [w, h] = effectiveDims();
    runEval(w, h, false);
  }

  function runEval(w: number, h: number, preview = false): void {
    const enabled = [...store.layers].reverse().filter((l) => l.enabled);

    pruneParseCache();

    // Reset per-field noise chips; valid layers repopulate them below.
    for (const l of store.layers)
      for (const ed of Object.values(l.editorRefs)) ed.setNoise([]);

    // Nothing to render (no layers, or all hidden): show nothing at all — not
    // even the plane — so an empty stack reads as a blank scene.
    if (!enabled.length) {
      getViewer().setPlaneVisible(false);
      updateReseedable([]);
      setStatus("idle");
      return;
    }

    const errors: ChannelError[] = [];
    const specLayers: Record<string, Expression2D>[] = [];
    const allNoise: NoiseExpression2D[] = [];
    const { version: defsVersion, env: defsEnv } = buildDefsEnv();
    for (const l of enabled) {
      const parsed: Record<string, Expression2D> = {};
      const layerErrors: Record<string, string> = {};
      for (const [key, src] of flatChannels(l.channels)) {
        try {
          const { expr, fresh } = parseField(
            l.id,
            key,
            src,
            defsEnv,
            defsVersion,
          );
          reconcileSeeds(l, key, expr, fresh);
          parsed[key] = expr;
        } catch (e) {
          const message = (e as Error).message;
          layerErrors[key] = message;
          errors.push({ layer: l.id, channel: key, message });
        }
      }
      if (Object.keys(layerErrors).length) continue;
      specLayers.push(engineFields(parsed, l));

      // Surface each field's auto-seeded noise as chips on its editor.
      for (const [key, expr] of Object.entries(parsed)) {
        const nz = collectNoise(expr).filter((n) => n.autoSeeded);
        if (!nz.length) continue;
        allNoise.push(...nz);
        l.editorRefs[key]?.setNoise(
          nz.map((node) => ({
            label: node.sourceText || node.noiseName,
            seed: node.currentSeed,
            range:
              node.sourceStart >= 0
                ? ([node.sourceStart, node.sourceEnd] as [number, number])
                : undefined,
            reroll: () => reseedNodes([node]),
          })),
        );
      }
    }
    applyErrors(errors);
    if (!specLayers.length) {
      setStatus("error");
      updateReseedable([]);
      return;
    }
    setStatus(errors.length ? "error" : "busy");

    // Only the auto-seeded noise of the layers that rendered is rerollable.
    updateReseedable(allNoise);

    // Serialize the resolved (seed-reconciled) trees into a transferable spec and
    // hand it to the backend. The result is asynchronous; the generation guard
    // drops it if a newer evaluation has started in the meantime.
    const region = effectiveRegion();
    const spec = buildSpec(specLayers, defsEnv, region, w, h);

    const gen = ++evalGen;
    const t0 = performance.now();
    // Preview passes may use the fast (approximate) preview backend; everything
    // else uses the authoritative settle backend.
    const active = preview && previewBackend ? previewBackend : settleBackend;
    active.evaluate(spec).then(
      (maps) => {
        if (gen !== evalGen) return; // superseded by a newer evaluation
        const ms = (performance.now() - t0).toFixed(0);
        getViewer().setPlaneVisible(true);
        // `!preview`: only the authoritative full-res frame may auto-re-fit the
        // camera to a changed relief depth (previews would jitter it while typing).
        // The region travels with the frame so the plane geometry and texture
        // update together (no stale-texture stretch on a crop).
        getViewer().applyFrame(
          {
            width: w,
            height: h,
            region: {
              x1: region.x1,
              y1: region.y1,
              x2: region.x2,
              y2: region.y2,
            },
            ...maps,
          },
          !preview,
        );
        setStatus(errors.length ? "error" : "ok");
        byId("eval-time").textContent = `${ms} ms`;
      },
      (e) => {
        if (gen !== evalGen) return;
        setStatus("error");
        console.error(e);
      },
    );
  }

  function applyErrors(errors: ChannelError[]): void {
    for (const l of store.layers) {
      for (const ed of Object.values(l.editorRefs)) ed.clearErr();
      l.errors = {};
    }
    for (const { layer, channel, message } of errors) {
      const l = store.layers.find((x) => x.id === layer);
      if (!l) continue;
      l.errors[channel] = message;
      l.editorRefs[channel]?.setErr(message);
    }
  }

  // Remember the rerollable noise of the latest render and notify the UI.
  function updateReseedable(nodes: NoiseExpression2D[]): void {
    reseedable = nodes;
    reseedListener?.(nodes.length);
  }

  // Reroll the given nodes and re-render. Used by both the global reseed and the
  // per-node chips. Renders at full resolution straight away (an explicit user
  // action); the reseeded nodes dropped their memos, so only they recompute,
  // and runEval refreshes every chip's revealed seed.
  function reseedNodes(nodes: NoiseExpression2D[]): void {
    let changed = false;
    for (const node of nodes) changed = node.reseed() || changed;
    if (!changed) return;
    const [w, h] = effectiveDims();
    runEval(w, h);
  }

  function reseed(): void {
    reseedNodes(reseedable);
  }

  return {
    scheduleEval,
    reseed,
    onReseedableChange(cb) {
      reseedListener = cb;
      cb(reseedable.length);
    },
    setSettleBackend(next) {
      if (next === settleBackend) return;
      settleBackend.dispose();
      settleBackend = next;
      scheduleEval();
    },
    setPreviewBackend(next) {
      if (next === previewBackend) return;
      previewBackend?.dispose();
      previewBackend = next;
    },
    backendKinds() {
      return {
        settle: settleBackend.kind,
        preview: previewBackend?.kind ?? null,
      };
    },
    setCropRegion(region) {
      cropRegion = region;
    },
    renderNow,
  };
}

// Resolve a layer's parsed channels into the engine-domain field expressions a
// MaterialSpec carries (one per FIELD_KEYS). Color channels in 0–255 mode are
// divided by 255 here so the spec is already in the engine's native [0,1]
// domain; the backend re-wraps each field in the Color/Layer model.
function engineFields(
  parsed: Record<string, Expression2D>,
  layer: LayerState,
): Record<string, Expression2D> {
  const scale255 = (expr: Expression2D) =>
    new DivisionExpression2D(expr, new Constant(255));
  const channel = (key: string, mode: "01" | "255") =>
    mode === "255" ? scale255(parsed[key]) : parsed[key];

  const bc = layer.colorMode.basecolor;
  const em = layer.colorMode.emissive;

  return {
    alpha: parsed["alpha"],
    "basecolor.r": channel("basecolor.r", bc),
    "basecolor.g": channel("basecolor.g", bc),
    "basecolor.b": channel("basecolor.b", bc),
    metallic: parsed["metallic"],
    roughness: parsed["roughness"],
    sheen: parsed["sheen"],
    coat: parsed["coat"],
    transmission: parsed["transmission"],
    ior: parsed["ior"],
    subsurface: parsed["subsurface"],
    anisotropy: parsed["anisotropy"],
    "emissive.r": channel("emissive.r", em),
    "emissive.g": channel("emissive.g", em),
    "emissive.b": channel("emissive.b", em),
    "emissive.strength": parsed["emissive.strength"],
    height: parsed["height"],
  };
}

function setStatus(state: Status): void {
  byId("status-dot").className = `dot dot-${state}`;
}
