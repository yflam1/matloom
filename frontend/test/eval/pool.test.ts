/**
 * pool.test.ts — the tile-parallel pool must produce exactly what whole-grid
 * evaluation produces.
 *
 * The core guarantee (no workers needed): splitting a material into row bands,
 * evaluating each band's channel stripe, reassembling, and running the finishing
 * pass yields maps byte-for-byte identical to a single full `evalSpec`. This is
 * checked across band counts that stress the seams — including the global
 * emissive max (a cross-band reduction) and the relief derivation (a
 * neighborhood op) that run after assembly. A fake-worker test then exercises
 * the PoolBackend plumbing (banded fan-out, out-of-order stripe arrival).
 */
import { describe, it, expect } from "vitest";
import {
  assembleChannels,
  finishMaps,
  resolveRegion,
  type EvalRegion,
  type Expression2D,
  type MaterialMaps,
} from "../../src/engine";
import { parseExpr } from "../../src/expr-lang";
import {
  buildSpec,
  evalSpec,
  SpecCache,
  FIELD_KEYS,
  type MaterialSpec,
} from "../../src/eval-core";
import { PoolBackend, splitBands } from "../../src/eval/pool";
import type { EvalRequest, EvalResponse } from "../../src/eval/protocol";

function specOf(
  layerSrcs: Record<string, string>[],
  region: EvalRegion = {},
  w = 20,
  h = 20,
): MaterialSpec {
  const env = new Map<string, Expression2D>();
  const layers = layerSrcs.map((s) => {
    const fields: Record<string, Expression2D> = {};
    for (const key of FIELD_KEYS)
      fields[key] = parseExpr(s[key] ?? (key === "alpha" ? "1" : "0"), env);
    return fields;
  });
  return buildSpec(layers, env, region, w, h);
}

function expectMapsEqual(a: MaterialMaps, b: MaterialMaps): void {
  expect(a.emissiveIntensity).toBe(b.emissiveIntensity);
  expect(Array.from(a.basecolor)).toEqual(Array.from(b.basecolor));
  expect(Array.from(a.orm)).toEqual(Array.from(b.orm));
  expect(Array.from(a.emissive)).toEqual(Array.from(b.emissive));
  expect(Array.from(a.normal)).toEqual(Array.from(b.normal));
  expect(Array.from(a.displacement)).toEqual(Array.from(b.displacement));
}

// Evaluate a spec by hand-tiling it into `bands` row bands and reassembling —
// the exact sequence the pool performs, minus the worker transport.
function evalTiled(spec: MaterialSpec, bandCount: number): MaterialMaps {
  const cache = new SpecCache();
  const bands = splitBands(spec.height, bandCount);
  const parts = bands.map((band) => ({
    y0: band.y0,
    data: cache.evalStripe(spec, band),
  }));
  const ch = assembleChannels(parts, spec.width, spec.height);
  return finishMaps(ch, spec.width, spec.height, resolveRegion(spec.region));
}

describe("splitBands", () => {
  it("partitions all rows into contiguous, gap-free bands", () => {
    for (const [h, k] of [
      [20, 4],
      [20, 1],
      [20, 20],
      [17, 4],
      [3, 8],
      [1, 4],
    ] as const) {
      const bands = splitBands(h, k);
      expect(bands[0].y0).toBe(0);
      expect(bands[bands.length - 1].y1).toBe(h);
      for (let i = 1; i < bands.length; i++)
        expect(bands[i].y0).toBe(bands[i - 1].y1); // no gaps / overlaps
      for (const b of bands) expect(b.y1).toBeGreaterThan(b.y0); // non-empty
    }
  });

  it("never makes more bands than rows", () => {
    expect(splitBands(3, 8)).toHaveLength(3);
    expect(splitBands(0, 8)).toHaveLength(1);
  });

  it("balances band heights (max one row apart)", () => {
    const bands = splitBands(17, 4); // 5,4,4,4
    const heights = bands.map((b) => b.y1 - b.y0);
    expect(Math.max(...heights) - Math.min(...heights)).toBeLessThanOrEqual(1);
  });
});

describe("tiled evaluation is byte-identical to full evaluation", () => {
  const cases: { name: string; spec: MaterialSpec }[] = [
    {
      name: "noise + threshold (per-band memo)",
      spec: specOf([
        {
          "basecolor.r": "fBm(base_freq=8, to_01=True, seed=1)",
          roughness:
            "Threshold(fBm(base_freq=12, to_01=True, seed=2), below_at=0.4, above_at=0.6)",
        },
      ]),
    },
    {
      name: "height variation (relief on the assembled field)",
      spec: specOf([
        {
          height: "(fBm(base_freq=6, to_01=True, seed=3) * 6)",
          "basecolor.g": "0.5",
        },
      ]),
    },
    {
      name: "emissive (global max spans bands)",
      spec: specOf([
        {
          "emissive.r": "(Y() * 2)", // brightest at one edge → max lives in one band
          "emissive.strength": "5",
          "basecolor.b": "0.2",
        },
      ]),
    },
    {
      name: "shapes + transforms",
      spec: specOf([
        {
          alpha: "Fill(Ellipse(0.5, 0.5, 0.3, 0.25))",
          "basecolor.r": "Rotate(Fill(Rect(0.2, 0.2, 0.5, 0.5)), 25)",
          height: "(Stroke(Ellipse(0.5, 0.5, 0.4, 0.4), 0.05) * 3)",
        },
      ]),
    },
    {
      name: "multi-layer composite",
      spec: specOf([
        { "basecolor.r": "0.3", "basecolor.g": "0.3", roughness: "0.7" },
        {
          alpha:
            "Threshold(fBm(base_freq=5, to_01=True, seed=9), below_at=0.5, below_to=0, above_at=0.5, above_to=1)",
          "basecolor.b": "0.8",
          height: "(fBm(base_freq=5, to_01=True, seed=9) * 4)",
        },
      ]),
    },
    {
      name: "constant height (relief fast path)",
      spec: specOf([{ height: "5", "basecolor.r": "0.6" }]),
    },
    {
      name: "yDown, offset region",
      spec: specOf(
        [
          {
            "basecolor.r": "X()",
            "basecolor.g": "Y()",
            height: "(fBm(seed=4) * 4)",
          },
        ],
        { x1: -0.5, y1: -0.5, x2: 0.5, y2: 0.5, yUp: false },
      ),
    },
  ];

  for (const { name, spec } of cases) {
    it(name, () => {
      const full = evalSpec(spec);
      // Band counts that stress different seam placements, incl. 1 (degenerate),
      // an awkward divisor, and one band per row.
      for (const k of [1, 2, 3, 7, spec.height]) {
        expectMapsEqual(evalTiled(spec, k), full);
      }
    });
  }
});

// A controllable stand-in for a dedicated worker that serves banded requests via
// the real evalStripe, so the assembled result is what the genuine pool yields.
class FakePoolWorker {
  onmessage: ((e: MessageEvent<EvalResponse>) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  terminated = false;
  private cache = new SpecCache();
  private queue: EvalRequest[] = [];

  postMessage(req: EvalRequest): void {
    this.queue.push(req);
  }
  terminate(): void {
    this.terminated = true;
  }
  flushAll(): void {
    while (this.queue.length) {
      const req = this.queue.shift()!;
      const stripe = this.cache.evalStripe(req.spec, req.band!);
      this.onmessage?.({
        data: { id: req.id, stripe, band: req.band! },
      } as MessageEvent<EvalResponse>);
    }
  }
  get pending(): number {
    return this.queue.length;
  }
}

describe("PoolBackend", () => {
  function makePool(size: number): {
    pool: PoolBackend;
    workers: FakePoolWorker[];
  } {
    const workers: FakePoolWorker[] = [];
    const pool = new PoolBackend(size, () => {
      const w = new FakePoolWorker();
      workers.push(w);
      return w as unknown as Worker;
    });
    return { pool, workers };
  }

  it("assembles fanned-out stripes into the full-grid maps", async () => {
    const { pool, workers } = makePool(4);
    const spec = specOf([
      {
        "basecolor.r": "fBm(base_freq=8, to_01=True, seed=7)",
        "emissive.g": "(X() * 2)",
        "emissive.strength": "4",
        height: "(fBm(base_freq=6, to_01=True, seed=7) * 5)",
      },
    ]);
    const p = pool.evaluate(spec);
    // Each worker received exactly its band.
    expect(workers.reduce((s, w) => s + w.pending, 0)).toBe(
      splitBands(spec.height, 4).length,
    );
    // Deliver stripes in a scrambled worker order to exercise out-of-order
    // assembly (the y0 carried in each reply must place them correctly).
    for (const w of [workers[2], workers[0], workers[3], workers[1]])
      w.flushAll();
    expectMapsEqual(await p, evalSpec(spec));
  });

  it("handles a grid shorter than the pool (idle workers)", async () => {
    const { pool, workers } = makePool(8);
    const spec = specOf([{ "basecolor.r": "0.5" }], {}, 6, 3); // 3 rows, 8 workers
    const p = pool.evaluate(spec);
    for (const w of workers) w.flushAll();
    expectMapsEqual(await p, evalSpec(spec));
    expect(pool.size).toBe(8);
  });

  it("dispose terminates every worker", () => {
    const { pool, workers } = makePool(3);
    pool.dispose();
    expect(workers.every((w) => w.terminated)).toBe(true);
  });
});
