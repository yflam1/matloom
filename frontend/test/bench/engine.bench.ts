/**
 * engine.bench.ts — a profiling harness for the material engine's grid
 * evaluation. NOT a unit test: it is excluded from `vitest run` (it does not
 * match `*.test.ts`) and is run on demand with `npm run bench`
 * (`vite-node test/bench/engine.bench.ts`).
 *
 * It builds a handful of representative materials and times
 * `evaluateMaterial()` at preview (256²) and full (512²) resolution under two
 * regimes:
 *
 *   - cold: a fresh expression tree per run, so the noise memo (keyed by grid
 *     extent — see noise.ts) never hits. Measures the full per-pixel kernel +
 *     allocation cost a noise-axis / resolution change incurs.
 *   - warm: one tree, evaluated repeatedly, so noise grids are memoized.
 *     Measures the compositing + non-noise channel cost a typical edit hits.
 *
 * Use it to compare a baseline against an optimization (Step 0 ↔ Step 1+ of the
 * eval-acceleration work): capture the table, make the change, re-run, diff.
 */
import {
  Color,
  Emissive,
  Layer,
  evaluateMaterial,
  type EvalRegion,
} from "../../src/engine";
import { parseExpr } from "../../src/expr-lang";

const p = (src: string) => parseExpr(src);

// Each entry is a *factory*: calling it builds a brand-new layer stack (fresh
// expression nodes, fresh noise memos), which is what the cold regime needs.
type Material = { name: string; build: () => Layer[] };

const MATERIALS: Material[] = [
  {
    // Constant channels: isolates pure compositing + quantization overhead.
    name: "flat (constants)",
    build: () => [
      new Layer({
        alpha: p("1"),
        basecolor: new Color({ r: p("0.8"), g: p("0.4"), b: p("0.2") }),
        metallic: p("0"),
        roughness: p("0.5"),
        height: p("0"),
      }),
    ],
  },
  {
    // One fBm driving several channels: the noise-heavy case.
    name: "noise (fBm)",
    build: () => [
      new Layer({
        alpha: p("1"),
        basecolor: new Color({
          r: p("fBm(base_freq=8, to_01=True)"),
          g: p("fBm(base_freq=8, to_01=True)"),
          b: p("fBm(base_freq=8, to_01=True)"),
        }),
        roughness: p(
          "Threshold(fBm(base_freq=16, to_01=True), below_at=0.4, above_at=0.6)",
        ),
        height: p("(fBm(base_freq=8, to_01=True) * 5)"),
      }),
    ],
  },
  {
    // Bricks lattice + threshold: pure Floor/arithmetic, no transcendentals.
    name: "bricks (arithmetic)",
    build: () => [
      new Layer({
        alpha: p("1"),
        basecolor: new Color({
          r: p("Bricks(brick_width=0.25, brick_height=0.12)"),
          g: p("(Bricks(brick_width=0.25, brick_height=0.12) * 0.5)"),
          b: p("0.2"),
        }),
        roughness: p("(1 - Bricks(brick_width=0.25, brick_height=0.12))"),
        height: p("(Bricks(brick_width=0.25, brick_height=0.12) * 3)"),
      }),
    ],
  },
  {
    // A heavier arithmetic expression tree (many binary nodes per channel),
    // which is exactly what kernel fusion targets.
    name: "deep arithmetic",
    build: () => {
      const expr =
        "((Sin((X() * 12)) * Cos((Y() * 12))) * 0.5 + (((X() - 0.5) ** 2) + ((Y() - 0.5) ** 2)))";
      return [
        new Layer({
          alpha: p("1"),
          basecolor: new Color({ r: p(expr), g: p(expr), b: p(expr) }),
          roughness: p(expr),
          metallic: p(expr),
          height: p(`(${expr} * 4)`),
        }),
      ];
    },
  },
  {
    // Three stacked layers with masks + emissive: a realistic composite.
    name: "multi-layer composite",
    build: () => [
      new Layer({
        alpha: p("1"),
        basecolor: new Color({ r: p("0.3"), g: p("0.3"), b: p("0.35") }),
        roughness: p("0.7"),
        height: p("0"),
      }),
      new Layer({
        alpha: p(
          "Threshold(fBm(base_freq=6, to_01=True), below_at=0.5, below_to=0, above_at=0.5, above_to=1)",
        ),
        basecolor: new Color({ r: p("0.6"), g: p("0.5"), b: p("0.1") }),
        roughness: p("0.4"),
        height: p("(fBm(base_freq=6, to_01=True) * 2)"),
      }),
      new Layer({
        alpha: p("Fill(Ellipse(0.5, 0.5, 0.2, 0.2))"),
        basecolor: new Color({ r: p("0.9"), g: p("0.1"), b: p("0.1") }),
        emissive: new Emissive({
          r: p("1"),
          g: p("0.2"),
          b: p("0.2"),
          strength: p("3"),
        }),
        roughness: p("0.2"),
        height: p("4"),
      }),
    ],
  },
];

const REGION: EvalRegion = {
  x1: 0,
  y1: 0,
  x2: 1,
  y2: 1,
  yUp: true,
};

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function timeCold(
  build: () => Layer[],
  w: number,
  h: number,
  runs: number,
): number[] {
  const out: number[] = [];
  for (let i = 0; i < runs; i++) {
    const layers = build(); // fresh tree → cold noise memo
    const t0 = performance.now();
    evaluateMaterial(layers, w, h, REGION);
    out.push(performance.now() - t0);
  }
  return out;
}

function timeWarm(
  build: () => Layer[],
  w: number,
  h: number,
  runs: number,
): number[] {
  const layers = build();
  evaluateMaterial(layers, w, h, REGION); // prime the memo
  const out: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    evaluateMaterial(layers, w, h, REGION);
    out.push(performance.now() - t0);
  }
  return out;
}

function fmt(ms: number): string {
  return ms.toFixed(2).padStart(8);
}

function main(): void {
  const RUNS = 25;
  const sizes: [string, number, number][] = [
    ["preview 256²", 256, 256],
    ["full 512²", 512, 512],
  ];

  // Warm up the JIT before any measurement.
  for (const m of MATERIALS) timeCold(m.build, 128, 128, 3);

  console.log(
    "\nmaterial                     resolution      cold(med/min ms)    warm(med/min ms)",
  );
  console.log("-".repeat(86));
  for (const m of MATERIALS) {
    for (const [label, w, h] of sizes) {
      const cold = timeCold(m.build, w, h, RUNS);
      const warm = timeWarm(m.build, w, h, RUNS);
      console.log(
        `${m.name.padEnd(28)} ${label.padEnd(14)} ` +
          `${fmt(median(cold))}/${fmt(Math.min(...cold))}   ` +
          `${fmt(median(warm))}/${fmt(Math.min(...warm))}`,
      );
    }
  }
  console.log("");
}

main();
