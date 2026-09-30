/** Tests for src/engine/compositing.ts — alpha resolution and weighted blend. */
import { describe, it, expect } from "vitest";
import { processAlphas, weightedBlend } from "../../src/engine/compositing";
import type { Grid } from "../../src/engine/grid";

const g = (...v: number[]): Grid => Float32Array.from(v);

describe("processAlphas (over operator, bottom-most first)", () => {
  it("two opaque layers: top takes all weight", () => {
    // layers ordered bottom-most first: [bottom, top]
    const { alpha, Vs } = processAlphas([g(1), g(1)]);
    expect(alpha[0]).toBeCloseTo(1);
    expect(Vs[1][0]).toBeCloseTo(1); // top fully visible
    expect(Vs[0][0]).toBeCloseTo(0); // bottom occluded
  });

  it("half-transparent top over opaque bottom", () => {
    const { alpha, Vs } = processAlphas([g(1), g(0.5)]);
    expect(alpha[0]).toBeCloseTo(1);
    expect(Vs[1][0]).toBeCloseTo(0.5);
    expect(Vs[0][0]).toBeCloseTo(0.5);
  });

  it("single fully transparent layer has zero coverage", () => {
    const { alpha, Vs } = processAlphas([g(0)]);
    expect(alpha[0]).toBeCloseTo(0);
    expect(Vs[0][0]).toBeCloseTo(0);
  });

  it("coverage follows 1 - prod(1 - a_i)", () => {
    const { alpha } = processAlphas([g(0.5), g(0.5)]);
    // 1 - (1-0.5)(1-0.5) = 0.75
    expect(alpha[0]).toBeCloseTo(0.75);
  });
});

describe("weightedBlend", () => {
  it("alpha-weighted average of channel grids", () => {
    // Two layers, alpha computed from those weights.
    const { alpha, Vs } = processAlphas([g(1), g(0.5)]);
    // bottom channel = 0.2, top channel = 0.8
    const out = weightedBlend([g(0.2), g(0.8)], alpha, Vs);
    // (V_bottom*0.2 + V_top*0.8)/alpha = (0.5*0.2 + 0.5*0.8)/1 = 0.5
    expect(out[0]).toBeCloseTo(0.5);
  });

  it("zero coverage yields zero", () => {
    const { alpha, Vs } = processAlphas([g(0)]);
    const out = weightedBlend([g(0.7)], alpha, Vs);
    expect(out[0]).toBe(0);
  });

  it("single opaque layer passes the channel through", () => {
    const { alpha, Vs } = processAlphas([g(1)]);
    const out = weightedBlend([g(0.42)], alpha, Vs);
    expect(out[0]).toBeCloseTo(0.42);
  });
});
