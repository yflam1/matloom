/** Tests for src/engine/shape.ts — Rect/Ellipse/Path shapes and Fill/Stroke. */
import { describe, it, expect } from "vitest";
import {
  Rect,
  Ellipse,
  Fill,
  Stroke,
  Path,
  lineTo,
  cubicTo,
} from "../../src/engine/shape";
import { Translate, Rotate } from "../../src/engine/transform";
import type { Expression2D } from "../../src/engine/expression";

const XS = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
const YS = new Float32Array([0, 0.5, 1]);

function expectGridMatchesCall(
  expr: Expression2D,
  xs: Float32Array,
  ys: Float32Array,
): void {
  const grid = expr.evalGrid(xs, ys, xs.length, ys.length);
  for (let i = 0; i < ys.length; i++)
    for (let j = 0; j < xs.length; j++)
      expect(grid[i * xs.length + j]).toBeCloseTo(expr.call(xs[j], ys[i]), 6);
}

describe("Fill", () => {
  it("rect: 1 inside, 0 outside", () => {
    const f = new Fill(new Rect(0.2, 0.2, 0.6, 0.6));
    expect(f.call(0.5, 0.5)).toBe(1);
    expect(f.call(0.05, 0.5)).toBe(0);
    expect(f.call(0.5, 0.95)).toBe(0);
  });

  it("ellipse: 1 inside, 0 outside", () => {
    const f = new Fill(new Ellipse(0.5, 0.5, 0.3, 0.2));
    expect(f.call(0.5, 0.5)).toBe(1);
    expect(f.call(0.5, 0.78)).toBe(0);
    expect(f.call(0.85, 0.5)).toBe(0);
  });

  it("feather is 0.5 on the boundary", () => {
    const f = new Fill(new Rect(0.2, 0.2, 0.6, 0.6), "NonZero", 0.1);
    expect(f.call(0.8, 0.5)).toBeCloseTo(0.5, 12);
  });

  it("rounded rect cuts the corner", () => {
    const sharp = new Fill(new Rect(0, 0, 1, 1));
    const round = new Fill(new Rect(0, 0, 1, 1, 0.3));
    expect(sharp.call(0.01, 0.01)).toBe(1);
    expect(round.call(0.01, 0.01)).toBe(0);
  });

  it("grid matches call (feather on and off)", () => {
    for (const feather of [0, 0.07]) {
      expectGridMatchesCall(
        new Fill(new Rect(0.2, 0.2, 0.6, 0.5), "NonZero", feather),
        XS,
        YS,
      );
      expectGridMatchesCall(
        new Fill(new Ellipse(0.5, 0.5, 0.3, 0.2), "NonZero", feather),
        XS,
        YS,
      );
    }
  });
});

describe("Stroke", () => {
  it("paints the boundary band", () => {
    const s = new Stroke(new Rect(0.2, 0.2, 0.6, 0.6), 0.08);
    expect(s.call(0.2, 0.5)).toBe(1);
    expect(s.call(0.5, 0.5)).toBe(0);
  });

  it("grid matches call", () => {
    expectGridMatchesCall(
      new Stroke(new Ellipse(0.5, 0.5, 0.3, 0.3), 0.08, 0.03),
      XS,
      YS,
    );
  });
});

describe("transformed shapes", () => {
  it("translate/rotate the filled field, grid matches call", () => {
    const star = new Fill(new Rect(0.2, 0.2, 0.4, 0.3), "NonZero", 0.04);
    expectGridMatchesCall(new Translate(star, 0.1, 0.2), XS, YS);
    expectGridMatchesCall(new Rotate(star, 33), XS, YS);
  });
});

describe("Path", () => {
  it("triangle fill is implicitly closed", () => {
    const tri = new Fill(
      new Path(0.2, 0.2, [lineTo(0.8, 0.2), lineTo(0.5, 0.8)]),
    );
    expect(tri.call(0.5, 0.4)).toBe(1);
    expect(tri.call(0.1, 0.1)).toBe(0);
    expect(tri.call(0.5, 0.9)).toBe(0);
  });

  it("needs at least one segment", () => {
    expect(() => new Path(0, 0, [])).toThrow(/at least one segment/);
  });

  it("stroke of an open path paints the segment", () => {
    const s = new Stroke(new Path(0.2, 0.5, [lineTo(0.8, 0.5)]), 0.06);
    expect(s.call(0.5, 0.5)).toBe(1);
    expect(s.call(0.5, 0.9)).toBe(0);
  });

  it("fill grid matches call (both fill rules, feather on/off)", () => {
    const p = new Path(0.2, 0.2, [
      lineTo(0.8, 0.3),
      cubicTo(0.7, 0.7, 0.4, 0.7, 0.3, 0.8),
    ]);
    for (const mode of ["NonZero", "EvenOdd"] as const)
      for (const feather of [0, 0.06])
        expectGridMatchesCall(new Fill(p, mode, feather), XS, YS);
  });

  it("stroke grid matches call", () => {
    const p = new Path(0.1, 0.5, [cubicTo(0.3, 0.9, 0.7, 0.1, 0.9, 0.5)]);
    expectGridMatchesCall(new Stroke(p, 0.06, 0.02), XS, YS);
  });
});
