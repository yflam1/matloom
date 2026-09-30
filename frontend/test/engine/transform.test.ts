/** Tests for src/engine/transform.ts — Translate, Scale and Rotate. */
import { describe, it, expect } from "vitest";
import { Translate, Scale, Rotate } from "../../src/engine/transform";
import {
  X,
  Y,
  Add,
  Mul,
  Sin,
  Threshold,
  Min,
} from "../../src/engine/expression";
import { fBm, Worley } from "../../src/engine/noise";
import { Bricks } from "../../src/engine/pattern";
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

describe("Translate", () => {
  it("shifts coordinates", () => {
    expect(new Translate(new X(), 0.3, 0).call(0.5, 0)).toBeCloseTo(0.2);
    expect(new Translate(new Y(), 0, 0.25).call(0, 1)).toBeCloseTo(0.75);
  });
  it("grid matches call", () => {
    expectGridMatchesCall(
      new Translate(Add(new X(), new Y()), 0.3, -0.2),
      XS,
      YS,
    );
  });
  it("exposes params", () => {
    expect(new Translate(new X(), 0.3, -0.2).exportParams).toEqual({
      dx: 0.3,
      dy: -0.2,
    });
  });
});

describe("Scale", () => {
  it("scales about the origin", () => {
    expect(new Scale(new X(), 2, 1).call(1, 0)).toBeCloseTo(0.5);
  });
  it("negative factor mirrors", () => {
    expect(new Scale(new X(), -1, 1).call(0.3, 0)).toBeCloseTo(-0.3);
  });
  it("rejects a zero factor", () => {
    expect(() => new Scale(new X(), 0, 1)).toThrow(/non-zero/);
    expect(() => new Scale(new X(), 1, 0)).toThrow(/non-zero/);
  });
  it("grid matches call", () => {
    expectGridMatchesCall(
      new Scale(new Sin(Mul(new X(), new Y())), 2, 0.5),
      XS,
      YS,
    );
  });
});

describe("Rotate", () => {
  it("rotating the X-gradient 90 CCW reads the Y coordinate", () => {
    const e = new Rotate(new X(), 90);
    expect(e.call(1, 0)).toBeCloseTo(0, 9);
    expect(e.call(0, 1)).toBeCloseTo(1, 9);
  });

  it("zero degrees is identity", () => {
    expect(new Rotate(Add(new X(), new Y()), 0).call(0.3, 0.4)).toBeCloseTo(
      0.7,
    );
  });

  it("grid matches call over many node types", () => {
    const nodes: Expression2D[] = [
      Add(new X(), new Y()),
      new Sin(new X()),
      Mul(new X(), new Y()),
      new Threshold(new X(), { below_at: 0.3, above_at: 0.7 }),
      new Bricks({ mortar: 0.1 }),
      new Min(new X(), new Y()),
      new Translate(Add(new X(), new Y()), 0.2, -0.1),
      new Scale(new Sin(new X()), 2, 0.5),
    ];
    for (const node of nodes)
      expectGridMatchesCall(new Rotate(node, 37), XS, YS);
  });

  it("noise rotates per cell, staying grid==call consistent", () => {
    expectGridMatchesCall(new Rotate(new fBm({ seed: 5 }), 30), XS, YS);
    expectGridMatchesCall(new Rotate(new Worley({ seed: 3 }), 25), XS, YS);
  });

  it("nested rotation composes", () => {
    expectGridMatchesCall(
      new Rotate(new Rotate(Add(new X(), new Y()), 20), 25),
      XS,
      YS,
    );
  });

  it("exposes the angle for serialization", () => {
    expect(new Rotate(new X(), 45).exportParams).toEqual({ degrees: 45 });
  });
});
