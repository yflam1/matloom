/** Tests for src/engine/grid.ts — grid primitives and color conversions. */
import { describe, it, expect } from "vitest";
import {
  makeGrid,
  gridFull,
  gridFromX,
  gridFromY,
  gridMap,
  gridAdd,
  gridSub,
  gridMul,
  gridDiv,
  gridPow,
  gridMin,
  gridMax,
  linearToSrgbScalar,
  srgbToLinearScalar,
  toByte,
} from "../../src/engine/grid";

describe("grid constructors", () => {
  it("makeGrid fills from index", () => {
    expect(Array.from(makeGrid(3, (i) => i * 2))).toEqual([0, 2, 4]);
  });

  it("gridFull fills a constant", () => {
    expect(Array.from(gridFull(3, 0.5))).toEqual([0.5, 0.5, 0.5]);
  });

  it("gridFromX broadcasts columns across rows", () => {
    const xs = new Float32Array([0, 1]);
    // w=2, h=2 -> row-major [x0,x1, x0,x1]
    expect(Array.from(gridFromX(xs, 2, 2))).toEqual([0, 1, 0, 1]);
  });

  it("gridFromY broadcasts rows across columns", () => {
    const ys = new Float32Array([0, 1]);
    // w=2, h=2 -> [y0,y0, y1,y1]
    expect(Array.from(gridFromY(ys, 2, 2))).toEqual([0, 0, 1, 1]);
  });
});

describe("grid element-wise ops", () => {
  const a = new Float32Array([1, 2, 3]);
  const b = new Float32Array([4, 5, 6]);

  it("gridMap", () => {
    expect(Array.from(gridMap(a, (v) => v + 1))).toEqual([2, 3, 4]);
  });
  it("gridAdd grid + grid", () => {
    expect(Array.from(gridAdd(a, b))).toEqual([5, 7, 9]);
  });
  it("gridAdd grid + scalar", () => {
    expect(Array.from(gridAdd(a, 10))).toEqual([11, 12, 13]);
  });
  it("gridSub", () => {
    expect(Array.from(gridSub(b, a))).toEqual([3, 3, 3]);
  });
  it("gridMul", () => {
    expect(Array.from(gridMul(a, 2))).toEqual([2, 4, 6]);
  });
  it("gridDiv", () => {
    expect(Array.from(gridDiv(b, 2))).toEqual([2, 2.5, 3]);
  });
  it("gridPow", () => {
    expect(Array.from(gridPow(a, 2))).toEqual([1, 4, 9]);
  });
  it("gridMin/gridMax with scalar", () => {
    expect(Array.from(gridMin(b, 5))).toEqual([4, 5, 5]);
    expect(Array.from(gridMax(a, 2))).toEqual([2, 2, 3]);
  });
  it("gridMin/gridMax with grid", () => {
    expect(Array.from(gridMin(a, b))).toEqual([1, 2, 3]);
    expect(Array.from(gridMax(a, b))).toEqual([4, 5, 6]);
  });
});

describe("color-space scalar conversions", () => {
  it("endpoints map to themselves", () => {
    expect(srgbToLinearScalar(0)).toBeCloseTo(0, 12);
    expect(srgbToLinearScalar(1)).toBeCloseTo(1, 12);
    expect(linearToSrgbScalar(0)).toBeCloseTo(0, 12);
    expect(linearToSrgbScalar(1)).toBeCloseTo(1, 12);
  });

  it("uses the linear segment below the cutoff", () => {
    expect(srgbToLinearScalar(0.04)).toBeCloseTo(0.04 / 12.92, 12);
    expect(linearToSrgbScalar(0.003)).toBeCloseTo(0.003 * 12.92, 12);
  });

  it("round-trips away from the seam", () => {
    for (const v of [0, 0.01, 0.2, 0.5, 0.9, 1]) {
      expect(linearToSrgbScalar(srgbToLinearScalar(v))).toBeCloseTo(v, 12);
    }
  });
});

describe("toByte quantizer", () => {
  it("maps [0,1] to [0,255] with rounding", () => {
    expect(toByte(0)).toBe(0);
    expect(toByte(1)).toBe(255);
    expect(toByte(0.4)).toBe(102); // 0.4*255 + 0.5 = 102.5 -> 102
  });
  it("clamps out-of-range", () => {
    expect(toByte(-1)).toBe(0);
    expect(toByte(2)).toBe(255);
  });
  it("rounds half up via +0.5 floor", () => {
    // 0.5 -> 127.5 + 0.5 = 128 -> 128
    expect(toByte(0.5)).toBe(128);
    // 200/255 ≈ 0.7843 -> *255=200 +0.5 =200.5 -> 200
    expect(toByte(200 / 255)).toBe(200);
  });
});
