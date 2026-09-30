/** Tests for src/viewer/framing.ts — relief Z extent + camera fit distance. */
import { describe, it, expect } from "vitest";
import { reliefZExtent, fitDistance } from "../../src/viewer/framing";

const F32 = (xs: number[]) => Float32Array.from(xs);

describe("reliefZExtent", () => {
  it("is flat when displacement is off", () => {
    expect(reliefZExtent(F32([1, 2, 3]), false)).toEqual({ zMin: 0, zMax: 0 });
  });

  it("is flat with no grid", () => {
    expect(reliefZExtent(null, true)).toEqual({ zMin: 0, zMax: 0 });
    expect(reliefZExtent(F32([]), true)).toEqual({ zMin: 0, zMax: 0 });
  });

  it("flips displacement to world Z (vertex.z = -displacement)", () => {
    // Displacement equals the composited height (coordinate units): heights
    // spanning -5 .. +20 give a surface spanning world Z -20 .. +5.
    expect(reliefZExtent(F32([-5, 10, 20, 0]), true)).toEqual({
      zMin: -20,
      zMax: 5,
    });
  });

  it("handles all-positive height (relief toward -Z)", () => {
    // A positive height of 10 displaces to world Z -10.
    expect(reliefZExtent(F32([10, 4, 10]), true)).toEqual({
      zMin: -10,
      zMax: -4,
    });
  });

  it("handles all-negative height (relief toward +Z)", () => {
    expect(reliefZExtent(F32([-10, -4, -10]), true)).toEqual({
      zMin: 4,
      zMax: 10,
    });
  });

  it("ignores a stray NaN, framing the finite samples", () => {
    expect(reliefZExtent(F32([1, NaN, 2]), true)).toEqual({
      zMin: -2,
      zMax: -1,
    });
  });

  it("falls back to flat when the extent is non-finite", () => {
    expect(reliefZExtent(F32([NaN, NaN]), true)).toEqual({ zMin: 0, zMax: 0 });
    expect(reliefZExtent(F32([1, Infinity]), true)).toEqual({
      zMin: 0,
      zMax: 0,
    });
  });
});

describe("fitDistance", () => {
  const base = { fov: 0.8, aspect: 1, margin: 1.5 };

  it("flat case matches the legacy rectangle fit", () => {
    const halfW = 0.5;
    const halfH = 0.5;
    const tan = Math.tan(base.fov / 2);
    const expected =
      Math.max(halfH / tan, halfW / (tan * base.aspect)) * base.margin;
    expect(fitDistance({ ...base, halfW, halfH, halfD: 0 })).toBeCloseTo(
      expected,
      12,
    );
  });

  it("a deep box backs the camera off more than a flat one", () => {
    const flat = fitDistance({ ...base, halfW: 0.5, halfH: 0.5, halfD: 0 });
    const deep = fitDistance({ ...base, halfW: 0.5, halfH: 0.5, halfD: 5 });
    expect(deep).toBeGreaterThan(flat);
  });

  it("grows monotonically with relief depth", () => {
    const d = (halfD: number) =>
      fitDistance({ ...base, halfW: 0.5, halfH: 0.5, halfD });
    expect(d(1)).toBeLessThan(d(5));
    expect(d(5)).toBeLessThan(d(20));
  });

  it("uses the limiting (smaller) FOV for a non-square viewport", () => {
    // Sphere fit with depth. A narrow (portrait) viewport has a *smaller*
    // horizontal FOV, so it limits → smaller half-FOV → larger distance. A wide
    // viewport's horizontal FOV exceeds the vertical one, so the vertical FOV
    // limits and the distance matches the square case.
    const box = { halfW: 0.5, halfH: 0.5, halfD: 5 };
    const square = fitDistance({ ...base, ...box, aspect: 1 });
    const narrow = fitDistance({ ...base, ...box, aspect: 0.5 });
    const wide = fitDistance({ ...base, ...box, aspect: 2 });
    expect(narrow).toBeGreaterThan(square);
    expect(wide).toBeCloseTo(square, 12);
  });

  it("never returns less than the floor", () => {
    expect(
      fitDistance({ ...base, halfW: 0, halfH: 0, halfD: 0 }),
    ).toBeGreaterThanOrEqual(0.1);
  });
});
