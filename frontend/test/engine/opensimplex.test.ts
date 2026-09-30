/** Tests for src/engine/opensimplex.ts — the OpenSimplex noise port. */
import { describe, it, expect } from "vitest";
import { openSimplex } from "../../src/engine/opensimplex";

describe("openSimplex", () => {
  it("is deterministic for a fixed seed", () => {
    const a = openSimplex(5);
    const b = openSimplex(5);
    expect(a.noise2(0.1, 0.2)).toBe(b.noise2(0.1, 0.2));
  });

  it("produces different fields for different seeds", () => {
    expect(openSimplex(1).noise2(0.1, 0.2)).not.toBe(
      openSimplex(2).noise2(0.1, 0.2),
    );
  });

  it("returns 0 at the origin (lattice point)", () => {
    // At an integer lattice point the simplex contributions cancel to 0.
    expect(openSimplex(5).noise2(0, 0)).toBeCloseTo(0, 12);
  });

  it("stays within the normalized [-1, 1] range", () => {
    const s = openSimplex(7);
    for (let i = 0; i < 30; i++) {
      for (let j = 0; j < 30; j++) {
        const v = s.noise2(i * 0.137, j * 0.211);
        expect(v).toBeGreaterThanOrEqual(-1);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  it("noise2array matches per-point noise2", () => {
    const s = openSimplex(3);
    const xs = new Float64Array([0.1, 0.5, 0.9]);
    const ys = new Float64Array([0.2, 0.6]);
    const arr = s.noise2array(xs, ys);
    for (let yi = 0; yi < ys.length; yi++) {
      for (let xi = 0; xi < xs.length; xi++) {
        expect(arr[yi * xs.length + xi]).toBe(s.noise2(xs[xi], ys[yi]));
      }
    }
  });

  it("matches known Python opensimplex values (byte-for-byte parity)", () => {
    // Reference values from the Python `opensimplex` package:
    //   OpenSimplex(5).noise2(0.1, 0.2) == 0.28038714071265863
    //   OpenSimplex(5).noise2(0.5, 0.5) == 0.546821875644804
    // The BigInt seed expansion must reproduce these bit-for-bit.
    const s = openSimplex(5);
    expect(s.noise2(0.1, 0.2)).toBe(0.28038714071265863);
    expect(s.noise2(0.5, 0.5)).toBe(0.546821875644804);
  });
});
