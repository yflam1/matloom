/** Tests for src/engine/noise.ts — fBm, Worley, seed lifecycle, collectNoise. */
import { describe, it, expect } from "vitest";
import { fBm, Worley, collectNoise } from "../../src/engine/noise";
import { Add, Sin, X, Constant } from "../../src/engine/expression";

const XS = new Float32Array([0, 0.5, 1, 1.5, 2]);
const YS = new Float32Array([0, 0.5, 1, 1.5]);

describe("fBm", () => {
  it("is deterministic for a fixed seed", () => {
    expect(new fBm({ seed: 42 }).call(0.1, 0.2)).toBe(
      new fBm({ seed: 42 }).call(0.1, 0.2),
    );
  });

  it("differs across seeds", () => {
    expect(new fBm({ seed: 1 }).call(0.1, 0.2)).not.toBe(
      new fBm({ seed: 2 }).call(0.1, 0.2),
    );
  });

  it("default output is signed and within [-1, 1]", () => {
    const f = new fBm({ seed: 7 });
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < 20; i++)
      for (let j = 0; j < 20; j++) {
        const v = f.call(i * 0.123, j * 0.321);
        min = Math.min(min, v);
        max = Math.max(max, v);
      }
    expect(min).toBeGreaterThanOrEqual(-1);
    expect(max).toBeLessThanOrEqual(1);
    expect(min).toBeLessThan(0);
  });

  it("to_01 stays within [0, 1]", () => {
    const f = new fBm({ seed: 7, to_01: true });
    for (let i = 0; i < 20; i++)
      for (let j = 0; j < 20; j++) {
        const v = f.call(i * 0.123, j * 0.321);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
  });

  it("grid matches scalar call", () => {
    const f = new fBm({ seed: 99 });
    const grid = f.evalGrid(XS, YS, XS.length, YS.length);
    for (let i = 0; i < YS.length; i++)
      for (let j = 0; j < XS.length; j++)
        expect(grid[i * XS.length + j]).toBeCloseTo(f.call(XS[j], YS[i]), 6);
  });

  it("matches Python fBm(seed=5)(0.1,0.2) exactly", () => {
    // Reference: matloom.engine.noise.fBm(seed=5)(0.1, 0.2) (octaves=6 default).
    expect(new fBm({ seed: 5 }).call(0.1, 0.2)).toBe(0.36799036667171914);
  });

  it("memoizes the grid for an unchanged sample region", () => {
    const f = new fBm({ seed: 5 });
    const a = f.evalGrid(XS, YS, XS.length, YS.length);
    const b = f.evalGrid(XS, YS, XS.length, YS.length);
    expect(a).toBe(b); // same instance returned on cache hit
  });

  it("exposes resolved params with seed excluded from the signature", () => {
    const f = new fBm({ octaves: 4, base_freq: 2, seed: 1 });
    expect(f.exportParams.octaves).toBe(4);
    expect(f.exportParams.base_freq_x).toBe(2);
    expect(f.argSignature()).toContain("fBm|4");
  });
});

describe("Worley", () => {
  it("is deterministic for a fixed seed", () => {
    expect(new Worley({ seed: 42 }).call(0.1, 0.2)).toBe(
      new Worley({ seed: 42 }).call(0.1, 0.2),
    );
  });

  it("grid matches scalar call across all modes", () => {
    for (const distance of ["euclidean", "manhattan", "chebyshev"] as const)
      for (const combination of ["F1", "F2", "F2-F1", "F2+F1"] as const) {
        const w = new Worley({ distance, combination, seed: 3 });
        const grid = w.evalGrid(XS, YS, XS.length, YS.length);
        for (let i = 0; i < YS.length; i++)
          for (let j = 0; j < XS.length; j++)
            expect(grid[i * XS.length + j]).toBeCloseTo(
              w.call(XS[j], YS[i]),
              5,
            );
      }
  });

  it("F1 is non-negative", () => {
    const w = new Worley({ combination: "F1", seed: 5 });
    for (let i = 0; i < 15; i++)
      for (let j = 0; j < 15; j++)
        expect(w.call(i * 0.21, j * 0.21)).toBeGreaterThanOrEqual(0);
  });

  it("F2-F1 is non-negative", () => {
    const w = new Worley({ combination: "F2-F1", seed: 5 });
    for (let i = 0; i < 15; i++)
      for (let j = 0; j < 15; j++)
        expect(w.call(i * 0.21, j * 0.21)).toBeGreaterThanOrEqual(0);
  });

  it("to_01 stays within [0, 1]", () => {
    const w = new Worley({ seed: 5, to_01: true });
    for (let i = 0; i < 20; i++)
      for (let j = 0; j < 20; j++) {
        const v = w.call(i * 0.21, j * 0.21);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
  });
});

describe("seed lifecycle", () => {
  it("auto-seeds when no seed is given", () => {
    const f = new fBm({});
    expect(f.autoSeeded).toBe(true);
    expect(Number.isInteger(f.currentSeed)).toBe(true);
  });

  it("a pinned seed is not auto-seeded and never re-rolls", () => {
    const f = new fBm({ seed: 5 });
    expect(f.autoSeeded).toBe(false);
    expect(f.reseed()).toBe(false);
    expect(f.currentSeed).toBe(5);
  });

  it("reseed changes an auto seed and reports the change", () => {
    const f = new fBm({});
    const before = f.currentSeed;
    const changed = f.reseed();
    expect(changed).toBe(true);
    expect(f.currentSeed).not.toBe(before);
  });

  it("restoreSeed reattaches a remembered auto seed", () => {
    const f = new fBm({});
    f.restoreSeed(12345);
    expect(f.currentSeed).toBe(12345);
  });

  it("restoreSeed ignores pinned nodes", () => {
    const f = new fBm({ seed: 5 });
    f.restoreSeed(999);
    expect(f.currentSeed).toBe(5);
  });

  it("argSignature is seed-independent", () => {
    const a = new fBm({ base_freq: 2 });
    const b = new fBm({ base_freq: 2, seed: 77 });
    expect(a.argSignature()).toBe(b.argSignature());
  });
});

describe("collectNoise", () => {
  it("finds noise nodes nested in an expression tree", () => {
    const n1 = new fBm({ seed: 1 });
    const n2 = new Worley({ seed: 2 });
    const expr = Add(new Sin(n1), n2);
    const found = collectNoise(expr);
    expect(found).toContain(n1);
    expect(found).toContain(n2);
    expect(found).toHaveLength(2);
  });

  it("returns empty for noise-free expressions", () => {
    expect(collectNoise(Add(new X(), new Constant(1)))).toHaveLength(0);
  });
});
