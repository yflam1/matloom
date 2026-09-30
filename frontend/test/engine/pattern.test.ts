/** Tests for src/engine/pattern.ts — the Bricks lattice and the Weave pattern. */
import { describe, it, expect } from "vitest";
import { Bricks, Weave } from "../../src/engine/pattern";

const XS = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
const YS = new Float32Array([0, 0.5, 1]);

describe("Bricks", () => {
  it("is 1 inside a brick body", () => {
    expect(new Bricks().call(0.5, 0.25)).toBe(1);
  });

  it("is 0 in the mortar gap", () => {
    const b = new Bricks({ mortar: 0.1 });
    expect(b.call(0.01, 0.25)).toBe(0); // near the x=0 cell boundary
    expect(b.call(0.5, 0.49)).toBe(0); // near the y=0.5 cell boundary
  });

  it("running bond (offset=0.5) staggers alternating rows", () => {
    const b = new Bricks({ offset: 0.5, mortar: 0.1 });
    expect(b.call(0.5, 0.25)).toBe(1); // row 0 brick
    expect(b.call(0.5, 0.75)).toBe(0); // row 1 joint shifted to x=0.5
  });

  it("stacked bond (offset=0) aligns rows", () => {
    const b = new Bricks({ offset: 0, mortar: 0.1 });
    expect(b.call(0.5, 0.25)).toBe(1);
    expect(b.call(0.5, 0.75)).toBe(1);
  });

  it("axis=column offsets columns", () => {
    const b = new Bricks({
      brick_width: 0.5,
      brick_height: 1.0,
      offset: 0.5,
      mortar: 0.1,
      axis: "column",
    });
    expect(b.call(0.25, 0.5)).toBe(1);
    expect(b.call(0.75, 0.5)).toBe(0);
  });

  it("feather produces a smooth edge", () => {
    const b = new Bricks({ mortar: 0.1, feather: 0.1 });
    expect(b.call(0.05, 0.25)).toBeCloseTo(0.5, 12); // on the body boundary
    const mid = b.call(0.07, 0.25);
    expect(mid).toBeGreaterThan(0);
    expect(mid).toBeLessThan(1);
  });

  it("feather=0 is binary", () => {
    const b = new Bricks({ mortar: 0.1 });
    for (let i = 0; i <= 16; i++)
      for (let j = 0; j <= 12; j++) {
        const v = b.call((i / 16) * 2, (j / 12) * 2);
        expect(v === 0 || v === 1).toBe(true);
      }
  });

  it("grid matches scalar call", () => {
    for (const axis of ["row", "column"] as const)
      for (const feather of [0, 0.08]) {
        const b = new Bricks({
          brick_width: 0.7,
          brick_height: 0.3,
          offset: 0.5,
          mortar: 0.06,
          axis,
          feather,
        });
        const grid = b.evalGrid(XS, YS, XS.length, YS.length);
        for (let i = 0; i < YS.length; i++)
          for (let j = 0; j < XS.length; j++)
            expect(grid[i * XS.length + j]).toBeCloseTo(
              b.call(XS[j], YS[i]),
              6,
            );
      }
  });

  it("exposes resolved params for serialization", () => {
    const b = new Bricks({ brick_width: 2, axis: "column", feather: 0.02 });
    expect(b.exportParams.brick_width).toBe(2);
    expect(b.exportParams.axis).toBe("column");
    expect(b.exportParams.feather).toBe(0.02);
  });
});

describe("Weave", () => {
  it("is 1 on the on-top thread, 0 in the inter-thread gap", () => {
    const w = new Weave({ base_freq: 1, warp_width: 0.5 });
    expect(w.call(0.5, 0.5)).toBe(1); // thread centre
    expect(w.call(0.05, 0.5)).toBe(0); // gap
  });

  it("plain weave (1/1) puts the warp on top in cell (0,0)", () => {
    const w = new Weave({ over: 1, under: 1, base_freq: 1, warp_width: 1 });
    expect(w.call(0.5, 0.5)).toBe(1);
  });

  it("feather 0 is binary", () => {
    const w = new Weave({ over: 2, under: 1, base_freq: 6, warp_width: 0.6 });
    for (let x = 0; x <= 2; x += 0.13)
      for (let y = 0; y <= 2; y += 0.17) expect([0, 1]).toContain(w.call(x, y));
  });

  it("grid path matches the scalar call", () => {
    for (const shift of [0, 1, 2])
      for (const feather of [0, 0.15]) {
        const w = new Weave({
          over: 2,
          under: 1,
          shift,
          base_freq_x: 5,
          base_freq_y: 7,
          warp_width: 0.6,
          feather,
        });
        const grid = w.evalGrid(XS, YS, XS.length, YS.length);
        for (let i = 0; i < YS.length; i++)
          for (let j = 0; j < XS.length; j++)
            expect(grid[i * XS.length + j]).toBeCloseTo(w.call(XS[j], YS[i]), 6);
      }
  });

  it("exposes resolved params for serialization", () => {
    const w = new Weave({ over: 2, shift: 2, base_freq: 12, warp_width: 0.7 });
    expect(w.exportParams.over).toBe(2);
    expect(w.exportParams.shift).toBe(2);
    expect(w.exportParams.base_freq_x).toBe(12);
    expect(w.exportParams.warp_width).toBe(0.7);
  });
});
