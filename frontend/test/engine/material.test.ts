/** Tests for src/engine/material.ts — evaluateMaterial compositing pipeline. */
import { describe, it, expect } from "vitest";
import { evaluateMaterial } from "../../src/engine/material";
import { Layer, Color, Emissive } from "../../src/engine/layer";
import {
  Constant,
  MultiplicationExpression2D,
  Threshold,
  X,
  Y,
} from "../../src/engine/expression";

describe("evaluateMaterial", () => {
  it("produces correctly sized maps (row-major w*h)", () => {
    const maps = evaluateMaterial([new Layer()], 4, 3);
    expect(maps.basecolor).toHaveLength(4 * 3 * 4); // RGBA8
    expect(maps.orm).toHaveLength(4 * 3 * 4);
    expect(maps.emissive).toHaveLength(4 * 3 * 3); // RGB8
    expect(maps.normal).toHaveLength(4 * 3 * 4); // RGBA8
  });

  it("renders a solid red opaque layer", () => {
    const red = new Layer({ basecolor: new Color({ r: new Constant(1) }) });
    const maps = evaluateMaterial([red], 2, 2);
    for (let i = 0; i < 4; i++) {
      expect(maps.basecolor[i * 4 + 0]).toBe(255); // R
      expect(maps.basecolor[i * 4 + 1]).toBe(0); // G
      expect(maps.basecolor[i * 4 + 2]).toBe(0); // B
      expect(maps.basecolor[i * 4 + 3]).toBe(255); // A
    }
  });

  it("packs ORM as AO(red) / roughness(green) / metallic(blue); flat → AO 255", () => {
    const l = new Layer({
      roughness: new Constant(1),
      metallic: new Constant(1),
    });
    const maps = evaluateMaterial([l], 2, 2);
    for (let i = 0; i < 4; i++) {
      expect(maps.orm[i * 4 + 0]).toBe(255); // AO (flat surface = fully lit)
      expect(maps.orm[i * 4 + 1]).toBe(255); // roughness
      expect(maps.orm[i * 4 + 2]).toBe(255); // metallic
      expect(maps.orm[i * 4 + 3]).toBe(255);
    }
  });

  it("packs OpenPBR channels into pbr1/pbr2", () => {
    const l = new Layer({
      sheen: new Constant(1),
      coat: new Constant(0),
      transmission: new Constant(1),
      subsurface: new Constant(0),
      anisotropy: new Constant(1),
      ior: new Constant(3), // (3-1)/(3-1)=1 → 255
    });
    const maps = evaluateMaterial([l], 2, 2);
    expect(maps.pbr1).toHaveLength(2 * 2 * 4);
    expect(maps.pbr2).toHaveLength(2 * 2 * 4);
    for (let i = 0; i < 4; i++) {
      expect(maps.pbr1[i * 4 + 0]).toBe(255); // sheen
      expect(maps.pbr1[i * 4 + 1]).toBe(0); // coat
      expect(maps.pbr1[i * 4 + 2]).toBe(255); // transmission
      expect(maps.pbr1[i * 4 + 3]).toBe(0); // subsurface
      expect(maps.pbr2[i * 4 + 0]).toBe(255); // anisotropy
      expect(maps.pbr2[i * 4 + 1]).toBe(255); // ior → max
      expect(maps.pbr2[i * 4 + 3]).toBe(255);
    }
  });

  it("a default layer packs inert OpenPBR values (zero weights, ior 1.5 → 64)", () => {
    const maps = evaluateMaterial([new Layer()], 2, 2);
    for (let i = 0; i < 4; i++) {
      expect(maps.pbr1[i * 4 + 0]).toBe(0);
      expect(maps.pbr1[i * 4 + 2]).toBe(0);
      expect(maps.pbr2[i * 4 + 0]).toBe(0);
      expect(maps.pbr2[i * 4 + 1]).toBe(64); // (1.5-1)/(3-1)=0.25 → 64
    }
  });

  it("a flat material has flat normals (128,128,255) and zero displacement", () => {
    const maps = evaluateMaterial([new Layer()], 2, 2);
    for (let i = 0; i < 4; i++) {
      expect(maps.normal[i * 4 + 0]).toBe(128); // nx → 0.5
      expect(maps.normal[i * 4 + 1]).toBe(128); // ny → 0.5
      expect(maps.normal[i * 4 + 2]).toBe(255); // nz → 1
      expect(maps.normal[i * 4 + 3]).toBe(255); // alpha unused (opaque)
      expect(maps.displacement[i]).toBe(0);
    }
  });

  it("composites height with max: a masked tall layer over a short base", () => {
    // Base covers everything at h=1; the top layer is present on the left half
    // (alpha = 1 where X ≤ 0.5) at h=9. Displacement equals the composited height
    // directly (coordinate units): 9 on the left (top present), 1 on the right.
    const base = new Layer({ height: new Constant(1) });
    const top = new Layer({
      // alpha: 1 for x ≤ 0.5, 0 for x > 0.5 (a hard vertical step).
      alpha: new Threshold(new X(), {
        below_at: 0.5,
        below_to: 1,
        above_at: 0.5,
        above_to: 0,
      }),
      height: new Constant(9),
    });
    const maps = evaluateMaterial([base, top], 4, 1, {
      x1: 0,
      x2: 1,
    });
    expect(maps.displacement[0]).toBeCloseTo(9, 6); // left: top present (h=9)
    expect(maps.displacement[3]).toBeCloseTo(1, 6); // right: base only (h=1)
    // The height step produces a non-flat normal somewhere along the row.
    let tilted = false;
    for (let i = 0; i < 4; i++) if (maps.normal[i * 4] !== 128) tilted = true;
    expect(tilted).toBe(true);
  });

  it("AO is crop-invariant: the world-fixed blur holds the step darkening steady", () => {
    // A hard height step at X=0.5. Rendering a crop (a sub-region at the same
    // resolution) shrinks the world texel size, so a fixed-texel AO blur darkens
    // the step — the reported regression. Passing the authored span as the relief
    // reference scales the blur radius with the zoom, holding the world radius
    // (and thus the darkening) constant.
    const W = 64;
    // A shallow step (depth 0.05) so the cavity stays unsaturated — a deep step
    // clamps AO to 0 on both crops and the ordering would be hidden.
    const stepLayers = () => [
      new Layer({ height: new Constant(0) }),
      new Layer({
        alpha: new Threshold(new X(), {
          below_at: 0.5,
          below_to: 1,
          above_at: 0.5,
          above_to: 0,
        }),
        height: new Constant(0.05),
      }),
    ];
    const aoMin = (m: { orm: Uint8Array }) => {
      let lo = 255;
      for (let i = 0; i < W; i++) lo = Math.min(lo, m.orm[i * 4]);
      return lo;
    };

    // Full authored view, and a 2× crop of the middle (same resolution).
    const full = aoMin(evaluateMaterial(stepLayers(), W, 1, { x1: 0, x2: 1 }));
    const cropWorld = aoMin(
      // reliefSpanX = authored span (1) → blur radius scales with the 2× zoom.
      evaluateMaterial(stepLayers(), W, 1, {
        x1: 0.25,
        x2: 0.75,
        reliefSpanX: 1,
      }),
    );
    const cropNaive = aoMin(
      // No reference span → relief uses the rendered span (the old behavior).
      evaluateMaterial(stepLayers(), W, 1, { x1: 0.25, x2: 0.75 }),
    );

    // The old fixed-texel behavior darkens the crop; the world-fixed one does not.
    expect(cropNaive).toBeLessThan(full);
    expect(cropWorld).toBeGreaterThan(cropNaive);
    // World-fixed crop stays close to the full-view darkening (small discrete
    // residual only) — i.e. AO is effectively crop-invariant.
    expect(Math.abs(cropWorld - full)).toBeLessThanOrEqual(8);
  });

  it("a fully transparent layer yields zero coverage", () => {
    const maps = evaluateMaterial(
      [new Layer({ alpha: new Constant(0) })],
      2,
      2,
    );
    for (let i = 0; i < 4; i++) expect(maps.basecolor[i * 4 + 3]).toBe(0);
  });

  it("reports emissiveIntensity >= 1 and normalizes emissive bytes", () => {
    const l = new Layer({
      emissive: new Emissive({ r: new Constant(1), strength: new Constant(4) }),
    });
    const maps = evaluateMaterial([l], 2, 2);
    expect(maps.emissiveIntensity).toBeGreaterThanOrEqual(1);
    // Normalized so the brightest channel maps to 255.
    let max = 0;
    for (const v of maps.emissive) max = Math.max(max, v);
    expect(max).toBe(255);
  });

  it("defaults emissiveIntensity to 1 when there is no emission", () => {
    const maps = evaluateMaterial([new Layer()], 2, 2);
    expect(maps.emissiveIntensity).toBe(1);
  });

  it("yUp reversal flips the row order of a Y-dependent channel", () => {
    // alpha = Y(): row 0 samples the max Y when yUp, the min Y when yDown.
    const l = new Layer({ alpha: new Y() });
    const up = evaluateMaterial([l], 1, 2, { y1: 0, y2: 1, yUp: true });
    const down = evaluateMaterial([l], 1, 2, { y1: 0, y2: 1, yUp: false });
    // Row 0 alpha (byte index 3) is larger under yUp (top = max Y).
    expect(up.basecolor[3]).toBeGreaterThan(down.basecolor[3]);
    // The two are row-reversed: up row0 == down row1 and vice versa.
    expect(up.basecolor[3]).toBe(down.basecolor[1 * 4 + 3]);
  });

  // A spatially-uniform (constant) height field exercises the relief fast path
  // in evaluateMaterial, which skips the AO box blur and normal central
  // differences. The result must be byte-identical to the full derivation: AO
  // 255 (no cavities), flat normal (128,128,255), and uniform displacement.
  it("a non-zero constant height stays flat (fast-path AO/normal)", () => {
    const maps = evaluateMaterial(
      [new Layer({ height: new Constant(5) })],
      8,
      8,
    );
    for (let i = 0; i < 64; i++) {
      expect(maps.orm[i * 4 + 0]).toBe(255); // AO: no cavity on a flat surface
      expect(maps.normal[i * 4 + 0]).toBe(128); // nx → 0
      expect(maps.normal[i * 4 + 1]).toBe(128); // ny → 0
      expect(maps.normal[i * 4 + 2]).toBe(255); // nz → 1
      expect(maps.displacement[i]).toBe(5); // displacement = height
    }
  });

  it("a negative constant height (carving) is also flat with negative displacement", () => {
    const maps = evaluateMaterial(
      [new Layer({ height: new Constant(-3) })],
      8,
      8,
    );
    for (let i = 0; i < 64; i++) {
      expect(maps.orm[i * 4 + 0]).toBe(255);
      expect(maps.normal[i * 4 + 2]).toBe(255);
      expect(maps.displacement[i]).toBe(-3); // displacement = height
    }
  });

  it("a varying height does NOT take the flat fast path (tilted normals appear)", () => {
    // height = X()*10 varies across the row, so the relief must derive a real
    // slope rather than the flat fast path.
    const maps = evaluateMaterial(
      [
        new Layer({
          height: new MultiplicationExpression2D(new X(), new Constant(10)),
        }),
      ],
      16,
      1,
      { x1: 0, x2: 1 },
    );
    let tilted = false;
    for (let i = 0; i < 16; i++) if (maps.normal[i * 4] !== 128) tilted = true;
    expect(tilted).toBe(true);
  });
});
