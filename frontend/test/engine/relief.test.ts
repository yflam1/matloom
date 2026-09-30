/** Tests for src/engine/relief.ts — height compositing + derived normal/AO. */
import { describe, it, expect } from "vitest";
import {
  compositeHeight,
  heightToNormal,
  heightToAo,
} from "../../src/engine/relief";

const F32 = (xs: number[]) => Float32Array.from(xs);

describe("compositeHeight", () => {
  it("takes maxᵢ αᵢ·hᵢ per cell", () => {
    // Two cells. Cell 0: both layers present (α=1) → max(1, 9) = 9.
    // Cell 1: top layer absent (α=0) → max(1·1, 0·9) = 1.
    const alpha = [F32([1, 1]), F32([1, 0])];
    const height = [F32([1, 1]), F32([9, 9])];
    const h = compositeHeight(alpha, height);
    expect(Array.from(h)).toEqual([9, 1]);
  });

  it("preserves negative height (no substrate floor)", () => {
    const h = compositeHeight([F32([1])], [F32([-5])]);
    expect(h[0]).toBe(-5);
  });

  it("is 0 where no layer is present (substrate)", () => {
    const h = compositeHeight([F32([0])], [F32([-5])]);
    expect(h[0]).toBe(0);
  });

  it("emits +0 (not −0) for a zero contribution", () => {
    // `0 * -5` is −0 in IEEE; the substrate must be +0 so it matches the Python
    // reference and never writes a negative zero into the height map.
    const h = compositeHeight([F32([0])], [F32([-5])]);
    expect(Object.is(h[0], 0)).toBe(true);
  });

  it("drops a non-finite contribution (poisoned layer treated as absent)", () => {
    // A height expression can hit Log(0) (−∞) or Sqrt(-1) (NaN); the finite
    // layer below must still win, mirroring the Python engine.
    const alpha = [F32([1, 1]), F32([1, 1])];
    const height = [F32([Infinity, NaN]), F32([2, 3])];
    const h = compositeHeight(alpha, height);
    expect(Array.from(h)).toEqual([2, 3]);
  });

  it("falls to substrate 0 when the only contribution is non-finite", () => {
    expect(compositeHeight([F32([1])], [F32([-Infinity])])[0]).toBe(0);
    expect(compositeHeight([F32([1])], [F32([NaN])])[0]).toBe(0);
  });
});

describe("heightToNormal", () => {
  it("flat height → (0, 0, 1) everywhere", () => {
    const h = F32([2, 2, 2, 2, 2, 2]); // 3×2 constant
    const { nx, ny, nz } = heightToNormal(h, 3, 2, 1, 1, true);
    for (let i = 0; i < 6; i++) {
      expect(nx[i]).toBeCloseTo(0, 6);
      expect(ny[i]).toBeCloseTo(0, 6);
      expect(nz[i]).toBeCloseTo(1, 6);
    }
  });

  it("a ramp rising in +X tilts the normal toward -X (nx < 0)", () => {
    // height increases left→right; ∂H/∂X > 0 → nx = -∂H/∂X < 0.
    const h = F32([0, 1, 2, 3]); // 4×1
    const { nx, nz } = heightToNormal(h, 4, 1, 1, 1, true);
    expect(nx[1]).toBeLessThan(0);
    expect(nz[1]).toBeGreaterThan(0);
  });

  it("yUp flips the sign of the Y-derivative", () => {
    const h = F32([0, 0, 1, 1]); // 2 rows × 2 cols: bottom rows higher
    const up = heightToNormal(h, 2, 2, 1, 1, true);
    const down = heightToNormal(h, 2, 2, 1, 1, false);
    expect(up.ny[0]).toBeCloseTo(-down.ny[0], 6);
  });
});

describe("heightToAo", () => {
  it("flat height → fully lit (AO 1)", () => {
    const h = F32(new Array(25).fill(3));
    const ao = heightToAo(h, 5, 5, 1, 2, 1);
    for (const v of ao) expect(v).toBeCloseTo(1, 6);
  });

  it("a pit (cell below its neighbors) is darkened, a peak is not", () => {
    // 5×5, mostly 1 with a deep pit at the center and a tall spike off-center.
    const g = new Array(25).fill(1);
    g[12] = -4; // center pit
    const ao = heightToAo(F32(g), 5, 5, 1, 1, 1);
    expect(ao[12]).toBeLessThan(1); // cavity darkened

    const g2 = new Array(25).fill(1);
    g2[12] = 5; // center peak
    const ao2 = heightToAo(F32(g2), 5, 5, 1, 1, 1);
    expect(ao2[12]).toBe(1); // peaks are not occluded
  });

  it("radiusY defaults to radiusX (isotropic)", () => {
    const g = new Array(49).fill(1);
    g[24] = -3; // center pit on a 7×7 grid
    const iso = heightToAo(F32(g), 7, 7, 1, 2, 1); // radiusY omitted
    const explicit = heightToAo(F32(g), 7, 7, 1, 2, 1, 2); // radiusY = 2
    expect(Array.from(iso)).toEqual(Array.from(explicit));
  });

  it("blurs the two axes independently (per-axis radius)", () => {
    // A full low row is constant along X, so an X-only blur (radiusY=0) leaves
    // it unchanged → nothing darkens; a Y-only blur (radiusX=0) smears it across
    // rows → the row reads as a cavity.
    const g = new Array(49).fill(1);
    for (let j = 0; j < 7; j++) g[3 * 7 + j] = -3; // entire row 3 low

    const aoX = heightToAo(F32(g), 7, 7, 1, 2, 1, 0); // X-only
    for (const v of aoX) expect(v).toBe(1); // identity on a constant row

    const aoY = heightToAo(F32(g), 7, 7, 1, 0, 1, 2); // Y-only
    expect(aoY[3 * 7 + 3]).toBeLessThan(1); // low row darkened
    expect(aoY[0 * 7 + 3]).toBe(1); // far rows stay lit
  });

  it("integer radius is bit-exact whether passed as int or float", () => {
    // Fractional support is a pure superset: an integer radius must produce the
    // same bytes whether the caller passes 2 or 2.0 (the lerp short-circuits at
    // frac 0).
    const g = new Array(49).fill(1);
    g[24] = -3;
    const intR = heightToAo(F32(g), 7, 7, 1, 2, 1);
    const floatR = heightToAo(F32(g), 7, 7, 1, 2.0, 1);
    expect(Array.from(intR)).toEqual(Array.from(floatR));
  });

  it("fractional radius darkens continuously between the enclosing integers", () => {
    // A shallow pit (no [0,1] saturation) so the darkening varies with radius.
    // radius 2.5 sits between radius 2 and 3, and the half step is smaller than
    // the full integer step (no `floor` jump).
    const g = new Array(81).fill(1);
    g[40] = -0.5; // center of a 9×9 grid
    const ao2 = heightToAo(F32(g), 9, 9, 1, 2, 1);
    const ao25 = heightToAo(F32(g), 9, 9, 1, 2.5, 1);
    const ao3 = heightToAo(F32(g), 9, 9, 1, 3, 1);
    const lo = Math.min(ao2[40], ao3[40]);
    const hi = Math.max(ao2[40], ao3[40]);
    expect(ao25[40]).toBeGreaterThanOrEqual(lo - 1e-9);
    expect(ao25[40]).toBeLessThanOrEqual(hi + 1e-9);
    const stepHalf = Math.abs(ao25[40] - ao2[40]);
    const stepFull = Math.abs(ao3[40] - ao2[40]);
    expect(stepHalf).toBeLessThan(stepFull);
  });
});
