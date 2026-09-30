/** Tests for src/viewer/crop.ts — the pure crop-region policy. */
import { describe, it, expect } from "vitest";
import {
  DEFAULT_CROP,
  clampRegion,
  computeCropRegion,
  regionsEqual,
  visibleRegionFromFrustum,
  type CropConfig,
  type CropMeasurement,
  type CropResolution,
  type FrustumPlane,
  type Region,
} from "../../src/viewer/crop";

const cfg: CropConfig = DEFAULT_CROP;
const UNIT: Region = { x1: 0, y1: 0, x2: 1, y2: 1 };
const RES: CropResolution = { w: 512, h: 512 };

describe("clampRegion", () => {
  it("intersects a sub-region with the full region", () => {
    const sub = { x1: 0.25, y1: 0.25, x2: 0.75, y2: 0.75 };
    expect(clampRegion(sub, UNIT)).toEqual(sub);
  });

  it("clips a sub-region that spills past the full bounds", () => {
    const sub = { x1: -1, y1: 0.5, x2: 0.5, y2: 2 };
    expect(clampRegion(sub, UNIT)).toEqual({ x1: 0, y1: 0.5, x2: 0.5, y2: 1 });
  });

  it("falls back to the full region on an empty intersection", () => {
    const sub = { x1: 5, y1: 5, x2: 6, y2: 6 };
    expect(clampRegion(sub, UNIT)).toEqual(UNIT);
  });

  it("inflates by the margin fraction before clipping", () => {
    const sub = { x1: 0.4, y1: 0.4, x2: 0.6, y2: 0.6 };
    const out = clampRegion(sub, UNIT, 0.5); // span 0.2 → ±0.1
    expect(out.x1).toBeCloseTo(0.3, 10);
    expect(out.x2).toBeCloseTo(0.7, 10);
  });

  it("preserves the full region's reversed orientation", () => {
    const full = { x1: 1, y1: 1, x2: 0, y2: 0 }; // both axes reversed
    const sub = { x1: 0.25, y1: 0.25, x2: 0.75, y2: 0.75 };
    expect(clampRegion(sub, full)).toEqual({
      x1: 0.75,
      y1: 0.75,
      x2: 0.25,
      y2: 0.25,
    });
  });
});

describe("computeCropRegion", () => {
  const base: CropMeasurement = {
    valid: true,
    overflow: false,
    visible: UNIT,
  };

  it("returns null for an invalid measurement", () => {
    expect(
      computeCropRegion({ ...base, valid: false }, UNIT, cfg, RES),
    ).toBeNull();
  });

  it("renders the whole authored region when it fits the viewport", () => {
    expect(computeCropRegion(base, UNIT, cfg, RES)).toEqual(UNIT);
  });

  it("crops to the (margin-inflated, clamped) visible sub-region when zoomed in", () => {
    const visible = { x1: 0.4, y1: 0.4, x2: 0.6, y2: 0.6 };
    const region = computeCropRegion(
      { ...base, overflow: true, visible },
      UNIT,
      cfg,
      RES,
    )!;
    // Strictly inside the full region, and inflated past the raw visible rect by
    // the margin (so the crop edges sit just outside the viewport).
    expect(region.x1).toBeGreaterThanOrEqual(0);
    expect(region.x2).toBeLessThanOrEqual(1);
    expect(region.x1).toBeLessThan(0.4);
    expect(region.x2).toBeGreaterThan(0.6);
  });

  it("clamps an overflowing visible rect that spills past the full region", () => {
    // Deep zoom near a corner: the visible bbox can extend beyond the plane.
    const visible = { x1: -0.5, y1: -0.5, x2: 0.2, y2: 0.2 };
    const region = computeCropRegion(
      { ...base, overflow: true, visible },
      UNIT,
      cfg,
      RES,
    )!;
    expect(region.x1).toBeGreaterThanOrEqual(0);
    expect(region.y1).toBeGreaterThanOrEqual(0);
    expect(region.x2).toBeLessThanOrEqual(1);
    expect(region.y2).toBeLessThanOrEqual(1);
  });

  it("preserves the authored region's reversed orientation when cropping", () => {
    const full: Region = { x1: 1, y1: 1, x2: 0, y2: 0 };
    const visible = { x1: 0.3, y1: 0.3, x2: 0.7, y2: 0.7 };
    const region = computeCropRegion(
      { ...base, overflow: true, visible },
      full,
      cfg,
      RES,
    )!;
    expect(region.x1).toBeGreaterThan(region.x2); // stays reversed on x
    expect(region.y1).toBeGreaterThan(region.y2); // stays reversed on y
  });

  it("snaps sub-texel pans to the same region", () => {
    const visible = { x1: 0.4, y1: 0.4, x2: 0.6, y2: 0.6 };
    const a = computeCropRegion(
      { ...base, overflow: true, visible },
      UNIT,
      cfg,
      RES,
    )!;
    // Pan by ~1/5 of a crop texel (the crop texel size here is much smaller than
    // the full-region texel size because we are zoomed in).
    const b = computeCropRegion(
      {
        ...base,
        overflow: true,
        visible: { ...visible, x1: 0.4001, x2: 0.6001 },
      },
      UNIT,
      cfg,
      RES,
    )!;
    expect(a).toEqual(b);
  });

  it("shifts by an integer number of texels when the pan crosses a grid line", () => {
    const visible = { x1: 0.4, y1: 0.4, x2: 0.6, y2: 0.6 };
    const a = computeCropRegion(
      { ...base, overflow: true, visible },
      UNIT,
      cfg,
      RES,
    )!;
    // Pan right by enough to cross exactly one crop texel grid line.
    const cropTexel = a.x2 - a.x1;
    const b = computeCropRegion(
      {
        ...base,
        overflow: true,
        visible: {
          ...visible,
          x1: visible.x1 + cropTexel,
          x2: visible.x2 + cropTexel,
        },
      },
      UNIT,
      cfg,
      RES,
    )!;
    expect(a).not.toEqual(b);
    expect(b.x1 - a.x1).toBeCloseTo(cropTexel, 10);
    expect(b.x2 - a.x2).toBeCloseTo(cropTexel, 10);
    expect(b.y1).toBeCloseTo(a.y1, 10);
    expect(b.y2).toBeCloseTo(a.y2, 10);
  });

  it("keeps a constant texel size across a same-width pan (no re-alias)", () => {
    const visible = { x1: 0.4, y1: 0.4, x2: 0.6, y2: 0.6 };
    const a = computeCropRegion(
      { ...base, overflow: true, visible },
      UNIT,
      cfg,
      RES,
    )!;
    // Same visible *width*, panned by 0.001: the texel size is unchanged, so the
    // re-render is a rigid translation rather than a re-alias.
    const b = computeCropRegion(
      {
        ...base,
        overflow: true,
        visible: { x1: 0.401, y1: 0.401, x2: 0.601, y2: 0.601 },
      },
      UNIT,
      cfg,
      RES,
    )!;
    expect(a.x2 - a.x1).toBeCloseTo(b.x2 - b.x1, 10);
    expect(a.y2 - a.y1).toBeCloseTo(b.y2 - b.y1, 10);
  });

  it("scales the texel size continuously with the visible span (no jumps)", () => {
    const a = computeCropRegion(
      {
        ...base,
        overflow: true,
        visible: { x1: 0.4, y1: 0.4, x2: 0.6, y2: 0.6 },
      },
      UNIT,
      cfg,
      RES,
    )!;
    // A 0.5% wider visible rect grows the crop span by ~0.5% — continuously, not
    // in a discrete ladder jump.
    const b = computeCropRegion(
      {
        ...base,
        overflow: true,
        visible: { x1: 0.4, y1: 0.4, x2: 0.602, y2: 0.6 },
      },
      UNIT,
      cfg,
      RES,
    )!;
    const spanA = a.x2 - a.x1;
    const spanB = b.x2 - b.x1;
    expect(spanB / spanA).toBeCloseTo(0.202 / 0.2, 6);
    expect(spanB).toBeGreaterThan(spanA);
  });
});

describe("regionsEqual", () => {
  it("is true for identical regions and handles nulls", () => {
    expect(regionsEqual(UNIT, { ...UNIT })).toBe(true);
    expect(regionsEqual(null, null)).toBe(true);
    expect(regionsEqual(UNIT, null)).toBe(false);
  });

  it("is false when any corner differs beyond the epsilon", () => {
    expect(regionsEqual(UNIT, { x1: 0, y1: 0, x2: 0.5, y2: 1 })).toBe(false);
  });

  it("treats sub-epsilon differences as equal", () => {
    expect(regionsEqual(UNIT, { ...UNIT, x2: 1 + 1e-6 })).toBe(true);
  });
});

// — visibleRegionFromFrustum —
//
// Frustum planes are built by hand (inward normals; inside ⇔ n·p + d ≥ 0) so
// the suite stays Babylon-free. pinholePlanes() was cross-checked against
// Babylon's Frustum.GetPlanes to <1e-6 on every plane but the far one (whose
// extracted `d` drifts by ~0.4 world units at maxZ = 1000 — float-extraction
// noise, irrelevant at region scale).

type Vec3 = [number, number, number];

/** The six inward planes of a pinhole-camera frustum (vertical-fixed fov). */
function pinholePlanes(
  eye: Vec3,
  target: Vec3,
  {
    fov = 0.8, // Babylon's camera default
    aspect = 1.6,
    minZ = 0.05,
    maxZ = 1000,
  }: { fov?: number; aspect?: number; minZ?: number; maxZ?: number } = {},
): FrustumPlane[] {
  const sub = (a: Vec3, b: Vec3): Vec3 => [
    a[0] - b[0],
    a[1] - b[1],
    a[2] - b[2],
  ];
  const add = (a: Vec3, b: Vec3): Vec3 => [
    a[0] + b[0],
    a[1] + b[1],
    a[2] + b[2],
  ];
  const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
  const cross = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a: Vec3): Vec3 => {
    const l = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / l, a[1] / l, a[2] / l];
  };
  const fwd = norm(sub(target, eye));
  const right = norm(cross([0, 1, 0], fwd));
  const up = cross(fwd, right);
  const tanV = Math.tan(fov / 2);
  const tanH = tanV * aspect;
  const through = (n: Vec3): FrustumPlane => ({
    normal: { x: n[0], y: n[1], z: n[2] },
    d: -dot(n, eye),
  });
  return [
    through(add(scale(fwd, tanH), right)), // left: ndc x ≥ −1
    through(add(scale(fwd, tanH), scale(right, -1))), // right
    through(add(scale(fwd, tanV), up)), // bottom
    through(add(scale(fwd, tanV), scale(up, -1))), // top
    {
      normal: { x: fwd[0], y: fwd[1], z: fwd[2] },
      d: -(dot(fwd, eye) + minZ),
    }, // near
    {
      normal: { x: -fwd[0], y: -fwd[1], z: -fwd[2] },
      d: dot(fwd, eye) + maxZ,
    }, // far
  ];
}

/** A plane from raw coefficients: n·p + d ≥ 0 keeps p. */
const plane = (
  nx: number,
  ny: number,
  nz: number,
  d: number,
): FrustumPlane => ({ normal: { x: nx, y: ny, z: nz }, d });

describe("visibleRegionFromFrustum", () => {
  const FULL: Region = { x1: 0, y1: 0, x2: 6, y2: 6 };
  const FLAT = [0, 0] as const;

  it("returns the full region when a zoomed-out camera sees it all", () => {
    const planes = pinholePlanes([3, 3, -20], [3, 3, 0]);
    expect(visibleRegionFromFrustum(planes, FULL, true, ...FLAT)).toEqual({
      x1: 0,
      y1: 0,
      x2: 6,
      y2: 6,
    });
  });

  it("returns a tight sub-region for a deep frontal zoom", () => {
    const planes = pinholePlanes([3, 3, -1.7], [3, 3, 0]);
    const vis = visibleRegionFromFrustum(planes, FULL, true, ...FLAT)!;
    expect(vis.x1).toBeCloseTo(1.85, 3);
    expect(vis.x2).toBeCloseTo(4.15, 3);
    expect(vis.y1).toBeCloseTo(2.281, 3);
    expect(vis.y2).toBeCloseTo(3.719, 3);
  });

  // The missing-region bug: camera near the positive x-axis looking towards
  // −x sees the wall at a grazing angle. Its far side projects into a thin
  // screen band at the plane's horizon, which the old 9×9 screen-space grid
  // straddled without a single sample — it measured x[3.0, 6.6] y[1.3, 4.7]
  // and the crop lopped off everything beyond. The frustum clip is exact.
  it("covers the far side of the plane at a grazing view angle", () => {
    const alpha = (5 * Math.PI) / 180; // 5° off the +x axis
    const eye: Vec3 = [3 + 4 * Math.cos(alpha), 3, 4 * Math.sin(alpha)];
    const planes = pinholePlanes(eye, [3, 3, 0]);
    const vis = visibleRegionFromFrustum(planes, FULL, true, ...FLAT)!;
    expect(vis.x1).toBeCloseTo(0, 5);
    expect(vis.x2).toBeCloseTo(6, 5);
    expect(vis.y1).toBeCloseTo(0.045, 3);
    expect(vis.y2).toBeCloseTo(5.955, 3);
  });

  it("expands the visible region to cover the relief slab", () => {
    const alpha = (5 * Math.PI) / 180;
    const eye: Vec3 = [3 + 4 * Math.cos(alpha), 3, 4 * Math.sin(alpha)];
    const planes = pinholePlanes(eye, [3, 3, 0]);
    const flat = visibleRegionFromFrustum(planes, FULL, true, ...FLAT)!;
    const slab = visibleRegionFromFrustum(planes, FULL, true, -0.4, 0.4)!;
    // Displaced geometry leaning into view is included: strictly wider cover…
    expect(slab.y1).toBeLessThan(flat.y1);
    expect(slab.y2).toBeGreaterThan(flat.y2);
    // …but never smaller than the flat answer.
    expect(slab.x1).toBeLessThanOrEqual(flat.x1);
    expect(slab.x2).toBeGreaterThanOrEqual(flat.x2);
  });

  it("returns null when the camera looks away from the region", () => {
    const planes = pinholePlanes([3, 3, -5], [3, 3, -10]);
    expect(visibleRegionFromFrustum(planes, FULL, true, ...FLAT)).toBeNull();
  });

  it("returns null when a plane clips the whole quad", () => {
    // x ≥ 10 keeps nothing of the [0, 6] quad.
    expect(
      visibleRegionFromFrustum([plane(1, 0, 0, -10)], FULL, true, ...FLAT),
    ).toBeNull();
  });

  it("honors the z caps: a near plane past zMin clips the flat quad but not the slab", () => {
    // z ≥ 0.2 keeps nothing of the flat quad (z = 0), but z = 0.4 satisfies it.
    const planes = [plane(0, 0, 1, -0.2), plane(0, 0, -1, 100)];
    expect(visibleRegionFromFrustum(planes, FULL, true, ...FLAT)).toBeNull();
    expect(visibleRegionFromFrustum(planes, FULL, true, 0, 0.4)).toEqual({
      x1: 0,
      y1: 0,
      x2: 6,
      y2: 6,
    });
  });

  it("resolves crossing z bounds (pair constraints) to a band", () => {
    // z ≥ wy and z ≤ wy force z = wy, achievable iff wy ∈ [zMin, zMax].
    const planes = [plane(0, -1, 1, 0), plane(0, 1, -1, 0)];
    const vis = visibleRegionFromFrustum(planes, FULL, true, 0, 0.4)!;
    expect(vis.x1).toBeCloseTo(0, 10);
    expect(vis.x2).toBeCloseTo(6, 10);
    expect(vis.y1).toBeCloseTo(0, 10);
    expect(vis.y2).toBeCloseTo(0.4, 10);
  });

  it("mirrors the band in region coords when Y points down", () => {
    // z ≥ −wy and z ≤ −wy force z = −wy: wy ∈ [−0.4, 0] ⇔ region y ∈ [0, 0.4].
    const planes = [plane(0, 1, 1, 0), plane(0, -1, -1, 0)];
    const vis = visibleRegionFromFrustum(planes, FULL, false, 0, 0.4)!;
    expect(vis.x1).toBeCloseTo(0, 10);
    expect(vis.x2).toBeCloseTo(6, 10);
    expect(vis.y1).toBeCloseTo(0, 10);
    expect(vis.y2).toBeCloseTo(0.4, 10);
  });
});
