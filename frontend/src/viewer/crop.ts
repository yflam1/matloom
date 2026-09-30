/**
 * crop.ts — pure region policy for "crop" rendering, factored out of viewer.ts
 * so it can be unit-tested without a Babylon engine/canvas (viewer.ts itself is
 * browser-only).
 *
 * The idea: the rendered texture is mapped across the material's *region*
 * rectangle (world XY), and the region and the pixel resolution (w×h) are fully
 * independent inputs to the engine. "Crop" mode leaves the resolution entirely
 * to the user (it's a normal, always-editable control) and only chooses the
 * *region* to render: the whole authored region while it fits the viewport, and
 * just the visible sub-rectangle once the user zooms in past it. Because the
 * resolution is fixed, those same texels then cover a tiny slice of the plane —
 * so effective sharpness climbs as you zoom in, for free, at the exact cost the
 * user picked.
 *
 * The viewer (viewer.ts) measures what is on screen by clipping the region
 * quad against the camera frustum — exact at any orbit angle, unlike
 * screen-space sampling, which misses the sliver of plane hugging its horizon
 * at grazing views — and hands over a {@link CropMeasurement}. The measurement
 * math ({@link visibleRegionFromFrustum}) and the policy turning a measurement
 * into the concrete {@link Region} to render live below, tested in isolation.
 */

/** A region rectangle in the material's coordinate space (may be inverted). */
export interface Region {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** The fixed render resolution that the crop window is snapped to. */
export interface CropResolution {
  w: number;
  h: number;
}

/** Tunables for the crop policy. */
export interface CropConfig {
  /** Fraction to inflate the visible sub-region by when zoomed in, so the crop
   * edges sit just outside the viewport (headroom, no visible seam). */
  margin: number;
}

export const DEFAULT_CROP: CropConfig = {
  margin: 0.04,
};

/** Subdivision of the displacement mesh. The crop snap quantum is an integer
 * number of texels, and with the default 512×512 resolution this makes the mesh
 * vertices land exactly on texels. */
export const MESH_SUBDIV = 512;

/**
 * Raw screen-space measurement produced by the viewer's Babylon projection.
 */
export interface CropMeasurement {
  /** False when the projection was degenerate (e.g. plane edge-on / behind the
   * camera). The caller should then keep the last good region rather than guess. */
  valid: boolean;
  /** True when the full region overflows the viewport (zoomed in): the policy
   * then narrows the rendered region to {@link visible}. */
  overflow: boolean;
  /** The visible sub-region (region coords) when overflowing; equals the full
   * region otherwise. */
  visible: Region;
}

/**
 * One world-space frustum plane with an inward-pointing normal: a point p is
 * inside the frustum iff normal·p + d ≥ 0. Structurally matches Babylon's
 * Plane (what Frustum.GetPlanes returns), so the viewer passes those directly
 * while this module stays Babylon-free and unit-testable.
 */
export interface FrustumPlane {
  normal: { x: number; y: number; z: number };
  d: number;
}

// Planes whose normal is this close to perpendicular to the Z axis bound the
// relief slab's z range negligibly; they are treated as z-free constraints.
const Z_AXIS_EPS = 1e-9;

/**
 * The exact visible sub-region of the region rectangle under a camera frustum.
 *
 * The surface is the region quad extruded along world Z over the relief depth
 * [zMin, zMax] (both 0 when displacement is off). A quad point (x, wy) is
 * visible — wy is the world Y, i.e. y or −y depending on the Y direction —
 * iff *some* z in that range puts it inside the frustum:
 *
 *   ∃ z ∈ [zMin, zMax] with nᵢ·(x, wy, z) + dᵢ ≥ 0 for every plane i.
 *
 * Eliminating z turns this into 2D half-plane constraints on the quad — with
 * sᵢ = nᵢ.x·x + nᵢ.y·wy + dᵢ:
 *
 *   nᵢ.z > 0 (z lower-bounded):      sᵢ + nᵢ.z·zMax ≥ 0
 *   nᵢ.z < 0 (z upper-bounded):      sᵢ + nᵢ.z·zMin ≥ 0
 *   nᵢ.z = 0 (z-free):               sᵢ ≥ 0
 *   pair (i: nᵢ.z > 0, j: nᵢ.z < 0): sⱼ·nᵢ.z − sᵢ·nⱼ.z ≥ 0
 *
 * Clipping the quad against all of them (Sutherland–Hodgman) yields the exact
 * visible polygon; its bbox, back in region coords, is the answer. This
 * replaces screen-space grid sampling: at grazing view angles the plane's far
 * side projects into an arbitrarily thin band around its horizon line, which
 * any fixed screen grid straddles and misses — silently cropping away surface
 * that is plainly on screen.
 *
 * Returns null when nothing of the quad is visible (looking away / fully
 * clipped), so the caller can keep its previous region instead of guessing.
 */
export function visibleRegionFromFrustum(
  planes: readonly FrustumPlane[],
  full: Region,
  yUp: boolean,
  zMin: number,
  zMax: number,
): Region | null {
  const xlo = Math.min(full.x1, full.x2);
  const xhi = Math.max(full.x1, full.x2);
  const ylo = Math.min(full.y1, full.y2);
  const yhi = Math.max(full.y1, full.y2);
  const worldY = (ry: number) => (yUp ? ry : -ry);
  const regionY = (wy: number) => (yUp ? wy : -wy);

  // 2D half-planes [A, B, C] meaning A·x + B·wy + C ≥ 0.
  const halfPlanes: [number, number, number][] = [];
  const lower: FrustumPlane[] = []; // n.z > 0: z ≥ …
  const upper: FrustumPlane[] = []; // n.z < 0: z ≤ …
  for (const pl of planes) {
    const n = pl.normal;
    if (n.z > Z_AXIS_EPS) {
      lower.push(pl);
      halfPlanes.push([n.x, n.y, n.z * zMax + pl.d]);
    } else if (n.z < -Z_AXIS_EPS) {
      upper.push(pl);
      halfPlanes.push([n.x, n.y, n.z * zMin + pl.d]);
    } else {
      halfPlanes.push([n.x, n.y, pl.d]);
    }
  }
  for (const pi of lower) {
    for (const pj of upper) {
      const ai = pi.normal.z;
      const aj = pj.normal.z;
      halfPlanes.push([
        pj.normal.x * ai - pi.normal.x * aj,
        pj.normal.y * ai - pi.normal.y * aj,
        pj.d * ai - pi.d * aj,
      ]);
    }
  }

  // Sutherland–Hodgman: clip the region quad (world coords) by each half-plane.
  let poly: [number, number][] = [
    [xlo, worldY(ylo)],
    [xhi, worldY(ylo)],
    [xhi, worldY(yhi)],
    [xlo, worldY(yhi)],
  ];
  for (const [A, B, C] of halfPlanes) {
    const out: [number, number][] = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      const da = A * a[0] + B * a[1] + C;
      const db = A * b[0] + B * b[1] + C;
      const aIn = da >= 0;
      const bIn = db >= 0;
      if (aIn) out.push(a);
      if (aIn !== bIn) {
        const t = da / (da - db);
        out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
      }
    }
    poly = out;
    if (poly.length === 0) return null;
  }

  // The polygon is convex, so its bbox is fixed by its vertices.
  let bxlo = Infinity;
  let bxhi = -Infinity;
  let bylo = Infinity;
  let byhi = -Infinity;
  for (const [x, wy] of poly) {
    const ry = regionY(wy);
    if (x < bxlo) bxlo = x;
    if (x > bxhi) bxhi = x;
    if (ry < bylo) bylo = ry;
    if (ry > byhi) byhi = ry;
  }
  return { x1: bxlo, y1: bylo, x2: bxhi, y2: byhi };
}

/**
 * Intersect a candidate sub-region with the full region (after inflating it by
 * `margin`), preserving the full region's axis orientation (x1>x2 / y1>y2).
 * Returns a copy of the full region if the intersection is empty/degenerate.
 */
export function clampRegion(sub: Region, full: Region, margin = 0): Region {
  const fxlo = Math.min(full.x1, full.x2);
  const fxhi = Math.max(full.x1, full.x2);
  const fylo = Math.min(full.y1, full.y2);
  const fyhi = Math.max(full.y1, full.y2);

  let xlo = Math.min(sub.x1, sub.x2);
  let xhi = Math.max(sub.x1, sub.x2);
  let ylo = Math.min(sub.y1, sub.y2);
  let yhi = Math.max(sub.y1, sub.y2);

  const mx = (xhi - xlo) * margin;
  const my = (yhi - ylo) * margin;
  xlo -= mx;
  xhi += mx;
  ylo -= my;
  yhi += my;

  xlo = Math.max(xlo, fxlo);
  xhi = Math.min(xhi, fxhi);
  ylo = Math.max(ylo, fylo);
  yhi = Math.min(yhi, fyhi);

  if (!(xhi > xlo) || !(yhi > ylo)) return { ...full };

  // Re-apply the full region's orientation so a reversed authored region stays
  // reversed (the engine mirrors the sample axis for x1>x2 / y1>y2).
  const xflip = full.x1 > full.x2;
  const yflip = full.y1 > full.y2;
  return {
    x1: xflip ? xhi : xlo,
    x2: xflip ? xlo : xhi,
    y1: yflip ? yhi : ylo,
    y2: yflip ? ylo : yhi,
  };
}

/**
 * Inflate a region by `margin` fraction of its own span, without clipping.
 */
function inflateRegion(sub: Region, margin: number): Region {
  let xlo = Math.min(sub.x1, sub.x2);
  let xhi = Math.max(sub.x1, sub.x2);
  let ylo = Math.min(sub.y1, sub.y2);
  let yhi = Math.max(sub.y1, sub.y2);

  const mx = (xhi - xlo) * margin;
  const my = (yhi - ylo) * margin;
  xlo -= mx;
  xhi += mx;
  ylo -= my;
  yhi += my;

  return { x1: xlo, y1: ylo, x2: xhi, y2: yhi };
}

/**
 * Greatest common divisor of two positive integers (Euclid's algorithm).
 */
function gcd(a: number, b: number): number {
  a = Math.round(Math.abs(a));
  b = Math.round(Math.abs(b));
  while (b !== 0) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a;
}

/**
 * Snap a candidate sub-region to a world-anchored texel grid. The returned region
 * always covers `sub`, has exactly `res.w × res.h` texels (so the user's
 * resolution is unchanged), and is anchored at the lower corner of `full`. Panning
 * by less than one grid cell therefore produces an identical region, and a pan by
 * exactly one cell shifts the sampling lattice by an integer number of texels —
 * the maps re-render as a rigid translation rather than re-aliasing.
 *
 * The texel *size* is continuous in the visible span (no ladder rungs): it is the
 * smallest size that covers the inflated visible region at the user's resolution,
 * capped at the full-region texel size so a non-zoomed view renders the whole
 * region. The relief (normal `dx`, AO radius) therefore changes smoothly with the
 * camera, never in the discrete 1% jumps the old ladder produced.
 */
function snapToTexelGrid(
  sub: Region,
  full: Region,
  res: CropResolution,
): Region {
  const fullXLo = Math.min(full.x1, full.x2);
  const fullXHi = Math.max(full.x1, full.x2);
  const fullYLo = Math.min(full.y1, full.y2);
  const fullYHi = Math.max(full.y1, full.y2);

  let xlo = Math.min(sub.x1, sub.x2);
  let xhi = Math.max(sub.x1, sub.x2);
  let ylo = Math.min(sub.y1, sub.y2);
  let yhi = Math.max(sub.y1, sub.y2);

  const needX = Math.max(0, xhi - xlo);
  const needY = Math.max(0, yhi - ylo);

  const fullTx = (fullXHi - fullXLo) / res.w;
  const fullTy = (fullYHi - fullYLo) / res.h;

  // Continuous texel size: the smallest that covers the inflated visible span at
  // the user's resolution, capped at the full-region texel size (a non-zoomed
  // view renders the whole region). A zero span (degenerate) falls back to the
  // full texel size.
  const tx = needX > 0 ? Math.min(needX / res.w, fullTx) : fullTx;
  const ty = needY > 0 ? Math.min(needY / res.h, fullTy) : fullTy;

  const Sx = res.w * tx;
  const Sy = res.h * ty;

  // Snap quantum that is an integer number of texels AND an integer number of
  // mesh vertex steps along each axis, so both the texture and the displacement
  // mesh stay phase-locked during a pan.
  const qx = tx * (res.w / gcd(res.w, MESH_SUBDIV));
  const qy = ty * (res.h / gcd(res.h, MESH_SUBDIV));

  let sx = fullXLo + qx * Math.floor((xlo - fullXLo) / qx);
  let sy = fullYLo + qy * Math.floor((ylo - fullYLo) / qy);

  // Keep the window inside the authored region when the crop is near an edge;
  // interior crops never hit these.
  if (sx + Sx > fullXHi) sx = fullXHi - Sx;
  if (sy + Sy > fullYHi) sy = fullYHi - Sy;
  if (sx < fullXLo) sx = fullXLo;
  if (sy < fullYLo) sy = fullYLo;

  return { x1: sx, y1: sy, x2: sx + Sx, y2: sy + Sy };
}

/**
 * Turn a screen-space measurement into the region to render. Returns null when
 * the measurement is invalid (caller keeps its previous region).
 *
 * - **Fits the viewport** (`!overflow`): render the whole authored region.
 * - **Overflows** (zoomed in): render only the visible sub-region, inflated by
 *   `margin` for headroom, then snapped to a world-anchored texel grid so the
 *   already-visible pixels stay identical during small pans. The resolution is
 *   unchanged (`res` is the user's manual width/height).
 */
export function computeCropRegion(
  m: CropMeasurement,
  full: Region,
  cfg: CropConfig = DEFAULT_CROP,
  res: CropResolution = { w: 512, h: 512 },
): Region | null {
  if (!m.valid) return null;
  if (!m.overflow) return { ...full };
  const inflated = inflateRegion(m.visible, cfg.margin);
  const snapped = snapToTexelGrid(inflated, full, res);
  return clampRegion(snapped, full, 0);
}

/** Whether two regions are close enough that re-rendering would be wasted. */
export function regionsEqual(a: Region | null, b: Region | null): boolean {
  if (!a || !b) return a === b;
  const eps = 1e-4;
  return (
    Math.abs(a.x1 - b.x1) < eps &&
    Math.abs(a.x2 - b.x2) < eps &&
    Math.abs(a.y1 - b.y1) < eps &&
    Math.abs(a.y2 - b.y2) < eps
  );
}
