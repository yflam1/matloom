/**
 * framing.ts — pure camera-framing math, factored out of viewer.ts so it can be
 * unit-tested without a Babylon engine/canvas (viewer.ts itself is browser-only).
 *
 * The viewer plane is a true 3D relief: its vertices are displaced along Z by the
 * height field (`vertex.z = -displacement`, see viewer.ts `applyDisplacement`).
 * These helpers turn that displacement into (a) the world-Z extent of the surface
 * and (b) an ArcRotateCamera distance that frames the whole 3D bounding box.
 */

/** World-Z range [zMin, zMax] of the displaced surface. */
export interface ZExtent {
  zMin: number;
  zMax: number;
}

/**
 * World-Z extent of the displaced plane. Vertices are placed at
 * `z = -displacement`, so the most *positive* displacement (tallest positive
 * height) is the most *negative* Z, and vice-versa. Returns a flat {0, 0} when
 * displacement is disabled or there is no grid yet.
 */
export function reliefZExtent(
  disp: Float32Array | null | undefined,
  displaceOn: boolean,
): ZExtent {
  if (!displaceOn || !disp || disp.length === 0) return { zMin: 0, zMax: 0 };
  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < disp.length; i++) {
    const v = disp[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  // Guard against a non-finite sample (e.g. a NaN height) collapsing the box.
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) return { zMin: 0, zMax: 0 };
  // z = -displacement flips the range: zMin = -max(disp), zMax = -min(disp).
  return { zMin: -mx, zMax: -mn };
}

export interface FitInput {
  /** Half the world-X extent of the plane. */
  halfW: number;
  /** Half the world-Y extent of the plane. */
  halfH: number;
  /** Half the world-Z (relief depth) extent; 0 for a flat plane. */
  halfD: number;
  /** Camera vertical field of view, in radians. */
  fov: number;
  /** Viewport aspect ratio (renderWidth / renderHeight). */
  aspect: number;
  /** Padding multiplier so the framed material doesn't touch the viewport edges. */
  margin: number;
}

/**
 * Distance for an ArcRotateCamera (targeting the box center) to frame the
 * material's bounding box.
 *
 * - **Flat** (`halfD ≈ 0`): the tight rectangle fit the viewer has always used —
 *   the limiting one of the vertical/horizontal FOV fits the plane exactly. This
 *   preserves the head-on look for the common flat case.
 * - **With relief depth**: fits the box's *bounding sphere* instead, so the whole
 *   3D extent stays visible from **any** orbit angle (revolve → see everything).
 *   A single fixed radius can't be tight at the head-on angle *and* cover the deep
 *   side views, so for thin-but-deep relief the home view reads more zoomed-out —
 *   the intended cost of "see the entire material when revolving".
 */
export function fitDistance({
  halfW,
  halfH,
  halfD,
  fov,
  aspect,
  margin,
}: FitInput): number {
  const tanHalfFov = Math.tan(fov / 2);
  if (halfD <= FLAT_EPS) {
    // Rectangle fit: vertical FOV is fixed; horizontal FOV is it scaled by aspect.
    const distV = halfH / tanHalfFov;
    const distH = halfW / (tanHalfFov * aspect);
    return Math.max(Math.max(distV, distH) * margin, MIN_DIST);
  }
  // Sphere fit: the circumscribed sphere is rotation-invariant, so framing it
  // frames the box at every orbit angle. Use the *limiting* (smaller) half-FOV.
  const r = Math.sqrt(halfW * halfW + halfH * halfH + halfD * halfD);
  const halfFovV = fov / 2;
  const halfFovH = Math.atan(tanHalfFov * aspect);
  const halfFov = Math.min(halfFovV, halfFovH);
  return Math.max((r / Math.sin(halfFov)) * margin, MIN_DIST);
}

/** Below this the relief counts as flat and the rectangle fit is used. */
const FLAT_EPS = 1e-9;
/** Floor on the framing distance, guarding a zero-size (degenerate) region. */
const MIN_DIST = 0.1;
