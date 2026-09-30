/**
 * OpenSimplex noise — faithful translation of opensimplex/internals.py
 * (Kurt Spencer / lmas/opensimplex).
 *
 * `init` uses BigInt for the 64-bit LCG seed expansion, matching Python's
 * ctypes.c_int64 overflow exactly. The permutation table is an Int32Array.
 * `noise2` / `noise2array` use plain IEEE-754 numbers, so output is
 * bit-for-bit identical to Python float64.
 */

export interface SimplexNoise {
  /** Sample the noise field at (x, y). Identical to the Python value. */
  noise2(x: number, y: number): number;
  /** Sample a grid; row-major Float64Array indexed [yi * xs.length + xi]. */
  noise2array(xs: Float64Array, ys: Float64Array): Float64Array;
}

const GRADIENTS2 = new Int8Array([
  5, 2, 2, 5, -5, 2, -2, 5, 5, -2, 2, -5, -5, -2, -2, -5,
]);
const STRETCH2 = -0.211324865405187;
const SQUISH2 = 0.366025403784439;
const NORM2 = 47;

// 64-bit LCG (BigInt for exact overflow, same as Python ctypes.c_int64).
const MUL = 6364136223846793005n;
const ADD = 1442695040888963407n;
const MASK = (1n << 64n) - 1n;
const SIGN = 1n << 63n;

function lcg(s: bigint): bigint {
  const v = (s * MUL + ADD) & MASK;
  return v >= SIGN ? v - (1n << 64n) : v;
}

function initPerm(seed: number): Int32Array {
  const perm = new Int32Array(256);
  const src = Int32Array.from({ length: 256 }, (_, i) => i);
  let s = BigInt(seed);
  s = lcg(lcg(lcg(s)));
  for (let i = 255; i >= 0; i--) {
    s = lcg(s);
    let r = Number((s + 31n) % BigInt(i + 1));
    if (r < 0) r += i + 1;
    perm[i] = src[r];
    src[r] = src[i];
  }
  return perm;
}

function extrapolate2(
  perm: Int32Array,
  xsb: number,
  ysb: number,
  dx: number,
  dy: number,
): number {
  const i = perm[(perm[xsb & 0xff] + ysb) & 0xff] & 0x0e;
  return GRADIENTS2[i] * dx + GRADIENTS2[i + 1] * dy;
}

function noise2(x: number, y: number, perm: Int32Array): number {
  const so = (x + y) * STRETCH2;
  const xs = x + so;
  const ys = y + so;
  const xsb = Math.floor(xs);
  const ysb = Math.floor(ys);
  const qo = (xsb + ysb) * SQUISH2;
  let dx0 = x - (xsb + qo);
  let dy0 = y - (ysb + qo);
  const xins = xs - xsb;
  const yins = ys - ysb;
  const inSum = xins + yins;

  let value = 0;

  // Contributions (1,0) and (0,1) — always evaluated.
  const dx1 = dx0 - 1 - SQUISH2;
  const dy1 = dy0 - SQUISH2;
  let a1 = 2 - dx1 * dx1 - dy1 * dy1;
  if (a1 > 0) {
    a1 *= a1;
    value += a1 * a1 * extrapolate2(perm, xsb + 1, ysb, dx1, dy1);
  }

  const dx2 = dx0 - SQUISH2;
  const dy2 = dy0 - 1 - SQUISH2;
  let a2 = 2 - dx2 * dx2 - dy2 * dy2;
  if (a2 > 0) {
    a2 *= a2;
    value += a2 * a2 * extrapolate2(perm, xsb, ysb + 1, dx2, dy2);
  }

  let xsvExt: number;
  let ysvExt: number;
  let dxExt: number;
  let dyExt: number;

  if (inSum <= 1) {
    // Inside triangle at (0,0).
    const zins = 1 - inSum;
    if (zins > xins || zins > yins) {
      if (xins > yins) {
        xsvExt = xsb + 1;
        ysvExt = ysb - 1;
        dxExt = dx0 - 1;
        dyExt = dy0 + 1;
      } else {
        xsvExt = xsb - 1;
        ysvExt = ysb + 1;
        dxExt = dx0 + 1;
        dyExt = dy0 - 1;
      }
    } else {
      xsvExt = xsb + 1;
      ysvExt = ysb + 1;
      dxExt = dx0 - 1 - 2 * SQUISH2;
      dyExt = dy0 - 1 - 2 * SQUISH2;
    }
    // Contribution (0,0).
    let a0 = 2 - dx0 * dx0 - dy0 * dy0;
    if (a0 > 0) {
      a0 *= a0;
      value += a0 * a0 * extrapolate2(perm, xsb, ysb, dx0, dy0);
    }
  } else {
    // Inside triangle at (1,1).
    const zins = 2 - inSum;
    if (zins < xins || zins < yins) {
      if (xins > yins) {
        xsvExt = xsb + 2;
        ysvExt = ysb;
        dxExt = dx0 - 2 - 2 * SQUISH2;
        dyExt = dy0 - 2 * SQUISH2;
      } else {
        xsvExt = xsb;
        ysvExt = ysb + 2;
        dxExt = dx0 - 2 * SQUISH2;
        dyExt = dy0 - 2 - 2 * SQUISH2;
      }
    } else {
      xsvExt = xsb;
      ysvExt = ysb;
      dxExt = dx0;
      dyExt = dy0;
    }
    // Contribution (1,1).
    dx0 -= 1 + 2 * SQUISH2;
    dy0 -= 1 + 2 * SQUISH2;
    let a0 = 2 - dx0 * dx0 - dy0 * dy0;
    if (a0 > 0) {
      a0 *= a0;
      value += a0 * a0 * extrapolate2(perm, xsb + 1, ysb + 1, dx0, dy0);
    }
  }

  // Extra vertex.
  let ae = 2 - dxExt * dxExt - dyExt * dyExt;
  if (ae > 0) {
    ae *= ae;
    value += ae * ae * extrapolate2(perm, xsvExt, ysvExt, dxExt, dyExt);
  }

  return value / NORM2;
}

function noise2array(
  xs: Float64Array,
  ys: Float64Array,
  perm: Int32Array,
): Float64Array {
  const out = new Float64Array(ys.length * xs.length);
  for (let yi = 0; yi < ys.length; yi++)
    for (let xi = 0; xi < xs.length; xi++)
      out[yi * xs.length + xi] = noise2(xs[xi], ys[yi], perm);
  return out;
}

export function openSimplex(seed: number): SimplexNoise {
  const perm = initPerm(seed);
  return {
    noise2: (x, y) => noise2(x, y, perm),
    noise2array: (xs, ys) => noise2array(xs, ys, perm),
  };
}
