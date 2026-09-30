/**
 * shader.ts — assemble the per-material WGSL compute shader for the WebGPU
 * preview tier.
 *
 * The shader mirrors the CPU {@link evalChannels}: per cell it evaluates every
 * layer's channels (codegen'd by wgsl-codegen), composites them with the
 * over-operator, and writes the same {@link ChannelData} outputs — packed
 * basecolor/ORM (RGBA8 in u32) plus raw emissive and height (f32). The
 * whole-grid finishing (AO + normal + emissive normalization) is left to the
 * CPU {@link finishMaps}, exactly as the tile pool does, so the GPU only does
 * the embarrassingly-parallel per-cell work.
 *
 * Output buffer layout (matches gpu-backend's bind group):
 *   binding 0  uniform Dims
 *   binding 1  basecolor : array<u32>   (pack4x8unorm RGBA8, length cells)
 *   binding 2  orm       : array<u32>   (R=0 placeholder, G=rough, B=metal, A=255)
 *   binding 3  emf       : array<f32>   (3 per cell, raw emissive)
 *   binding 4  height    : array<f32>   (1 per cell)
 */
import type { Expression2D, Layer } from "../engine";
import { compileExpr, WGSL_PRELUDE } from "./wgsl-codegen";

export const WORKGROUP = 8;

// Below this sentinel a height composite counts as "no contribution" → 0.
const H_SENTINEL = "-3.5e38";
const F32_MAX = "3.4e38";

// A blend `Σ Vᵢ·gᵢ` over layers, each gᵢ the codegen'd channel expression.
function blend(layers: Layer[], channel: (l: Layer) => string): string {
  return layers.map((l, i) => `V${i} * (${channel(l)})`).join(" + ");
}

/**
 * Build the compute shader for a layer stack (bottom-most first). Throws
 * {@link GpuUnsupportedError} (from the codegen) if any channel uses a node the
 * GPU path can't translate, so the backend can fall back to the CPU.
 */
export function buildComputeShader(layers: Layer[]): string {
  const n = layers.length;
  const X = "x";
  const Y = "y";
  const ce = (e: Expression2D): string => compileExpr(e, X, Y);

  // Per-layer alpha + over-operator visibility weights (Vᵢ), computed top-down.
  const lines: string[] = [];
  for (let i = 0; i < n; i++)
    lines.push(`  let a${i} = ${ce(layers[i].alpha)};`);
  lines.push(`  var run = 1.0;`);
  for (let i = n - 1; i >= 0; i--) {
    lines.push(`  let V${i} = a${i} * run;`);
    lines.push(`  run = run * (1.0 - a${i});`);
  }
  lines.push(`  let aTot = 1.0 - run;`);
  lines.push(`  let invA = select(0.0, 1.0 / aTot, aTot > 0.0);`);

  // Basecolor: blend per-channel linear values, divide by coverage, encode sRGB.
  lines.push(
    `  let brl = (${blend(layers, (l) => ce(l.basecolor.r))}) * invA;`,
  );
  lines.push(
    `  let bgl = (${blend(layers, (l) => ce(l.basecolor.g))}) * invA;`,
  );
  lines.push(
    `  let bbl = (${blend(layers, (l) => ce(l.basecolor.b))}) * invA;`,
  );
  lines.push(
    `  basecolor[idx] = pack4x8unorm(vec4<f32>(l2s(brl), l2s(bgl), l2s(bbl), aTot));`,
  );

  // ORM: R (AO) is filled on the CPU; G/B carry roughness/metallic.
  lines.push(
    `  let rough = (${blend(layers, (l) => ce(l.roughness))}) * invA;`,
  );
  lines.push(`  let metal = (${blend(layers, (l) => ce(l.metallic))}) * invA;`);
  lines.push(`  orm[idx] = pack4x8unorm(vec4<f32>(0.0, rough, metal, 1.0));`);

  // Emissive: blend (channel · strength) per layer; left un-normalized for the CPU.
  const em = (attr: "r" | "g" | "b") =>
    blend(
      layers,
      (l) => `(${ce(l.emissive[attr])}) * (${ce(l.emissive.strength)})`,
    );
  lines.push(`  emf[idx * 3u + 0u] = (${em("r")}) * invA;`);
  lines.push(`  emf[idx * 3u + 1u] = (${em("g")}) * invA;`);
  lines.push(`  emf[idx * 3u + 2u] = (${em("b")}) * invA;`);

  // Height: max over layers of αᵢ·hᵢ, ignoring non-finite contributions.
  lines.push(`  var hmax = ${H_SENTINEL};`);
  for (let i = 0; i < n; i++) {
    lines.push(`  let hv${i} = a${i} * (${ce(layers[i].height)});`);
    lines.push(
      `  if (hv${i} == hv${i} && hv${i} < ${F32_MAX} && hv${i} > -${F32_MAX} && hv${i} > hmax) { hmax = hv${i}; }`,
    );
  }
  lines.push(`  if (hmax <= ${H_SENTINEL} || hmax == 0.0) { hmax = 0.0; }`);
  lines.push(`  height[idx] = hmax;`);

  return `${WGSL_PRELUDE}
struct Dims { w: u32, h: u32, x1: f32, x2: f32, y1: f32, y2: f32, yUp: u32, pad: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read_write> basecolor: array<u32>;
@group(0) @binding(2) var<storage, read_write> orm: array<u32>;
@group(0) @binding(3) var<storage, read_write> emf: array<f32>;
@group(0) @binding(4) var<storage, read_write> height: array<f32>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= dims.w || gid.y >= dims.h) { return; }
  let idx = gid.y * dims.w + gid.x;
  // Cell-center sample coordinates; ys is reversed when yUp (row 0 = max Y).
  let fx = (f32(gid.x) + 0.5) / f32(dims.w);
  let fy = (f32(gid.y) + 0.5) / f32(dims.h);
  let x = dims.x1 + fx * (dims.x2 - dims.x1);
  var y = dims.y1 + fy * (dims.y2 - dims.y1);
  if (dims.yUp != 0u) { y = dims.y2 + fy * (dims.y1 - dims.y2); }
${lines.join("\n")}
}
`;
}
