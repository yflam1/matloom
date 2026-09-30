/**
 * gpu-backend.ts — the WebGPU compute *preview* tier (Step 4).
 *
 * It evaluates a material's per-cell channels on the GPU (one compute thread per
 * texel, shader codegen'd from the expression trees) and then runs the same CPU
 * {@link finishMaps} the tile pool does for relief + emissive normalization. The
 * GPU result is an APPROXIMATE preview — GPU transcendentals and fused
 * multiply-adds diverge from the CPU/libm engine beyond the cross-engine
 * tolerance — so it is wired only as the low-res preview backend; the
 * authoritative, parity-faithful frame always comes from a CPU tier.
 *
 * Robustness: anything the GPU can't do — no WebGPU, device-init failure, a
 * material using an un-codegen'able node (noise/shapes/bricks), or a dispatch
 * error — transparently falls back to a wrapped CPU backend, so a frame is
 * always produced and is always correct.
 *
 * The WebGPU dispatch path runs only in a browser with a GPU; the node test
 * runner has no WebGPU, so it is exercised by the fallback/capability tests plus
 * manual browser verification. The codegen and shader assembly it depends on are
 * unit-tested directly.
 */
import {
  finishMaps,
  resolveRegion,
  IOR_PACK_MAX,
  type ChannelData,
  type MaterialMaps,
} from "../engine";
import { specToLayers, type MaterialSpec } from "../eval-core";
import type { EvalBackend } from "../eval/backend";
import { buildComputeShader, WORKGROUP } from "./shader";
import { GpuUnsupportedError } from "./wgsl-codegen";

// --- Minimal WebGPU surface (no @webgpu/types dependency) -------------------
// Standardized GPUBufferUsage bit flags and GPUMapMode.READ.
const USAGE = {
  STORAGE: 0x80,
  UNIFORM: 0x40,
  COPY_DST: 0x08,
  COPY_SRC: 0x04,
  MAP_READ: 0x01,
};
const MAP_READ = 0x01;

interface GpuBuffer {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}
interface GpuQueue {
  writeBuffer(b: GpuBuffer, offset: number, data: BufferSource): void;
  submit(cmds: unknown[]): void;
}
interface GpuDevice {
  createShaderModule(d: { code: string }): unknown;
  createBuffer(d: { size: number; usage: number }): GpuBuffer;
  createComputePipeline(d: unknown): { getBindGroupLayout(i: number): unknown };
  createBindGroup(d: unknown): unknown;
  createCommandEncoder(): GpuCommandEncoder;
  queue: GpuQueue;
  destroy?(): void;
}
interface GpuCommandEncoder {
  beginComputePass(): GpuComputePass;
  copyBufferToBuffer(
    s: GpuBuffer,
    so: number,
    d: GpuBuffer,
    dof: number,
    size: number,
  ): void;
  finish(): unknown;
}
interface GpuComputePass {
  setPipeline(p: unknown): void;
  setBindGroup(i: number, g: unknown): void;
  dispatchWorkgroups(x: number, y: number, z?: number): void;
  end(): void;
}
interface GpuAdapter {
  requestDevice(): Promise<GpuDevice>;
}
interface Gpu {
  requestAdapter(): Promise<GpuAdapter | null>;
}

function getGpu(): Gpu | null {
  const nav =
    typeof navigator !== "undefined" ? (navigator as { gpu?: Gpu }) : null;
  return nav?.gpu ?? null;
}

/** Whether the environment exposes a WebGPU entry point at all. */
export function webgpuSupported(): boolean {
  return getGpu() !== null;
}

interface CachedPipeline {
  pipeline: { getBindGroupLayout(i: number): unknown };
}

export class GpuBackend implements EvalBackend {
  readonly kind = "gpu";
  private device: GpuDevice | null = null;
  private initPromise: Promise<boolean> | null = null;
  // Compute pipelines keyed by shader source (one per distinct material shape).
  private pipelines = new Map<string, CachedPipeline>();

  /** @param fallback the CPU backend used whenever the GPU path can't run. */
  constructor(private readonly fallback: EvalBackend) {}

  private async ensureDevice(): Promise<boolean> {
    if (this.device) return true;
    if (this.initPromise) return this.initPromise;
    this.initPromise = (async (): Promise<boolean> => {
      const gpu = getGpu();
      if (!gpu) return false;
      try {
        const adapter = await gpu.requestAdapter();
        if (!adapter) return false;
        this.device = await adapter.requestDevice();
        return true;
      } catch {
        return false;
      }
    })();
    return this.initPromise;
  }

  /** Whether a spec's channels can all be codegen'd to WGSL (else CPU fallback). */
  supports(spec: MaterialSpec): boolean {
    try {
      buildComputeShader(specToLayers(spec));
      return true;
    } catch (e) {
      if (e instanceof GpuUnsupportedError) return false;
      throw e;
    }
  }

  async evaluate(spec: MaterialSpec): Promise<MaterialMaps> {
    if (await this.ensureDevice()) {
      try {
        const shader = buildComputeShader(specToLayers(spec)); // may throw GpuUnsupported
        const ch = await this.runCompute(spec, shader);
        return finishMaps(
          ch,
          spec.width,
          spec.height,
          resolveRegion(spec.region),
        );
      } catch (e) {
        if (!(e instanceof GpuUnsupportedError))
          console.warn("WebGPU eval failed; using CPU fallback.", e);
        // fall through
      }
    }
    return this.fallback.evaluate(spec);
  }

  private getPipeline(shader: string): CachedPipeline {
    const hit = this.pipelines.get(shader);
    if (hit) return hit;
    const device = this.device!;
    const module = device.createShaderModule({ code: shader });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const cached: CachedPipeline = { pipeline };
    this.pipelines.set(shader, cached);
    return cached;
  }

  private async runCompute(
    spec: MaterialSpec,
    shader: string,
  ): Promise<ChannelData> {
    const device = this.device!;
    const { width: w, height: h } = spec;
    const n = w * h;
    const { pipeline } = this.getPipeline(shader);

    // Uniform: Dims { w, h, x1, x2, y1, y2, yUp, pad } (32 bytes).
    const r = resolveRegion(spec.region);
    const dims = new ArrayBuffer(32);
    const dv = new DataView(dims);
    dv.setUint32(0, w, true);
    dv.setUint32(4, h, true);
    dv.setFloat32(8, r.x1, true);
    dv.setFloat32(12, r.x2, true);
    dv.setFloat32(16, r.y1, true);
    dv.setFloat32(20, r.y2, true);
    dv.setUint32(24, r.yUp ? 1 : 0, true);
    const dimsBuf = device.createBuffer({
      size: 32,
      usage: USAGE.UNIFORM | USAGE.COPY_DST,
    });
    device.queue.writeBuffer(dimsBuf, 0, dims);

    // Output storage buffers: basecolor/orm packed u32 (4 bytes/cell), emf f32×3,
    // height f32. Each also COPY_SRC so it can be staged for readback.
    const storage = USAGE.STORAGE | USAGE.COPY_SRC;
    const bcBuf = device.createBuffer({ size: n * 4, usage: storage });
    const ormBuf = device.createBuffer({ size: n * 4, usage: storage });
    const emfBuf = device.createBuffer({ size: n * 3 * 4, usage: storage });
    const hBuf = device.createBuffer({ size: n * 4, usage: storage });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dimsBuf } },
        { binding: 1, resource: { buffer: bcBuf } },
        { binding: 2, resource: { buffer: ormBuf } },
        { binding: 3, resource: { buffer: emfBuf } },
        { binding: 4, resource: { buffer: hBuf } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(w / WORKGROUP), Math.ceil(h / WORKGROUP));
    pass.end();

    // Stage every output for MAP_READ readback.
    const mk = (size: number): GpuBuffer =>
      device.createBuffer({ size, usage: USAGE.COPY_DST | USAGE.MAP_READ });
    const bcRead = mk(n * 4);
    const ormRead = mk(n * 4);
    const emfRead = mk(n * 3 * 4);
    const hRead = mk(n * 4);
    enc.copyBufferToBuffer(bcBuf, 0, bcRead, 0, n * 4);
    enc.copyBufferToBuffer(ormBuf, 0, ormRead, 0, n * 4);
    enc.copyBufferToBuffer(emfBuf, 0, emfRead, 0, n * 3 * 4);
    enc.copyBufferToBuffer(hBuf, 0, hRead, 0, n * 4);
    device.queue.submit([enc.finish()]);

    const readBytes = async (
      buf: GpuBuffer,
      size: number,
    ): Promise<ArrayBuffer> => {
      await buf.mapAsync(MAP_READ);
      const copy = buf.getMappedRange().slice(0, size);
      buf.unmap();
      buf.destroy();
      return copy;
    };
    const [bcAB, ormAB, emfAB, hAB] = await Promise.all([
      readBytes(bcRead, n * 4),
      readBytes(ormRead, n * 4),
      readBytes(emfRead, n * 3 * 4),
      readBytes(hRead, n * 4),
    ]);

    // Release device-side buffers.
    for (const b of [dimsBuf, bcBuf, ormBuf, emfBuf, hBuf]) b.destroy();

    // The GPU preview tier does not compute the OpenPBR extras (sheen / coat /
    // transmission / subsurface / anisotropy / ior); emit inert defaults so the
    // ChannelData is well-formed. The CPU tiers are the authoritative path for
    // these channels.
    const pbr1 = new Uint8Array(n * 4);
    const pbr2 = new Uint8Array(n * 4);
    const iorByte = Math.round(((1.5 - 1) / (IOR_PACK_MAX - 1)) * 255);
    for (let i = 0; i < n; i++) {
      pbr2[i * 4 + 1] = iorByte;
      pbr2[i * 4 + 3] = 255;
    }

    return {
      basecolor: new Uint8Array(bcAB), // packed RGBA8 little-endian → [r,g,b,a,…]
      orm: new Uint8Array(ormAB),
      pbr1,
      pbr2,
      emf: new Float32Array(emfAB),
      height: new Float32Array(hAB),
    };
  }

  dispose(): void {
    this.pipelines.clear();
    this.device?.destroy?.();
    this.device = null;
    this.fallback.dispose();
  }
}
