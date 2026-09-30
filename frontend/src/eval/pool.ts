/**
 * pool.ts — the tile-parallel evaluation backend: the numba `parallel=True`
 * analog for the browser. It splits the grid into contiguous row bands, fans
 * each out to a worker that returns just that band's per-cell channel stripe,
 * then reassembles the stripes and runs the whole-grid finishing pass (relief +
 * emissive normalization) once on the main thread.
 *
 * Why this is byte-identical to single-threaded evaluation:
 *   - every per-cell output depends only on its own (x, y), so a stripe computes
 *     exactly the rows it owns — no cross-tile coupling, no reduction reordering;
 *   - the one whole-grid reduction (the global emissive max) and the one
 *     neighborhood op (the AO box blur + normal central differences) run after
 *     assembly, on the complete field, exactly as before.
 *
 * Transfer is zero-copy via transferable ArrayBuffers, so this needs no
 * SharedArrayBuffer and therefore no COOP/COEP cross-origin isolation — which
 * matters because GitHub Pages (where this deploys) cannot set those headers.
 *
 * Bands are assigned to workers by index, so a given worker keeps evaluating the
 * same rows across edits and its per-band noise memo keeps hitting.
 */
import {
  assembleChannels,
  finishMaps,
  resolveRegion,
  type ChannelData,
  type MaterialMaps,
} from "../engine";
import type { MaterialSpec, RowBand } from "../eval-core";
import { spawnEvalWorker, type EvalBackend } from "./backend";
import {
  isStripeResponse,
  type EvalRequest,
  type EvalResponse,
} from "./protocol";

/**
 * Partition `h` rows into at most `count` contiguous bands of near-equal height
 * (the first `h % count` bands get one extra row). Never returns empty bands, so
 * a grid shorter than `count` simply uses fewer workers.
 */
export function splitBands(h: number, count: number): RowBand[] {
  const k = Math.min(Math.max(1, count), Math.max(1, h));
  const base = Math.floor(h / k);
  const extra = h % k;
  const bands: RowBand[] = [];
  let y = 0;
  for (let i = 0; i < k; i++) {
    const rows = base + (i < extra ? 1 : 0);
    bands.push({ y0: y, y1: y + rows });
    y += rows;
  }
  return bands;
}

interface PoolJob {
  resolve: (m: MaterialMaps) => void;
  reject: (e: unknown) => void;
  spec: MaterialSpec;
  parts: { y0: number; data: ChannelData }[];
  want: number;
}

export class PoolBackend implements EvalBackend {
  readonly kind = "pool";
  private workers: Worker[];
  private nextId = 1;
  private jobs = new Map<number, PoolJob>();

  constructor(size: number, spawn: () => Worker = spawnEvalWorker) {
    this.workers = Array.from({ length: Math.max(1, size) }, () => spawn());
    for (const w of this.workers) {
      w.onmessage = (e: MessageEvent<EvalResponse>): void =>
        this.onMessage(e.data);
      w.onerror = (e: ErrorEvent): void =>
        this.failAll(new Error(e.message || "pool worker error"));
    }
  }

  /** Number of workers in the pool. */
  get size(): number {
    return this.workers.length;
  }

  evaluate(spec: MaterialSpec): Promise<MaterialMaps> {
    const bands = splitBands(spec.height, this.workers.length);
    const id = this.nextId++;
    return new Promise<MaterialMaps>((resolve, reject) => {
      this.jobs.set(id, {
        resolve,
        reject,
        spec,
        parts: [],
        want: bands.length,
      });
      bands.forEach((band, i) => {
        const req: EvalRequest = { id, spec, band };
        this.workers[i].postMessage(req);
      });
    });
  }

  private onMessage(data: EvalResponse): void {
    if (!isStripeResponse(data)) return; // pool requests are always banded
    const job = this.jobs.get(data.id);
    if (!job) return;
    job.parts.push({ y0: data.band.y0, data: data.stripe });
    if (job.parts.length < job.want) return;
    this.jobs.delete(data.id);
    const ch = assembleChannels(job.parts, job.spec.width, job.spec.height);
    job.resolve(
      finishMaps(
        ch,
        job.spec.width,
        job.spec.height,
        resolveRegion(job.spec.region),
      ),
    );
  }

  private failAll(err: Error): void {
    for (const job of this.jobs.values()) job.reject(err);
    this.jobs.clear();
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
    this.jobs.clear();
  }
}

/** Default pool size: leave one core for the main thread, capped to keep spawn cost bounded. */
export function defaultPoolSize(): number {
  const cores =
    (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
  return Math.min(8, Math.max(1, cores - 1));
}

/** Create a tile-parallel pool backend (defaults to {@link defaultPoolSize}). */
export function createPoolBackend(
  size: number = defaultPoolSize(),
): PoolBackend {
  return new PoolBackend(size);
}
