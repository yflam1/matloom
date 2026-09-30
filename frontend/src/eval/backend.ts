/**
 * backend.ts — pluggable evaluation backends. Every backend takes a
 * {@link MaterialSpec} and returns the composited {@link MaterialMaps}; they
 * differ only in *where* and *how* the work runs:
 *
 *   - `sync`   — on the main thread (fallback; used where workers are absent).
 *   - `worker` — on one dedicated worker, off the UI thread (Step 2).
 *   - `pool`   — row-tiled across N workers for multi-core throughput (Step 3).
 *   - `gpu`    — a WebGPU compute *preview* tier (Step 4); NOT parity-faithful.
 *
 * The `sync`, `worker` and `pool` tiers all run the identical `eval-core`
 * engine, so they are byte-for-byte equivalent — only the GPU tier diverges
 * (documented on {@link GpuBackend}).
 */
import { SpecCache, type MaterialSpec } from "../eval-core";
import type { MaterialMaps } from "../engine";
import {
  isStripeResponse,
  type EvalRequest,
  type EvalResponse,
} from "./protocol";

export type EvalBackendKind = "sync" | "worker" | "pool" | "gpu";

export interface EvalBackend {
  readonly kind: EvalBackendKind;
  /** Evaluate a spec to its texture maps. */
  evaluate(spec: MaterialSpec): Promise<MaterialMaps>;
  /** Release any owned workers / GPU resources. */
  dispose(): void;
}

/** Main-thread evaluation. Always available; the universal fallback. */
export class SyncBackend implements EvalBackend {
  readonly kind = "sync";
  private cache = new SpecCache();
  evaluate(spec: MaterialSpec): Promise<MaterialMaps> {
    return Promise.resolve(this.cache.eval(spec));
  }
  dispose(): void {}
}

/** Spawn the dedicated eval worker (Vite resolves the URL at build time). */
export function spawnEvalWorker(): Worker {
  return new Worker(new URL("./eval.worker.ts", import.meta.url), {
    type: "module",
  });
}

/**
 * One dedicated worker. Keeps every in-flight request keyed by id, so the
 * caller can fire a new evaluation before an old one returns; the app's
 * generation guard then discards superseded results. The worker's own
 * {@link SpecCache} keeps unchanged channels memoized across messages.
 */
export class WorkerBackend implements EvalBackend {
  readonly kind = "worker";
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (m: MaterialMaps) => void; reject: (e: unknown) => void }
  >();

  constructor(worker: Worker = spawnEvalWorker()) {
    this.worker = worker;
    this.worker.onmessage = (e: MessageEvent<EvalResponse>): void => {
      const data = e.data;
      if (isStripeResponse(data)) return; // a single worker only issues full requests
      const p = this.pending.get(data.id);
      if (!p) return;
      this.pending.delete(data.id);
      p.resolve(data.maps);
    };
    this.worker.onerror = (e: ErrorEvent): void => {
      // A worker-level failure poisons every outstanding request; surface it so
      // the app can show an error rather than hang.
      const err = new Error(e.message || "eval worker error");
      for (const { reject } of this.pending.values()) reject(err);
      this.pending.clear();
    };
  }

  evaluate(spec: MaterialSpec): Promise<MaterialMaps> {
    const id = this.nextId++;
    return new Promise<MaterialMaps>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const req: EvalRequest = { id, spec };
      this.worker.postMessage(req);
    });
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}

/** Whether dedicated workers are available in this environment. */
export function workersSupported(): boolean {
  return typeof Worker !== "undefined";
}

/**
 * Create a backend of the requested kind, degrading gracefully: an unsupported
 * or failed worker/pool/gpu tier falls back to the next simpler tier so the
 * viewer always renders. `gpu` and `pool` are loaded lazily by their owners
 * (Steps 3–4); this factory covers the always-present `sync`/`worker` tiers.
 */
export function createCpuBackend(kind: "sync" | "worker"): EvalBackend {
  if (kind === "worker" && workersSupported()) {
    try {
      return new WorkerBackend();
    } catch {
      // Worker construction can throw under strict CSPs; fall back to sync.
    }
  }
  return new SyncBackend();
}
