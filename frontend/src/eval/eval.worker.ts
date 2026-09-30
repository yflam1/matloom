/**
 * eval.worker.ts — the dedicated worker that evaluates a material off the main
 * thread, so a full-resolution re-render never freezes the UI. It serves both
 * CPU tiers: a plain request returns the finished maps; a banded request (tile
 * pool) returns just the per-cell channel stripe for its rows.
 *
 * It owns one persistent {@link SpecCache}, so across edits an unchanged channel
 * reuses its parsed tree and memoized noise grids — the same incremental
 * behavior the main thread had, now off-thread. Results are transferred back
 * zero-copy via their backing ArrayBuffers.
 *
 * This file is browser-only (it is never imported by the node test runner). Its
 * computation lives entirely in `eval-core` / `engine`, which *are* unit-tested;
 * the worker is the thin message-loop glue around them.
 */
import { SpecCache, mapsTransferList, channelTransferList } from "../eval-core";
import type { EvalRequest, EvalResponse } from "./protocol";

// Type the dedicated-worker globals without pulling the "WebWorker" lib (which
// would clash with the "DOM" lib this project compiles against).
const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<EvalRequest>) => void) | null;
  postMessage: (message: EvalResponse, transfer: Transferable[]) => void;
};

const cache = new SpecCache();

ctx.onmessage = (e: MessageEvent<EvalRequest>): void => {
  const { id, spec, band } = e.data;
  if (band) {
    const stripe = cache.evalStripe(spec, band);
    ctx.postMessage({ id, stripe, band }, channelTransferList(stripe));
  } else {
    const maps = cache.eval(spec);
    ctx.postMessage({ id, maps }, mapsTransferList(maps));
  }
};
