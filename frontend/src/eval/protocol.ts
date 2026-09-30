/**
 * protocol.ts — the message shapes exchanged with the eval worker(s).
 *
 * A request carries a self-contained {@link MaterialSpec} and a monotonically
 * increasing `id`; the matching response echoes the `id` so a backend can pair
 * replies with their requests even when several are in flight. The tile pool
 * additionally tags a request with the row {@link RowBand} a worker should
 * compute and gets back a per-cell channel stripe; a single worker leaves
 * `band` unset and gets back the finished maps.
 */
import type { MaterialSpec, RowBand } from "../eval-core";
import type { MaterialMaps, ChannelData } from "../engine";

export type { RowBand };

export interface EvalRequest {
  id: number;
  spec: MaterialSpec;
  /** When set, compute only these rows and return a channel stripe (pool). */
  band?: RowBand;
}

/** Reply to a full-grid request. */
export interface MapsResponse {
  id: number;
  maps: MaterialMaps;
}

/** Reply to a banded (tile-pool) request: the stripe and the rows it covers. */
export interface StripeResponse {
  id: number;
  stripe: ChannelData;
  band: RowBand;
}

export type EvalResponse = MapsResponse | StripeResponse;

/** Narrow a response to the banded (stripe) variant. */
export function isStripeResponse(r: EvalResponse): r is StripeResponse {
  return "stripe" in r;
}
