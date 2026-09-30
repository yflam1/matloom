/**
 * crop-controller.ts — drives "crop" rendering.
 *
 * When enabled, it watches the camera (orbit / zoom / pan / resize) and keeps
 * the *rendered region* matched to what's actually on screen: the whole authored
 * region while it fits the viewport, and just the visible sub-rectangle once the
 * user zooms in past it. The render *resolution* is never touched — it stays the
 * user's manual width/height — so zooming in spreads the same texels over a
 * smaller slice of the plane and sharpens for free, at the cost the user picked.
 *
 * The geometry (region → screen, screen → plane) is measured by the viewer
 * (viewer.ts `measureCrop`); the policy that turns a measurement into a region
 * is pure (viewer/crop.ts). This class is the glue: debounce camera events,
 * dedupe redundant regions, push the result to the evaluator (region override)
 * and — crucially — leave the camera entirely under the user's control while on
 * (`setAutoFrame(false)`), so nothing fights their orbit/zoom.
 *
 * The authored View (store.settings.x1..y2) and the manual resolution are never
 * mutated; the View remains the "full" extent the policy measures against and is
 * restored verbatim when crop mode is turned off.
 */
import {
  DEFAULT_CROP,
  computeCropRegion,
  regionsEqual,
  type CropConfig,
  type Region,
} from "../viewer/crop";
import type { Evaluator } from "./eval";
import type { ViewerHandle } from "../viewer";
import type { Store } from "./state";

// Camera events fire in bursts during a drag; coalesce them with a short delay
// before re-measuring. Kept small so the render feels prompt once the view
// settles (the controller renders immediately afterwards, with no extra debounce).
const CAMERA_DEBOUNCE_MS = 50;

export class CropController {
  private enabled = false;
  private unsub: (() => void) | null = null;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private last: Region | null = null;

  constructor(
    private readonly store: Store,
    private readonly evaluator: Evaluator,
    private readonly getViewer: () => ViewerHandle,
    private readonly cfg: CropConfig = DEFAULT_CROP,
  ) {}

  isEnabled(): boolean {
    return this.enabled;
  }

  setEnabled(on: boolean): void {
    if (on === this.enabled) return;
    this.enabled = on;
    const v = this.getViewer();
    if (on) {
      // Keep the user's current camera — toggling crop on must not reset the
      // view. Just stop the automatic relief-driven reframe (so crops don't move
      // the camera) and start measuring the crop from wherever the view already
      // is. The first computeAndApply repositions the plane via the frame.
      v.setAutoFrame(false);
      this.subscribe();
      this.last = null;
      this.computeAndApply(true);
    } else {
      this.unsubscribe();
      clearTimeout(this.timer);
      this.last = null;
      v.setAutoFrame(true);
      // Restore the full authored region without touching the camera — toggling
      // crop off leaves the view exactly where it is. Clearing the override makes
      // the next render use the full region; renderNow repositions the plane to it
      // immediately (no debounce gap showing the stale crop).
      this.evaluator.setCropRegion(null);
      this.evaluator.renderNow();
    }
  }

  /**
   * Re-apply the crop state to a freshly built viewer (after a WebGL/WebGPU
   * backend switch rebuilds it). No-op when disabled.
   */
  rebind(): void {
    if (!this.enabled) return;
    this.unsubscribe();
    const v = this.getViewer();
    v.setAutoFrame(false);
    this.frameFull();
    this.subscribe();
    this.last = null;
    this.computeAndApply(true);
  }

  /**
   * Recompute from the current authored region — call after the user edits the
   * region / Y direction (or imports a View) while crop mode is on. Re-frames
   * the camera on the new full region (so it stays centered on the plane), then
   * re-derives the crop. No-op when disabled.
   */
  refresh(): void {
    if (!this.enabled) return;
    this.frameFull();
    this.last = null;
    this.computeAndApply(true);
  }

  /** Reset the camera to its home view (full region) and re-derive the crop. */
  resetView(): void {
    this.getViewer().resetView();
    this.last = null;
    this.computeAndApply(true);
  }

  // Frame the camera on the full authored region (and reset the plane geometry to
  // it). The next computeAndApply re-derives the crop from there.
  private frameFull(): void {
    const { x1, y1, x2, y2 } = this.store.settings;
    this.getViewer().setRegion(x1, y1, x2, y2);
  }

  private subscribe(): void {
    this.unsub = this.getViewer().onCameraChange(() => this.onCamera());
  }

  private unsubscribe(): void {
    this.unsub?.();
    this.unsub = null;
  }

  private onCamera(): void {
    clearTimeout(this.timer);
    this.timer = setTimeout(
      () => this.computeAndApply(false),
      CAMERA_DEBOUNCE_MS,
    );
  }

  private fullRegion(): Region {
    const { x1, y1, x2, y2 } = this.store.settings;
    return { x1, y1, x2, y2 };
  }

  // Measure → derive the region → (if changed) push it to the evaluator and
  // schedule a single render. `force` bypasses the dedupe (used on enable/refresh).
  // The plane geometry is *not* moved here — it follows the texture in applyFrame
  // (region travels with the frame), so a crop never flashes a stale stretch.
  private computeAndApply(force: boolean): void {
    if (!this.enabled) return;
    const v = this.getViewer();
    const full = this.fullRegion();
    const { width: w, height: h } = this.store.settings;
    const region = computeCropRegion(v.measureCrop(full), full, this.cfg, {
      w,
      h,
    });
    if (!region) return; // degenerate projection — keep the previous region
    if (!force && regionsEqual(region, this.last)) return;
    this.last = region;

    this.evaluator.setCropRegion(region);
    // Render immediately — onCamera already debounced the motion, so a second
    // (scheduleEval) debounce would just add lag after the view settles.
    this.evaluator.renderNow();
  }
}
