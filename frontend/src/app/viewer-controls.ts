/**
 * viewer-controls.ts — the View section of the left panel (above Definitions):
 * region and resolution number inputs, a Crop toggle (render only the visible
 * region), the Axes / Y-direction / Ambient / Direct / Normal / Occlusion /
 * Displace toggle buttons, a Reset-view action (snaps the camera back to its
 * home angle), and their keyboard shortcuts (X / Y / A / D / N / O / H / R,
 * plus C for Crop). Keeping these off the canvas leaves the material
 * unobstructed at every size.
 */
import { byId } from "./dom";
import { DEFAULTS, type Store } from "./state";
import type { Evaluator } from "./eval";
import type { CropController } from "./crop-controller";
import type { ViewerHandle } from "../viewer";

// Keys bound to numeric inputs (a subset of Settings with numeric values).
type NumericKey = "x1" | "y1" | "x2" | "y2" | "width" | "height";

export class ViewerControls {
  constructor(
    private readonly store: Store,
    private readonly evaluator: Evaluator,
    private readonly getViewer: () => ViewerHandle,
    private readonly crop: CropController,
  ) {}

  init(): void {
    this.bindNumber("in-x1", "x1");
    this.bindNumber("in-x2", "x2");
    this.bindNumber("in-y1", "y1");
    this.bindNumber("in-y2", "y2");
    this.bindNumber("in-w", "width", true);
    this.bindNumber("in-h", "height", true);

    // Resolution inputs: limit to 4 digits (max 9999).
    const resInputs = ["in-w", "in-h"];
    for (const id of resInputs) {
      const el = byId<HTMLInputElement>(id);
      el.addEventListener("input", () => {
        const raw = el.value.slice(0, 4);
        if (raw !== el.value) {
          el.value = raw;
        }
      });
    }

    byId("btn-axes").addEventListener("click", () =>
      this.setAxes(!this.store.settings.axes),
    );
    byId("btn-yflip").addEventListener("click", () =>
      this.setYUp(!this.store.settings.yUp),
    );
    byId("btn-ambient").addEventListener("click", () =>
      this.setAmbient(!this.store.settings.ambient),
    );
    byId("btn-direct").addEventListener("click", () =>
      this.setDirect(!this.store.settings.direct),
    );
    byId("btn-normal").addEventListener("click", () =>
      this.setNormal(!this.store.settings.normal),
    );
    byId("btn-occlusion").addEventListener("click", () =>
      this.setOcclusion(!this.store.settings.occlusion),
    );
    byId("btn-displace").addEventListener("click", () =>
      this.setDisplace(!this.store.settings.displace),
    );
    byId("btn-crop").addEventListener("click", () =>
      this.setCrop(!this.store.settings.crop),
    );
    byId("btn-swap-x").addEventListener("click", () => this.swapX());
    byId("btn-swap-y").addEventListener("click", () => this.swapY());
    byId("btn-reset").addEventListener("click", () => this.resetView());
    window.addEventListener("keydown", (e) => this.onKey(e));

    this.setAxes(this.store.settings.axes);
    this.setYUp(this.store.settings.yUp);
    this.setAmbient(this.store.settings.ambient);
    this.setDirect(this.store.settings.direct);
    this.setNormal(this.store.settings.normal);
    this.setOcclusion(this.store.settings.occlusion);
    this.setDisplace(this.store.settings.displace);
    this.setCrop(this.store.settings.crop);
  }

  /**
   * Re-read region + scale from the store into the inputs and the viewer, and
   * re-render. Called after an import applies a `View(...)` to `store.settings`.
   */
  syncFromSettings(): void {
    const { settings } = this.store;
    this.syncInput("in-x1", settings.x1);
    this.syncInput("in-y1", settings.y1);
    this.syncInput("in-x2", settings.x2);
    this.syncInput("in-y2", settings.y2);
    // In crop mode the controller owns the rendered region; just recompute
    // from the new authored extent. Otherwise commit the region and re-render.
    if (settings.crop) {
      this.crop.refresh();
      return;
    }
    this.getViewer().setRegion(
      settings.x1,
      settings.y1,
      settings.x2,
      settings.y2,
    );
    this.evaluator.scheduleEval();
  }

  private setAxes(visible: boolean): void {
    this.store.settings.axes = visible;
    this.getViewer().setAxesVisible(visible);
    byId("btn-axes").classList.toggle("active", visible);
  }

  private setYUp(yUp: boolean): void {
    this.store.settings.yUp = yUp;
    const btn = byId("btn-yflip");
    btn.classList.toggle("active", yUp);
    btn.querySelector(".tool-ic")!.textContent = yUp ? "↑" : "↓";
    btn.querySelector(".tool-txt")!.textContent = yUp ? "Y-up" : "Y-down";
    this.getViewer().setYAxisDirection(yUp);
    if (this.store.settings.crop) this.crop.refresh();
    else this.evaluator.scheduleEval();
  }

  private setAmbient(on: boolean): void {
    this.store.settings.ambient = on;
    this.getViewer().setAmbientEnabled(on);
    byId("btn-ambient").classList.toggle("active", on);
  }

  private setDirect(on: boolean): void {
    this.store.settings.direct = on;
    this.getViewer().setDirectEnabled(on);
    byId("btn-direct").classList.toggle("active", on);
  }

  private setNormal(on: boolean): void {
    this.store.settings.normal = on;
    this.getViewer().setNormalEnabled(on);
    byId("btn-normal").classList.toggle("active", on);
  }

  private setOcclusion(on: boolean): void {
    this.store.settings.occlusion = on;
    this.getViewer().setOcclusionEnabled(on);
    byId("btn-occlusion").classList.toggle("active", on);
  }

  private setDisplace(on: boolean): void {
    this.store.settings.displace = on;
    this.getViewer().setDisplaceEnabled(on);
    byId("btn-displace").classList.toggle("active", on);
  }

  // Crop rendering. The manual W/H inputs stay live (the user owns the
  // resolution at all times); turning this on only narrows the rendered region
  // to what's visible, and turning it off restores the full authored region.
  private setCrop(on: boolean): void {
    this.store.settings.crop = on;
    byId("btn-crop").classList.toggle("active", on);
    this.crop.setEnabled(on);
  }

  // After a region / Y-direction edit: in crop mode the controller recomputes
  // the rendered region from the new authored extent; otherwise commit the region
  // to the viewer and re-render directly. Reaching here means the user authored a
  // region edit (input/swap), so mark the view authored for export.
  private afterRegionChange(lowRes: boolean): void {
    this.store.settings.viewSet = true;
    if (this.store.settings.crop) {
      this.crop.refresh();
      return;
    }
    const { settings } = this.store;
    this.getViewer().setRegion(
      settings.x1,
      settings.y1,
      settings.x2,
      settings.y2,
    );
    this.evaluator.scheduleEval(lowRes);
  }

  private bindNumber(id: string, key: NumericKey, integer = false): void {
    const el = byId<HTMLInputElement>(id);
    const { settings } = this.store;
    el.value = String(settings[key]);

    el.addEventListener("input", () => {
      let v = integer ? parseInt(el.value, 10) : parseFloat(el.value);
      if (!Number.isFinite(v)) return; // wait for a complete value
      if (integer) v = Math.max(1, Math.round(v));
      settings[key] = v;
      // Integer keys are width/height (the manual resolution, always live —
      // crop mode keeps using it); region keys re-commit + re-render.
      if (integer) this.evaluator.scheduleEval(true);
      else this.afterRegionChange(true);
    });
    el.addEventListener("change", () => {
      // Normalize the displayed value on blur/commit.
      let v = integer ? parseInt(el.value, 10) : parseFloat(el.value);
      if (!Number.isFinite(v)) v = DEFAULTS[key];
      if (integer) v = Math.max(1, Math.round(v));
      el.value = String(v);
      settings[key] = v;
      if (integer) this.evaluator.scheduleEval();
      else this.afterRegionChange(false);
    });
  }

  private onKey(e: KeyboardEvent): void {
    const t = e.target as HTMLElement | null;
    if (
      t &&
      (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)
    )
      return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "x") {
      e.preventDefault();
      this.setAxes(!this.store.settings.axes);
    } else if (k === "y") {
      e.preventDefault();
      this.setYUp(!this.store.settings.yUp);
    } else if (k === "a") {
      e.preventDefault();
      this.setAmbient(!this.store.settings.ambient);
    } else if (k === "d") {
      e.preventDefault();
      this.setDirect(!this.store.settings.direct);
    } else if (k === "n") {
      e.preventDefault();
      this.setNormal(!this.store.settings.normal);
    } else if (k === "o") {
      e.preventDefault();
      this.setOcclusion(!this.store.settings.occlusion);
    } else if (k === "h") {
      e.preventDefault();
      this.setDisplace(!this.store.settings.displace);
    } else if (k === "c") {
      e.preventDefault();
      this.setCrop(!this.store.settings.crop);
    } else if (k === "r") {
      e.preventDefault();
      this.resetView();
    }
  }

  /**
   * Snap the camera back to its head-on home angle and re-frame to fit the
   * current region. Region, resolution and the display toggles are left
   * untouched — this restores only the orbit/zoom, which has no other way back.
   */
  private resetView(): void {
    // In crop mode the controller resets the camera to the full-region home
    // view and re-derives the crop; otherwise reset the viewer directly.
    if (this.store.settings.crop) {
      this.crop.resetView();
      return;
    }
    this.getViewer().resetView();
  }

  private swapX(): void {
    const { settings } = this.store;
    const tmp = settings.x1;
    settings.x1 = settings.x2;
    settings.x2 = tmp;
    this.syncInput("in-x1", settings.x1);
    this.syncInput("in-x2", settings.x2);
    this.afterRegionChange(false);
  }

  private swapY(): void {
    const { settings } = this.store;
    const tmp = settings.y1;
    settings.y1 = settings.y2;
    settings.y2 = tmp;
    this.syncInput("in-y1", settings.y1);
    this.syncInput("in-y2", settings.y2);
    this.afterRegionChange(false);
  }

  private syncInput(id: string, value: number): void {
    const el = byId<HTMLInputElement>(id);
    el.value = String(value);
  }
}
