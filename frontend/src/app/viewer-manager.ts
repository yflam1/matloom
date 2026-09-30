/**
 * viewer-manager.ts — owns the single ViewerHandle and the WebGL/WebGPU switch.
 *
 * A <canvas> permanently binds to its first graphics context, so switching
 * backends means swapping in a brand-new canvas and rebuilding the viewer on
 * it. The freshly built viewer is re-synced to the current region/Y/axes state
 * and the scene is re-evaluated so the new backend shows the same material.
 */
import { initViewer, type Backend, type ViewerHandle } from "../viewer";
import { byId } from "./dom";
import type { Store } from "./state";
import type { Evaluator } from "./eval";
import type { CropController } from "./crop-controller";
import { WebGPUEngine } from "../viewer/babylon";
import { createCpuBackend } from "../eval/backend";
import { GpuBackend, webgpuSupported } from "../gpu/gpu-backend";

export class ViewerManager {
  private viewer: ViewerHandle | null = null;
  private evaluator: Evaluator | null = null;
  private crop: CropController | null = null;

  constructor(private readonly store: Store) {}

  /** The evaluator is created after the manager, so it is injected here. */
  setEvaluator(evaluator: Evaluator): void {
    this.evaluator = evaluator;
  }

  /** The crop controller (rebound to the fresh viewer on backend switch). */
  setCrop(crop: CropController): void {
    this.crop = crop;
  }

  /** Accessor handed to modules that render frames; viewer must exist by then. */
  get(): ViewerHandle {
    if (!this.viewer) throw new Error("viewer not initialized");
    return this.viewer;
  }

  /** Build the initial viewer on the existing canvas and sync region state. */
  async init(): Promise<void> {
    this.viewer = await initViewer(this.store.settings.backend);
    const { x1, y1, x2, y2 } = this.store.settings;
    this.viewer.setRegion(x1, y1, x2, y2);
    this.syncEvalPreviewBackend();
  }

  /** Wire the WebGL/WebGPU buttons, disabling WebGPU when unsupported. */
  async initSwitch(): Promise<void> {
    const gpuBtn = byId<HTMLButtonElement>("backend-webgpu");
    const glBtn = byId<HTMLButtonElement>("backend-webgl");
    const supported =
      typeof WebGPUEngine !== "undefined" &&
      (await WebGPUEngine.IsSupportedAsync);

    if (!supported) {
      gpuBtn.disabled = true;
      byId("backend-switch").setAttribute(
        "data-tip",
        "Your device doesn't support WebGPU",
      );
    } else {
      gpuBtn.addEventListener("click", () => this.setBackend("webgpu"));
    }
    glBtn.addEventListener("click", () => this.setBackend("webgl"));
    this.syncButtons();
  }

  private async setBackend(backend: Backend): Promise<void> {
    if (backend === this.store.settings.backend) return;
    this.store.settings.backend = backend;
    this.syncButtons();
    if (this.viewer) this.viewer.dispose();
    await this.build(backend);
    this.syncEvalPreviewBackend();
    // Re-apply crop state (camera subscription, autoFrame, crop) to the fresh
    // viewer; when crop is off this is a no-op and the plain eval below runs.
    this.crop?.rebind();
    this.evaluator?.scheduleEval();
  }

  // Tie the GPU compute *preview* tier to the WebGPU render backend: when the
  // user selects WebGPU (and it's supported), low-res preview frames are
  // evaluated on the GPU (with a CPU worker fallback for materials the GPU path
  // can't run); the authoritative full-res frame always stays on the CPU pool.
  // Selecting WebGL clears the preview tier (previews then use the settle pool).
  private syncEvalPreviewBackend(): void {
    if (!this.evaluator) return;
    const useGpu =
      this.store.settings.backend === "webgpu" && webgpuSupported();
    this.evaluator.setPreviewBackend(
      useGpu ? new GpuBackend(createCpuBackend("worker")) : null,
    );
  }

  // Swap in a fresh canvas (a canvas can't change context type) and rebuild.
  private async build(backend: Backend): Promise<void> {
    const oldCanvas = byId<HTMLCanvasElement>("render-canvas");
    const fresh = oldCanvas.cloneNode(false) as HTMLCanvasElement; // copy id + attrs, drop GL state
    oldCanvas.replaceWith(fresh);

    const { settings } = this.store;
    this.viewer = await initViewer(backend, fresh);
    this.viewer.setRegion(settings.x1, settings.y1, settings.x2, settings.y2);
    this.viewer.setYAxisDirection(settings.yUp);
    this.viewer.setAxesVisible(settings.axes);
    this.viewer.setAmbientEnabled(settings.ambient);
    this.viewer.setDirectEnabled(settings.direct);
    this.viewer.setNormalEnabled(settings.normal);
    this.viewer.setOcclusionEnabled(settings.occlusion);
    this.viewer.setDisplaceEnabled(settings.displace);
  }

  private syncButtons(): void {
    for (const btn of document.querySelectorAll<HTMLButtonElement>(".bk-opt"))
      btn.classList.toggle(
        "active",
        btn.dataset.backend === this.store.settings.backend,
      );
  }
}
