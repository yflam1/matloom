/**
 * main.ts — application entry point. Builds the shared store, the viewer
 * manager, the evaluator and every UI module, then wires them together in the
 * same order the original app used:
 *   viewer → controls → divider → expr popups → backend switch → first layer.
 */
import "../style.css";
import { createStore } from "./state";
import { createEvaluator } from "./eval";
import { CropController } from "./crop-controller";
import { ViewerManager } from "./viewer-manager";
import { ExprEditorUI } from "./expr-editor";
import { LayerPanel } from "./layer-panel";
import { DefsPanel } from "./defs-panel";
import { ViewerControls } from "./viewer-controls";
import { IoControls } from "./io-controls";
import { initDivider } from "./divider";
import { initLayout } from "./layout";
import { byId } from "./dom";
import { DEFAULT_MATERIAL } from "./default-material";

async function main(): Promise<void> {
  const store = createStore();

  // The viewer manager owns the ViewerHandle (and rebuilds it on backend
  // switch), so everything that renders a frame goes through manager.get().
  const viewerManager = new ViewerManager(store);
  const evaluator = createEvaluator(store, () => viewerManager.get());
  viewerManager.setEvaluator(evaluator);

  // Crop rendering: watches the camera and narrows the rendered region to
  // what's on screen (resolution stays the user's manual setting). Needs the
  // viewer rebound after a backend switch, so the manager holds a reference.
  const crop = new CropController(store, evaluator, () => viewerManager.get());
  viewerManager.setCrop(crop);

  await viewerManager.init();

  const editorUI = new ExprEditorUI();
  const layerPanel = new LayerPanel(store, evaluator, editorUI);
  const defsPanel = new DefsPanel(store, evaluator, editorUI);
  const controls = new ViewerControls(
    store,
    evaluator,
    () => viewerManager.get(),
    crop,
  );
  const ioControls = new IoControls(
    store,
    layerPanel,
    defsPanel,
    evaluator,
    controls,
  );

  layerPanel.init();
  controls.init();
  ioControls.init();
  initDivider();
  initLayout();
  editorUI.init();
  defsPanel.init();
  await viewerManager.initSwitch();

  // Reseed control: visible only while the material uses auto-seeded noise; its
  // count shows how many seeds a click will reroll.
  const reseedBtn = byId<HTMLButtonElement>("btn-reseed");
  const reseedCount = byId("reseed-count");
  reseedBtn.addEventListener("click", () => evaluator.reseed());
  evaluator.onReseedableChange((count) => {
    reseedBtn.hidden = count === 0;
    reseedCount.textContent = String(count);
  });

  // Boot with the default material preloaded (the paper's black slate
  // flooring, seeds included), so the demo opens on a real authored material.
  // Fall back to a plain default layer if the embedded program ever fails to
  // parse — it shouldn't.
  if (ioControls.applyMaterialText(DEFAULT_MATERIAL) !== null) {
    layerPanel.addLayer(); // fallback: one default layer
  }

  // Reveal the app only now: the static shell (viewer/index.html) stays
  // visibility:hidden — with the #boot placeholder covering the page — until
  // the UI is fully wired and the default material is loaded, so nothing
  // half-built ever paints.
  document.getElementById("app")?.classList.add("app-ready");
  document.getElementById("boot")?.classList.add("app-ready");
}

main();
