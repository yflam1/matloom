/**
 * divider.ts — drag the vertical divider to resize the layer panel.
 */
import { byId } from "./dom";

export function initDivider(): void {
  const divider = byId("divider");
  const panel = byId("panel");
  let dragging = false;

  divider.addEventListener("mousedown", (e) => {
    dragging = true;
    e.preventDefault();
    document.body.style.cursor = "col-resize";
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    // Drive width through a custom property (not an inline `width`, which would
    // out-rank the collapse/drawer rules in the cascade and block collapsing).
    panel.style.setProperty(
      "--panel-width",
      Math.min(600, Math.max(240, e.clientX)) + "px",
    );
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    document.body.style.cursor = "";
  });
}
