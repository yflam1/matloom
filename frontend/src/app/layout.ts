/**
 * layout.ts — chrome that adapts the viewer to small screens.
 *
 *   • Collapsible left panel — one state class (`panel-collapsed` on #app) drives
 *     both the desktop "slide shut" and the mobile slide-in drawer. The toolbar
 *     chevron (and a backdrop tap on mobile) collapses it; the floating #panel-fab
 *     reopens it.
 *   • Toolbar overflow menu — the rarely-used Import / Export / Randomize / Renderer controls
 *     live in a #toolbar-menu popover behind the ⋯ button (closes on outside-click
 *     or Escape, and after a menu action opens its own modal).
 *   • Collapsible View section (#viewer-controls, a panel section above
 *     Definitions) via its header toggle.
 *   • Responsive defaults — the panel starts collapsed at the mobile breakpoint
 *     (slide-in drawer) and expanded above it; the View section travels with
 *     the panel, so it needs no breakpoint handling of its own.
 */
import { byId } from "./dom";

const MOBILE_QUERY = "(max-width: 760px)";

export function initLayout(): void {
  const app = byId("app");
  const vc = byId("viewer-controls");

  const setPanelCollapsed = (collapsed: boolean): void =>
    void app.classList.toggle("panel-collapsed", collapsed);

  byId("btn-panel-collapse").addEventListener("click", () =>
    setPanelCollapsed(true),
  );
  byId("panel-fab").addEventListener("click", () => setPanelCollapsed(false));
  byId("panel-backdrop").addEventListener("click", () =>
    setPanelCollapsed(true),
  );

  initOverflowMenu();

  // View section: its header is the collapse toggle.
  const vcToggle = byId("vc-toggle");
  vcToggle.addEventListener("click", () => {
    const collapsed = vc.classList.toggle("collapsed");
    vcToggle.setAttribute("aria-expanded", String(!collapsed));
  });

  // Default the panel to the mobile drawer at the breakpoint — and follow it as
  // the window is resized / the device is rotated. The View section lives
  // inside the panel, so it needs no breakpoint handling of its own.
  const mq = window.matchMedia(MOBILE_QUERY);
  const applyBreakpoint = (mobile: boolean): void => {
    setPanelCollapsed(mobile);
  };
  applyBreakpoint(mq.matches);
  mq.addEventListener("change", (e) => applyBreakpoint(e.matches));
}

function initOverflowMenu(): void {
  const btn = byId("btn-menu");
  const menu = byId("toolbar-menu");

  const setOpen = (open: boolean): void => {
    menu.hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
  };

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(menu.hidden);
  });
  // Dismiss on outside-click and Escape.
  document.addEventListener("click", (e) => {
    if (!menu.hidden && !menu.contains(e.target as Node)) setOpen(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setOpen(false);
  });
  // Import / Export / Randomize open their own modals — close the menu behind them.
  byId("btn-import").addEventListener("click", () => setOpen(false));
  byId("btn-export").addEventListener("click", () => setOpen(false));
  byId("btn-randomize").addEventListener("click", () => setOpen(false));
}
