/**
 * layer-panel.ts — renders the layer list: each layer is a collapsible card of
 * channel-group editors, with an eye toggle, delete button, a 0–1 / 0–255
 * range toggle per color channel, and header drag-to-reorder.
 */
import { byId } from "./dom";
import {
  CHANNEL_SPEC,
  channelKeyOfFk,
  getChannelValue,
  setChannelValue,
} from "./channels";
import { defaultChannels, uid, type LayerState, type Store } from "./state";
import type { Evaluator } from "./eval";
import type { ExprEditorUI } from "./expr-editor";

export class LayerPanel {
  private dragId: string | null = null;
  private armedCard: HTMLElement | null = null; // card whose drag is armed by a header press

  constructor(
    private readonly store: Store,
    private readonly evaluator: Evaluator,
    private readonly editorUI: ExprEditorUI,
  ) {
    // A single global mouseup disarms the armed card after a press that did not
    // turn into a drag (a completed drag disarms via its own `dragend`). This
    // keeps drags startable only from the header without a per-card listener.
    window.addEventListener("mouseup", () => {
      if (this.armedCard) {
        this.armedCard.draggable = false;
        this.armedCard = null;
      }
    });
  }

  /** Wire the "+ Layer" button and render the initial (empty) list. */
  init(): void {
    byId("btn-add").addEventListener("click", () => this.addLayer());
  }

  addLayer(): void {
    this.store.layers.unshift({
      id: uid(),
      enabled: true,
      collapsed: false,
      channels: defaultChannels(),
      // Sparse: a new layer exports `Layer()` until a channel is edited.
      setChannels: new Set(),
      colorMode: { basecolor: "255", emissive: "255" },
      errors: {},
      editorRefs: {},
      seedMemory: {},
    });
    this.render();
    this.evaluator.scheduleEval();
  }

  private removeLayer(id: string): void {
    this.store.layers = this.store.layers.filter((l) => l.id !== id);
    this.render();
    this.evaluator.scheduleEval();
  }

  private duplicateLayer(id: string): void {
    const idx = this.store.layers.findIndex((l) => l.id === id);
    if (idx === -1) return;
    const src = this.store.layers[idx];
    // Deep-copy channels (expression strings) and the remembered noise seeds so
    // the duplicate's unseeded noises reproduce the source's seeds; new uid;
    // editorRefs will be re-created by render().
    const copy: LayerState = {
      id: uid(),
      enabled: src.enabled,
      collapsed: src.collapsed,
      channels: JSON.parse(JSON.stringify(src.channels)),
      setChannels: new Set(src.setChannels),
      colorMode: { ...src.colorMode },
      errors: {},
      editorRefs: {},
      seedMemory: JSON.parse(JSON.stringify(src.seedMemory)),
    };
    // Insert immediately after the source layer.
    this.store.layers.splice(idx + 1, 0, copy);
    this.render();
    this.evaluator.scheduleEval();
  }

  render(): void {
    const list = byId("layer-list");
    list.replaceChildren();
    // `layers` is stored top-most first, so render in order: top-most at the
    // top of the list, bottom-most ("Layer 1") at the bottom.
    const n = this.store.layers.length;
    this.store.layers.forEach((l, i) =>
      list.appendChild(this.buildLayerEl(l, `Layer ${n - i}`)),
    );
  }

  private buildLayerEl(layer: LayerState, label: string): HTMLDivElement {
    const card = document.createElement("div");
    card.className = "lcard" + (layer.collapsed ? "" : " open");
    card.dataset.id = layer.id;
    card.draggable = false; // armed only while a drag starts on the header

    // — Header —
    const hdr = document.createElement("div");
    hdr.className = "lhdr";

    const eye = document.createElement("button");
    eye.className = "leye" + (layer.enabled ? " on" : "");
    eye.textContent = layer.enabled ? "●" : "○";
    eye.title = "Toggle visibility";
    eye.addEventListener("click", (e) => {
      e.stopPropagation();
      layer.enabled = !layer.enabled;
      eye.className = "leye" + (layer.enabled ? " on" : "");
      eye.textContent = layer.enabled ? "●" : "○";
      this.evaluator.scheduleEval();
    });

    // Name reflects stacking order, so it always shows the layer's position.
    const name = document.createElement("span");
    name.className = "lname";
    name.textContent = label;

    const caret = document.createElement("span");
    caret.className = "lcaret";
    caret.textContent = "▶";

    const dup = document.createElement("button");
    dup.className = "ldup";
    dup.textContent = "⧉";
    dup.title = "Duplicate layer";
    dup.addEventListener("click", (e) => {
      e.stopPropagation();
      this.duplicateLayer(layer.id);
    });

    const del = document.createElement("button");
    del.className = "ldel";
    del.textContent = "✕";
    del.title = "Delete layer";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      this.removeLayer(layer.id);
    });

    hdr.append(eye, name, caret, dup, del);
    hdr.addEventListener("click", () => {
      layer.collapsed = !card.classList.toggle("open");
    });

    // — Body —
    const body = document.createElement("div");
    body.className = "lbody";
    layer.editorRefs = {};

    for (const spec of CHANNEL_SPEC) {
      const grp = document.createElement("div");
      grp.className = "cgrp open";

      const ghdr = document.createElement("div");
      ghdr.className = "cghdr";
      const gc = document.createElement("span");
      gc.className = "cgcaret";
      gc.textContent = "▶";
      ghdr.append(gc);
      // Keep a trailing unit like "(mm)" lowercase even though the header is
      // CSS-uppercased (so "Height (mm)" reads "HEIGHT (mm)", not "HEIGHT (MM)").
      const unitMatch = spec.label.match(/^(.*?)\s*(\([^)]*\))$/);
      if (unitMatch) {
        ghdr.append(document.createTextNode(" " + unitMatch[1] + " "));
        const unit = document.createElement("span");
        unit.className = "cunit";
        unit.textContent = unitMatch[2];
        ghdr.append(unit);
      } else {
        ghdr.append(document.createTextNode(" " + spec.label));
      }
      ghdr.addEventListener("click", () => grp.classList.toggle("open"));

      if (spec.hasRangeToggle) {
        const modeKey = spec.key as "basecolor" | "emissive";
        const btn = document.createElement("button");
        btn.className = "crng";
        btn.title = "Toggle input range: 0–1 (normalized) or 0–255 (integer)";
        const updateBtn = () => {
          const mode = layer.colorMode[modeKey];
          btn.textContent = mode === "255" ? "0–255" : "0–1";
          btn.classList.toggle("crng-int", mode === "255");
        };
        updateBtn();
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          layer.colorMode[modeKey] =
            layer.colorMode[modeKey] === "255" ? "01" : "255";
          updateBtn();
          this.evaluator.scheduleEval(true);
        });
        ghdr.appendChild(btn);
      }

      const gbody = document.createElement("div");
      gbody.className = "cgbody";
      const showLabels = spec.fields.length > 1;
      for (const { fk, lbl } of spec.fields) {
        const wrap = document.createElement("div");
        wrap.className = "cfield";
        if (showLabels) {
          const l = document.createElement("div");
          l.className = "clbl";
          l.textContent = lbl ?? "";
          wrap.appendChild(l);
        }
        const edWrap = document.createElement("div");
        edWrap.className = "ceditor";
        layer.editorRefs[fk] = this.editorUI.makeEditor(
          edWrap,
          getChannelValue(layer.channels, fk),
          (val) => {
            setChannelValue(layer.channels, fk, val);
            // Sticky: an edited channel is now part of the export set, even if
            // the value is later returned to the default.
            layer.setChannels.add(channelKeyOfFk(fk));
            this.evaluator.scheduleEval(true);
          },
        );
        wrap.appendChild(edWrap);
        gbody.appendChild(wrap);
      }
      grp.append(ghdr, gbody);
      body.appendChild(grp);
    }

    card.append(hdr, body);
    this.attachDragReorder(card, hdr);
    return card;
  }

  private attachDragReorder(card: HTMLDivElement, hdr: HTMLDivElement): void {
    // Arm dragging only when the press starts on the header so clicking or
    // dragging inside a channel input never reorders the layer. The global
    // mouseup / dragend handlers disarm it again.
    hdr.addEventListener("mousedown", () => {
      card.draggable = true;
      this.armedCard = card;
    });

    card.addEventListener("dragstart", () => {
      this.dragId = card.dataset.id ?? null;
      card.classList.add("dragging");
    });
    card.addEventListener("dragend", () => {
      card.draggable = false;
      if (this.armedCard === card) this.armedCard = null;
      card.classList.remove("dragging");
    });
    card.addEventListener("dragover", (e) => {
      e.preventDefault();
      card.classList.add("dragover");
    });
    card.addEventListener("dragleave", () => card.classList.remove("dragover"));
    card.addEventListener("drop", (e) => {
      e.preventDefault();
      card.classList.remove("dragover");
      if (!this.dragId || this.dragId === card.dataset.id) return;
      const layers = this.store.layers;
      const from = layers.findIndex((l) => l.id === this.dragId);
      const to = layers.findIndex((l) => l.id === card.dataset.id);
      if (from === -1 || to === -1) return;
      const [moved] = layers.splice(from, 1);
      layers.splice(to, 0, moved);
      this.render(); // re-labels so names follow the new order
      this.evaluator.scheduleEval();
    });
  }
}
