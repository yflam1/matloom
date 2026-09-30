/** Tests for src/app/material-io.ts — Define preamble import/export round-trip. */
import { describe, it, expect } from "vitest";
import {
  serializeMaterial,
  parseImportedLayers,
} from "../../src/app/material-io";
import { channelKeyOfFk } from "../../src/app/channels";
import { createStore } from "../../src/app/state";

// The canonical form the Python engine emits for the same input (see
// tests/engine/test_main.py `_MATERIAL_WITH_DEF`); both engines must agree.
const CANONICAL = `View(0, 0, 1, 1)
Define(star, Fill(Path(0.4, 0.4, LineTo(0.6, 0.4), LineTo(0.5, 0.7))))
Material(
  Layer(Rotate(star, 30))
    .basecolor(200, 120, 60)
    .metallic(0)
    .roughness(0.5)
    .sheen(0)
    .coat(0)
    .transmission(0)
    .ior(1.5)
    .subsurface(0)
    .anisotropy(0)
    .emissive(0, 0, 0, 0)
    .height(0)
)`;

describe("material-io Define round-trip", () => {
  it("imports and re-serializes a definition identically", () => {
    const r = parseImportedLayers(CANONICAL);
    const store = createStore();
    store.layers = r.layers;
    store.defs = r.defs;
    // CANONICAL has a `View(...)` line: apply it + mark the view authored so the
    // re-serialized output keeps the `View(...)` preamble (mirrors the production
    // import path in io-controls.applyMaterialText).
    if (r.view) {
      Object.assign(store.settings, r.view);
      store.settings.viewSet = true;
    }
    expect(serializeMaterial(store)).toBe(CANONICAL);
  });

  it("exposes imported definitions by name", () => {
    const { defs } = parseImportedLayers(CANONICAL);
    expect(defs.map((d) => d.name)).toEqual(["star"]);
  });

  it("resolves a later def referencing an earlier one", () => {
    const src =
      "Define(box, Fill(Rect(0.2, 0.2, 0.3, 0.3)))\n" +
      "Define(shifted, Translate(box, 0.4, 0.4))\n" +
      "Material(Layer(shifted).basecolor(255, 255, 255)" +
      ".metallic(0).roughness(0.5).emissive(0, 0, 0, 0))";
    const { layers, defs } = parseImportedLayers(src);
    const store = createStore();
    store.layers = layers;
    store.defs = defs;
    const out = serializeMaterial(store);
    expect(out).toContain("Define(box, ");
    expect(out).toContain("Define(shifted, Translate(box, 0.4, 0.4))");
  });

  it("rejects a reference to an undefined name", () => {
    expect(() =>
      parseImportedLayers(
        "Material(Layer(missing).basecolor(0, 0, 0)" +
          ".metallic(0).roughness(0.5).emissive(0, 0, 0, 0))",
      ),
    ).toThrow(/unknown name/);
  });

  it("rejects a malformed Define", () => {
    expect(() => parseImportedLayers("Define(a)\nMaterial()")).toThrow(
      /two arguments/,
    );
  });

  it("round-trips a non-default height expression", () => {
    const src =
      "Material(\n" +
      "  Layer(1)\n" +
      "    .basecolor(0, 0, 0)\n" +
      "    .metallic(0)\n" +
      "    .roughness(0.5)\n" +
      "    .emissive(0, 0, 0, 0)\n" +
      "    .height(fBm(base_freq=4, seed=3)),\n" +
      "  Layer(Fill(Rect(0.2, 0.2, 0.6, 0.6)))\n" +
      "    .basecolor(0, 0, 0)\n" +
      "    .metallic(0)\n" +
      "    .roughness(0.5)\n" +
      "    .emissive(0, 0, 0, 0)\n" +
      "    .height(9)\n" +
      ")";
    const { layers, defs } = parseImportedLayers(src);
    const store = createStore();
    store.layers = layers;
    store.defs = defs;
    const out = serializeMaterial(store);
    expect(out).toContain(".height(fBm(base_freq=4, seed=3))");
    expect(out).toContain(".height(9)");
    // Re-importing the output reproduces it exactly.
    const store2 = createStore();
    const r2 = parseImportedLayers(out);
    store2.layers = r2.layers;
    store2.defs = r2.defs;
    expect(serializeMaterial(store2)).toBe(out);
  });

  it("defaults height to 0 when a layer omits .height (backward compatible)", () => {
    const { layers } = parseImportedLayers(
      "Material(Layer(1).basecolor(0, 0, 0).metallic(0)" +
        ".roughness(0.5).emissive(0, 0, 0, 0))",
    );
    expect(layers[0].channels.height).toBe("0.0");
  });

  it("omits a View line when the region was never authored", () => {
    // No `View(...)` in the imported source and no region edit: export omits the
    // `View(...)` preamble (set-tracking: an unauthored default view is omitted,
    // mirroring the Python engine).
    const store = createStore();
    store.layers = parseImportedLayers("Material(Layer(1))").layers;
    const out = serializeMaterial(store);
    expect(out.startsWith("Material(")).toBe(true);
    expect(out.includes("View(")).toBe(false);
  });

  it("emits and round-trips a non-default View (region)", () => {
    const store = createStore();
    // A user-authored region edit (mirrors the panel input path, which marks
    // the view authored via afterRegionChange).
    store.settings.x2 = 2;
    store.settings.y2 = 1.5;
    store.settings.viewSet = true;
    store.layers = parseImportedLayers("Material(Layer(1))").layers;
    const out = serializeMaterial(store);
    expect(out.startsWith("View(0, 0, 2, 1.5)")).toBe(true);
    // Importing it back yields the same view.
    const r = parseImportedLayers(out);
    expect(r.view).toEqual({ x1: 0, y1: 0, x2: 2, y2: 1.5 });
  });

  it("parses View before Define and Material", () => {
    const { view, layers } = parseImportedLayers(
      "View(0, 0, 4, 4)\nDefine(b, Fill(Rect(0.2, 0.2, 0.3, 0.3)))\n" +
        "Material(Layer(b))",
    );
    expect(view).toEqual({ x1: 0, y1: 0, x2: 4, y2: 4 });
    expect(layers).toHaveLength(1);
  });

  it("rejects a malformed View", () => {
    // Too few args.
    expect(() => parseImportedLayers("View(0, 0, 1)\nMaterial()")).toThrow(
      /4 arguments/,
    );
    // The old 5-arg form (with a mm/unit scale) is no longer accepted.
    expect(() =>
      parseImportedLayers("View(0, 0, 1, 1, 100)\nMaterial()"),
    ).toThrow(/4 arguments/);
  });
});

describe("channel set-tracking (emit only what was set)", () => {
  function roundTrip(src: string): string {
    const { layers, defs, view } = parseImportedLayers(src);
    const store = createStore();
    store.layers = layers;
    store.defs = defs;
    // Apply an imported `View(...)` line the way the production import path
    // does (io-controls.applyMaterialText): coords + viewSet=true. Sources
    // without a `View(...)` line leave viewSet=false, so export omits it.
    if (view) {
      Object.assign(store.settings, view);
      store.settings.viewSet = true;
    }
    return serializeMaterial(store);
  }

  it("re-serializes a sparse layer to only its authored channels", () => {
    // Byte-identical to the Python engine's output for the same input. The
    // source has no `View(...)` line, so the preamble is omitted.
    const expected = `Material(
  Layer(0.8)
    .basecolor(150, 120, 100)
    .metallic(0.2)
)`;
    expect(
      roundTrip("Material(Layer(0.8).basecolor(150, 120, 100).metallic(0.2))"),
    ).toBe(expected);
  });

  it("preserves an explicit default channel on round-trip", () => {
    // .roughness(0) and .metallic(0) are defaults, but the author wrote them
    // explicitly -> they are kept. Byte-identical to the Python engine.
    const expected = `Material(
  Layer(0.5)
    .basecolor(10, 20, 30)
    .metallic(0)
    .roughness(0)
)`;
    expect(
      roundTrip(
        "Material(Layer(0.5).basecolor(10, 20, 30).roughness(0).metallic(0))",
      ),
    ).toBe(expected);
  });

  it("emits Layer() for a layer with no set channels", () => {
    const expected = `Material(
  Layer()
)`;
    expect(roundTrip("Material(Layer())")).toBe(expected);
  });

  it("omits an untouched alpha (Layer() with a channel but no alpha)", () => {
    const expected = `Material(
  Layer()
    .basecolor(1, 2, 3)
)`;
    expect(roundTrip("Material(Layer().basecolor(1, 2, 3))")).toBe(expected);
  });

  it("channelKeyOfFk maps field keys to atomic channel keys", () => {
    expect(channelKeyOfFk("alpha")).toBe("alpha");
    expect(channelKeyOfFk("metallic")).toBe("metallic");
    expect(channelKeyOfFk("basecolor.r")).toBe("basecolor");
    expect(channelKeyOfFk("basecolor.g")).toBe("basecolor");
    expect(channelKeyOfFk("emissive.strength")).toBe("emissive");
  });
});
