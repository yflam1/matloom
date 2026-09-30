import { describe, it, expect } from "vitest";
import {
  DEFAULT_CONFIG,
  GeneratedMaterial,
  MaterialGenerator,
  generateMaterial,
} from "../src/generate";
import { parseImportedLayers, serializeMaterial } from "../src/app/material-io";
import { createStore } from "../src/app/state";

const SEEDS = Array.from({ length: 12 }, (_, i) => i);

function asString(seed: number, config = {}): string {
  return generateMaterial(config, { seed, asString: true }) as string;
}

// Rebuild a Store from a generated string and re-serialize it — the canonical
// round-trip the app itself performs on import/export.
function roundTrip(src: string): string {
  const { layers, defs, view } = parseImportedLayers(src);
  const store = createStore();
  store.layers = layers;
  store.defs = defs;
  if (view) {
    store.settings.x1 = view.x1;
    store.settings.y1 = view.y1;
    store.settings.x2 = view.x2;
    store.settings.y2 = view.y2;
    // A generated string always carries a `View(...)` line, so mark the view
    // authored (mirrors the production import path) or re-serialization would
    // drop the preamble and break the byte-identical round-trip.
    store.settings.viewSet = true;
  }
  return serializeMaterial(store);
}

describe("generateMaterial return types", () => {
  it("returns a GeneratedMaterial by default", () => {
    expect(generateMaterial({}, { seed: 0 })).toBeInstanceOf(GeneratedMaterial);
  });

  it("returns serialized DSL when asString", () => {
    const s = asString(0);
    expect(typeof s).toBe("string");
    expect(s.startsWith("View(")).toBe(true);
    expect(s).toContain("Material(");
  });

  it("asString matches material.serialize()", () => {
    const material = generateMaterial({}, { seed: 5 }) as GeneratedMaterial;
    expect(material.serialize()).toBe(asString(5));
  });
});

describe("determinism", () => {
  it("same seed is deterministic", () => {
    expect(asString(123)).toBe(asString(123));
  });

  it("different seeds differ", () => {
    expect(asString(1)).not.toBe(asString(2));
  });
});

describe("round-trip / validity", () => {
  it.each(SEEDS)("generated material round-trips stably (seed %i)", (seed) => {
    const once = asString(seed);
    expect(roundTrip(once)).toBe(once);
  });

  it.each(SEEDS)("deserializes to the same shape (seed %i)", (seed) => {
    const cfg = { numLayers: 3, numDefs: 2 };
    const src = asString(seed, cfg);
    const { layers, defs } = parseImportedLayers(src);
    expect(layers.length).toBe(3);
    expect(defs.length).toBe(2);
  });
});

describe("layer counts", () => {
  it("numLayers is exact", () => {
    const m = generateMaterial(
      { numLayers: 4 },
      { seed: 0 },
    ) as GeneratedMaterial;
    expect(m.layers.length).toBe(4);
  });

  it.each(SEEDS)("layer count within range (seed %i)", (seed) => {
    const m = generateMaterial(
      { minLayers: 2, maxLayers: 5 },
      { seed },
    ) as GeneratedMaterial;
    expect(m.layers.length).toBeGreaterThanOrEqual(2);
    expect(m.layers.length).toBeLessThanOrEqual(5);
  });

  it("default layer count is within 1..3", () => {
    for (let seed = 0; seed < 40; seed++) {
      const m = generateMaterial({}, { seed }) as GeneratedMaterial;
      expect(m.layers.length).toBeGreaterThanOrEqual(1);
      expect(m.layers.length).toBeLessThanOrEqual(3);
    }
  });
});

describe("define preamble", () => {
  it("names defines def1, def2, … in order", () => {
    const m = generateMaterial(
      { numDefs: 3 },
      { seed: 0 },
    ) as GeneratedMaterial;
    expect(m.defs.map((d) => d.name)).toEqual(["def1", "def2", "def3"]);
  });

  it("emits no Define when numDefs is 0", () => {
    expect(asString(0, { numDefs: 0 })).not.toContain("Define(");
  });

  it("defines only reference earlier defines (import never throws)", () => {
    const cfg = { numDefs: 5, numLayers: 2, maxDepth: 5 };
    for (let seed = 0; seed < 20; seed++) {
      expect(() => parseImportedLayers(asString(seed, cfg))).not.toThrow();
    }
  });
});

describe("opaque base", () => {
  it("forces the bottom-most layer alpha to 1", () => {
    const src = asString(0, { opaqueBase: true });
    // Bottom-most layer is the first in the canonical string.
    const firstLayer = src.slice(src.indexOf("Layer("));
    expect(firstLayer.startsWith("Layer(1)")).toBe(true);
  });

  it("non-opaque base still round-trips", () => {
    for (let seed = 0; seed < 10; seed++) {
      const once = asString(seed, { opaqueBase: false });
      expect(roundTrip(once)).toBe(once);
    }
  });
});

describe("depth control", () => {
  it("maxDepth 0 produces only terminals and still round-trips", () => {
    const cfg = { maxDepth: 0, numLayers: 2, numDefs: 1 };
    for (let seed = 0; seed < 15; seed++) {
      const once = asString(seed, cfg);
      expect(roundTrip(once)).toBe(once);
    }
  });
});

describe("view region", () => {
  it.each(SEEDS)("is non-degenerate (seed %i)", (seed) => {
    const m = generateMaterial({}, { seed }) as GeneratedMaterial;
    const [x1, y1, x2, y2] = m.view;
    expect(x2).toBeGreaterThan(x1);
    expect(y2).toBeGreaterThan(y1);
  });

  it.each(SEEDS)(
    "pins to (0,0,1,1) when randomView is off (seed %i)",
    (seed) => {
      const m = generateMaterial(
        { randomView: false },
        { seed },
      ) as GeneratedMaterial;
      expect(m.view).toEqual([0, 0, 1, 1]);
      expect(m.serialize().startsWith("View(0, 0, 1, 1)")).toBe(true);
    },
  );
});

describe("type-faithful sampling", () => {
  it("respects numeric domains", () => {
    const gen = new MaterialGenerator(
      {
        realLo: -3,
        realHi: 4,
        posRealMin: 0.01,
        posRealMax: 2,
        nonNegRealMax: 5,
      },
      0,
    );
    for (let i = 0; i < 2000; i++) {
      const r = gen.real();
      expect(r).toBeGreaterThanOrEqual(-3);
      expect(r).toBeLessThanOrEqual(4);
      const p = gen.posReal();
      expect(p).toBeGreaterThanOrEqual(0.01);
      expect(p).toBeLessThanOrEqual(2);
      const nn = gen.nonNegReal();
      expect(nn).toBeGreaterThanOrEqual(0);
      expect(nn).toBeLessThanOrEqual(5);
      expect(gen.nonzeroReal()).not.toBe(0);
      const z = gen.zeroToOne();
      expect(z).toBeGreaterThanOrEqual(0);
      expect(z).toBeLessThan(1);
      const d = gen.degrees();
      expect(d).toBeGreaterThanOrEqual(0);
      expect(d).toBeLessThan(360);
      const byte = gen.byte();
      expect(byte).toBeGreaterThanOrEqual(0);
      expect(byte).toBeLessThanOrEqual(255);
      const s = gen.seed();
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(2 ** 31 - 1);
    }
  });
});

describe("config validation", () => {
  const bad: Array<[Partial<typeof DEFAULT_CONFIG>, string]> = [
    [{ minLayers: 3, maxLayers: 2 }, "minLayers"],
    [{ numLayers: 0 }, "numLayers"],
    [{ minLayers: 0 }, "positive"],
    [{ minDefs: 2, maxDefs: 1 }, "minDefs"],
    [{ numDefs: -1 }, "numDefs"],
    [{ maxDepth: -1 }, "maxDepth"],
    [{ leafBias: 1.5 }, "leafBias"],
    [{ realLo: 5, realHi: 1 }, "realLo"],
    [{ posRealMin: 0 }, "posRealMin"],
    [{ maxOctaves: 0 }, "maxOctaves"],
    [{ maxPathSegments: 0 }, "maxPathSegments"],
  ];
  it.each(bad)("rejects %o", (config, match) => {
    expect(() => new MaterialGenerator(config)).toThrow(match);
  });
});
