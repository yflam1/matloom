/**
 * eval-core.test.ts — the worker eval path must compute exactly what the
 * synchronous path always did.
 *
 * The guarantee under test: serializing a resolved layer stack to a
 * MaterialSpec and re-evaluating it (`buildSpec` → `evalSpec`) yields maps that
 * are byte-for-byte identical to building the same `Layer[]` directly and
 * calling `evaluateMaterial`. Because the spec is the boundary the eval worker
 * and the tile pool cross, this is what makes off-thread evaluation safe: same
 * input, same deterministic engine, same output.
 */
import { describe, it, expect } from "vitest";
import {
  Color,
  Emissive,
  Layer,
  evaluateMaterial,
  type EvalRegion,
  type Expression2D,
  type MaterialMaps,
} from "../../src/engine";
import { parseExpr } from "../../src/expr-lang";
import {
  buildSpec,
  evalSpec,
  SpecCache,
  mapsTransferList,
  FIELD_KEYS,
  type MaterialSpec,
} from "../../src/eval-core";

// A layer expressed as raw engine-domain field expressions (what `Color`/`Layer`
// wrap). Missing fields fall back to the channel defaults below.
type FieldExprs = Record<string, Expression2D>;

const DEFAULT_FIELD_SRC: Record<string, string> = {
  alpha: "1",
  "basecolor.r": "0",
  "basecolor.g": "0",
  "basecolor.b": "0",
  metallic: "0",
  roughness: "0",
  sheen: "0",
  coat: "0",
  transmission: "0",
  ior: "1.5",
  subsurface: "0",
  anisotropy: "0",
  "emissive.r": "0",
  "emissive.g": "0",
  "emissive.b": "0",
  "emissive.strength": "0",
  height: "0",
};

// Fill in any unspecified fields so both paths see a complete layer.
function complete(
  partial: Record<string, string>,
  env: Map<string, Expression2D>,
): FieldExprs {
  const out: FieldExprs = {};
  for (const key of FIELD_KEYS)
    out[key] = parseExpr(partial[key] ?? DEFAULT_FIELD_SRC[key], env);
  return out;
}

function fieldsToLayer(f: FieldExprs): Layer {
  return new Layer({
    alpha: f["alpha"],
    basecolor: new Color({
      r: f["basecolor.r"],
      g: f["basecolor.g"],
      b: f["basecolor.b"],
    }),
    metallic: f["metallic"],
    roughness: f["roughness"],
    sheen: f["sheen"],
    coat: f["coat"],
    transmission: f["transmission"],
    ior: f["ior"],
    subsurface: f["subsurface"],
    anisotropy: f["anisotropy"],
    emissive: new Emissive({
      r: f["emissive.r"],
      g: f["emissive.g"],
      b: f["emissive.b"],
      strength: f["emissive.strength"],
    }),
    height: f["height"],
  });
}

function expectMapsEqual(a: MaterialMaps, b: MaterialMaps): void {
  expect(a.emissiveIntensity).toBe(b.emissiveIntensity);
  expect(Array.from(a.basecolor)).toEqual(Array.from(b.basecolor));
  expect(Array.from(a.orm)).toEqual(Array.from(b.orm));
  expect(Array.from(a.pbr1)).toEqual(Array.from(b.pbr1));
  expect(Array.from(a.pbr2)).toEqual(Array.from(b.pbr2));
  expect(Array.from(a.emissive)).toEqual(Array.from(b.emissive));
  expect(Array.from(a.normal)).toEqual(Array.from(b.normal));
  expect(Array.from(a.displacement)).toEqual(Array.from(b.displacement));
}

// Build both a direct Layer[] and the equivalent spec, then assert byte-equal.
function assertSpecMatchesDirect(
  layerSrcs: Record<string, string>[],
  defSrcs: { name: string; src: string }[] = [],
  region: EvalRegion = {},
  w = 24,
  h = 24,
): { spec: MaterialSpec; reference: MaterialMaps } {
  const env = new Map<string, Expression2D>();
  for (const d of defSrcs) env.set(d.name, parseExpr(d.src, env));
  const fieldLayers = layerSrcs.map((s) => complete(s, env));

  const direct = fieldLayers.map(fieldsToLayer);
  const reference = evaluateMaterial(direct, w, h, region);

  const spec = buildSpec(fieldLayers, env, region, w, h);
  const viaSpec = evalSpec(spec);

  expectMapsEqual(viaSpec, reference);
  return { spec, reference };
}

describe("evalSpec reproduces the direct evaluateMaterial path byte-for-byte", () => {
  it("constants and simple arithmetic", () => {
    assertSpecMatchesDirect([
      {
        alpha: "1",
        "basecolor.r": "0.8",
        "basecolor.g": "(X() * Y())",
        "basecolor.b": "(1 - X())",
        roughness: "0.5",
        metallic: "0",
      },
    ]);
  });

  it("transcendentals and threshold", () => {
    assertSpecMatchesDirect([
      {
        "basecolor.r":
          "Threshold((Sin((X() * 10)) * 0.5 + 0.5), below_at=0.3, above_at=0.7)",
        roughness: "Abs(Cos((Y() * 8)))",
        height: "(Sqrt((X() + 0.01)) * 3)",
      },
    ]);
  });

  it("fBm with a pinned seed", () => {
    assertSpecMatchesDirect([
      {
        "basecolor.r": "fBm(base_freq=8, to_01=True, seed=12345)",
        roughness: "fBm(base_freq=16, to_01=True, seed=999)",
        height: "(fBm(base_freq=6, to_01=True, seed=7) * 5)",
      },
    ]);
  });

  it("fBm with an auto-rolled seed (same instance serialized into the spec)", () => {
    // The raw node is auto-seeded (no seed= given). buildSpec serializes *that*
    // instance, baking its current random seed, so the spec reproduces it.
    assertSpecMatchesDirect([
      {
        "basecolor.g": "fBm(base_freq=10, to_01=True)",
        height: "fBm(base_freq=4)",
      },
    ]);
  });

  it("Worley noise", () => {
    assertSpecMatchesDirect([
      {
        "basecolor.b":
          'Worley(combination="F2-F1", base_freq=6, to_01=True, seed=42)',
      },
    ]);
  });

  it("bricks pattern", () => {
    assertSpecMatchesDirect([
      {
        "basecolor.r": "Bricks(brick_width=0.25, brick_height=0.12)",
        height: "(Bricks(brick_width=0.25, brick_height=0.12) * 3)",
      },
    ]);
  });

  it("shapes and transforms", () => {
    assertSpecMatchesDirect([
      {
        alpha: "Fill(Ellipse(0.5, 0.5, 0.3, 0.2))",
        "basecolor.r": "Rotate(Fill(Rect(0.2, 0.2, 0.4, 0.4)), 30)",
        height: "(Stroke(Rect(0.1, 0.1, 0.8, 0.8), 0.05) * 2)",
      },
    ]);
  });

  it("multi-layer composite with masks and emissive", () => {
    assertSpecMatchesDirect([
      {
        "basecolor.r": "0.3",
        "basecolor.g": "0.3",
        "basecolor.b": "0.35",
        roughness: "0.7",
      },
      {
        alpha:
          "Threshold(fBm(base_freq=6, to_01=True, seed=1), below_at=0.5, below_to=0, above_at=0.5, above_to=1)",
        "basecolor.r": "0.6",
        height: "(fBm(base_freq=6, to_01=True, seed=1) * 2)",
      },
      {
        alpha: "Fill(Ellipse(0.5, 0.5, 0.2, 0.2))",
        "emissive.r": "1",
        "emissive.strength": "3",
      },
    ]);
  });

  it("definitions referenced via Ref", () => {
    assertSpecMatchesDirect(
      [{ "basecolor.r": "Rotate(star, 30)", alpha: "star" }],
      [
        {
          name: "star",
          src: "Fill(Path(0.4, 0.4, LineTo(0.6, 0.4), LineTo(0.5, 0.7)))",
        },
      ],
    );
  });

  it("chained definitions (later refers to earlier)", () => {
    assertSpecMatchesDirect(
      [{ alpha: "shifted" }],
      [
        { name: "box", src: "Fill(Rect(0.2, 0.2, 0.3, 0.3))" },
        { name: "shifted", src: "Translate(box, 0.4, 0.4)" },
      ],
    );
  });

  it("the 0-255 color convention (a /255 scale baked into the field)", () => {
    // The app scales 0–255 colors by 1/255 before evaluation; the spec carries
    // the already-scaled expression.
    assertSpecMatchesDirect([
      {
        "basecolor.r": "(200 / 255)",
        "basecolor.g": "(120 / 255)",
        "basecolor.b": "(60 / 255)",
      },
    ]);
  });

  it("a non-default region (offset window, yDown)", () => {
    assertSpecMatchesDirect(
      [
        {
          "basecolor.r": "X()",
          "basecolor.g": "Y()",
          height: "(fBm(seed=3) * 4)",
        },
      ],
      [],
      { x1: -0.5, y1: -0.5, x2: 0.5, y2: 0.5, yUp: false },
      20,
      20,
    );
  });
});

describe("MaterialSpec is structured-clone safe (survives a postMessage round-trip)", () => {
  it("a JSON-cloned spec evaluates identically to the original", () => {
    const { spec, reference } = assertSpecMatchesDirect([
      {
        "basecolor.r": "fBm(base_freq=8, to_01=True, seed=11)",
        alpha: "Fill(Ellipse(0.5, 0.5, 0.3, 0.3))",
        height: "(Worley(base_freq=5, to_01=True, seed=2) * 3)",
      },
    ]);
    // JSON is a strict subset of the structured-clone algorithm postMessage
    // uses; if a spec round-trips through JSON it round-trips through a worker.
    const cloned = JSON.parse(JSON.stringify(spec)) as MaterialSpec;
    expectMapsEqual(evalSpec(cloned), reference);
  });
});

describe("SpecCache memoization", () => {
  const region: EvalRegion = {};

  function specOf(layerSrcs: Record<string, string>[]): MaterialSpec {
    const env = new Map<string, Expression2D>();
    return buildSpec(
      layerSrcs.map((s) => complete(s, env)),
      env,
      region,
      16,
      16,
    );
  }

  it("re-evaluating an unchanged spec is deterministic", () => {
    const cache = new SpecCache();
    const spec = specOf([
      { "basecolor.r": "fBm(base_freq=8, to_01=True, seed=5)" },
    ]);
    expectMapsEqual(cache.eval(spec), cache.eval(spec));
  });

  it("editing one field leaves the others' output unchanged", () => {
    const cache = new SpecCache();
    const before = cache.eval(
      specOf([
        {
          "basecolor.r": "fBm(base_freq=8, to_01=True, seed=5)",
          roughness: "0.4",
        },
      ]),
    );
    // Change only roughness; basecolor must be byte-identical to `before`.
    const after = cache.eval(
      specOf([
        {
          "basecolor.r": "fBm(base_freq=8, to_01=True, seed=5)",
          roughness: "0.9",
        },
      ]),
    );
    expect(Array.from(after.basecolor)).toEqual(Array.from(before.basecolor));
    // ...while roughness (ORM green) actually changed.
    let ormChanged = false;
    for (let i = 0; i < after.orm.length; i += 4)
      if (after.orm[i + 1] !== before.orm[i + 1]) ormChanged = true;
    expect(ormChanged).toBe(true);
  });

  it("a fresh cache matches a reused cache (no stale state leaks)", () => {
    const spec = specOf([
      {
        "basecolor.r": "fBm(base_freq=8, to_01=True, seed=5)",
        height: "(X() * 4)",
      },
    ]);
    const reused = new SpecCache();
    reused.eval(specOf([{ "basecolor.r": "0.5" }])); // prime with something else
    expectMapsEqual(reused.eval(spec), new SpecCache().eval(spec));
  });
});

describe("mapsTransferList", () => {
  it("lists the five backing buffers for zero-copy transfer", () => {
    const maps = evaluateMaterial([new Layer()], 4, 4);
    const list = mapsTransferList(maps);
    expect(list).toHaveLength(5);
    expect(list).toContain(maps.basecolor.buffer);
    expect(list).toContain(maps.displacement.buffer);
  });
});
