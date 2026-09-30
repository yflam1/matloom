/** Tests for src/expr-lang/serialize.ts — Expression2D -> canonical DSL. */
import { describe, it, expect } from "vitest";
import { serializeExpr } from "../../src/expr-lang/serialize";
import { parseExpr } from "../../src/expr-lang/parser";
import {
  Constant,
  X,
  Y,
  Sin,
  Min,
  Max,
  Log,
  Threshold,
  fBm,
  Worley,
  Bricks,
  Add,
  Mul,
} from "../../src/engine";

describe("primitives", () => {
  it("integral constants stay integral", () => {
    expect(serializeExpr(new Constant(5))).toBe("5");
  });
  it("float constants", () => {
    expect(serializeExpr(new Constant(0.25))).toBe("0.25");
  });
  it("X and Y", () => {
    expect(serializeExpr(new X())).toBe("X()");
    expect(serializeExpr(new Y())).toBe("Y()");
  });
  it("unary minus renders without -1 *", () => {
    expect(serializeExpr(parseExpr("-X()"))).toBe("-X()");
  });
  it("binary operators are parenthesized", () => {
    expect(serializeExpr(Add(new X(), new Y()))).toBe("(X() + Y())");
    expect(serializeExpr(Mul(new X(), new Y()))).toBe("(X() * Y())");
  });
  it("Min/Max", () => {
    expect(serializeExpr(new Min(new X(), new Y()))).toBe("Min(X(), Y())");
    expect(serializeExpr(new Max(new X(), new Y()))).toBe("Max(X(), Y())");
  });
  it("Sin", () => {
    expect(serializeExpr(new Sin(new X()))).toBe("Sin(X())");
  });
  it("Log natural omits base; explicit base included", () => {
    expect(serializeExpr(new Log(new X()))).toBe("Log(X())");
    expect(serializeExpr(new Log(new X(), 10))).toBe("Log(X(), 10)");
  });
});

describe("Threshold", () => {
  it("omits redundant below_to", () => {
    expect(serializeExpr(new Threshold(new X(), { below_at: 0.3 }))).toBe(
      "Threshold(X(), below_at=0.3)",
    );
  });
  it("includes all set params", () => {
    const s = serializeExpr(
      new Threshold(new X(), {
        below_at: 0.1,
        below_to: 0,
        above_at: 0.9,
        above_to: 1,
        transition_width: 0.05,
      }),
    );
    expect(s).toContain("below_at=0.1");
    expect(s).toContain("below_to=0");
    expect(s).toContain("above_at=0.9");
    expect(s).toContain("above_to=1");
    expect(s).toContain("transition_width=0.05");
  });
});

describe("noise serialization bakes the seed", () => {
  it("fBm omits defaults, includes seed", () => {
    expect(serializeExpr(new fBm({ seed: 7 }))).toBe("fBm(seed=7)");
  });
  it("fBm includes non-default params", () => {
    const s = serializeExpr(
      new fBm({ octaves: 4, base_freq: 2, to_01: true, seed: 9 }),
    );
    expect(s).toContain("octaves=4");
    expect(s).toContain("base_freq=2");
    expect(s).toContain("to_01=True");
    expect(s).toContain("seed=9");
  });
  it("fBm separate axis frequencies", () => {
    const s = serializeExpr(
      new fBm({ base_freq_x: 2, base_freq_y: 3, seed: 1 }),
    );
    expect(s).toContain("base_freq_x=2");
    expect(s).toContain("base_freq_y=3");
  });
  it("Worley defaults", () => {
    expect(serializeExpr(new Worley({ seed: 2 }))).toBe("Worley(seed=2)");
  });
  it("Worley non-defaults", () => {
    const s = serializeExpr(
      new Worley({ distance: "manhattan", combination: "F2", seed: 4 }),
    );
    expect(s).toContain('distance="manhattan"');
    expect(s).toContain('combination="F2"');
  });
});

describe("noise set-tracking (parse path)", () => {
  // A parsed noise node carries the arg names the author wrote, so the
  // serializer emits only those, preserving explicit defaults like `octaves=6`
  // or `to_01=False` that value-comparison would drop.
  it("fBm keeps an explicit default octaves=6", () => {
    expect(
      serializeExpr(parseExpr("fBm(octaves=6, base_freq=2, seed=9)")),
    ).toBe("fBm(octaves=6, base_freq=2, seed=9)");
  });
  it("fBm keeps an explicit default to_01=False", () => {
    expect(
      serializeExpr(parseExpr("fBm(base_freq=2, to_01=False, seed=9)")),
    ).toBe("fBm(base_freq=2, to_01=False, seed=9)");
  });
  it("fBm omits unset defaults", () => {
    expect(serializeExpr(parseExpr("fBm(base_freq=2, seed=9)"))).toBe(
      "fBm(base_freq=2, seed=9)",
    );
  });
  it("fBm one-axis frequency omits the other axis", () => {
    expect(serializeExpr(parseExpr("fBm(base_freq_x=2, seed=9)"))).toBe(
      "fBm(base_freq_x=2, seed=9)",
    );
  });
  it("Worley keeps explicit default distance/combination", () => {
    expect(
      serializeExpr(
        parseExpr('Worley(distance="euclidean", combination="F1", seed=4)'),
      ),
    ).toBe('Worley(distance="euclidean", combination="F1", seed=4)');
  });
});

describe("Bricks serialization", () => {
  it("omits defaults", () => {
    expect(serializeExpr(new Bricks())).toBe("Bricks()");
  });
  it("includes non-default params", () => {
    const s = serializeExpr(
      new Bricks({
        brick_width: 2,
        offset: 0.25,
        axis: "column",
        feather: 0.02,
      }),
    );
    expect(s).toContain("brick_width=2");
    expect(s).toContain("offset=0.25");
    expect(s).toContain('axis="column"');
    expect(s).toContain("feather=0.02");
    expect(s).not.toContain("brick_height");
    expect(s).not.toContain("mortar");
  });
});

describe("round-trip stability", () => {
  const cases = [
    "5",
    "0.25",
    "X()",
    "Y()",
    "(X() + Y())",
    "(X() * 2)",
    "Sin(X())",
    "Min(X(), Y())",
    "Log(X(), 10)",
    "Threshold(X(), below_at=0.2, above_at=0.8)",
    "fBm(octaves=4, seed=9)",
    "fBm(octaves=6, base_freq=2, seed=9)",
    "fBm(base_freq_x=2, base_freq_y=3, seed=9)",
    'Worley(distance="euclidean", combination="F1", seed=4)',
    'Worley(distance="manhattan", seed=4)',
    "Bricks()",
    'Bricks(brick_width=2, offset=0.25, axis="column", feather=0.02)',
    "Translate(X(), 0.3, 0)",
    "Scale(X(), 2, 0.5)",
    "Rotate(X(), 45)",
    "Fill(Rect(0.2, 0.2, 0.6, 0.6))",
    "Fill(Rect(0, 0, 1, 1, 0.1), feather=0.02)",
    'Fill(Ellipse(0.5, 0.5, 0.3, 0.2), mode="EvenOdd")',
    "Stroke(Rect(0.2, 0.2, 0.6, 0.6), 0.04, feather=0.01)",
    "Fill(Path(0.2, 0.2, LineTo(0.8, 0.2), LineTo(0.5, 0.8)))",
    "Stroke(Path(0.1, 0.5, CubicTo(0.3, 0.9, 0.7, 0.1, 0.9, 0.5)), 0.05)",
  ];
  for (const src of cases) {
    it(`is a fixed point: ${src}`, () => {
      const once = serializeExpr(parseExpr(src));
      const twice = serializeExpr(parseExpr(once));
      expect(twice).toBe(once);
    });
  }
});
