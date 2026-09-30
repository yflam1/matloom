/** Tests for src/engine/expression.ts — the Expression2D tree. */
import { describe, it, expect } from "vitest";
import {
  Constant,
  X,
  Y,
  Sin,
  Cos,
  Log,
  Abs,
  Sqrt,
  Floor,
  Ceil,
  sRGB2Linear,
  Linear2sRGB,
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  AdditionExpression2D,
  DivisionExpression2D,
  Min,
  Max,
  Threshold,
  coerceConstant,
  coerceFloat,
  coerceInt,
} from "../../src/engine/expression";
import type { Axis } from "../../src/engine/grid";

// Helper: every node's evalGrid must agree with its scalar call() path.
// Grids are Float32Array, so values are compared at single-precision tolerance
// (~1e-6) rather than full double precision.
function expectGridMatchesCall(
  expr: { evalGrid: Function; call: Function },
  xs: Axis,
  ys: Axis,
) {
  const w = xs.length;
  const h = ys.length;
  const grid = expr.evalGrid(xs, ys, w, h);
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      expect(grid[i * w + j]).toBeCloseTo(expr.call(xs[j], ys[i]), 6);
    }
  }
}

const XS = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
const YS = new Float32Array([0, 0.5, 1]);

describe("leaf nodes", () => {
  it("Constant returns its value everywhere", () => {
    expect(new Constant(0.7).call()).toBe(0.7);
  });
  it("Constant default is 0", () => {
    expect(new Constant().call()).toBe(0);
  });
  it("Constant exposes literal", () => {
    expect(new Constant(1.5).literal).toBe(1.5);
  });
  it("X and Y select coordinates", () => {
    expect(new X().call(0.3)).toBe(0.3);
    expect(new Y().call(0.3, 0.9)).toBe(0.9);
  });
  it("leaf grids match call", () => {
    expectGridMatchesCall(new Constant(0.25), XS, YS);
    expectGridMatchesCall(new X(), XS, YS);
    expectGridMatchesCall(new Y(), XS, YS);
  });
});

describe("binary operators", () => {
  it("Add/Sub/Mul/Div/Pow scalar", () => {
    expect(Add(new X(), new Y()).call(0.3, 0.4)).toBeCloseTo(0.7);
    expect(Sub(new X(), new Y()).call(0.5, 0.2)).toBeCloseTo(0.3);
    expect(Mul(new X(), new Y()).call(0.5, 0.4)).toBeCloseTo(0.2);
    expect(Div(new X(), new Y()).call(0.5, 0.4)).toBeCloseTo(1.25);
    expect(Pow(new X(), new Y()).call(2, 3)).toBeCloseTo(8);
  });
  it("constructors coerce bare numbers", () => {
    const e = Add(1, 2);
    expect(e).toBeInstanceOf(AdditionExpression2D);
    expect(e.call()).toBe(3);
  });
  it("binary grids match call", () => {
    expectGridMatchesCall(Add(new X(), new Y()), XS, YS);
    expectGridMatchesCall(Mul(new X(), new Y()), XS, YS);
    expectGridMatchesCall(Pow(new X(), Add(new Y(), 1)), XS, YS);
  });
});

describe("division-by-zero guard", () => {
  it("scalar returns 0", () => {
    const e = Div(new Constant(1), new Constant(0));
    expect(e).toBeInstanceOf(DivisionExpression2D);
    expect(e.call()).toBe(0);
  });
  it("grid zeroes the divide-by-zero cells", () => {
    const e = Div(new Constant(1), new X()); // x==0 in column 0
    const grid = e.evalGrid(XS, YS, XS.length, YS.length);
    for (let i = 0; i < YS.length; i++) {
      expect(grid[i * XS.length]).toBe(0);
    }
    expect(grid.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe("unary functions", () => {
  it("match Math counterparts", () => {
    expect(new Sin(new Constant(0.7)).call()).toBeCloseTo(Math.sin(0.7), 12);
    expect(new Cos(new Constant(0.7)).call()).toBeCloseTo(Math.cos(0.7), 12);
    expect(new Abs(new Constant(-0.7)).call()).toBeCloseTo(0.7, 12);
    expect(new Sqrt(new Constant(0.49)).call()).toBeCloseTo(0.7, 12);
    expect(new Floor(new Constant(2.7)).call()).toBe(2);
    expect(new Ceil(new Constant(2.1)).call()).toBe(3);
  });
  it("unary grids match call", () => {
    for (const Cls of [Sin, Cos, Abs, Sqrt, Floor, Ceil]) {
      expectGridMatchesCall(new Cls(Add(new X(), 0.1)), XS, YS);
    }
  });
  it("sRGB nodes", () => {
    expectGridMatchesCall(new sRGB2Linear(new X()), XS, YS);
    expectGridMatchesCall(new Linear2sRGB(new X()), XS, YS);
  });
});

describe("Min/Max", () => {
  it("scalar", () => {
    expect(new Min(new Constant(0.2), new Constant(0.8)).call()).toBe(0.2);
    expect(new Max(new Constant(0.2), new Constant(0.8)).call()).toBe(0.8);
  });
  it("grids match call", () => {
    expectGridMatchesCall(new Min(new X(), new Y()), XS, YS);
    expectGridMatchesCall(new Max(new X(), new Y()), XS, YS);
  });
});

describe("Log", () => {
  it("natural base by default", () => {
    expect(new Log(new Constant(Math.E)).call()).toBeCloseTo(1, 12);
    expect(new Log(new Constant(Math.E)).logBase).toBe(Math.E);
  });
  it("special bases", () => {
    expect(new Log(new Constant(1000), 10).call()).toBeCloseTo(3, 9);
    expect(new Log(new Constant(8), 2).call()).toBeCloseTo(3, 12);
  });
  it("arbitrary base", () => {
    expect(new Log(new Constant(27), 3).call()).toBeCloseTo(3, 12);
  });
  it("grid special bases match call", () => {
    expectGridMatchesCall(new Log(Add(new X(), 1), 10), XS, YS);
    expectGridMatchesCall(new Log(Add(new X(), 1), 2), XS, YS);
    expectGridMatchesCall(new Log(Add(new X(), 1), 5), XS, YS);
  });
});

describe("coercion helpers", () => {
  it("coerceConstant wraps numbers", () => {
    expect(coerceConstant(5)).toBeInstanceOf(Constant);
    const x = new X();
    expect(coerceConstant(x)).toBe(x);
  });
  it("coerceFloat evaluates expressions at the origin", () => {
    expect(coerceFloat(2.5)).toBe(2.5);
    expect(coerceFloat(new Sqrt(new Constant(4)))).toBeCloseTo(2, 12);
  });
  it("coerceInt rounds expression values (banker's rounding)", () => {
    expect(coerceInt(new Constant(2.5))).toBe(2); // round-half-to-even
    expect(coerceInt(new Constant(1.5))).toBe(2);
    expect(coerceInt(new Constant(2.4))).toBe(2);
    expect(coerceInt(7)).toBe(7);
  });
});

describe("Threshold", () => {
  it("hard above", () => {
    const t = new Threshold(new X(), { above_at: 0.5, above_to: 1 });
    expect(t.call(0.4)).toBeCloseTo(0.4);
    expect(t.call(0.5)).toBeCloseTo(1);
    expect(t.call(0.9)).toBeCloseTo(1);
  });
  it("hard below", () => {
    const t = new Threshold(new X(), { below_at: 0.5, below_to: 0 });
    expect(t.call(0.6)).toBeCloseTo(0.6);
    expect(t.call(0.5)).toBeCloseTo(0);
  });
  it("below_to defaults to below_at", () => {
    const t = new Threshold(new X(), { below_at: 0.3 });
    expect(t.call(0.1)).toBeCloseTo(0.3);
  });
  it("clamp both ends ([0,1])", () => {
    const t = new Threshold(new X(), { below_at: 0, above_at: 1 });
    expect(t.call(-0.5)).toBeCloseTo(0);
    expect(t.call(0.5)).toBeCloseTo(0.5);
    expect(t.call(1.5)).toBeCloseTo(1);
  });
  it("smoothstep midpoint", () => {
    const t = new Threshold(new X(), {
      above_at: 1,
      above_to: 2,
      transition_width: 1,
    });
    // window [0,1]; at v=0.5 smoothstep s=0.5; out = 0.5*0.5 + 0.5*2
    expect(t.call(0.5)).toBeCloseTo(0.5 * 0.5 + 0.5 * 2, 12);
  });
  it("exposes resolved export params", () => {
    const t = new Threshold(new X(), { below_at: 0.3 });
    expect(t.exportParams.below_at).toBe(0.3);
    expect(t.exportParams.below_to).toBe(0.3);
  });
  it("grids match call (hard + smooth)", () => {
    expectGridMatchesCall(
      new Threshold(new X(), { below_at: 0.2, above_at: 0.8 }),
      XS,
      YS,
    );
    expectGridMatchesCall(
      new Threshold(new X(), {
        below_at: 0.2,
        below_to: 0,
        transition_width: 0.3,
      }),
      XS,
      YS,
    );
    expectGridMatchesCall(
      new Threshold(new X(), {
        above_at: 0.7,
        above_to: 1,
        transition_width: 0.3,
      }),
      XS,
      YS,
    );
  });
});
