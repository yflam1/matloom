/**
 * codegen.test.ts — the WGSL codegen and shader assembly for the WebGPU preview
 * tier. These run in node (no GPU): they assert the generated WGSL text and the
 * supported/unsupported capability split, which is what decides whether a
 * material runs on the GPU or falls back to the CPU.
 */
import { describe, it, expect } from "vitest";
import {
  Layer,
  Color,
  X,
  sRGB2Linear,
  Linear2sRGB,
  type Expression2D,
} from "../../src/engine";
import { parseExpr } from "../../src/expr-lang";
import {
  compileExpr,
  exprSupported,
  f32lit,
  GpuUnsupportedError,
  WGSL_PRELUDE,
} from "../../src/gpu/wgsl-codegen";
import { buildComputeShader } from "../../src/gpu/shader";

const wgsl = (src: string, env?: Map<string, Expression2D>): string =>
  compileExpr(parseExpr(src, env), "x", "y");

describe("f32lit", () => {
  it("emits valid WGSL f32 literals", () => {
    expect(f32lit(2)).toBe("2f");
    expect(f32lit(0.5)).toBe("0.5f");
    expect(f32lit(-3)).toBe("-3f");
    expect(f32lit(1e-7)).toBe("1e-7f");
  });
  it("rejects non-finite constants", () => {
    expect(() => f32lit(Infinity)).toThrow(GpuUnsupportedError);
    expect(() => f32lit(NaN)).toThrow(GpuUnsupportedError);
  });
});

describe("compileExpr — supported nodes", () => {
  it("leaves and constants", () => {
    expect(wgsl("5")).toBe("5f");
    expect(wgsl("X()")).toBe("x");
    expect(wgsl("Y()")).toBe("y");
  });
  it("arithmetic", () => {
    expect(wgsl("(X() + 2)")).toBe("(x + 2f)");
    expect(wgsl("(X() - Y())")).toBe("(x - y)");
    expect(wgsl("(X() * 2)")).toBe("(x * 2f)");
    expect(wgsl("((X() + 1) / 2)")).toBe("((x + 1f) / 2f)");
    expect(wgsl("(X() ** 2)")).toBe("pow(x, 2f)");
    expect(wgsl("Min(X(), Y())")).toBe("min(x, y)");
    expect(wgsl("Max(X(), 0)")).toBe("max(x, 0f)");
  });
  it("unary math", () => {
    expect(wgsl("Sin(X())")).toBe("sin(x)");
    expect(wgsl("Cos(Y())")).toBe("cos(y)");
    expect(wgsl("Abs(X())")).toBe("abs(x)");
    expect(wgsl("Sqrt(X())")).toBe("sqrt(x)");
    expect(wgsl("Floor(X())")).toBe("floor(x)");
    expect(wgsl("Ceil(X())")).toBe("ceil(x)");
  });
  it("log bases", () => {
    expect(wgsl("Log(X())")).toBe("log(x)");
    expect(wgsl("Log(X(), 2)")).toBe("log2(x)");
    expect(wgsl("Log(X(), 10)")).toContain("(log(x) / ");
  });
  it("sRGB conversions and threshold", () => {
    // sRGB2Linear / Linear2sRGB are internal nodes (produced by the Color
    // getters), not user-facing DSL, so construct them directly.
    expect(compileExpr(new sRGB2Linear(new X()), "x", "y")).toBe("s2l(x)");
    expect(compileExpr(new Linear2sRGB(new X()), "x", "y")).toBe("l2s(x)");
    expect(wgsl("Threshold(X(), below_at=0.2, above_at=0.8)")).toBe(
      "thr(x, true, 0.2f, 0.2f, true, 0.8f, 0.8f, 0f)",
    );
  });
  it("transforms are coordinate substitutions", () => {
    expect(wgsl("Translate(X(), 0.3, 0.1)")).toBe("(x - 0.3f)");
    expect(wgsl("Translate(Y(), 0.3, 0.1)")).toBe("(y - 0.1f)");
    expect(wgsl("Scale(X(), 2, 4)")).toBe("(x / 2f)");
    // Rotate mixes x and y (per-cell substitution).
    const rot = wgsl("Rotate((X() + Y()), 90)");
    expect(rot).toContain("* x +");
    expect(rot).toContain("* y)");
  });
  it("resolves named refs through their target", () => {
    const env = new Map<string, Expression2D>();
    env.set("g", parseExpr("(X() * 2)", env));
    expect(wgsl("(g + 1)", env)).toBe("((x * 2f) + 1f)");
  });
  it("reports supported expressions", () => {
    expect(exprSupported(parseExpr("Sin((X() * 10))"))).toBe(true);
    expect(exprSupported(parseExpr("Threshold(Cos(Y()), above_at=0.5)"))).toBe(
      true,
    );
  });
});

describe("compileExpr — unsupported nodes fall back", () => {
  for (const src of [
    "fBm(seed=1)",
    "Worley(seed=1)",
    "Bricks()",
    "Fill(Rect(0.2, 0.2, 0.3, 0.3))",
    "Stroke(Ellipse(0.5, 0.5, 0.3, 0.3), 0.05)",
  ]) {
    it(`${src} → unsupported`, () => {
      expect(exprSupported(parseExpr(src))).toBe(false);
      expect(() => compileExpr(parseExpr(src), "x", "y")).toThrow(
        GpuUnsupportedError,
      );
    });
  }
  it("an unsupported node anywhere in the tree taints the whole expression", () => {
    expect(exprSupported(parseExpr("(Sin(X()) + fBm(seed=2))"))).toBe(false);
    expect(exprSupported(parseExpr("Rotate(Fill(Rect(0,0,1,1)), 30)"))).toBe(
      false,
    );
  });
});

describe("buildComputeShader", () => {
  it("emits a complete compute shader with the prelude and outputs", () => {
    const shader = buildComputeShader([
      new Layer({ basecolor: new Color({ r: parseExpr("(X() * Y())") }) }),
    ]);
    expect(shader).toContain(WGSL_PRELUDE.trim().split("\n")[1]); // a prelude line
    expect(shader).toContain("@compute @workgroup_size(8, 8)");
    expect(shader).toContain("fn main(");
    expect(shader).toContain("pack4x8unorm");
    expect(shader).toContain("basecolor[idx]");
    expect(shader).toContain("emf[idx * 3u + 0u]");
    expect(shader).toContain("height[idx]");
  });

  it("unrolls the over-operator across multiple layers", () => {
    const shader = buildComputeShader([
      new Layer({ basecolor: new Color({ r: parseExpr("0.2") }) }),
      new Layer({
        alpha: parseExpr("0.5"),
        basecolor: new Color({ r: parseExpr("0.8") }),
      }),
    ]);
    expect(shader).toContain("let a0 =");
    expect(shader).toContain("let a1 =");
    expect(shader).toContain("let V0 =");
    expect(shader).toContain("let V1 =");
  });

  it("propagates GpuUnsupportedError for a material the GPU can't run", () => {
    expect(() =>
      buildComputeShader([new Layer({ alpha: parseExpr("fBm(seed=1)") })]),
    ).toThrow(GpuUnsupportedError);
  });
});
