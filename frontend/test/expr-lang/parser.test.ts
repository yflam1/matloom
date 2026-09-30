/** Tests for src/expr-lang/parser.ts — parsing the channel DSL. */
import { describe, it, expect } from "vitest";
import { parseExpr } from "../../src/expr-lang/parser";
import {
  Constant,
  AdditionExpression2D,
  DivisionExpression2D,
  SubtractionExpression2D,
  MultiplicationExpression2D,
  PowerExpression2D,
  fBm,
  Worley,
  Threshold,
  Bricks,
  Ref,
  NoiseExpression2D,
} from "../../src/engine";

describe("literals", () => {
  it("parses integers and floats", () => {
    expect(parseExpr("42").call()).toBe(42);
    expect(parseExpr("3.5").call()).toBe(3.5);
  });
  it("parses scientific notation", () => {
    expect(parseExpr("1e3").call()).toBe(1000);
    expect(parseExpr("1.5e-2").call()).toBeCloseTo(0.015);
  });
  it("parses pi and e constants", () => {
    expect(parseExpr("pi").call()).toBeCloseTo(Math.PI);
    expect(parseExpr("e").call()).toBeCloseTo(Math.E);
  });
  it("wraps a bare number as Constant", () => {
    expect(parseExpr("7")).toBeInstanceOf(Constant);
  });
});

describe("operators", () => {
  it("addition", () => {
    expect(parseExpr("1 + 2")).toBeInstanceOf(AdditionExpression2D);
    expect(parseExpr("1 + 2").call()).toBe(3);
  });
  it("respects precedence", () => {
    expect(parseExpr("1 + 2 * 3").call()).toBe(7);
  });
  it("parentheses override precedence", () => {
    expect(parseExpr("(1 + 2) * 3").call()).toBe(9);
  });
  it("power is right-associative", () => {
    expect(parseExpr("2 ** 3 ** 2").call()).toBe(512);
  });
  it("unary minus and plus", () => {
    expect(parseExpr("-5").call()).toBe(-5);
    expect(parseExpr("+5").call()).toBe(5);
    expect(parseExpr("-X()").call(0.3)).toBeCloseTo(-0.3);
  });
  it("operator node types", () => {
    expect(parseExpr("5 - 2")).toBeInstanceOf(SubtractionExpression2D);
    expect(parseExpr("5 * 2")).toBeInstanceOf(MultiplicationExpression2D);
    expect(parseExpr("6 / 2")).toBeInstanceOf(DivisionExpression2D);
    expect(parseExpr("5 ** 2")).toBeInstanceOf(PowerExpression2D);
  });
  it("coordinate expressions", () => {
    expect(parseExpr("X() * 2 + 1").call(0.5)).toBeCloseTo(2);
  });
});

describe("function calls", () => {
  it("parses fBm with keyword args", () => {
    const e = parseExpr("fBm(base_freq=2, seed=5)");
    expect(e).toBeInstanceOf(fBm);
    expect((e as fBm).exportParams.base_freq_x).toBe(2);
    expect((e as fBm).currentSeed).toBe(5);
  });
  it("parses fBm with positional args", () => {
    const e = parseExpr("fBm(3, 2.0, 0.5, seed=1)");
    expect((e as fBm).exportParams.octaves).toBe(3);
  });
  it("parses Worley with a string arg", () => {
    const e = parseExpr('Worley(combination="F2-F1", seed=3)');
    expect(e).toBeInstanceOf(Worley);
    expect((e as Worley).exportParams.combination).toBe("F2-F1");
  });
  it("parses Threshold", () => {
    const e = parseExpr("Threshold(X(), below_at=0.2, above_at=0.8)");
    expect(e).toBeInstanceOf(Threshold);
  });
  it("parses Bricks with positional and keyword args", () => {
    const e = parseExpr('Bricks(0.4, 0.2, offset=0.25, axis="column")');
    expect(e).toBeInstanceOf(Bricks);
    expect((e as Bricks).exportParams.brick_width).toBe(0.4);
    expect((e as Bricks).exportParams.offset).toBe(0.25);
    expect((e as Bricks).exportParams.axis).toBe("column");
  });
  it("parses nested calls", () => {
    expect(parseExpr("Sin(Mul(X(), 2))").call(Math.PI / 4)).toBeCloseTo(
      Math.sin(Math.PI / 2),
    );
  });
  it("tags noise nodes with their source span", () => {
    const e = parseExpr("fBm(seed=5)") as NoiseExpression2D;
    expect(e.sourceText).toBe("fBm(seed=5)");
    expect(e.sourceStart).toBe(0);
    expect(e.sourceEnd).toBe(11);
  });
});

describe("errors", () => {
  it("unknown function", () => {
    expect(() => parseExpr("Bogus(1)")).toThrow(/unknown function/);
  });
  it("unknown name", () => {
    expect(() => parseExpr("foobar")).toThrow(/unknown name/);
  });
  it("function referenced without call", () => {
    expect(() => parseExpr("fBm")).toThrow(/is a function/);
  });
  it("too many positional args", () => {
    expect(() => parseExpr("X(1)")).toThrow(/too many positional/);
  });
  it("unknown parameter", () => {
    expect(() => parseExpr("fBm(bogus=1)")).toThrow(/unknown parameter/);
  });
  it("duplicate keyword arg: last value wins (NOTE: diverges from Python, which raises)", () => {
    // The TS parser collapses repeated keywords in an object, so the last
    // value silently wins; matloom/engine/parser.py raises on duplicates.
    // Captured here to document the divergence rather than assert a guarantee.
    const e = parseExpr("fBm(seed=1, seed=2)") as fBm;
    expect(e.currentSeed).toBe(2);
  });
  it("missing required argument", () => {
    expect(() => parseExpr("Sin()")).toThrow(/missing required argument/);
  });
  it("trailing comma", () => {
    expect(() => parseExpr("Min(X(), )")).toThrow(/trailing comma/);
  });
  it("unterminated string", () => {
    expect(() => parseExpr('Worley(combination="F1)')).toThrow(
      /unterminated string/,
    );
  });
  it("unexpected character", () => {
    expect(() => parseExpr("X() @ 2")).toThrow(/unexpected character/);
  });
  it("trailing input", () => {
    expect(() => parseExpr("1 2")).toThrow(/trailing input/);
  });
  it("empty input", () => {
    expect(() => parseExpr("")).toThrow(/expected expression/);
  });
});

describe("definitions (env)", () => {
  it("resolves a name to a Ref that delegates to the stored expression", () => {
    const box = parseExpr("Fill(Rect(0.2, 0.2, 0.6, 0.6))");
    const env = new Map([["box", box]]);
    const ref = parseExpr("box", env);
    expect(ref).toBeInstanceOf(Ref);
    expect(ref.call(0.5, 0.5)).toBe(box.call(0.5, 0.5));
    // Usable anywhere an expression is.
    expect(parseExpr("Translate(box, 0.1, 0)", env).call(0.45, 0.5)).toBe(
      box.call(0.35, 0.5),
    );
  });

  it("an unknown name without an env still throws", () => {
    expect(() => parseExpr("box")).toThrow(/unknown name/);
  });

  it("a bare shape must be wrapped in Fill/Stroke", () => {
    expect(() => parseExpr("Rect(0, 0, 1, 1)")).toThrow(/Fill/);
  });

  it("a bare segment is only valid inside Path", () => {
    expect(() => parseExpr("LineTo(1, 2)")).toThrow(/Path/);
  });
});
