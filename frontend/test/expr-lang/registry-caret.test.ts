/** Tests for src/expr-lang/registry.ts and caret.ts. */
import { describe, it, expect } from "vitest";
import { EXPR_REGISTRY, REGISTRY_MAP } from "../../src/expr-lang/registry";
import { callContextAt, tokenAt } from "../../src/expr-lang/caret";

describe("registry", () => {
  it("entry names are unique", () => {
    const names = EXPR_REGISTRY.map((e) => e.name);
    expect(new Set(names).size).toBe(names.length);
  });

  it("REGISTRY_MAP indexes every entry", () => {
    for (const e of EXPR_REGISTRY) {
      expect(REGISTRY_MAP.get(e.name)).toBe(e);
    }
  });

  it("exposes the documented callable functions", () => {
    for (const name of ["fBm", "Worley", "Threshold", "Sin", "Min", "Max"]) {
      expect(REGISTRY_MAP.has(name)).toBe(true);
      expect(REGISTRY_MAP.get(name)!.callForm).toBe("call");
    }
  });

  it("pi and e are value constants", () => {
    expect(REGISTRY_MAP.get("pi")!.callForm).toBe("value");
    expect(REGISTRY_MAP.get("e")!.callForm).toBe("value");
    expect(REGISTRY_MAP.get("pi")!.build(undefined)).toBeCloseTo(Math.PI);
    expect(REGISTRY_MAP.get("e")!.build(undefined)).toBeCloseTo(Math.E);
  });

  it("each call entry has params and a builder", () => {
    for (const e of EXPR_REGISTRY) {
      expect(typeof e.build).toBe("function");
      if (e.callForm === "call") expect(Array.isArray(e.params)).toBe(true);
    }
  });
});

describe("callContextAt", () => {
  it("returns null outside any call", () => {
    expect(callContextAt("1 + 2", 5)).toBeNull();
  });

  it("identifies the enclosing function and arg index", () => {
    const ctx = callContextAt("fBm(2, ", 7);
    expect(ctx?.name).toBe("fBm");
    expect(ctx?.argIndex).toBe(1);
  });

  it("detects the active keyword name", () => {
    const ctx = callContextAt("fBm(seed=", 9);
    expect(ctx?.name).toBe("fBm");
    expect(ctx?.activeName).toBe("seed");
  });

  it("resolves the innermost call when nested", () => {
    const ctx = callContextAt("Sin(Mul(X(), ", 13);
    expect(ctx?.name).toBe("Mul");
    expect(ctx?.argIndex).toBe(1);
  });

  it("ignores commas inside string literals", () => {
    const ctx = callContextAt('Worley(combination="F2,F1", ', 28);
    expect(ctx?.name).toBe("Worley");
    expect(ctx?.argIndex).toBe(1);
  });

  it("returns null for an unknown function name", () => {
    expect(callContextAt("Bogus(", 6)).toBeNull();
  });

  it("never throws on incomplete input", () => {
    for (const s of ["fBm(", "((((", "X(),,", '"unterminated']) {
      expect(() => callContextAt(s, s.length)).not.toThrow();
    }
  });
});

describe("tokenAt", () => {
  it("returns the identifier under the caret", () => {
    const t = tokenAt("fBm", 2);
    expect(t?.text).toBe("fBm");
    expect(t?.start).toBe(0);
    expect(t?.end).toBe(3);
  });

  it("returns null inside whitespace", () => {
    expect(tokenAt("a  b", 2)).toBeNull();
  });

  it("does not treat a leading digit run as an identifier", () => {
    expect(tokenAt("123", 2)).toBeNull();
  });

  it("captures a partial identifier for completion", () => {
    const t = tokenAt("Wor", 3);
    expect(t?.text).toBe("Wor");
  });
});
