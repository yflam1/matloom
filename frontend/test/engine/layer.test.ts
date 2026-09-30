/** Tests for src/engine/layer.ts — Color, Emissive, Layer channel clamps. */
import { describe, it, expect } from "vitest";
import { Color, Emissive, Layer } from "../../src/engine/layer";
import { Constant, X } from "../../src/engine/expression";
import { srgbToLinearScalar } from "../../src/engine/grid";

describe("Color", () => {
  it("defaults to black", () => {
    const c = new Color();
    expect(c.raw_r.call()).toBe(0);
    expect(c.raw_g.call()).toBe(0);
    expect(c.raw_b.call()).toBe(0);
  });

  it("channel is sRGB->linear of the clamped raw value", () => {
    const c = new Color({ r: new Constant(0.5) });
    expect(c.r.call()).toBeCloseTo(srgbToLinearScalar(0.5), 12);
  });

  it("clamps out-of-range raw values to [0,1] before conversion", () => {
    expect(new Color({ r: new Constant(2) }).r.call()).toBeCloseTo(
      srgbToLinearScalar(1),
      12,
    );
    expect(new Color({ r: new Constant(-1) }).r.call()).toBeCloseTo(
      srgbToLinearScalar(0),
      12,
    );
  });
});

describe("Emissive", () => {
  it("strength defaults to 0", () => {
    expect(new Emissive().strength.call()).toBe(0);
  });

  it("strength is clamped to non-negative", () => {
    expect(new Emissive({ strength: new Constant(-3) }).strength.call()).toBe(
      0,
    );
  });

  it("allows large strength values", () => {
    expect(new Emissive({ strength: new Constant(100) }).strength.call()).toBe(
      100,
    );
  });

  it("inherits color channels", () => {
    const e = new Emissive({ r: new Constant(1) });
    expect(e.r.call()).toBeCloseTo(srgbToLinearScalar(1), 12);
  });
});

describe("Layer", () => {
  it("has sensible defaults", () => {
    const l = new Layer();
    expect(l.alpha.call()).toBeCloseTo(1);
    expect(l.metallic.call()).toBe(0);
    expect(l.roughness.call()).toBe(0);
    expect(l.height.call()).toBe(0);
    expect(l.basecolor).toBeInstanceOf(Color);
    expect(l.emissive).toBeInstanceOf(Emissive);
  });

  it("clamps alpha/metallic/roughness to [0,1]", () => {
    const l = new Layer({
      alpha: new Constant(2),
      metallic: new Constant(5),
      roughness: new Constant(-2),
    });
    expect(l.alpha.call()).toBeCloseTo(1);
    expect(l.metallic.call()).toBeCloseTo(1);
    expect(l.roughness.call()).toBeCloseTo(0);
  });

  it("leaves height unclamped (may be any value)", () => {
    expect(new Layer({ height: new Constant(9) }).height.call()).toBe(9);
    expect(new Layer({ height: new Constant(-3) }).height.call()).toBe(-3);
  });

  it("defaults the OpenPBR channels to inert values (weights 0, ior 1.5)", () => {
    const l = new Layer();
    expect(l.sheen.call()).toBe(0);
    expect(l.coat.call()).toBe(0);
    expect(l.transmission.call()).toBe(0);
    expect(l.ior.call()).toBeCloseTo(1.5);
    expect(l.subsurface.call()).toBe(0);
    expect(l.anisotropy.call()).toBe(0);
  });

  it("clamps the OpenPBR weight channels to [0,1]", () => {
    const l = new Layer({
      sheen: new Constant(2),
      coat: new Constant(-1),
      transmission: new Constant(5),
      subsurface: new Constant(-0.5),
      anisotropy: new Constant(3),
    });
    expect(l.sheen.call()).toBeCloseTo(1);
    expect(l.coat.call()).toBeCloseTo(0);
    expect(l.transmission.call()).toBeCloseTo(1);
    expect(l.subsurface.call()).toBeCloseTo(0);
    expect(l.anisotropy.call()).toBeCloseTo(1);
  });

  it("clamps ior to >= 1", () => {
    expect(new Layer({ ior: new Constant(0.2) }).ior.call()).toBeCloseTo(1);
    expect(new Layer({ ior: new Constant(2.4) }).ior.call()).toBeCloseTo(2.4);
  });

  it("accepts a coordinate expression for alpha", () => {
    const l = new Layer({ alpha: new X() });
    expect(l.alpha.call(0.3)).toBeCloseTo(0.3);
    expect(l.alpha.call(1.5)).toBeCloseTo(1); // clamped
  });
});
