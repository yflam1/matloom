/**
 * gpu-backend.test.ts — the GpuBackend's CPU-fallback and capability behavior.
 *
 * The node test runner has no WebGPU, so `ensureDevice()` resolves false and
 * every evaluate() transparently falls back to the wrapped CPU backend. That is
 * exactly the property that matters for correctness: with or without a GPU, the
 * GpuBackend must always return a valid frame — and, on fallback, the
 * byte-identical CPU result. The GPU dispatch itself is browser-only and
 * verified manually.
 */
import { describe, it, expect, vi } from "vitest";
import { type Expression2D } from "../../src/engine";
import { parseExpr } from "../../src/expr-lang";
import {
  buildSpec,
  evalSpec,
  FIELD_KEYS,
  type MaterialSpec,
} from "../../src/eval-core";
import { SyncBackend, type EvalBackend } from "../../src/eval/backend";
import { GpuBackend, webgpuSupported } from "../../src/gpu/gpu-backend";

function specOf(
  fieldSrc: Record<string, string>,
  w = 12,
  h = 12,
): MaterialSpec {
  const env = new Map<string, Expression2D>();
  const fields: Record<string, Expression2D> = {};
  for (const key of FIELD_KEYS)
    fields[key] = parseExpr(
      fieldSrc[key] ?? (key === "alpha" ? "1" : "0"),
      env,
    );
  return buildSpec([fields], env, {}, w, h);
}

describe("webgpuSupported", () => {
  it("is false under the node test runner (no navigator.gpu)", () => {
    expect(webgpuSupported()).toBe(false);
  });
});

describe("GpuBackend.supports", () => {
  it("accepts the analytic subset", () => {
    const gpu = new GpuBackend(new SyncBackend());
    expect(
      gpu.supports(
        specOf({ "basecolor.r": "Threshold(Sin((X() * 10)), above_at=0.5)" }),
      ),
    ).toBe(true);
  });
  it("rejects materials using noise / shapes (→ CPU fallback)", () => {
    const gpu = new GpuBackend(new SyncBackend());
    expect(gpu.supports(specOf({ "basecolor.r": "fBm(seed=1)" }))).toBe(false);
    expect(
      gpu.supports(specOf({ alpha: "Fill(Ellipse(0.5, 0.5, 0.3, 0.3))" })),
    ).toBe(false);
  });
});

describe("GpuBackend falls back to the CPU backend with no GPU present", () => {
  it("returns the byte-identical CPU result for a GPU-supported material", async () => {
    const spec = specOf({
      "basecolor.r": "0.6",
      "basecolor.g": "(X() * Y())",
      roughness: "Threshold(Cos((Y() * 8)), below_at=0.3, above_at=0.7)",
      height: "(Sin((X() * 6)) * 4)",
    });
    const gpu = new GpuBackend(new SyncBackend());
    const maps = await gpu.evaluate(spec);
    expect(Array.from(maps.basecolor)).toEqual(
      Array.from(evalSpec(spec).basecolor),
    );
    expect(Array.from(maps.orm)).toEqual(Array.from(evalSpec(spec).orm));
    expect(Array.from(maps.displacement)).toEqual(
      Array.from(evalSpec(spec).displacement),
    );
  });

  it("also falls back (correctly) for an unsupported material", async () => {
    const spec = specOf({
      "basecolor.r": "fBm(base_freq=8, to_01=True, seed=3)",
    });
    const gpu = new GpuBackend(new SyncBackend());
    const maps = await gpu.evaluate(spec);
    expect(Array.from(maps.basecolor)).toEqual(
      Array.from(evalSpec(spec).basecolor),
    );
  });

  it("reports kind 'gpu'", () => {
    expect(new GpuBackend(new SyncBackend()).kind).toBe("gpu");
  });

  it("dispose disposes the wrapped fallback", () => {
    const fallback: EvalBackend = {
      kind: "sync",
      evaluate: () => Promise.reject(new Error("unused")),
      dispose: vi.fn(),
    };
    new GpuBackend(fallback).dispose();
    expect(fallback.dispose).toHaveBeenCalledOnce();
  });
});
