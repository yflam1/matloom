/**
 * layer.ts — Color / Emissive / Layer data classes. Each exposes its channels
 * as Threshold-clamped expressions, matching matloom/engine/layer.py.
 */
import { Constant, Expression2D, Threshold, sRGB2Linear } from "./expression";

// Channel clamps: [0,1] for normalized channels, [0,∞) for emissive strength,
// [1,∞) for the refractive index.
const clamp01 = (e: Expression2D) =>
  new Threshold(e, { below_at: 0, above_at: 1 });
const clampPositive = (e: Expression2D) => new Threshold(e, { below_at: 0 });
const clampIor = (e: Expression2D) => new Threshold(e, { below_at: 1 });

export interface ColorChannels {
  r?: Expression2D;
  g?: Expression2D;
  b?: Expression2D;
}

export class Color {
  readonly raw_r: Expression2D;
  readonly raw_g: Expression2D;
  readonly raw_b: Expression2D;

  constructor({
    r = new Constant(0),
    g = new Constant(0),
    b = new Constant(0),
  }: ColorChannels = {}) {
    this.raw_r = r;
    this.raw_g = g;
    this.raw_b = b;
  }

  get r(): Expression2D {
    return new sRGB2Linear(clamp01(this.raw_r));
  }
  get g(): Expression2D {
    return new sRGB2Linear(clamp01(this.raw_g));
  }
  get b(): Expression2D {
    return new sRGB2Linear(clamp01(this.raw_b));
  }
}

export interface EmissiveChannels extends ColorChannels {
  strength?: Expression2D;
}

export class Emissive extends Color {
  readonly raw_strength: Expression2D;

  constructor({
    r = new Constant(0),
    g = new Constant(0),
    b = new Constant(0),
    strength = new Constant(0),
  }: EmissiveChannels = {}) {
    super({ r, g, b });
    this.raw_strength = strength;
  }

  get strength(): Expression2D {
    return clampPositive(this.raw_strength);
  }
}

export interface LayerChannels {
  alpha?: Expression2D;
  basecolor?: Color;
  metallic?: Expression2D;
  roughness?: Expression2D;
  // OpenPBR-aligned reflectance channels (each a [0,1] scalar like metallic,
  // except `ior` which is a refractive index >= 1).
  sheen?: Expression2D;
  coat?: Expression2D;
  transmission?: Expression2D;
  ior?: Expression2D;
  subsurface?: Expression2D;
  anisotropy?: Expression2D;
  emissive?: Emissive;
  height?: Expression2D;
}

export class Layer {
  readonly raw_alpha: Expression2D;
  readonly basecolor: Color;
  readonly raw_metallic: Expression2D;
  readonly raw_roughness: Expression2D;
  readonly raw_sheen: Expression2D;
  readonly raw_coat: Expression2D;
  readonly raw_transmission: Expression2D;
  readonly raw_ior: Expression2D;
  readonly raw_subsurface: Expression2D;
  readonly raw_anisotropy: Expression2D;
  readonly emissive: Emissive;
  readonly raw_height: Expression2D;

  constructor({
    alpha = new Constant(1),
    basecolor = new Color(),
    metallic = new Constant(0),
    roughness = new Constant(0),
    sheen = new Constant(0),
    coat = new Constant(0),
    transmission = new Constant(0),
    ior = new Constant(1.5),
    subsurface = new Constant(0),
    anisotropy = new Constant(0),
    emissive = new Emissive(),
    height = new Constant(0),
  }: LayerChannels = {}) {
    this.raw_alpha = alpha;
    this.basecolor = basecolor;
    this.raw_metallic = metallic;
    this.raw_roughness = roughness;
    this.raw_sheen = sheen;
    this.raw_coat = coat;
    this.raw_transmission = transmission;
    this.raw_ior = ior;
    this.raw_subsurface = subsurface;
    this.raw_anisotropy = anisotropy;
    this.emissive = emissive;
    this.raw_height = height;
  }

  get alpha(): Expression2D {
    return clamp01(this.raw_alpha);
  }
  get metallic(): Expression2D {
    return clamp01(this.raw_metallic);
  }
  get roughness(): Expression2D {
    return clamp01(this.raw_roughness);
  }
  // Microfiber sheen weight (cloth/velvet/fabric fuzz lobe).
  get sheen(): Expression2D {
    return clamp01(this.raw_sheen);
  }
  // Clearcoat weight (lacquer, car paint, glaze).
  get coat(): Expression2D {
    return clamp01(this.raw_coat);
  }
  // Transmission weight (glass, ice, gel): how much light passes through.
  get transmission(): Expression2D {
    return clamp01(this.raw_transmission);
  }
  // Index of refraction (>= 1; default 1.5).
  get ior(): Expression2D {
    return clampIor(this.raw_ior);
  }
  // Subsurface-scattering weight (skin, wax, jade); scatter color = basecolor.
  get subsurface(): Expression2D {
    return clamp01(this.raw_subsurface);
  }
  // Specular anisotropy strength (brushed metal, satin); direction = tangent X.
  get anisotropy(): Expression2D {
    return clamp01(this.raw_anisotropy);
  }
  // Absolute displacement in coordinate units (same space as the X/Y plane);
  // left unclamped (it may be any real value, including negative). The per-layer
  // fields are composited with the max operator (alpha as a coverage mask) in
  // relief.ts compositeHeight.
  get height(): Expression2D {
    return this.raw_height;
  }
}
