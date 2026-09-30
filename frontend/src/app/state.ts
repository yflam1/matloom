/**
 * state.ts — the document model: global viewer settings and the layer stack.
 *
 * Layer ordering follows the Photoshop convention: the top-most layer is shown
 * at the top of the list and labelled "Layer N"; the bottom-most layer is at
 * the bottom and labelled "Layer 1". `layers` is stored top-most first; the
 * engine expects bottom-most first, so the list is reversed before evaluation.
 */
import type { EditorHandle } from "./expr-editor";

export type Backend = "webgl" | "webgpu";

export interface Settings {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /**
   * Whether the viewing window was explicitly authored (by import of a
   * `View(...)` line or by editing the region in the panel). Only when true is a
   * `View(...)` line emitted on export; an unauthored default region is omitted,
   * mirroring the Python engine's `_view_set`. Sticky: once authored it stays
   * authored even if the coords are returned to the default.
   */
  viewSet: boolean;
  width: number;
  height: number;
  yUp: boolean;
  axes: boolean;
  ambient: boolean;
  direct: boolean;
  /** Apply the derived normal map (relief shading). */
  normal: boolean;
  /** Apply the derived ambient-occlusion map. */
  occlusion: boolean;
  /** Real vertex displacement of the plane mesh by the height. */
  displace: boolean;
  /**
   * Crop rendering: render only the visible region — the whole plane while it
   * fits the viewport, just the visible sub-rectangle once zoomed in past it —
   * at the user's chosen (always-editable) resolution, so zooming in sharpens
   * for free. Off by default; not persisted.
   */
  crop: boolean;
  backend: Backend;
}

export const DEFAULTS: Settings = {
  x1: 0,
  y1: 0,
  x2: 1,
  y2: 1,
  viewSet: false,
  width: 512,
  height: 512,
  yUp: true,
  axes: false,
  ambient: true,
  direct: true,
  normal: true,
  occlusion: true,
  displace: true,
  crop: false,
  backend: "webgl",
};

/** Channel source text. Multi-component channels nest their fields. */
export interface Channels {
  alpha: string;
  basecolor: { r: string; g: string; b: string };
  metallic: string;
  roughness: string;
  sheen: string;
  coat: string;
  transmission: string;
  ior: string;
  subsurface: string;
  anisotropy: string;
  emissive: { r: string; g: string; b: string; strength: string };
  height: string;
}

/** "01" = inputs are normalized [0,1]; "255" = inputs are integer [0,255]. */
export type ColorMode = "01" | "255";

/**
 * A channel key as used for set-tracking. `basecolor` and `emissive` are atomic
 * (the whole color is set or not), matching how the textual format expresses
 * them (`.basecolor(r, g, b)`). Mirrors the Python engine's channel set.
 */
export type ChannelKey = keyof Channels;

/**
 * Remembered auto-generated noise seeds for one layer, so an unseeded
 * `fBm()`/`Worley()` keeps its seed while the expression is edited around it
 * (and across pin/un-pin). Keyed `fieldKey → argSignature → seeds`; the inner
 * array is a queue so several identical noises in one field stay distinct.
 * Plain JSON so it deep-copies when a layer is duplicated.
 */
export type SeedMemory = Record<string, Record<string, number[]>>;

export interface LayerState {
  id: string;
  enabled: boolean;
  /** Whether the layer card is collapsed in the panel (persists across renders). */
  collapsed: boolean;
  channels: Channels;
  /**
   * The channels the author explicitly set (by import or by editing in the
   * panel). Only these are emitted on export; untouched channels are omitted
   * rather than re-emitted at their default. Sticky: once a channel is edited
   * it stays set even if returned to the default value. A freshly-added layer
   * starts empty (exports `Layer()`). Mirrors the Python engine's
   * `Layer.model_fields_set` tracking.
   */
  setChannels: Set<ChannelKey>;
  colorMode: { basecolor: ColorMode; emissive: ColorMode };
  errors: Record<string, string>;
  editorRefs: Record<string, EditorHandle>;
  seedMemory: SeedMemory;
}

export const defaultChannels = (): Channels => ({
  alpha: "1.0",
  basecolor: { r: "127", g: "127", b: "127" },
  metallic: "0.0",
  roughness: "0.5",
  sheen: "0.0",
  coat: "0.0",
  transmission: "0.0",
  ior: "1.5",
  subsurface: "0.0",
  anisotropy: "0.0",
  emissive: { r: "0", g: "0", b: "0", strength: "0.0" },
  height: "0.0",
});

let uidCounter = 0;
export const uid = (): string => `l${++uidCounter}`;

/**
 * A named, material-wide definition (`Define(name, expr)`). Stored as source
 * text; referenced by bare name from any channel and resolved at evaluation
 * time. Definitions are emitted before `Material(...)` and may reference
 * earlier ones.
 */
export interface MaterialDef {
  name: string;
  src: string;
}

/** Shared, mutable document state. A single instance is created in main.ts. */
export interface Store {
  settings: Settings;
  layers: LayerState[];
  defs: MaterialDef[];
}

export function createStore(): Store {
  return { settings: { ...DEFAULTS }, layers: [], defs: [] };
}
