/**
 * channels.ts — the channel layout spec plus typed accessors for the nested
 * channel source text. A "field key" (fk) like "basecolor.r" addresses one
 * editable value; single-field channels use their own name ("alpha").
 */
import type { Channels } from "./state";

export interface ChannelField {
  /** Field key, e.g. "alpha" or "basecolor.r". */
  fk: string;
  /** Per-field label; omitted for single-field channels. */
  lbl?: string;
}

export interface ChannelSpec {
  key: keyof Channels;
  label: string;
  fields: ChannelField[];
  /** basecolor/emissive carry a 0–1 / 0–255 input-range toggle. */
  hasRangeToggle?: boolean;
}

// Single-field channels (alpha, metallic, roughness) need no per-field label —
// the group header already names them. Multi-field channels (basecolor,
// emissive) label each field. The rule is purely "more than one field", so no
// channel name is special-cased.
export const CHANNEL_SPEC: ChannelSpec[] = [
  { key: "alpha", label: "Alpha", fields: [{ fk: "alpha" }] },
  {
    key: "basecolor",
    label: "Basecolor",
    fields: [
      { fk: "basecolor.r", lbl: "r" },
      { fk: "basecolor.g", lbl: "g" },
      { fk: "basecolor.b", lbl: "b" },
    ],
    hasRangeToggle: true,
  },
  { key: "metallic", label: "Metallic", fields: [{ fk: "metallic" }] },
  { key: "roughness", label: "Roughness", fields: [{ fk: "roughness" }] },
  { key: "sheen", label: "Sheen", fields: [{ fk: "sheen" }] },
  { key: "coat", label: "Coat", fields: [{ fk: "coat" }] },
  {
    key: "transmission",
    label: "Transmission",
    fields: [{ fk: "transmission" }],
  },
  { key: "ior", label: "IOR", fields: [{ fk: "ior" }] },
  { key: "subsurface", label: "Subsurface", fields: [{ fk: "subsurface" }] },
  { key: "anisotropy", label: "Anisotropy", fields: [{ fk: "anisotropy" }] },
  {
    key: "emissive",
    label: "Emissive",
    fields: [
      { fk: "emissive.r", lbl: "r" },
      { fk: "emissive.g", lbl: "g" },
      { fk: "emissive.b", lbl: "b" },
      { fk: "emissive.strength", lbl: "strength" },
    ],
    hasRangeToggle: true,
  },
  { key: "height", label: "Height", fields: [{ fk: "height" }] },
];

/** Channel keys that hold a single scalar source string (no nested fields). */
type ScalarChannelKey =
  | "alpha"
  | "metallic"
  | "roughness"
  | "sheen"
  | "coat"
  | "transmission"
  | "ior"
  | "subsurface"
  | "anisotropy"
  | "height";

/**
 * The atomic channel key a field key belongs to: `"basecolor.r"` -> `"basecolor"`,
 * `"alpha"` -> `"alpha"`. Used for set-tracking (a channel is set as a whole).
 */
export function channelKeyOfFk(fk: string): keyof Channels {
  return fk.split(".")[0] as keyof Channels;
}

export function getChannelValue(ch: Channels, fk: string): string {
  if (fk.startsWith("basecolor."))
    return ch.basecolor[fk.slice(10) as "r" | "g" | "b"];
  if (fk.startsWith("emissive."))
    return ch.emissive[fk.slice(9) as "r" | "g" | "b" | "strength"];
  return ch[fk as ScalarChannelKey];
}

export function setChannelValue(ch: Channels, fk: string, val: string): void {
  if (fk.startsWith("basecolor."))
    ch.basecolor[fk.slice(10) as "r" | "g" | "b"] = val;
  else if (fk.startsWith("emissive."))
    ch.emissive[fk.slice(9) as "r" | "g" | "b" | "strength"] = val;
  else ch[fk as ScalarChannelKey] = val;
}

/** Yield every [fieldKey, sourceText] pair of a channel set, in render order. */
export function* flatChannels(ch: Channels): Generator<[string, string]> {
  yield ["alpha", ch.alpha];
  yield ["basecolor.r", ch.basecolor.r];
  yield ["basecolor.g", ch.basecolor.g];
  yield ["basecolor.b", ch.basecolor.b];
  yield ["metallic", ch.metallic];
  yield ["roughness", ch.roughness];
  yield ["sheen", ch.sheen];
  yield ["coat", ch.coat];
  yield ["transmission", ch.transmission];
  yield ["ior", ch.ior];
  yield ["subsurface", ch.subsurface];
  yield ["anisotropy", ch.anisotropy];
  yield ["emissive.r", ch.emissive.r];
  yield ["emissive.g", ch.emissive.g];
  yield ["emissive.b", ch.emissive.b];
  yield ["emissive.strength", ch.emissive.strength];
  yield ["height", ch.height];
}
