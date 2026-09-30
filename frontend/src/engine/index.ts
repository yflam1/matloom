/**
 * engine — public surface of the material engine. Re-exports the expression
 * tree, noise, layer data classes and the material evaluator. This mirrors
 * matloom/engine and is the only module the UI and expr-lang import from.
 */
export * from "./expression";
export * from "./noise";
export * from "./pattern";
export * from "./transform";
export * from "./shape";
export * from "./layer";
export * from "./relief";
export {
  evaluateMaterial,
  evalChannels,
  finishMaps,
  assembleChannels,
  axisCoords,
  resolveRegion,
  IOR_PACK_MAX,
} from "./material";
export type {
  EvalRegion,
  ResolvedRegion,
  MaterialMaps,
  ChannelData,
} from "./material";
