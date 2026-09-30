/**
 * babylon.ts — the single entry point for Babylon.js. Importing classes from
 * their concrete (non-".pure") module paths both pulls in only what the viewer
 * uses (tree-shaking) AND applies the runtime side effects those classes need
 * — most importantly engine.rawTexture, which attaches createRawTexture() to
 * the engine prototype so RawTexture.CreateRGBATexture works. Every other
 * Babylon import in the app goes through this module.
 */
export { Engine } from "@babylonjs/core/Engines/engine.js";
export { WebGPUEngine } from "@babylonjs/core/Engines/webgpuEngine.js";
export { Scene } from "@babylonjs/core/scene.js";
export { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera.js";
export { DirectionalLight } from "@babylonjs/core/Lights/directionalLight.js";
export { Color3, Color4 } from "@babylonjs/core/Maths/math.color.js";
export { Vector3, Matrix } from "@babylonjs/core/Maths/math.vector.js";
export { Viewport } from "@babylonjs/core/Maths/math.viewport.js";
// Pure math (frustum plane extraction for the crop measurement), no side effects.
export { Frustum } from "@babylonjs/core/Maths/math.frustum.js";
export { Mesh } from "@babylonjs/core/Meshes/mesh.js";
export { VertexData } from "@babylonjs/core/Meshes/mesh.vertexData.js";
export {
  CreateLines,
  CreateDashedLines,
} from "@babylonjs/core/Meshes/Builders/linesBuilder.js";
export { PBRMaterial } from "@babylonjs/core/Materials/PBR/pbrMaterial.js";
export { RawTexture } from "@babylonjs/core/Materials/Textures/rawTexture.js";
// RawCubeTexture backs the procedural environment used for image-based ambient
// lighting. Its constructor calls engine.createRawCubeTexture, which lives in
// the same engine.rawTexture extension that engine.js/webgpuEngine.js already
// pull in for createRawTexture (see RawTexture above) — so both backends have
// it without an extra side-effect import. The diffuse-irradiance accessor it
// relies on (BaseTexture.sphericalPolynomial) is registered by the PBR material
// import below (pbrBaseMaterial.js imports baseTexture.polynomial.js).
export { RawCubeTexture } from "@babylonjs/core/Materials/Textures/rawCubeTexture.js";
export { Texture } from "@babylonjs/core/Materials/Textures/texture.js";
export { ImageProcessingConfiguration } from "@babylonjs/core/Materials/imageProcessingConfiguration.js";
// Pure math type (no runtime side effects) used to hand-build a flat, DC-only
// irradiance for the neutral environment instead of convolving a cube map.
export { SphericalPolynomial } from "@babylonjs/core/Maths/sphericalPolynomial.js";
