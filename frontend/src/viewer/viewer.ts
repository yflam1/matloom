/**
 * viewer.ts — Babylon.js scene: material plane, camera, key light and axes.
 *
 * Coordinate system (Babylon is left-handed):
 *   +X → right, +Y → up, +Z → into the screen.
 * The camera looks straight down +Z at the plane from distance CAM_RADIUS.
 * (CAM_RADIUS is only the initial default; the camera is re-framed to fit the
 * plane after the first setRegion.)
 *
 * Because the height field physically displaces the plane's vertices along Z (a
 * true 3D relief), the camera frames the surface's full *3D* bounding box, not
 * just its flat XY footprint: it targets the box center (mid-depth, so orbiting
 * revolves around the relief rather than the z = 0 origin) and backs off far
 * enough that the whole displaced extent stays visible from any orbit angle. The
 * framing math lives in framing.ts; see placePlane() / applyFrame() below for when
 * it is (re-)applied.
 *
 * The plane is a subdivided unit grid with its corner at the origin, scaled and
 * moved so it lies *along* the Y axis ray in the z = 0 plane: spanning world
 * Y ∈ [y1, y2] (above the origin) when Y points up, and the mirrored
 * Y ∈ [−y2, −y1] (below the origin) when Y points down. Either way the origin
 * corner sits where the X and Y rays meet, so the surface reads as a realization
 * of the XY plane. Its UVs map texel (col, row) → world (x, y), and textures are
 * uploaded with invertY = true so data row 0 (the engine's first row) sits at the
 * plane's top — max-Y for Y-up, the origin row for Y-down. The grid is tessellated
 * so the height field can physically displace its vertices (a true 3D relief).
 *
 * initViewer(backend, canvas) is async and returns an imperative handle.
 * `backend` is "webgl" (default) or "webgpu". A <canvas> locks to its first
 * context type, so callers switch backends by handing in a fresh canvas.
 */
import {
  Engine,
  WebGPUEngine,
  Scene,
  ArcRotateCamera,
  DirectionalLight,
  Color3,
  Color4,
  Vector3,
  Frustum,
  Mesh,
  VertexData,
  CreateLines,
  CreateDashedLines,
  PBRMaterial,
  RawTexture,
  RawCubeTexture,
  Texture,
  ImageProcessingConfiguration,
  SphericalPolynomial,
} from "./babylon";
import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine.js";
import type { LinesMesh } from "@babylonjs/core/Meshes/linesMesh.js";
import { reliefZExtent, fitDistance } from "./framing";
import {
  MESH_SUBDIV,
  visibleRegionFromFrustum,
  type CropMeasurement,
  type Region,
} from "./crop";
import { IOR_PACK_MAX } from "../engine/material";

export type Backend = "webgl" | "webgpu";

/** One frame of evaluated texture data, as produced by evaluateMaterial(). */
export interface Frame {
  width: number;
  height: number;
  basecolor: Uint8Array;
  orm: Uint8Array;
  /** RGBA8 OpenPBR weights: R=sheen, G=coat, B=transmission, A=subsurface. */
  pbr1: Uint8Array;
  /** RGBA8: R=anisotropy, G=ior (normalized over [1,3]), B=0, A=255. */
  pbr2: Uint8Array;
  emissive: Uint8Array;
  emissiveIntensity: number;
  /** RGBA8 OpenGL (+Y) tangent normal; alpha unused. */
  normal: Uint8Array;
  /** Per-texel surface displacement in coordinate units (row-major w*h). */
  displacement: Float32Array;
  /**
   * The region this frame was rendered for. When present, the plane geometry is
   * repositioned to it *atomically* with the texture upload, so crops
   * never flash a stale texture stretched onto new geometry. The camera is not
   * touched.
   */
  region?: { x1: number; y1: number; x2: number; y2: number };
}

/** Imperative handle returned by initViewer(). */
export interface ViewerHandle {
  /**
   * Push freshly evaluated texture data. `settle` marks the authoritative
   * full-resolution frame (vs a low-res preview); only settle frames may
   * auto-re-fit the camera to a changed relief depth, so previews never make the
   * camera jump while typing. Defaults to true.
   */
  applyFrame(frame: Frame, settle?: boolean): void;
  /** Show or hide the material plane (hidden = a blank scene, no surface). */
  setPlaneVisible(visible: boolean): void;
  /** Move + resize the plane, recenter & re-frame the camera to fit it. */
  setRegion(x1: number, y1: number, x2: number, y2: number): void;
  /**
   * Enable/disable the *automatic* relief-driven camera reframe. Off in crop
   * mode, where the camera is reframed only on an explicit {@link setRegion}
   * (authored-region change), never on a content edit or a crop.
   * Defaults to on.
   */
  setAutoFrame(on: boolean): void;
  /**
   * Subscribe to camera view changes (orbit / zoom / pan / programmatic re-frame)
   * and canvas resizes. Returns an unsubscribe function. Used by crop
   * rendering to recompute the rendered region when the view changes.
   */
  onCameraChange(cb: () => void): () => void;
  /**
   * Measure how the given (authored, full) region projects onto the current
   * viewport, for the crop region policy. Pure projection math — no side
   * effects. Returns an invalid measurement when the projection is degenerate
   * (the caller then keeps its previous region).
   */
  measureCrop(full: Region): CropMeasurement;
  /** Return the orbit camera to its head-on home angle and re-frame the plane. */
  resetView(): void;
  /** Show/hide the XYZ axes. */
  setAxesVisible(visible: boolean): void;
  /** Point the Y axis up (true) or down (false). */
  setYAxisDirection(up: boolean): void;
  /**
   * Turn the ambient (image-based) light on or off. This is the soft, fill
   * light from the neutral environment — the term an occlusion map darkens.
   */
  setAmbientEnabled(on: boolean): void;
  /** Turn the directional key light on or off — the crisp highlight term. */
  setDirectEnabled(on: boolean): void;
  /** Apply the derived normal map (relief shading). Independent of displacement. */
  setNormalEnabled(on: boolean): void;
  /** Apply the derived ambient-occlusion map (ORM red channel). */
  setOcclusionEnabled(on: boolean): void;
  /** Real vertex displacement of the plane mesh by the height (a true 3D pop). */
  setDisplaceEnabled(on: boolean): void;
  /** Tear down the engine, scene and listeners. */
  dispose(): void;
}

const CAM_RADIUS = 3.5;
// Head-on home orientation: looking straight down +Z with +Y up. resetView()
// returns the orbit camera to these angles (the constructor's initial values).
const CAM_ALPHA = -Math.PI / 2;
const CAM_BETA = Math.PI / 2;
const FRAME_MARGIN = 1.5; // padding so the framed plane doesn't touch the viewport edges (tunable)
const AXIS_LEN = 1000; // long enough to read as "infinite" within the view
// X and Y axes are coplanar with the flat material plane (both in z = 0). To
// behave like the Z axis — depth-tested and occluded by the surface — they are
// nudged this far behind the plane (+Z, away from the camera) so the flat
// surface (z = 0) wins the depth test over them instead of z-fighting. See the
// axes block below.
const AXIS_Z_BIAS = 1e-3;
const INVERT_Y = true; // data row 0 → top of the plane
// Overflow/fit hysteresis thresholds (fraction of the full region missing from
// the visible bbox). Engage the crop once zoomed in past SET; disengage only
// once zoomed back out past CLEAR. The gap prevents a tiny orbit near the
// just-fits zoom level from toggling the rendered region frame to frame.
const OVERFLOW_SET = 0.02;
const OVERFLOW_CLEAR = 0.005;

// Ambient (image-based) light level. This is the constant irradiance the neutral
// environment emits in every direction; it fills the shadows the single key
// light leaves black and is the term an occlusion (AO) map darkens. Kept below 1
// so the key light still reads as a distinct highlight (tunable).
const AMBIENT_LEVEL = 0.35;

const SAMPLING = Texture.BILINEAR_SAMPLINGMODE;
const UBYTE = Engine.TEXTURETYPE_UNSIGNED_BYTE;

// Plane tessellation (subdivisions per side) for real vertex displacement.
// With the default 512×512 render resolution this makes mesh vertices land on
// texel centers, so the displaced silhouette stays phase-locked with the
// snapped crop lattice. ≈ (SUBDIV+1)² vertices.
const SUBDIV = MESH_SUBDIV;
// Whether Babylon needs the normal map's green channel flipped. Our engine emits
// OpenGL (+Y-up) normals; textures upload with invertY, so verify on screen and
// flip this if relief reads carved-where-it-should-be-raised.
const INVERT_NORMAL_Y = true;

// Thin wrapper over RawTexture.CreateRGBATexture that preserves the exact
// positional-argument shape the app has always used. Note the `gamma` boolean
// is passed in the `creationFlags` slot (the 9th positional arg) — this is the
// long-standing call shape; Babylon's own signature types that position as a
// number, so the call is funneled through here with a single documented cast
// instead of repeating it at four call sites.
function createRGBATexture(
  scene: Scene,
  data: Uint8Array,
  w: number,
  h: number,
  gamma: boolean,
): RawTexture {
  return (RawTexture.CreateRGBATexture as unknown as RGBATextureFactory)(
    data,
    w,
    h,
    scene,
    false,
    INVERT_Y,
    SAMPLING,
    UBYTE,
    gamma,
  );
}

type RGBATextureFactory = (
  data: Uint8Array,
  width: number,
  height: number,
  scene: Scene,
  generateMipMaps: boolean,
  invertY: boolean,
  samplingMode: number,
  type: number,
  gamma: boolean,
) => RawTexture;

// WebGPU needs an async init handshake; WebGL is ready on construction. Both
// take the antialias flag the same way, so the rest of the scene is identical.
async function createEngine(
  canvas: HTMLCanvasElement,
  backend: Backend,
): Promise<AbstractEngine> {
  if (backend === "webgpu") {
    const engine = new WebGPUEngine(canvas, { antialias: true });
    await engine.initAsync();
    return engine;
  }
  return new Engine(canvas, true);
}

export async function initViewer(
  backend: Backend = "webgl",
  canvas: HTMLCanvasElement = document.getElementById(
    "render-canvas",
  ) as HTMLCanvasElement,
): Promise<ViewerHandle> {
  const engine = await createEngine(canvas, backend);
  const scene = new Scene(engine);
  scene.clearColor = new Color4(0.07, 0.07, 0.08, 1);

  // — Tone mapping: ACES filmic, matching Blender's view transform —
  // Without it the linear render is clamped per-channel before the sRGB display
  // conversion, so a bright pure-red emissive just pins R at 255 and never
  // whitens. ACES runs the color through input/output matrices that mix the
  // channels in the highlights, so as emissive strength climbs the surface rolls
  // off toward white — the same bloom Blender shows when an emission shader's
  // Strength is cranked. This applies to the whole scene (lit base color too),
  // exactly as Blender's view transform does.
  scene.imageProcessingConfiguration.toneMappingEnabled = true;
  scene.imageProcessingConfiguration.toneMappingType =
    ImageProcessingConfiguration.TONEMAPPING_ACES;

  // — Ambient: a neutral image-based environment (no skybox) —
  // Set as scene.environmentTexture so every PBR material picks it up as the
  // indirect light source. It only fills shadow and adds reflections; the
  // visible background is still the near-black clearColor above. The Ambient
  // button gates its strength via scene.environmentIntensity (see
  // setAmbientEnabled); the texture itself stays attached.
  const env = makeNeutralEnvironment(scene, AMBIENT_LEVEL);
  scene.environmentTexture = env;

  // — Camera: faces the plane head-on, +X right / +Y up / +Z into screen —
  const cam = new ArcRotateCamera(
    "cam",
    CAM_ALPHA,
    CAM_BETA,
    CAM_RADIUS,
    new Vector3(0.5, 0.5, 0),
    scene,
  );
  cam.lowerRadiusLimit = 0.4;
  cam.upperRadiusLimit = 60;
  // Pull the near clip in close so zooming right up to the surface doesn't clip
  // it away (the default minZ of 1 would clip the plane well before the
  // lowerRadiusLimit of 0.4). Kept above 0 to preserve depth precision.
  cam.minZ = 0.05;
  cam.wheelDeltaPercentage = 0.02;
  // noPreventDefault = false: Babylon must preventDefault() wheel/pointer
  // events on the canvas. Otherwise the browser still performs the default
  // action — on the standalone page that's nothing, but when the app is
  // embedded in an iframe on the paper page the unconsumed wheel scroll
  // chains up and scrolls the host page while the camera zooms.
  cam.attachControl(canvas, false);
  // Left-drag to orbit only. `buttons` exists on the pointers input at runtime
  // but isn't on the narrow ICameraInput base type.
  (cam.inputs.attached.pointers as unknown as { buttons: number[] }).buttons = [
    0,
  ];

  // Fit radius from the last explicit reframe; used to scale pan sensitivity so
  // a drag moves by a screen-constant amount at any zoom level.
  let fitRadius = CAM_RADIUS;

  // — Single key light from the (−X, +Y, −Z) region (travels toward +X,−Y,+Z) —
  // Slightly dimmer than before: the neutral environment now fills what used to
  // be black shadow, so the key light only needs to supply the directional
  // highlight on top of that ambient base.
  const light = new DirectionalLight(
    "key",
    new Vector3(1, -1, 1).normalize(),
    scene,
  );
  light.intensity = 1.2;

  // — Material plane —
  // A tessellated grid (not a single quad) so it can be physically displaced by
  // the height field for a true 3D relief.
  const { mesh: plane, positions: basePositions } = makeGrid(scene, SUBDIV);
  // Displacement rewrites the vertices' Z each frame but Babylon's auto bounding
  // box is only computed from the flat base grid (updateVerticesData below doesn't
  // refresh extents). A stale flat box at z = 0 makes frustum culling drop the
  // whole plane once the camera orbits around the relief's mid-depth target — the
  // taller the relief, the sooner it vanishes. This is the single focal mesh, so
  // skip frustum culling entirely rather than recompute extents every frame.
  plane.alwaysSelectAsActiveMesh = true;
  // Full PBRMaterial (not the simplified PBRMetallicRoughnessMaterial) so we can
  // drive AO (from the ORM red channel) and a normal map, with the flags below
  // as public setters (the simplified subclass required writing private `_` fields).
  const mat = new PBRMaterial("mat", scene);
  mat.backFaceCulling = false;
  mat.twoSidedLighting = true; // light whichever face we look at
  // The plane is alpha-blended (for coverage), which normally skips depth writes;
  // force them on so the displaced relief self-occludes correctly (raised parts
  // hide the parts behind them) instead of drawing in submission order.
  mat.forceDepthWrite = true;
  plane.material = mat;
  mat.transparencyMode = PBRMaterial.PBRMATERIAL_ALPHABLEND;

  // Metallic-roughness workflow driven entirely by the ORM texture: the scalar
  // metallic/roughness multiply the texture, so keep them at 1.
  mat.metallic = 1;
  mat.roughness = 1;
  mat.useRoughnessFromMetallicTextureAlpha = false;
  mat.useRoughnessFromMetallicTextureGreen = true;
  mat.useMetallnessFromMetallicTextureBlue = true;
  // AO lives in the ORM red channel; toggled by setOcclusionEnabled.
  mat.useAmbientOcclusionFromMetallicTextureRed = true;
  // Coverage comes from the albedo alpha (alpha-blend transparency).
  mat.useAlphaFromAlbedoTexture = true;

  // Tie specular highlights and environment reflections to the surface alpha so
  // a fully transparent plane (alpha 0) shows no light glinting off it.
  mat.useSpecularOverAlpha = false;
  mat.useRadianceOverAlpha = false;

  let baseTex = solidRGBA(scene, [128, 128, 128, 255], true);
  baseTex.hasAlpha = true;
  let ormTex = solidRGBA(scene, [255, 128, 0, 255], false);
  let emiTex = solidRGB(scene, [0, 0, 0]);
  // Flat normal (0,0,1) → encoded (128,128,255); alpha unused.
  let nrmTex = solidRGBA(scene, [128, 128, 255, 255], false);
  mat.albedoTexture = baseTex;
  mat.metallicTexture = ormTex;
  mat.emissiveTexture = emiTex;
  mat.emissiveColor = Color3.Black();
  mat.bumpTexture = nrmTex;
  mat.invertNormalMapY = INVERT_NORMAL_Y;

  // Relief feature toggles. Normal (the bump texture) and AO (the ORM red
  // channel) are independent shading terms; displacement is independent geometry
  // handled separately (applyDisplacement).
  let normalOn = true;
  let occlusionOn = true;
  function applyReliefMode(): void {
    mat.bumpTexture = normalOn ? nrmTex : null;
    mat.useAmbientOcclusionFromMetallicTextureRed = occlusionOn;
  }

  // OpenPBR reflectance extras (sheen / coat / transmission / subsurface /
  // anisotropy / ior) are evaluated per-texel by the engine and packed into the
  // pbr1/pbr2 maps. The browser preview drives Babylon's PBRMaterial sub-blocks
  // from each channel's MEAN over the frame — these channels are constant for
  // the overwhelming majority of materials, so the mean is exact there and a
  // faithful approximation when they vary (the Python export + Blender render
  // path is fully spatially-varying). Each sub-block is enabled only when its
  // mean weight is non-zero, so default materials keep the original shader.
  function meanByte(buf: Uint8Array, comp: number): number {
    let sum = 0;
    const n = buf.length / 4;
    for (let i = 0; i < n; i++) sum += buf[i * 4 + comp];
    return n === 0 ? 0 : sum / n / 255;
  }
  function applyPbrExtras(pbr1: Uint8Array, pbr2: Uint8Array): void {
    const sheen = meanByte(pbr1, 0);
    const coat = meanByte(pbr1, 1);
    const transmission = meanByte(pbr1, 2);
    const subsurface = meanByte(pbr1, 3);
    const anisotropy = meanByte(pbr2, 0);
    const ior = 1 + meanByte(pbr2, 1) * (IOR_PACK_MAX - 1);

    mat.sheen.isEnabled = sheen > 0;
    mat.sheen.intensity = sheen;

    mat.clearCoat.isEnabled = coat > 0;
    mat.clearCoat.intensity = coat;
    mat.clearCoat.indexOfRefraction = ior;

    // Transmission and subsurface both live under Babylon's subSurface block.
    mat.subSurface.isRefractionEnabled = transmission > 0;
    mat.subSurface.refractionIntensity = transmission;
    mat.subSurface.indexOfRefraction = ior;
    mat.subSurface.isTranslucencyEnabled = subsurface > 0;
    mat.subSurface.translucencyIntensity = subsurface;

    mat.anisotropy.isEnabled = anisotropy > 0;
    mat.anisotropy.intensity = anisotropy;
  }
  applyReliefMode();

  // Displacement state: the latest height field (coordinate units) and a working
  // position buffer reused across updates. Only the Z of each vertex changes;
  // X/Y stay at their flat grid values.
  let displaceOn = true;
  let dispGrid: Float32Array | null = null;
  let dispW = 0;
  let dispH = 0;
  const dispPositions = basePositions.slice();
  // World-Z extent of the displaced surface (0 when flat / displacement off): the
  // current authoritative value, plus the value the camera was last framed at, so
  // applyFrame can re-fit only when the relief depth actually changes.
  let reliefZMin = 0;
  let reliefZMax = 0;
  let framedZMin = 0;
  let framedZMax = 0;
  // Crop-mode frustum-clip slab: a stable world-Z extent used by measureCrop to
  // extrude the region quad, decoupled from the per-crop refit so the visible
  // region never depends on the *previous* crop's relief (which would oscillate
  // frame to frame and flicker). Seeded from the last full-region settle extent
  // when crop turns on, then grown monotonically to cover any deeper relief seen
  // since. The full-region extent is a superset of any crop's, so growing only
  // ever over-covers — the frustum clip never crops away visible surface.
  let cropSlabZMin = 0;
  let cropSlabZMax = 0;
  // Hysteresis for the overflow/fit decision in measureCrop. Once the view is
  // zoomed in enough to engage a crop, it stays engaged until the view zooms out
  // enough that almost the whole authored region is back on screen; once it
  // disengages, it only re-engages past a small zoom-in threshold. The band
  // stops a tiny orbit near the just-fits zoom level from toggling the rendered
  // region between the full authored region and a crop sub-region.
  let lastOverflow = false;
  function applyDisplacement(): void {
    const stride = SUBDIV + 1;
    for (let i = 0; i <= SUBDIV; i++) {
      const ly = i / SUBDIV;
      for (let j = 0; j <= SUBDIV; j++) {
        const vi = (i * stride + j) * 3;
        let dz = 0;
        if (displaceOn && dispGrid)
          dz = sampleGrid(dispGrid, dispW, dispH, j / SUBDIV, ly);
        dispPositions[vi + 2] = -dz; // displace toward the camera (−Z)
      }
    }
    plane.updateVerticesData("position", dispPositions);
  }

  // — XYZ axes: solid colored rays from the origin to +infinity (no fade) —
  // The Z axis points into the screen, so the plane sits between it and the
  // camera and depth-tests it out naturally (group 0). X and Y, though, are
  // coplanar with the flat material plane (both in z = 0): a depth-tested axis
  // there would z-fight with an undisplaced surface, and — worse — once the
  // surface is displaced toward the camera (−Z) by a positive height, the X/Y
  // rays at z = 0 fall *in front* of the displaced surface and draw right
  // through it (the bug: red X / green Y visible through a raised plane). The
  // fix is to nudge the X/Y rays slightly behind the plane (+Z, toward
  // AXIS_Z_BIAS) so the surface — flat or displaced — wins the depth test over
  // them, exactly as it does over the Z ray. All three axes then read as
  // occluded by the material, with no z-fighting.
  const xAxis = makeAxis(
    scene,
    new Vector3(AXIS_LEN, 0, AXIS_Z_BIAS),
    new Color3(1, 0.27, 0.27),
    0,
  );
  const yAxis = makeAxis(
    scene,
    new Vector3(0, AXIS_LEN, AXIS_Z_BIAS),
    new Color3(0.3, 0.95, 0.4),
    0,
  );
  const zAxis = makeAxis(
    scene,
    new Vector3(0, 0, AXIS_LEN),
    new Color3(0.36, 0.56, 1),
    0,
  );
  const negXAxis = makeDashedAxis(
    scene,
    new Vector3(-AXIS_LEN, 0, AXIS_Z_BIAS),
    new Color3(1, 0.27, 0.27),
    0,
  );
  const negYAxis = makeDashedAxis(
    scene,
    new Vector3(0, -AXIS_LEN, AXIS_Z_BIAS),
    new Color3(0.3, 0.95, 0.4),
    0,
  );
  const negZAxis = makeDashedAxis(
    scene,
    new Vector3(0, 0, -AXIS_LEN),
    new Color3(0.36, 0.56, 1),
    0,
  );
  const axes = [xAxis, yAxis, zAxis, negXAxis, negYAxis, negZAxis];
  axes.forEach((a) => a.setEnabled(false));

  engine.runRenderLoop(() => scene.render());

  // Camera-change fan-out: orbit/zoom/pan (and programmatic re-frames) fire the
  // camera's view-matrix observable; a canvas resize changes the projection too.
  // Crop rendering subscribes here to recompute its resolution/region.
  const camListeners = new Set<() => void>();
  function updatePanSensibility(): void {
    const ptr = cam.inputs.attached.pointers as unknown as
      | { panningSensibility: number }
      | undefined;
    if (!ptr) return;
    // 1000 is Babylon's default at the framed fit distance. Zooming in makes
    // the world-texel size smaller, so the same pixel drag must move fewer world
    // units to feel screen-constant; panningSensibility is inverse to speed.
    const sens = (1000 * fitRadius) / Math.max(1e-6, cam.radius);
    ptr.panningSensibility = Math.max(10, Math.min(1_000_000, sens));
  }
  const notifyCam = (): void => {
    updatePanSensibility();
    for (const cb of camListeners) cb();
  };
  const camObserver = cam.onViewMatrixChangedObservable.add(() => notifyCam());
  const onResize = () => {
    engine.resize();
    // Repaint the freshly-resized backbuffer in the same frame. engine.resize()
    // reallocates (and clears) the render target; without an immediate render the
    // cleared canvas can composite black for a frame — a flash visible through the
    // 0.2s panel-collapse animation, which grows the canvas every frame.
    scene.render();
    notifyCam();
  };
  // Watch the canvas element itself, not just window resizes: collapsing the
  // left panel or dragging the divider changes the canvas's CSS size via the
  // flex layout *without* a window resize event. Without
  // this, the backbuffer keeps its old size and the browser stretches it onto the
  // new canvas box — distorting the plane's aspect/scale. ResizeObserver fires for
  // every layout-driven size change, so engine.resize() always matches the canvas.
  const resizeObserver = new ResizeObserver(() => onResize());
  resizeObserver.observe(canvas);
  window.addEventListener("resize", onResize);

  // Plane placement depends on both the region and the Y direction, which are
  // set through independent calls; keep the latest of each and recompute from
  // both. When Y points down the plane is translated into the −Y region so it
  // lies along the down-pointing Y ray: it spans world Y ∈ [−y2, −y1], putting
  // the origin corner at its top. The texels flip to match for free — the engine
  // already emits its first row as min-Y in Y-down mode (vs max-Y in Y-up), and
  // invertY pins that first row to the plane's top edge either way.
  // Two regions, deliberately separate:
  //  • renderRegion — what the plane geometry currently shows (and the texture is
  //    rendered for). In crop mode this is the cropped visible sub-region;
  //    otherwise the authored region. Updated *atomically with the texture* in
  //    applyFrame, so the plane never shows a stale texture stretched onto new
  //    geometry.
  //  • frameTarget — the authored region the camera frames. Only the explicit
  //    setRegion changes it, so orbit/zoom and crop rendering never move the
  //    camera (no "camera drifts off the plane" while the crop shrinks).
  let renderRegion = { x1: 0, y1: 0, x2: 1, y2: 1 };
  let frameTarget = { x1: 0, y1: 0, x2: 1, y2: 1 };
  let yUp = true;
  // Gates the *automatic* relief-driven reframe (refitToRelief). Off in crop
  // mode, where the camera is reframed only on an explicit authored-region change
  // (setRegion), never on a content edit or a crop.
  let autoFrame = true;

  // The two lights toggle independently, but the raw-color "unlit" escape hatch
  // depends on *both* being off, so their state is kept here and the material /
  // tone-mapping mode is recomputed from the pair. With any light on the surface
  // is shaded normally; with both off it drops to verbatim sRGB base color.
  //
  // Why unlit (rather than just dimming both to zero): UNLIT also bypasses ACES
  // tone mapping's highlight roll-off, so a linear 1.0 still encodes to pure
  // #FFFFFF. PBRMaterial exposes `unlit` publicly (the setter handles the shader
  // recompile); that define drops the IBL diffuse/reflection terms too, so the
  // detached-vs-zeroed ambient distinction doesn't matter in this mode.
  let ambientOn = true;
  let directOn = true;
  function applyLightingMode(): void {
    const anyLight = ambientOn || directOn;
    mat.unlit = !anyLight;
    scene.imageProcessingConfiguration.toneMappingEnabled = anyLight;
  }

  function positionPlane(): void {
    const { x1, y1, x2, y2 } = renderRegion;
    // Occupy the [min, max] rectangle on each axis with *positive* extent, so
    // the plane sits at a fixed world location regardless of input order. The
    // flip then comes from the texture alone: evaluateMaterial() already samples
    // xs/ys in reverse when x1>x2 / y1>y2 (matching export()), so reversed bounds
    // surface as a visible mirror instead of being cancelled by a negative scale.
    plane.position.set(
      Math.min(x1, x2),
      yUp ? Math.min(y1, y2) : -Math.max(y1, y2),
      0,
    );
    plane.scaling.set(Math.abs(x2 - x1), Math.abs(y2 - y1), 1);
  }

  function placePlane(): void {
    positionPlane();
    // In crop mode the camera is user-owned — position the plane but never
    // move the camera.
    if (autoFrame) frameCamera();
  }

  // The camera half of placePlane: target the surface's 3D box center and back
  // off to frame it. Always frames the authored region (frameTarget), never the
  // rendered crop, so the camera stays centered on the whole plane.
  function frameCamera(): void {
    const { x1, y1, x2, y2 } = frameTarget;
    // Center of the surface's 3D bounding box. Z is the mid-depth of the relief
    // (0 when flat), so orbiting revolves around the material rather than the
    // z = 0 origin — the relief stays centered as the user drags.
    const cx = (x1 + x2) / 2;
    const cy = (yUp ? y1 + y2 : -(y1 + y2)) / 2;
    const cz = (reliefZMin + reliefZMax) / 2;

    // Re-frame so the whole plane is visible, without throwing away the user's
    // current orbit orientation. setTarget() (default args) recomputes
    // alpha/beta/radius to hold the camera's world position, so capture the
    // viewing angle first and restore it, then set the fit distance.
    const alpha = cam.alpha;
    const beta = cam.beta;
    cam.setTarget(new Vector3(cx, cy, cz));
    cam.alpha = alpha;
    cam.beta = beta;

    // Distance that frames the full 3D box. Uses the camera's vertical FOV
    // (default fovMode = FOVMODE_VERTICAL_FIXED); aspect maps it to the
    // horizontal FOV. With relief depth this becomes a bounding-sphere fit so the
    // surface stays in view at any orbit angle (see framing.ts).
    const halfW = Math.abs(x2 - x1) / 2;
    const halfH = Math.abs(y2 - y1) / 2;
    const halfD = (reliefZMax - reliefZMin) / 2;
    const fit = fitDistance({
      halfW,
      halfH,
      halfD,
      fov: cam.fov,
      aspect: engine.getRenderWidth() / engine.getRenderHeight(),
      margin: FRAME_MARGIN,
    });
    // Raise the zoom-out cap so large/deep planes aren't clamped below their fit
    // distance (max() keeps the original 60 floor for small/normal regions), and
    // push the far clip plane out past the framed sphere so deep relief never
    // gets far-clipped.
    cam.upperRadiusLimit = Math.max(60, fit * 3);
    cam.maxZ = Math.max(cam.maxZ, fit * 4);
    cam.radius = fit; // clamped to [lowerRadiusLimit, upperRadiusLimit] on assignment

    // Remember the depth we framed at, so applyFrame only re-fits when it changes.
    framedZMin = reliefZMin;
    framedZMax = reliefZMax;

    fitRadius = fit;
    updatePanSensibility();
  }

  // Recompute the relief's world-Z extent from the latest displacement grid
  // (respecting the displace toggle), and re-fit the camera if it changed enough
  // to matter. Tolerance scales with the extent so sub-noise wiggles don't nudge
  // the camera, while real height edits do.
  function refitToRelief(): void {
    const { zMin, zMax } = reliefZExtent(dispGrid, displaceOn);
    reliefZMin = zMin;
    reliefZMax = zMax;
    // Crop mode owns the camera and the slab: grow the frustum-clip slab to
    // cover the deepest relief seen since crop turned on (monotonic, never
    // shrinks) so measureCrop stops tracking the per-crop extent and the visible
    // region stops oscillating frame to frame. The full-region extent is a
    // superset of any crop's, so this only ever over-covers — safe.
    if (!autoFrame) {
      cropSlabZMin = Math.min(cropSlabZMin, zMin);
      cropSlabZMax = Math.max(cropSlabZMax, zMax);
      return;
    }
    const tol =
      1e-4 * Math.max(1, Math.abs(zMin), Math.abs(zMax), Math.abs(framedZMin));
    if (Math.abs(zMin - framedZMin) > tol || Math.abs(zMax - framedZMax) > tol)
      placePlane();
  }

  // Measure how much of the authored full region is actually on screen, for
  // the crop region policy (crop.ts): the region quad — extruded over the
  // relief depth, so displaced geometry leaning into view counts — clipped
  // against the camera frustum. Exact at any orbit angle; screen-space
  // sampling is not an option, because at grazing angles the plane's far side
  // projects into an arbitrarily thin band around its horizon line, which any
  // fixed screen grid straddles and misses — cropping away surface that is
  // plainly on screen. Pure read.
  function measureCrop(full: Region): CropMeasurement {
    const rw = engine.getRenderWidth();
    const rh = engine.getRenderHeight();
    const invalid: CropMeasurement = {
      valid: false,
      overflow: false,
      visible: full,
    };
    if (rw < 1 || rh < 1) return invalid;

    const transform = cam.getViewMatrix().multiply(cam.getProjectionMatrix());
    const visible = visibleRegionFromFrustum(
      Frustum.GetPlanes(transform),
      full,
      yUp,
      // Use the stable crop slab (not the per-crop reliefZMin/Max) so the
      // visible region never depends on the previous crop's relief extent.
      cropSlabZMin,
      cropSlabZMax,
    );
    // Looking away from the surface (or fully clipped): nothing to measure —
    // report the full region rather than guess.
    if (!visible) {
      lastOverflow = false;
      return { valid: true, overflow: false, visible: full };
    }
    // Overflow hysteresis: `missing` is the fraction of the full region NOT
    // covered by the visible bbox (0 = full on screen, >0 = zoomed in). Once
    // overflowing, stay overflowing until the view zooms back out past the clear
    // threshold; once not, only engage past the set threshold. The band stops a
    // tiny orbit near the just-fits zoom level from toggling the rendered region.
    const fullDx = Math.abs(full.x2 - full.x1) || 1;
    const fullDy = Math.abs(full.y2 - full.y1) || 1;
    const missing =
      1 -
      Math.min(
        Math.abs(visible.x2 - visible.x1) / fullDx,
        Math.abs(visible.y2 - visible.y1) / fullDy,
      );
    const engage = lastOverflow
      ? missing >= OVERFLOW_CLEAR
      : missing >= OVERFLOW_SET;
    if (engage) {
      lastOverflow = true;
      return { valid: true, overflow: true, visible };
    }
    lastOverflow = false;
    return { valid: true, overflow: false, visible: full };
  }

  return {
    applyFrame(
      {
        width: w,
        height: h,
        basecolor,
        orm,
        pbr1,
        pbr2,
        emissive,
        emissiveIntensity,
        normal,
        displacement,
        region: frameRegion,
      },
      settle = true,
    ) {
      // Reposition the plane to the rendered region *with* the texture swap, so
      // a crop never shows the old texture stretched onto new geometry.
      if (frameRegion) {
        renderRegion = { ...frameRegion };
        positionPlane();
      }
      baseTex = uploadRGBA(scene, baseTex, basecolor, w, h, true);
      baseTex.hasAlpha = true;
      ormTex = uploadRGBA(scene, ormTex, orm, w, h, false);
      emiTex = uploadRGB(scene, emiTex, emissive, w, h);
      nrmTex = uploadRGBA(scene, nrmTex, normal, w, h, false);
      mat.albedoTexture = baseTex;
      mat.metallicTexture = ormTex;
      mat.emissiveTexture = emiTex;
      const e = emissiveIntensity;
      mat.emissiveColor = new Color3(e, e, e);
      applyPbrExtras(pbr1, pbr2);
      applyReliefMode(); // re-attach the (possibly detached) bump texture
      dispGrid = displacement;
      dispW = w;
      dispH = h;
      applyDisplacement();
      // Re-fit only on the authoritative full-res frame: previews (low-res) would
      // report a different extent and make the camera jump while typing.
      if (settle) refitToRelief();
    },
    setPlaneVisible(visible) {
      plane.setEnabled(visible);
    },
    setRegion(x1, y1, x2, y2) {
      // The authored-region setter: it defines both what the camera frames and
      // (until the next crop) what the plane shows, and always re-frames
      // the camera — so changing the region in crop mode re-centers on it.
      frameTarget = { x1, y1, x2, y2 };
      renderRegion = { x1, y1, x2, y2 };
      positionPlane();
      frameCamera();
    },
    setAutoFrame(on) {
      autoFrame = on;
      // Seed/reset the crop frustum-clip slab. Called exactly on crop enable
      // (on=false), disable (on=true) and rebind. On enable, snapshot the last
      // full-region settle extent (reliefZMin/Max still hold it) as a stable,
      // conservative slab; on disable, clear it (measureCrop isn't used then).
      if (on) {
        cropSlabZMin = 0;
        cropSlabZMax = 0;
      } else {
        cropSlabZMin = reliefZMin;
        cropSlabZMax = reliefZMax;
      }
    },
    onCameraChange(cb) {
      camListeners.add(cb);
      return () => camListeners.delete(cb);
    },
    measureCrop(full) {
      return measureCrop(full);
    },
    // Snap the orbit back to the head-on home angle and re-frame the authored
    // region. Restores the full plane geometry too (a crop is replaced
    // on the next frame), so Reset always returns a sane, centered home view.
    resetView() {
      cam.alpha = CAM_ALPHA;
      cam.beta = CAM_BETA;
      renderRegion = { ...frameTarget };
      positionPlane();
      frameCamera();
    },
    setAxesVisible(visible) {
      axes.forEach((a) => a.setEnabled(visible));
    },
    // Flip the Y arrow and mirror the plane to match the coordinate system:
    // +Y up when yUp, −Y up when not.
    setYAxisDirection(up) {
      yUp = up;
      yAxis.scaling.y = up ? 1 : -1;
      negYAxis.scaling.y = up ? 1 : -1;
      placePlane();
    },
    // Ambient (image-based) light. Scaling scene.environmentIntensity is a plain
    // uniform update — no shader recompile — and it gates both the IBL diffuse
    // irradiance and the environment reflection (both ride vLightingIntensity.z).
    // 0 fully removes the ambient contribution; 1 restores it. The unlit escape
    // hatch only engages once the direct light is also off (see applyLightingMode).
    setAmbientEnabled(on) {
      ambientOn = on;
      scene.environmentIntensity = on ? 1 : 0;
      applyLightingMode();
    },
    // Directional key light — the crisp highlight. Toggled via setEnabled(); the
    // unlit escape hatch engages only once the ambient is also off.
    setDirectEnabled(on) {
      directOn = on;
      light.setEnabled(on);
      applyLightingMode();
    },
    // Relief feature toggles — all independent. Normal is the bump (normal-map)
    // shading term, Occlusion is the AO term, Displacement is real geometry.
    setNormalEnabled(on) {
      normalOn = on;
      applyReliefMode();
    },
    setOcclusionEnabled(on) {
      occlusionOn = on;
      applyReliefMode();
    },
    setDisplaceEnabled(on) {
      displaceOn = on;
      applyDisplacement();
      // Track the relief's new Z extent (used by later refits / crop projection),
      // but do NOT re-frame the camera: toggling displacement is a pure geometry
      // change, so the camera stays exactly where it is regardless of crop mode
      // (which already held the camera). positionPlane() is a no-op here — the
      // region hasn't changed — but keeps the plane placement consistent.
      const { zMin, zMax } = reliefZExtent(dispGrid, displaceOn);
      reliefZMin = zMin;
      reliefZMax = zMax;
      // In crop mode, keep the frustum-clip slab monotonic (see refitToRelief).
      // Toggling displacement off yields zMin=zMax=0, which leaves a non-trivial
      // slab unchanged (min/max against 0) — the slab survives a displacement
      // toggle, so the crop region doesn't jump when relief is switched off/on.
      if (!autoFrame) {
        cropSlabZMin = Math.min(cropSlabZMin, zMin);
        cropSlabZMax = Math.max(cropSlabZMax, zMax);
      }
      positionPlane();
    },
    dispose() {
      window.removeEventListener("resize", onResize);
      resizeObserver.disconnect();
      cam.onViewMatrixChangedObservable.remove(camObserver);
      camListeners.clear();
      scene.dispose();
      engine.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// Environment (image-based ambient light)
// ---------------------------------------------------------------------------
// A 1×1×6 neutral cube used purely as a light source, never shown: there is no
// skybox, so the near-black clearColor stays the visible background. Two things
// come off this texture:
//   • diffuse ambient — driven by the texture's spherical polynomial, NOT its
//     pixels. A uniform environment is a DC-only (constant) polynomial, so we
//     hand-build it with addAmbient() instead of convolving a cube map. This is
//     the irradiance that fills the key light's shadows and that an occlusion
//     (AO) map darkens.
//   • specular reflections — sampled from the cube pixels; a 1×1 cube reflects a
//     flat neutral tint, which is what lets metals read as metallic at all.
// The pixels are written in linear space (gammaSpace = false) so the byte level
// and the ambient level match without an sRGB detour.
function makeNeutralEnvironment(scene: Scene, level: number): RawCubeTexture {
  const v = Math.round(level * 255); // same linear level for the 6 reflection faces
  const faces = Array.from({ length: 6 }, () => new Uint8Array([v, v, v, 255]));
  const env = new RawCubeTexture(
    scene,
    faces,
    1,
    Engine.TEXTUREFORMAT_RGBA,
    UBYTE,
    false, // no mip maps — a 1×1 cube has none to generate
    false, // invertY irrelevant for a constant face
    Texture.NEAREST_SAMPLINGMODE,
  );
  env.gammaSpace = false; // pixels are already linear
  // Flat, view-independent irradiance: addAmbient writes only the constant
  // (xx/yy/zz) polynomial terms, so every surface normal receives `level`.
  const sp = new SphericalPolynomial();
  sp.addAmbient(new Color3(level, level, level));
  env.sphericalPolynomial = sp;
  return env;
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
// A subdivided unit grid in the XY plane, corner at the origin, front face
// toward −Z, with UV (col, row) → (x, y). Returned `updatable` so the Z of each
// vertex can be rewritten for displacement; `positions` is the flat base copy.
function makeGrid(
  scene: Scene,
  n: number,
): { mesh: Mesh; positions: Float32Array } {
  const stride = n + 1;
  const count = stride * stride;
  const positions = new Float32Array(count * 3);
  const uvs = new Float32Array(count * 2);
  const normals = new Float32Array(count * 3);
  for (let i = 0; i <= n; i++) {
    for (let j = 0; j <= n; j++) {
      const v = i * stride + j;
      const lx = j / n;
      const ly = i / n;
      positions[v * 3] = lx;
      positions[v * 3 + 1] = ly;
      positions[v * 3 + 2] = 0;
      normals[v * 3 + 2] = -1; // flat front normal (shading comes from the map)
      uvs[v * 2] = lx;
      uvs[v * 2 + 1] = ly;
    }
  }
  const indices: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const a = i * stride + j;
      const b = a + 1;
      const c = a + stride;
      const d = c + 1;
      indices.push(a, b, d, a, d, c); // matches the original quad's winding
    }
  }
  const mesh = new Mesh("plane", scene);
  const data = new VertexData();
  data.positions = positions as unknown as number[];
  data.uvs = uvs as unknown as number[];
  data.normals = normals as unknown as number[];
  data.indices = indices;
  data.applyToMesh(mesh, true); // updatable = true
  return { mesh, positions: positions.slice() };
}

// Bilinear sample of a row-major w*h grid at local (lx, ly) ∈ [0,1]². Local
// y = 1 is the plane's top edge, which is the grid's first row (row 0) — the
// same orientation the textures use (uploaded invertY), so relief lines up.
function sampleGrid(
  g: Float32Array,
  w: number,
  h: number,
  lx: number,
  ly: number,
): number {
  const fx = lx * (w - 1);
  const fy = (1 - ly) * (h - 1);
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, w - 1);
  const y1 = Math.min(y0 + 1, h - 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const top = g[y0 * w + x0] * (1 - tx) + g[y0 * w + x1] * tx;
  const bot = g[y1 * w + x0] * (1 - tx) + g[y1 * w + x1] * tx;
  return top * (1 - ty) + bot * ty;
}

function makeAxis(
  scene: Scene,
  end: Vector3,
  color: Color3,
  groupId = 1,
): LinesMesh {
  const line = CreateLines("axis", { points: [Vector3.Zero(), end] }, scene);
  line.color = color;
  line.renderingGroupId = groupId; // group 1 = always on top; group 0 = depth-tested
  line.isPickable = false;
  return line;
}

function makeDashedAxis(
  scene: Scene,
  end: Vector3,
  color: Color3,
  groupId = 1,
): LinesMesh {
  const line = CreateDashedLines(
    "axis-neg",
    { points: [Vector3.Zero(), end], dashSize: 3, gapSize: 2, dashNb: 5000 },
    scene,
  );
  line.color = color;
  line.renderingGroupId = groupId;
  line.isPickable = false;
  return line;
}

// ---------------------------------------------------------------------------
// Raw textures  (invertY = true → data row 0 maps to the top)
// ---------------------------------------------------------------------------
function solidRGBA(scene: Scene, px: number[], gamma: boolean): RawTexture {
  return createRGBATexture(scene, new Uint8Array(px), 1, 1, gamma);
}

function solidRGB(scene: Scene, px: number[]): RawTexture {
  // WebGPU has no 3-channel texture format, so emissive is stored RGBA with a
  // padded (unused) alpha. WebGL accepts RGBA too, keeping one code path.
  return createRGBATexture(scene, rgbToRgba(new Uint8Array(px)), 1, 1, false);
}

function uploadRGBA(
  scene: Scene,
  tex: RawTexture,
  data: Uint8Array,
  w: number,
  h: number,
  gamma: boolean,
): RawTexture {
  if (tex && tex.getSize().width === w && tex.getSize().height === h) {
    tex.update(data);
    return tex;
  }
  if (tex) tex.dispose();
  return createRGBATexture(scene, data, w, h, gamma);
}

// Expand n*3 RGB bytes to n*4 RGBA with opaque alpha (emissive alpha is unused).
function rgbToRgba(rgb: Uint8Array): Uint8Array {
  const n = (rgb.length / 3) | 0;
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = rgb[i * 3];
    out[i * 4 + 1] = rgb[i * 3 + 1];
    out[i * 4 + 2] = rgb[i * 3 + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

function uploadRGB(
  scene: Scene,
  tex: RawTexture,
  data: Uint8Array,
  w: number,
  h: number,
): RawTexture {
  const rgba = rgbToRgba(data);
  if (tex && tex.getSize().width === w && tex.getSize().height === h) {
    tex.update(rgba);
    return tex;
  }
  if (tex) tex.dispose();
  return createRGBATexture(scene, rgba, w, h, false);
}
