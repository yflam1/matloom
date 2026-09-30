# MatLoom Viewer (frontend)

An interactive, fully client-side viewer for the MatLoom layered-material DSL.
You build a stack of layers, write per-channel expressions (a small Python-like
language), and the material is evaluated **in the browser** and rendered live on
a 3D plane with [Babylon.js](https://www.babylonjs.com/).

There is no backend: the entire material engine — a faithful port of
`matloom/engine/*.py` — runs in TypeScript and produces the same texture maps the
Python pipeline does.

## Scripts

```bash
npm install        # install dependencies (first time only)
npm run dev        # live-reloading dev server: / is the project page, /viewer/ the app
npm run build      # type-check + bundle + obfuscate → dist/
npm run build:check   # static post-build check: no leaked Vite placeholders
npm run test:smoke    # headless-Chromium smoke test of the built dist/
npm run build:verify  # build + build:check + test:smoke (the full pre-push gate)
npm run preview     # serve the production build locally
npm run typecheck  # type-check only, no emit
```

`npm run build` is what GitHub Pages deploys: `dist/index.html` is the paper
project page served at the site root, and `dist/viewer/` is this app under the
`/viewer/` sub-path (one Vite build with two HTML entries; the relative vite
base makes the nesting work unchanged). The project page also embeds the viewer
itself: `demoSection()` in `src/page/render.ts` places a same-origin
`<iframe src="viewer/">` between the teaser strip and the abstract, so the app
boots live inside the page. An iframe because the two documents share element
ids (`#app`, `#panel`, …) and conflicting global styles (the app's stylesheet
resets everything and paints `html`/`body` dark), so an in-page mount is not
viable. The page uses one uniform container width for every section, demo
included (`.page-container`, 1120px, in `src/page/style.css` — between Bulma's
960px `is-max-desktop` and 1216px `is-widescreen`). The frame is immediately
interactive: the camera
is attached with `noPreventDefault = false` (`src/viewer/viewer.ts`), so
wheel/drag over the canvas act on the material instead of chaining the scroll
up to the host page. The standalone `/viewer/` page remains linked as the
full-screen mode.

Because the obfuscator runs only on
`build` (never `dev`), a regression can compile green yet break at runtime — so
`build:check` and `test:smoke` verify the _built_ bundle actually functions
(see [Build guarantee](#build-guarantee)).

## Tooling

- **[Vite](https://vite.dev)** — dev server and production bundler.
- **TypeScript** (`strict`) — every module is fully typed.
- **[@babylonjs/core](https://www.npmjs.com/package/@babylonjs/core)** —
  imported module-by-module (not the monolithic UMD bundle) so the build only
  ships the parts the viewer uses. See [`src/viewer/babylon.ts`](src/viewer/babylon.ts).
- **[vite-plugin-javascript-obfuscator](https://www.npmjs.com/package/vite-plugin-javascript-obfuscator)**
  — obfuscates the application bundle on `build` only (never in `dev`). The
  Babylon chunk is excluded (it is large and already minified), and the
  performance-sensitive transforms (control-flow flattening, etc.) are disabled
  so the per-pixel engine loops stay fast. See [`vite.config.ts`](vite.config.ts).
- **[Playwright](https://playwright.dev)** (`@playwright/test`) — a build-artifact
  smoke test (`test:smoke`), run against the served `dist/` in headless Chromium.
  It is the only test that boots the real app and the real eval-worker spawn
  path, so it catches build-only regressions the node unit suite cannot. It runs
  under its own config and is excluded from `vitest run` (see
  [Build guarantee](#build-guarantee)).

## Source layout

```
src/
├── app/        UI + application controller (the only DOM-aware code)
├── engine/     material evaluation engine (mirror of matloom/engine/*.py)
├── expr-lang/  the channel expression language (parser + registry)
├── eval/       evaluation backends (sync / worker / tile pool) + the worker
├── gpu/        WebGPU compute preview tier (WGSL codegen + GPU backend)
├── eval-core.ts  the DOM-free spec → maps core shared by every backend
├── generate.ts   grammar-free random material generator (mirror of matloom/engine/generate.py)
├── viewer/     Babylon.js 3D scene
└── style.css   all styling (imported by app/main.ts)
index.html      Vite entry; loads src/app/main.ts as a module
```

The dependency direction is one-way and acyclic:

```
app  ──▶  eval(-core) ──▶  expr-lang  ──▶  engine
 │            │   └─────────▶  gpu  ────────▶  engine
 └────────▶  viewer  ──────────────────────▶  engine   (viewer depends only on engine types)
```

### `engine/` — material evaluation (no DOM, no Babylon)

A direct, **numerically byte-identical** port of the Python engine. Pure
computation over `Float32Array` grids.

| File             | Responsibility                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grid.ts`        | `Float32Array` grid primitives (`gridAdd`, `gridMin`, `gridMul`, …), the `Grid`/`Axis` types, sRGB↔linear conversions, and `toByte` (the [0,1]→uint8 quantizer).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `opensimplex.ts` | OpenSimplex noise, ported from `opensimplex`. Uses `BigInt` for the 64-bit seed LCG so values match Python exactly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `expression.ts`  | The `Expression2D` tree: leaves (`Constant`, `X`, `Y`), unary (`Sin`, `Cos`, `Log`, `Abs`, `Sqrt`, `Floor`, `Ceil`, sRGB conversions), binary (`Add`/`Sub`/`Mul`/`Div`/`Pow`/`Min`/`Max`), and `Threshold`. A `children()` accessor makes the tree walkable, and a scalar `call()` evaluates a node at a single point. Bare numbers coerce to `Constant` where an expression is expected; conversely an expression passed where a plain number is expected (e.g. `fBm({ octaves: new Log(1000, 10) })`) is evaluated to a scalar via `call()`.                                                                                                        |
| `noise.ts`       | The `fBm` and `Worley` noise expressions and their shared `NoiseExpression2D` base (seed tracking, per-node `reseed()`, grid memoization). `collectNoise()` walks an expression for its noise nodes.                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `layer.ts`       | `Color`, `Emissive`, `Layer` — group channels and expose them as threshold-clamped expressions. `Layer` also carries the `height` channel (absolute coordinate units, unclamped).                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `relief.ts`      | `compositeHeight` (the max operator `maxᵢ { hᵢ : αᵢ > 0 }` — alpha gates coverage only, no scaling), `heightToNormal` (OpenGL +Y tangent normal) and `heightToAo` (cavity AO; per-axis blur radius `radiusX`/`radiusY` so `finishMaps` can scale it by the crop zoom and hold the blur's **world** radius — and thus the darkening — invariant as a crop narrows the rendered region; the normal stays the raw per-texel gradient and sharpens on zoom). Byte-for-byte mirror of `matloom/engine/relief.py`.                                                                                                                                          |
| `compositing.ts` | `processAlphas` (over-operator coverage) and `weightedBlend` (per-channel alpha-weighted blend).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `material.ts`    | `evaluateMaterial()` — composites a layer stack into the basecolor / ORM (AO in red) / emissive maps plus the RGBA normal map and displacement; derives the relief via `relief.ts` (height shares the X/Y coordinate unit; lateral spacing is region span ÷ resolution). Split into a per-cell phase (`evalChannels`, parallelizable by row band — see [Evaluation backends](#evaluation-backends)) and a whole-grid finishing phase (`finishMaps`: relief + emissive normalization); `assembleChannels` stitches row stripes back together. A constant height field skips the AO box-blur / normal derivation (a provably byte-identical fast path). |
| `index.ts`       | Public barrel; the only entry point `app`/`expr-lang` import from.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

### `expr-lang/` — the channel expression language

Turns a string like `fBm(base_freq=2, seed=5)` or `Sin(X())*0.5 + 0.5` into an
`Expression2D`. Also powers the editor's autocomplete and signature help.

| File           | Responsibility                                                                                                                                                                                                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `registry.ts`  | The single ordered list of callable functions and value constants (`fBm`, `Worley`, `Sin`, `Min`, `pi`, …). Each entry carries its parameter specs (for the signature popup) and a `build()` (for the parser). **Adding an entry here exposes a new function in both the language and the UI.** |
| `parser.ts`    | Tokenizer + recursive-descent parser. `parseExpr(src)` returns an `Expression2D` (positional + keyword args, operator precedence, Python literals). Tokens carry source offsets, so each noise call is tagged with its source span.                                                             |
| `serialize.ts` | The inverse of `parser.ts`. `serializeExpr(expr)` turns a parsed tree back into canonical DSL source (`Constant`s become bare numbers, auto-seeded noise has its seed baked in). Used by the material import/export feature; the output re-parses on both this engine and the Python one.       |
| `caret.ts`     | `callContextAt` / `tokenAt` — caret-position analysis used by the editor popups. Never throw, so they run on every keystroke.                                                                                                                                                                   |
| `index.ts`     | Public barrel.                                                                                                                                                                                                                                                                                  |

### `viewer/` — the Babylon.js scene

| File         | Responsibility                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `babylon.ts` | The **single** place Babylon is imported. Re-exports the specific classes used, via their non-`.pure` module paths so the required runtime side effects (notably `engine.rawTexture`, which also backs the `RawCubeTexture` used for the ambient environment) are applied.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `viewer.ts`  | `initViewer(backend, canvas)` builds the material plane (a full Babylon `PBRMaterial`), camera (auto-framed to the region **and the relief's 3D depth** — see `framing.ts`), a directional key light, a neutral image-based ambient environment, and XYZ axes, and returns an imperative `ViewerHandle` (`applyFrame`, `setRegion`, `resetView`, `setAxesVisible`, `setYAxisDirection`, `setAmbientEnabled`, `setDirectEnabled`, `setNormalEnabled`, `setOcclusionEnabled`, `setDisplaceEnabled`, plus `setAutoFrame` / `onCameraChange` / `measureCrop` for crop rendering, and `dispose`). `applyFrame` can carry the rendered `region`, repositioning the plane atomically with the texture. The displacement mesh is tessellated to match the default 512×512 resolution, and pan sensitivity scales inversely with zoom so a drag moves screen-constant amounts. Supports WebGL and WebGPU. |
| `framing.ts` | Pure camera-framing math (no Babylon, unit-tested): `reliefZExtent` (world-Z range of the displaced surface) and `fitDistance` (rectangle fit when flat; bounding-sphere fit when there's relief depth, so the surface stays visible at any orbit angle).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `crop.ts`    | Pure crop-region policy (no Babylon, unit-tested): turns an on-screen `CropMeasurement` (from the viewer's `measureCrop`) into the `Region` to render. It chooses _only the region_, never the resolution (that stays the user's manual width/height): the whole authored region while it fits the viewport (`!overflow`), and the visible sub-rectangle once zoomed past it — `computeCropRegion` inflates it by `margin`, snaps it to a **world-anchored texel grid** (so small pans keep the same texels and only reveal new ones at the edges), and clamps to the authored region. Because the resolution is fixed, the same texels cover a smaller slice as you zoom in, so effective sharpness climbs for free at flat cost.                                                                                                                                                               |
| `index.ts`   | Public barrel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

### `app/` — UI and application controller

The only DOM-aware code. A single mutable `Store` (settings + layer stack) is
shared by the modules below; editing a channel debounces an evaluation, which
parses each layer, calls `evaluateMaterial`, and pushes the result to the viewer.

| File                 | Responsibility                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.ts`            | Entry point. Imports the CSS, builds the store / viewer manager / evaluator / UI modules, and wires them in the original init order.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `state.ts`           | The document model: `Settings`, `LayerState`, `Channels`, and the `Store`. Documents the top-most-first layer ordering convention.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `dom.ts`             | `byId()` — a typed, throwing `getElementById`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `channels.ts`        | The `CHANNEL_SPEC` layout table plus typed get/set/iterate helpers over the nested channel source text.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `eval.ts`            | `createEvaluator` — debounced scheduling, low-res preview, parsing each layer, the noise-seed lifecycle, status dot, and per-field parse errors. Serializes the resolved trees into a transferable `MaterialSpec` and dispatches it to a pluggable [evaluation backend](#evaluation-backends) (the CPU tile pool by default, off the main thread), applying the result behind a generation guard. Caches parsed expressions per field so unchanged channels are not re-parsed, and drives the reseed UI.                                                                                                                                                                                                                                |
| `expr-editor.ts`     | `ExprEditorUI` — the channel `<input>`s plus the two body-level singletons (completion dropdown + signature popup) that assist whichever input is focused, and the per-field noise chips (reveal seed, click-to-copy, per-node reroll, hover-to-locate).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `layer-panel.ts`     | `LayerPanel` — renders the layer cards (channel groups, eye/duplicate/delete, 0–1/0–255 toggle) and the header drag-to-reorder.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `defs-panel.ts`      | `DefsPanel` — the "Definitions" section: add/edit/delete `Define(name, expr)` bindings (a name field + a full expression editor each), reference them by bare name from any channel. Resolves in order and shows per-card name/expression errors. Backed by `store.defs`.                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `material-io.ts`     | Import/export of the layer stack (and the `Define(…)` preamble) to the `Material(Layer(…)…)` text format: `serializeMaterial` (visible layers + defs → string, seeds baked, colors in 0–255) and `parseImportedLayers` (string → fresh layer states + defs). Mirrors `matloom/engine`'s `serialize()` / `deserialize()`.                                                                                                                                                                                                                                                                                                                                                                                                                |
| `io-controls.ts`     | `IoControls` — the Import / Export actions (in the toolbar's ⋯ overflow menu). Export opens a modal showing the string read-only with **Copy** and timestamped **Download**; Import opens a modal (textarea + file picker) and rebuilds the stack on Apply, validating the whole string first so a bad import never clobbers the current material.                                                                                                                                                                                                                                                                                                                                                                                      |
| `viewer-controls.ts` | `ViewerControls` — the View section of the left panel (above Definitions, so the canvas stays unobstructed): region/resolution inputs, a **Crop** toggle (renders only the visible region; the manual W/H resolution stays live, so zooming in sharpens for free), the Axes / Y-direction / Ambient / Direct / Normal / AO / Displace toggles, a Reset-view action (snaps the camera back to its head-on home angle and re-frames the full 3D relief — centering on the relief's mid-depth and fitting its whole displaced extent; region/resolution are left untouched), and `syncFromSettings()` (re-applies the region after a `View(…)` import), plus their `X` / `Y` / `A` / `D` / `N` / `O` / `H` / `C` / `R` keyboard shortcuts. |
| `crop-controller.ts` | `CropController` — drives the **Crop** toggle. While on, it watches the camera (`onCameraChange`, debounced), measures how the authored region projects on screen (`measureCrop`), derives the render region (`crop.ts`), and pushes it to the evaluator (`setCropRegion`); the cropped region then travels with the frame and repositions the plane atomically. The resolution is never touched (the user owns it). Suppresses the automatic relief reframe (`setAutoFrame(false)`) so crops don't fight the orbit/zoom, but re-frames the camera on an authored-region change/reset so it stays centered on the plane; the authored View + manual resolution are preserved and restored when toggled off.                             |
| `viewer-manager.ts`  | `ViewerManager` — owns the single `ViewerHandle` and the WebGL/WebGPU switch (swaps in a fresh canvas and rebuilds on backend change; rebinds the `CropController` to the fresh viewer). Selecting WebGPU also enables the GPU compute [preview backend](#evaluation-backends); WebGL clears it.                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `divider.ts`         | Drag-to-resize for the layer panel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `layout.ts`          | `initLayout()` — responsive chrome: the collapsible left panel (one `panel-collapsed` class drives both the desktop slide-shut and the mobile slide-in drawer, with a floating opener + backdrop), the toolbar's ⋯ overflow menu (Import / Export / Renderer), the collapsible View section header, and the `matchMedia` breakpoint defaults.                                                                                                                                                                                                                                                                                                                                                                                           |

## Import / Export

The layer stack can be saved to and loaded from a text format via the **Import**
/ **Export** items in the toolbar's ⋯ overflow menu. The same format is understood by the
Python engine (`matloom/engine`), so a material authored in the browser can be
reconstructed server-side and vice versa — the two engines emit byte-for-byte
identical strings.

A third ⋯-menu item, **Randomize**, opens a dialog with the full
`GeneratorConfig` controls (layer/define counts, depth, leaf bias, numeric
ranges, opaque base, randomize-View toggle, seed) over
[`src/generate.ts`](src/generate.ts). It shows
the generated program in an editable preview; **🎲 Regenerate** rolls a fresh
material (blank seed) or reproduces a fixed one, and **Apply** routes the text
through the very same import pipeline (`parseImportedLayers` → store → eval), so
a generated material is just an imported one whose source happens to be random.

### Format

The output is pretty-printed with a 2-space indent:

```
View(<x1>, <y1>, <x2>, <y2>)
Material(
  Layer(<alpha>)
    .basecolor(<r>, <g>, <b>)
    .metallic(<m>)
    .roughness(<rough>)
    .emissive(<r>, <g>, <b>, <strength>)
    .height(<height>),
  Layer(<alpha>)
    …
)
```

- The **`View(…)`** line records the sampled window (`x1, y1, x2, y2`). It is
  always written, and on import it restores the region inputs and re-renders.
- The **first** `Layer` is the bottom-most ("Layer 1"); the store keeps layers
  top-most first, so the order is reversed on the way in and out.
- Each `<…>` is a channel expression in the same DSL the editors use (`fBm`,
  `Worley`, `Sin`, `X()`, operators, …). The parser ignores the indentation and
  newlines, so a hand-flattened single-line string imports just as well.
- **basecolor / emissive RGB use the 0–255 convention**; `alpha`, `metallic`,
  `roughness` and emissive `strength` are in the engine's native domain, and
  `height` is in coordinate units (the same space as X/Y). A color
  group shown in 0–1 mode is scaled (`(<expr> * 255)`) on export, and imported
  colors come in as 0–255 mode.
- Auto-seeded noise has its **current seed baked in** (`fBm(…, seed=N)`), so a
  re-import reproduces the exact same noise instead of re-rolling.
- Only **visible** layers are exported. Visibility is a UI-only flag with no slot
  in the format, so imported layers all come in visible.

### Behavior

- **Export** opens a dialog showing the material string in a read-only textarea,
  with **Copy** (to the clipboard) and **Download** (to
  `material_YYYYMMDD-HHMMSS.txt` — timestamped so repeated downloads never
  collide). It serializes from the parsed expression tree, so formatting is
  canonicalized (`X()*2` → `(X() * 2)`).
- **Import** opens a dialog with a textarea (paste a string) and a _Load file…_
  button (read a `.txt` into the textarea). _Apply_ validates the whole string —
  including every channel expression — before replacing the stack, so a
  malformed import never clobbers the current material.

The browser side lives in [`app/material-io.ts`](src/app/material-io.ts) (the
`Material(…)` grammar) and [`expr-lang/serialize.ts`](src/expr-lang/serialize.ts)
(expression → source). The Python mirror is `LayeredMaterial.serialize()` /
`LayeredMaterial.deserialize()`, backed by `matloom/engine/parser.py` (string →
`Expression2D`) and `matloom/engine/serialize.py` (`Expression2D` → string). The
two parsers/serializers cover the identical expression set, so strings cross
between the engines unchanged.

## Incremental evaluation

Editing one channel does not recompute the rest. `eval.ts` caches each field's
parsed `Expression2D` (keyed by its source text), so unchanged channels are
neither re-parsed nor re-evaluated; `fBm`/`Worley` additionally memoize their
last grid for a given sample resolution. This is purely about skipping redundant
work — the composited maps are unchanged.

## Evaluation backends

A full-resolution re-evaluation is hundreds of thousands of per-cell expression
evaluations. To keep it fast and off the UI thread, evaluation is **pluggable**:
`eval.ts` resolves the layer stack to `Expression2D` trees (owning the seed
lifecycle + noise chips), serializes them to a `MaterialSpec`, and hands that to
an `EvalBackend`. Every backend runs the _same_ engine; they differ only in
_where_ the work happens.

```
eval.ts ──(MaterialSpec)──▶ EvalBackend ──▶ MaterialMaps ──▶ viewer.applyFrame
                              ├─ sync    main thread (fallback)
                              ├─ worker  one dedicated worker
                              ├─ pool    N workers, row-tiled  ← default
                              └─ gpu     WebGPU compute preview (approximate)
```

### The spec boundary (`eval-core.ts`)

A `MaterialSpec` is a plain, structured-cloneable value: the layer stack as
**canonical DSL strings** (via `serializeExpr`, with every auto-seed baked in),
plus the region and resolution. It crosses a `postMessage` boundary unchanged
and re-evaluates to a **byte-identical** result on the far side, because
`serializeExpr` → `parseExpr` is loss-free within one JS engine (numbers
round-trip through `String`/`Number`) and the engine math is deterministic.
`SpecCache.eval()` rebuilds the stack and runs the very same `evaluateMaterial`
the synchronous path always used; a per-cache parse + noise-memo cache keeps the
[incremental evaluation](#incremental-evaluation) behavior, now per worker.

### CPU tiers — `sync`, `worker`, `pool` (`eval/`)

All three are **parity-faithful**: they run `eval-core` unchanged, so their
output is byte-for-byte identical to the single-threaded engine (and to the
Python reference, within the existing cross-engine tolerance).

- **`worker`** moves evaluation onto one dedicated worker
  ([`eval/eval.worker.ts`](src/eval/eval.worker.ts)), so a re-render never
  freezes the UI.
- **`pool`** ([`eval/pool.ts`](src/eval/pool.ts)) is the default and the
  multi-core (numba-`parallel=True`) analog: it splits the grid into contiguous
  **row bands**, evaluates each band's per-cell channel stripe on a separate
  worker, then reassembles and runs the whole-grid finishing pass
  (`finishMaps`) once. This is safe because every per-cell output depends only
  on its own `(x, y)` — no cross-tile coupling and **no reordering of any
  reduction**. The two whole-grid operations (the global emissive max and the
  AO box-blur / normal central-differences) run after assembly, on the complete
  field, exactly as before. Transfer is zero-copy via **transferable
  `ArrayBuffer`s**, so no `SharedArrayBuffer` and therefore no COOP/COEP
  cross-origin isolation is needed — which matters because GitHub Pages (where
  this deploys) cannot set those headers.

Bands are assigned to workers by index, so a worker keeps evaluating the same
rows across edits and its per-band noise memo keeps hitting.

### GPU tier — `gpu` (`gpu/`)

A **WebGPU compute preview** tier. [`gpu/wgsl-codegen.ts`](src/gpu/wgsl-codegen.ts)
translates the analytic node subset (constants, X/Y, arithmetic, unary math,
sRGB, `Threshold`, named refs, and the `Translate`/`Scale`/`Rotate` transforms —
expressed as per-cell coordinate substitutions) into a WGSL expression;
[`gpu/shader.ts`](src/gpu/shader.ts) assembles the per-cell compositing compute
shader, and [`gpu/gpu-backend.ts`](src/gpu/gpu-backend.ts) dispatches it and runs
the same CPU `finishMaps` for relief + emissive normalization.

This tier is **NOT parity-faithful**: GPU transcendentals and fused
multiply-adds are spec'd only to a few ULP and diverge from the CPU/libm engine
beyond the cross-engine tolerance. It is therefore wired only as the _preview_
backend (the low-res pass during rapid edits); the authoritative full-res frame
always comes from a CPU tier. Anything the GPU can't run — no WebGPU,
device-init failure, or a material using a node outside the analytic subset
(noise, shapes, bricks) — **transparently falls back to a CPU worker**, so a
correct frame is always produced.

The GPU tier is tied to the **WebGPU** renderer toggle (`viewer-manager.ts`):
selecting WebGPU turns on the GPU compute preview; WebGL uses the CPU pool for
both passes. The GPU dispatch path runs only in a browser with a GPU, so it is
covered by the codegen/capability/fallback unit tests plus manual verification,
while the CPU tiers are fully unit-tested (including a byte-identity check that
tiled evaluation equals the full grid). The real worker spawn path, Vite's
build-time worker-asset resolution, and the full app boot are exercised by the
[build-guarantee smoke test](#build-guarantee) against the built `dist/`.

### Picking a tier

`createEvaluator` defaults the **settle** (authoritative) backend to the CPU
pool (→ single worker → `sync`, degrading if workers are unavailable) and leaves
the **preview** backend unset until WebGPU is selected. `setSettleBackend` /
`setPreviewBackend` swap tiers at runtime.

### Profiling

`npm run bench` (`test/bench/engine.bench.ts`, excluded from `vitest run`) times
`evaluateMaterial` over representative materials at preview/full resolution,
under cold (fresh tree, noise memo cold) and warm (memo hot) regimes — used to
measure the single-thread kernel cost (the pool/GPU speedups are wall-clock and
browser-only).

## Build guarantee

The obfuscator runs only on `build` (never in `dev`), so a regression can
compile green yet break only in the deployed bundle. The canonical case is
commit `88dfd82`: the obfuscator's `stringArray`/base64 encoding moved Vite's
`__VITE_WORKER_ASSET__<hash>__` placeholder into its decoder array, so Vite's
post-build literal replace found nothing — the raw placeholder shipped, the eval
worker 404'd, and the pool rejected every evaluation with `"pool worker error"`.
This was invisible to `vite build`'s exit code and to the node unit suite (which
drives the pool via fake workers, never the real `new Worker(new URL(...))`
spawn). Two checks close that gap, both runnable locally and run in CI:

1. **`npm run build:check`** (`scripts/check-build-artifacts.mjs`) — a fast,
   browser-free post-build scan. It fails if any `__VITE_WORKER_ASSET__` /
   `__VITE_ASSET__` / `__VITE_PUBLIC_ASSET__` placeholder survives into `dist/`,
   and asserts the `eval.worker-<hash>.js` chunk URL appears as a literal in the
   index chunk (i.e. Vite's placeholder→URL replacement actually happened).

2. **`npm run test:smoke`** (`test/smoke/boot.spec.ts`, Playwright) — boots the
   _built_ `dist/` (served by `vite preview`, same relative base as Pages) in
   headless Chromium and verifies the app actually runs: the eval-worker chunk
   loads with HTTP 200, no `"pool worker error"` or uncaught page error fires,
   and `#status-dot` reaches `dot-ok` on the first evaluation. This is the only
   test that exercises the real worker spawn path and the full app boot.

`npm run build:verify` chains `build` + `build:check` + `test:smoke`. Both
checks run inside the `deploy-web.yml` build job and gate its Pages deploy (a
failing check blocks the artifact upload; the smoke specs navigate to
`/viewer/`, where the app lives inside the built site). The smoke spec lives
under
`test/smoke/` and is excluded from `vitest run`, so the unit suite stays
node-only and browser-free.

## Noise seeds

An unseeded `fBm()` / `Worley()` is given a system seed **once** and then keeps
it: editing the rest of the expression — or any other channel — leaves its noise
put (it does not re-roll on every keystroke). The seed is remembered per layer,
keyed by the noise's arguments, so it survives a re-parse as long as you don't
change that noise's own arguments. Two consequences follow:

- adding a `seed=` to pin a node and then **removing it again** brings the
  original auto-generated seed back, and
- **duplicating a layer** copies its auto seeds, so the copy reproduces the
  original's noise rather than rolling its own.

Each auto-seeded node shows a chip under its field that:

- **reveals** its seed (otherwise hidden),
- **copies** it on click (paste back as `seed=N` to pin it permanently),
- **rerolls** just that node, and
- **highlights** its call in the expression on hover — focusing the field if
  needed, so the noise lights up even when the field isn't focused — making it
  clear which chip maps to which noise when a field has several.

A pinned `seed=` is never touched, and a toolbar button rerolls every auto-seeded
node at once.

## Lighting

The plane is lit by two independent, separately toggled sources, with buttons
under Axes and Y-up:

- **Ambient** (the `A` key) — a neutral, image-based environment that fills the
  shadows the key light leaves dark and adds faint reflections (so metals read
  as metallic). There is **no skybox**: the environment is used purely as a
  light source, so the background stays the near-black clear color. The button
  gates `scene.environmentIntensity`, a plain uniform with no shader recompile.
- **Direct** (the `D` key) — a single directional key light supplying the crisp
  highlight that makes surface relief read as three-dimensional.

Both the specular highlight and the environment reflection are tied to the
surface alpha (the default PBR "specular/radiance over alpha" is turned off), so
a fully transparent plane (alpha 0) shows no light glinting off it. An empty
layer stack (no layers, or all hidden) renders nothing at all — the plane itself
is hidden, not just cleared to transparent.

The ambient environment is procedural: a 1×1×6 neutral cube whose diffuse
irradiance is a flat, DC-only spherical polynomial (built directly rather than
convolved from pixels), so every surface normal receives the same fill. Its
brightness and the key-light intensity are tunable constants in `viewer.ts`.

### Raw sRGB inspection (both off)

Turning **both** lights off drops the plane to its **base color verbatim** —
handy for checking that the displayed color is the exact sRGB value the engine
exported, so a basecolor of `(255, 255, 255)` reads as pure `#FFFFFF`. Two things
give way together to make that readout exact:

- the material is put in **`unlit`** mode, so no diffuse/specular/IBL term scales
  the surface (the key light and the whole ambient contribution drop out); and
- **ACES tone mapping** is bypassed — it mixes channels in the highlights and
  would roll a linear `1.0` down to roughly `0.8`, so true white would never
  read as white.

The base texture is uploaded as sRGB and the linear→sRGB encode on output lives
outside the tone-mapping path, so dropping ACES still leaves an exact
sample-decode / output-encode round-trip. Turning either light back on restores
the lit PBR look (key light and/or ambient + ACES).

## Relief (height → normal, AO, displacement)

Each layer has a `height` channel (absolute **coordinate units** — the same space
as the X/Y plane — default `0`). The layers are composited with a **max** operator
— `H = maxᵢ { hᵢ : αᵢ > 0 }` — not the alpha-weighted blend the other channels
use. Alpha is a pure **coverage mask** for height: it gates _where_ a layer
contributes but does **not** scale the magnitude (a half-opaque layer is not half
as tall), so you type absolute heights (bricks 9, mortar 1) and the tallest
_present_ layer wins. From that height field `relief.ts` derives an OpenGL (+Y-up)
tangent **normal map** and an approximate **ambient-occlusion** map (cavity
darkening); no normal/AO authoring needed.

Because height shares the coordinate-unit space of the plane, the normal/AO
derivation just uses the per-texel lateral spacing (region span ÷ resolution) —
there is no separate scale control, and the displacement equals the height
directly. Three independent toggles (under the light buttons, keys **N** /
**O** / **P**):

- **Normal** (`N`) — apply the derived normal map (relief shading via lighting).
- **AO** (`O`) — apply the ambient-occlusion map (it rides the ORM texture's red
  channel, read via `useAmbientOcclusionFromMetallicTextureRed`).
- **Displace** (`P`) — physically **displace** the plane's mesh vertices by the
  height field, for a true 3D relief with a real silhouette (the plane is a
  tessellated grid, not a single quad; `forceDepthWrite` keeps the raised parts
  self-occluding correctly). Positive height pushes toward the camera (−Z),
  negative away (+Z). The camera follows the relief: it targets the surface's 3D
  mid-depth (so orbiting revolves around the material, not the z = 0 origin) and
  backs off to fit the whole displaced extent, so revolving keeps it all in view.
  It re-fits on Reset, load, region change, this toggle, and whenever the relief's
  min/max depth changes — but not on unrelated edits (color, roughness) or low-res
  previews, so it won't jump while you type. (See `framing.ts`.)

These are genuinely independent (normal = a shading texture, AO = a shading
texture, displacement = real geometry). The exported `height`/`normal`/AO map
**values** match the Python engine (`evaluateMaterial` ↔ `LayeredMaterial.export`,
parity-checked); the displacement is viewer geometry, not part of that contract.

## Behavior parity

This app was ported from a previous vanilla-JavaScript implementation with a hard
"no behavior change" requirement. The engine and parser were validated to be
byte-for-byte identical to the originals across constants, coordinates, trig,
arithmetic, power, thresholds, sRGB conversions, fBm and Worley noise (every
distance/combination), compositing, and both Y-up/Y-down regions.

The element-wise `Abs`, `Sqrt`, `Floor`, `Ceil`, `Log` (natural, base-2,
base-10, arbitrary base), and `Min`/`Max` operators were added afterward to
mirror `matloom/engine/expr.py`. They have no counterpart in the original
vanilla-JS engine, so they fall outside that byte-for-byte baseline. Because any
expression can also be evaluated to a scalar (via `call()`), these double as the
former lowercase `sqrt`/`log`/`abs`/`min`/`max`/`floor`/`ceil` helpers — e.g.
`fBm(base_freq=Sqrt(2))` — which were removed in favor of reusing the operators.

The caching above preserves this — a given expression and seed still evaluate to
the same bytes. The one intentional interactive change is the seed lifecycle: an
unseeded noise now holds its seed (remembered per layer by its argument
signature, and copied when a layer is duplicated) until you reroll it, rather
than re-rolling on each evaluation.
