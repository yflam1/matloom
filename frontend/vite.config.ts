import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import obfuscator from "vite-plugin-javascript-obfuscator";

// Relative base so the built site works under any GitHub Pages sub-path
// (https://<user>.github.io/<repo>/) without hard-coding the repo name.
export default defineConfig({
  base: "./",

  build: {
    target: "es2022",
    sourcemap: false,
    rollupOptions: {
      // Two HTML entries in one site: the paper project page at the root and
      // the interactive viewer at /viewer/ (emitted as dist/viewer/index.html,
      // with its entry chunk named after the input key, not "index").
      input: {
        index: fileURLToPath(new URL("./index.html", import.meta.url)),
        viewer: fileURLToPath(new URL("./viewer/index.html", import.meta.url)),
      },
      output: {
        // Babylon is large and already minified upstream; keep it in its own
        // chunk so it is cached independently and excluded from obfuscation.
        // Function form (Rolldown/Vite 8 dropped the object form's typing).
        manualChunks(id) {
          return id.includes("@babylonjs/core") ? "babylon" : undefined;
        },
      },
    },
  },

  plugins: [
    // Moderate obfuscation: string-array encoding, identifier renaming and a
    // light dead-code pass — deliberately WITHOUT control-flow flattening,
    // numbers-to-expressions or self-defending, which would slow the engine's
    // per-pixel loops. Build-only: the dev server stays readable and fast.
    obfuscator({
      apply: "build",
      // Never touch the Babylon vendor chunk (huge + already minified).
      exclude: [/node_modules/, /babylon/],
      options: {
        compact: true,
        simplify: true,
        identifierNamesGenerator: "hexadecimal",
        renameGlobals: false,
        stringArray: true,
        stringArrayEncoding: ["base64"],
        stringArrayThreshold: 0.75,
        stringArrayRotate: true,
        stringArrayWrappersCount: 2,
        unicodeEscapeSequence: true,
        // Vite emits build-time placeholders like __VITE_WORKER_ASSET__<hash>__
        // (and __VITE_ASSET__ / __VITE_PUBLIC_ASSET__) as plain string literals,
        // then does a literal text replace AFTER this plugin runs. Without
        // reserving them, stringArray/base64 encoding moves them into the
        // decoder array, the literal replace finds nothing, and the raw
        // placeholder ships — so `new Worker(new URL(...))` 404s in prod
        // ("pool worker error"). Dev is unaffected (obfuscator is build-only).
        reservedStrings: [
          "__VITE_WORKER_ASSET__",
          "__VITE_ASSET__",
          "__VITE_PUBLIC_ASSET__",
        ],
        // PERMANENTLY OFF — transformObjectKeys is semantically broken (still
        // in javascript-obfuscator 5.5.0): an all-constant object literal
        // inside a multi-call function is hoisted to module scope and shared
        // across calls. It turned defaultChannels()' nested basecolor/emissive
        // `{r,g,b}` (and toLayerState's colorMode/errors/editorRefs/seedMemory)
        // into singletons, so every layer aliased ONE basecolor object and the
        // last layer processed won — the deployed app rendered every layer in
        // the bottom layer's color (top-layer ink invisible on an opaque
        // papyrus base). Dev/unit tests can't see it (build-only transform);
        // test/smoke/layer-colors.spec.ts guards the class.
        transformObjectKeys: false,
        deadCodeInjection: true,
        deadCodeInjectionThreshold: 0.2,
        // Explicitly off — these are the transforms that hurt hot-loop perf.
        controlFlowFlattening: false,
        numbersToExpressions: false,
        splitStrings: false,
        selfDefending: false,
        debugProtection: false,
      },
    }),
  ],
});
