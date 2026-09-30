/**
 * playwright.config.ts — config for the build-artifact smoke suite.
 *
 * This is the *runtime* layer of the deploy guarantee: it boots the real built
 * `dist/` (served by `vite preview`, same relative base as GitHub Pages) in
 * headless Chromium and verifies the app actually functions — that the eval
 * worker pool spawns, the worker chunk loads with 200 (not the 404 from
 * commit 88dfd82's leaked-placeholder regression), and the first evaluation
 * reaches the OK status.
 *
 * It is deliberately decoupled from vitest (its own config, its own runner) so
 * the node-only unit suite never pulls in a browser environment. Run it via
 * `npm run test:smoke`, never `npm test`.
 */
import { defineConfig } from "@playwright/test";

const PORT = 4173;
const baseURL = `http://localhost:${PORT}`;
const isCI = !!process.env.CI;

export default defineConfig({
  testDir: "./test/smoke",
  // CI runs only the CPU-only deploy-gate specs: boot (eval worker pool),
  // page-math (MathJax on the paper page), and demo-embed (the viewer iframe
  // embedded in that page). layer-colors and randomize drive the GPU/render
  // canvas, which needs a working WebGL backend; CI's headless Linux Chromium
  // has no GPU and gates SwiftShader off by default, so WebGL renders nothing
  // there and those specs can't pass (and burn ~2min each trying). Run the
  // full suite locally with `npm run test:smoke`, where the GPU works.
  testMatch: isCI
    ? ["**/boot.spec.ts", "**/page-math.spec.ts", "**/demo-embed.spec.ts"]
    : ["**/*.spec.ts"],
  // One browser session boots one pool; keep it sequential.
  fullyParallel: false,
  workers: 1,
  forbidOnly: isCI,
  retries: 0, // don't mask real failures; the smoke test must be reliable
  // Per-test budget for the local GPU specs (layer-colors, randomize). Each
  // waits on the first GPU frame via waitForRender() (90s budget) before
  // driving the UI, and layer-colors polls for the recompiled 2-layer frame
  // (60s). The first material-shader compile is slow and variable even on a
  // local GPU (observed 9s to 47s on macOS Metal, cold vs warm driver cache),
  // so the stacked budgets can overrun Playwright's default 30s on a cold run.
  // CI runs only boot (fast), so this only affects local runs. A genuine logic
  // failure still fails fast at each step's own inner timeout (5s layer-count,
  // 15s actionTimeout, 20s dot-ok), so this only spares slow-but-correct runs.
  timeout: 180_000,
  reporter: isCI ? [["list"], ["html", { open: "never" }]] : "list",
  outputDir: "test-results",
  use: {
    baseURL,
    headless: true,
    trace: "on-first-reuse",
    screenshot: "only-on-failure",
    // Generous: cold Chromium + Babylon init + first eval.
    actionTimeout: 15_000,
    navigationTimeout: 15_000,
  },
  // Serve the production build. `vite preview` honors the relative base ("./")
  // from vite.config.ts, reproducing GitHub Pages URL resolution faithfully.
  webServer: {
    command: `npm run preview -- --port ${PORT} --strictPort`,
    url: baseURL,
    reuseExistingServer: !isCI,
    timeout: 60_000,
    cwd: new URL(".", import.meta.url).pathname,
  },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
});
