/**
 * _helpers.ts — shared utilities for the Playwright smoke suite.
 *
 * Playwright's testDir (see playwright.config.ts) only matches files whose name
 * contains a `spec`/`test` segment, so this module is import-only and is never
 * run as a test file.
 */
import type { Page } from "@playwright/test";

/** Count non-background pixels in a PNG screenshot buffer (decoded in-page). */
async function nonBlankPixels(page: Page, png: Buffer): Promise<number> {
  return await page.evaluate(async (b64) => {
    const img = new Image();
    await new Promise<void>((res, rej) => {
      img.onload = () => res();
      img.onerror = () => rej(new Error("screenshot decode failed"));
      img.src = `data:image/png;base64,${b64}`;
    });
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    if (!ctx) return 0;
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] | d[i + 1] | d[i + 2]) n++;
    return n;
  }, png.toString("base64"));
}

/**
 * Wait until the render canvas has painted a real frame.
 *
 * The eval status dot reaches "ok" the moment the worker returns the texture
 * maps, but the GPU still has to compile the material shader and paint the first
 * frame (`engine.runRenderLoop(() => scene.render())` in viewer.ts). While that
 * first compile runs the main thread is busy and CDP `Input.dispatchMouseEvent`
 * (real `locator.click()`) stalls: the click event eventually fires (the menu
 * opens) but Playwright times out waiting for the input ack — the original
 * click-stall failure at `#btn-menu`. A canvas screenshot goes through the
 * compositor, so polling it for non-blank content only resolves once the first
 * frame is painted; after that the thread is idle and UI clicks land promptly,
 * keeping the 15s actionTimeout meaningful for genuine failures.
 *
 * The specs using this run locally, not on CI: headless CI Chromium has no GPU
 * and gates SwiftShader off, so WebGL renders nothing there (boot, which checks
 * the eval worker not the canvas, is the CI gate — see playwright.config.ts).
 * The 90s default absorbs the slow, variable first compile on a local GPU
 * (observed 9s to 47s on macOS Metal, cold vs warm driver cache; the 30s it
 * replaced timed out on the cold end).
 */
export async function waitForRender(
  page: Page,
  timeout = 90_000,
): Promise<void> {
  const deadline = Date.now() + timeout;
  for (;;) {
    try {
      const png = await page.locator("#render-canvas").screenshot();
      if ((await nonBlankPixels(page, png)) > 0) return;
    } catch {
      // Screenshot not ready yet (compositor mid-frame) — keep polling.
    }
    if (Date.now() >= deadline)
      throw new Error(`render canvas stayed blank for ${timeout}ms`);
    await page.waitForTimeout(500);
  }
}
