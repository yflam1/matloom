/**
 * layer-colors.spec.ts — regression guard for the obfuscator's
 * `transformObjectKeys` hoisting bug, invisible to the node unit suite.
 *
 * javascript-obfuscator (4.2.2, still 5.5.0) hoists an all-constant object
 * literal out of a multi-call function into module scope, so every call shares
 * one object. In the built bundle that turned `defaultChannels()`' nested
 * `basecolor`/`emissive` `{r,g,b}` (and `toLayerState()`'s
 * `colorMode`/`errors`/`editorRefs`/`seedMemory`) into singletons: importing
 * an N-layer material wrote every layer's basecolor into the SAME object and
 * the last layer processed won — the viewer rendered every layer in the
 * bottom layer's color (a dark top layer became invisible on an opaque base).
 * Dev mode is never affected (the obfuscator is build-only), and the unit
 * tests exercise the correct source, so only a built-bundle end-to-end check
 * sees it. The option is pinned off in vite.config.ts; this spec catches the
 * class if it is ever re-enabled (or a similar transform is added).
 *
 * The check: import a two-layer material whose bottom layer is opaque red and
 * whose top layer paints the right half blue, let it render, then count
 * red-ish and blue-ish pixels in a canvas screenshot. Both color families
 * must be present; with the hoisting bug the plane is uniformly red (the
 * bottom layer's color wins) and the blue count collapses to ~0.
 *
 * Part of the Playwright smoke suite — run via `npm run test:smoke` (consumes
 * an existing `dist/`; run `npm run build` first). Excluded from CI, which runs
 * only boot: this spec drives the GPU/render canvas, and CI's headless Chromium
 * can't render WebGL (no GPU, SwiftShader gated off). See playwright.config.ts.
 */
import { test, expect, type Page } from "@playwright/test";
import { waitForRender } from "./_helpers";

/** Bottom opaque red; top paints the right half opaque blue (alpha = X >= .5). */
const TWO_COLOR_PROGRAM = `View(0, 0, 1, 1)
Material(
  Layer(1)
    .basecolor(255, 0, 0)
    .roughness(1.0),
  Layer(Floor((X() + 0.5)))
    .basecolor(0, 0, 255)
    .roughness(1.0)
)`;

async function dot(page: Page): Promise<string | null> {
  const cls = await page.locator("#status-dot").getAttribute("class");
  return cls?.match(/dot-(idle|busy|ok|error)/)?.[1] ?? null;
}

/** Count strongly red / strongly blue pixels in a PNG screenshot buffer. */
async function countColorFamilies(
  page: Page,
  png: Buffer,
): Promise<{ red: number; blue: number; total: number }> {
  return await page.evaluate(async (b64) => {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("screenshot decode failed"));
      img.src = `data:image/png;base64,${b64}`;
    });
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    if (!ctx) throw new Error("no 2d context");
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let red = 0;
    let blue = 0;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i];
      const b = d[i + 2];
      if (r > 100 && r > b + 40) red++;
      else if (b > 100 && b > r + 40) blue++;
    }
    return { red, blue, total: d.length / 4 };
  }, png.toString("base64"));
}

test("imported layers keep their distinct basecolors on screen", async ({
  page,
}) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));

  await page.goto("/viewer/");
  await expect.poll(() => dot(page), { timeout: 20_000 }).toBe("ok");
  // Wait for the first GPU frame before driving the UI: the material shader
  // compiles on the first scene.render(), and real clicks stall until that
  // frame is painted (see _helpers.ts). This spec runs locally only.
  await waitForRender(page);

  // Import via the ⋯ menu → Import material → paste → Apply.
  await page.locator("#btn-menu").click();
  await page.locator("#btn-import").click();
  const modal = page.locator(".io-modal:not([hidden])");
  await expect(modal).toBeVisible();
  await expect(modal.locator(".io-title")).toHaveText("Import material");
  await modal.locator("textarea.io-text").fill(TWO_COLOR_PROGRAM);
  await modal.getByRole("button", { name: "Apply" }).click();
  await expect(modal).toBeHidden();

  // Two layers in the panel, and the eval settles.
  const layers = page.locator("#layer-list > *");
  await expect
    .poll(async () => await layers.count(), { timeout: 5000 })
    .toBe(2);
  await expect.poll(() => dot(page), { timeout: 20_000 }).toBe("ok");

  // Wait for the recompiled 2-layer frame to paint, then count color families.
  // On CI's SwiftShader the imported material's shader compiles on the first
  // post-Apply scene.render() and can take many seconds; a fixed 500ms wait can
  // snapshot the canvas before the new frame lands (showing the prior material
  // or just the clear color), so poll until BOTH color families are present.
  // With the transformObjectKeys hoisting bug the top layer renders red too and
  // `blue` stays ~0 forever, so the poll exhausts its budget and the assertion
  // below fails with the diagnostic message.
  const deadline = Date.now() + 60_000;
  let red = 0;
  let blue = 0;
  let total = 0;
  let planePx = 0;
  for (;;) {
    const shot = await page.locator("#render-canvas").screenshot();
    ({ red, blue, total } = await countColorFamilies(page, shot));
    planePx = total * 0.02; // very conservative floor per color family
    if ((red > planePx && blue > planePx) || Date.now() >= deadline) break;
    await page.waitForTimeout(500);
  }
  expect(
    red,
    `left half should be red (got ${red} red-ish px of ${total})`,
  ).toBeGreaterThan(planePx);
  expect(
    blue,
    `right half should be blue (got ${blue} blue-ish px of ${total}) — ` +
      `a near-zero count means every layer took the bottom layer's color ` +
      `(the transformObjectKeys shared-literal bug) or the 2-layer frame had ` +
      `not yet painted`,
  ).toBeGreaterThan(planePx);

  expect(pageErrors).toEqual([]);
});
