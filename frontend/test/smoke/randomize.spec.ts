/**
 * randomize.spec.ts — end-to-end check of the Randomize feature against the
 * built `dist/`. Drives the real DOM path that node unit tests can't reach:
 * the ⋯ menu item, the controls dialog, Regenerate, and Apply routing a
 * generated material through the same import pipeline (parse → store → eval).
 *
 * Part of the Playwright smoke suite — run via `npm run test:smoke` (consumes
 * an existing `dist/`; run `npm run build` first). Excluded from CI, which runs
 * only boot: this spec drives the GPU/render canvas, and CI's headless Chromium
 * can't render WebGL (no GPU, SwiftShader gated off). See playwright.config.ts.
 */
import { test, expect, type Page } from "@playwright/test";
import { waitForRender } from "./_helpers";

async function dot(page: Page): Promise<string | null> {
  const cls = await page.locator("#status-dot").getAttribute("class");
  return cls?.match(/dot-(idle|busy|ok|error)/)?.[1] ?? null;
}

test("randomize: menu → dialog → apply populates layers and evaluates", async ({
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

  // Open the ⋯ menu and click Randomize.
  await page.locator("#btn-menu").click();
  await page.locator("#btn-randomize").click();

  const modal = page.locator(".io-modal:not([hidden])");
  await expect(modal).toBeVisible();
  await expect(modal.locator(".io-title")).toHaveText("Randomize material");

  // The preview textarea should hold a generated program on open.
  const preview = modal.locator("textarea.io-text");
  await expect(preview).toHaveValue(/View\(/);
  await expect(preview).toHaveValue(/Material\(/);

  // Force exactly 2 layers, then regenerate.
  await modal.locator(".rnd-input").first().waitFor();
  await page.locator(".io-modal:not([hidden]) input").nth(0).fill("2"); // minLayers
  await page.locator(".io-modal:not([hidden]) input").nth(1).fill("2"); // maxLayers
  await modal.getByRole("button", { name: /Regenerate/ }).click();
  await expect(preview).toHaveValue(/Material\(/);

  // Apply → dialog closes, stack rebuilt, eval reaches ok.
  await modal.getByRole("button", { name: "Apply" }).click();
  await expect(modal).toBeHidden();

  const layers = page.locator("#layer-list > *");
  await expect
    .poll(async () => await layers.count(), { timeout: 5000 })
    .toBe(2);
  await expect.poll(() => dot(page), { timeout: 20_000 }).toBe("ok");

  expect(pageErrors).toEqual([]);
});
