/**
 * demo-embed.spec.ts — the paper page embeds the live viewer (Playwright).
 *
 * src/page/render.ts demoSection() places the /viewer/ app in a same-origin
 * iframe between the teaser strip and the abstract, immediately interactive
 * (no activation overlay). This spec pins that contract: the frame sits at
 * the required position in the page and the embedded viewer actually boots
 * (its #app reaches app-ready — the same boot boot.spec.ts drives on the
 * standalone page, proving the iframe URL, the worker spawn, and the first
 * evaluation all work in the embedded context).
 *
 * CPU-only (DOM assertions, no canvas pixels), so it runs in CI like
 * boot.spec.ts and page-math.spec.ts. Run via `npm run test:smoke` (needs a
 * built dist/).
 */
import { test, expect } from "@playwright/test";

test("the paper page embeds a booting viewer", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));

  await page.goto("/");

  // Placement: directly below the teaser section, directly above the abstract.
  const neighbors = await page.evaluate(() => {
    const demo = document.querySelector("#main-content .demo");
    return {
      prev: demo?.previousElementSibling?.className ?? "",
      next: demo?.nextElementSibling?.id ?? "",
    };
  });
  expect(neighbors.prev, "the demo sits right after the teaser").toContain(
    "teaser",
  );
  expect(neighbors.next, "the abstract follows the demo").toBe("abstract");

  // The embedded app boots inside the iframe: #app flips to app-ready only
  // after the full UI is wired and the default material is loaded
  // (src/app/main.ts).
  const app = page.frameLocator(".demo-frame > iframe").locator("#app");
  await expect(app, "embedded viewer reaches app-ready").toHaveClass(
    /app-ready/,
    { timeout: 30_000 },
  );

  expect(pageErrors, "no uncaught page errors (parent or iframe)").toEqual([]);
});
