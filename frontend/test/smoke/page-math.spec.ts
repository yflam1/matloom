/**
 * page-math.spec.ts — MathJax renders the paper page's math (Playwright).
 *
 * The page's math (\( ... \) in the abstract) is injected by the deferred app
 * bundle while MathJax arrives as an async CDN script. Chrome can execute the
 * warm-cached CDN script before deferred modules; MathJax then sees readyState
 * "interactive" and typesets immediately — before the abstract exists — which
 * left raw \( delimiters on screen in Chrome only (Safari and Firefox run the
 * async script during parsing and so wait for DOMContentLoaded, after the
 * bundle has rendered). src/page/main.ts closes both orderings by re-typesetting
 * once MathJax is up; the second test below forces the losing ordering by
 * delaying the app bundle past the CDN script.
 *
 * CPU-only (DOM assertions, no WebGL), so it runs in CI like boot.spec.ts.
 * Run via `npm run test:smoke` (needs a built dist/).
 */
import { test, expect, type Page } from "@playwright/test";

/** Inline \( ... \) spans in the abstract: 141, 21, 30, 20, 59.2%, 19.3%. */
const EXPECTED_MATH_COUNT = 6;

async function mathState(page: Page): Promise<{
  mathJaxLoaded: boolean;
  containers: number;
  rawDelims: number;
}> {
  return await page.evaluate(() => ({
    mathJaxLoaded: typeof window.MathJax?.typesetPromise === "function",
    containers: document.querySelectorAll("mjx-container").length,
    rawDelims: (
      document
        .querySelector("#abstract .content")
        ?.textContent?.match(/\\\(|\\\)/g) ?? []
    ).length,
  }));
}

async function expectAbstractTypeset(page: Page): Promise<void> {
  // MathJax must actually load: if the CDN is unreachable nothing typesets,
  // and the deployed page would show raw delimiters, so fail loudly.
  await expect
    .poll(() => mathState(page).then((s) => s.mathJaxLoaded), {
      timeout: 20_000,
      message: "MathJax API should load from the CDN",
    })
    .toBe(true);
  await expect
    .poll(() => mathState(page).then((s) => s.containers), {
      timeout: 10_000,
      message: "every inline math in the abstract should be typeset",
    })
    .toBe(EXPECTED_MATH_COUNT);
  // Typeset containers exist; now no raw delimiter may remain anywhere.
  const { rawDelims } = await mathState(page);
  expect(rawDelims, "no raw \\( delimiters should remain").toBe(0);
}

test("abstract math typesets when the bundle runs first (natural order)", async ({
  page,
}) => {
  await page.goto("/");
  await expectAbstractTypeset(page);
});

test("abstract math typesets when MathJax runs first (delayed bundle)", async ({
  page,
}) => {
  // Reproduce real Chrome: the warm-cached async CDN script executes before
  // the deferred bundle renders the abstract.
  await page.route("**/assets/*.js", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    await route.continue();
  });
  await page.goto("/");
  await expectAbstractTypeset(page);
});
