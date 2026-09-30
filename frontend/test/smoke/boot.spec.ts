/**
 * boot.spec.ts — the deploy-guarantee smoke test (Playwright).
 *
 * The only test in the repo that boots the *built* app and exercises the real
 * `new Worker(new URL("./eval.worker.ts", import.meta.url))` path (see
 * `src/eval/backend.ts`). Every other eval test injects a fake worker under
 * node, so this is what catches build-only regressions like commit 88dfd82 — a
 * leaked Vite placeholder shipping the raw `__VITE_WORKER_ASSET__<hash>__`
 * string, the worker script 404'ing, and the pool rejecting every eval with
 * `"pool worker error"` — all invisible to the node suite and to `vite build`'s
 * exit code.
 *
 * Run via `npm run test:smoke` (Playwright, not vitest). It consumes an
 * existing `dist/` — run `npm run build` first (or `npm run build:verify` for
 * the full gate).
 */
import { test, expect, type Page } from "@playwright/test";

/** Status-dot CSS classes set by `setStatus()` in `src/app/eval.ts`. */
type DotState = "idle" | "busy" | "ok" | "error";

async function dotState(page: Page): Promise<DotState | null> {
  const cls = await page.locator("#status-dot").getAttribute("class");
  if (!cls) return null;
  const m = cls.match(/dot-(idle|busy|ok|error)/);
  return (m?.[1] as DotState) ?? null;
}

test("app boots and the eval pool reaches dot-ok", async ({ page }) => {
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const workerResponses: { url: string; status: number }[] = [];

  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => pageErrors.push(e.message));
  page.on("response", (r) => {
    if (/\/eval\.worker-[^/]+\.js(\?.*)?$/.test(r.url())) {
      workerResponses.push({ url: r.url(), status: r.status() });
    }
  });

  await page.goto("/viewer/");

  // The boot path: createStore → ViewerManager → createEvaluator (default =
  // pool) → viewerManager.init (Babylon) → UI wiring → initSwitch → preload of
  // the default material (black slate flooring; falls back to one plain layer) →
  // scheduleEval → runEval → PoolBackend.evaluate.
  // A healthy first eval flips #status-dot to `dot-ok` (and shows the plane);
  // a worker 404 rejects with "pool worker error" and flips it to `dot-error`.

  // Poll the status dot until it settles to a terminal state, within budget.
  // `dot-busy` is transient (eval in flight); `dot-ok`/`dot-error` are sticky.
  const deadline = Date.now() + 20_000;
  let state = (await dotState(page)) as DotState | null;
  while (state !== "ok" && state !== "error" && Date.now() < deadline) {
    await page.waitForTimeout(150);
    state = (await dotState(page)) as DotState | null;
  }

  // Surface diagnostics if the boot failed.
  if (state !== "ok") {
    const detail = [
      `final status-dot state: ${state ?? "<none>"}`,
      pageErrors.length
        ? `page errors:\n  - ${pageErrors.join("\n  - ")}`
        : "page errors: (none)",
      consoleErrors.length
        ? `console errors:\n  - ${consoleErrors.join("\n  - ")}`
        : "console errors: (none)",
      workerResponses.length
        ? `worker responses:\n  - ${workerResponses.map((r) => `${r.status} ${r.url}`).join("\n  - ")}`
        : "worker responses: (none — pool never spawned a worker)",
    ].join("\n");
    test.info().attach("boot-diagnostics.txt", {
      body: detail,
      contentType: "text/plain",
    });
  }

  // --- Assertions ----------------------------------------------------------
  expect(state, "status dot should reach dot-ok").toBe("ok");

  expect(pageErrors, "no uncaught page errors during boot").toEqual([]);

  const poolErrors = consoleErrors.filter((t) =>
    t.includes("pool worker error"),
  );
  expect(poolErrors, "no pool worker error").toEqual([]);

  expect(
    workerResponses.length,
    "the pool should have requested the eval worker chunk",
  ).toBeGreaterThan(0);
  for (const r of workerResponses) {
    expect(
      r.status,
      `eval worker chunk should load with 200 (got ${r.status} for ${r.url})`,
    ).toBe(200);
  }
});
