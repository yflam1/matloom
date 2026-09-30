/**
 * check-build-artifacts.mjs — a fast, browser-free post-build guard.
 *
 * Run after `vite build` (which also obfuscates). The site is ONE build with
 * TWO HTML entries: the paper project page (dist/index.html, served at the
 * GitHub Pages root) and the interactive viewer (dist/viewer/index.html,
 * served under /viewer/), sharing dist/assets/.
 *
 * It catches the class of regression fixed in commit 88dfd82: the obfuscator's
 * `stringArray`/base64 encoding moved Vite's `__VITE_WORKER_ASSET__<hash>__`
 * placeholder into its decoder array, so Vite's post-build literal text-replace
 * found nothing and the raw placeholder shipped — `new Worker(new
 * URL("./eval.worker.ts", ...))` then 404'd at runtime ("pool worker error")
 * and the deployed site was broken, while `npm run dev` (obfuscator is
 * build-only) stayed fine.
 *
 * Assertions, each must pass:
 *
 *  1. No leaked Vite placeholders anywhere in `dist/`. The three placeholder
 *     families Vite emits as plain literals and replaces post-build must never
 *     survive into the shipped bundle.
 *  2. The eval-worker chunk URL is present as a *literal* in the viewer entry
 *     chunk (assets/viewer-*.js) — i.e. Vite's `__VITE_WORKER_ASSET__<hash>__`
 *     → `eval.worker-<hash>.js` replacement actually happened. Exactly one
 *     worker chunk must be emitted.
 *  3. Both HTML entries and the hosting insurance files exist: index.html,
 *     viewer/index.html, .nojekyll, and all committed page figures.
 *  4. Every local asset each HTML file references resolves to a file in dist/
 *     (relative to that HTML's own directory — the viewer's ../assets/… links
 *     live or die here).
 *
 * Exits non-zero on any failure so it can gate CI and the Pages deploy.
 */
import { readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const FRONTEND_ROOT = fileURLToPath(new URL("..", import.meta.url));
const DIST = join(FRONTEND_ROOT, "dist");
const ASSETS = join(DIST, "assets");

const PLACEHOLDERS = [
  "__VITE_WORKER_ASSET__",
  "__VITE_ASSET__",
  "__VITE_PUBLIC_ASSET__",
];

const TEASER = [
  "wood-planks",
  "black-slate-flooring",
  "hammered-rusty-metal",
  "black-and-white-marble-checker-tile",
  "triangle-ceramic-tiles",
  "brown-and-white-checkered-cotton-fabric",
].map((name) => `figures/teaser/${name}.png`);

const REFINE = ["0", "1", "2", "3", "4", "5"].map(
  (round) => `figures/refine/r${round}.png`,
);

const EDITS = ["original", "blue-glaze", "higher-roughness", "wider-grout"].map(
  (name) => `figures/edits/${name}.png`,
);

const COMPARISON = [];
for (const prompt of [
  "acoustic-panels",
  "cracked-ice",
  "holiday-wrapping-paper",
]) {
  for (const method of ["ours", "stablematerials", "intrinsix", "matfuse"]) {
    COMPARISON.push(`figures/comparison/${prompt}-${method}.png`);
  }
}

const REQUIRED_FILES = [
  "index.html",
  "viewer/index.html",
  ".nojekyll",
  "figures/overview.png",
  ...TEASER,
  ...REFINE,
  ...EDITS,
  ...COMPARISON,
];

// Text bundle formats Vite emits; binary assets (fonts/images) are skipped.
const TEXT_EXT = new Set([".js", ".mjs", ".cjs", ".html", ".css"]);

let failures = 0;
const fail = (msg) => {
  console.error(`  FAIL: ${msg}`);
  failures++;
};
const ok = (msg) => console.log(`  PASS: ${msg}`);

async function exists(p) {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

/** Recursively collect text-bundle files under `dir`. */
async function collectTextFiles(dir) {
  const out = [];
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return out; // missing dir handled by the caller's existence check
  }
  for (const e of entries) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...(await collectTextFiles(p)));
    else if (TEXT_EXT.has(e.name.slice(e.name.lastIndexOf(".")))) out.push(p);
  }
  return out;
}

async function main() {
  console.log("Checking built artifacts in dist/…");

  if (!(await exists(DIST))) {
    fail(`dist/ not found — run \`vite build\` first.`);
    process.exit(1);
  }

  // --- Assertion 1: no leaked Vite placeholders ---------------------------
  const files = await collectTextFiles(DIST);
  if (!files.length) fail(`no text bundle files found under dist/.`);
  const leaked = [];
  for (const f of files) {
    const src = await readFile(f, "utf8");
    for (const ph of PLACEHOLDERS) {
      if (src.includes(ph)) leaked.push({ file: f, placeholder: ph });
    }
  }
  if (leaked.length) {
    for (const { file, placeholder } of leaked) {
      fail(`leaked placeholder \`${placeholder}\` in ${file}`);
    }
  } else {
    ok(`no leaked Vite placeholders (${files.length} file(s) scanned).`);
  }

  // --- Assertion 2: worker chunk URL is a literal in the viewer chunk -----
  const assetFiles = (
    await readdir(ASSETS, { withFileTypes: true }).catch(() => [])
  )
    .filter((e) => e.isFile())
    .map((e) => e.name);
  const workerChunks = assetFiles.filter((n) =>
    /^eval\.worker-.*\.js$/.test(n),
  );
  const viewerChunks = assetFiles.filter((n) => /^viewer-.*\.js$/.test(n));

  if (workerChunks.length !== 1) {
    fail(
      `expected exactly 1 eval.worker-*.js chunk, found ${workerChunks.length}` +
        (workerChunks.length ? ` (${workerChunks.join(", ")})` : "") +
        `. If workers were split into multiple chunks, update this check.`,
    );
  } else if (viewerChunks.length !== 1) {
    fail(
      `expected exactly 1 viewer-*.js entry chunk, found ${viewerChunks.length}` +
        (viewerChunks.length ? ` (${viewerChunks.join(", ")})` : "") +
        `.`,
    );
  } else {
    const viewerSrc = await readFile(join(ASSETS, viewerChunks[0]), "utf8");
    if (viewerSrc.includes(workerChunks[0])) {
      ok(
        `worker URL \`${workerChunks[0]}\` is a literal in ${viewerChunks[0]}.`,
      );
    } else {
      fail(
        `worker URL \`${workerChunks[0]}\` not found as a literal in ` +
          `${viewerChunks[0]} — Vite's __VITE_WORKER_ASSET__ replacement may ` +
          `have been skipped.`,
      );
    }
  }

  // --- Assertion 3: both entries + hosting insurance + figures exist ------
  const missing = [];
  for (const file of REQUIRED_FILES) {
    if (!(await exists(join(DIST, file)))) missing.push(file);
  }
  if (missing.length) {
    for (const file of missing)
      fail(`required file missing from dist/: ${file}`);
  } else {
    ok(`all ${REQUIRED_FILES.length} required files present`);
  }

  // --- Assertion 4: local asset refs in each HTML resolve -----------------
  for (const html of ["index.html", "viewer/index.html"]) {
    const htmlPath = join(DIST, html);
    if (!(await exists(htmlPath))) continue; // already failed in assertion 3
    const slash = html.lastIndexOf("/");
    const dir = slash === -1 ? DIST : join(DIST, html.slice(0, slash));
    const src = await readFile(htmlPath, "utf8");
    const refs = [...src.matchAll(/(?:src|href)="([^"]+)"/g)].map((m) => m[1]);
    let missingRefs = 0;
    for (const ref of refs) {
      if (/^(https?:|data:|#|mailto:)/.test(ref)) continue;
      const local = join(dir, ref.replace(/^\.\//, "").split(/[?#]/)[0]);
      if (!(await exists(local))) {
        fail(`${html} references missing local asset: ${ref}`);
        missingRefs++;
      }
    }
    if (missingRefs === 0) {
      ok(`all ${refs.length} ${html} asset references resolve`);
    }
  }

  // Soft reminder, not a gate: the arXiv button and BibTeX carry a placeholder
  // ID until the preprint is posted (the reference page shipped the same
  // placeholder), so a forgotten swap is surfaced here instead of failing CI.
  try {
    const contentSrc = await readFile(
      join(FRONTEND_ROOT, "src", "page", "content.ts"),
      "utf8",
    );
    if (contentSrc.includes("XXXX.XXXXX")) {
      console.log(
        "  WARN: arXiv placeholder (XXXX.XXXXX) still in src/page/content.ts — fill in the real ID once the preprint is posted (does not fail this check).",
      );
    }
  } catch {
    // Source not available; skip the reminder.
  }

  if (failures) {
    console.error(`\n${failures} build-artifact check(s) failed.`);
    process.exit(1);
  }
  console.log("\nAll build-artifact checks passed.");
}

await main();
