/**
 * vitest.config.ts — unit-test runner config for the frontend.
 *
 * Kept separate from vite.config.ts so the obfuscator/build pipeline never
 * touches the tests. The engine and expr-lang are pure computation (no DOM,
 * no Babylon), so the default "node" environment is all that is needed.
 */
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
    // The Playwright build-artifact smoke suite (test/smoke/**, *.spec.ts) runs
    // under its own config via `npm run test:smoke` — never vitest — so the
    // node-only unit suite stays browser-free. Excluded defensively too.
    exclude: ["test/smoke/**", "node_modules/**", "dist/**"],
    coverage: {
      provider: "v8",
      reportsDirectory: "coverage",
      include: ["src/engine/**", "src/expr-lang/**"],
    },
  },
});
