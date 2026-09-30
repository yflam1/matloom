/**
 * dom.ts — tiny DOM helpers shared across the UI modules.
 */

/** Look up an element by id, throwing if it is missing (all ids are static). */
export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`element #${id} not found`);
  return el as T;
}
