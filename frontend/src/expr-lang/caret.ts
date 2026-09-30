/**
 * caret.ts — caret-context analysis for the editor UI (signature popup +
 * completion dropdown). These run on raw, possibly-incomplete text on every
 * keystroke, so they never throw.
 */
import { REGISTRY_MAP } from "./registry";

export interface CallContext {
  /** Function name owning the parentheses the caret sits in. */
  name: string;
  /** Zero-based index of the argument segment the caret is in. */
  argIndex: number;
  /** Keyword name if the current segment looks like `name=…`, else null. */
  activeName: string | null;
}

const isIdentStart = (c: string) => /[A-Za-z_]/.test(c);
const isIdentChar = (c: string) => /[A-Za-z0-9_]/.test(c);

/**
 * Which call's parentheses is the caret inside? Forward-scans up to `caret`,
 * maintaining a stack of open-paren frames; quotes are skipped wholesale so
 * commas/parens inside strings don't confuse the counter. Returns the innermost
 * enclosing known call, or null when the caret is not inside one.
 */
export function callContextAt(
  text: string,
  caret: number | null,
): CallContext | null {
  text = text || "";
  if (caret == null) caret = text.length;

  const stack: { name: string | null; argIndex: number; segStart: number }[] =
    [];
  let i = 0;
  while (i < caret) {
    const c = text[i];
    if (c === '"' || c === "'") {
      // Advance just past the matching close quote (or to caret/end).
      i++;
      while (i < text.length && text[i] !== c) i++;
      i++; // skip the closing quote (or step past end)
      continue;
    }
    if (c === "(") {
      // Identifier ending immediately before `i` (skipping spaces) names the
      // call; absent (e.g. a grouping paren) → null.
      let j = i - 1;
      while (j >= 0 && /\s/.test(text[j])) j--;
      const end = j + 1;
      while (j >= 0 && isIdentChar(text[j])) j--;
      const start = j + 1;
      let name: string | null = null;
      if (start < end && isIdentStart(text[start]))
        name = text.slice(start, end);
      stack.push({ name, argIndex: 0, segStart: i + 1 });
    } else if (c === ")") {
      if (stack.length) stack.pop();
    } else if (c === "," && stack.length) {
      const top = stack[stack.length - 1];
      top.argIndex += 1;
      top.segStart = i + 1;
    }
    i++;
  }

  if (!stack.length) return null;
  const top = stack[stack.length - 1];
  if (top.name == null || !REGISTRY_MAP.has(top.name)) return null;

  const segment = text.slice(top.segStart, caret);
  const m = segment.match(/^\s*([A-Za-z_]\w*)\s*=/);
  return {
    name: top.name,
    argIndex: top.argIndex,
    activeName: m ? m[1] : null,
  };
}

export interface IdentToken {
  text: string;
  start: number;
  end: number;
}

/**
 * The identifier token the caret is inside or adjacent to on its left, or null.
 * Used by the completion dropdown to replace a partially-typed name.
 */
export function tokenAt(text: string, caret: number | null): IdentToken | null {
  text = text || "";
  if (caret == null) caret = text.length;

  let start = caret;
  while (start > 0 && isIdentChar(text[start - 1])) start--;
  let end = caret;
  while (end < text.length && isIdentChar(text[end])) end++;
  if (start === end) return null;
  // Identifiers cannot start with a digit; if the run does, it's a number.
  if (!/[A-Za-z_]/.test(text[start])) return null;
  return { text: text.slice(start, end), start, end };
}
