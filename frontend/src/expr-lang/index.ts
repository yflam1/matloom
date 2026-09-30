/**
 * expr-lang — the channel expression language: a registry of functions/consts,
 * a Python-like parser (parseExpr), and caret-context helpers for the editor.
 */
export { parseExpr, EXPR_REGISTRY } from "./parser";
export { serializeExpr } from "./serialize";
export { callContextAt, tokenAt } from "./caret";
export type { CallContext, IdentToken } from "./caret";
export type { RegistryEntry, ParamSpec, ExprValue } from "./registry";
