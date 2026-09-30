/**
 * parser.ts — a small Python-like expression parser. Users write
 *   fBm(base_freq=2, seed=5),  Worley(combination="F2-F1"),
 *   Sin(X()),  Mul(X(), 2),  X()*2+1
 * with positional args allowed for required params. JavaScript object syntax
 * (`{key: …}`) is NOT supported. parseExpr() returns an Expression2D.
 */
import {
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Constant,
  Expression2D,
  NoiseExpression2D,
  Shape,
  Ref,
} from "../engine";
import {
  EXPR_REGISTRY,
  REGISTRY_MAP,
  type ExprValue,
  type RegistryEntry,
} from "./registry";

const LITERAL_MAP: Record<string, boolean | null> = {
  True: true,
  False: false,
  None: null,
  true: true,
  false: false,
  null: null,
};

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
// Tokens carry their [start, end) source offsets so the parser can record the
// span of each call (used to locate a noise node back in the field's text).
type Token = { start: number; end: number } & (
  | { type: "STRING"; value: string }
  | { type: "NUMBER"; value: number }
  | { type: "IDENT"; value: string }
  | { type: "OP"; value: string }
);

function tokenize(src: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];

    // Skip whitespace.
    if (/\s/.test(c)) {
      i++;
      continue;
    }

    const start = i;

    // String (single or double quote).
    if (c === "'" || c === '"') {
      let j = i + 1;
      while (j < src.length && src[j] !== c) j++;
      if (j >= src.length) throw new SyntaxError("unterminated string literal");
      tokens.push({
        type: "STRING",
        value: src.slice(i + 1, j),
        start,
        end: j + 1,
      });
      i = j + 1;
      continue;
    }

    // Number.
    if (/[0-9.]/.test(c)) {
      let j = i;
      while (j < src.length && /[0-9]/.test(src[j])) j++;
      if (j < src.length && src[j] === ".") {
        j++;
        while (j < src.length && /[0-9]/.test(src[j])) j++;
      }
      if (j < src.length && (src[j] === "e" || src[j] === "E")) {
        j++;
        if (j < src.length && (src[j] === "+" || src[j] === "-")) j++;
        while (j < src.length && /[0-9]/.test(src[j])) j++;
      }
      tokens.push({
        type: "NUMBER",
        value: parseFloat(src.slice(i, j)),
        start,
        end: j,
      });
      i = j;
      continue;
    }

    // Identifier or keyword.
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_]/.test(src[j])) j++;
      tokens.push({ type: "IDENT", value: src.slice(i, j), start, end: j });
      i = j;
      continue;
    }

    // Operators & punctuation (match ** before *).
    if (c === "*" && src[i + 1] === "*") {
      tokens.push({ type: "OP", value: "**", start, end: i + 2 });
      i += 2;
    } else if ("+-*/()=,".includes(c)) {
      tokens.push({ type: "OP", value: c, start, end: i + 1 });
      i++;
    } else {
      throw new SyntaxError(`unexpected character '${c}'`);
    }
  }
  return tokens;
}

// ---------------------------------------------------------------------------
// Parser + evaluator (recursive descent)
// ---------------------------------------------------------------------------
type Arg =
  | { kind: "kw"; key: string; value: ExprValue }
  | { kind: "pos"; value: ExprValue };

function evaluate(src: string, env?: Map<string, Expression2D>): ExprValue {
  const tokens = tokenize(src);
  let pos = 0;

  const peek = (): Token | null => (pos < tokens.length ? tokens[pos] : null);
  const advance = (): Token | null =>
    pos < tokens.length ? tokens[pos++] : null;
  const fail = (msg: string): never => {
    throw new SyntaxError(msg);
  };
  const expect = (op: string): Token => {
    const t = advance();
    if (!t || t.value !== op)
      fail(`expected '${op}' but found '${t ? t.value : "end of input"}'`);
    return t as Token;
  };

  function expr(): ExprValue {
    return addsub();
  }

  function addsub(): ExprValue {
    let left = muldiv();
    for (;;) {
      const t = peek();
      if (!t) break;
      if (t.value === "+") {
        advance();
        left = Add(left as any, muldiv() as any);
      } else if (t.value === "-") {
        advance();
        left = Sub(left as any, muldiv() as any);
      } else break;
    }
    return left;
  }

  function muldiv(): ExprValue {
    let left = unary();
    for (;;) {
      const t = peek();
      if (!t) break;
      if (t.value === "*") {
        advance();
        left = Mul(left as any, unary() as any);
      } else if (t.value === "/") {
        advance();
        left = Div(left as any, unary() as any);
      } else break;
    }
    return left;
  }

  function unary(): ExprValue {
    const t = peek();
    if (t && (t.value === "-" || t.value === "+")) {
      advance();
      const operand = power();
      return t.value === "-" ? Mul(-1, operand as any) : operand;
    }
    return power();
  }

  function power(): ExprValue {
    const left = atom();
    const t = peek();
    if (t && t.value === "**") {
      advance();
      return Pow(left as any, unary() as any); // right-assoc
    }
    return left;
  }

  function atom(): ExprValue {
    const t = peek();
    if (!t) return fail("expected expression but found end of input");

    if (t.type === "NUMBER") {
      advance();
      return t.value;
    }
    if (t.type === "STRING") {
      advance();
      return t.value;
    }
    if (t.type === "IDENT") {
      advance();
      const name = t.value;
      const next = peek();
      if (next && next.value === "(") {
        advance();
        const args = arglist();
        const close = expect(")");
        const result = callFn(name, args);
        // Tag noise nodes with their call's source span so the UI can echo the
        // exact call text and highlight it back in the field.
        if (result instanceof NoiseExpression2D) {
          result.sourceText = src.slice(t.start, close.end);
          result.sourceStart = t.start;
          result.sourceEnd = close.end;
        }
        return result;
      }
      // Bare identifier.
      if (name in LITERAL_MAP) return LITERAL_MAP[name];
      const entry = REGISTRY_MAP.get(name);
      if (entry) {
        if (entry.callForm === "value") return entry.build(undefined);
        return fail(`'${name}' is a function — call it as ${name}()`);
      }
      if (env && env.has(name)) return new Ref(name, env.get(name)!);
      return fail(`unknown name: ${name}`);
    }
    if (t.type === "OP" && t.value === "(") {
      advance();
      const inner = expr();
      expect(")");
      return inner;
    }
    return fail(`unexpected token '${t.value}'`);
  }

  // arglist: (arg (',' arg)*)?  — empty is OK for no-arg calls.
  function arglist(): Arg[] {
    const head = peek();
    if (!head || head.value === ")") return [];
    const args = [arg()];
    while (peek() && peek()!.value === ",") {
      advance();
      if (peek() && peek()!.value === ")")
        fail("trailing comma in argument list");
      args.push(arg());
    }
    return args;
  }

  // arg: keyword arg (`name=value`) or positional.
  function arg(): Arg {
    const t = peek();
    if (t && t.type === "IDENT") {
      const next = tokens[pos + 1];
      if (next && next.value === "=") {
        const key = (advance() as Token).value as string;
        advance(); // consume '='
        return { kind: "kw", key, value: expr() };
      }
    }
    return { kind: "pos", value: expr() };
  }

  function callFn(name: string, rawArgs: Arg[]): ExprValue {
    const entry = REGISTRY_MAP.get(name);
    if (!entry) return fail(`unknown function: ${name}`);
    if (entry.callForm === "value")
      return fail(`'${name}' is a constant; use it without parentheses`);

    const positional: ExprValue[] = [];
    const kw: Record<string, ExprValue> = {};
    for (const a of rawArgs) {
      if (a.kind === "kw") kw[a.key] = a.value;
      else positional.push(a.value);
    }

    if (entry.variadic) {
      if (Object.keys(kw).length > 0)
        fail(`keyword arguments not allowed for variadic function: ${name}`);
      return entry.build(positional);
    }

    return entry.build(resolveArgs(entry, positional, kw));
  }

  function resolveArgs(
    entry: RegistryEntry,
    positional: ExprValue[],
    kw: Record<string, ExprValue>,
  ): Record<string, ExprValue> {
    const params = entry.params ?? [];
    const args: Record<string, ExprValue> = {};
    const pnames = params.map((p) => p.name);
    const rest: ExprValue[] = [];

    for (let i = 0; i < positional.length; i++) {
      if (i < params.length) args[params[i].name] = positional[i];
      else if (entry.restParam) rest.push(positional[i]);
      else fail(`too many positional arguments for ${entry.name}`);
    }

    for (const k of Object.keys(kw)) {
      if (!pnames.includes(k))
        fail(`unknown parameter '${k}' for ${entry.name}`);
      if (k in args)
        fail(`duplicate value for parameter '${k}' in ${entry.name}`);
      args[k] = kw[k];
    }

    for (const p of params) {
      if (p.required && !(p.name in args))
        fail(`missing required argument '${p.name}' for ${entry.name}`);
    }

    if (entry.restParam)
      (args as Record<string, unknown>)[entry.restParam.name] = rest;

    return args;
  }

  const result = expr();
  if (pos < tokens.length) {
    fail(
      `unexpected trailing input: ${tokens
        .slice(pos)
        .map((t) => t.value)
        .join(" ")}`,
    );
  }
  return result;
}

function isExpression2D(v: ExprValue): v is Expression2D {
  return v instanceof Expression2D;
}

/**
 * Parse a channel expression. Mirrors the old Function-based contract:
 * numbers → Constant, Expression2D → Expression2D, anything else → error.
 * `env` maps definition names to expressions, so a bare name referencing a
 * `Define` resolves to a Ref.
 */
export function parseExpr(
  src: string,
  env?: Map<string, Expression2D>,
): Expression2D {
  const result = evaluate((src || "").trim(), env);
  if (typeof result === "number") return new Constant(result);
  if (isExpression2D(result)) return result;
  if (result instanceof Shape)
    throw new TypeError(
      "a shape is only valid inline inside Fill(...) or Stroke(...); it cannot be bound by Define or referenced by name",
    );
  if (result !== null && typeof result === "object" && "kind" in result)
    throw new TypeError(
      "LineTo(...)/CubicTo(...) are only valid inside Path(...)",
    );
  throw new TypeError(`expected an expression, got ${typeof result}`);
}

export { EXPR_REGISTRY };
