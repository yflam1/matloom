/**
 * expr-editor.ts — channel input editors plus the two body-level popups that
 * assist them: a completion dropdown and a signature-help box.
 *
 * Both popups are singletons on document.body (position: fixed) — NOT inside
 * the overflow-hidden panel — so the panel can never clip them. They key off
 * whichever input currently has focus (`activeInput`). Inputs are rebuilt on
 * every layer re-render; the singletons survive because they only reference the
 * *focused* input, and an input's blur handler hides them so a destroyed input
 * can never strand a popup. A single ExprEditorUI instance owns all of this.
 */
import {
  EXPR_REGISTRY,
  callContextAt,
  tokenAt,
  type ParamSpec,
  type RegistryEntry,
  type ExprValue,
} from "../expr-lang";

/** One auto-seeded noise node found in a field's expression. */
export interface NoiseChip {
  /** Call snippet identifying the node, e.g. "fBm(base_freq=2)". */
  label: string;
  /** The revealed seed value. */
  seed: number;
  /** [start, end) of the call in the field's text, for hover-highlighting. */
  range?: [number, number];
  /** Reroll just this node (and re-render). */
  reroll: () => void;
}

/** Per-editor handle used to surface parse errors on the field. */
export interface EditorHandle {
  clearErr(): void;
  setErr(msg: string): void;
  /** Show one chip per auto-seeded noise node (reveal seed + reroll); [] hides. */
  setNoise(chips: NoiseChip[]): void;
}

// Channel inputs carry a back-reference to their change handler so the dropdown
// can re-fire onChange after programmatically inserting a completion.
type ChannelInput = HTMLInputElement & {
  _exprOnChange?: (val: string) => void;
};

export class ExprEditorUI {
  private menuEl!: HTMLDivElement; // completion dropdown
  private sigEl!: HTMLDivElement; // signature help
  private activeInput: ChannelInput | null = null;
  private menuOpen = false;
  private menuIndex = -1; // highlighted item; -1 = nothing highlighted

  /** Build the two body-level singletons. Call once, after DOM is ready. */
  init(): void {
    this.menuEl = document.createElement("div");
    this.menuEl.className = "expr-menu";
    this.menuEl.style.display = "none";
    // One item per registry entry, in registry order (name + dimmed summary).
    EXPR_REGISTRY.forEach((entry, i) => {
      const item = document.createElement("div");
      item.className = "expr-menu-item";
      const name = document.createElement("span");
      name.className = "name";
      name.textContent = entry.name;
      const summary = document.createElement("span");
      summary.className = "summary";
      summary.textContent = entry.summary || "";
      item.append(name, summary);
      // Use mousedown (fires before the input's blur) + preventDefault so the
      // click keeps focus in the input; accepting an item is identical to Enter.
      item.addEventListener("mousedown", (e) => {
        e.preventDefault();
        this.menuIndex = i;
        this.acceptEntry();
      });
      this.menuEl.appendChild(item);
    });

    this.sigEl = document.createElement("div");
    this.sigEl.className = "expr-sig";
    this.sigEl.style.display = "none";

    document.body.append(this.menuEl, this.sigEl);

    // Moving the caret (arrow keys, clicks) updates the active-param highlight.
    document.addEventListener("selectionchange", () => {
      if (this.activeInput) this.refreshSig(this.activeInput);
    });
  }

  /** Create a channel editor in `container`, wired to the popups. */
  makeEditor(
    container: HTMLElement,
    initVal: string,
    onChange: (val: string) => void,
  ): EditorHandle {
    const inp = document.createElement("input") as ChannelInput;
    inp.type = "text";
    inp.value = initVal;
    inp.spellcheck = false;
    inp.autocomplete = "off";
    inp.autocapitalize = "off";
    inp.setAttribute("autocorrect", "off");
    inp._exprOnChange = onChange; // used by acceptEntry() after insertion

    inp.addEventListener("input", () => {
      onChange(inp.value);
      this.refreshSig(inp);
    });
    inp.addEventListener("focus", () => {
      this.activeInput = inp;
      this.refreshSig(inp);
    });
    // Esc is the only *keyboard* way to close the dropdown (handled in keydown).
    // But a body-level popup must also hide on blur (and on accepting an item);
    // otherwise clicking elsewhere would strand a dropdown over the page, and a
    // re-render could destroy the input beneath it.
    inp.addEventListener("blur", () => {
      this.closeMenu();
      this.hideSig();
      if (this.activeInput === inp) this.activeInput = null;
    });
    // Caret moves not covered by selectionchange in every browser.
    inp.addEventListener("keyup", () => this.refreshSig(inp));
    inp.addEventListener("click", () => this.refreshSig(inp));
    inp.addEventListener("keydown", (e) => this.onKeydown(e, inp));

    container.appendChild(inp);
    let errDiv: HTMLDivElement | null = null;
    let noiseDiv: HTMLDivElement | null = null;
    return {
      clearErr() {
        container.classList.remove("err");
        errDiv?.remove();
        errDiv = null;
      },
      setErr(msg: string) {
        container.classList.add("err");
        if (!errDiv) {
          errDiv = document.createElement("div");
          errDiv.className = "cerr";
          inp.after(errDiv);
        }
        errDiv.textContent = msg;
      },
      setNoise(chips) {
        if (!chips.length) {
          noiseDiv?.remove();
          noiseDiv = null;
          return;
        }
        if (!noiseDiv) {
          noiseDiv = document.createElement("div");
          noiseDiv.className = "cnoise";
          // Sits below the input (and below any error strip).
          (errDiv ?? inp).after(noiseDiv);
        }
        noiseDiv.replaceChildren(...chips.map((c) => makeNoiseChip(c, inp)));
      },
    };
  }

  private onKeydown(e: KeyboardEvent, inp: ChannelInput): void {
    if (this.handleBrackets(e, inp)) return;
    if (e.key === "Tab") {
      // Cycle focus through this layer's property editors instead of the
      // default tab order. Shift+Tab walks backwards; both wrap around.
      // (Leaving the input blurs it, which closes any open menu/signature.)
      e.preventDefault();
      const card = inp.closest(".lcard");
      if (!card) return;
      // Skip editors hidden by a collapsed group (offsetParent is null for
      // display:none elements, which cannot take focus).
      const editors = [
        ...card.querySelectorAll<HTMLInputElement>(".ceditor input"),
      ].filter((el) => el === inp || el.offsetParent !== null);
      const i = editors.indexOf(inp);
      if (i === -1 || editors.length < 2) return;
      const next =
        editors[(i + (e.shiftKey ? -1 : 1) + editors.length) % editors.length];
      next.focus();
      next.select();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!this.menuOpen) this.openMenu(inp);
      else this.moveMenu(+1); // wraps at the end
    } else if (e.key === "ArrowUp") {
      // Only hijack ArrowUp while the menu is open; otherwise let the caret move.
      if (this.menuOpen) {
        e.preventDefault();
        this.moveMenu(-1); // wraps at the start
      }
    } else if (e.key === "Enter") {
      if (this.menuOpen) {
        e.preventDefault();
        if (this.menuIndex >= 0) this.acceptEntry();
        else this.closeMenu();
      }
    } else if (e.key === "Escape") {
      if (this.menuOpen) {
        e.preventDefault();
        this.closeMenu(); // keyboard close path (does not blur)
      }
    }
  }

  // Auto-close parentheses to match how a code editor behaves:
  //   • typing "(" inserts a matching ")" (wrapping the selection, if any) and
  //     leaves the caret inside;
  //   • typing ")" when the caret already sits just before a ")" steps over it
  //     instead of inserting a second one;
  //   • Backspace inside an empty "()" deletes both halves.
  // Returns true when it handled the key (so onKeydown stops). A programmatic
  // value change does not fire the input event, so onChange is re-fired here.
  private handleBrackets(e: KeyboardEvent, inp: ChannelInput): boolean {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    const start = inp.selectionStart ?? inp.value.length;
    const end = inp.selectionEnd ?? start;
    const val = inp.value;

    if (e.key === "(") {
      e.preventDefault();
      const inner = val.slice(start, end);
      inp.value = val.slice(0, start) + "(" + inner + ")" + val.slice(end);
      inp.setSelectionRange(start + 1, start + 1 + inner.length);
      this.closeMenu();
      inp._exprOnChange?.(inp.value);
      this.refreshSig(inp);
      return true;
    }
    if (e.key === ")" && start === end && val[start] === ")") {
      e.preventDefault();
      inp.setSelectionRange(start + 1, start + 1);
      this.refreshSig(inp);
      return true;
    }
    if (
      e.key === "Backspace" &&
      start === end &&
      val[start - 1] === "(" &&
      val[start] === ")"
    ) {
      e.preventDefault();
      inp.value = val.slice(0, start - 1) + val.slice(start + 1);
      inp.setSelectionRange(start - 1, start - 1);
      inp._exprOnChange?.(inp.value);
      this.refreshSig(inp);
      return true;
    }
    return false;
  }

  // --- completion dropdown ---------------------------------------------------
  private openMenu(inp: ChannelInput): void {
    this.menuOpen = true;
    this.menuIndex = -1;
    this.positionMenu(inp);
    this.highlightMenu();
    this.menuEl.style.display = "block";
    this.hideSig(); // never show the menu and the signature at the same time
  }

  private closeMenu(): void {
    this.menuOpen = false;
    this.menuIndex = -1;
    if (this.menuEl) this.menuEl.style.display = "none";
  }

  private positionMenu(inp: ChannelInput): void {
    const r = inp.getBoundingClientRect();
    this.menuEl.style.left = r.left + "px";
    this.menuEl.style.top = r.bottom + 2 + "px";
    this.menuEl.style.minWidth = r.width + "px";
  }

  private highlightMenu(): void {
    const items = this.menuEl.children;
    for (let i = 0; i < items.length; i++)
      items[i].classList.toggle("active", i === this.menuIndex);
    if (this.menuIndex >= 0 && items[this.menuIndex])
      items[this.menuIndex].scrollIntoView({ block: "nearest" });
  }

  private moveMenu(delta: number): void {
    const n = EXPR_REGISTRY.length;
    if (delta > 0)
      this.menuIndex = this.menuIndex < 0 ? 0 : (this.menuIndex + 1) % n;
    else this.menuIndex = this.menuIndex <= 0 ? n - 1 : this.menuIndex - 1;
    this.highlightMenu();
  }

  // Accept the highlighted entry into `activeInput`: insert `name()` (caret
  // between the parens) for calls, or bare `name` (caret after) for value consts.
  private acceptEntry(): void {
    const inp = this.activeInput;
    if (!inp || this.menuIndex < 0) {
      this.closeMenu();
      return;
    }
    const entry = EXPR_REGISTRY[this.menuIndex];
    const caret = inp.selectionStart ?? inp.value.length;
    const tok = tokenAt(inp.value, caret); // partially-typed name to replace
    const start = tok ? tok.start : caret;
    const end = tok ? tok.end : caret;

    let insert: string;
    let caretPos: number;
    if (entry.callForm === "call") {
      insert = entry.name + "()";
      caretPos = start + entry.name.length + 1; // between the parentheses
    } else {
      insert = entry.name;
      caretPos = start + entry.name.length; // after the value
    }
    inp.value = inp.value.slice(0, start) + insert + inp.value.slice(end);
    inp.setSelectionRange(caretPos, caretPos);
    this.closeMenu();
    inp._exprOnChange?.(inp.value);
    this.refreshSig(inp); // for calls the caret is now inside the parens → show sig
  }

  // --- signature popup -------------------------------------------------------
  private hideSig(): void {
    if (this.sigEl) this.sigEl.style.display = "none";
  }

  private refreshSig(inp: ChannelInput): void {
    if (!this.sigEl || inp !== this.activeInput) return;
    if (this.menuOpen) {
      this.hideSig(); // avoid overlapping the dropdown
      return;
    }
    const ctx = callContextAt(inp.value, inp.selectionStart);
    if (!ctx) {
      this.hideSig();
      return;
    }
    const entry = EXPR_REGISTRY.find((e) => e.name === ctx.name);
    if (!entry) {
      this.hideSig();
      return;
    }
    const params = sigParams(entry);

    // Active param: keyword match wins; else positional argIndex (clamped); for
    // variadic the single var slot is always active.
    let activeIdx = -1;
    if (ctx.activeName)
      activeIdx = params.findIndex((p) => p.name === ctx.activeName);
    if (activeIdx < 0 && params.length)
      activeIdx = entry.variadic
        ? 0
        : Math.min(ctx.argIndex, params.length - 1);

    this.renderSig(entry, params, activeIdx);
    this.sigEl.style.display = "block";
    this.positionSig(inp);
  }

  private renderSig(
    entry: RegistryEntry,
    params: ParamSpec[],
    activeIdx: number,
  ): void {
    this.sigEl.replaceChildren();

    const line = document.createElement("div");
    line.className = "expr-sig-params";
    line.appendChild(document.createTextNode(entry.name + "("));
    params.forEach((p, i) => {
      if (i > 0) line.appendChild(document.createTextNode(", "));
      const span = document.createElement("span");
      span.className = "expr-sig-param" + (i === activeIdx ? " active" : "");
      let txt = p.name + ": " + p.type;
      if (!p.required && p.default !== undefined)
        txt += " = " + formatDefault(p.default);
      if (p.variadic) txt += ", …";
      span.textContent = txt;
      line.appendChild(span);
    });
    line.appendChild(document.createTextNode(")"));
    this.sigEl.appendChild(line);

    const active = params[activeIdx];
    if (active) {
      const doc = document.createElement("div");
      doc.className = "expr-sig-doc";
      let txt =
        active.name + " (" + (active.required ? "required" : "optional");
      if (!active.required && active.default !== undefined)
        txt += ", default " + formatDefault(active.default);
      txt += ")";
      if (active.doc) txt += " — " + active.doc;
      doc.textContent = txt;
      this.sigEl.appendChild(doc);
    }
  }

  // Prefer placing the popup above the input; fall back to below if there's no
  // room. Measured after display:block so offsetHeight reflects real content.
  private positionSig(inp: ChannelInput): void {
    const r = inp.getBoundingClientRect();
    const h = this.sigEl.offsetHeight;
    const top = r.top - h - 4 >= 0 ? r.top - h - 4 : r.bottom + 4;
    this.sigEl.style.left = r.left + "px";
    this.sigEl.style.top = top + "px";
  }
}

// Build a noise chip: "call  seed  ⟳". The label echoes the node's exact call
// text so it's clear which noise it is; hovering selects that call back in the
// input. The seed is click-to-copy (pin a good one as `seed=N`); ⟳ rerolls it.
function makeNoiseChip(
  chip: NoiseChip,
  inp: HTMLInputElement,
): HTMLSpanElement {
  const el = document.createElement("span");
  el.className = "cnoise-chip";
  el.title = `${chip.label} — seed ${chip.seed}`;

  const label = document.createElement("span");
  label.className = "cnoise-call";
  label.textContent = chip.label;

  const seed = document.createElement("button");
  seed.type = "button";
  seed.className = "cnoise-seed";
  seed.textContent = String(chip.seed);
  seed.title = "Click to copy seed (paste as seed=… to keep it)";
  seed.addEventListener("mousedown", (e) => e.preventDefault()); // keep input focus
  seed.addEventListener("click", () => {
    if (!navigator.clipboard) return;
    void navigator.clipboard.writeText(String(chip.seed));
    seed.classList.add("copied");
    setTimeout(() => seed.classList.remove("copied"), 600);
  });

  const roll = document.createElement("button");
  roll.type = "button";
  roll.className = "cnoise-roll";
  roll.textContent = "⟳";
  roll.title = "Regenerate this seed";
  roll.addEventListener("mousedown", (e) => e.preventDefault());
  roll.addEventListener("click", chip.reroll);

  el.append(label, seed, roll);

  // Hover → highlight this noise's call in the input so it's obvious which chip
  // controls which noise. The call is shown by selecting it, but a text
  // selection only paints while the input is focused — so focus the input on
  // hover even if it wasn't (e.g. focus was elsewhere, or nowhere), and on leave
  // put focus and the caret back exactly where they were.
  if (chip.range) {
    const [s, e] = chip.range;
    let restore: (() => void) | null = null;
    el.addEventListener("mouseenter", () => {
      const prev = activeTextField();
      inp.focus({ preventScroll: true });
      inp.setSelectionRange(s, e);
      inp.classList.add("locating");
      restore = () => {
        inp.classList.remove("locating");
        if (prev && prev.el !== inp) {
          prev.el.focus({ preventScroll: true });
          prev.el.setSelectionRange(prev.start, prev.end);
        } else if (prev) {
          inp.setSelectionRange(prev.start, prev.end);
        } else {
          inp.blur();
        }
      };
    });
    el.addEventListener("mouseleave", () => {
      restore?.();
      restore = null;
    });
  }

  return el;
}

// Snapshot the currently focused text input and its caret, so a transient focus
// change (e.g. lighting up a noise on chip hover) can be undone afterwards.
function activeTextField(): {
  el: HTMLInputElement;
  start: number | null;
  end: number | null;
} | null {
  const a = document.activeElement;
  if (a instanceof HTMLInputElement && a.type === "text")
    return { el: a, start: a.selectionStart, end: a.selectionEnd };
  return null;
}

// Normalize a registry entry's params for display. A trailing rest parameter
// (Path's segments) is shown as a variadic slot after the fixed params, so the
// signature makes clear that more arguments follow. Variadic entries (min/max)
// carry no `params` array, so synthesize a single var slot to render.
function sigParams(entry: RegistryEntry): ParamSpec[] {
  const base = entry.params ?? [];
  if (entry.restParam) return [...base, { ...entry.restParam, variadic: true }];
  if (base.length) return base;
  if (entry.variadic)
    return [
      {
        name: "args",
        type: "float",
        required: true,
        doc: "One or more numeric values to combine.",
        variadic: true,
      },
    ];
  return [];
}

function formatDefault(v: ExprValue): string {
  if (v === null) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (typeof v === "string") return JSON.stringify(v);
  return String(v);
}
