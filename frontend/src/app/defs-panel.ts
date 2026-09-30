/**
 * defs-panel.ts — the "Definitions" section: author the material-level
 * `Define(name, expr)` bindings that channels reference by bare name.
 *
 * Each definition is a card with a name field and a full expression editor
 * (the same completion / signature popups the channel editors use). Definitions
 * resolve in order — a later one may reference an earlier one — so validation
 * builds the environment top-to-bottom and reports name/expression errors per
 * card. The store's `defs` array is the single source of truth; editing here
 * reschedules an evaluation, which rebuilds the resolved environment.
 */
import { byId } from "./dom";
import { parseExpr, EXPR_REGISTRY } from "../expr-lang";
import type { Expression2D } from "../engine";
import type { MaterialDef, Store } from "./state";
import type { Evaluator } from "./eval";
import type { EditorHandle, ExprEditorUI } from "./expr-editor";

// Names that cannot be used for a definition: every callable/const in the
// language (a bare function name is a call error, never a reference) plus the
// boolean/None literals.
const RESERVED = new Set<string>([
  ...EXPR_REGISTRY.map((e) => e.name),
  "pi",
  "e",
  "True",
  "False",
  "None",
  "true",
  "false",
  "null",
]);

const VALID_NAME = /^[A-Za-z_]\w*$/;

interface DefCard {
  def: MaterialDef;
  expr: EditorHandle;
  setNameError(msg: string): void;
}

export class DefsPanel {
  private cards: DefCard[] = [];

  constructor(
    private readonly store: Store,
    private readonly evaluator: Evaluator,
    private readonly editorUI: ExprEditorUI,
  ) {}

  /** Wire the "+ Define" button and render the initial list. */
  init(): void {
    byId("btn-add-def").addEventListener("click", () => this.addDef());
    this.render();
  }

  private addDef(): void {
    // A valid, illustrative starting point so the new card renders immediately.
    this.store.defs.push({
      name: this.uniqueName(),
      src: "Fill(Rect(0.3, 0.3, 0.4, 0.4))",
    });
    this.render();
    this.evaluator.scheduleEval();
  }

  private removeDef(def: MaterialDef): void {
    this.store.defs = this.store.defs.filter((d) => d !== def);
    this.render();
    this.evaluator.scheduleEval();
  }

  private uniqueName(): string {
    const used = new Set(this.store.defs.map((d) => d.name));
    let i = 1;
    while (used.has(`shape${i}`)) i++;
    return `shape${i}`;
  }

  /** Rebuild the card list from the store and (re)validate. */
  render(): void {
    const list = byId("defs-list");
    list.replaceChildren();
    this.cards = this.store.defs.map((d) => this.buildDefEl(list, d));
    byId("defs-section").classList.toggle(
      "empty",
      this.store.defs.length === 0,
    );
    this.validate();
  }

  private buildDefEl(list: HTMLElement, def: MaterialDef): DefCard {
    const card = document.createElement("div");
    card.className = "dcard";

    const row = document.createElement("div");
    row.className = "dcard-row";

    const name = document.createElement("input");
    name.type = "text";
    name.className = "dname";
    name.value = def.name;
    name.spellcheck = false;
    name.autocomplete = "off";
    name.placeholder = "name";
    name.title = "Definition name — reference it by this name in any channel";
    name.addEventListener("input", () => {
      def.name = name.value.trim();
      this.validate();
      this.evaluator.scheduleEval(true);
    });

    const eq = document.createElement("span");
    eq.className = "deq";
    eq.textContent = "=";

    const del = document.createElement("button");
    del.type = "button";
    del.className = "ddel";
    del.textContent = "✕";
    del.title = "Delete definition";
    del.addEventListener("click", () => this.removeDef(def));

    row.append(name, eq, del);

    const nameErr = document.createElement("div");
    nameErr.className = "dname-err";
    nameErr.hidden = true;

    const edWrap = document.createElement("div");
    edWrap.className = "ceditor";
    const expr = this.editorUI.makeEditor(edWrap, def.src, (val) => {
      def.src = val;
      this.validate();
      this.evaluator.scheduleEval(true);
    });

    card.append(row, nameErr, edWrap);
    list.appendChild(card);

    return {
      def,
      expr,
      setNameError(msg: string) {
        nameErr.textContent = msg;
        nameErr.hidden = !msg;
        name.classList.toggle("err", msg !== "");
      },
    };
  }

  // Resolve definitions in order, surfacing name and expression errors. Mirrors
  // the env-building in material-io / eval so the panel's diagnostics match what
  // the evaluator will actually accept.
  private validate(): void {
    const env = new Map<string, Expression2D>();
    const seen = new Set<string>();
    for (const { def, expr, setNameError } of this.cards) {
      let nameErr = "";
      if (!def.name) nameErr = "name required";
      else if (!VALID_NAME.test(def.name))
        nameErr = "use letters, digits or _ (not starting with a digit)";
      else if (RESERVED.has(def.name))
        nameErr = `'${def.name}' is a reserved name`;
      else if (seen.has(def.name)) nameErr = "duplicate name";
      setNameError(nameErr);

      let parsed: Expression2D | null = null;
      try {
        parsed = parseExpr(def.src, env);
        expr.clearErr();
      } catch (e) {
        expr.setErr((e as Error).message);
      }

      if (!nameErr && parsed) {
        seen.add(def.name);
        env.set(def.name, parsed);
      }
    }
  }
}
