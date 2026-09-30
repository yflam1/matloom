/**
 * io-controls.ts — the Import / Export toolbar buttons.
 *
 *   Export → opens a modal showing the (pretty-printed) material string in a
 *            read-only textarea, with Copy and Download buttons. The download
 *            file name is timestamped (material_YYYYMMDD-HHMMSS.txt) so repeated
 *            downloads never collide.
 *   Import → opens a modal with an editable textarea (paste a material string)
 *            and a "Load file…" button (read a .txt into the textarea); Apply
 *            rebuilds the layer stack, replacing whatever is currently there.
 *
 * A bad import never clobbers the current stack: the string is fully parsed and
 * validated before anything is replaced, and any error is shown in the modal.
 */
import { byId } from "./dom";
import { serializeMaterial, parseImportedLayers } from "./material-io";
import { generateMaterial, type GeneratorConfig } from "../generate";
import type { Evaluator } from "./eval";
import type { LayerPanel } from "./layer-panel";
import type { DefsPanel } from "./defs-panel";
import type { ViewerControls } from "./viewer-controls";
import type { Store } from "./state";

export class IoControls {
  // Import modal.
  private importModal!: HTMLDivElement;
  private importText!: HTMLTextAreaElement;
  private importFile!: HTMLInputElement;
  private importErr!: HTMLDivElement;
  // Export modal.
  private exportModal!: HTMLDivElement;
  private exportText!: HTMLTextAreaElement;
  private copyBtn!: HTMLButtonElement;
  private copyResetTimer: ReturnType<typeof setTimeout> | undefined;
  // Randomize modal.
  private rndModal!: HTMLDivElement;
  private rndText!: HTMLTextAreaElement;
  private rndErr!: HTMLDivElement;
  private rndInputs!: Record<string, HTMLInputElement>;

  constructor(
    private readonly store: Store,
    private readonly layerPanel: LayerPanel,
    private readonly defsPanel: DefsPanel,
    private readonly evaluator: Evaluator,
    private readonly viewerControls: ViewerControls,
  ) {}

  init(): void {
    this.buildImportModal();
    this.buildExportModal();
    this.buildRandomizeModal();
    byId("btn-export").addEventListener("click", () => this.openExport());
    byId("btn-import").addEventListener("click", () => this.openImport());
    byId("btn-randomize").addEventListener("click", () => this.openRandomize());
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      if (!this.exportModal.hidden) this.closeExport();
      else if (!this.importModal.hidden) this.closeImport();
      else if (!this.rndModal.hidden) this.closeRandomize();
    });
  }

  // — Export —

  private openExport(): void {
    let text: string;
    try {
      text = serializeMaterial(this.store);
    } catch (e) {
      window.alert(`Export failed: ${(e as Error).message}`);
      return;
    }
    this.exportText.value = text;
    this.resetCopyLabel();
    this.exportModal.hidden = false;
    this.exportText.focus();
    this.exportText.setSelectionRange(0, 0);
  }

  private closeExport(): void {
    this.exportModal.hidden = true;
  }

  private async copyExport(): Promise<void> {
    const text = this.exportText.value;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard API unavailable (e.g. insecure context): fall back to a
      // manual selection so the user can press Ctrl/Cmd+C themselves.
      this.exportText.focus();
      this.exportText.select();
      this.copyBtn.textContent = "Press Ctrl+C";
      return;
    }
    clearTimeout(this.copyResetTimer);
    this.copyBtn.textContent = "Copied!";
    this.copyResetTimer = setTimeout(() => this.resetCopyLabel(), 1200);
  }

  private resetCopyLabel(): void {
    clearTimeout(this.copyResetTimer);
    this.copyBtn.textContent = "Copy";
  }

  private downloadExport(): void {
    const blob = new Blob([this.exportText.value], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `material_${timestamp()}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // — Import —

  private openImport(): void {
    this.importErr.textContent = "";
    this.importModal.hidden = false;
    this.importText.focus();
  }

  private closeImport(): void {
    this.importModal.hidden = true;
    this.importFile.value = "";
  }

  private applyImport(): void {
    const err = this.applyMaterialText(this.importText.value);
    if (err) {
      this.importErr.textContent = err;
      return;
    }
    this.closeImport();
  }

  /**
   * Parse a material string and replace the current stack with it. Shared by
   * Import, Randomize, and the boot-time preload of the default material
   * (main.ts). Returns an error message (without mutating anything) on
   * failure, or null on success — a bad string never clobbers the stack.
   */
  applyMaterialText(text: string): string | null {
    const trimmed = text.trim();
    if (!trimmed) return "Nothing to apply.";
    let imported;
    try {
      imported = parseImportedLayers(trimmed);
    } catch (e) {
      return (e as Error).message;
    }
    this.store.layers = imported.layers;
    this.store.defs = imported.defs;
    this.layerPanel.render();
    this.defsPanel.render();
    if (imported.view) {
      // Apply the imported region and mark the view authored (so export emits a
      // `View(...)` line), then re-sync the inputs/viewer/eval.
      Object.assign(this.store.settings, imported.view);
      this.store.settings.viewSet = true;
      this.viewerControls.syncFromSettings();
    } else {
      // The imported material has no `View(...)` line, so the new document has
      // no authored view: drop the flag so export omits `View(...)`.
      this.store.settings.viewSet = false;
      this.evaluator.scheduleEval();
    }
    return null;
  }

  // — Randomize —

  private openRandomize(): void {
    this.rndErr.textContent = "";
    this.regenerate(); // a fresh roll every time the dialog opens
    this.rndModal.hidden = false;
    this.rndText.focus();
  }

  private closeRandomize(): void {
    this.rndModal.hidden = true;
  }

  // Read the form, generate a material, and show its DSL in the preview box.
  // An empty seed rolls a fresh random material; a set seed is reproducible.
  private regenerate(): void {
    this.rndErr.textContent = "";
    let config: Partial<GeneratorConfig>;
    let seed: number | undefined;
    try {
      ({ config, seed } = this.readRandomizeForm());
    } catch (e) {
      this.rndErr.textContent = (e as Error).message;
      return;
    }
    try {
      this.rndText.value = generateMaterial(config, {
        seed,
        asString: true,
      }) as string;
    } catch (e) {
      // Invalid config (e.g. min > max) surfaces here.
      this.rndErr.textContent = (e as Error).message;
    }
  }

  private applyRandomize(): void {
    const err = this.applyMaterialText(this.rndText.value);
    if (err) {
      this.rndErr.textContent = err;
      return;
    }
    this.closeRandomize();
  }

  // Pull a GeneratorConfig (and optional seed) out of the form inputs. Throws on
  // a non-numeric required field; the generator itself validates the values.
  private readRandomizeForm(): {
    config: Partial<GeneratorConfig>;
    seed: number | undefined;
  } {
    const num = (key: string): number => {
      const v = Number(this.rndInputs[key].value);
      if (!Number.isFinite(v)) throw new Error(`${key} must be a number`);
      return v;
    };
    const config: Partial<GeneratorConfig> = {
      minLayers: num("minLayers"),
      maxLayers: num("maxLayers"),
      minDefs: num("minDefs"),
      maxDefs: num("maxDefs"),
      maxDepth: num("maxDepth"),
      leafBias: num("leafBias"),
      opaqueBase: this.rndInputs.opaqueBase.checked,
      randomView: this.rndInputs.randomView.checked,
      realLo: num("realLo"),
      realHi: num("realHi"),
      posRealMin: num("posRealMin"),
      posRealMax: num("posRealMax"),
      nonNegRealMax: num("nonNegRealMax"),
      maxOctaves: num("maxOctaves"),
      maxPathSegments: num("maxPathSegments"),
    };
    const seedRaw = this.rndInputs.seed.value.trim();
    let seed: number | undefined;
    if (seedRaw !== "") {
      const s = Number(seedRaw);
      if (!Number.isFinite(s))
        throw new Error("seed must be a number (or blank)");
      seed = s;
    }
    return { config, seed };
  }

  // — Modal construction —

  private buildRandomizeModal(): void {
    const { modal, dialog, textarea } = makeModalShell("Randomize material");
    textarea.spellcheck = false;

    // Build the controls form and slot it above the DSL preview textarea.
    const inputs: Record<string, HTMLInputElement> = {};
    const form = document.createElement("div");
    form.className = "rnd-form";
    const addGroup = (label: string, fields: RndField[]): void => {
      const group = document.createElement("div");
      group.className = "rnd-group";
      const title = document.createElement("div");
      title.className = "rnd-group-title";
      title.textContent = label;
      group.appendChild(title);
      const grid = document.createElement("div");
      grid.className = "rnd-grid";
      for (const f of fields) {
        const { field, input } = makeRndField(f);
        inputs[f.key] = input;
        grid.appendChild(field);
      }
      group.appendChild(grid);
      form.appendChild(group);
    };

    addGroup("Structure", [
      { key: "minLayers", label: "Min layers", type: "int", value: 1, min: 1 },
      { key: "maxLayers", label: "Max layers", type: "int", value: 3, min: 1 },
      { key: "minDefs", label: "Min defines", type: "int", value: 0, min: 0 },
      { key: "maxDefs", label: "Max defines", type: "int", value: 2, min: 0 },
    ]);
    addGroup("Expressions", [
      { key: "maxDepth", label: "Max depth", type: "int", value: 6, min: 0 },
      {
        key: "leafBias",
        label: "Leaf bias",
        type: "float",
        value: 0.5,
        min: 0,
        max: 1,
        step: 0.05,
      },
      {
        key: "maxOctaves",
        label: "Max octaves",
        type: "int",
        value: 8,
        min: 1,
      },
      {
        key: "maxPathSegments",
        label: "Max path segs",
        type: "int",
        value: 4,
        min: 1,
      },
      {
        key: "opaqueBase",
        label: "Opaque base layer",
        type: "check",
        value: 1,
      },
      { key: "randomView", label: "Randomize View", type: "check", value: 1 },
    ]);
    addGroup("Numeric ranges", [
      {
        key: "realLo",
        label: "real min",
        type: "float",
        value: -10,
        step: 0.5,
      },
      { key: "realHi", label: "real max", type: "float", value: 10, step: 0.5 },
      {
        key: "posRealMin",
        label: "pos_real min",
        type: "float",
        value: 0.001,
        step: 0.1,
      },
      {
        key: "posRealMax",
        label: "pos_real max",
        type: "float",
        value: 10,
        step: 0.5,
      },
      {
        key: "nonNegRealMax",
        label: "non_neg max",
        type: "float",
        value: 10,
        step: 0.5,
      },
    ]);
    addGroup("Seed", [
      { key: "seed", label: "Seed (blank = random)", type: "text", value: "" },
    ]);

    dialog.insertBefore(form, textarea);

    const err = document.createElement("div");
    err.className = "io-err";
    dialog.appendChild(err);

    const actions = document.createElement("div");
    actions.className = "io-actions";
    const regen = makeBtn("🎲 Regenerate");
    const spacer = document.createElement("span");
    spacer.className = "io-spacer";
    const cancel = makeBtn("Cancel");
    const apply = makeBtn("Apply", "io-apply");
    actions.append(regen, spacer, cancel, apply);
    dialog.appendChild(actions);

    document.body.appendChild(modal);
    this.rndModal = modal;
    this.rndText = textarea;
    this.rndErr = err;
    this.rndInputs = inputs;

    regen.addEventListener("click", () => this.regenerate());
    cancel.addEventListener("click", () => this.closeRandomize());
    apply.addEventListener("click", () => this.applyRandomize());
    modal.addEventListener("click", (e) => {
      if (e.target === modal) this.closeRandomize();
    });
  }

  private buildExportModal(): void {
    const { modal, dialog, textarea } = makeModalShell("Export material");
    textarea.readOnly = true;

    const actions = document.createElement("div");
    actions.className = "io-actions";
    const spacer = document.createElement("span");
    spacer.className = "io-spacer";
    const download = makeBtn("Download");
    const copy = makeBtn("Copy", "io-apply");
    const close = makeBtn("Close");
    // Keep the action layout parallel to the Import modal: a utility button on
    // the left, then [dismiss] [primary] on the right.
    actions.append(download, spacer, close, copy);
    dialog.appendChild(actions);

    document.body.appendChild(modal);
    this.exportModal = modal;
    this.exportText = textarea;
    this.copyBtn = copy;

    copy.addEventListener("click", () => void this.copyExport());
    download.addEventListener("click", () => this.downloadExport());
    close.addEventListener("click", () => this.closeExport());
    modal.addEventListener("click", (e) => {
      if (e.target === modal) this.closeExport();
    });
  }

  private buildImportModal(): void {
    const { modal, dialog, textarea } = makeModalShell("Import material");
    textarea.placeholder = "Paste a Material(...) string, or load a .txt file…";

    const err = document.createElement("div");
    err.className = "io-err";
    dialog.appendChild(err);

    const actions = document.createElement("div");
    actions.className = "io-actions";

    const fileLabel = document.createElement("label");
    fileLabel.className = "io-btn io-file";
    fileLabel.textContent = "Load file…";
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".txt,text/plain";
    fileInput.hidden = true;
    fileLabel.appendChild(fileInput);

    const spacer = document.createElement("span");
    spacer.className = "io-spacer";
    const cancel = makeBtn("Cancel");
    const apply = makeBtn("Apply", "io-apply");
    actions.append(fileLabel, spacer, cancel, apply);
    dialog.appendChild(actions);

    document.body.appendChild(modal);
    this.importModal = modal;
    this.importText = textarea;
    this.importFile = fileInput;
    this.importErr = err;

    fileInput.addEventListener("change", () => {
      const file = fileInput.files?.[0];
      if (!file) return;
      file
        .text()
        .then((t) => {
          textarea.value = t;
          err.textContent = "";
        })
        .catch((e) => (err.textContent = `Could not read file: ${e.message}`));
    });
    cancel.addEventListener("click", () => this.closeImport());
    apply.addEventListener("click", () => this.applyImport());
    modal.addEventListener("click", (e) => {
      if (e.target === modal) this.closeImport();
    });
  }
}

// Backdrop + dialog + title + textarea; callers append their own action row.
function makeModalShell(title: string): {
  modal: HTMLDivElement;
  dialog: HTMLDivElement;
  textarea: HTMLTextAreaElement;
} {
  const modal = document.createElement("div");
  modal.className = "io-modal";
  modal.hidden = true;

  const dialog = document.createElement("div");
  dialog.className = "io-dialog";

  const heading = document.createElement("div");
  heading.className = "io-title";
  heading.textContent = title;

  const textarea = document.createElement("textarea");
  textarea.className = "io-text";
  textarea.spellcheck = false;

  dialog.append(heading, textarea);
  modal.appendChild(dialog);
  return { modal, dialog, textarea };
}

// One control in the Randomize form. `int`/`float` render a number input,
// `check` a checkbox, `text` a free text field (used for the optional seed).
interface RndField {
  key: string;
  label: string;
  type: "int" | "float" | "check" | "text";
  value: number | string;
  min?: number;
  max?: number;
  step?: number;
}

// A labelled field for the Randomize form; returns the wrapper and its input.
function makeRndField(f: RndField): {
  field: HTMLLabelElement;
  input: HTMLInputElement;
} {
  const field = document.createElement("label");
  field.className = f.type === "check" ? "rnd-field rnd-check" : "rnd-field";

  const caption = document.createElement("span");
  caption.className = "rnd-label";
  caption.textContent = f.label;

  const input = document.createElement("input");
  if (f.type === "check") {
    input.type = "checkbox";
    input.checked = Boolean(f.value);
  } else if (f.type === "text") {
    input.type = "text";
    input.value = String(f.value);
  } else {
    input.type = "number";
    input.value = String(f.value);
    if (f.min !== undefined) input.min = String(f.min);
    if (f.max !== undefined) input.max = String(f.max);
    input.step = String(f.step ?? (f.type === "int" ? 1 : "any"));
  }
  input.className = "rnd-input";

  // Checkbox sits before its label; everything else after.
  if (f.type === "check") field.append(input, caption);
  else field.append(caption, input);
  return { field, input };
}

function makeBtn(label: string, extra = ""): HTMLButtonElement {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = extra ? `io-btn ${extra}` : "io-btn";
  btn.textContent = label;
  return btn;
}

// Local timestamp YYYYMMDD-HHMMSS for unique download file names.
function timestamp(): string {
  const d = new Date();
  const p = (n: number): string => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
    `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
  );
}
