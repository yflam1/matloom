// DOM builders for the page. Content lives in content.ts; this module only
// knows how to turn that data into nodes, mirroring the Academic Project Page
// Template's Bulma markup.
import {
  ABSTRACT_HTML,
  BIBTEX,
  DEMO,
  HERO,
  SECTIONS,
  TEASER,
  TEMPLATE_CREDIT_HTML,
} from "./content";
import type { Figure, FigureImage, Section } from "./content";

type Attrs = Record<string, string>;

function el(tag: string, cls?: string, attrs?: Attrs): HTMLElement {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  }
  return node;
}

function setHtml(node: HTMLElement, markup: string): HTMLElement {
  node.innerHTML = markup;
  return node;
}

function heading(
  parent: HTMLElement,
  cls: string,
  text: string,
  tag = "h2",
): void {
  const h = el(tag, cls);
  h.textContent = text;
  parent.appendChild(h);
}

function cellImage(cell: FigureImage, lazy = true): HTMLElement {
  const img = document.createElement("img");
  img.src = cell.src;
  img.alt = cell.alt;
  if (lazy) img.loading = "lazy";
  return img;
}

/** A labeled image cell: image with an optional small tag underneath. */
function figureCell(cell: FigureImage): HTMLElement {
  const wrap = el("div", "fig-cell");
  wrap.appendChild(cellImage(cell));
  if (cell.label) {
    const label = el("span", "fig-cell-label");
    label.textContent = cell.label;
    wrap.appendChild(label);
  }
  return wrap;
}

function heroSection(): HTMLElement {
  const section = el("section", "hero");
  const body = el("div", "hero-body");
  const container = el("div", "page-container");
  const columns = el("div", "columns is-centered");
  const column = el("div", "column has-text-centered");

  heading(column, "title is-1 publication-title", HERO.title, "h1");

  const authors = el("div", "is-size-5 publication-authors");
  for (const a of HERO.authors) {
    const block = el("span", "author-block");
    const link = el("a", undefined, { href: a.url, target: "_blank" });
    link.textContent = a.name;
    block.appendChild(link);
    if (a.sup) {
      const sup = document.createElement("sup");
      sup.innerHTML = a.sup;
      block.appendChild(sup);
    }
    block.append(",");
    authors.appendChild(block);
  }
  column.appendChild(authors);

  const affil = el("div", "is-size-5 publication-authors");
  const affilBlock = setHtml(el("span", "author-block"), HERO.affiliation);
  affil.appendChild(affilBlock);
  const note = el("span", "eql-cntrb");
  const small = document.createElement("small");
  small.appendChild(document.createElement("br"));
  small.innerHTML += HERO.note;
  note.appendChild(small);
  affil.appendChild(note);
  column.appendChild(affil);

  const linkColumn = el("div", "column has-text-centered");
  const links = el("div", "publication-links");
  for (const b of HERO.buttons) {
    const block = el("span", "link-block");
    const isExternal = /^https?:\/\//.test(b.url);
    const attrs: Attrs = { href: b.url };
    if (isExternal) attrs.target = "_blank";
    const a = el(
      "a",
      "external-link button is-normal is-rounded is-dark",
      attrs,
    );
    const icon = el("span", "icon");
    const i = document.createElement("i");
    i.className = b.icon;
    icon.appendChild(i);
    const label = el("span");
    label.textContent = b.label;
    a.append(icon, label);
    block.appendChild(a);
    links.appendChild(block);
  }
  linkColumn.appendChild(links);
  column.appendChild(linkColumn);

  columns.appendChild(column);
  container.appendChild(columns);
  body.appendChild(container);
  section.appendChild(body);
  return section;
}

/**
 * Paper teaser: a strip of direct outputs between the hero and the abstract.
 * Images first, then one plain-language note underneath.
 */
function teaserSection(): HTMLElement {
  const section = el("section", "hero teaser");
  const body = el("div", "hero-body");
  const container = el("div", "page-container");
  const grid = el("div", "teaser-grid");
  for (const cell of TEASER.images) {
    grid.appendChild(figureCell(cell));
  }
  container.appendChild(grid);
  const content = el("div", "content paper-content teaser-note");
  setHtml(content.appendChild(el("p")), TEASER.note);
  container.appendChild(content);
  body.appendChild(container);
  section.appendChild(body);
  return section;
}

/**
 * The embedded interactive viewer: the standalone /viewer/ app in a same-origin
 * iframe, below the teaser and above the abstract, immediately interactive
 * (the canvas preventDefaults wheel/pointer events, so scrolling over it zooms
 * the material rather than scrolling the page). Same page-wide container as
 * every other section, so the page reads as one uniform column.
 */
function demoSection(): HTMLElement {
  const section = el("section", "hero demo");
  const body = el("div", "hero-body");
  const container = el("div", "page-container");
  heading(container, "title is-3", DEMO.heading);

  const frame = el("div", "demo-frame");
  const iframe = document.createElement("iframe");
  iframe.src = DEMO.src;
  iframe.title = DEMO.iframeTitle;
  iframe.loading = "lazy";
  frame.appendChild(iframe);
  container.appendChild(frame);

  const content = el("div", "content paper-content demo-note");
  setHtml(content.appendChild(el("p")), DEMO.note);
  container.appendChild(content);

  body.appendChild(container);
  section.appendChild(body);
  return section;
}

function abstractSection(): HTMLElement {
  const section = el("section", "section hero is-light", { id: "abstract" });
  const container = el("div", "page-container");
  const columns = el("div", "columns is-centered has-text-centered");
  const column = el("div", "column is-four-fifths");
  heading(column, "title is-3", "Abstract");
  const content = setHtml(
    el("div", "content has-text-justified"),
    ABSTRACT_HTML,
  );
  column.appendChild(content);
  columns.appendChild(column);
  container.appendChild(columns);
  section.appendChild(container);
  return section;
}

function singleFigure(figure: Figure): HTMLElement {
  const node = el("figure", "paper-fig", { id: figure.id });
  const img = cellImage(figure.images[0]);
  img.style.maxWidth = figure.maxWidth;
  node.appendChild(img);
  return node;
}

function rowFigure(figure: Figure): HTMLElement {
  const node = el("figure", "paper-fig", { id: figure.id });
  const row = el("div", "fig-row");
  row.style.maxWidth = figure.maxWidth;
  for (const cell of figure.images) {
    row.appendChild(figureCell(cell));
  }
  node.appendChild(row);
  return node;
}

function gridFigure(figure: Figure): HTMLElement {
  const node = el("figure", "paper-fig", { id: figure.id });
  const grid = el("div", "fig-grid");
  grid.style.maxWidth = figure.maxWidth;
  // A grid without column labels degrades to a single row rather than
  // dropping every image (cols = 0 would make i % cols NaN forever).
  const cols = figure.columnLabels?.length ?? figure.images.length;
  if (figure.columnLabels) {
    const head = el("div", "fig-grid-head");
    head.appendChild(el("span", "fig-corner"));
    for (const label of figure.columnLabels) {
      const span = el("span");
      span.textContent = label;
      head.appendChild(span);
    }
    grid.appendChild(head);
  }
  for (let i = 0; i < figure.images.length; i++) {
    const row = i % cols;
    if (row === 0) {
      const rowNode = el("div", "fig-grid-row");
      if (figure.rowLabels) {
        const label = el("span", "fig-row-label");
        label.textContent = figure.rowLabels[i / cols] ?? "";
        rowNode.appendChild(label);
      }
      grid.appendChild(rowNode);
    }
    grid.lastElementChild?.appendChild(cellImage(figure.images[i]));
  }
  node.appendChild(grid);
  return node;
}

function paperSection(s: Section): HTMLElement {
  const light = s.light === true;
  const section = el("section", light ? "section hero is-light" : "section");
  const container = el("div", "page-container");
  heading(container, "title is-3", s.heading);
  const content = el("div", "content paper-content");
  if (s.figure) {
    const builders: Record<Figure["kind"], (f: Figure) => HTMLElement> = {
      single: singleFigure,
      row: rowFigure,
      grid: gridFigure,
    };
    content.appendChild(builders[s.figure.kind](s.figure));
  }
  for (const p of s.paragraphs) {
    setHtml(content.appendChild(el("p")), p);
  }
  container.appendChild(content);
  section.appendChild(container);
  return section;
}

function bibtexSection(onCopy: () => void): HTMLElement {
  const section = el("section", "section", { id: "BibTeX" });
  const container = el("div", "page-container content");
  const header = el("div", "bibtex-header");
  heading(header, "title", "BibTeX");
  const button = el("button", "copy-bibtex-btn", {
    title: "Copy BibTeX to clipboard",
  });
  button.addEventListener("click", onCopy);
  const icon = document.createElement("i");
  icon.className = "fas fa-copy";
  const label = el("span", "copy-text");
  label.textContent = "Copy";
  button.append(icon, label);
  header.appendChild(button);
  container.appendChild(header);

  const pre = el("pre", undefined, { id: "bibtex-code" });
  const code = document.createElement("code");
  code.textContent = BIBTEX;
  pre.appendChild(code);
  container.appendChild(pre);
  section.appendChild(container);
  return section;
}

function footer(): HTMLElement {
  // The credit spans the page container directly (no centered column relic
  // from the template): its left edge aligns with every section above, and
  // the wider measure avoids one-word widows in the license sentence.
  const footer = el("footer", "footer");
  const container = el("div", "page-container");
  const content = el("div", "content");
  content.appendChild(setHtml(el("p"), TEMPLATE_CREDIT_HTML));
  container.appendChild(content);
  footer.appendChild(container);
  return footer;
}

/** Build the whole page under <main id="main-content">. */
export function renderPage(onCopyBibTeX: () => void): HTMLElement {
  const main = el("main", undefined, { id: "main-content" });
  main.appendChild(heroSection());
  main.appendChild(teaserSection());
  main.appendChild(demoSection());
  main.appendChild(abstractSection());
  for (const s of SECTIONS) main.appendChild(paperSection(s));
  main.appendChild(bibtexSection(onCopyBibTeX));
  main.appendChild(footer());
  return main;
}

/** The template's scroll-to-top floating button. */
export function renderScrollButton(): HTMLButtonElement {
  const button = el("button", "scroll-to-top", {
    title: "Scroll to top",
    "aria-label": "Scroll to top",
  }) as HTMLButtonElement;
  const i = document.createElement("i");
  i.className = "fas fa-chevron-up";
  button.appendChild(i);
  return button;
}
