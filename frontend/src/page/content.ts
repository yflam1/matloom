// Page content, separated from rendering so edits happen here, not in DOM code.
// Prose paragraphs are page-editorial text (plain-language summaries of the
// paper's results); the abstract is the paper abstract verbatim.

export interface FigureImage {
  src: string;
  alt: string;
  label?: string; // small caption-like tag under/next to the image
}

export interface Figure {
  id: string;
  /** single: one img; row: labeled image strip; grid: comparison grid. */
  kind: "single" | "row" | "grid";
  maxWidth: string;
  images: FigureImage[];
  /** grid only: one label per image column (method names). */
  columnLabels?: string[];
  /** grid only: one label per image row (prompt names). */
  rowLabels?: string[];
}

export interface Section {
  heading: string;
  /** Alternate the background: sections render white / light down the page. */
  light?: boolean;
  figure?: Figure;
  paragraphs: string[];
}

export interface Author {
  name: string;
  url: string;
  sup: string; // superscript mark after the name ("" for none)
}

export interface ButtonLink {
  icon: string; // fontawesome/academicons class
  label: string;
  url: string;
}

export const HERO = {
  title:
    "MatLoom: Layered Text-to-Material Generation in a Compact Program Space",
  authors: [
    { name: "Anson Y. Lam", url: "https://yflam1.github.io/", sup: "" },
    {
      name: "Shuqing Li",
      url: "https://shuqing-li.github.io/",
      sup: "*",
    },
    {
      name: "Michael R. Lyu",
      url: "https://www.cse.cuhk.edu.hk/lyu/home",
      sup: "",
    },
  ] as Author[],
  affiliation:
    "Department of Computer Science and Engineering<br>The Chinese University of Hong Kong<br>2026",
  note: "*Corresponding author.",
  buttons: [
    {
      icon: "fab fa-github",
      label: "Code",
      url: "https://github.com/yflam1/matloom",
    },
    {
      icon: "ai ai-arxiv",
      label: "arXiv",
      url: "https://arxiv.org/abs/2609.40322",
    },
  ] as ButtonLink[],
};

// Paper teaser: a strip of direct text-to-material outputs, shown right after
// the hero and BEFORE the abstract (labels are the text prompts).
export const TEASER = {
  images: [
    {
      src: "figures/teaser/wood-planks.png",
      alt: "Flat render of wood planks",
      label: "Wood planks",
    },
    {
      src: "figures/teaser/black-slate-flooring.png",
      alt: "Flat render of black slate flooring",
      label: "Black slate flooring",
    },
    {
      src: "figures/teaser/hammered-rusty-metal.png",
      alt: "Flat render of hammered rusty metal",
      label: "Hammered rusty metal",
    },
    {
      src: "figures/teaser/black-and-white-marble-checker-tile.png",
      alt: "Flat render of a black and white marble checker tile pattern",
      label: "Black and white marble checker tile",
    },
    {
      src: "figures/teaser/triangle-ceramic-tiles.png",
      alt: "Flat render of triangle ceramic tiles",
      label: "Triangle ceramic tiles",
    },
    {
      src: "figures/teaser/brown-and-white-checkered-cotton-fabric.png",
      alt: "Flat render of brown and white checkered cotton fabric",
      label: "Brown and white checkered cotton fabric",
    },
  ] as FigureImage[],
  note: "Every image above is the output of a short program that a language model wrote from the short label underneath it. Across the paper's 141-prompt benchmark, such programs have a median length of 21 lines. Because the program is saved alongside the picture, the material can be rebuilt at any resolution, changed by editing a line, or re-rolled with a new random seed.",
};

// The embedded interactive viewer: the standalone /viewer/ app itself, live in
// a same-origin iframe between the teaser and the abstract (an iframe because
// the app and this page share element ids and global styles, so an in-page
// mount is not viable). Immediately interactive — the canvas captures wheel
// and drag, so those gestures act on the material instead of the page.
export const DEMO = {
  heading: "Try it in your browser",
  src: "viewer/",
  iframeTitle: "MatLoom interactive material viewer",
  note: 'This demo is the full viewer running in your browser, with nothing to install. It opens with the black slate flooring material from the teaser strip already loaded, seeds included. Edit its layers and channel expressions and watch the surface re-render live; drag to orbit, scroll to zoom, switch the lighting, toggle relief channels such as normals and displacement, or press Randomize for a fresh material on demand. Prefer more room? <a href="viewer/">Open the viewer full-screen</a>. Either way, you are touching the paper\'s core claim directly: a material is a program you can read and tweak, not a static picture.',
};

// Paper abstract, verbatim. Math stays as \( ... \) for MathJax.
export const ABSTRACT_HTML =
  "<p>Material generation should produce not only an appearance, but also the rules that construct it. We introduce MatLoom, a compact, layer-oriented language for text-to-material generation with pretrained language models. Each program composes alpha-masked layers whose shared spatial expressions define coverage and physically based rendering (PBR) channels, making dependencies between patterns, color, and relief explicit. A standalone interpreter evaluates the program into material maps, while the source retains named fields and layer parameters for subsequent authoring. Without task-specific fine-tuning, our pipeline uses parser-guided repair and preview-based critique to revise material designs, then searches noise seeds while keeping each candidate's remaining source fixed. On a curated benchmark of \\(141\\) prompts evaluated with six backbones, our best-performing configuration achieves higher mean scores than three diffusion baselines on all four flat-layout prompt-alignment metrics. Its initial programs already exceed all three baselines on mean BLIPScore, before critique or seed search. Retained programs have a median length of \\(21\\) lines when pooled across backbones. In a blind four-way comparison involving \\(30\\) participants and \\(20\\) prompts, our renders receive \\(59.2\\%\\) of choices, compared with \\(19.3\\%\\) for the most-preferred baseline. Compact executable programs thus offer a way to generate prompt-aligned materials while retaining their construction as part of the asset.</p>";

// One section per figure: image first, then plain-language paragraphs.
export const SECTIONS = [
  {
    heading: "A material you can read and edit",
    figure: {
      id: "fig-overview",
      kind: "single",
      maxWidth: "100%",
      images: [
        {
          src: "figures/overview.png",
          alt: "The MatLoom pipeline: a text request becomes a layered program, which is critiqued with a preview and revised, then explored across noise seeds",
        },
      ],
    },
    paragraphs: [
      "MatLoom describes a surface material as a short program, median length just 21 lines, that a computer runs to produce the material at any resolution. The program is a stack of layers, like coats of paint applied from the bottom up: each layer has a mask saying where it shows up, plus surface properties such as color, roughness, and height. One named pattern can drive several of these at once, so the same tile outline decides where the glaze appears and how raised each tile edge is. Editing one number, the mortar width, widens the grout and adjusts the tile relief together.",
      "The diagram above shows how a request becomes a material. A standard pretrained language model, with no task-specific training, writes a program from the text while a parser reports errors it fixes along the way. A critic step then looks at a fast preview, statistics of the material's color and relief channels, and the program source, and revises the design. A final search tries many random seeds for the fixed design and keeps the best-scoring one.",
    ],
  },
  {
    heading: "A critic that reviews its own work",
    light: true,
    figure: {
      id: "fig-refine",
      kind: "row",
      maxWidth: "100%",
      images: [
        {
          src: "figures/refine/r0.png",
          alt: "First draft of the giraffe-skin material: rounded, partly merged patches",
          label: "Start",
        },
        {
          src: "figures/refine/r1.png",
          alt: "Giraffe-skin material after critique round one",
          label: "Round 1",
        },
        {
          src: "figures/refine/r2.png",
          alt: "Giraffe-skin material after critique round two",
          label: "Round 2",
        },
        {
          src: "figures/refine/r3.png",
          alt: "Giraffe-skin material after critique round three",
          label: "Round 3",
        },
        {
          src: "figures/refine/r4.png",
          alt: "Giraffe-skin material after critique round four",
          label: "Round 4",
        },
        {
          src: "figures/refine/r5.png",
          alt: "Giraffe-skin material after critique round five, with large polygonal patches",
          label: "Round 5",
        },
      ],
    },
    paragraphs: [
      "Six snapshots of one material, generated from the prompt &ldquo;giraffe skin with large polygonal patches&rdquo;. The first draft misses the request: the patches come out rounded and partly merged. Over five critique rounds the patch shape, color, and scale move toward the request, until the large cream-and-brown polygons of giraffe skin emerge. An automatic score of how well the render matches its prompt rises from 2.8 to 99.8 out of 100 across the rounds shown.",
    ],
  },
  {
    heading: "Change one thing, keep everything else",
    figure: {
      id: "fig-edits",
      kind: "row",
      maxWidth: "90%",
      images: [
        {
          src: "figures/edits/original.png",
          alt: "Original glazed tile material",
          label: "Original",
        },
        {
          src: "figures/edits/blue-glaze.png",
          alt: "The same tile material with a bluer glaze",
          label: "Bluer glaze",
        },
        {
          src: "figures/edits/higher-roughness.png",
          alt: "The same tile material with a rougher finish",
          label: "Rougher finish",
        },
        {
          src: "figures/edits/wider-grout.png",
          alt: "The same tile material with wider grout between tiles",
          label: "Wider grout",
        },
      ],
    },
    paragraphs: [
      "Because the output is a program, changing your mind does not mean starting over. Each image above edits one part of the same tile program: a bluer glaze, a rougher finish, or wider grout between the tiles. The remaining source and all random seeds stay untouched, so nothing outside the edited expression moves. When an expression is shared, the change travels with it by design: widening the grout also reshapes the tile edges built from the same field. The same separation powers rerolling: a new seed gives a fresh variation of an unchanged design.",
    ],
  },
  {
    heading: "Side by side with three prior systems",
    light: true,
    figure: {
      id: "fig-comparison",
      kind: "grid",
      maxWidth: "100%",
      columnLabels: [
        "MatLoom (ours)",
        "StableMaterials",
        "IntrinsiX",
        "MatFuse",
      ],
      rowLabels: ["Holiday wrapping paper", "Acoustic panels", "Cracked ice"],
      images: [
        {
          src: "figures/comparison/holiday-wrapping-paper-ours.png",
          alt: "Holiday wrapping paper rendered by MatLoom in a lit scene",
        },
        {
          src: "figures/comparison/holiday-wrapping-paper-stablematerials.png",
          alt: "Holiday wrapping paper rendered by StableMaterials in a lit scene",
        },
        {
          src: "figures/comparison/holiday-wrapping-paper-intrinsix.png",
          alt: "Holiday wrapping paper rendered by IntrinsiX in a lit scene",
        },
        {
          src: "figures/comparison/holiday-wrapping-paper-matfuse.png",
          alt: "Holiday wrapping paper rendered by MatFuse in a lit scene",
        },
        {
          src: "figures/comparison/acoustic-panels-ours.png",
          alt: "Acoustic panels rendered by MatLoom in a lit scene",
        },
        {
          src: "figures/comparison/acoustic-panels-stablematerials.png",
          alt: "Acoustic panels rendered by StableMaterials in a lit scene",
        },
        {
          src: "figures/comparison/acoustic-panels-intrinsix.png",
          alt: "Acoustic panels rendered by IntrinsiX in a lit scene",
        },
        {
          src: "figures/comparison/acoustic-panels-matfuse.png",
          alt: "Acoustic panels rendered by MatFuse in a lit scene",
        },
        {
          src: "figures/comparison/cracked-ice-ours.png",
          alt: "Cracked ice rendered by MatLoom in a lit scene",
        },
        {
          src: "figures/comparison/cracked-ice-stablematerials.png",
          alt: "Cracked ice rendered by StableMaterials in a lit scene",
        },
        {
          src: "figures/comparison/cracked-ice-intrinsix.png",
          alt: "Cracked ice rendered by IntrinsiX in a lit scene",
        },
        {
          src: "figures/comparison/cracked-ice-matfuse.png",
          alt: "Cracked ice rendered by MatFuse in a lit scene",
        },
      ],
    },
    paragraphs: [
      "On a benchmark of 141 prompts, MatLoom is compared with three published diffusion systems, AI models that generate material images directly, pixel by pixel. Each row above shows the same prompt rendered by every method as a lit scene; the row captions name the requested material, and the examples were chosen to show structural differences rather than typical performance. The best MatLoom configuration ranks first on all four automatic text-matching scores, whether materials are scored as flat swatches or as lit scenes. On one of them, BLIPScore, it scores 56.06 against 29.94 for the closest system, and its unrefined first drafts, before any critique or seed search, already clear all three. Every one of the six language models tested also beats that closest score.",
    ],
  },
  {
    heading: "Human preference study",
    paragraphs: [
      "Automatic scores are one thing; do people agree? In a blind study, 30 participants each saw 20 unlabeled four-way grids and picked the render that best matched its prompt. MatLoom received 59.2% of the 600 choices, about three times the 19.3% share of the runner-up, and every participant chose it more often than any single baseline. It also earned the highest fidelity rating, 4.98 out of 7.",
      "The paper is candid about limits: the seed-search stage uses a larger generation budget than the baselines, and the metrics measure prompt alignment rather than physical accuracy. The results show that compact programs are competitive with diffusion outputs, not that this particular language beats every other program-based approach.",
    ],
  },
] as Section[];

export const BIBTEX =
  "@article{lam2026matloom,\n  title={MatLoom: Layered Text-to-Material Generation in a Compact Program Space},\n  author={Lam, Anson Y. and Li, Shuqing and Lyu, Michael R.},\n  journal={arXiv preprint arXiv:2609.40322},\n  year={2026}\n}";

export const TEMPLATE_CREDIT_HTML =
  'This page was built using the <a href="https://github.com/eliahuhorwitz/Academic-project-page-template" target="_blank">Academic Project Page Template</a> which was adopted from the <a href="https://nerfies.github.io" target="_blank">Nerfies</a> project page. ' +
  "You are free to borrow the source code of this website, we just ask that you link back to this page in the footer. <br> This website is licensed under a " +
  '<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/" target="_blank">Creative Commons Attribution-ShareAlike 4.0 International License</a>.';
