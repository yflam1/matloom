"""Build the curated evaluation benchmark from the four raw prompt datasets.

This module is the single entry point for benchmark construction. It replaces
the old two-file split (``matloom/eval/select.py`` for the sampling logic and
``scripts/select_prompts.py`` for the CLI/orchestration), collapsing both into
one place next to the dataset loaders under :mod:`matloom.benchmark`.

Pipeline
--------
For each of the four datasets (GenProc, MatSynth, StableMaterials, text2fabric)
we load every available prompt via the per-dataset loader in
:mod:`matloom.benchmark.loaders`, then keep at most ``max_per_dataset`` of them
using *farthest-point sampling* over Sentence-BERT embeddings. The result is
written to ``datasets/benchmark.json`` as::

    {"seed": 42, "datasets": {"GenProc": ["...", ...], "MatSynth": [...], ...}}

and is consumed by the ``matloom.eval`` runners.

Why farthest-point sampling
---------------------------
A random subset can cluster around common material families (wood, metal,
fabric) and miss the long tail. Farthest-point (max-min) sampling greedily
grows the selected set so that every new pick is as far as possible, in cosine
space, from everything already chosen. This spreads the benchmark across the
full breadth of the dataset with a small, fixed budget. Sampling is greedy and
therefore deterministic given the seed/start index, so a fixed ``seed``
reproduces the exact same benchmark.

Heavy deps stay out of the import path
-------------------------------------
``numpy`` is light and imported at the top. The sentence embedder
(``SentenceBert``, which pulls in torch) is imported lazily inside
:func:`create_benchmark`, so importing this module is torch-free and the pure
selection logic in :func:`select_diverse` can be unit-tested with a fake
``encode`` callable and no network.

CLI
---
    python -m matloom.benchmark.create                    # 50/dataset, seed 42
    python -m matloom.benchmark.create --per-dataset 25
    python -m matloom.benchmark.create --seed 7
    python -m matloom.benchmark.create --out-name dev.json
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from matloom.benchmark.loaders import (
    genproc,
    matsynth,
    stablematerials,
    text2fabric,
)
from matloom.constants import DEFAULT_DATASET_DIR
from matloom.utils.dtypes import SDict
from matloom.utils.maths import clamp


def select_diverse(
    prompts: list[str],
    k: int,
    *,
    encode: Callable[[list[str]], np.ndarray],
    start: int = 0,
) -> list[str]:
    """Pick ``k`` diverse prompts by farthest-point sampling over embeddings.

    Embeddings come from the injected ``encode`` callable (so the pure sampling
    logic is testable without torch or a network). They are assumed
    L2-normalized, as ``SentenceBert.encode`` produces, so the dot product is
    cosine similarity and ``distance = 1 - similarity``.

    The algorithm seeds the selected set with index ``start``, then repeatedly
    adds the prompt whose *minimum* cosine distance to the already-selected set
    is largest (the classic max-min / farthest-point heuristic). ``start`` is
    clamped into range, so a fixed seed reproduces the same selection.

    Returns exactly ``min(k, len(prompts))`` prompts in selection order, or an
    empty list when ``prompts`` is empty or ``k <= 0``. Note that when
    ``k >= len(prompts)`` the prompts come back in *selection* order, not
    original order; callers that want all prompts unchanged should short-circuit
    before calling (as :func:`create_benchmark` does).
    """
    if len(prompts) == 0 or k <= 0:
        return []
    k = min(k, len(prompts))
    start = clamp(start, 0, len(prompts) - 1)
    emb = encode(prompts)
    selected = [start]
    # Min cosine distance of each point to the selected set, seeded by `start`.
    sims = emb @ emb[start]
    min_dist = 1.0 - sims
    min_dist[start] = -np.inf  # Never reselect the seed.
    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        sims = emb @ emb[nxt]
        np.minimum(min_dist, 1.0 - sims, out=min_dist)  # In-place update.
        min_dist[nxt] = -np.inf  # Never reselect the pick.
    return [prompts[i] for i in selected]


def create_benchmark(
    max_per_dataset: int | None = None,
    dataset_dir: Path | None = None,
    seed: int | None = None,
) -> SDict[int | SDict[list[str]]]:
    """Build the benchmark dict ``{"seed", "datasets"}`` for all four datasets.

    ``max_per_dataset`` (default 50) caps how many prompts each dataset
    contributes; when a dataset has no more than that many prompts it is kept
    wholesale (original order), otherwise :func:`select_diverse` picks a diverse
    subset. ``seed`` (default 42) chooses each dataset's farthest-point start
    index, which makes the whole selection reproducible.

    The Sentence-BERT embedder is constructed here (lazily imported) and passed
    to :func:`select_diverse` as the ``encode`` callable. The returned dict is
    JSON-serializable and matches the format written to
    ``datasets/benchmark.json``.
    """
    from tqdm import tqdm

    from matloom.models.sbert import SentenceBert

    max_per_dataset = 50 if max_per_dataset is None else max_per_dataset
    seed = 42 if seed is None else seed

    sbert = SentenceBert()

    def encode(texts: list[str]) -> np.ndarray:
        return sbert.encode(texts).cpu().numpy()

    benchmark: SDict[int | SDict[list[str]]] = {"seed": seed, "datasets": {}}
    for ds in tqdm(
        [genproc, matsynth, stablematerials, text2fabric],
        desc="Creating benchmark",
    ):
        prompts: list[str] = ds.load(dataset_dir)
        # `seed` chooses the farthest-point start so a fixed seed reproduces it.
        start = seed % max(len(prompts), 1)
        chosen = (
            select_diverse(
                prompts, max_per_dataset, encode=encode, start=start
            )
            if len(prompts) > max_per_dataset
            else prompts
        )
        benchmark["datasets"][ds.DATASET_NAME] = chosen
        print(f"{ds.DATASET_NAME}: selected {len(chosen)} prompts")

    # Check that no prompt appears in more than one dataset
    all_prompts = [
        p for prompts in benchmark["datasets"].values() for p in prompts
    ]
    assert len(all_prompts) == len(set(all_prompts)), (
        "Duplicate prompts across datasets"
    )

    return benchmark


if __name__ == "__main__":
    import argparse

    from matloom.utils.file import save_json

    parser = argparse.ArgumentParser(
        prog="matloom.benchmark.create",
        description="Build a diverse evaluation benchmark from the four prompt datasets.",
    )
    parser.add_argument(
        "--per-dataset",
        type=int,
        default=None,
        help="max prompts to pick from each dataset (default 50)",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=None, help="raw datasets dir"
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="benchmark.json",
        help="output benchmark JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seeds the farthest-point start index (reproducible selection)",
    )
    args = parser.parse_args()

    benchmark = create_benchmark(args.per_dataset, args.dataset_dir, args.seed)
    out_dir = DEFAULT_DATASET_DIR
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / args.out_name
    save_json(benchmark, out_path, indent=2)
    print(f"Saved benchmark to {out_path}")
