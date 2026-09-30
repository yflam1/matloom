"""Print paper LaTeX tables from the eval result JSONs under ``experiments/results/``.

Reads the newest file of each result family (overridable via CLI flags):
``baselines*.json``, ``ours_main_all*.json``, ``ours_main_final*.json``, and
``ours_ablations*.json``. Every family is saved incrementally by its scoring
script, so partial runs are tolerated: unreadable files are skipped, missing
cells render as ``--``, and coverage warnings go to stderr. An absent family
only removes the rows/tables that need it.

JSON shapes (all written by the ``matloom/eval/eval_*.py`` scripts):

- ``baselines*.json`` — ``{method: {"plane"|"scene": {metric: {prompt: {seed:
  {"score", "path"}}}}, "mobileclip2": {prompt: seed}}}``; one score per
  (method, layout, metric, prompt, seed).
- ``ours_main_all*.json`` — the trajectory-wide family for the flagship
  generator (gemini): same as baselines, but every score is a 0-based *list*
  over programs and the generator dict adds ``idx``. Indices 0..5 are the
  trajectory (0 = initial generation, 1..5 = refinement rounds) and the tail
  holds the polished variants in the order they were polished (6 = polished
  first, 7 = polished chosen or last, 8 = polished last when the chosen index
  is 1..4), so lists have length 8 or 9. ``idx[prompt][seed]`` gives
  ``{"before": <mobileclip2-chosen trajectory index 0..5>, "final": <index of
  the run's final program, always >= 6>}``.
- ``ours_main_final*.json`` — ``{generator: <baselines-method shape>}`` for
  every remaining generator (gpt/glm/deepseek/gemma/qwen as scored); one
  score per seed (the final program).
- ``ours_ablations*.json`` — ``{ablation: {"plane"|"scene": {metric: {prompt:
  float}}}}`` for the seed-42 GPT ablation runs; one bare float per prompt.

Tables (cells are ``plane/scene`` unless noted):

- main — baselines first, then the Ours rows (GPT, Gemini, GLM, DeepSeek,
  Gemma, Qwen as data allows). Cells are one-line ``plane/scene`` seed means
  (the mean over prompts of the per-prompt seed mean; the all-family
  generator's seeds use their ``idx.final`` programs). The best value of each
  column half across all rows is bolded.
- refinement — round 0's mean trajectory-program score, then rounds 1..5 as
  signed deltas from it, over every (prompt, seed) run of the all-family
  generator. The best round of each column half is bolded (its cell shows
  round 0's absolute when round 0 wins, else the delta).
- polish — signed mean delta from polishing the first program, the
  mobileclip2-chosen program, and the last program, plus the mean of the three
  rows. All rows average over all runs; the chosen comparison maps onto
  first/last when the chosen index is 0 or 5 (length-8 lists).
- selector + winrate — mean scores of the first / mobileclip2-chosen / last
  trajectory programs plus the per-metric oracle best over indices 0..5, then
  the pick-vs-last win rate and mean delta, answering whether the fast
  selector beats always taking the last program.
- ablations — the full-configuration row (GPT seed 42, absolute means,
  omitted unless its prompt set matches the ablations) then the ablation rows
  as signed deltas against it in README order, falling back to absolute means
  when the full row is unavailable. The full row reads the gpt generator's
  seed-42 finals from the final family, falling back to a gpt all-family JSON.
- significance — the flagship generator (the all-family generator, gemini by
  convention) vs. every baseline, per metric and layout: paired two-sided
  Wilcoxon signed-rank test and a 95% bootstrap CI of the mean delta, both
  over per-prompt seed means (scipy required; skipped when absent).

Run-tree tables (these walk ``experiments/runs/ours/main`` via ``--runs-dir``
instead of the result JSONs):

- polish-budget — Stage III's sweep budget ablated from stored data, no
  reruns: each run's ``polish[i].candidates`` holds the quick-scorer score of
  the original (index 0) plus every re-seeded variant in evaluation order, so
  truncating to the first ``k`` re-seeds replays the accept-if-better
  selection exactly. Reports the mean quick-scorer gain over the best
  pre-polish program per budget (``--budgets``), the share of the k=1000
  gain, and the share of runs already at their full best. Flagship backbone
  only by default (``--backbone``).
- program-stats — median final-program size (lines and bytes) per backbone
  and pooled; the final program is the last ``<program>`` of trajectory.xml,
  the text the eval scripts score.
- runtime — per-backbone median wall-clock minutes, median polish share of
  the total, and the polish p90.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

METRICS = [
    "blipscore",
    "clipscore.OpenClip",
    "vqascore.VqaScore",
    "vqascore.VqaDirect",
]

# Paper-facing display names keyed by the raw method/metric/ablation names.
_METRIC_NAMES = {
    "blipscore": "BLIPScore",
    "clipscore.OpenClip": "CLIPScore",
    "vqascore.VqaScore": "VQAScore",
    "vqascore.VqaDirect": "MLLM Judge",
}
_BASELINE_NAMES = {
    "intrinsix": "IntrinsiX",
    "matfuse": "MatFuse",
    "stablematerials": "StableMaterials",
}
# Ours rows in paper order.
_OURS_NAMES = (
    ("gemma", "Ours (gemma-4-26b-a4b-it)"),
    ("qwen", "Ours (qwen3.6-35b-a3b)"),
    ("deepseek", "Ours (deepseek-v4-flash-0731)"),
    ("glm", "Ours (glm-5.2)"),
    ("gpt", "Ours (gpt-5.6-luna)"),
    ("gemini", "Ours (gemini-3.6-flash)"),
)
# Ablation rows in the README order, grouped as prompt aids / critic signals /
# critic models; the group index only drives \midrule placement.
_ABLATIONS = (
    ("no-playbook", "w/o playbook", 0),
    ("no-few-shot", "w/o few-shot", 0),
    ("no-playbook_no-few-shot", "w/o playbook + few-shot", 0),
    ("no-critic-render", "w/o render critique", 1),
    ("no-critic-code", "w/o code critique", 1),
    ("no-critic-diagnostics", "w/o diagnostics", 1),
    ("no-critic-render_no-critic-code", "w/o render + code critique", 1),
    (
        "no-critic-render_no-critic-diagnostics",
        "w/o render critique + diagnostics",
        1,
    ),
    (
        "no-critic-code_no-critic-diagnostics",
        "w/o code critique + diagnostics",
        1,
    ),
    ("critic-model-gemini", "critic = Gemini", 2),
    ("critic-model-glm", "critic = GLM", 2),
    ("critic-model-deepseek", "critic = DeepSeek", 2),
    ("critic-model-qwen", "critic = Qwen", 2),
    ("critic-model-gemma", "critic = Gemma", 2),
)
_LAYOUTS = ("plane", "scene")


def _warn(message: str) -> None:
    print(f"print_tables: {message}", file=sys.stderr)


def latest_json(results_dir: Path, pattern: str) -> Path | None:
    """Newest JSON matching ``pattern`` under ``results_dir``, or None."""
    matches = sorted(results_dir.glob(pattern))
    return matches[-1] if matches else None


def load_result(path: Path) -> dict | None:
    """Load a result JSON, tolerating the truncated files of partial runs.

    Unreadable files (missing, a directory, permission denied, or truncated
    mid-write — including mid-codepoint UTF-8) warn and skip the family.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        _warn(f"{path.name}: unreadable ({e}); family omitted")
        return None
    return data


def final_scores(
    res: dict, layout: str, metric: str, idx: dict | None = None
) -> dict[str, dict[str, float]]:
    """Per-prompt per-seed score of each run's final program.

    Final-family results store one score per seed; all-family results store
    one 0-based list per seed, whose ``idx`` entry names the final program.
    Unresolvable entries (partial runs) are skipped.
    """
    out: dict[str, dict[str, float]] = {}
    for prompt, seeds in res.get(layout, {}).get(metric, {}).items():
        row: dict[str, float] = {}
        for seed, entries in seeds.items():
            if idx is None:
                row[seed] = entries["score"]
                continue
            i = idx.get(prompt, {}).get(seed, {}).get("final")
            if i is not None and i < len(entries):
                row[seed] = entries[i]["score"]
        if row:
            out[prompt] = row
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _prompt_mean(
    per_prompt: dict[str, dict[str, float]], where: str
) -> float | None:
    """Mean over prompts of the per-prompt seed mean, warning when empty."""
    if not per_prompt:
        _warn(f"{where}: no scores (partial run?)")
        return None
    return _mean(
        [_mean(list(seeds.values())) for seeds in per_prompt.values()]
    )


def _pair(a: float | None, b: float | None) -> str:
    """``a/b`` at two decimals; a missing side renders as ``--``."""
    if a is None and b is None:
        return "--"
    left = "--" if a is None else f"{a:.2f}"
    right = "--" if b is None else f"{b:.2f}"
    return f"{left}/{right}"


def _signed_pair(a: float | None, b: float | None) -> str:
    if a is None and b is None:
        return "--"
    left = "--" if a is None else f"{a:+.2f}"
    right = "--" if b is None else f"{b:+.2f}"
    return f"{left}/{right}"


def _half(value: float | None, bold: bool, signed: bool = False) -> str:
    """One half of a plane/scene cell, wrapped in ``\\textbf`` when bold."""
    if value is None:
        return "--"
    text = f"{value:+.2f}" if signed else f"{value:.2f}"
    return rf"\textbf{{{text}}}" if bold else text


def _row(label: str, cells: list[str]) -> str:
    return f"    {label} & " + " & ".join(cells) + r" \\"


def _tabular(header: list[str], body: list[str]) -> str:
    lines = [
        r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def _all_res(data: dict | None) -> tuple[str | None, dict | None]:
    """Unwrap an ours_main_all JSON to ``(generator name, its result dict)``.

    The scorer writes a single-generator wrapper (``{"gemini": {...}}``); a
    bare generator dict is accepted with an unknown name, and any other shape
    warns and yields nothing.
    """
    if data is None:
        return None, None
    if "plane" in data or "scene" in data or "idx" in data:
        return None, data
    wrapped = [
        key
        for key, value in data.items()
        if isinstance(value, dict) and ("plane" in value or "scene" in value)
    ]
    if len(wrapped) == 1:
        return wrapped[0], data[wrapped[0]]
    _warn("ours_main_all: unexpected JSON shape; family ignored")
    return None, None


def _runs(res: dict, layout: str, metric: str):
    """Yield ``(prompt, seed, entries)`` for every scored program list."""
    for prompt, seeds in res.get(layout, {}).get(metric, {}).items():
        for seed, entries in seeds.items():
            yield prompt, seed, entries


def _total_runs(res: dict) -> int:
    """Run count of the fullest (layout, metric) cell, as a coverage baseline."""
    for layout in _LAYOUTS:
        for metric in METRICS:
            per_prompt = res.get(layout, {}).get(metric, {})
            if per_prompt:
                return sum(len(seeds) for seeds in per_prompt.values())
    return 0


def _main_pairs(
    res: dict, label: str, idx: dict | None = None
) -> list[tuple[float | None, float | None]] | None:
    """One ``(plane, scene)`` seed-mean pair per metric for one method row."""
    pairs = []
    scored = False
    for metric in METRICS:
        pair: list[float | None] = []
        for layout in _LAYOUTS:
            per_prompt = final_scores(res, layout, metric, idx)
            where = f"main/{label}/{layout}/{metric}"
            pair.append(_prompt_mean(per_prompt, where))
            scored = scored or bool(per_prompt)
        pairs.append((pair[0], pair[1]))
    return pairs if scored else None


def main_table(
    baselines: dict | None,
    final: dict | None = None,
    all_data: dict | None = None,
) -> str | None:
    """Main comparison: baselines first, then the Ours rows.

    Every cell is a one-line ``plane/scene`` pair of seed means: the mean over
    prompts of the per-prompt seed mean. Each generator renders from the final
    family, except the all-family generator (gemini), whose entry it overrides
    so its seeds use their ``idx.final`` programs. MobileCLIP2 seed selection
    is deliberately not shown here; the selector tables measure it in
    isolation. The best value of each column half (baselines and Ours in one
    pool) is bolded.
    """
    entries: list[tuple[str, list[tuple[float | None, float | None]]]] = []
    for method in sorted(baselines or {}):
        name = _BASELINE_NAMES.get(method, method)
        pairs = _main_pairs(baselines[method], name)  # type: ignore[index]
        if pairs:
            entries.append((name, pairs))
    all_gen, all_res = _all_res(all_data)
    for key, label in _OURS_NAMES:
        if key == all_gen and all_res is not None:
            pairs = _main_pairs(all_res, label, idx=all_res.get("idx"))
        else:
            res = (final or {}).get(key)
            if res is None:
                continue
            pairs = _main_pairs(res, label)
        if pairs:
            entries.append((label, pairs))
    if not entries:
        return None
    bests: list[tuple[float | None, float | None]] = []
    for mi in range(len(METRICS)):
        col_best: list[float | None] = []
        for j in (0, 1):
            vals = [
                pairs[mi][j]
                for _, pairs in entries
                if pairs[mi][j] is not None
            ]
            col_best.append(max(vals) if vals else None)
        bests.append((col_best[0], col_best[1]))
    body = []
    for i, (label, pairs) in enumerate(entries):
        cells = []
        for mi, (best_a, best_b) in enumerate(bests):
            a, b = pairs[mi]
            cells.append(f"{_half(a, a == best_a)}/{_half(b, b == best_b)}")
        body.append(_row(label, cells))
    header = ["Method", *(_METRIC_NAMES[m] for m in METRICS)]
    return _tabular(header, body)


def refinement_table(all_data: dict | None) -> str | None:
    """Refinement rounds: round 0 absolute, rounds 1..5 as deltas from it.

    Each cell is a mean trajectory-program score over every (prompt, seed) run
    of the all-family generator. Rows 1..5 restate theirs as the signed change
    from round 0, so the table reads as the refinement effect itself; a side
    renders as ``--`` when either round's mean is unresolvable. The best round
    of each column half is bolded: its cell still shows the row's own value
    (the absolute for round 0, the delta otherwise), and round 0 winning a
    column means refinement never beat the initial program there.
    """
    gen, res = _all_res(all_data)
    if res is None:
        return None
    total = _total_runs(res)
    if total == 0:
        _warn(f"{gen or 'ours_main_all'}: no scored runs (partial run?)")
    # means[round][metric][layout] -> that round's mean, or None (partial run).
    means: list[list[list[float | None]]] = []
    for i in range(6):
        row: list[list[float | None]] = []
        for metric in METRICS:
            pair: list[float | None] = []
            for layout in _LAYOUTS:
                vals = [
                    entries[i]["score"]
                    for _, _, entries in _runs(res, layout, metric)
                    if i < len(entries)
                ]
                if i == 0 and layout == "plane" and len(vals) < total:
                    _warn(
                        f"refinement/{metric}: {len(vals)}/{total} runs (partial?)"
                    )
                pair.append(_mean(vals))
            row.append(pair)
        means.append(row)
    # bests[metric][layout] -> the best round's mean, whose cell gets bolded.
    bests: list[list[float | None]] = []
    for mi in range(len(METRICS)):
        per_layout: list[float | None] = []
        for j in range(len(_LAYOUTS)):
            vals = [
                means[i][mi][j]
                for i in range(6)
                if means[i][mi][j] is not None
            ]
            per_layout.append(max(vals) if vals else None)
        bests.append(per_layout)
    body = []
    for i in range(6):
        cells = []
        for mi in range(len(METRICS)):
            halves = []
            for j in range(len(_LAYOUTS)):
                base, cur = means[0][mi][j], means[i][mi][j]
                if i == 0:
                    halves.append(_half(cur, cur == bests[mi][j]))
                else:
                    delta = None if base is None or cur is None else cur - base
                    halves.append(
                        _half(delta, cur == bests[mi][j], signed=True)
                    )
            cells.append("/".join(halves))
        body.append(_row("0 (base)" if i == 0 else str(i), cells))
    header = ["Refinements", *(_METRIC_NAMES[m] for m in METRICS)]
    return _tabular(header, body)


def polish_table(all_data: dict | None) -> str | None:
    """Signed mean deltas from polishing first/chosen/last, over all runs.

    ``chosen`` is the mobileclip2-selected trajectory program (``idx.before``);
    when it coincides with the first or last program (length-8 lists) its
    comparison is that program's own polish delta. The final ``Mean`` row
    averages the three rows, i.e. the mean polish effect over the three
    starting points.
    """
    gen, res = _all_res(all_data)
    if res is None:
        return None
    total = _total_runs(res)
    if total == 0:
        _warn(f"{gen or 'ours_main_all'}: no scored runs (partial run?)")
    labels = ("First", "Chosen", "Last", "Mean")
    # deltas[metric][row][layout] -> mean delta, or None when unresolvable.
    deltas = [[[None for _ in _LAYOUTS] for _ in labels] for _ in METRICS]
    for mi, metric in enumerate(METRICS):
        for j, layout in enumerate(_LAYOUTS):
            fs, cs, ls = [], [], []
            for prompt, seeds in res.get("idx", {}).items():
                for seed, meta in seeds.items():
                    entries = (
                        res.get(layout, {})
                        .get(metric, {})
                        .get(prompt, {})
                        .get(seed)
                    )
                    b = meta.get("before")
                    if (
                        entries is None
                        or len(entries) < 8
                        or b is None
                        or b > 5
                    ):
                        continue
                    fs.append(entries[6]["score"] - entries[0]["score"])
                    if b == 0:
                        cs.append(entries[6]["score"] - entries[0]["score"])
                    elif b == 5:
                        cs.append(entries[7]["score"] - entries[5]["score"])
                    else:
                        cs.append(entries[7]["score"] - entries[b]["score"])
                    tail = 7 if len(entries) == 8 else 8
                    ls.append(entries[tail]["score"] - entries[5]["score"])
            if layout == "plane" and len(fs) < total:
                _warn(f"polish/{metric}: {len(fs)}/{total} runs (partial?)")
            deltas[mi][0][j] = _mean(fs)
            deltas[mi][1][j] = _mean(cs)
            deltas[mi][2][j] = _mean(ls)
            row_means = [deltas[mi][r][j] for r in range(3)]
            deltas[mi][3][j] = _mean([v for v in row_means if v is not None])
    body = []
    for r, label in enumerate(labels):
        row_cells = [
            _signed_pair(deltas[m][r][0], deltas[m][r][1])
            for m in range(len(METRICS))
        ]
        body.append(_row(label, row_cells))
    header = ["Polish", *(_METRIC_NAMES[m] for m in METRICS)]
    return _tabular(header, body)


def selector_table(all_data: dict | None) -> str | None:
    """Mean scores of first/pick/last trajectory programs plus oracle best."""
    gen, res = _all_res(all_data)
    if res is None:
        return None
    total = _total_runs(res)
    if total == 0:
        _warn(f"{gen or 'ours_main_all'}: no scored runs (partial run?)")
    labels = ("First", "MobileCLIP2 pick", "Last", "Oracle best")
    # values[metric][row][layout] -> that row's per-run program scores.
    values = [[[[] for _ in _LAYOUTS] for _ in labels] for _ in METRICS]
    for mi, metric in enumerate(METRICS):
        for j, layout in enumerate(_LAYOUTS):
            for prompt, seeds in res.get("idx", {}).items():
                for seed, meta in seeds.items():
                    entries = (
                        res.get(layout, {})
                        .get(metric, {})
                        .get(prompt, {})
                        .get(seed)
                    )
                    b = meta.get("before")
                    if (
                        entries is None
                        or len(entries) < 6
                        or b is None
                        or b > 5
                    ):
                        continue
                    values[mi][0][j].append(entries[0]["score"])
                    values[mi][1][j].append(entries[b]["score"])
                    values[mi][2][j].append(entries[5]["score"])
                    values[mi][3][j].append(
                        max(e["score"] for e in entries[:6])
                    )
            if layout == "plane" and len(values[mi][0][j]) < total:
                _warn(
                    f"selector/{metric}: "
                    f"{len(values[mi][0][j])}/{total} runs (partial?)"
                )
    body = []
    for r, label in enumerate(labels):
        row_cells = [
            _pair(_mean(values[m][r][0]), _mean(values[m][r][1]))
            for m in range(len(METRICS))
        ]
        body.append(_row(label, row_cells))
    header = ["Program", *(_METRIC_NAMES[m] for m in METRICS)]
    return _tabular(header, body)


def winrate_table(all_data: dict | None) -> str | None:
    """MobileCLIP2 pick vs the last program: win rate and mean delta."""
    gen, res = _all_res(all_data)
    if res is None:
        return None
    if _total_runs(res) == 0:
        _warn(f"{gen or 'ours_main_all'}: no scored runs (partial run?)")
    rates: list[list[float | None]] = [
        [None for _ in _LAYOUTS] for _ in METRICS
    ]
    deltas: list[list[float | None]] = [
        [None for _ in _LAYOUTS] for _ in METRICS
    ]
    for mi, metric in enumerate(METRICS):
        for j, layout in enumerate(_LAYOUTS):
            wins, diffs = [], []
            for prompt, seeds in res.get("idx", {}).items():
                for seed, meta in seeds.items():
                    entries = (
                        res.get(layout, {})
                        .get(metric, {})
                        .get(prompt, {})
                        .get(seed)
                    )
                    b = meta.get("before")
                    if (
                        entries is None
                        or len(entries) < 6
                        or b is None
                        or b > 5
                    ):
                        continue
                    pick, last = entries[b]["score"], entries[5]["score"]
                    wins.append(pick > last)
                    diffs.append(pick - last)
            rates[mi][j] = 100.0 * sum(wins) / len(wins) if wins else None
            deltas[mi][j] = _mean(diffs)
    body = [
        _row(
            r"Win rate (\%)",
            [
                "--"
                if rates[m][0] is None or rates[m][1] is None
                else f"{rates[m][0]:.1f}/{rates[m][1]:.1f}"
                for m in range(len(METRICS))
            ],
        ),
        _row(
            "Mean delta",
            [
                _signed_pair(deltas[m][0], deltas[m][1])
                for m in range(len(METRICS))
            ],
        ),
    ]
    header = ["Pick vs. last", *(_METRIC_NAMES[m] for m in METRICS)]
    return _tabular(header, body)


def _full_row(final: dict | None, all_data: dict | None) -> dict | None:
    """Absolute seed-42 full-configuration means for the gpt generator.

    The ablations are gpt seed-42 runs, so the Full row reads the gpt
    generator's seed-42 finals from the final family, falling back to a gpt
    all-family JSON (whose ``idx`` names each seed's final program).
    """
    res = (final or {}).get("gpt")
    idx = None
    if res is None:
        all_gen, all_res = _all_res(all_data)
        if all_gen == "gpt" and all_res is not None:
            res, idx = all_res, all_res.get("idx")
    if res is None:
        _warn("ablations: no gpt seed-42 finals; Full (ours) row omitted")
        return None
    scores: dict[tuple[str, str], float | None] = {}
    prompts: set[str] | None = None
    for layout in _LAYOUTS:
        for metric in METRICS:
            fs = final_scores(res, layout, metric, idx)
            seed42 = {p: s["42"] for p, s in fs.items() if "42" in s}
            if prompts is not None and len(seed42) < len(prompts):
                _warn(
                    f"ablations/full/{layout}/{metric}: "
                    f"{len(seed42)}/{len(prompts)} prompts (partial?)"
                )
            scores[(layout, metric)] = _mean(list(seed42.values()))
            if prompts is None:
                prompts = set(seed42)
    return {"prompts": prompts, "scores": scores}


def ablation_table(
    abl: dict | None, final: dict | None = None, all_data: dict | None = None
) -> str | None:
    """Ablation rows as signed deltas against the full GPT seed-42 run.

    The Full row is omitted (and the ablations fall back to absolute means)
    unless the GPT seed-42 prompt set matches the ablations exactly.
    """
    if not abl:
        return None
    full = _full_row(final, all_data)
    ref_prompts: set[str] | None = None
    for key in abl:
        for metric in METRICS:
            prompts = set(abl[key].get("plane", {}).get(metric, {}))
            if prompts:
                ref_prompts = prompts
                break
        if ref_prompts is not None:
            break
    if full is not None and full["prompts"] != ref_prompts:
        n_ref = len(ref_prompts) if ref_prompts is not None else 0
        _warn(
            f"ablations: Full (ours) row omitted (gpt seed-42 has "
            f"{len(full['prompts'])} prompts, ablations have {n_ref})"
        )
        full = None
    known = {key: (label, group) for key, label, group in _ABLATIONS}
    unknown = sorted(key for key in abl if key not in known)
    if unknown:
        _warn(f"ablations: unknown keys appended: {', '.join(unknown)}")
    keys = [key for key, _, _ in _ABLATIONS if key in abl] + unknown
    body = []
    if full is not None:
        cells = [
            _pair(full["scores"][("plane", m)], full["scores"][("scene", m)])
            for m in METRICS
        ]
        body.append(_row("Full (ours)", cells))
    prev_group = (
        -1 if full is not None else None
    )  # -1: rule after the Full row
    for key in keys:
        label, group = known.get(key, (key, None))
        if body and group != prev_group:
            body.append(r"    \midrule")
        cells = []
        for metric in METRICS:
            pair = []
            for layout in _LAYOUTS:
                vals = abl[key].get(layout, {}).get(metric, {})
                if ref_prompts is not None and len(vals) < len(ref_prompts):
                    _warn(
                        f"ablations/{label}/{layout}/{metric}: "
                        f"{len(vals)}/{len(ref_prompts)} prompts (partial?)"
                    )
                mu = _mean(list(vals.values()))
                if full is not None:
                    base = full["scores"][(layout, metric)]
                    pair.append(
                        None if mu is None or base is None else mu - base
                    )
                else:
                    pair.append(mu)
            cells.append(
                _signed_pair(*pair) if full is not None else _pair(*pair)
            )
        body.append(_row(label, cells))
        prev_group = group
    header = ["Ablation", *(_METRIC_NAMES[m] for m in METRICS)]
    return _tabular(header, body)


def significance_table(
    baselines: dict | None,
    final: dict | None = None,
    all_data: dict | None = None,
) -> str | None:
    """Flagship generator vs. each baseline: mean delta, bootstrap CI, Wilcoxon p.

    Pairs are per-prompt seed means of the final program (the all-family
    generator's seeds use their ``idx.final`` programs). One row per baseline
    and layout; each cell reads ``+<delta> [<lo>, <hi>] p=<p>`` for the mean
    delta (ours minus baseline), a 95% percentile bootstrap CI over 10,000
    prompt-level resamples, and the paired two-sided Wilcoxon signed-rank p.
    A metric with no shared prompts renders as ``--``.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        _warn("significance: scipy unavailable; table omitted")
        return None
    import random

    _, all_res = _all_res(all_data)
    ours_res, ours_idx = all_res, (all_res or {}).get("idx")
    if ours_res is None:
        if not final:
            return None
        # Fall back to the first final-family generator present.
        key = next(iter(final))
        ours_res = final[key]
        _warn(f"significance: no all-family JSON; using generator {key!r}")

    def _cell(layout: str, metric: str, base: dict) -> str:
        ours_pp = final_scores(ours_res, layout, metric, ours_idx)  # type: ignore[arg-type]
        base_pp = final_scores(base, layout, metric)
        shared = sorted(set(ours_pp) & set(base_pp))
        if not shared:
            return "--"
        deltas = [
            _mean(list(ours_pp[p].values())) - _mean(list(base_pp[p].values()))
            for p in shared
        ]
        try:
            p_value: float | None = wilcoxon(deltas).pvalue
        except ValueError:  # all-zero deltas
            p_value = 1.0
        rng = random.Random(0)
        boots = sorted(
            _mean([deltas[rng.randrange(len(deltas))] for _ in deltas])
            for _ in range(10000)
        )
        lo, hi = boots[249], boots[9749]
        p_text = (
            "p<1e-4"
            if p_value is not None and p_value < 1e-4
            else f"p={p_value:.2g}"
        )
        return f"{_mean(deltas):+.1f} [{lo:+.1f},{hi:+.1f}] {p_text}"

    body = []
    for method in sorted(baselines or {}):
        name = _BASELINE_NAMES.get(method, method)
        for j, layout in enumerate(_LAYOUTS):
            label = name if j == 0 else f"{name} (staged)"
            cells = [_cell(layout, m, baselines[method]) for m in METRICS]  # type: ignore[index]
            body.append(_row(label, cells))
    if not body:
        return None
    header = ["Ours vs.", *(_METRIC_NAMES[m] for m in METRICS)]
    return _tabular(header, body)


# ---------------------------------------------------------------------------
# Run-tree statistics (record.json / trajectory.xml), not the result JSONs.


def _iter_records(runs_dir: Path, backbone: str | None = None):
    """Yield ``(backbone, record, path)`` for every main run under ``runs_dir``."""
    if not runs_dir.is_dir():
        _warn(f"runs directory not found: {runs_dir}")
        return
    for path in sorted(runs_dir.glob("*/seed*/*/*/record.json")):
        fam = path.relative_to(runs_dir).parts[0]
        if backbone is not None and fam != backbone:
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            _warn(f"unreadable record: {path}")
            continue
        yield fam, record, path


def _final_program_text(traj_path: Path) -> str | None:
    """The last ``<program>`` of a trajectory.xml, i.e. the text the eval
    scripts score (``eval_ours_*.py`` read the same node)."""
    try:
        root = ET.parse(traj_path).getroot()
    except (OSError, ET.ParseError):
        return None
    programs = root.findall(".//program")
    if not programs or programs[-1].text is None:
        return None
    return programs[-1].text.strip()


def polish_budget_table(
    runs_dir: Path,
    backbone: str = "gemini",
    budgets: tuple[int, ...] = (1, 10, 50, 100, 250, 500, 1000),
) -> str | None:
    """Quick-scorer gain vs. polish sweep budget, from stored per-variant scores.

    Every run polishes up to three programs (the trajectory's first, the
    mobileclip2-chosen, and the last), sweeping ``seed_sweep`` re-seeds plus
    the original (``candidates[0]``, whose score equals ``initial_score``),
    and keeps the best accept-if-better result over the entries
    (``matloom/engine/text2dsl.py``). Truncating each sweep to its first ``k``
    re-seeds replays exactly that selection over ``candidates[:k+1]``, so the
    per-run gain at ``k`` is non-negative and non-decreasing in ``k``. All
    scores are quick-scorer (MobileCLIP2) values, not evaluation metrics.
    """
    k_max = max(budgets)
    gains: dict[int, list[float]] = {k: [] for k in budgets}
    at_full: dict[int, int] = {k: 0 for k in budgets}
    n_runs = 0
    baselines: list[float] = []
    for _, record, path in _iter_records(runs_dir, backbone):
        entries = [
            e for e in (record.get("polish") or []) if e.get("candidates")
        ]
        if not entries:
            continue
        n_runs += 1
        best_prefix = {k: -math.inf for k in budgets}
        for entry in entries:
            scores = [c["scores"][0] for c in entry["candidates"]]
            if abs(max(scores) - entry["final_score"]) > 1e-6:
                _warn(
                    f"{path}: polish[{entry.get('index')}] final_score != sweep max"
                )
            for k in budgets:
                best_prefix[k] = max(best_prefix[k], max(scores[: k + 1]))
        baseline = max(e["initial_score"] for e in entries)
        baselines.append(baseline)
        for k in budgets:
            gains[k].append(best_prefix[k] - baseline)
            if best_prefix[k] >= best_prefix[k_max] - 1e-9:
                at_full[k] += 1
    if n_runs == 0:
        _warn(f"no polish records under {runs_dir} for {backbone}")
        return None
    full_gain = _mean(gains[k_max]) or 0.0
    header = [
        "Budget $k$",
        "Mean gain",
        f"\\% of $k{{=}}{k_max}$",
        "Runs at best (\\%)",
    ]
    body = []
    for k in budgets:
        gain = _mean(gains[k]) or 0.0
        share = (100.0 * gain / full_gain) if full_gain > 0 else 0.0
        body.append(
            _row(
                f"${k}$",
                [
                    f"{gain:+.2f}",
                    f"{share:.0f}" if k != k_max else "100",
                    f"{100.0 * at_full[k] / n_runs:.0f}",
                ],
            )
        )
    note = (
        f"% polish-budget ({backbone}, n={n_runs} runs; quick-scorer gain over the "
        f"best pre-polish program, median baseline {statistics.median(baselines):.2f})"
    )
    return note + "\n" + _tabular(header, body)


def program_stats_table(runs_dir: Path) -> str | None:
    """Median size of the final program per backbone and pooled.

    The final program is the last ``<program>`` of ``trajectory.xml`` (the
    text the eval scripts score). Lines count the program's own newlines;
    size is UTF-8 bytes.
    """
    sizes: dict[str, list[tuple[int, int]]] = {}
    for fam, _, path in _iter_records(runs_dir):
        text = _final_program_text(path.parent / "trajectory.xml")
        if text is None:
            _warn(f"no final program in {path.parent / 'trajectory.xml'}")
            continue
        sizes.setdefault(fam, []).append(
            (text.count("\n") + 1, len(text.encode()))
        )
    if not sizes:
        return None
    rows = []
    pool: list[tuple[int, int]] = []

    def _cells(vals: list[tuple[int, int]]) -> list[str]:
        lines = statistics.median(v[0] for v in vals)
        nbytes = statistics.median(v[1] for v in vals)
        return [f"{lines:.0f}", f"{nbytes:.0f}", f"{nbytes / 1024:.2f}"]

    for fam, label in _OURS_NAMES:
        vals = sizes.get(fam)
        if not vals:
            continue
        pool += vals
        rows.append(_row(label, _cells(vals)))
    rows.append(_row("Pooled (all six)", _cells(pool)))
    return _tabular(["Backbone", "Median lines", "Median bytes", "KB"], rows)


def runtime_table(runs_dir: Path) -> str | None:
    """Per-backbone wall-clock: median total minutes and median polish share."""
    totals: dict[str, list[float]] = {}
    shares: dict[str, list[float]] = {}
    polish_s: dict[str, list[float]] = {}
    for fam, record, _ in _iter_records(runs_dir):
        duration = record.get("duration_seconds") or {}
        total = duration.get("total")
        polish = duration.get("polish") or []
        if not isinstance(total, (int, float)) or not polish:
            continue
        totals.setdefault(fam, []).append(float(total))
        polish_s.setdefault(fam, []).append(float(sum(polish)))
        shares.setdefault(fam, []).append(float(sum(polish)) / float(total))
    if not totals:
        return None
    rows = []
    for fam, label in _OURS_NAMES:
        if fam not in totals:
            continue
        t = totals[fam]
        p90 = (
            statistics.quantiles(polish_s[fam], n=10)[8]
            if len(t) >= 10
            else None
        )
        rows.append(
            _row(
                label,
                [
                    f"{statistics.median(t) / 60:.1f}",
                    f"{100 * statistics.median(shares[fam]):.0f}",
                    f"{p90 / 60:.1f}" if p90 is not None else "--",
                ],
            )
        )
    return _tabular(
        [
            "Backbone",
            "Median total (min)",
            "Polish share (\\%)",
            "Polish p90 (min)",
        ],
        rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("experiments") / "results"
    )
    parser.add_argument(
        "--baselines", type=Path, help="override the baselines JSON"
    )
    parser.add_argument(
        "--all", type=Path, help="override the ours_main_all JSON"
    )
    parser.add_argument(
        "--final", type=Path, help="override the ours_main_final JSON"
    )
    parser.add_argument(
        "--ablations", type=Path, help="override the ablations JSON"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments") / "runs" / "ours" / "main",
        help="run tree for polish-budget / program-stats / runtime",
    )
    parser.add_argument(
        "--budgets",
        default="1,10,50,100,250,500,1000",
        help="comma-separated polish sweep budgets for --table polish-budget",
    )
    parser.add_argument(
        "--backbone",
        default="gemini",
        help="backbone family for --table polish-budget (default gemini)",
    )
    parser.add_argument(
        "--table",
        default="all",
        choices=(
            "all",
            "main",
            "refinement",
            "polish",
            "selector",
            "winrate",
            "ablations",
            "significance",
            "polish-budget",
            "program-stats",
            "runtime",
        ),
    )
    args = parser.parse_args(argv)

    cache: dict[str, dict | None] = {}

    def family(
        name: str, flag: Path | None, pattern: str, missing: str
    ) -> dict | None:
        if name not in cache:
            path = (
                flag
                if flag is not None
                else latest_json(args.results_dir, pattern)
            )
            data = load_result(path) if path is not None else None
            if data is None:
                _warn(f"no usable {pattern}; {missing}")
            cache[name] = data
        return cache[name]

    def want(table: str) -> bool:
        return args.table in ("all", table)

    if want("main"):
        table = main_table(
            family(
                "baselines",
                args.baselines,
                "baselines*.json",
                "baseline rows omitted from the main table",
            ),
            final=family(
                "final",
                args.final,
                "ours_main_final*.json",
                "Ours generator rows omitted",
            ),
            all_data=family(
                "all",
                args.all,
                "ours_main_all*.json",
                "the all-family row and the refinement/polish/selector tables omitted",
            ),
        )
        if table:
            print(
                "% main results (per cell: plane mean / scene mean; "
                "bold = best of each column half)"
            )
            print(table)
    if want("refinement"):
        table = refinement_table(
            family(
                "all",
                args.all,
                "ours_main_all*.json",
                "refinement table omitted",
            )
        )
        if table:
            print(
                "% refinement rounds (row 0 absolute; rows 1--5 signed delta vs "
                "row 0; per cell: plane / scene; bold = best round)"
            )
            print(table)
    if want("polish"):
        table = polish_table(
            family(
                "all", args.all, "ours_main_all*.json", "polish table omitted"
            )
        )
        if table:
            print("% polish deltas (per cell: plane delta / scene delta)")
            print(table)
    if want("selector"):
        table = selector_table(
            family(
                "all",
                args.all,
                "ours_main_all*.json",
                "selector table omitted",
            )
        )
        if table:
            print("% selector: mean scores (per cell: plane / scene)")
            print(table)
    if want("winrate"):
        table = winrate_table(
            family(
                "all",
                args.all,
                "ours_main_all*.json",
                "win-rate table omitted",
            )
        )
        if table:
            print(
                "% selector: mobileclip2 pick vs last (per cell: plane / scene)"
            )
            print(table)
    if want("ablations"):
        table = ablation_table(
            family(
                "ablations",
                args.ablations,
                "ours_ablations*.json",
                "ablation table omitted",
            ),
            final=family(
                "final",
                args.final,
                "ours_main_final*.json",
                "the Full (ours) row falls back to a gpt all-family JSON",
            ),
            all_data=family(
                "all",
                args.all,
                "ours_main_all*.json",
                "the Full (ours) row falls back to the final family's gpt entry",
            ),
        )
        if table:
            print(
                "% ablations (per cell: plane / scene; Full absolute, rest delta)"
            )
            print(table)
    if want("significance"):
        table = significance_table(
            family(
                "baselines",
                args.baselines,
                "baselines*.json",
                "significance table omitted",
            ),
            final=family(
                "final",
                args.final,
                "ours_main_final*.json",
                "the significance rows fall back to a final-family generator",
            ),
            all_data=family(
                "all",
                args.all,
                "ours_main_all*.json",
                "the significance rows fall back to the final family",
            ),
        )
        if table:
            print(
                "% significance (flagship generator vs baselines; delta [95% CI] Wilcoxon p)"
            )
            print(table)
    if want("polish-budget"):
        budgets = tuple(
            int(k) for k in str(args.budgets).split(",") if k.strip()
        )
        if not budgets:
            _warn(
                "--budgets is empty; falling back to the default budget list"
            )
            budgets = (1, 10, 50, 100, 250, 500, 1000)
        table = polish_budget_table(args.runs_dir, args.backbone, budgets)
        if table:
            print(table)
    if want("program-stats"):
        table = program_stats_table(args.runs_dir)
        if table:
            print(
                "% program stats (final program = last <program> of trajectory.xml)"
            )
            print(table)
    if want("runtime"):
        table = runtime_table(args.runs_dir)
        if table:
            print(
                "% runtime (per run: median total, median polish share, polish p90)"
            )
            print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
