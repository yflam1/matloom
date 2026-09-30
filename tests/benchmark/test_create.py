"""Tests for matloom.benchmark.create — diverse prompt selection.

All pure: embeddings are fakes (no torch / network). The selection logic is
exercised through :func:`select_diverse` with an injected ``encode`` callable,
the same seam ``create_benchmark`` uses for the real Sentence-Bert embedder.
"""

from __future__ import annotations

import numpy as np

from matloom.benchmark.create import select_diverse


# --------------------------------------------------------------------------- #
# Farthest-point sampling (via select_diverse with a fixed embedding table)
# --------------------------------------------------------------------------- #
def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_select_diverse_picks_spread_out_points():
    # Four prompts whose embeddings are two near-duplicates and two far apart.
    # From start=0, FPS should avoid the near-duplicate and reach the far ones.
    emb = _norm(
        [
            [1.0, 0.0],
            [0.99, 0.01],  # near-duplicate of row 0
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    prompts = [f"p{i}" for i in range(len(emb))]

    def encode(texts):
        return emb

    chosen = select_diverse(prompts, 3, encode=encode, start=0)

    assert chosen[0] == "p0"
    assert "p1" not in chosen  # the near-duplicate is the last thing FPS would add
    assert set(chosen) == {"p0", "p2", "p3"}


def test_select_diverse_deterministic_given_start():
    rng = np.random.default_rng(0)
    emb = _norm(rng.standard_normal((20, 8)))
    prompts = [f"p{i}" for i in range(20)]

    def encode(texts):
        return emb

    a = select_diverse(prompts, 6, encode=encode, start=3)
    b = select_diverse(prompts, 6, encode=encode, start=3)

    assert a == b
    assert a[0] == "p3"
    assert len(set(a)) == 6  # no repeats


def test_select_diverse_k_geq_n_returns_all():
    prompts = ["p0", "p1"]
    emb = _norm([[1.0, 0.0], [0.0, 1.0]])

    def encode(texts):
        return emb

    # k > n: every prompt is selected (in selection order, not original order).
    assert set(select_diverse(prompts, 5, encode=encode, start=0)) == {
        "p0",
        "p1",
    }
    # k == 0: nothing selected.
    assert select_diverse(prompts, 0, encode=encode, start=0) == []


def test_select_diverse_empty_prompts():
    emb = _norm([[1.0, 0.0]])

    def encode(texts):
        return emb

    assert select_diverse([], 3, encode=encode, start=0) == []


def test_select_diverse_uses_encoder_and_count():
    # Encoder spreads the points on a line so FPS picks endpoints first.
    prompts = [f"text {i}" for i in range(10)]

    def encode(texts):
        return _norm([[float(i), 1.0] for i in range(len(texts))])

    chosen = select_diverse(prompts, 4, encode=encode, start=0)

    assert len(chosen) == 4
    assert len(set(chosen)) == 4
