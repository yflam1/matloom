"""Tests for matloom.engine.polish — tunable extraction, the fixed-light preview,
and the CMA-ES driver. The optimizer tests skip when ``cma`` is not installed;
the extraction/preview tests are pure numpy + engine (no torch, no Blender)."""

import os

import numpy as np
import pytest

from matloom.engine.main import LayeredMaterial
from matloom.engine.polish import (
    _SEED_RE,
    _apply_values,
    _extract_tunables,
    _seed_variants,
    polish_program,
)
from matloom.engine.preview import fixed_light_preview

_BRICK = (
    "View(0, 0, 6, 6)\n"
    "Define(brick, Bricks(brick_width=1, brick_height=0.45, mortar=0.06))\n"
    "Define(tone, fBm(octaves=4, base_freq=2.5, to_01=True, seed=7))\n"
    "Material(\n"
    "  Layer(1).basecolor(175, 170, 160).roughness(0.95),\n"
    "  Layer(brick).basecolor((150 + (tone * 50)), 55, 45).height((0.12 + (tone * 0.05)))\n)"
)
_METAL = (
    "View(0, 0, 1, 1)\nMaterial(\n"
    "  Layer(1).basecolor(180, 182, 188).metallic(1).roughness(0.3)\n)"
)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _origins(program):
    return [t.origin for t in _extract_tunables(program)]


def test_extract_excludes_structural_args():
    o = _origins(_BRICK)
    # seed, octaves, to_01 are excluded; View bounds are excluded.
    assert not any("seed" in x or "octaves" in x or "to_01" in x for x in o)
    assert not any("View" in x for x in o)
    # base_freq is tunable (log space).
    assert any("base_freq" in x for x in o)


def test_extract_has_no_engine_artifacts():
    # The authored text has no 255 conversion divisor and no default channels.
    vals = [_BRICK[t.start : t.end] for t in _extract_tunables(_BRICK)]
    assert "255" not in vals
    # brick_width=1, base_freq=2.5, roughness=0.95 are all present.
    assert "1" in vals and "2.5" in vals and "0.95" in vals


def test_extract_frequencies_use_log_bounds():
    bf = next(t for t in _extract_tunables(_BRICK) if "base_freq" in t.origin)
    assert bf.scale == "log"
    # log10 bounds [-1, 1.7] -> v in [10^-1, 10^1.7] ~= [0.1, 50.1]
    assert abs(bf.from_unit(0.0) - 1e-1) < 1e-9
    assert abs(bf.from_unit(1.0) - 10**1.7) < 1e-3
    assert abs(bf.to_unit(2.5) - (np.log10(2.5) + 1) / 2.7) < 1e-9


def test_apply_is_idempotent_at_original():
    tuns = _extract_tunables(_BRICK)
    x0 = [t.to_unit(float(_BRICK[t.start : t.end])) for t in tuns]
    assert all(0.0 <= x <= 1.0 for x in x0)  # every authored value is in-bounds
    out = _apply_values(_BRICK, x0)
    assert (
        LayeredMaterial.deserialize(out).serialize()
        == LayeredMaterial.deserialize(_BRICK).serialize()
    )


def test_apply_changes_values_and_still_parses():
    tuns = _extract_tunables(_BRICK)
    out = _apply_values(_BRICK, [0.5] * len(tuns))
    LayeredMaterial.deserialize(out)  # must parse
    # base_freq was 2.5; at theta=0.5 in log [-1,1.7] -> 10^(-1 + 0.5*2.7) = 10^0.35
    new_tuns = _extract_tunables(out)
    bf = next(t for t in new_tuns if "base_freq" in t.origin)
    assert abs(float(out[bf.start : bf.end]) - 10**0.35) < 1e-3


def test_apply_channel_values_stay_in_valid_ranges():
    tuns = _extract_tunables(_METAL)
    out = _apply_values(_METAL, [0.5] * len(tuns))
    for t in _extract_tunables(out):
        v = float(out[t.start : t.end])
        assert t.lo <= v <= t.hi, f"{t.origin}={v} out of [{t.lo},{t.hi}]"


def test_apply_at_theta_half_parses_rect_ellipse_stroke():
    # Rect/Ellipse/Stroke have positional args the parser requires > 0
    # (width/height/rx/ry/stroke-width). _apply_values at theta=0.5 must keep
    # the program parseable (the per-position bounds must not drive those to 0).
    prog = (
        "View(0, 0, 1, 1)\n"
        "Define(frame, Stroke(Rect(0.08, 0.08, 0.84, 0.84, 0.04), 0.06, feather=0.005))\n"
        "Define(disc, Fill(Ellipse(0.5, 0.5, 0.3, 0.3)))\n"
        "Material(\n"
        "  Layer(frame).basecolor(180, 182, 188).metallic(1).roughness(0.35),\n"
        "  Layer(disc).basecolor(50, 100, 200).roughness(0.5)\n)"
    )
    tuns = _extract_tunables(prog)
    assert len(tuns) >= 8  # several Rect coords + stroke width + channels
    out = _apply_values(prog, [0.5] * len(tuns))
    LayeredMaterial.deserialize(out)  # must not raise


def test_extract_skips_out_of_bounds_structural_literals():
    # A large hash multiplier (43758.5453) is a structural constant the LLM
    # chose, not a tunable parameter; it must be skipped so _apply_values cannot
    # clamp it and corrupt the program (the default positional bound is wide,
    # but a value beyond it is left untouched).
    prog = (
        "View(0, 0, 1, 1)\n"
        "Define(r0, (Sin((X() * 12.9898)) * 43758.5453))\n"
        "Material(\n  Layer(1).basecolor(150, 150, 150).roughness(0.5)\n)"
    )
    tuns = _extract_tunables(prog)
    vals = [prog[t.start : t.end] for t in tuns]
    assert "43758.5453" not in vals  # the hash multiplier is skipped
    # 12.9898 (inside the default [-1000, 1000] bound) is still tunable.
    assert "12.9898" in vals
    # Idempotent: applying the original theta leaves the program unchanged.
    x0 = [t.to_unit(float(prog[t.start : t.end])) for t in tuns]
    out = _apply_values(prog, x0)
    assert (
        LayeredMaterial.deserialize(out).serialize()
        == LayeredMaterial.deserialize(prog).serialize()
    )


# --------------------------------------------------------------------------- #
# Fixed-light preview
# --------------------------------------------------------------------------- #
def test_preview_shape_dtype_and_finite():
    img = fixed_light_preview(LayeredMaterial.deserialize(_BRICK), 64, 64)
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    assert np.isfinite(img.astype(float)).all()


def test_preview_metal_is_not_black():
    # A flat metal has no diffuse; the ambient fill must keep it visible so the
    # verifier can see the metal's color.
    img = fixed_light_preview(LayeredMaterial.deserialize(_METAL), 64, 64)
    assert img.mean() > 10.0


def test_preview_distinguishes_materials():
    metal = fixed_light_preview(LayeredMaterial.deserialize(_METAL), 64, 64).mean()
    matte = fixed_light_preview(
        LayeredMaterial.deserialize(
            "View(0, 0, 1, 1)\nMaterial(\n  Layer(1).basecolor(150, 40, 40).roughness(0.95)\n)"
        ),
        64,
        64,
    ).mean()
    # A bright metal and a dark-red matte differ in mean luminance.
    assert abs(metal - matte) > 5.0


# --------------------------------------------------------------------------- #
# Seed variants (the discrete seed sweep)
# --------------------------------------------------------------------------- #
def _seeds(prog: str) -> list[int]:
    """All seed= values in source order (duplicates kept)."""
    return [int(m.group(1)) for m in _SEED_RE.finditer(prog)]


def test_seed_variants_first_is_original():
    prog = "View(0,0,1,1)\nDefine(t, fBm(seed=7))\nMaterial(\n  Layer(t)\n)"
    variants = _seed_variants(prog, 4, np.random.default_rng(0))
    assert variants[0] == prog


def test_seed_variants_no_seed_returns_original_only():
    prog = "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(150,150,150)\n)"
    assert _seed_variants(prog, 4, np.random.default_rng(0)) == [prog]


def test_seed_variants_k_zero_returns_original_only():
    prog = "View(0,0,1,1)\nDefine(t, fBm(seed=7))\nMaterial(\n  Layer(t)\n)"
    assert _seed_variants(prog, 0, np.random.default_rng(0)) == [prog]


def test_seed_variants_count_is_k_plus_one():
    prog = "View(0,0,1,1)\nDefine(t, fBm(seed=7))\nMaterial(\n  Layer(t)\n)"
    assert len(_seed_variants(prog, 5, np.random.default_rng(0))) == 6


def test_seed_variants_are_deterministic_given_rng():
    prog = "View(0,0,1,1)\nDefine(t, fBm(seed=7))\nMaterial(\n  Layer(t)\n)"
    v1 = _seed_variants(prog, 3, np.random.default_rng(42))
    v2 = _seed_variants(prog, 3, np.random.default_rng(42))
    assert v1 == v2


def test_seed_variants_distinct_seeds_redrawn_independently():
    # A canonical program bakes a distinct seed per fBm/Worley; the sweep must
    # redraw each independently so it samples the real N-D seed space rather than
    # collapsing every noise to one shared realization.
    prog = (
        "View(0,0,1,1)\n"
        "Define(a, fBm(seed=11))\n"
        "Define(b, fBm(seed=22))\n"
        "Material(\n  Layer(a)\n)"
    )
    new = _seeds(_seed_variants(prog, 2, np.random.default_rng(0))[1])
    assert len(new) == 2
    # both moved off their authored values
    assert new[0] not in (11, 22) and new[1] not in (11, 22)
    # independently redrawn -> distinct (correlation structure not collapsed)
    assert new[0] != new[1]


def test_seed_variants_repeated_seed_stays_correlated():
    # Two noise calls the LLM deliberately correlated by sharing seed=7 must
    # stay correlated: remapped together to one new shared value.
    prog = (
        "View(0,0,1,1)\n"
        "Define(a, fBm(seed=7))\n"
        "Define(b, fBm(seed=7))\n"
        "Material(\n  Layer(a)\n)"
    )
    new = _seeds(_seed_variants(prog, 2, np.random.default_rng(0))[1])
    assert new == [new[0], new[0]]  # both got the same new value
    assert new[0] != 7  # and it moved off the original


def test_seed_variants_negative_seed_remapped_nonnegative():
    # The regex admits a leading '-'; the redraw must emit a valid non-negative
    # seed (the engine's seed field is ge=0).
    prog = "View(0,0,1,1)\nDefine(t, fBm(seed=-3))\nMaterial(\n  Layer(t)\n)"
    assert _seeds(_seed_variants(prog, 2, np.random.default_rng(0))[1])[0] >= 0


# --------------------------------------------------------------------------- #
# Robustness: a degenerate candidate must not abort the sweep / CMA-ES
# --------------------------------------------------------------------------- #
def test_polish_degenerate_candidate_scores_floor_and_loses(monkeypatch):
    # A candidate that fails to render must not abort the sweep: it scores the
    # finite floor (_BROKEN_SCORE) and loses to any valid candidate. Uses the
    # seed-sweep-only path (max_cma_evals=0) so it runs without the `cma` package.
    import matloom.engine.polish as polish_mod

    prog = (
        "View(0, 0, 1, 1)\n"
        "Define(t, fBm(octaves=3, base_freq=2, to_01=True, seed=7))\n"
        "Material(\n  Layer(1).basecolor(200, 200, 200).roughness(0.5),\n"
        "  Layer(t).basecolor(50, 50, 50).roughness(0.5)\n)"
    )
    real_preview = polish_mod.fixed_light_preview
    calls = {"n": 0}

    def flaky_preview(material, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:  # the second seed variant's preview fails
            raise RuntimeError("degenerate candidate")
        return real_preview(material, *a, **kw)

    monkeypatch.setattr(polish_mod, "fixed_light_preview", flaky_preview)
    _, log = polish_program(
        "x",
        prog,
        seed_sweep=3,
        max_cma_evals=0,
        score_fn=lambda imgs, txt: [float(img.mean()) / 255.0 for img in imgs],
        seed=0,
    )
    variants = [c["trajectory"][0] for c in log["candidates"]]
    floor = [s for c in log["candidates"] for s in c["scores"] if s < -1e5]
    assert len(floor) == 1  # exactly the one broken variant hit the floor
    assert log["final_score"] > -1e5  # the floor did not win
    assert log["final_program"] != variants[1]  # the broken variant lost


# --------------------------------------------------------------------------- #
# Render pool (opt-in process pool for the seed sweep)
# --------------------------------------------------------------------------- #
_TWO_LAYER = (
    "View(0, 0, 1, 1)\n"
    "Define(t, fBm(octaves=3, base_freq=2, to_01=True, seed=7))\n"
    "Material(\n"
    "  Layer(1).basecolor(50, 50, 50).roughness(0.5),\n"
    "  Layer(t).basecolor(200, 200, 200).roughness(0.5)\n)"
)
_DEGENERATE = "not a program at all"


def _mean_brightness(imgs, txt):
    return [float(img.mean()) / 255.0 for img in imgs]


def test_resolve_pool_workers_param_beats_env_beats_off(monkeypatch):
    from matloom.engine.polish import _default_pool_workers, _resolve_pool_workers

    monkeypatch.delenv("MATLOOM_POLISH_WORKERS", raising=False)
    # Explicit param wins (0 is a valid "off"; negatives clamp to 0).
    assert _resolve_pool_workers(0) == 0
    assert _resolve_pool_workers(2) == 2
    assert _resolve_pool_workers(5, env="8") == 5
    assert _resolve_pool_workers(-3) == 0
    # No param -> env: counts pass through, words gate, garbage is off.
    assert _resolve_pool_workers(None, env="") == 0
    assert _resolve_pool_workers(None, env="0") == 0
    assert _resolve_pool_workers(None, env="off") == 0
    assert _resolve_pool_workers(None, env="7") == 7
    assert _resolve_pool_workers(None, env="banana") == 0
    assert _resolve_pool_workers(None, env="auto") == _default_pool_workers()
    # No param and no env in the real environment -> off (the library default).
    assert _resolve_pool_workers(None) == 0
    assert _default_pool_workers() == min(4, os.cpu_count() or 1)


def test_pool_never_fires_on_the_default_path(monkeypatch):
    # Hermeticity pin: with no parallel_renders (and no env) the sweep must
    # render in-process. Poison BOTH the module-level worker and the executor
    # constructor: any accidental pool creation or parent-side worker call
    # raises AssertionError instead of silently falling back to a pass.
    import matloom.engine.polish as polish_mod

    def _poisoned(*a, **kw):
        raise AssertionError("the render pool fired on the default path")

    monkeypatch.setattr(polish_mod, "_render_program", _poisoned)
    monkeypatch.setattr(polish_mod, "ProcessPoolExecutor", _poisoned)
    _, log = polish_program(
        "x",
        _TWO_LAYER,
        seed_sweep=2,
        max_cma_evals=0,
        score_fn=_mean_brightness,
        seed=0,
    )
    assert len(log["candidates"]) == 3
    assert log["config"]["parallel_renders"] == 0


def test_env_is_consulted_when_param_is_none(monkeypatch):
    # parallel_renders=None (not 0) defers to MATLOOM_POLISH_WORKERS: an operator
    # can force sequential rendering for library callers that leave the param
    # unset. Env "0" keeps this run spawn-free and fast.
    monkeypatch.setenv("MATLOOM_POLISH_WORKERS", "0")
    _, log = polish_program(
        "x",
        _TWO_LAYER,
        seed_sweep=1,
        max_cma_evals=0,
        score_fn=_mean_brightness,
        seed=0,
        parallel_renders=None,
    )
    assert log["config"]["parallel_renders"] == 0


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="spawns a 2-worker pool")
def test_render_program_in_spawned_worker_is_bit_identical():
    # REAL spawn pool (not a stub): covers worker picklability, the spawn
    # re-import path, and per-candidate error handling on every CI Python.
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from matloom.engine.polish import _render_program

    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        f_valid = pool.submit(_render_program, _TWO_LAYER, 64, 64)
        f_bad = pool.submit(_render_program, _DEGENERATE, 64, 64)
        img, err = f_valid.result()
        bad_img, bad_err = f_bad.result()
    assert err is None
    expected = fixed_light_preview(LayeredMaterial.deserialize(_TWO_LAYER), 64, 64)
    assert img.shape == expected.shape and img.dtype == expected.dtype
    assert img.tobytes() == expected.tobytes()
    # The degenerate program comes back as a serializable error marker carrying
    # exactly the text the sequential path would log for the same exception.
    assert bad_img is None
    try:
        LayeredMaterial.deserialize(_DEGENERATE)
        raise AssertionError("expected the degenerate program to fail to parse")
    except ValueError as e:
        assert bad_err == str(e)


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="spawns a 2-worker pool")
def test_polish_pool_matches_sequential_output_and_scores():
    # Renders are pure functions of the program text and executor.map yields in
    # submission order, so the pooled sweep must return the identical program
    # and identical per-candidate scores as the sequential one (same seed).
    _, seq = polish_program(
        "x",
        _TWO_LAYER,
        seed_sweep=4,
        max_cma_evals=0,
        score_fn=_mean_brightness,
        seed=0,
        parallel_renders=0,
    )
    final_pool, pooled = polish_program(
        "x",
        _TWO_LAYER,
        seed_sweep=4,
        max_cma_evals=0,
        score_fn=_mean_brightness,
        seed=0,
        parallel_renders=2,
    )
    assert seq["config"]["parallel_renders"] == 0
    assert pooled["config"]["parallel_renders"] == 2
    assert final_pool == seq["final_program"]
    assert [c["scores"] for c in pooled["candidates"]] == [
        c["scores"] for c in seq["candidates"]
    ]


def test_polish_pool_death_falls_back_to_sequential(monkeypatch):
    # BrokenProcessPool (worker segfault / OOM kill) is raised by the executor,
    # not the worker, so it must be caught in fitness: the batch re-renders
    # in-process, the run completes, and the results match the sequential run
    # exactly (a failure mode the sequential path structurally cannot have).
    from concurrent.futures.process import BrokenProcessPool

    import matloom.engine.polish as polish_mod

    class _DyingPool:
        def __init__(self, *a, **kw):
            pass

        def map(self, *a, **kw):
            raise BrokenProcessPool("worker died")

        def shutdown(self, *a, **kw):
            pass

    warnings: list[str] = []
    monkeypatch.setattr(polish_mod, "ProcessPoolExecutor", _DyingPool)
    # The matloom logger does not propagate, so capture at the source.
    monkeypatch.setattr(polish_mod.logger, "warning", lambda msg: warnings.append(msg))
    final, log = polish_program(
        "x",
        _TWO_LAYER,
        seed_sweep=4,
        max_cma_evals=0,
        score_fn=_mean_brightness,
        seed=0,
        parallel_renders=2,
    )
    assert any("Render pool failed" in w for w in warnings)
    _, seq = polish_program(
        "x",
        _TWO_LAYER,
        seed_sweep=4,
        max_cma_evals=0,
        score_fn=_mean_brightness,
        seed=0,
        parallel_renders=0,
    )
    assert final == seq["final_program"]
    assert [c["scores"] for c in log["candidates"]] == [
        c["scores"] for c in seq["candidates"]
    ]


# --------------------------------------------------------------------------- #
# CMA-ES driver (skipped without the vetted `cma` package)
# --------------------------------------------------------------------------- #
pytest.importorskip("cma")


def _brightness(imgs, txt):
    """Toy batched verifier: one mean-luminance score per image (torch-free).
    The polish fitness is batched (``score_fn(images, prompt) -> list[float]``),
    so a test verifier must return a list, not a single wrapped value."""
    return [float(img.mean()) / 255.0 for img in imgs]


def test_polish_accepts_if_better_and_logs():
    # Toy fitness: reward a bright preview. CMA-ES should raise the albedo.
    prog = "View(0, 0, 1, 1)\nMaterial(\n  Layer(1).basecolor(60, 60, 60).roughness(0.5)\n)"
    polished, log = polish_program(
        "white plastic",
        prog,
        max_cma_evals=80,
        cma_sigma0=0.3,
        score_fn=_brightness,
        seed=0,
    )
    # The program has no `seed=`, so the sweep is just the original: one
    # candidate, and CMA-ES appends a polished second step to its trajectory.
    assert len(log["candidates"]) == 1
    cand = log["candidates"][0]
    assert len(cand["trajectory"]) == 2
    assert cand["cma"]["n_tunables"] >= 1
    assert cand["cma"]["n_evals"] > 0
    # Accept-if-better via the max: the polish raised the verifier score.
    assert log["final_score"] > log["initial_score"]
    # The polished program parses and is brighter than the original.
    assert LayeredMaterial.deserialize(polished)
    base_img = fixed_light_preview(LayeredMaterial.deserialize(prog))
    pol_img = fixed_light_preview(LayeredMaterial.deserialize(polished))
    assert pol_img.mean() > base_img.mean()


def test_polish_no_tunables_and_no_seeds_is_noop():
    # No `seed=` literals -> the seed sweep collapses to the original; no numeric
    # tunables (octaves/to_01 are structural; View is excluded) -> CMA-ES has
    # nothing to optimize. A true no-op: the program is returned unchanged.
    prog = (
        "View(0, 0, 1, 1)\n"
        "Define(t, fBm(octaves=4, to_01=True))\n"
        "Material(\n  Layer(t)\n)"
    )
    polished, log = polish_program(
        "x",
        prog,
        max_cma_evals=10,
        score_fn=lambda imgs, txt: [0.5] * len(imgs),
        seed=0,
    )
    assert polished == prog
    assert log["final_program"] == prog
    assert log["final_score"] == log["initial_score"]
    assert log["candidates"][0]["cma"]["n_tunables"] == 0


def test_polish_seed_sweep_without_cma_picks_a_variant():
    # max_cma_evals=0 (default): only the discrete seed sweep runs. A two-layer
    # material whose top-layer mask is a seeded fBm varies in luminance across
    # re-seeds, so the sweep has a real argmax and CMA-ES never runs.
    prog = (
        "View(0, 0, 1, 1)\n"
        "Define(t, fBm(octaves=3, base_freq=2, to_01=True, seed=7))\n"
        "Material(\n  Layer(1).basecolor(50, 50, 50).roughness(0.5),\n"
        "  Layer(t).basecolor(200, 200, 200).roughness(0.5)\n)"
    )
    _, log = polish_program(
        "x", prog, seed_sweep=8, max_cma_evals=0, score_fn=_brightness, seed=0
    )
    variants = [c["trajectory"][0] for c in log["candidates"]]
    assert len(variants) == 9  # original + 8 re-seeds
    # No CMA-ES: each candidate has a one-step trajectory and no `cma` log.
    for cand in log["candidates"]:
        assert len(cand["trajectory"]) == 1
        assert "cma" not in cand
    # The winner is one of the seed variants, at the max variant score.
    assert log["final_program"] in variants
    assert log["final_score"] == max(c["scores"][0] for c in log["candidates"])


def test_polish_cma_runs_on_every_seed_variant():
    # max_cma_evals > 0: CMA-ES polishes EVERY seed variant, not just the best
    # (the new "CMA-ES on every variant" design).
    prog = (
        "View(0, 0, 1, 1)\n"
        "Define(t, fBm(octaves=3, base_freq=2, to_01=True, seed=7))\n"
        "Material(\n  Layer(1).basecolor(80, 80, 80).roughness(0.5),\n"
        "  Layer(t).basecolor(200, 200, 200).roughness(0.5)\n)"
    )
    _, log = polish_program(
        "bright",
        prog,
        seed_sweep=3,
        max_cma_evals=30,
        cma_sigma0=0.3,
        score_fn=_brightness,
        seed=0,
    )
    assert len(log["candidates"]) == 4  # original + 3 re-seeds
    for cand in log["candidates"]:
        assert len(cand["trajectory"]) == 2  # [seed_variant, cma_polished]
        assert cand["cma"]["n_tunables"] >= 1
        assert cand["cma"]["n_evals"] > 0


def test_polish_cma_polish_can_beat_seed_sweep():
    # Regression for the score-sign bug: the CMA-ES objective is -fitness, so
    # _cma_one_program stores es.result.fbest (a NEGATED, negative value). The
    # final max must un-negate it (-cma_log["score"]) or a CMA-ES polish can
    # never beat the positive seed-sweep scores. With a brightness reward and a
    # dark basecolor, CMA-ES (which can drive basecolor to 255) beats any re-seed.
    prog = (
        "View(0, 0, 1, 1)\n"
        "Define(t, fBm(octaves=3, base_freq=2, to_01=True, seed=7))\n"
        "Material(\n  Layer(t).basecolor(60, 60, 60).roughness(0.5)\n)"
    )
    _, log = polish_program(
        "white",
        prog,
        seed_sweep=3,
        max_cma_evals=40,
        cma_sigma0=0.3,
        score_fn=_brightness,
        seed=0,
    )
    seed_scores = [c["scores"][0] for c in log["candidates"]]
    polished_progs = {c["trajectory"][1] for c in log["candidates"]}
    # The winner is a CMA-ES-polished program, not a seed variant...
    assert log["final_program"] in polished_progs
    # ...and it strictly beats every seed-variant score. Without the un-negation
    # the CMA-ES scores would be negative and final_score would equal
    # max(seed_scores) (a seed variant wins), not exceed it.
    assert log["final_score"] > max(seed_scores)


def test_preview_uses_all_pbr_channels():
    """Every tunable PBR channel must affect the preview (else CMA-ES has no
    gradient on it). Verified by toggling each channel and checking the mean
    luminance changes vs the no-extra-channel baseline."""

    def preview(extra):
        prog = (
            "View(0, 0, 1, 1)\nMaterial(\n"
            "  Layer(1).basecolor(150, 150, 160).roughness(0.3)" + extra + "\n)"
        )
        return fixed_light_preview(LayeredMaterial.deserialize(prog), 96, 96).mean()

    base = preview("")
    # sheen, coat, subsurface, anisotropy all brighten/soften; ior is monotonic.
    assert preview(".sheen(0.8)") != base
    assert preview(".coat(0.8)") != base
    assert preview(".subsurface(0.8)") != base
    assert preview(".anisotropy(0.8)") != base
    # higher IOR -> more reflective (larger Schlick F0) -> brighter.
    assert preview(".ior(2.5)") > preview(".ior(1.2)")


def test_preview_uses_basecolor_roughness_metallic_transmission_emissive():
    """The channels the original `test_preview_uses_all_pbr_channels` omits
    (basecolor, roughness, metallic, transmission, emissive) must each move the
    preview mean, so CMA-ES has a gradient on them too."""

    def m(prog):
        return fixed_light_preview(LayeredMaterial.deserialize(prog), 80, 80).mean()

    assert m(
        "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(200,200,200).roughness(0.5)\n)"
    ) > m("View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(50,50,50).roughness(0.5)\n)")
    assert m(
        "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(150,150,150).roughness(0.1)\n)"
    ) != m(
        "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(150,150,150).roughness(0.9)\n)"
    )
    assert m(
        "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(180,180,180).metallic(1).roughness(0.3)\n)"
    ) != m(
        "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(180,180,180).metallic(0).roughness(0.3)\n)"
    )
    assert m(
        "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(200,50,50).transmission(0.9).ior(1.5)\n)"
    ) != m(
        "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(200,50,50).transmission(0.0).ior(1.5)\n)"
    )
    assert m(
        "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(50,50,50).emissive(200,50,50,1.0)\n)"
    ) > m(
        "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(50,50,50).emissive(0,0,0,1.0)\n)"
    )


def test_preview_height_and_alpha_mask_affect_output():
    """Height (relief -> normal/AO) and the alpha mask (over-operator) must
    affect the preview. Uses a two-layer material with a thresholded fBm mask so
    the over-composited height has real relief and the top layer partially
    occludes the substrate."""
    flat = fixed_light_preview(
        LayeredMaterial.deserialize(
            "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(150,150,150).roughness(0.5)\n)"
        ),
        80,
        80,
    ).mean()
    relief = fixed_light_preview(
        LayeredMaterial.deserialize(
            "View(0,0,1,1)\n"
            "Define(m, Threshold(fBm(octaves=4, base_freq=5, to_01=True, seed=1),"
            " below_at=0.4, below_to=1, above_at=0.6, above_to=0))\n"
            "Material(\n  Layer(1).basecolor(150,150,150).roughness(0.5),\n"
            "  Layer(m).basecolor(150,150,150).roughness(0.5).height(0.3)\n)"
        ),
        80,
        80,
    ).mean()
    assert relief != flat
    # Alpha mask: a partial top layer vs no top layer changes the composite.
    masked = fixed_light_preview(
        LayeredMaterial.deserialize(
            "View(0,0,1,1)\n"
            "Define(m, Threshold(fBm(octaves=4, base_freq=5, to_01=True, seed=1),"
            " below_at=0.4, below_to=1, above_at=0.6, above_to=0))\n"
            "Material(\n  Layer(1).basecolor(200,50,50).roughness(0.5),\n"
            "  Layer(m).basecolor(50,50,200).roughness(0.5)\n)"
        ),
        80,
        80,
    ).mean()
    solo = fixed_light_preview(
        LayeredMaterial.deserialize(
            "View(0,0,1,1)\nMaterial(\n  Layer(1).basecolor(200,50,50).roughness(0.5)\n)"
        ),
        80,
        80,
    ).mean()
    assert masked != solo


def test_preview_robust_to_partial_alpha():
    """A material with zero-alpha regions (a partial mask) must not produce
    NaN/inf in the preview: the blend's divide-by-alpha is masked, the relief
    falls back to the substrate, and the shader clamps all inputs."""
    prog = (
        "View(0,0,1,1)\n"
        "Define(m, Threshold(fBm(octaves=4, base_freq=8, to_01=True, seed=3),"
        " below_at=0.3, below_to=1, above_at=0.5, above_to=0))\n"
        "Material(\n  Layer(1).basecolor(100,100,100).roughness(0.5).height(0.2),\n"
        "  Layer(m).basecolor(200,200,200).roughness(0.5).height(0.5)\n)"
    )
    img = fixed_light_preview(LayeredMaterial.deserialize(prog), 80, 80)
    assert img.shape == (80, 80, 3) and img.dtype == np.uint8
    assert np.isfinite(img.astype(float)).all()
