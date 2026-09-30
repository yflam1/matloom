"""Tests for matloom.engine.text2dsl — the LLM text-to-DSL pipeline.

These exercise the generation/parse-retry/refinement orchestration without a
network or a real model: ``build_llm`` is monkeypatched to return an ``Llm``
whose OpenAI client is a fake responder (the same technique as
tests/utils/test_llm.py). No torch, no Blender, no API calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("openai")
pytest.importorskip("json_repair")

from matloom.engine import text2dsl
from matloom.engine.main import LayeredMaterial
from matloom.engine.text2dsl import (
    FEW_SHOT_EXAMPLES,
    Critique,
    Text2DslError,
    _build_system_prompt,
    extract_preamble,
    extract_program,
    text_to_material,
)
from matloom.utils.llm import Llm

# A minimal valid program the fake model can "emit".
_GOOD = (
    "View(0, 0, 1, 1)\nMaterial(\n  Layer(1).basecolor(200, 30, 30).roughness(0.3)\n)"
)
_GOOD2 = (
    "View(0, 0, 1, 1)\nMaterial(\n  Layer(1).basecolor(20, 30, 200).roughness(0.6)\n)"
)


# --------------------------------------------------------------------------- #
# Fakes (duck-typed openai client)
# --------------------------------------------------------------------------- #
def _completion(content: str):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeCompletions:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.seen = []  # api-format messages of each create() call

    def create(self, *, messages, stream, **params):
        self.seen.append(messages)
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return _completion(reply)


def _fake_llm(replies):
    llm = Llm("fake-model", api_key="sk-test")
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(replies))
    )
    return llm


def _patch_build_llm(monkeypatch, replies):
    """Make every build_llm(...) return a fresh fake with the given replies."""
    monkeypatch.setattr(text2dsl, "build_llm", lambda *a, **k: _fake_llm(replies))


# --------------------------------------------------------------------------- #
# Program extraction
# --------------------------------------------------------------------------- #
def test_extract_program_from_tags():
    text = f"Here you go:\n<material>\n{_GOOD}\n</material>\nDone."
    assert extract_program(text) == _GOOD


def test_extract_program_missing_returns_none():
    assert extract_program("no tags here") is None


def test_extract_preamble_recovers_text_before_tags():
    text = f"Here is my plan.\n<material>\n{_GOOD}\n</material>\nDone."
    assert extract_preamble(text) == "Here is my plan."


def test_extract_preamble_empty_when_no_prose():
    assert extract_preamble(f"<material>\n{_GOOD}\n</material>") == ""


def test_extract_preamble_empty_when_no_tags():
    # No tag block -> nothing to precede; returns "" (defensive, like extract_program
    # returns None, but the preamble default is the empty string).
    assert extract_preamble("no tags here") == ""


# --------------------------------------------------------------------------- #
# Generation (happy path / retries)
# --------------------------------------------------------------------------- #
def test_text_to_material_happy_path(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    trajectory, _ = text_to_material("red plastic", few_shot=False, verbose=False)
    # Returns a parseable program.
    assert LayeredMaterial.deserialize(trajectory[-1]).serialize()


def test_text_to_material_retries_on_unparseable(monkeypatch):
    bad = "<material>\nMaterial(Layer(1).bogus(2))\n</material>"
    good = f"<material>\n{_GOOD}\n</material>"
    _patch_build_llm(monkeypatch, [bad, good])
    trajectory, _ = text_to_material(
        "red plastic", few_shot=False, parse_retries=2, verbose=False
    )
    assert LayeredMaterial.deserialize(trajectory[-1])


def test_text_to_material_retries_on_missing_tags(monkeypatch):
    good = f"<material>\n{_GOOD}\n</material>"
    _patch_build_llm(monkeypatch, ["I cannot help with that.", good])
    trajectory, _ = text_to_material(
        "red plastic", few_shot=False, parse_retries=1, verbose=False
    )
    assert LayeredMaterial.deserialize(trajectory[-1])


def test_text_to_material_gives_up_after_parse_retries(monkeypatch):
    bad = "<material>\nMaterial(Layer(1).bogus(2))\n</material>"
    _patch_build_llm(monkeypatch, [bad])
    with pytest.raises(Text2DslError):
        text_to_material("x", few_shot=False, parse_retries=0, verbose=False)


# --------------------------------------------------------------------------- #
# System prompt / few-shot toggle (the ablation)
# --------------------------------------------------------------------------- #
def test_system_prompt_few_shot_toggle():
    with_ex = _build_system_prompt(few_shot=True)
    without = _build_system_prompt(few_shot=False)
    assert "# Examples" in with_ex
    assert "# Examples" not in without
    # The DSL reference and output format are always present.
    for s in (with_ex, without):
        assert "layered-material DSL" in s
        assert "<material>" in s


def test_few_shot_examples_are_valid_and_stable():
    assert len(FEW_SHOT_EXAMPLES) == 6
    for prompt, program in FEW_SHOT_EXAMPLES:
        once = LayeredMaterial.deserialize(program).serialize()
        twice = LayeredMaterial.deserialize(once).serialize()
        assert once == twice, prompt


def test_few_shot_examples_vary_layer_count_and_expressions():
    # The examples must teach layering and expression use, not just syntax: they
    # should span single- and multi-layer materials (here up to 4 layers) and the
    # set actually uses Define, the pattern-in-the-mask idiom, and expression-
    # driven channels — not be a row of one-layer constants.
    counts = sorted(
        len(LayeredMaterial.deserialize(p)._layers) for _, p in FEW_SHOT_EXAMPLES
    )
    assert counts[0] == 1, "want at least one single-layer example"
    assert counts[-1] >= 4, "want at least one example with 4+ layers"
    assert len(set(counts)) >= 3, "layer counts should vary across examples"
    corpus = "\n".join(p for _, p in FEW_SHOT_EXAMPLES)
    assert "Define(" in corpus
    # Expression-driven channels appear (a channel argument that is not a bare
    # number/identifier — e.g. arithmetic inside basecolor/roughness/height).
    assert "* " in corpus and "+ " in corpus


# --------------------------------------------------------------------------- #
# Critique (structured output) + refinement loop
# --------------------------------------------------------------------------- #
def test_critique_json_response_parses():
    raw = (
        '{"match_score": 72, "differences": ["too glossy", "color too orange"], '
        '"suggestions": ["raise roughness", "shift hue toward red"]}'
    )
    c = Critique.from_str(raw)
    assert c.match_score == 72
    assert "too glossy" in c.differences
    assert len(c.suggestions) == 2


def test_critique_to_str_is_a_template():
    tmpl = Critique.to_str()
    assert "match_score" in tmpl
    assert "differences" in tmpl
    assert "suggestions" in tmpl


def test_refine_loop_applies_critique(monkeypatch):
    # Generation returns _GOOD first, then _GOOD2 after the critique feedback.
    _patch_build_llm(
        monkeypatch,
        [
            f"<material>\n{_GOOD}\n</material>",
            f"<material>\n{_GOOD2}\n</material>",
        ],
    )
    # Stub the (Blender-backed) critic so no render is needed.
    seen: dict = {"programs": []}

    def fake_critique(critic, prompt, program, **kw):
        seen["programs"].append(program)
        return Critique(match_score=40, differences=["too red"], suggestions=["bluer"])

    monkeypatch.setattr(text2dsl, "_critique", fake_critique)
    trajectory, _ = text_to_material(
        "a material",
        few_shot=False,
        refine=1,
        critic_render=True,
        verbose=False,
    )
    # One refine round drives a revision, then a trailing final critique scores
    # the revised program (no further revision). So the critic is called twice.
    assert len(seen["programs"]) == 2
    assert "200, 30, 30" in seen["programs"][0]  # _GOOD drove the revision
    assert "20, 30, 200" in seen["programs"][1]  # _GOOD2 trailing critique
    assert LayeredMaterial.deserialize(trajectory[-1])
    assert "20, 30, 200" in trajectory[-1]  # _GOOD2 was returned after refinement


# --------------------------------------------------------------------------- #
# Seed pinning: canonicalize each program + rewrite the convo's <material> turn
# --------------------------------------------------------------------------- #
_NOISE = (
    "View(0, 0, 1, 1)\nMaterial(\n"
    "  Layer(1).basecolor(200, 30, 30).height(fBm(base_freq=3, to_01=True))\n)"
)
_NOISE2 = (
    "View(0, 0, 1, 1)\nMaterial(\n"
    "  Layer(1).basecolor(20, 30, 200).height(fBm(base_freq=3, to_01=True))\n)"
)


def _msg_text(msg: dict) -> str:
    """Flatten an api-format message's content to a string (user turns carry a
    list of content parts; assistant turns carry a plain string)."""
    c = msg["content"]
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c)


def test_canonicalize_pins_seeds_and_rewrites_convo(monkeypatch):
    # Both replies contain an UNSEEDED fBm; canonicalization must bake a seed.
    fake = _fake_llm(
        [
            f"<material>\n{_NOISE}\n</material>",
            f"<material>\n{_NOISE2}\n</material>",
        ]
    )
    monkeypatch.setattr(text2dsl, "build_llm", lambda *a, **k: fake)
    monkeypatch.setattr(
        text2dsl,
        "_critique",
        lambda *a, **k: Critique(match_score=40, differences=["x"], suggestions=["y"]),
    )
    trajectory, _ = text_to_material(
        "a material",
        few_shot=False,
        refine=1,
        critic_render=True,
        verbose=False,
    )
    # Every trajectory program has its fBm seed baked (no unseeded noise left).
    assert "seed=" in trajectory[0]
    assert "seed=" in trajectory[-1]

    # The refine-turn create() call saw the canonical (seed-baked) depth-0
    # program as the assistant turn — i.e. the convo's <material> reply was
    # rewritten so the LLM copies baked seeds forward for unchanged layers.
    refine_msgs = fake._client.chat.completions.seen[1]
    assistant_text = [_msg_text(m) for m in refine_msgs if m["role"] == "assistant"]
    assert any(trajectory[0] in t for t in assistant_text)
    # The refine prompt instructs the LLM to copy seed= values verbatim.
    user_text = [_msg_text(m) for m in refine_msgs if m["role"] == "user"]
    assert any("seed=" in t and "unchanged" in t for t in user_text)


def test_preamble_survives_seed_pinning_rewrite(monkeypatch):
    # _OUTPUT_INSTRUCTIONS invites the model to "think artistically" before the
    # <material> block. That preamble must survive _pin_seeds so the refine turn
    # sees the model's own rationale ahead of the canonicalized program (and the
    # preamble must stay BEFORE the tag, with the seed-baked program inside it).
    pre = "I will layer a low-frequency fBm for broad red form, then add grain."
    fake = _fake_llm(
        [
            f"{pre}\n<material>\n{_NOISE}\n</material>",
            f"{pre}\n<material>\n{_NOISE2}\n</material>",
        ]
    )
    monkeypatch.setattr(text2dsl, "build_llm", lambda *a, **k: fake)
    monkeypatch.setattr(
        text2dsl,
        "_critique",
        lambda *a, **k: Critique(match_score=40, differences=["x"], suggestions=["y"]),
    )
    trajectory, _ = text_to_material(
        "a material",
        few_shot=False,
        refine=1,
        critic_render=True,
        verbose=False,
    )
    # seen[1] is the refine create() call; its messages include the rewritten
    # depth-0 assistant turn (preamble + canonical <material> block).
    refine_msgs = fake._client.chat.completions.seen[1]
    assistant_text = "\n".join(
        _msg_text(m) for m in refine_msgs if m["role"] == "assistant"
    )
    assert pre in assistant_text  # preamble preserved
    assert "seed=" in assistant_text and trajectory[0] in assistant_text  # + baked
    assert assistant_text.index(pre) < assistant_text.index("<material>")  # before tag


def test_seed_arg_makes_runs_reproducible(monkeypatch):
    def run() -> str:
        _patch_build_llm(monkeypatch, [f"<material>\n{_NOISE}\n</material>"])
        traj, _ = text_to_material(
            "a material", few_shot=False, seed=123, verbose=False
        )
        return traj[0]

    a = run()
    b = run()
    # Same seed arg -> same baked auto-seed -> identical canonical program.
    assert a == b
    assert "seed=" in a  # the seed was actually baked (not left unseeded)


def test_no_seed_arg_does_not_pin_reproducibly(monkeypatch):
    # With seed=None the baked auto-seed is drawn from the global RNG and is not
    # guaranteed to repeat across runs (it still gets baked for within-run
    # stability, just not cross-run reproducible).
    _patch_build_llm(monkeypatch, [f"<material>\n{_NOISE}\n</material>"])
    traj, record = text_to_material(
        "a material", few_shot=False, seed=None, verbose=False
    )
    assert "seed=" in traj[0]
    assert record["settings"]["seed"] is None


def test_refine_stops_when_render_unavailable(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    # _critique returns None (rendering unavailable) → refinement is skipped.
    monkeypatch.setattr(text2dsl, "_critique", lambda *a, **k: None)
    trajectory, _ = text_to_material(
        "a material",
        few_shot=False,
        refine=3,
        critic_render=True,
        verbose=False,
    )
    assert LayeredMaterial.deserialize(trajectory[-1])


# --------------------------------------------------------------------------- #
# Refine trajectory (one refine=N run answers every depth 1..N at once)
# --------------------------------------------------------------------------- #
def test_trajectory_returns_program_after_each_refine(monkeypatch):
    # Generation: p0, then a revised program after each of two critique rounds.
    _GOOD3 = "View(0, 0, 1, 1)\nMaterial(\n  Layer(1).basecolor(10, 200, 30).roughness(0.5)\n)"
    gen = _fake_llm(
        [
            f"<material>\n{_GOOD}\n</material>",
            f"<material>\n{_GOOD2}\n</material>",
            f"<material>\n{_GOOD3}\n</material>",
        ]
    )
    crit = _fake_llm(
        ['{"match_score": 30, "differences": ["x"], "suggestions": ["y"]}'] * 2
    )
    calls = {"n": 0}

    def fake_build(*a, **k):
        i = calls["n"]
        calls["n"] += 1
        return gen if i == 0 else crit

    monkeypatch.setattr(text2dsl, "build_llm", fake_build)
    # Stub the render-backed critic so no Blender is needed but the loop still runs.
    monkeypatch.setattr(
        text2dsl,
        "_critique",
        lambda *a, **k: Critique(match_score=30, differences=["x"], suggestions=["y"]),
    )
    traj, _ = text_to_material(
        "a material",
        few_shot=False,
        refine=2,
        critic_render=True,
        verbose=False,
    )
    # [p0 (0 refines), p1 (1 refine), p2 (2 refines)].
    assert isinstance(traj, list) and len(traj) == 3
    assert "200, 30, 30" in traj[0]
    assert "20, 30, 200" in traj[1]
    assert "10, 200, 30" in traj[2]
    for p in traj:
        assert LayeredMaterial.deserialize(p)


def test_trajectory_with_no_refine_is_singleton(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    traj, _ = text_to_material("a material", few_shot=False, verbose=False)
    assert isinstance(traj, list) and len(traj) == 1
    assert "200, 30, 30" in traj[0]


def test_trajectory_shortens_when_refine_stops_early(monkeypatch):
    # refine=3 requested, but rendering becomes unavailable after p0 → just [p0].
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    monkeypatch.setattr(text2dsl, "_critique", lambda *a, **k: None)
    traj, _ = text_to_material(
        "a material",
        few_shot=False,
        refine=3,
        critic_render=True,
        verbose=False,
    )
    assert isinstance(traj, list) and len(traj) == 1


def test_default_return_is_final_program_string(monkeypatch):
    # Without return_trajectory the API is unchanged (returns the final str).
    _patch_build_llm(
        monkeypatch,
        [
            f"<material>\n{_GOOD}\n</material>",
            f"<material>\n{_GOOD2}\n</material>",
        ],
    )
    monkeypatch.setattr(
        text2dsl,
        "_critique",
        lambda *a, **k: Critique(match_score=30, differences=[], suggestions=[]),
    )
    trajectory, _ = text_to_material(
        "a material",
        few_shot=False,
        refine=1,
        critic_render=True,
        verbose=False,
    )
    assert isinstance(trajectory, list) and len(trajectory) == 2
    assert isinstance(trajectory[-1], str)
    assert "20, 30, 200" in trajectory[-1]  # the final (revised) program


# --------------------------------------------------------------------------- #
# return_meta: the per-round critiques ride along with the trajectory
# --------------------------------------------------------------------------- #
def test_return_meta_records_each_critique(monkeypatch):
    _GOOD3 = "View(0, 0, 1, 1)\nMaterial(\n  Layer(1).basecolor(10, 200, 30).roughness(0.5)\n)"
    _patch_build_llm(
        monkeypatch,
        [
            f"<material>\n{_GOOD}\n</material>",
            f"<material>\n{_GOOD2}\n</material>",
            f"<material>\n{_GOOD3}\n</material>",
        ],
    )
    verdicts = [
        Critique(match_score=41, differences=["too red"], suggestions=["bluer"]),
        Critique(match_score=77, differences=["flat"], suggestions=["add relief"]),
        Critique(match_score=88, differences=["done"], suggestions=[]),
    ]
    it = iter(verdicts)
    monkeypatch.setattr(text2dsl, "_critique", lambda *a, **k: next(it))
    traj, record = text_to_material(
        "a material",
        few_shot=False,
        refine=2,
        critic_render=True,
        verbose=False,
    )
    assert len(traj) == 3
    # Two refine rounds drive revisions, then a trailing final critique scores the
    # last program (no further revision). The trailing entry produces no program.
    assert [c["match_score"] for c in record["critiques"]] == [41, 77, 88]
    assert record["critiques"][0]["differences"] == ["too red"]
    assert record["critiques"][1]["suggestions"] == ["add relief"]


def test_return_meta_empty_without_refinement(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    trajectory, record = text_to_material("a material", few_shot=False, verbose=False)
    assert isinstance(trajectory[-1], str)
    assert record["critiques"] == []


def test_return_meta_empty_when_refine_stops_early(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    monkeypatch.setattr(text2dsl, "_critique", lambda *a, **k: None)
    traj, record = text_to_material(
        "a material",
        few_shot=False,
        refine=3,
        critic_render=True,
        verbose=False,
    )
    assert len(traj) == 1
    assert record["critiques"] == []


# --------------------------------------------------------------------------- #
# The REAL _critique LLM call (no stub) — regression for three live-only bugs:
#   1. the critic is an Llm, whose one-shot API is __call__, not .send();
#   2. the uninformed appearance-only system prompt (_build_critic_system_prompt) must reach it;
#   3. CoStar.Json needs a non-None `context` or its template render raises.
# All three are invisible if the test stubs _critique itself, so these drive the
# genuine code path with a fake OpenAI client + a stubbed (Blender-free) render.
# --------------------------------------------------------------------------- #
def _stub_render(monkeypatch):
    """Stub the no-Blender preview (``matloom.engine.preview.fixed_light_preview``)
    so ``_critique``'s render is a tiny RGB image without the real numpy composite."""
    import numpy as np

    monkeypatch.setattr(
        "matloom.engine.preview.fixed_light_preview",
        lambda material, width=256, height=256: np.zeros((8, 8, 3), np.uint8),
    )


def test_critique_invokes_blind_vision_critic(monkeypatch):
    pytest.importorskip("PIL")
    pytest.importorskip("cv2")
    _stub_render(monkeypatch)
    critic = _fake_llm(
        ['{"match_score": 30, "differences": ["too red"], "suggestions": ["go blue"]}']
    )
    verdict = text2dsl._critique(
        critic,
        "deep blue plastic",
        _GOOD,
        render=True,
        code=False,
        width=8,
        height=8,
        temperature=None,
        reasoning_effort=None,
        verbose=False,
    )
    assert isinstance(verdict, Critique)
    assert verdict.match_score == 30
    # The critic client was actually queried exactly once...
    fake = critic._client.chat.completions
    assert fake.calls == 1
    msgs = fake.seen[0]
    # ...with the blind appearance-only system prompt...
    assert msgs[0]["role"] == "system"
    assert "appearance" in msgs[0]["content"].lower()
    # ...and an attached image (so it judges the render, not the DSL).
    user = msgs[-1]
    assert any(part.get("type") == "image_url" for part in user["content"])
    # The DSL string must NOT appear anywhere in what the critic sees (it is blind).
    flat = str(msgs)
    assert "Layer(" not in flat and "basecolor" not in flat


def test_refine_loop_runs_real_critique(monkeypatch):
    pytest.importorskip("PIL")
    pytest.importorskip("cv2")
    _stub_render(monkeypatch)
    gen = _fake_llm(
        [
            f"<material>\n{_GOOD}\n</material>",
            f"<material>\n{_GOOD2}\n</material>",
        ]
    )
    crit = _fake_llm(
        ['{"match_score": 20, "differences": ["too red"], "suggestions": ["bluer"]}']
    )
    calls = {"n": 0}

    def fake_build(*a, **k):
        i = calls["n"]
        calls["n"] += 1
        return gen if i == 0 else crit

    monkeypatch.setattr(text2dsl, "build_llm", fake_build)
    trajectory, _ = text_to_material(
        "blue plastic",
        few_shot=False,
        refine=1,
        critic_render=True,
        verbose=False,
    )
    # The real _critique ran twice: one refine round, plus the trailing final
    # critique of the revised program. The round's feedback produced _GOOD2.
    assert crit._client.chat.completions.calls == 2
    assert "20, 30, 200" in trajectory[-1]
    assert LayeredMaterial.deserialize(trajectory[-1])


# --------------------------------------------------------------------------- #
# The render path: stub the no-Blender preview to assert when it is / isn't run
# --------------------------------------------------------------------------- #
def _capture_render(monkeypatch):
    """Stub ``fixed_light_preview`` to record whether ``_critique``'s render path
    is touched (returns a tiny image), so render=False configs can assert no
    image was produced."""
    import numpy as np

    seen: dict = {}

    def fake(material, width=256, height=256):
        seen["called"] = True
        return np.zeros((8, 8, 3), np.uint8)

    monkeypatch.setattr("matloom.engine.preview.fixed_light_preview", fake)
    return seen


# --------------------------------------------------------------------------- #
# Critic signals: render / code / diagnostics (the 7 refinement configs)
# --------------------------------------------------------------------------- #
def _last_user_parts(msgs):
    """The content parts of the final (user) message of the first critic call."""
    user = msgs[-1]
    return user["content"], user


def test_critique_informed_shows_program_and_dsl(monkeypatch):
    pytest.importorskip("PIL")
    _stub_render(monkeypatch)
    critic = _fake_llm(
        ['{"match_score": 30, "differences": ["x"], "suggestions": ["y"]}']
    )
    text2dsl._critique(
        critic,
        "deep blue plastic",
        _GOOD,
        render=True,
        code=True,
        width=8,
        height=8,
        temperature=None,
        reasoning_effort=None,
        verbose=False,
    )
    msgs = critic._client.chat.completions.seen[0]
    # The informed critic's system prompt is the CRITIC role + the DSL spec (no
    # generator role), with the line-targeted-edit preamble.
    sys_content = msgs[0]["content"].lower()
    assert "line-targeted" in sys_content
    assert "# program structure" in sys_content  # the DSL spec is attached
    assert "expert technical artist" not in sys_content  # no generator role
    # ...and the user message shows the program source + an attached render.
    parts, user = _last_user_parts(msgs)
    flat = str(user)
    assert "Layer(" in flat and "basecolor" in flat  # the DSL source is shown
    assert any(p.get("type") == "image_url" for p in parts)


def test_critique_diagnostics_only_is_text_only(monkeypatch):
    pytest.importorskip("PIL")
    # Capture whether the render path is touched; it must NOT be in this mode.
    seen = _capture_render(monkeypatch)
    critic = _fake_llm(['{"match_score": 50, "differences": [], "suggestions": []}'])
    verdict = text2dsl._critique(
        critic,
        "deep blue plastic",
        _GOOD,
        render=False,
        code=False,
        diagnostics=True,
        width=8,
        height=8,
        temperature=None,
        reasoning_effort=None,
        verbose=False,
    )
    assert verdict is not None and verdict.match_score == 50
    assert seen == {}  # no preview render was produced
    msgs = critic._client.chat.completions.seen[0]
    assert (
        "appearance scalars" in msgs[0]["content"].lower()
    )  # diagnostics system prompt
    parts, user = _last_user_parts(msgs)
    flat = str(user)
    assert "appearance scalars" in flat.lower()  # the scalar block lead-in
    assert (
        "Albedo luminance mean" in flat
    )  # a descriptor bullet (human label) is present
    assert "deep blue plastic" in flat
    assert not any(p.get("type") == "image_url" for p in parts)  # text-only
    assert "Layer(" not in flat  # no DSL source (not informed)


def test_critique_blind_plus_diagnostics_has_image_and_scalars(monkeypatch):
    pytest.importorskip("PIL")
    _stub_render(monkeypatch)
    critic = _fake_llm(
        ['{"match_score": 40, "differences": ["x"], "suggestions": ["y"]}']
    )
    text2dsl._critique(
        critic,
        "red plastic",
        _GOOD,
        render=True,
        code=False,
        diagnostics=True,
        width=8,
        height=8,
        temperature=None,
        reasoning_effort=None,
        verbose=False,
    )
    msgs = critic._client.chat.completions.seen[0]
    # Uninformed system prompt (no DSL spec), but it tells the critic to use the
    # scalars; the user message presents the scalars + the image.
    sys_content = msgs[0]["content"].lower()
    assert "layered-material dsl" not in sys_content  # no DSL spec
    assert "use them to ground" in sys_content  # diagnostics framing in the role
    assert "appearance scalars" in sys_content  # the scalars are part of the role
    parts, user = _last_user_parts(msgs)
    flat = str(user)
    assert "appearance scalars" in flat.lower()  # the scalar block lead-in
    assert "Albedo luminance mean" in flat  # a descriptor bullet (human label)
    assert any(p.get("type") == "image_url" for p in parts)
    assert "Layer(" not in flat  # uninformed: no program source


def test_critique_code_only_is_text_only(monkeypatch):
    pytest.importorskip("PIL")
    # No render path is touched in this mode (code only, no image/vision).
    seen = _capture_render(monkeypatch)
    critic = _fake_llm(['{"match_score": 50, "differences": [], "suggestions": []}'])
    verdict = text2dsl._critique(
        critic,
        "deep blue plastic",
        _GOOD,
        render=False,
        code=True,
        width=8,
        height=8,
        temperature=None,
        reasoning_effort=None,
        verbose=False,
    )
    assert verdict is not None and verdict.match_score == 50
    assert seen == {}  # no preview render was produced
    msgs = critic._client.chat.completions.seen[0]
    # The code-only critic gets the DSL spec + a line-targeted, no-render judge.
    sys_content = msgs[0]["content"].lower()
    assert "# program structure" in sys_content  # the DSL spec is attached
    assert "line-targeted" in sys_content
    assert "do not see a render" in sys_content  # text-only framing
    parts, user = _last_user_parts(msgs)
    flat = str(user)
    assert "Layer(" in flat and "basecolor" in flat  # the DSL source is shown
    assert "deep blue plastic" in flat
    assert not any(p.get("type") == "image_url" for p in parts)  # text-only
    assert "appearance scalars" not in flat.lower()  # no diagnostics


def test_critique_code_plus_diagnostics_is_text_only(monkeypatch):
    pytest.importorskip("PIL")
    seen = _capture_render(monkeypatch)
    critic = _fake_llm(['{"match_score": 50, "differences": [], "suggestions": []}'])
    verdict = text2dsl._critique(
        critic,
        "deep blue plastic",
        _GOOD,
        render=False,
        code=True,
        diagnostics=True,
        width=8,
        height=8,
        temperature=None,
        reasoning_effort=None,
        verbose=False,
    )
    assert verdict is not None and verdict.match_score == 50
    assert seen == {}  # no preview render was produced
    msgs = critic._client.chat.completions.seen[0]
    sys_content = msgs[0]["content"].lower()
    assert "# program structure" in sys_content  # DSL spec attached (code)
    assert "line-targeted" in sys_content
    assert "cross-check the scalars" in sys_content  # the {A,C} judge clause
    assert "no render is shown" in sys_content  # scalars caveat (no render)
    parts, user = _last_user_parts(msgs)
    flat = str(user)
    assert "Layer(" in flat and "basecolor" in flat  # DSL source shown
    assert "Albedo luminance mean" in flat  # scalars present
    assert "deep blue plastic" in flat
    assert not any(p.get("type") == "image_url" for p in parts)  # text-only


def test_critique_no_signal_returns_none_without_calling_builder(monkeypatch):
    # All three signals off -> nothing to critique -> None, and neither the
    # prompt builder nor the critic LLM is ever invoked.
    calls = {"builder": 0}

    def spy(*a, **k):
        calls["builder"] += 1
        raise AssertionError("builder must not be called with no signal")

    monkeypatch.setattr(text2dsl, "_build_critic_system_prompt", spy)
    critic = _fake_llm(['{"match_score": 50, "differences": [], "suggestions": []}'])
    verdict = text2dsl._critique(
        critic,
        "deep blue plastic",
        _GOOD,
        render=False,
        code=False,
        diagnostics=False,
        width=8,
        height=8,
        temperature=None,
        reasoning_effort=None,
        verbose=False,
    )
    assert verdict is None
    assert calls["builder"] == 0  # the no-signal guard returned before the builder
    assert critic._client.chat.completions.calls == 0


def test_text_to_material_informed_mode_is_plumbed(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    captured = {}

    def fake_critique(
        critic,
        prompt,
        program,
        *,
        render=False,
        code=False,
        diagnostics=False,
        **kw,
    ):
        captured["render"] = render
        captured["code"] = code
        captured["diagnostics"] = diagnostics
        return Critique(match_score=30, differences=["x"], suggestions=["y"])

    monkeypatch.setattr(text2dsl, "_critique", fake_critique)
    _, record = text_to_material(
        "a material",
        few_shot=False,
        refine=1,
        critic_render=True,
        critic_code=True,
        verbose=False,
    )
    assert captured["render"] is True
    assert captured["code"] is True
    assert captured["diagnostics"] is False
    assert record["settings"]["critic_render"] is True
    assert record["settings"]["critic_code"] is True


def test_text_to_material_critic_diagnostics_only_refines_without_visual(
    monkeypatch,
):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    captured = {}

    def fake_critique(
        critic,
        prompt,
        program,
        *,
        render=False,
        code=False,
        diagnostics=False,
        **kw,
    ):
        captured["render"] = render
        captured["code"] = code
        captured["diagnostics"] = diagnostics
        return Critique(
            match_score=55, differences=["flat"], suggestions=["add relief"]
        )

    monkeypatch.setattr(text2dsl, "_critique", fake_critique)
    # No render, no code, but diagnostics=True -> diagnostics-only refinement.
    _, record = text_to_material(
        "a material",
        few_shot=False,
        refine=1,
        critic_diagnostics=True,
        verbose=False,
    )
    assert captured["render"] is False
    assert captured["code"] is False
    assert captured["diagnostics"] is True
    assert record["settings"]["critic_render"] is False
    assert record["settings"]["critic_code"] is False
    assert record["settings"]["critic_diagnostics"] is True


def test_text_to_material_records_mode_in_meta(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    monkeypatch.setattr(
        text2dsl,
        "_critique",
        lambda *a, **k: Critique(match_score=30, differences=["x"], suggestions=["y"]),
    )
    _, record = text_to_material(
        "a material",
        few_shot=False,
        refine=1,
        critic_render=True,
        verbose=False,
    )
    # The mode lives in the record's settings, not duplicated per critique.
    assert record["settings"]["critic_render"] is True
    assert record["settings"]["critic_code"] is False
    assert record["settings"]["critic_diagnostics"] is False


# --------------------------------------------------------------------------- #
# Playbook / few-shot independence + select_best + record shape
# --------------------------------------------------------------------------- #
def test_playbook_and_few_shot_are_independent():
    from matloom.engine.text2dsl import _ORGANIC_PLAYBOOK, _build_system_prompt

    both_off = _build_system_prompt(few_shot=False, playbook=False)
    assert "Organic texture playbook" not in both_off and "# Examples" not in both_off
    pb_only = _build_system_prompt(few_shot=False, playbook=True)
    assert "Organic texture playbook" in pb_only and "# Examples" not in pb_only
    fs_only = _build_system_prompt(few_shot=True, playbook=False)
    assert "# Examples" in fs_only and "Organic texture playbook" not in fs_only
    both_on = _build_system_prompt(few_shot=True, playbook=True)
    assert "Organic texture playbook" in both_on and "# Examples" in both_on
    assert _ORGANIC_PLAYBOOK  # non-empty


def test_record_has_full_shape(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    _, record = text_to_material(
        "red plastic", few_shot=False, playbook=False, verbose=False
    )
    assert set(record) == {
        "prompt",
        "timestamp",
        "settings",
        "duration_seconds",
        "critiques",
        "selected",
        "chosen_before_polish",
        "polish",
    }
    assert record["prompt"] == "red plastic"
    assert record["selected"] == {"critic": None, "mobileclip2": None}
    assert record["critiques"] == []  # no refinement
    assert record["polish"] is None  # polish off by default
    assert record["settings"]["critic_render"] is False
    assert record["settings"]["critic_code"] is False
    assert record["settings"]["playbook"] is False
    assert record["settings"]["polish"] is False


def test_select_best_picks_critic_argmax(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    # The MobileCLIP2 selector needs torch + Blender; stub it so this test
    # exercises only the (free, deterministic) critic-score selector.
    monkeypatch.setattr(text2dsl, "_select_by_mobileclip2", lambda *a, **k: None)
    # 2 refine rounds + a trailing final critique = 3 critiques with these scores.
    scores = iter([40, 70, 50])
    monkeypatch.setattr(
        text2dsl,
        "_critique",
        lambda *a, **k: Critique(
            match_score=next(scores), differences=[], suggestions=[]
        ),
    )
    _, record = text_to_material(
        "a material",
        few_shot=False,
        refine=2,
        critic_render=True,
        select_best=True,
        verbose=False,
    )
    # argmax of [40, 70, 50] is index 1 (shallowest-tie rule irrelevant; 70 is unique).
    assert record["selected"]["critic"] == 1
    assert record["selected"]["mobileclip2"] is None


def test_select_best_critic_argmax_breaks_ties_toward_shallowest(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    monkeypatch.setattr(text2dsl, "_select_by_mobileclip2", lambda *a, **k: None)
    scores = iter([70, 70, 30])
    monkeypatch.setattr(
        text2dsl,
        "_critique",
        lambda *a, **k: Critique(
            match_score=next(scores), differences=[], suggestions=[]
        ),
    )
    _, record = text_to_material(
        "a material",
        few_shot=False,
        refine=2,
        critic_render=True,
        select_best=True,
        verbose=False,
    )
    # 70 ties at depths 0 and 1; the shallowest (0) wins.
    assert record["selected"]["critic"] == 0


def test_select_best_single_program_trajectory_skips_selectors(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    called = {"mobileclip2": False}

    def fake_mobileclip(*a, **k):
        called["mobileclip2"] = True
        return 0

    monkeypatch.setattr(text2dsl, "_select_by_mobileclip2", fake_mobileclip)
    _, record = text_to_material(
        "a material", few_shot=False, select_best=True, verbose=False
    )
    # No refinement -> single-program trajectory -> the selectors are skipped
    # (no verifier load, no render); the only candidate is trivially index 0.
    assert not called["mobileclip2"]
    assert record["selected"] == {"critic": 0, "mobileclip2": 0}


# --------------------------------------------------------------------------- #
# _build_critic_system_prompt: (render, code, diagnostics) -> system prompt
# --------------------------------------------------------------------------- #
def test__build_critic_system_prompt_uninformed_no_diagnostics():
    from matloom.engine.text2dsl import _build_critic_system_prompt

    s = _build_critic_system_prompt(render=True, code=False, diagnostics=False)
    assert s.startswith("You are a meticulous material-appearance critic.")
    assert "single rendered image" in s and "target text description" in s
    assert "DSL program source" not in s  # not informed
    assert "appearance scalars" not in s  # no diagnostics
    assert "# Program structure" not in s  # no DSL spec for the uninformed critic
    assert "expert technical artist" not in s  # no generator role


def test__build_critic_system_prompt_informed_appends_dsl_spec():
    from matloom.engine.text2dsl import _build_critic_system_prompt

    s = _build_critic_system_prompt(render=True, code=True, diagnostics=False)
    assert "DSL program source" in s
    assert "# Program structure" in s  # the DSL spec is appended
    assert "line-targeted" in s
    assert "expert technical artist" not in s  # critic role, not generator


def test__build_critic_system_prompt_diagnostics_tells_visual_critic_to_use_scalars():
    from matloom.engine.text2dsl import _build_critic_system_prompt

    s = _build_critic_system_prompt(render=True, code=False, diagnostics=True)
    assert "appearance scalars" in s  # listed in "you are shown"
    assert "use them to ground" in s  # the visual-critic framing
    assert "# Program structure" not in s  # still no DSL spec

    # The informed + diagnostics critic gets both the DSL spec and the framing.
    s2 = _build_critic_system_prompt(render=True, code=True, diagnostics=True)
    assert "use them to ground" in s2 and "# Program structure" in s2


def test__build_critic_system_prompt_none_is_text_only():
    from matloom.engine.text2dsl import _build_critic_system_prompt

    s = _build_critic_system_prompt(render=False, code=False, diagnostics=True)
    assert "no render is shown" in s
    assert "single rendered image" not in s
    assert "reason from the scalars and the description only" in s.lower()
    # The text-only critic is told to reason from scalars; it does NOT get the
    # visual "use them to ground your visual judgment" note.
    assert "use them to ground" not in s
    assert "# Program structure" not in s


def test__build_critic_system_prompt_flat_layout_warns_relief_invisible():
    from matloom.engine.text2dsl import _build_critic_system_prompt

    # The render is always a flat head-on swatch (no scene staging), so the
    # layout note warns relief/reflectance/transmission are barely visible.
    s = _build_critic_system_prompt(render=True, code=False, diagnostics=False)
    assert "head-on" in s
    assert "little relief" in s  # the flat-layout caveat
    assert "upright plane" not in s  # no scene language


def test__build_critic_system_prompt_informed_has_layout_and_spec():
    from matloom.engine.text2dsl import _build_critic_system_prompt

    s = _build_critic_system_prompt(render=True, code=True, diagnostics=False)
    assert "# Program structure" in s  # informed
    assert "head-on" in s  # flat layout note present
    assert "line-targeted" in s


def test_build_critic_system_prompt_no_signal_raises():
    # The builder is unreachable with no signal (_critique guards it out), but it
    # self-defends: all-False raises instead of silently emitting a scalars judge.
    from matloom.engine.text2dsl import _build_critic_system_prompt

    with pytest.raises(ValueError):
        _build_critic_system_prompt(False, False, False)


# --------------------------------------------------------------------------- #
# CMA-ES polish wiring (polish_program is stubbed so no cma/MobileCLIP2 needed)
# --------------------------------------------------------------------------- #
def test_polish_wires_p0_chosen_last_and_picks_best(monkeypatch):
    _patch_build_llm(
        monkeypatch,
        [
            f"<material>\n{_GOOD}\n</material>",
            f"<material>\n{_GOOD2}\n</material>",
        ],
    )
    monkeypatch.setattr(
        text2dsl,
        "_critique",
        lambda *a, **k: Critique(match_score=50, differences=["x"], suggestions=["y"]),
    )
    # Stub the verifier loader so polish=True doesn't pull in torch/MobileCLIP2.
    monkeypatch.setattr(
        text2dsl,
        "_load_mobileclip2",
        lambda: lambda imgs, txt: [0.5] * len(imgs),
    )
    seen: list[str] = []

    def fake_polish(prompt, program, **kw):
        seen.append(program)
        return program + "\n# polished", {
            "config": kw,
            "evals": 10,
            "prompt": prompt,
            "initial_score": 0.5,
            "initial_program": program,
            "final_score": 0.7,
            "final_program": program + "\n# polished",
            "candidates": [{"trajectory": [program], "scores": [0.5]}],
        }

    import matloom.engine.polish as polish_mod

    monkeypatch.setattr(polish_mod, "polish_program", fake_polish)
    traj, record = text_to_material(
        "a material",
        few_shot=False,
        refine=1,
        critic_render=True,
        select_best=False,
        polish=True,
        verbose=False,
    )
    # Refinement produced 2 programs; with select_best off, chosen = last, so
    # polish runs on {p0, last} (deduped) -> two per-program logs.
    assert len(record["polish"]) == 2
    assert {log["index"] for log in record["polish"]} == {0, 1}
    assert seen == [traj[0], traj[1]]  # p0 then last, in index order
    # The best polished program (both tie at 0.7; max is stable -> index 0) is
    # appended to the trajectory as the CLI's final output.
    assert len(traj) == 3
    assert traj[-1].endswith("# polished")
    assert record["settings"]["polish"] is True


def test_polish_off_leaves_record_polish_none(monkeypatch):
    _patch_build_llm(monkeypatch, [f"<material>\n{_GOOD}\n</material>"])
    _, record = text_to_material("a material", few_shot=False, verbose=False)
    assert record["polish"] is None
