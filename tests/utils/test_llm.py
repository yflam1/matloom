"""Tests for matloom.utils.llm — the OpenAI-compatible chat client.

These exercise the conversation/branching/concurrency layer without hitting a
real endpoint: the ``Llm._client`` is replaced with a fake that returns canned
completions. ``openai``/``json_repair`` are lightweight pure-Python deps in
requirements-test.txt; the module is skipped if they are somehow unavailable.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("openai")
pytest.importorskip("json_repair")

from matloom.utils.llm import JsonResponseModel, Llm, LlmChatError, concurrent_map
from matloom.utils.llm.msg import Messages


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def _make_completion(
    content: str,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
    logprobs_content=None,
):
    """Build a duck-typed stand-in for openai's ChatCompletion."""
    message = SimpleNamespace(content=content)
    logprobs = (
        None if logprobs_content is None else SimpleNamespace(content=logprobs_content)
    )
    choice = SimpleNamespace(
        message=message, finish_reason=finish_reason, logprobs=logprobs
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_no_choices_completion():
    """A completion whose ``choices`` field is None: some providers return
    ``choices: null`` under load, which the SDK passes through as None."""
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=None, usage=usage)


class FakeCompletions:
    def __init__(self, responder):
        self._responder = responder
        self.calls = []
        self._lock = threading.Lock()

    def create(self, *, messages, stream, **params):
        with self._lock:
            self.calls.append({"messages": messages, "params": params})
        return self._responder(messages, params)


class FakeClient:
    def __init__(self, responder):
        self.chat = SimpleNamespace(completions=FakeCompletions(responder))


def make_llm(responder, **kwargs):
    """An ``Llm`` whose network client is replaced by a fake responder."""
    kwargs.setdefault("api_key", "sk-test")
    llm = Llm("fake-model", **kwargs)
    llm._client = FakeClient(responder)
    return llm


def echo_responder(reply="hello"):
    def _responder(messages, params):
        return _make_completion(reply)

    return _responder


# --------------------------------------------------------------------------- #
# Basic completion
# --------------------------------------------------------------------------- #
def test_one_shot_call_returns_content():
    llm = make_llm(echo_responder("world"))
    out = llm("hi", verbose=False)
    assert out.response == "world"
    assert out.content == "world"
    assert out.finish_reason == "stop"
    assert out.prompt_tokens == 10
    assert out.completion_tokens == 5


def test_one_shot_is_stateless():
    """Each __call__ is an independent conversation: no leaked history."""
    llm = make_llm(echo_responder("a"))
    llm("first", verbose=False)
    llm("second", verbose=False)
    # Two separate one-turn dialogues => two completion calls, each with only
    # the system + a single user message.
    completions = llm._client.chat.completions
    assert len(completions.calls) == 2
    for call in completions.calls:
        roles = [m["role"] for m in call["messages"]]
        assert roles == ["system", "user"]


def test_cost_accumulates_thread_safely():
    llm = make_llm(echo_responder("x"), input_cost=1.0, output_cost=2.0)
    # 10 prompt * 1/1e6 + 5 completion * 2/1e6
    out = llm("hi", verbose=False)
    expected = 10 * 1e-6 + 5 * 2e-6
    assert out.cost == pytest.approx(expected)
    llm("hi", verbose=False)
    assert llm.cost == pytest.approx(2 * expected)


# --------------------------------------------------------------------------- #
# Conversations & multi-turn
# --------------------------------------------------------------------------- #
def test_conversation_accumulates_turns():
    replies = iter(["r1", "r2"])

    def responder(messages, params):
        return _make_completion(next(replies))

    llm = make_llm(responder)
    convo = llm.conversation(sys_prompt="be brief", verbose=False)
    convo.send("turn one")
    convo.send("turn two")
    roles = [m.role for m in convo.messages.messages]
    assert roles == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    # The second request must carry the full prior context (system + first
    # exchange + the new user turn; the new assistant reply is appended after).
    last_call = llm._client.chat.completions.calls[-1]
    assert len(last_call["messages"]) == 4


def test_conversation_each_has_own_system_prompt():
    llm = make_llm(echo_responder("ok"))
    a = llm.conversation(sys_prompt="system A", verbose=False)
    b = llm.conversation(sys_prompt="system B", verbose=False)
    assert a.messages.messages[0].content == "system A"
    assert b.messages.messages[0].content == "system B"


def test_send_rolls_back_user_turn_on_failure():
    def boom(messages, params):
        raise LlmChatError("LLM response is None")

    llm = make_llm(boom)
    convo = llm.conversation(sys_prompt="s", verbose=False)
    with pytest.raises(LlmChatError):
        convo.send("hello", max_parse_retries=0)
    # The user turn must have been removed so the dialogue stays consistent.
    assert [m.role for m in convo.messages.messages] == ["system"]


# --------------------------------------------------------------------------- #
# Branching
# --------------------------------------------------------------------------- #
def test_branch_full_copy_is_independent():
    llm = make_llm(echo_responder("ok"))
    convo = llm.conversation(sys_prompt="s", verbose=False)
    convo.send("a")
    fork = convo.branch()
    fork.send("b only on fork")
    # Original is unchanged by the fork's extra turn.
    assert len(convo) == 3  # system + user + assistant
    assert len(fork) == 5
    assert convo.messages.messages is not fork.messages.messages


def test_branch_at_index_truncates():
    llm = make_llm(echo_responder("ok"))
    convo = llm.conversation(sys_prompt="s", verbose=False)
    convo.send("a")
    convo.send("b")
    assert len(convo) == 5
    fork = convo.branch(at=1)  # keep only the system message
    assert [m.role for m in fork.messages.messages] == ["system"]
    # Forking does not disturb the source.
    assert len(convo) == 5


def test_branch_at_too_large_raises():
    llm = make_llm(echo_responder("ok"))
    convo = llm.conversation(sys_prompt="s", verbose=False)
    with pytest.raises(ValueError):
        convo.branch(at=99)


# --------------------------------------------------------------------------- #
# Retry semantics
# --------------------------------------------------------------------------- #
def test_semantic_failure_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def responder(messages, params):
        calls["n"] += 1
        if calls["n"] == 1:
            return _make_completion(None)  # triggers LlmChatError
        return _make_completion("recovered")

    llm = make_llm(responder)
    out = llm("hi", verbose=False, max_parse_retries=2)
    assert out.response == "recovered"
    assert calls["n"] == 2


def test_empty_string_content_is_retried(monkeypatch):
    # Some providers (e.g. gemini under load) return an empty/blank string
    # rather than None; it must be treated as a transient failure and retried,
    # not passed through to fail downstream validation.
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def responder(messages, params):
        calls["n"] += 1
        if calls["n"] == 1:
            return _make_completion("")  # empty string
        if calls["n"] == 2:
            return _make_completion("   \n ")  # whitespace-only
        return _make_completion("recovered")

    llm = make_llm(responder)
    out = llm("hi", verbose=False, max_parse_retries=3)
    assert out.response == "recovered"
    assert calls["n"] == 3


def test_semantic_failure_exhausts_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    def responder(messages, params):
        return _make_completion(None)

    llm = make_llm(responder)
    with pytest.raises(LlmChatError):
        llm("hi", verbose=False, max_parse_retries=2)
    # 1 initial + 2 retries
    assert len(llm._client.chat.completions.calls) == 3


def test_none_choices_is_retried_then_succeeds(monkeypatch):
    # Some providers return ``choices: null`` under load; it must be treated
    # as a transient failure and retried, not crash with a TypeError.
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def responder(messages, params):
        calls["n"] += 1
        if calls["n"] == 1:
            return _make_no_choices_completion()
        return _make_completion("recovered")

    llm = make_llm(responder)
    out = llm("hi", verbose=False, max_parse_retries=2)
    assert out.response == "recovered"
    assert calls["n"] == 2


def test_none_choices_exhausts_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    def responder(messages, params):
        return _make_no_choices_completion()

    llm = make_llm(responder)
    with pytest.raises(LlmChatError):
        llm("hi", verbose=False, max_parse_retries=2)
    # 1 initial + 2 retries
    assert len(llm._client.chat.completions.calls) == 3


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #
class _Answer(JsonResponseModel):
    value: int


def test_response_model_parsing():
    def responder(messages, params):
        return _make_completion('Here you go: {"value": 42} done')

    llm = make_llm(responder)
    out = llm("give me json", _Answer, verbose=False)
    assert isinstance(out.response, _Answer)
    assert out.response.value == 42


# --------------------------------------------------------------------------- #
# Vision (image attachments on user turns)
# --------------------------------------------------------------------------- #
def test_user_turn_can_attach_image():
    captured = {}

    def responder(messages, params):
        captured["user"] = messages[-1]["content"]
        return _make_completion("seen")

    llm = make_llm(responder)
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    out = llm("describe", images=[img], verbose=False)
    assert out.response == "seen"
    parts = captured["user"]
    assert any(p.get("type") == "image_url" for p in parts)
    url = next(p for p in parts if p.get("type") == "image_url")
    assert url["image_url"]["url"].startswith("data:image/jpeg;base64,")


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def test_concurrent_map_preserves_order():
    def work(x):
        # Sleep inversely so out-of-order completion is likely if unordered.
        time.sleep(0.01 * (5 - x))
        return x * x

    assert concurrent_map(work, [1, 2, 3, 4, 5], max_concurrency=5) == [
        1,
        4,
        9,
        16,
        25,
    ]


def test_concurrent_map_empty():
    assert concurrent_map(lambda x: x, [], max_concurrency=4) == []


def test_concurrent_map_bounds_concurrency():
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def work(x):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.02)
        with lock:
            active["now"] -= 1
        return x

    concurrent_map(work, list(range(20)), max_concurrency=3)
    assert active["max"] <= 3


def test_batch_runs_all_prompts_concurrently():
    def responder(messages, params):
        # echo back the user prompt text
        user = messages[-1]["content"]
        text = user[0]["text"] if isinstance(user, list) else user
        return _make_completion(f"reply:{text}")

    llm = make_llm(responder)
    outs = llm.batch(["p0", "p1", "p2"], max_concurrency=3)
    assert [o.response for o in outs] == ["reply:p0", "reply:p1", "reply:p2"]


def test_many_conversations_concurrently_are_isolated():
    def responder(messages, params):
        user = messages[-1]["content"]
        text = user[0]["text"] if isinstance(user, list) else user
        return _make_completion(f"echo:{text}")

    llm = make_llm(responder)

    def run(i):
        convo = llm.conversation(sys_prompt=f"sys{i}", verbose=False)
        out = convo.send(f"msg{i}")
        # Each conversation keeps its own system prompt and single exchange.
        assert convo.messages.messages[0].content == f"sys{i}"
        assert len(convo) == 3
        return out.response

    results = concurrent_map(run, list(range(25)), max_concurrency=8)
    assert results == [f"echo:msg{i}" for i in range(25)]


# --------------------------------------------------------------------------- #
# Messages helpers (branching primitives)
# --------------------------------------------------------------------------- #
def test_messages_copy_is_deep():
    msgs = Messages()
    msgs.set_system("s")
    msgs.add_user("u")
    clone = msgs.copy()
    clone.add_assistant("a")
    assert len(msgs) == 2
    assert len(clone) == 3


def test_messages_truncate_and_counts():
    msgs = Messages()
    msgs.set_system("s")
    msgs.add_user("u1")
    msgs.add_assistant("a1")
    msgs.add_user("u2")
    assert msgs.has_system
    assert msgs.assistant_count == 1
    msgs.truncate(2)
    assert [m.role for m in msgs.messages] == ["system", "user"]
