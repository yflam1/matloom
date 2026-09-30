"""
matloom.utils.llm — a thread-safe client over an OpenAI-compatible Chat
Completions endpoint.

Kept on Chat Completions (not the Responses API) so non-OpenAI ``base_url``
providers keep working. The split:

- :mod:`matloom.utils.llm.base` (``OpenAiApi``) — owns the shared ``OpenAI``
  client + a lock-guarded cost total.
- :mod:`matloom.utils.llm.llm` (``Llm``) — a stateless client/factory:
  ``conversation(...)`` spawns a :class:`Conversation`, ``__call__``/
  ``complete(...)`` run one-shot completions, ``batch(...)`` fans many out
  concurrently.
- :mod:`matloom.utils.llm.conversation` (``Conversation``) — one stateful
  multi-turn dialogue: its own system prompt + ``Messages`` (each user turn may
  attach images for vision models) + history, all guarded by an ``RLock``, with
  ``branch(at=None)`` to fork an independent copy (deep-copy the message list,
  optionally truncated to ``at`` leading messages).
- :mod:`matloom.utils.llm.concurrency` — ``concurrent_map``, a bounded
  ``ThreadPoolExecutor`` helper (``max_concurrency`` caps in-flight requests).
- :mod:`matloom.utils.llm.output` — ``ChatParameters``/``LlmOutput``/
  ``LlmChatError``.
- :mod:`matloom.utils.llm.response` — the ``ResponseModel`` JSON structured-output
  parsing (brace-counting + ``json_repair``, provider-agnostic), plus
  ``JsonResponseModel`` for single-object responses.
- :mod:`matloom.utils.llm.msg`, :mod:`matloom.utils.llm.parser`,
  :mod:`matloom.utils.llm.template`, :mod:`matloom.utils.llm.embedder` — message
  models, prompt templates, and the embeddings client.

Retries are split correctly: transport errors (429/5xx/connection) are left to
the SDK's own ``max_retries``; only *semantic* failures (empty response / parse
error) are retried locally. Unit-tested torch-free via a mock client
(``tests/utils/test_llm.py``); the embeddings/heavier paths are not.
"""

from .concurrency import concurrent_map
from .conversation import Conversation
from .llm import DEFAULT_SYSTEM_PROMPT, Llm
from .output import ChatParameters, LlmChatError, LlmOutput
from .response import (
    ZERO_SHOT_COT_REASONING,
    JsonResponseModel,
    ResponseModel,
    ResponseModelParsingError,
)
from .template import CoStar, PromptTemplate

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ZERO_SHOT_COT_REASONING",
    "ChatParameters",
    "CoStar",
    "Conversation",
    "JsonResponseModel",
    "Llm",
    "LlmChatError",
    "LlmOutput",
    "PromptTemplate",
    "ResponseModel",
    "ResponseModelParsingError",
    "concurrent_map",
]
