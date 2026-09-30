"""matloom.metrics — image/prompt scoring used by the evaluation pipeline.

- :mod:`matloom.metrics.clipscore` (``CLIPScore``), :mod:`matloom.metrics.blipscore`
  (``BLIPScore`` ITM head), :mod:`matloom.metrics.clipiqa`, and
  :mod:`matloom.metrics.vqascore` (t2v-metrics) — the four metrics. ``base.py``
  imports torch. Console entry points: ``clipscore``, ``blipscore`` (see
  ``pyproject.toml``).
- :mod:`matloom.metrics.vqascore` also holds the **LLM-judge** evaluators
  (``VqaDirect`` / ``VqaMultiPerspectives``, reasoning-based scoring over any
  :class:`matloom.utils.llm.Llm`); the ``vqascore.VqaDirect`` metric in the
  ``matloom/eval/eval_*.py`` scripts (shown as ``MLLM Judge`` by
  ``print_tables``) reuses ``VqaDirect`` with ``claude-sonnet-5``.
"""
