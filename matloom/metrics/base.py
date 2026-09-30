from collections.abc import Callable
from typing import Any

import torch
from pydantic import BaseModel


class VlmResult(BaseModel, arbitrary_types_allowed=True):
    scores: torch.Tensor  # [len(texts), len(images)]


def _invalid_metric_name(name: str) -> str:
    return f"Invalid metric name: {name}"


def load_metric(name: str, **kwargs: Any) -> Callable[..., VlmResult]:
    parts = name.split(".")
    parts[0] = parts[0].lower()
    if parts[0] == "blipscore":
        if len(parts) != 1:
            raise ValueError(_invalid_metric_name(name))
        from .blipscore import BlipScore

        return BlipScore(**kwargs)
    if parts[0] == "clipiqa":
        if len(parts) != 1:
            raise ValueError(_invalid_metric_name(name))
        from .clipiqa import ClipIqa

        return ClipIqa(**kwargs)
    if parts[0] == "clipscore":
        if len(parts) != 2:
            raise ValueError(_invalid_metric_name(name))
        from .clipscore import ClipScore

        if not hasattr(ClipScore, parts[1]):
            raise ValueError(_invalid_metric_name(name))
        return getattr(ClipScore, parts[1])(**kwargs)
    if parts[0] == "vqascore":
        if len(parts) != 2:
            raise ValueError(_invalid_metric_name(name))
        from . import vqascore

        if not hasattr(vqascore, parts[1]):
            raise ValueError(_invalid_metric_name(name))
        return getattr(vqascore, parts[1])(**kwargs)
    raise ValueError(_invalid_metric_name(name))
