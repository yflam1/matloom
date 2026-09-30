from pathlib import Path
from typing import Annotated, Any, TypeVar

import numpy as np
from PIL import Image
from pydantic import AfterValidator, StringConstraints

UNSET = object()  # Sentinel for unset values

PathLike = Path | str
Url = str
ImgLike = Image.Image | np.ndarray | Url | bytes | PathLike

# Type for dictionary with string as key
T = TypeVar("T")
SDict = dict[str, T]

JsonObject = Any
YamlObject = Any
NonEmptyStr = Annotated[str, StringConstraints(strict=True, min_length=1)]


def _check_non_zero(v: float) -> float:
    if v == 0.0:
        raise ValueError(f"Value cannot be exactly zero, got {v}")
    return v


NonZeroFloat = Annotated[float, AfterValidator(_check_non_zero)]
