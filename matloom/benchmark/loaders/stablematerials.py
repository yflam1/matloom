from pathlib import Path

from matloom.constants import DEFAULT_DATASET_DIR
from matloom.utils.dtypes import PathLike
from matloom.utils.file import read_file

DATASET_NAME = "StableMaterials"


def load(dataset_dir: PathLike | None = None) -> list[str]:
    dataset_dir = (
        DEFAULT_DATASET_DIR if dataset_dir is None else Path(dataset_dir)
    )
    return list(
        read_file(
            dataset_dir / f"{DATASET_NAME}.txt",
            remove_spaces=True,
            remove_empty=True,
        )
    )
