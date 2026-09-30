"""
The dataset from ["Generating Procedural Materials from Text or Image Prompts"](https://doi.org/10.1145/3588432.3591520) (Hu et al., SIGGRAPH 2023).

The supplemental zip is served behind a Cloudflare challenge, so it cannot be
fetched programmatically. Download it manually from the ACM Digital Library
supplemental materials link and place it at
`<out_dir>/GenProc/siggraph23conferenceproceedings-40_supplemental_materials.zip`.
"""

from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Literal, overload

from PIL import Image

from matloom.constants import DEFAULT_DATASET_DIR
from matloom.utils.dtypes import PathLike, SDict

DATASET_NAME = "GenProc"
GENPROC_URL = "https://dl.acm.org/doi/suppl/10.1145/3588432.3591520/suppl_file/siggraph23conferenceproceedings-40_supplemental_materials.zip"


def _load_zip(zip_path: Path) -> SDict[list[bytes]]:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{DATASET_NAME} zip not found at {zip_path}. Download it from\n"
            f"  {GENPROC_URL}\n"
            f"and place it at {zip_path}."
        )
    from zipfile import ZipFile

    data: SDict[list[bytes]] = defaultdict(list)
    with ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/"):
                continue
            parts = PurePosixPath(name).parts
            if "Text Prompts" not in parts:
                continue
            if not parts[-1].lower().endswith(".jpg"):
                continue
            prompt = parts[parts.index("Text Prompts") + 1]
            data[prompt].append(zf.read(name))
    return data


def prepare(out_dir: PathLike | None = None) -> Path:
    out_dir = DEFAULT_DATASET_DIR if out_dir is None else Path(out_dir)
    out_path = out_dir / f"{DATASET_NAME}.json.gz"
    if out_path.exists():
        return out_path
    from tqdm import tqdm

    from matloom.metrics.clipscore import ClipScore
    from matloom.utils.file import save_json
    from matloom.utils.misc import bytes_to_base64

    zip_path = out_dir / DATASET_NAME / GENPROC_URL.split("/")[-1]
    data = _load_zip(zip_path)
    clipscore = ClipScore.OpenClip()

    def find_best_image(images: list[bytes], prompt: str) -> bytes:
        result = clipscore(images, [f"A photo of {prompt}"], progress=False)
        best = result.scores.argmax().item()
        return images[best]

    out = {
        prompt: bytes_to_base64(find_best_image(images, prompt))
        for prompt, images in tqdm(data.items(), desc=DATASET_NAME)
    }
    out_dir.mkdir(exist_ok=True)
    save_json(out, out_path, indent=2)
    return out_path


@overload
def load(
    dataset_dir: PathLike | None = ..., full: Literal[False] = False
) -> list[str]: ...
@overload
def load(
    dataset_dir: PathLike | None = ..., full: Literal[True] = True
) -> SDict[Image.Image]: ...
def load(
    dataset_dir: PathLike | None = None, full: bool = False
) -> list[str] | SDict[Image.Image]:
    from matloom.utils.file import load_json

    data: SDict[str] = load_json(prepare(dataset_dir))
    if full:
        import io

        from matloom.utils.misc import base64_to_bytes

        return {
            prompt: Image.open(io.BytesIO(base64_to_bytes(b64)))
            for prompt, b64 in data.items()
        }
    else:
        return list(data.keys())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()

    prepare(args.dataset_dir)
