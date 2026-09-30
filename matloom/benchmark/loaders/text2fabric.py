"""
The [Text2Fabric](https://valentin.deschaintre.fr/text2fabric) dataset.

``descriptions.csv`` (prompts) and ``renderings.zip`` (512x512 swatches) are
fetched from S3 on first build into ``<out_dir>/text2fabric/`` and need no
manual download. Image bytes are read directly from the zip in memory (no
extraction); only the ``renderings/plane/baseline_illumination`` subtree is
used and the other geometries/lightings are skipped.

Each fabric group yields multiple (prompt, image) candidates; the prompt best
matching the image is selected via CLIPScore. With ``one_per_group`` a
SentenceBERT pass further picks one entry per group.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Literal, overload

from PIL import Image

from matloom.constants import DEFAULT_DATASET_DIR
from matloom.utils.dtypes import PathLike, SDict

DATASET_NAME = "text2fabric"
DESCRIPTIONS_URL = (
    "https://language-fabric-pub.s3.us-west-2.amazonaws.com/descriptions.csv"
)
RENDERINGS_URL = (
    "https://language-fabric-pub.s3.us-west-2.amazonaws.com/renderings.zip"
)
# Subtree of the renderings zip that is used (read in memory, not extracted).
RENDERINGS_SUBDIR = "renderings/plane/baseline_illumination"


def _download(url: str, dest: Path) -> None:
    from urllib.request import urlretrieve

    from tqdm import tqdm

    with tqdm(
        desc=f"Downloading {dest.name}", unit="B", unit_scale=True
    ) as pbar:

        def reporthook(
            _block_num: int, block_size: int, total_size: int
        ) -> None:
            if pbar.total is None and total_size > 0:
                pbar.total = total_size
            pbar.update(block_size)

        urlretrieve(url, str(dest), reporthook=reporthook)


def _maybe_download(out_dir: Path) -> tuple[Path, Path]:
    """Download ``descriptions.csv`` and ``renderings.zip`` into
    ``out_dir/DATASET_NAME`` if absent. Returns ``(csv_path, zip_path)``; the
    zip is read in memory by :func:`prepare`, not extracted."""
    base = out_dir / DATASET_NAME
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "descriptions.csv"
    zip_path = base / "renderings.zip"
    if not csv_path.exists():
        _download(DESCRIPTIONS_URL, csv_path)
    if not zip_path.exists():
        _download(RENDERINGS_URL, zip_path)
    return csv_path, zip_path


def _load_data(csv_path: Path, present: set[str]) -> SDict[list[str]]:
    import pandas as pd

    def handle_punctuation(prompt: str, match: re.Match) -> str:
        full_match = match.group(0)
        start_index = match.start()

        # Check if it's preceded by "i.e"
        if prompt[start_index - 3 : start_index].lower() == "i.e":
            return full_match

        # Look for the next non-space character
        suffix = prompt[match.end() :]
        next_char_search = re.search(r"\S", suffix)

        if next_char_search:
            next_char = next_char_search.group(0)
            if next_char.islower():
                return ","
        return "."

    df = pd.read_csv(csv_path)
    data = defaultdict(list)
    for idx, row in df.iterrows():
        prompt: str = row["fabric_description_prod"]
        image: str = row["image_url"]
        filename = image.split("_4096")[0]

        # Filter out entries without a matching 512 rendering in the zip
        if (
            "_4096" not in image
            or not image.endswith(".png")
            or filename not in present
        ):
            continue

        # Clean up the prompt
        prompt = prompt.replace("\r\n", " ").replace("\n", " ")
        prompt = prompt.replace("\\", "")
        prompt = prompt.replace(",.", ".")
        prompt = prompt.replace(" .", ". ")
        prompt = prompt.replace(" ,", ", ")
        prompt = prompt.strip(' "')
        prompt = re.sub(
            r"\.,", lambda m, p=prompt: handle_punctuation(p, m), prompt
        )
        prompt = re.sub(r" +", " ", prompt)
        if not prompt.endswith((".", "!", "?")):
            prompt += "."

        data[filename].append(prompt)

    return data


def prepare(
    out_dir: PathLike | None = None, one_per_group: bool = False
) -> Path:
    out_dir = DEFAULT_DATASET_DIR if out_dir is None else Path(out_dir)
    out_path = out_dir / f"{DATASET_NAME}.json.gz"
    if out_path.exists():
        return out_path

    csv_path, zip_path = _maybe_download(out_dir)

    from zipfile import ZipFile

    from tqdm import tqdm

    from matloom.metrics.clipscore import ClipScore
    from matloom.utils.file import save_json
    from matloom.utils.misc import bytes_to_base64

    with ZipFile(zip_path) as zf:
        present = {
            m[len(RENDERINGS_SUBDIR) + 1 : -len("_512.png")]
            for m in zf.namelist()
            if m.startswith(f"{RENDERINGS_SUBDIR}/") and m.endswith("_512.png")
        }
        data = _load_data(csv_path, present)
        clipscore = ClipScore.OpenClip()

        def find_best_prompt(img_bytes: bytes, prompts: list[str]) -> str:
            result = clipscore(
                [img_bytes],
                [f"A photo of {prompt}" for prompt in prompts],
                progress=False,
            )
            best = result.scores.argmax().item()
            return prompts[best]

        final: SDict[list[SDict[str]]] = defaultdict(list)
        for name, prompts in tqdm(data.items(), desc=DATASET_NAME):
            group, _ = name.rsplit("_", maxsplit=1)
            img_bytes = zf.read(f"{RENDERINGS_SUBDIR}/{name}_512.png")
            best = find_best_prompt(img_bytes, prompts)
            final[group].append(
                {
                    "name": name,
                    "prompt": best,
                    "base64": bytes_to_base64(img_bytes),
                }
            )

    if one_per_group:
        from matloom.models.sbert import SentenceBert

        sbert = SentenceBert()

        def pick_one(group: str, prompts: list[str]) -> int:
            grp = group.replace("_", " ")
            result = sbert(sbert.encode(prompts), sbert.encode([grp]))
            return result.argmax().item()

        for group, entries in final.items():
            prompts = [entry["prompt"] for entry in entries]
            idx = pick_one(group, prompts)
            final[group] = [entries[idx]]

    out_dir.mkdir(exist_ok=True)
    save_json(dict(final), out_path, indent=2)
    return out_path


@overload
def load(
    dataset_dir: PathLike | None = ..., full: Literal[False] = False
) -> list[str]: ...
@overload
def load(
    dataset_dir: PathLike | None = ..., full: Literal[True] = True
) -> SDict[list[SDict[str | Image.Image]]]: ...
def load(
    dataset_dir: PathLike | None = None, full: bool = False
) -> list[str] | SDict[list[SDict[str | Image.Image]]]:
    from matloom.utils.file import load_json

    data = load_json(prepare(dataset_dir))
    if full:
        import io

        from matloom.utils.misc import base64_to_bytes

        return {
            group: [
                {
                    "name": entry["name"],
                    "prompt": entry["prompt"],
                    "image": Image.open(
                        io.BytesIO(base64_to_bytes(entry["base64"]))
                    ),
                }
                for entry in entries
            ]
            for group, entries in data.items()
        }
    else:
        return [
            entry["prompt"] for entries in data.values() for entry in entries
        ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()

    prepare(args.dataset_dir)
