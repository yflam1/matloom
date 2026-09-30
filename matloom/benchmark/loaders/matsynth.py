from pathlib import Path
from typing import Literal, overload

from PIL import Image

from matloom.constants import DEFAULT_DATASET_DIR
from matloom.utils.dtypes import PathLike, SDict

DATASET_NAME = "MatSynth"


def prepare(out_dir: PathLike | None = None) -> Path:
    out_dir = DEFAULT_DATASET_DIR if out_dir is None else Path(out_dir)
    out_path = out_dir / f"{DATASET_NAME}.json.gz"
    if out_path.exists():
        return out_path
    from tempfile import NamedTemporaryFile, TemporaryDirectory

    from tqdm import tqdm

    from datasets import load_dataset
    from matloom.render import textures_to_material
    from matloom.utils.file import save_json
    from matloom.utils.misc import bytes_to_base64

    # `load_dataset("gvecchio/MatSynth", split="test")` prepares the whole config
    # (train + test, ~437 GB) before applying `split=`, and the card's dataset_info
    # declares both splits so `data_files` restriction trips `ExpectedMoreSplitsError`.
    # Load the 8 test shards directly via the parquet builder: no predeclared splits,
    # single split named "train", ~7.4 GB total.
    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/gvecchio/MatSynth/data/test-*.parquet",
        split="train",
    )
    out: SDict[SDict[str]] = {}
    for datum in tqdm(ds, desc=DATASET_NAME):
        prompt: str = datum["metadata"]["description"]
        if prompt == "":
            continue
        with (
            NamedTemporaryFile(suffix=".png") as tmp1,
            NamedTemporaryFile(suffix=".png") as tmp2,
            NamedTemporaryFile(suffix=".png") as tmp3,
            NamedTemporaryFile(suffix=".png") as tmp4,
            NamedTemporaryFile(suffix=".png") as tmp5,
            NamedTemporaryFile(suffix=".png") as tmp6,
            NamedTemporaryFile(suffix=".png") as tmp7,
            TemporaryDirectory() as tmpdir,
        ):
            datum["basecolor"].save(tmp1.name)
            datum["displacement"].save(tmp2.name)
            datum["height"].save(tmp3.name)
            datum["metallic"].save(tmp4.name)
            datum["normal"].save(tmp5.name)
            datum["roughness"].save(tmp6.name)
            datum["specular"].save(tmp7.name)
            renders = textures_to_material(
                tmpdir,
                basecolor=tmp1.name,
                metallic=tmp4.name,
                roughness=tmp6.name,
                specular=tmp7.name,
                normal=tmp5.name,
                height=(tmp3.name, 0.1),
                displacement=(tmp2.name, 0.5, 0.1),
                single_view=True,
            )
            with renders[0].open("rb") as f:
                base64 = bytes_to_base64(f.read())
        out[datum["name"]] = {"prompt": prompt, "base64": base64}

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
) -> SDict[SDict[str | Image.Image]]: ...
def load(
    dataset_dir: PathLike | None = None, full: bool = False
) -> list[str] | SDict[SDict[str | Image.Image]]:
    from matloom.utils.file import load_json

    data: SDict[SDict[str]] = load_json(prepare(dataset_dir))
    if full:
        import io

        from matloom.utils.misc import base64_to_bytes

        return {
            name: {
                "prompt": entry["prompt"],
                "image": Image.open(
                    io.BytesIO(base64_to_bytes(entry["base64"]))
                ),
            }
            for name, entry in data.items()
        }
    else:
        return [v["prompt"] for v in data.values()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()

    prepare(args.dataset_dir)
