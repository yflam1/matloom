import argparse
import gc
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from lxml import etree

from matloom import logger
from matloom.metrics.base import load_metric
from matloom.render import program_to_material
from matloom.utils.dtypes import SDict
from matloom.utils.file import load_json, save_json
from matloom.utils.llm import Llm
from matloom.utils.progress import progress_bar

METRICS = [
    "blipscore",
    "clipiqa",
    "clipscore.OpenClip",
    "vqascore.VqaScore",
    "vqascore.VqaDirect",
]


def main(base_dir: Path) -> None:
    if not base_dir.is_dir():
        raise ValueError(f"Invalid base directory: {base_dir}")
    now = datetime.now(timezone.utc).strftime(r"%Y%m%d-%H%M%S-%f")
    out_path = Path("experiments") / "results" / f"ours_ablations_{now}.json"
    ablations = sorted(p for p in base_dir.iterdir() if p.is_dir())
    prompt_to_imgs: SDict[list[tuple[str, str, Path]]] = defaultdict(list)
    result = {}
    with progress_bar() as pbar:
        tid_abla = pbar.add_task("Ablations", total=len(ablations))
        for a in ablations:
            name = a.name
            result[name] = {
                "plane": defaultdict(dict),
                "scene": defaultdict(dict),
            }
            records = sorted(a.glob("**/record.json"))
            tid_record = pbar.add_task(name, total=len(records))
            for r in records:
                info = load_json(r)
                prompt = info.get("prompt", None)
                if prompt is None:
                    raise RuntimeError(f"Missing prompt in {r}")
                parent = r.parent
                traj = parent / "trajectory.xml"
                if not traj.is_file():
                    raise RuntimeError(f"Missing trajectory.xml in {parent}")
                prog: str = (
                    etree.parse(traj)
                    .getroot()
                    .xpath("//program")[-1]
                    .text.strip()
                )
                plane_paths = sorted(parent.glob("render_plane*.png"))
                plane_path = (
                    plane_paths[0]
                    if len(plane_paths) > 0
                    else program_to_material(prog, parent, scene=False)[0]
                )
                scene_paths = sorted(parent.glob("render_scene*.png"))
                scene_path = (
                    scene_paths[0]
                    if len(scene_paths) > 0
                    else program_to_material(prog, parent, scene=True)[0]
                )
                prompt_to_imgs[prompt].append((name, "plane", plane_path))
                prompt_to_imgs[prompt].append((name, "scene", scene_path))

                # Advance progress bar
                pbar.advance(tid_record)
            pbar.remove_task(tid_record)
            pbar.advance(tid_abla)
        pbar.remove_task(tid_abla)

    with progress_bar() as pbar:
        tid_metric = pbar.add_task("Metrics", total=len(METRICS))
        for metric in METRICS:
            logger.info(f"Loading {metric}")
            evaluator = load_metric(
                metric,
                llm=Llm("claude-sonnet-5", max_retries=10),
                reasoning_effort="medium",
            )
            logger.info(f"Loaded {metric}")

            tid_prompt = pbar.add_task("Prompts", total=len(prompt_to_imgs))
            for prompt, imgs in prompt_to_imgs.items():
                scores = (
                    evaluator(
                        images=[img[2] for img in imgs],
                        texts=[
                            f"A photo of {prompt}"
                            if metric.startswith("clipscore")
                            else prompt
                        ],
                    )
                    .scores[0]
                    .tolist()
                )
                for (name, style, img), score in zip(imgs, scores):
                    result[name][style][metric][prompt] = score
                save_json(result, out_path, indent=2)
                pbar.advance(tid_prompt)
            pbar.remove_task(tid_prompt)

            # Clean up to free memory
            del evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # Advance progress bar
            pbar.advance(tid_metric)
        pbar.remove_task(tid_metric)
        logger.info("Metrics done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("experiments") / "runs" / "ours" / "ablations",
    )
    args = parser.parse_args()
    main(args.base_dir)
