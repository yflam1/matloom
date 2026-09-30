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
    out_path = Path("experiments") / "results" / f"ours_main_all_{now}.json"
    mains = sorted(
        p for p in base_dir.iterdir() if p.is_dir() and p.name == "gemini"
    )
    prompt_to_imgs: SDict[list[tuple[str, str, str, Path, bool]]] = (
        defaultdict(list)
    )
    result = {}
    with progress_bar() as pbar:
        tid_main = pbar.add_task("Main", total=len(mains))
        for m in mains:
            name = m.name
            result[name] = {
                "plane": defaultdict(
                    lambda: defaultdict(lambda: defaultdict(list))
                ),
                "scene": defaultdict(
                    lambda: defaultdict(lambda: defaultdict(list))
                ),
                "idx": defaultdict(dict),
                "mobileclip2": {},
            }
            records = sorted(m.glob("**/record.json"))
            tid_record = pbar.add_task(name, total=len(records))
            for r in records:
                info = load_json(r)
                prompt: str | None = info.get("prompt", None)
                seed = info.get("settings", {}).get("seed", None)
                if prompt is None:
                    raise RuntimeError(f"Missing prompt in {r}")
                if seed is None:
                    raise RuntimeError(f"Missing seed in {r}")
                seed = str(seed)
                parent = r.parent
                traj = parent / "trajectory.xml"
                if not traj.is_file():
                    raise RuntimeError(f"Missing trajectory.xml in {parent}")
                xpath = etree.parse(traj).getroot().xpath("//program")
                final_prog: str = xpath[-1].text.strip()
                final_idx: int | None = None
                progs: list[str] = [xpath[i].text.strip() for i in range(6)]
                for polished in info["polish"]:
                    cand: str = polished["final_program"].strip()
                    if cand == final_prog:
                        if final_idx is not None:
                            raise RuntimeError(
                                f"Duplicate final program in {r}"
                            )
                        final_idx = len(progs)
                    progs.append(cand)
                if final_idx is None:
                    raise RuntimeError(f"Final program not found in {r}")
                if final_idx < 6:
                    raise RuntimeError(
                        f"Final program index is less than 6 in {r}"
                    )
                before_idx: int | None = info.get("chosen_before_polish", None)
                if before_idx is None:
                    raise RuntimeError(f"Missing chosen_before_polish in {r}")
                if before_idx == 0 or before_idx == 5:
                    if len(progs) != 8:
                        raise RuntimeError(
                            f"Unexpected number of programs in {r}"
                        )
                else:
                    if len(progs) != 9:
                        raise RuntimeError(
                            f"Unexpected number of programs in {r}"
                        )
                result[name]["idx"][prompt][seed] = {
                    "before": before_idx,
                    "final": final_idx,
                }
                for i, prog in enumerate(progs):
                    out_dir = parent / "renders" / f"program{i:02d}"
                    plane_paths = sorted(out_dir.glob("render_plane*.png"))
                    plane_path = (
                        plane_paths[0]
                        if len(plane_paths) > 0
                        else program_to_material(prog, out_dir, scene=False)[0]
                    )
                    scene_paths = sorted(out_dir.glob("render_scene*.png"))
                    scene_path = (
                        scene_paths[0]
                        if len(scene_paths) > 0
                        else program_to_material(prog, out_dir, scene=True)[0]
                    )
                    prompt_to_imgs[prompt].append(
                        (name, seed, "plane", plane_path, i == final_idx)
                    )
                    prompt_to_imgs[prompt].append(
                        (name, seed, "scene", scene_path, i == final_idx)
                    )

                # Advance progress bar
                pbar.advance(tid_record)
            pbar.remove_task(tid_record)
            pbar.advance(tid_main)
        pbar.remove_task(tid_main)

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
                        images=[img[3] for img in imgs],
                        texts=[
                            f"A photo of {prompt}"
                            if metric.startswith("clipscore")
                            else prompt
                        ],
                    )
                    .scores[0]
                    .tolist()
                )
                for (name, seed, style, img, _), score in zip(imgs, scores):
                    result[name][style][metric][prompt][seed].append(
                        {"score": score, "path": str(img)}
                    )
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

        # Select best by MobileCLIP2
        mobileclip2 = load_metric("clipscore.MobileClip2")
        mc2_result = {}
        tid_mc2 = pbar.add_task(
            "Selecting by MobileCLIP2", total=len(prompt_to_imgs)
        )
        for prompt, imgs in prompt_to_imgs.items():
            paths = [
                str(img[3]) for img in imgs if img[2] == "plane" and img[4]
            ]
            mc2_scores = (
                mobileclip2(images=paths, texts=[f"A photo of {prompt}"])
                .scores[0]
                .tolist()
            )
            mc2_result[prompt] = dict(zip(paths, mc2_scores))
            pbar.advance(tid_mc2)
        for res in result.values():
            for prompt, runs in res["plane"][METRICS[0]].items():
                res["mobileclip2"][prompt] = max(
                    [(seed, entries) for seed, entries in runs.items()],
                    key=lambda x: mc2_result[prompt][
                        x[1][res["idx"][prompt][x[0]]["final"]]["path"]
                    ],
                )[0]
                save_json(result, out_path, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("experiments") / "runs" / "ours" / "main",
    )
    args = parser.parse_args()
    main(args.base_dir)
