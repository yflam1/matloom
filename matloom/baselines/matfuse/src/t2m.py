import datetime as dt
import json
import os
import time
from dataclasses import dataclass, fields
from os.path import join
from typing import Optional

import matplotlib

# Set the backend to non-interactive "Agg" before importing other modules
matplotlib.use("Agg")

from PIL import Image
from tqdm import tqdm
from utils.inference_helpers import run_generation

DEFAULT_OUTDIR = join("outputs", "MatFuse")
DEFAULT_RESOLUTION = 512
DEFAULT_STEPS = 50
DEFAULT_SCALE = 5.0


@dataclass
class MatFuseMaps(object):
    basecolor: Image.Image
    normal: Image.Image
    roughness: Image.Image
    specular: Image.Image


@dataclass
class MatFuseOutput(object):
    basic: MatFuseMaps
    ema: MatFuseMaps
    cfg: MatFuseMaps


def run_matfuse(
    prompt: str,
    resolution: Optional[int] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple[MatFuseOutput, int, int, float, int]:
    resolution = resolution if resolution is not None else DEFAULT_RESOLUTION
    steps = steps if steps is not None else DEFAULT_STEPS
    scale = guidance_scale if guidance_scale is not None else DEFAULT_SCALE
    if seed is None:
        seed = -1
    results, seed = run_generation(
        render_emb=None,
        palette_source=None,
        sketch=None,
        prompt=prompt,
        num_samples=1,
        image_resolution=resolution,
        ddim_steps=steps,
        seed=seed,
        ddim_eta=0.0,
        ucg_scale=scale,
    )
    return (
        MatFuseOutput(
            basic=MatFuseMaps(
                basecolor=Image.fromarray(results[2][:resolution, :resolution, :]),
                normal=Image.fromarray(results[2][resolution:, :resolution, :]),
                roughness=Image.fromarray(results[2][:resolution, resolution:, :]),
                specular=Image.fromarray(results[2][resolution:, resolution:, :]),
            ),
            ema=MatFuseMaps(
                basecolor=Image.fromarray(results[3][:resolution, :resolution, :]),
                normal=Image.fromarray(results[3][resolution:, :resolution, :]),
                roughness=Image.fromarray(results[3][:resolution, resolution:, :]),
                specular=Image.fromarray(results[3][resolution:, resolution:, :]),
            ),
            cfg=MatFuseMaps(
                basecolor=Image.fromarray(results[4][:resolution, :resolution, :]),
                normal=Image.fromarray(results[4][resolution:, :resolution, :]),
                roughness=Image.fromarray(results[4][:resolution, resolution:, :]),
                specular=Image.fromarray(results[4][resolution:, resolution:, :]),
            ),
        ),
        resolution,
        steps,
        scale,
        seed,
    )


if __name__ == "__main__":
    # Get prompts
    prompts: list[str] = []
    while True:
        prompt = input(f"Enter prompt {len(prompts) + 1} (skip to end): ").strip()
        if prompt == "":
            break
        prompts.append(prompt)
    if len(prompts) == 0:
        raise ValueError("No prompt entered. Exiting.")

    # Get output directory
    out_dir = input(f"Enter output directory (skip means {DEFAULT_OUTDIR}): ").strip()
    if out_dir == "":
        out_dir = DEFAULT_OUTDIR

    # Get resolution
    resolution = input(f"Enter resolution (skip means {DEFAULT_RESOLUTION}): ").strip()
    resolution = None if resolution == "" else int(resolution)

    # Get number of steps
    steps = input(f"Enter steps (skip means {DEFAULT_STEPS}): ").strip()
    if steps == "":
        steps = None
    else:
        steps = int(steps)
        if steps <= 3:
            raise ValueError("Steps must be greater than 3.")

    # Get guidance scale
    scale = input(f"Enter guidance scale (skip means {DEFAULT_SCALE}): ").strip()
    scale = None if scale == "" else float(scale)

    seeds: list[Optional[int]] = []
    for i in range(len(prompts)):
        seed = input(f"Enter seed for prompt {i + 1} (skip means random): ").strip()
        seeds.append(None if seed == "" else int(seed))

    # Warm up the pipeline so the first real prompt isn't inflated by one-time
    # CUDA/cuDNN init, memory allocation, etc. (excluded from per-prompt timing).
    print("Warming up MatFuse...")
    try:
        run_matfuse(
            "warmup",
            resolution=resolution,
            steps=steps,
            guidance_scale=scale,
            seed=0,
        )
    except Exception as e:
        print(f"Warm-up generation failed (continuing anyway): {e}")

    for prompt, seed in tqdm(
        zip(prompts, seeds), desc="Running MatFuse", total=len(prompts)
    ):
        now = dt.datetime.now(dt.timezone.utc).strftime(r"%Y%m%d-%H%M%S-%f")
        run_dir = join(out_dir, f"matfuse_{now}")
        start_time = time.perf_counter()
        try:
            output, resolution, steps, scale, seed = run_matfuse(
                prompt,
                resolution=resolution,
                steps=steps,
                guidance_scale=scale,
                seed=seed,
            )
        except Exception:
            continue
        duration = time.perf_counter() - start_time

        for i, mode in enumerate([f.name for f in fields(MatFuseOutput)]):
            _out_dir = join(run_dir, f"{i + 1}_{mode}")
            os.makedirs(_out_dir, exist_ok=True)
            for cat in [f.name for f in fields(MatFuseMaps)]:
                img: Image.Image = getattr(getattr(output, mode), cat)
                img.save(join(_out_dir, f"{cat}.png"))
        with open(join(run_dir, "info.json"), "w") as f:
            json.dump(
                {
                    "prompt": prompt,
                    "resolution": resolution,
                    "steps": steps,
                    "guidance_scale": scale,
                    "seed": seed,
                    "duration_seconds": duration,
                },
                f,
                indent=2,
            )
