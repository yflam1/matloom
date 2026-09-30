import datetime as dt
import json
import os
import random
import time
from dataclasses import dataclass, fields
from os.path import join
from typing import Optional

import torch
from diffusers import DiffusionPipeline, LCMScheduler, UNet2DConditionModel
from PIL import Image
from pydantic import validate_call
from pytorch_lightning import seed_everything

from matloom.utils.anybase import AnyBase
from matloom.utils.misc import finalize_device, is_cuda, loading
from matloom.utils.progress import progress_bar


@dataclass
class StableMaterialsOutput(object):
    basecolor: Image.Image
    normal: Image.Image
    height: Image.Image
    roughness: Image.Image
    metallic: Image.Image


@dataclass
class StableMaterialsConfig(object):
    use_lcm: bool
    resolution: int
    steps: int
    guidance_scale: float
    seed: int


class StableMaterials(AnyBase):
    @validate_call
    def __init__(
        self,
        use_lcm: bool = False,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._use_lcm = use_lcm
        self._device = finalize_device(device)
        self._default_steps = 4 if use_lcm else 50
        # NOTE: the gvecchio/StableMaterials custom pipeline mixes fp16 and fp32
        # tensors in its CLIP text projection under newer transformers, raising
        # "expected mat1 and mat2 to have the same dtype, but got float != Half".
        # Loading in fp32 avoids the mismatch; a 512x512 generation fits a 24 GB
        # GPU comfortably, so the speed cost is acceptable for evaluation.
        dtype = torch.float32 if is_cuda(self._device) else None
        kwargs = dict(
            trust_remote_code=True,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        if use_lcm:
            kwargs["unet"] = UNet2DConditionModel.from_pretrained(
                "gvecchio/StableMaterials",
                subfolder="unet_lcm",
                torch_dtype=dtype,
                cache_dir=cache_dir,
            )
        pipe = DiffusionPipeline.from_pretrained(
            "gvecchio/StableMaterials", **kwargs
        )
        if use_lcm:
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        self._pipe = pipe.to(torch.device(self._device))

    @torch.inference_mode()
    @validate_call
    def __call__(
        self,
        prompt: str,
        resolution: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> tuple[StableMaterialsOutput, StableMaterialsConfig]:
        with loading(f"Generating material with {self.cname_}..."):
            resolution = resolution if resolution is not None else 512
            steps = steps if steps is not None else self._default_steps
            scale = guidance_scale if guidance_scale is not None else 10.0
            if seed is None:
                seed = random.randint(0, 2**31 - 1)
            seed_everything(seed)
            result = self._pipe(
                prompt=prompt,
                height=resolution,
                width=resolution,
                num_inference_steps=steps,
                guidance_scale=scale,
                tileable=True,
                num_images_per_prompt=1,
            ).images[0]
            output = StableMaterialsOutput(
                basecolor=result.basecolor,
                normal=result.normal,
                height=result.height,
                roughness=result.roughness,
                metallic=result.metallic,
            )
            config = StableMaterialsConfig(
                use_lcm=self._use_lcm,
                resolution=resolution,
                steps=steps,
                guidance_scale=scale,
                seed=seed,
            )
            return output, config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, action="append")
    parser.add_argument(
        "--out-dir", type=str, default=join("outputs", "StableMaterials")
    )
    parser.add_argument("--use-lcm", action="store_true")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    if not (
        args.seed is None
        or len(args.seed) == 1
        or len(args.seed) == len(args.prompt)
    ):
        parser.error(
            "Number of seeds must be either 1 or equal to the number of prompts"
        )

    # Prepare seeds
    seeds = (
        (args.seed for _ in args.prompt)
        if args.seed is None
        else (args.seed[0] for _ in args.prompt)
        if len(args.seed) == 1
        else args.seed
    )

    model = StableMaterials(use_lcm=args.use_lcm)

    # Warm up the pipeline so the first real prompt isn't inflated by one-time
    # CUDA/cuDNN init, memory allocation, etc. (excluded from per-prompt timing).
    print("Warming up StableMaterials...")
    try:
        model(
            prompt="warmup",
            resolution=args.resolution,
            steps=args.steps,
            guidance_scale=args.scale,
            seed=0,
        )
    except Exception as e:
        print(f"Warm-up generation failed (continuing anyway): {e}")

    with progress_bar(transient=False) as pbar:
        task_id = pbar.add_task(
            "Running StableMaterials", total=len(args.prompt)
        )
        for prompt, seed in zip(args.prompt, seeds):
            now = dt.datetime.now(dt.timezone.utc).strftime(
                r"%Y%m%d-%H%M%S-%f"
            )
            start_time = time.perf_counter()
            try:
                output, config = model(
                    prompt=prompt,
                    resolution=args.resolution,
                    steps=args.steps,
                    guidance_scale=args.scale,
                    seed=seed,
                )
            except Exception:
                continue
            duration = time.perf_counter() - start_time

            # Save output images
            out_dir = join(args.out_dir, f"stablematerials_{now}")
            os.makedirs(out_dir, exist_ok=True)
            for cat in [f.name for f in fields(StableMaterialsOutput)]:
                path = join(out_dir, f"{cat}.png")
                getattr(output, cat).save(path)

            # Save config for this run
            with open(join(out_dir, "info.json"), "w") as f:
                json.dump(
                    {
                        "prompt": prompt,
                        "use_lcm": config.use_lcm,
                        "resolution": config.resolution,
                        "steps": config.steps,
                        "guidance_scale": config.guidance_scale,
                        "seed": config.seed,
                        "duration_seconds": duration,
                    },
                    f,
                    indent=2,
                )

            # Update progress bar
            pbar.advance(task_id)
