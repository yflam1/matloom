"""
IntrinsiX text-to-PBR baseline wrapper.

IntrinsiX (Kocsis et al., NeurIPS 2025) generates intrinsic PBR maps from a text
prompt by injecting LoRA modules into FLUX.1-dev. Unlike MatFuse / StableMaterials
the upstream project is **not** a pip package, so the pipeline class is vendored
in full under `intrinsix/` next to this file (see this folder's README.md). With
`num_components=3` it emits three images: albedo, a packed "material" map
(R=roughness, G=metallic), and a tangent normal, and **no** height/displacement map.

This script is self-contained: it imports only the vendored `intrinsix` package
and third-party deps, never `matloom` (render.py is the only file that touches
matloom, via `matloom.render`).
"""

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from dataclasses import dataclass, fields
from os.path import abspath, dirname, join
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Put the vendored intrinsix/ package (this directory) on the path so it imports
# without cloning the upstream repo or setting PYTHONPATH.
_BASELINE_DIR = abspath(dirname(__file__))
if _BASELINE_DIR not in sys.path:
    sys.path.insert(0, _BASELINE_DIR)

from model.intrinsix import IntrinsiXPipeline  # noqa: E402


def _is_mps(device: str) -> bool:
    return device.lower() == "mps"


def _is_cuda(device: str) -> bool:
    return device.lower().startswith("cuda")


def _finalize_device(device: Optional[str]) -> str:
    """Pick the best available device: CUDA > MPS > CPU."""
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if _is_cuda(device) and not torch.cuda.is_available():
        print(f"WARNING: '{device}' is not available, using CPU instead")
        return "cpu"
    return device


# Hub ids: the released LoRA, and the (gated) base model it is injected into.
LORA_REPO = "PeterKocsis/IntrinsiX"
BASE_MODEL = "black-forest-labs/FLUX.1-dev"


@dataclass
class IntrinsiXOutput(object):
    # `basecolor` (not `albedo`) so the maps match render.py / textures_to_material.
    basecolor: Image.Image
    roughness: Image.Image
    metallic: Image.Image
    normal: Image.Image


@dataclass
class IntrinsiXConfig(object):
    resolution: int
    steps: int
    guidance_scale: float
    seed: int


class IntrinsiX(object):
    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[str] = "models",
    ) -> None:
        self._device = _finalize_device(device)
        # bfloat16 on CUDA (FLUX.1-dev) and on MPS (Apple Silicon M-series
        # supports bf16). CPU stays float32 (impractically slow, but valid).
        if _is_cuda(self._device) or _is_mps(self._device):
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        print(f"Loading IntrinsiX on {self._device} ({dtype})...")
        self._pipe = IntrinsiXPipeline.from_pretrained(
            pretrained_model_name_or_path=LORA_REPO,
            base_model_path=BASE_MODEL,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        if _is_cuda(self._device):
            self._configure_cuda()
        elif _is_mps(self._device):
            self._pipe = self._pipe.to(self._device)

    def _configure_cuda(self) -> None:
        """Place the pipeline across N visible CUDA GPUs.

        FLUX.1-dev is ~12 B params (~24 GB in bf16) and does not fit on a single
        24 GB GPU with activations. diffusers' ``device_map="balanced"`` overloads
        GPU 0: it parks T5-XXL + the VAE there and balances only weights, so the
        denoising activations push GPU 0 over. Instead, keep the text encoders and
        VAE on GPU 0 (``encode_prompt`` assumes one execution device) and shard
        only the transformer across all N visible GPUs via accelerate, giving GPU 0
        a small budget because it also hosts T5-XXL (~9.4 GB bf16) + the VAE.
        N = ``torch.cuda.device_count()``, so control it with CUDA_VISIBLE_DEVICES.
        """
        n = torch.cuda.device_count()
        pipe = self._pipe
        if n >= 2:
            from accelerate import dispatch_model, infer_auto_device_map

            # Text encoders + VAE stay on GPU 0 so encode_prompt / vae.decode run
            # on a single device; only the transformer is sharded.
            pipe.text_encoder.to("cuda:0")
            pipe.text_encoder_2.to("cuda:0")
            pipe.vae.to("cuda:0")

            # GPU 0 also hosts T5-XXL + CLIP + VAE, so give the transformer a small
            # budget there and split the rest evenly across the other N-1 GPUs.
            max_memory = {0: "8GiB"}
            for i in range(1, n):
                max_memory[i] = "22GiB"
            # no_split_module_classes takes class-NAME STRINGS: accelerate matches
            # `module.__class__.__name__ in no_split_module_classes`. Passing class
            # objects silently never matches, so blocks get split mid-block and the
            # forward crashes mixing devices inside one block.
            device_map = infer_auto_device_map(
                pipe.transformer,
                max_memory=max_memory,
                no_split_module_classes=[
                    "FluxTransformerBlock",
                    "FluxSingleTransformerBlock",
                ],
            )
            pipe.transformer = dispatch_model(
                pipe.transformer, device_map=device_map
            )

            # The dispatched transformer returns its output on the GPU of its last
            # block. Move it back to GPU 0 so the scheduler step sees noise_pred
            # and latents on the same device.
            def _to_gpu0(_, __, output):
                return tuple(
                    o.to("cuda:0") if isinstance(o, torch.Tensor) else o
                    for o in output
                )

            pipe.transformer.register_forward_hook(_to_gpu0)
            print(
                f"IntrinsiX transformer sharded across {n} GPUs (GPU 0 light)."
            )
        else:
            # Single 24 GB-class GPU: stream transformer blocks to the GPU one at
            # a time so peak VRAM stays small (slower, but will not OOM).
            pipe.enable_sequential_cpu_offload()
            print("IntrinsiX on 1 GPU with sequential CPU offload.")

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        resolution: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> tuple[IntrinsiXOutput, IntrinsiXConfig]:
        resolution = resolution if resolution is not None else 512
        steps = steps if steps is not None else 28
        scale = guidance_scale if guidance_scale is not None else 3.5
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        gen_device = (
            "cuda"
            if _is_cuda(self._device)
            else "mps"
            if _is_mps(self._device)
            else "cpu"
        )
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        images = self._pipe(
            prompt=prompt,
            height=resolution,
            width=resolution,
            num_components=3,  # albedo, material(rough+metal), normal
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        ).images
        material = np.asarray(images[1])
        output = IntrinsiXOutput(
            basecolor=images[0],
            roughness=Image.fromarray(
                material[..., 0]
            ),  # R of the material map
            metallic=Image.fromarray(
                material[..., 1]
            ),  # G of the material map
            normal=images[2],
        )
        config = IntrinsiXConfig(
            resolution=resolution,
            steps=steps,
            guidance_scale=scale,
            seed=seed,
        )
        return output, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, action="append")
    parser.add_argument(
        "--out-dir", type=str, default=join("outputs", "IntrinsiX")
    )
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

    # Prepare seeds (mirrors the StableMaterials baseline CLI).
    seeds = (
        (args.seed for _ in args.prompt)
        if args.seed is None
        else (args.seed[0] for _ in args.prompt)
        if len(args.seed) == 1
        else args.seed
    )

    model = IntrinsiX()

    # Warm up the pipeline so the first real prompt isn't inflated by one-time
    # CUDA/cuDNN init, memory allocation, etc. (excluded from per-prompt timing).
    print("Warming up IntrinsiX...")
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

    for prompt, seed in tqdm(
        zip(args.prompt, seeds),
        desc="Running IntrinsiX",
        total=len(args.prompt),
    ):
        now = dt.datetime.now(dt.timezone.utc).strftime(r"%Y%m%d-%H%M%S-%f")
        start_time = time.perf_counter()
        try:
            output, config = model(
                prompt=prompt,
                resolution=args.resolution,
                steps=args.steps,
                guidance_scale=args.scale,
                seed=seed,
            )
        except Exception as e:
            print(f"ERROR generating '{prompt}': {e}")
            continue
        duration = time.perf_counter() - start_time

        out_dir = join(args.out_dir, f"intrinsix_{now}")
        os.makedirs(out_dir, exist_ok=True)
        for cat in [f.name for f in fields(IntrinsiXOutput)]:
            getattr(output, cat).save(join(out_dir, f"{cat}.png"))

        with open(join(out_dir, "info.json"), "w") as f:
            json.dump(
                {
                    "prompt": prompt,
                    "resolution": config.resolution,
                    "steps": config.steps,
                    "guidance_scale": config.guidance_scale,
                    "seed": config.seed,
                    "duration_seconds": duration,
                },
                f,
                indent=2,
            )
