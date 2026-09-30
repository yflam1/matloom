import argparse
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, action="append")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--resolution", type=str, default="")
    parser.add_argument("--steps", type=str, default="")
    parser.add_argument("--scale", type=str, default="")
    parser.add_argument("--seed", type=str, action="append", default=None)
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
        ["" for _ in args.prompt]
        if args.seed is None
        else [args.seed[0] for _ in args.prompt]
        if len(args.seed) == 1
        else args.seed
    )

    # Prepare inputs
    inputs = [
        *args.prompt,
        "",
        args.out_dir,
        args.resolution,
        args.steps,
        args.scale,
        *seeds,
    ]

    command = [
        sys.executable,
        "matloom/baselines/matfuse/src/t2m.py",
        "--ckpt",
        "models/matfuse-full.ckpt",
        "--config",
        "matloom/baselines/matfuse/src/configs/diffusion/matfuse-ldm-vq_f8.yaml",
    ]
    subprocess.run(
        command, input="\n".join(inputs) + "\n", text=True, check=True
    )
