# IntrinsiX

Text-to-PBR baseline using [IntrinsiX](https://github.com/Peter-Kocsis/IntrinsiX) (Kocsis et al., NeurIPS 2025).
IntrinsiX injects LoRA modules into `black-forest-labs/FLUX.1-dev` and generates intrinsic maps from a text prompt: **albedo (base color), roughness, metallic, normal**.
There is no height/displacement map, so the rendered plane is flat (normal-map relief only).

This baseline is self-contained, like the MatFuse baseline: the minimal upstream pipeline package is vendored under `intrinsix/` here, and `run.py` imports only that package plus third-party deps (never `matloom`).
No `git clone`, no `PYTHONPATH`.
`render.py` is the only file that imports `matloom` (via `matloom.render`), and it runs in the `matloom` env, not this one.

> **Hardware.** Inference loads FLUX.1-dev in `bfloat16` and needs roughly **35 GB of VRAM** on CUDA (tested on an RTX A6000). On Apple Silicon it runs on MPS and fits in ~30 GB of unified memory (tested on an M4 Pro, 48 GB). It will not run on CPU in practice.

## Installation

```bash
conda create -n intrinsix python=3.10 -y
conda activate intrinsix
python -m pip install -r matloom/baselines/intrinsix/requirements.txt

# Authenticate for the gated base model (accept the FLUX.1-dev license on the
# Hugging Face model page first), then log in:
huggingface-cli login
```

> **License.** IntrinsiX's code and weights are **CC-BY-NC-SA-4.0** and FLUX.1-dev carries its own non-commercial license — research use only.

## Usage

```text
python matloom/baselines/intrinsix/run.py --prompt PROMPT [--out-dir OUT_DIR] [--resolution RESOLUTION] [--steps STEPS] [--scale SCALE] [--seed SEED]

options:
  --prompt PROMPT          (repeatable) text description of the material
  --out-dir OUT_DIR        output root (default: outputs/IntrinsiX)
  --resolution RESOLUTION  square render size (default: 512)
  --steps STEPS            diffusion steps (default: 28)
  --scale SCALE            guidance scale (default: 3.5)
  --seed SEED              (repeatable) one seed, or one per prompt
```

Each run writes `basecolor.png` / `roughness.png` / `metallic.png` / `normal.png` plus an `info.json` into a timestamped subfolder.
Render the saved maps with:

```bash
python matloom/baselines/intrinsix/render.py --base-dir outputs/IntrinsiX
```

## Citation

```bibtex
@inproceedings{kocsis2025intrinsix,
  title={IntrinsiX: High-Quality PBR Generation using Image Priors},
  author={Kocsis, Peter and H{\"o}llein, Lukas and Nie{\ss}ner, Matthias},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
