# MatFuse

## Installation

```bash
# On macOS 26+ use python 3.12: scipy wheels for 3.10 cap out at 1.15,
# whose `scipy.sparse.linalg` PROPACK extension fails the stricter dyld checks,
# and t2m.py imports it at startup. On Linux, python 3.10 works too.
conda create -n matfuse python=3.12 -y
conda activate matfuse
# openai-clip does `from pkg_resources import packaging`. setuptools 70 stopped
# vendoring packaging (the standalone `packaging` package must be installed) and
# 81+ drops pkg_resources entirely, so keep setuptools in [78.1.1, 81).
python -m pip install "setuptools>=78.1.1,<81" wheel packaging
python -m pip install -r matloom/baselines/matfuse/requirements.txt
wget -P models https://huggingface.co/gvecchio/MatFuse/resolve/main/checkpoints/matfuse-full.ckpt
```

## Usage

```text
python matloom/baselines/matfuse/run.py --prompt PROMPT [--out-dir OUT_DIR] [--resolution RESOLUTION] [--steps STEPS] [--scale SCALE] [--seed SEED]

options:
  --prompt PROMPT
  --out-dir OUT_DIR
  --resolution RESOLUTION
  --steps STEPS
  --scale SCALE
  --seed SEED
```

## Citation

```bibtex
@inproceedings{vecchio2024matfuse,
  author    = {Vecchio, Giuseppe and Sortino, Renato and Palazzo, Simone and Spampinato, Concetto},
  title     = {MatFuse: Controllable Material Generation with Diffusion Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {4429-4438}
}
```
