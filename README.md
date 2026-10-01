<h1 align="center">MatLoom: Layered Text-to-Material Generation in a Compact Program Space</h1>

<p align="center">
  <strong>
    <a href="https://yflam1.github.io/">Anson Y. Lam</a>,
    <a href="https://shuqing-li.github.io/">Shuqing Li</a><sup>*</sup>,
    <a href="https://www.cse.cuhk.edu.hk/lyu/home">Michael R. Lyu</a>
  </strong>
</p>

<p align="center">
  Department of Computer Science and Engineering, The Chinese University of Hong Kong<br>
  <sup>*</sup>Corresponding author.
</p>

<!-- Add an arXiv link here once the preprint is up. -->
<p align="center">
  <a href="https://yflam1.github.io/matloom/"><strong>Project Page</strong></a> |
  <a href="https://arxiv.org/abs/2609.40322"><strong>arXiv</strong></a>
</p>

## 📝 Abstract

Material generation should produce not only an appearance, but also the rules that construct it.
We introduce MatLoom, a compact, layer-oriented language for text-to-material generation with pretrained language models.
Each program composes alpha-masked layers whose shared spatial expressions define coverage and physically based rendering (PBR) channels, making dependencies between patterns, color, and relief explicit.
A standalone interpreter evaluates the program into material maps, while the source retains named fields and layer parameters for subsequent authoring.
Without task-specific fine-tuning, our pipeline uses parser-guided repair and preview-based critique to revise material designs, then searches noise seeds while keeping each candidate's remaining source fixed.
On a curated benchmark of 141 prompts evaluated with six backbones, our best-performing configuration achieves higher mean scores than three diffusion baselines on all four flat-layout prompt-alignment metrics.
Its initial programs already exceed all three baselines on mean BLIPScore, before critique or seed search.
Retained programs have a median length of 21 lines when pooled across backbones.
In a blind four-way comparison involving 30 participants and 20 prompts, our renders receive 59.2% of choices, compared with 19.3% for the most-preferred baseline.
Compact executable programs thus offer a way to generate prompt-aligned materials while retaining their construction as part of the asset.

## 📦 Installation

```bash
conda create -n matloom python=3.13 -y
conda activate matloom
python -m pip install "setuptools<70.0.0" wheel
python -m pip install -e ".[blender,vqascore]" --no-build-isolation
python -m pip install \
  "t2v-metrics@git+https://github.com/linzhiqiu/t2v_metrics.git@98f184daaaa4a28be494c7dd8eb3b9e9f763aace" \
  --no-deps
python -m pip install image-reward --no-deps
python -m pip install iopath fairscale tiktoken sentencepiece
python -m pip install \
  "salesforce-lavis@git+https://github.com/anson416/LAVIS.git@single-env" \
  "transformers==4.49.0" --no-build-isolation
```

## 🎬 Quickstart

A material is a short program: a `View` preamble sets the sampled window, each `Define` names a reusable spatial pattern, and a `Material` composes alpha-masked layers with later layers on top.
The program below lays gray bricks over a red mortar substrate, with the brick pattern driving both coverage and height.
The `View(...)` line must precede every `Define(...)`.

```bash
cat > bricks.dsl <<'EOF'
View(0, 0, 10, 10)
Define(brick, Bricks())

Material(
  Layer(1).basecolor(255, 0, 0).roughness(0.5).height(0),
  Layer(brick).basecolor(127, 127, 127).roughness(0.5).height(brick * 0.5)
)
EOF

# Evaluate the program into 13 material maps under out/maps (CPU-only, no API key).
# --program takes the program text itself, hence "$(cat ...)".
matloom-export --program "$(cat bricks.dsl)" --output-dir out/maps --width 256 --height 256

# Floating-point maps are written as .hdr by default.
# Set OPENCV_IO_ENABLE_OPENEXR=1 to write .exr instead, which preserves negative heights.
OPENCV_IO_ENABLE_OPENEXR=1 matloom-export --program "$(cat bricks.dsl)" --output-dir out/maps-exr --width 256 --height 256

# Optional: render the maps into a preview image with Blender Cycles (needs the [blender] extra).
matloom-render --maps-dir out/maps --output-dir out/render

# Optional: sample a random program and evaluate it the same way.
matloom-generate --seed 42 --num-layers 3 --num-defs 2 > random.dsl
matloom-export --program "$(cat random.dsl)" --output-dir out/maps --width 256 --height 256

# Text-to-material generation with an LLM: prints a program for the prompt (OpenAI-compatible endpoint required).
matloom-text2dsl --prompt "gray bricks over red mortar" --model MODEL --api-key KEY > bricks_llm.dsl
matloom-export --program "$(cat bricks_llm.dsl)" --output-dir out/maps --width 256 --height 256
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📚 Citation

If you find this code useful, please cite our paper.

```bibtex
@article{lam2026matloom,
  title={{MatLoom}: Layered Text-to-Material Generation in a Compact Program Space},
  author={Lam, Anson Y. and Li, Shuqing and Lyu, Michael R.},
  journal={arXiv preprint arXiv:2609.40322},
  year={2026}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).
