"""
Deactivate the `matfuse` environment first.
"""

import argparse
from os import listdir
from os.path import isdir, join

from tqdm import tqdm

from matloom.render import textures_to_material


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Render MatFuse baseline outputs (basecolor/roughness/specular/normal "
            "maps) in Blender. Walks --base-dir/<name>/<variant> for each variant."
        )
    )
    p.add_argument("--base-dir", type=str, default=join("outputs", "MatFuse"))
    p.add_argument(
        "--scene",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "render in the realistic angled scene (floor, props, HDRI + sun) "
            "instead of a flat head-on plane"
        ),
    )
    p.add_argument(
        "--all-views",
        action="store_true",
        help="render 4 views instead of a single one",
    )
    p.add_argument(
        "--shadow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let the geometry cast shadow rays (default: on)",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    single_view = not args.all_views

    for name in tqdm(sorted(listdir(args.base_dir)), desc="Rendering MatFuse"):
        path = join(args.base_dir, name)
        if not isdir(path):
            continue
        for subdir in ["1_basic", "2_ema", "3_cfg"]:
            out_dir = join(path, subdir)
            if not isdir(out_dir):
                continue
            textures_to_material(
                out_dir,
                basecolor=join(out_dir, "basecolor.png"),
                roughness=join(out_dir, "roughness.png"),
                specular=join(out_dir, "specular.png"),
                normal=join(out_dir, "normal.png"),
                shadow=args.shadow,
                single_view=single_view,
                scene=args.scene,
            )
