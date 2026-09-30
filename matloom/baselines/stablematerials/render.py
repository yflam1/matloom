import argparse
from os import listdir
from os.path import isdir, join

from tqdm import tqdm

from matloom.render import textures_to_material


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Render StableMaterials baseline outputs (basecolor/metallic/roughness/"
            "normal + height-as-bump) in Blender. Walks --base-dir/<name>."
        )
    )
    p.add_argument(
        "--base-dir", type=str, default=join("outputs", "StableMaterials")
    )
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
    p.add_argument(
        "--height-distance",
        type=float,
        default=0.1,
        help="bump strength applied to the height map (world units; default 0.1)",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    single_view = not args.all_views

    for name in tqdm(
        sorted(listdir(args.base_dir)), desc="Rendering StableMaterials"
    ):
        path = join(args.base_dir, name)
        if not isdir(path):
            continue
        textures_to_material(
            path,
            basecolor=join(path, "basecolor.png"),
            metallic=join(path, "metallic.png"),
            roughness=join(path, "roughness.png"),
            normal=join(path, "normal.png"),
            height=(join(path, "height.png"), args.height_distance),
            shadow=args.shadow,
            single_view=single_view,
            scene=args.scene,
        )
