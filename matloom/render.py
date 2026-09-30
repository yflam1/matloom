"""matloom.render — turn texture maps into rendered previews via Blender (Cycles).

Public entry points (all accept ``scene=`` to pick the layout):

- :func:`textures_to_material` — apply a dict of texture maps to the hero panel
  and render it. The low-level primitive used by everything below.
- :func:`maps_to_material` — render the maps written by
  :meth:`matloom.engine.main.LayeredMaterial.export` from a directory; routes
  height to real geometry displacement (``scale = 2.0 / region_span`` so the
  relief matches the browser viewer).
- :func:`program_to_material` — export a DSL ``program`` to a temp dir and
  render it, reading ``region_span`` from its ``View``.
- :func:`program_to_image` — render a single head-on view and return it in
  memory (nothing on disk); used by the text-to-DSL critique loop.

Two layouts, selected by ``scene``:

- **flat swatch** (``scene=False``, default): the material on a single plane,
  ``single_view=True`` → one head-on view, else 4 views rotated 90° each.
  Cheapest, but hides relief seen edge-on and gives transmission nothing to
  refract. Uses the ``HDRI`` (``city.exr``) for soft fill.
- **realistic scene** (``scene=True``): ONE consistent world with no per-ray
  visibility tricks. A real enclosed studio light box
  (``_build_studio_room``: floor + 4 walls + ceiling, neutral-grey) that a mirror
  reflects coherently; real, fully-shadowed props behind the panel
  (``_build_back_props``: camera context + something for glass to refract) and
  in front of it (``_build_front_props``: a colorful still-life a mirror
  reflects, placed outside the camera's ~40° frustum so a matte panel's frame
  is unchanged). Lighting is HDRI-free: a faint flat-grey world
  (``SCENE_WORLD_GREY``) plus real area lights (``SCENE_LIGHTS``, only the key
  casts shadows, so every shadow is parallel). Shot from a fixed three-quarter
  angle so displacement relief casts a clean shadow and reads as 3D. The
  ``SCENE_*`` constants at the top of the module fix the room, lights, and
  camera. Closer to the photographs the CLIP/BLIP/VQA evaluators in
  :mod:`matloom.metrics` expect.

CLI: ``matloom-render`` — ``--program``, ``--maps-dir``, or ``--text`` (a prompt
→ text-to-DSL → program → render; ``--model`` selects the LLM), plus optional
``--output-dir`` (when omitted the render is shown in a blocking OpenCV window
and nothing is left on disk), ``--all-views``, ``--scene``/``--no-scene``,
``--shadow``/``--no-shadow``, ``--export-width``/``--export-height``,
``--region-span``/``--height-scale``, and a per-channel ``--<channel>``/
``--no-<channel>`` toggle for each of :data:`CHANNELS`. The three baselines
(``baselines/{matfuse,stablematerials,intrinsix}/render.py``) call
:func:`textures_to_material` directly, wired for the maps their models emit.
"""

import argparse
import math
import shutil
import tempfile
from os.path import exists
from pathlib import Path

from matloom import logger
from matloom.blender import bpa
from matloom.blender.bpa import Vector
from matloom.utils.dtypes import PathLike

# from mathutils import Vector  # type: ignore


HDRI = Path(bpa.__file__).parent / "hdri" / "city.exr"
# NOTE: the realistic *scene* preset is HDRI-free (a studio softbox setup; see the
# scene section below). Only the flat-swatch path uses this HDRI for soft fill.

# The texture channels `maps_to_material` can route into the render, in the order
# they appear in the CLI help. Each is individually toggleable via `enabled`
# (a `--<channel>` / `--no-<channel>` flag on the CLI) so a single map can be
# isolated to confirm it is contributing — e.g. `--no-ao --no-normal …` to see
# the bare base color, or only `--height` to inspect displacement alone.
CHANNELS = (
    "basecolor",
    "metallic",
    "roughness",
    "sheen",
    "coat",
    "transmission",
    "ior",
    "subsurface",
    "anisotropy",
    "normal",
    "ao",
    "emissive",
    "height",
)

# Blender's default plane (``primitive_plane_add``) spans 2 world units per side.
# The browser viewer keeps height in *raw coordinate units* while scaling the
# plane's X/Y by the View region span, so to reproduce its relief proportions the
# displacement scale must map a coordinate-unit height onto the 2-unit plane:
#   displacement_scale = PLANE_SIZE / region_span   (see `_displacement_for`).
PLANE_SIZE = 2.0

# |======================================================|
# |   The realistic "scene" preset (vs. the flat plane)  |
# |======================================================|
#
# WHY a scene at all: a head-on render of a material on a flat plane hides two
# things the DSL can express — surface *relief* (height/displacement, invisible
# when viewed face-on) and *transmission/IOR* (you need something behind the
# material to refract). It is also out-of-distribution for the CLIP/BLIP/VQA
# evaluators in `matloom/metrics`, which were trained on photographs of objects in
# real environments, not orthographic swatches. The scene preset addresses both:
# the material panel stays the hero, but it is stood upright and viewed from a
# fixed three-quarter angle (so relief casts shadows and reads as 3D), grounded
# inside a studio with neutral primitives around it.
#
# DESIGN: ONE CONSISTENT WORLD — no ray-visibility tricks. Earlier versions used an
# HDRI and/or "reflection probes" and "softboxes" that were visible to some ray
# types but not others; that builds an *inconsistent* world and leaks artifacts
# (a mirror reflected a blown-out emissive softbox quad, and reflection-only props
# floated with no shadow). Instead the scene is a real enclosed studio "light box":
# a closed neutral-grey room (floor + four walls + ceiling) lit by real area
# lights, with real props. A mirror simply reflects the actual room — coherent,
# fully shadowed, no surprises.
#
# Props sit BOTH behind the panel (camera context + something for glass to refract)
# and IN FRONT of it (what the mirror reflects). The front props are real, fully
# shadowed objects placed OUTSIDE the camera frustum (the camera's ~40° cone does
# not see them directly) but within the panel's reflected field of view — so a
# matte panel's frame is unchanged while a mirror reflects grounded, shadowed
# objects. Nothing relies on per-ray visibility flags.
#
# All distances are in the same world units as the 2-unit hero panel.
SCENE_PANEL_LIFT = (
    1.02  # panel centre height (≈ half-height, so it rests on floor)
)
# Enclosed studio "light box": a closed neutral room the mirror reflects coherently.
# Kept fairly TIGHT — a deep/tall room shows a big slab of featureless grey wall
# above the panel and pushes the props far from their contact shadows. The back
# wall sits just behind the props and the ceiling just above the lights.
SCENE_ROOM_HALF = 5.5  # half-extent of floor/ceiling and wall width
SCENE_ROOM_HEIGHT = 4.5  # ceiling height (keep all lights below this)
SCENE_ROOM_GREY = (197, 198, 202)  # walls/floor
SCENE_CEILING_GREY = (221, 222, 226)  # ceiling (slightly brighter)
SCENE_WORLD_GREY = (
    210,
    211,
    214,
)  # world background (visible only through gaps)
# Real area lights: (position, aim_at, energy, size, casts_shadow). Only the key
# casts shadows, so every shadow is parallel; fills are shadowless and soft. All
# are kept BELOW the ceiling (z < SCENE_ROOM_HEIGHT) so the room does not clip them.
_PANEL_AIM = (0.0, 0.0, SCENE_PANEL_LIFT)
SCENE_LIGHTS: tuple[tuple, ...] = (
    # (position, aim_at, energy, size, casts_shadow)
    ((-3.0, -3.0, 4.0), _PANEL_AIM, 900.0, 2.6, True),  # key, upper front-left
    ((0.0, 0.0, 4.2), _PANEL_AIM, 220.0, 3.6, False),  # soft top
)
# Fixed three-quarter hero camera: elevation tilt (down from horizontal) and the
# azimuth swing about the upright panel. Chosen so grazing light rakes the relief
# while keeping the panel close to frame-filling.
SCENE_CAM_ELEVATION = 18.0
SCENE_CAM_AZIMUTH = 26.0
# frame the panel's bounding sphere (margin leaves room for props)
SCENE_CAM_FIT_RATIO = 0.0
# padding around the panel; ~1 fills the frame, higher pulls back
SCENE_CAM_MARGIN = 0.9
SCENE_SAMPLES = (
    160  # glass/refraction + an enclosed room is noisier than a swatch
)
# Panel bounding-sphere radius: the 2×2 panel has half-diagonal √2; the margin
# pulls the camera back just enough that the floor and props stay in shot.
_PANEL_RADIUS = math.sqrt(2.0)


def textures_to_material(
    out_dir: PathLike,
    *,
    basecolor: str,
    metallic: str | None = None,
    roughness: str | None = None,
    specular: str | None = None,
    sheen: str | None = None,
    coat: str | None = None,
    transmission: str | None = None,
    ior: str | None = None,
    subsurface: str | None = None,
    anisotropy: str | None = None,
    normal: str | None = None,
    ao: str | None = None,
    height: tuple[str, float] | None = None,
    displacement: tuple[str | float, float, float] | None = None,
    emission: tuple[str, float] | None = None,
    shadow: bool = True,
    single_view: bool = False,
    scene: bool = False,
) -> list[Path]:
    """Apply the given texture maps to the hero material panel and render it.

    Two layouts are available via ``scene``:

    * ``scene=False`` (default): the historical *flat swatch* — the material on a
      single plane viewed head-on, optionally rotated through four 90° views.
      Cheapest and most direct, but it hides surface relief (seen edge-on) and
      gives transmissive materials nothing to refract.
    * ``scene=True``: a small *realistic scene* — the panel stood upright on a
      floor in front of a back wall, with two neutral primitives placed behind it,
      lit by an HDRI plus a key sun and viewed from a fixed three-quarter angle.
      This makes displacement relief cast shadows (so it reads as 3D) and gives
      transmission/IOR something to refract, and looks closer to the photographs
      the CLIP/BLIP/VQA evaluators in :mod:`matloom.metrics` expect.
    """
    material = dict(
        basecolor=basecolor,
        metallic=metallic,
        roughness=roughness,
        specular=specular,
        sheen=sheen,
        coat=coat,
        transmission=transmission,
        subsurface=subsurface,
        anisotropy=anisotropy,
        normal=normal,
        ao=ao,
        height=height,
        displacement=displacement,
        emission=emission,
        **({"ior": ior} if ior is not None else {}),
    )
    _render = _render_scene if scene else _render_plane
    return _render(out_dir, material, shadow=shadow, single_view=single_view)


def _render_plane(
    out_dir: PathLike, material: dict, *, shadow: bool, single_view: bool
) -> list[Path]:
    """Flat-swatch render: the material on a single head-on plane (the original
    behaviour). With ``single_view=False`` the plane is rotated through four 90°
    views (each de-rotated back to upright in the saved image)."""
    bpa.clear()
    bpa.initialize(transparent=True, environment_map=(str(HDRI), 0.5))

    builder = bpa.Builder()
    plane = builder.new_plane()
    builder.add_material(plane, **material)

    renderer = bpa.Renderer()
    # When shadows are off the plane casts no shadow rays, so the only crevice
    # darkening comes from the AO map — useful for confirming AO is contributing
    # (best paired with a flat plane, i.e. no displacement / `--no-height`).
    if not shadow:
        renderer.disable_shadow()
    center, radius = renderer.compute_bounding_sphere()
    out_dir = Path(out_dir)
    paths: list[Path] = []
    for i in range(1 if single_view else 4):
        path = out_dir / f"render_plane_view-{i}_shadow-{shadow}.png"
        paths.append(path)
        if i > 0:
            bpa.transform(plane, rotation=(0, 0, 90))
        renderer.render_perspective(
            str(path), center, radius, resolution=512, fit_ratio=1.0
        )
        if i > 0:
            renderer.rotate_image(str(path), str(path), angle=-(i * 90))
    return paths


def _render_scene(
    out_dir: PathLike, material: dict, *, shadow: bool, single_view: bool
) -> list[Path]:
    """Realistic-scene render: the hero material panel stood upright inside a real
    enclosed studio (a closed neutral-grey room lit by real area lights), with real
    props both behind it (camera context + something for glass to refract) and in
    front of it (what a mirror reflects), shot from a fixed three-quarter angle.

    The world is ONE consistent scene — no per-ray visibility tricks. A mirror
    reflects the actual room and the real, fully-shadowed front props; a matte
    panel's frame is unchanged because those front props sit outside the camera
    frustum. The panel is the same 2×2 plane the flat path uses (so displacement
    scaling is unchanged), rotated to stand vertical. ``single_view`` renders the
    one hero angle; otherwise the camera orbits a front-facing fan of azimuths."""
    bpa.clear()
    # A faint world grey shows only through any gaps; the enclosed room provides
    # the real environment. More samples — a closed room + refraction is noisy.
    bpa.initialize(
        transparent=False,
        background_color=SCENE_WORLD_GREY,
        samples=SCENE_SAMPLES,
    )

    builder = bpa.Builder()

    # The hero panel: a 2×2 plane stood upright (rotate 90° about X so its front
    # face looks toward the camera at -Y), lifted so it rests on the floor.
    panel = builder.new_plane(name="Panel")
    builder.add_material(panel, **material)
    bpa.transform(
        panel, rotation=(90, 0, 0), position=(0, 0, SCENE_PANEL_LIFT)
    )

    _build_studio_room(builder)
    _build_scene_props(builder)

    renderer = bpa.Renderer()
    # Real area lights. Only the key casts shadows (the fills are shadowless), so
    # every cast shadow is parallel — one coherent key direction.
    for position, aim_at, energy, size, casts in SCENE_LIGHTS:
        renderer.add_area_light(
            position=position,
            aim_at=aim_at,
            energy=int(energy),
            size=size,
            use_shadow=casts,
        )
    if not shadow:
        renderer.disable_shadow()

    center = (0.0, 0.0, SCENE_PANEL_LIFT)
    radius = _PANEL_RADIUS * SCENE_CAM_MARGIN
    # Single hero angle, or a front-facing fan of azimuths around the panel.
    if single_view:
        offsets = [0.0]
    else:
        offsets = [0.0, -75.0, -45.0, -15.0, 15.0, 45.0]
    out_dir = Path(out_dir)
    paths: list[Path] = []
    for i, daz in enumerate(offsets):
        path = out_dir / f"render_scene_view-{i}_shadow-{shadow}.png"
        paths.append(path)
        rotation = (90.0 - SCENE_CAM_ELEVATION, 0.0, SCENE_CAM_AZIMUTH + daz)
        renderer.render_perspective(
            str(path),
            Vector(center),
            radius,
            rotation=rotation,
            resolution=512,
            fit_ratio=SCENE_CAM_FIT_RATIO,
        )
    return paths


def _build_studio_room(builder: "bpa.Builder") -> None:
    """Build the enclosed studio "light box": a closed neutral-grey room (floor +
    four walls + ceiling) around the panel. This is the *real* environment a mirror
    reflects — coherent and complete from every angle, with no HDRI to tear and no
    emissive light panels to pop as white quads in the reflection."""
    half = SCENE_ROOM_HALF
    h = SCENE_ROOM_HEIGHT

    def surface(name, position, rotation, sx, sy, color):
        plane = builder.new_plane(name=name)
        bpa.transform(
            plane, rotation=rotation, scale=(sx, sy, 1.0), position=position
        )
        builder.add_material(plane, basecolor=color, roughness=0.85)

    surface("Floor", (0, 0, 0), (0, 0, 0), half, half, SCENE_ROOM_GREY)
    surface("Ceiling", (0, 0, h), (0, 0, 0), half, half, SCENE_CEILING_GREY)
    # Four upright walls (each a plane rotated to stand vertical, centred at h/2).
    surface(
        "WallBack", (0, half, h / 2), (90, 0, 0), half, h / 2, SCENE_ROOM_GREY
    )
    surface(
        "WallFront",
        (0, -half, h / 2),
        (90, 0, 0),
        half,
        h / 2,
        SCENE_ROOM_GREY,
    )
    surface(
        "WallLeft",
        (-half, 0, h / 2),
        (90, 0, 90),
        half,
        h / 2,
        SCENE_ROOM_GREY,
    )
    surface(
        "WallRight",
        (half, 0, h / 2),
        (90, 0, 90),
        half,
        h / 2,
        SCENE_ROOM_GREY,
    )


def _build_scene_props(builder: "bpa.Builder") -> None:
    """Place real, fully-shadowed props around the panel — back props (behind) and
    front props (in front, reflected by a mirror). See `_build_back_props` /
    `_build_front_props`."""
    _build_back_props(builder)
    _build_front_props(builder)


def _build_back_props(builder: "bpa.Builder") -> None:
    """Back props (positive Y, behind the panel): a sphere and a cube that straddle
    the panel's side edges — part shows around the silhouette (clearly a separate
    object) and part shows refracted through a transmissive panel. They give an
    opaque material scene context and a glass material something to refract.

    Kept CLOSE behind the panel so their contact shadow lands near the panel and
    reads clearly (placed far back against the distant wall, they look ungrounded —
    "floating")."""
    sphere = builder.new_uv_sphere(name="BackSphere", radius=0.8)
    bpa.transform(sphere, position=(1.5, 1.2, 0.8))
    builder.add_material(sphere, basecolor=(205, 120, 55), roughness=0.45)

    cube = builder.new_cube(name="BackCube")
    # Spin 30° about Z so the camera (looking down ~18° and across) sees two side
    # faces + the top — a clear 3D cube, not a single flat blue face.
    bpa.transform(
        cube,
        rotation=(0, 0, 30),
        scale=(1.3, 1.3, 1.3),
        position=(-1.9, 1.2, 0.65),
    )
    builder.add_material(cube, basecolor=(70, 110, 200), roughness=0.55)


def _build_front_props(builder: "bpa.Builder") -> None:
    """Front props (negative Y, in front of the panel): a colorful little
    still-life that a mirror/metal panel reflects. These are ordinary objects — they
    cast and receive real shadows — but are placed OUTSIDE the camera's ~40° frustum
    (>19° off-axis; see the frustum analysis in the module docstring), so the camera
    never sees them directly (a matte panel's frame is unchanged) while the panel's
    reflected field of view does include them. No ray-visibility tricks."""
    # (builder, position, scale, rotation_deg, color). Orientation makes each prop
    # read as solid 3D in the reflection: the camera (and so the mirror) looks down
    # ~18° and across, so an UPRIGHT 3-sided cone already shows two slanted faces +
    # the vertical edge (a clear pyramid) while resting flat on the floor — tilting
    # it would only sink its base through the floor. A small Z spin keeps the edge
    # off-centre. The tetra sits at y=-3.7 so a clear part clears the camera frustum.
    front: list[tuple] = [
        (
            lambda: builder.new_uv_sphere(name="FrontSphere", radius=0.7),
            (-1.9, -2.6, 0.7),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
            (210, 70, 60),
        ),
        (
            lambda: builder.new_cone(
                name="FrontTetra", radius=0.8, depth=1.5, vertices=3
            ),
            (-0.2, -3.7, 0.75),  # depth 1.5 → base at z=0 (rests on the floor)
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),  # upright; small spin so the edge is off-centre
            (80, 175, 95),
        ),
        (
            lambda: builder.new_torus(
                name="FrontTorus", major_radius=0.7, minor_radius=0.24
            ),
            (-2.8, -2.0, 0.9),
            (1.0, 1.0, 1.0),
            (80.0, 0.0, 20.0),
            (235, 195, 60),
        ),
    ]
    for make, position, scale, rotation, color in front:
        obj = make()
        bpa.transform(obj, rotation=rotation, scale=scale, position=position)
        builder.add_material(obj, basecolor=color, roughness=0.4)


# |======================================================|
# |   Render the texture maps written by main.py.export  |
# |======================================================|


def _find_map(maps_dir: Path, stem: str, exts: tuple[str, ...]) -> str | None:
    """Return the first ``<stem>.<ext>`` that exists in ``maps_dir`` (or None)."""
    for ext in exts:
        path = maps_dir / f"{stem}.{ext}"
        if path.exists():
            return str(path)
    return None


def _displacement_for(
    height_path: str | None,
    region_span: float,
    height_scale: float | None,
) -> tuple[str, float, float] | None:
    """Build the ``displacement=(path, midlevel, scale)`` tuple from an exported
    height map, matching the viewer's relief proportions.

    The exported height is an absolute value in coordinate units with the
    substrate at 0, so ``midlevel`` is 0 (negative heights carve below it). The
    scale maps that coordinate-unit height onto the 2-unit Blender plane:
    ``PLANE_SIZE / region_span`` by default, or ``height_scale`` verbatim when
    the caller overrides it.
    """
    if height_path is None:
        return None
    scale = (
        height_scale
        if height_scale is not None
        else PLANE_SIZE / max(region_span, 1e-9)
    )
    return (height_path, 0.0, scale)


def maps_to_material(
    maps_dir: PathLike,
    out_dir: PathLike,
    *,
    region_span: float = 1.0,
    height_scale: float | None = None,
    enabled: set[str] | None = None,
    shadow: bool = True,
    single_view: bool = True,
    scene: bool = False,
) -> list[Path]:
    """Render the texture maps written by ``LayeredMaterial.export`` (basecolor,
    metallic, roughness, normal, ao, emissive, height) from a directory.

    Height drives *real geometry displacement* (not a bump illusion), so the
    render shows the same 3D relief as the browser viewer; ``region_span`` is the
    View region's extent in coordinate units, used to scale that displacement.

    ``enabled`` selects which of :data:`CHANNELS` to route into the material
    (default: all). A disabled channel is skipped even if its map is present, so
    a single texture can be isolated to confirm it is rendering correctly.

    ``scene`` chooses the layout: a flat head-on swatch (default) or a realistic
    angled scene that reveals relief and transmission (see
    :func:`textures_to_material`).
    """
    src = Path(maps_dir)
    if not src.is_dir():
        raise NotADirectoryError(f"maps directory not found: {maps_dir}")

    on = set(CHANNELS) if enabled is None else enabled

    def pick(stem: str, exts: tuple[str, ...]) -> str | None:
        return _find_map(src, stem, exts) if stem in on else None

    basecolor = pick("basecolor", ("png",))
    if "basecolor" in on and basecolor is None:
        raise FileNotFoundError(f"basecolor.png not found in {maps_dir}")
    metallic = pick("metallic", ("png",))
    roughness = pick("roughness", ("png",))
    sheen = pick("sheen", ("png",))
    coat = pick("coat", ("png",))
    transmission = pick("transmission", ("png",))
    subsurface = pick("subsurface", ("png",))
    anisotropy = pick("anisotropy", ("png",))
    ior = pick("ior", ("exr", "hdr"))
    normal = pick("normal", ("png",))
    ao = pick("ao", ("png",))
    emissive = pick("emissive", ("exr", "hdr"))
    height = pick("height", ("exr", "hdr"))

    return textures_to_material(
        out_dir,
        basecolor=basecolor,
        metallic=metallic,
        roughness=roughness,
        sheen=sheen,
        coat=coat,
        transmission=transmission,
        ior=ior,
        subsurface=subsurface,
        anisotropy=anisotropy,
        normal=normal,
        ao=ao,
        # Real displacement from the height field (the viewer physically
        # displaces the mesh); the bump `height` channel is left unused.
        displacement=_displacement_for(height, region_span, height_scale),
        emission=(emissive, 1.0) if emissive is not None else None,
        shadow=shadow,
        single_view=single_view,
        scene=scene,
    )


def program_to_material(
    program: str,
    out_dir: PathLike,
    *,
    width: int = 512,
    height: int = 512,
    height_scale: float | None = None,
    enabled: set[str] | None = None,
    shadow: bool = True,
    single_view: bool = True,
    scene: bool = False,
) -> list[Path]:
    """Export the maps of a DSL ``program`` to a temporary directory, then render
    them. The displacement scale is derived from the program's own View span.
    ``scene`` selects the flat swatch (default) or the realistic scene layout."""
    # Imported lazily so the engine's heavy import chain is paid only when a
    # program (rather than a pre-exported maps dir) is rendered.
    from matloom.engine.main import LayeredMaterial

    material = LayeredMaterial.deserialize(program)
    x1, _, x2, _ = material._view
    region_span = abs(x2 - x1)

    with tempfile.TemporaryDirectory() as tmp:
        material.export(Path(tmp), width=width, height=height)
        return maps_to_material(
            tmp,
            out_dir,
            region_span=region_span,
            height_scale=height_scale,
            enabled=enabled,
            shadow=shadow,
            single_view=single_view,
            scene=scene,
        )


def program_to_image(
    program: str,
    *,
    width: int = 512,
    height: int = 512,
    height_scale: float | None = None,
    enabled: set[str] | None = None,
    shadow: bool = True,
    scene: bool = False,
):
    """Render a DSL ``program`` and return the head-on render as a **BGR**
    ``numpy.ndarray`` (H, W, 3) **without leaving anything on disk** — the render
    goes to a temp dir that is removed before returning. Used by the text-to-DSL
    refinement loop to feed the rendered material to a vision critic. ``scene``
    selects the flat swatch (default) or the realistic scene layout. Requires a
    working Blender (Cycles) install; raises if the render did not produce an
    image."""
    import cv2

    with tempfile.TemporaryDirectory(prefix="matloom-render-") as tmp:
        renders = program_to_material(
            program,
            tmp,
            width=width,
            height=height,
            height_scale=height_scale,
            enabled=enabled,
            shadow=shadow,
            single_view=True,
            scene=scene,
        )
        path = renders[0]
        if not path.exists():
            raise RuntimeError("Render produced no image")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise RuntimeError(f"Failed to read render at {path}")
        return img.copy()  # Owned (temp dir is about to vanish)


# |======================================================|
# |                         CLI                          |
# |======================================================|


def _show_images(paths: list[PathLike]) -> None:
    """Display rendered images in a blocking OpenCV window.

    The images are read into memory here, so the caller's temp files can be
    deleted immediately afterwards (nothing is left on disk). ``imshow`` blocks
    until a key is pressed; it needs a GUI-enabled OpenCV build and a display, so
    on a headless box (no display / a ``headless`` cv2 wheel) it raises — we fall
    back to just printing the paths the caller already persisted.
    """
    import cv2

    imgs = [
        (Path(p), cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
        for p in paths
        if exists(p)
    ]
    imgs = [(p, im) for p, im in imgs if im is not None]
    if not imgs:
        return
    try:
        for path, im in imgs:
            cv2.imshow(path.name, im)
        logger.info(
            "Showing render(s); press any key in the image window to close."
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        # Headless OpenCV (no HighGUI) — can't pop a window.
        raise RuntimeError("OpenCV has no display support") from e


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matloom-render",
        description=(
            "Render a layered material in Blender (Cycles) and show the result. "
            "Provide either a DSL --program (the string matloom-generate prints), "
            "a --maps-dir of already-exported texture maps, or a --text prompt "
            "that the LLM text-to-DSL pipeline turns into a program first."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--program",
        help="the material DSL program to render (exported internally)",
    )
    src.add_argument(
        "--maps-dir",
        type=Path,
        help="directory of exported maps (output of matloom-export)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "directory to save the render into (default: a temp dir opened in "
            "the OS image viewer, not persisted)"
        ),
    )
    p.add_argument(
        "--all-views",
        action="store_true",
        help="render 4 views (rotated 90° each) instead of a single view",
    )
    p.add_argument(
        "--scene",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "render the material in a realistic angled scene (floor, back wall, "
            "background props, HDRI + sun) instead of a flat head-on plane; "
            "reveals surface relief and transmission/IOR and looks closer to the "
            "photographs the CLIP/BLIP/VQA evaluators expect"
        ),
    )
    p.add_argument(
        "--shadow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "let the plane cast shadow rays (default: on); --no-shadow isolates "
            "the AO map's darkening (pair with --no-height for a flat plane)"
        ),
    )
    p.add_argument(
        "--export-width",
        type=int,
        default=512,
        help="export width in pixels (--program only)",
    )
    p.add_argument(
        "--export-height",
        type=int,
        default=512,
        help="export height in pixels (--program only)",
    )
    p.add_argument(
        "--region-span",
        type=float,
        default=1.0,
        help=(
            "View region extent in coordinate units, for displacement scaling "
            "(--maps-dir only; --program reads it from the program's View)"
        ),
    )
    p.add_argument(
        "--height-scale",
        type=float,
        default=None,
        help="override the displacement scale (default: 2 / region span)",
    )
    # Per-channel on/off toggles: `--ao`/`--no-ao`, etc. Default None ("not
    # specified") so an unset channel stays enabled. Disabling a channel skips
    # its map even when present, which is the point — render one texture at a
    # time to confirm it is contributing.
    ch = p.add_argument_group(
        "channels",
        "enable/disable individual texture channels (default: all on); "
        "e.g. --no-ao --no-normal to isolate the base color",
    )
    for name in CHANNELS:
        ch.add_argument(
            f"--{name}",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"render the {name} channel (use --no-{name} to disable)",
        )
    return p


def _enabled_channels(args: argparse.Namespace) -> set[str]:
    """Resolve the per-channel toggles into the set of channels to render. A
    channel left unspecified (None) stays on; only an explicit ``--no-<channel>``
    drops it."""
    return {name for name in CHANNELS if getattr(args, name) is not False}


def _render_into(args: argparse.Namespace, out_dir: str) -> list[Path]:
    single_view = not args.all_views
    enabled = _enabled_channels(args)
    # A --text prompt is turned into a DSL program by the LLM pipeline first,
    # then rendered exactly like a --program. The pipeline is imported lazily so
    # the LLM stack is only loaded when text generation is actually requested.
    program = args.program
    if program is not None:
        return program_to_material(
            program,
            out_dir,
            width=args.export_width,
            height=args.export_height,
            height_scale=args.height_scale,
            enabled=enabled,
            shadow=args.shadow,
            single_view=single_view,
            scene=args.scene,
        )
    else:
        return maps_to_material(
            str(args.maps_dir),
            out_dir,
            region_span=args.region_span,
            height_scale=args.height_scale,
            enabled=enabled,
            shadow=args.shadow,
            single_view=single_view,
            scene=args.scene,
        )


def _cli(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in _render_into(args, str(out_dir)):
            if p.exists():
                logger.warning(p)
        return

    # No output dir: render into a self-cleaning temp dir. Because the OpenCV
    # viewer blocks (and reads the images into memory), the temp dir can be
    # removed as soon as the window closes — nothing is left on disk.
    with tempfile.TemporaryDirectory(prefix="matloom-render-") as tmp:
        pngs = [p for p in _render_into(args, tmp) if p.exists()]
        try:
            _show_images(pngs)
        except RuntimeError:
            # Headless OpenCV can't show a window; persist instead so the render
            # isn't lost with the temp dir, and tell the user where it went.
            fallback = Path(tempfile.mkdtemp(prefix="matloom-render-"))
            for p in pngs:
                shutil.copy2(p, fallback / p.name)
            logger.error(
                "No display available (headless OpenCV); saved render(s) to:\n"
                + "\n".join(str(fallback / p.name) for p in pngs)
            )


if __name__ == "__main__":
    _cli()
