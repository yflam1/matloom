"""Tests for matloom.render — the Blender (Cycles) rendering pipeline.

These exercise the real renderer, so they require a working Blender install
(``bpy``) and OpenCV (``cv2``); both are skipped when unavailable, which is the
case in the torch-free / Blender-free CI unit job (see AGENTS.md "Testing"). They
are intentionally small (low resolution + sample count) so they stay fast.

The focus is the scene-vs-plane layout switch added for revealing surface relief
and transmission/IOR: that both layouts produce an image, and that the realistic
scene actually contains more than the bare swatch (a floor, background props,
lighting) so a vision evaluator sees a scene rather than a flat plane.
"""

from __future__ import annotations

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")
pytest.importorskip("cv2")

from matloom import render


@pytest.fixture(autouse=True)
def _quiet_blender(monkeypatch):
    """Neutralize bpa's fd-level ``redirect_stdout`` for the duration of a test.

    The real context manager closes and re-opens file descriptor 1 to silence
    Blender's render spam; under pytest's output capture that corrupts the
    captured stream and crashes the session at teardown. Replacing it with a
    no-op keeps the genuine Cycles render running (the thing under test) while
    leaving pytest's stdout intact."""
    import contextlib

    @contextlib.contextmanager
    def _null(*args, **kwargs):
        yield

    monkeypatch.setattr(render.bpa, "redirect_stdout", _null)


# A program with strong, varied relief (so displacement is visible) — the case a
# flat head-on swatch hides and the scene's grazing light is meant to reveal.
_RELIEF = """View(0, 0, 4, 4)
Define(brick, Bricks(brick_width=1, brick_height=0.5, mortar=0.08, feather=0.01))
Material(
  Layer(1).basecolor(170, 165, 155).roughness(0.95),
  Layer(brick).basecolor(160, 70, 50).roughness(0.85).height(0.25)
)"""

# A clear, smooth, transmissive panel — needs something behind it to refract,
# which only the scene layout provides.
_GLASS = """View(0, 0, 1, 1)
Material(
  Layer(1).basecolor(235, 240, 245).roughness(0.05).transmission(0.98).ior(1.5)
)"""

# A perfect mirror — reflects the studio room and the real front props in front of
# the panel (a consistent world; no HDRI, no ray-visibility tricks).
_MIRROR = """View(0, 0, 1, 1)
Material(
  Layer(1).basecolor(255, 255, 255).roughness(0).metallic(1)
)"""


def _render(program: str, tmp_path, *, scene: bool) -> np.ndarray:
    import cv2

    out = tmp_path / ("scene" if scene else "plane")
    out.mkdir()
    renders = render.program_to_material(
        program, str(out), width=128, height=128, scene=scene, single_view=True
    )
    path = renders[0]
    assert path.exists(), "render produced no image"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def test_plane_render_produces_image(tmp_path):
    img = _render(_RELIEF, tmp_path, scene=False)
    assert img.shape[2] == 3
    # A real render, not a uniform fill.
    assert img.std() > 1.0


def test_scene_render_produces_image(tmp_path):
    img = _render(_RELIEF, tmp_path, scene=True)
    assert img.shape[2] == 3
    assert img.std() > 1.0


def test_scene_is_richer_than_bare_plane(tmp_path):
    """The scene layout adds a floor and background props around the hero panel,
    so its bottom strip (the floor) should differ markedly from the plane render,
    whose bottom is empty background."""
    plane = _render(_RELIEF, tmp_path, scene=False).astype(float)
    scene = _render(_RELIEF, tmp_path, scene=True).astype(float)
    h = scene.shape[0]
    # The scene's lower third is dominated by the lit floor; the flat swatch's is
    # background. They must not be near-identical.
    scene_floor = scene[2 * h // 3 :].mean()
    plane_floor = plane[2 * h // 3 :].mean()
    assert abs(scene_floor - plane_floor) > 5.0


def test_scene_transmission_refracts_background(tmp_path):
    """Through a clear transmissive panel, the scene's background props are
    refracted, so the panel centre shows spatial variation rather than a flat
    color (which is all a plane-only render of glass could show)."""
    img = _render(_GLASS, tmp_path, scene=True).astype(float)
    h, w, _ = img.shape
    center = img[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    # Refracted scene detail means a non-trivial spread of values in the centre.
    assert center.std() > 2.0


def test_program_to_image_scene_returns_rgb(tmp_path):
    img = render.program_to_image(_RELIEF, width=128, height=128, scene=True)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3


def test_new_uv_sphere_is_a_mesh():
    bpy_render = render.bpa
    bpy_render.clear()
    sphere = bpy_render.Builder.new_uv_sphere(radius=0.5)
    assert bpy_render.is_mesh(sphere)
    assert len(sphere.data.vertices) > 0


def test_scene_backdrop_is_flat_not_hdri(tmp_path):
    """The scene's world is a flat neutral grey (HDRI-free studio), so the top
    corners (above the panel and props) should be a near-uniform color rather than
    the varied content of an HDRI environment, and match left-to-right."""
    img = _render(_RELIEF, tmp_path, scene=True).astype(float)
    corner = 16
    tl = img[:corner, :corner]
    tr = img[:corner, -corner:]
    # Each corner is near-uniform (a flat grey world has tiny variance)...
    assert tl.std() < 6.0 and tr.std() < 6.0
    # ...and both corners are the SAME color (an HDRI would differ left vs right).
    assert abs(tl.mean(axis=(0, 1)) - tr.mean(axis=(0, 1))).max() < 8.0


def test_environment_invisible_keeps_lighting(tmp_path):
    """`bpa.initialize`'s ``environment_visible=False`` (a bpa feature still used by
    external callers; the scene itself is now HDRI-free) must keep the HDRI lighting
    the scene (a lit lambert sphere is not black) while the camera sees the flat
    backdrop."""
    import cv2

    bpa = render.bpa
    bpa.clear()
    bpa.initialize(
        transparent=False,
        environment_map=(str(render.HDRI), 1.0),
        environment_visible=False,
        background_color=(40, 40, 40),
        samples=16,
    )
    builder = bpa.Builder()
    sphere = builder.new_uv_sphere(radius=1.0)
    builder.add_material(sphere, basecolor=(200, 200, 200), roughness=0.6)
    renderer = bpa.Renderer()
    center, radius = renderer.compute_bounding_sphere()
    out = tmp_path / "env.png"
    renderer.render_perspective(str(out), center, radius, resolution=128, fit_ratio=1.0)
    assert out.exists()
    img = cv2.imread(str(out), cv2.IMREAD_COLOR).astype(float)
    # The sphere is lit by the HDRI (mean brightness well above black).
    assert img.mean() > 20.0


def test_scene_world_is_flat_grey_no_hdri():
    """The realistic scene is HDRI-free: after initializing it the world must have
    NO environment-texture node (just a flat-color Background) — a regression guard
    that the scene never reintroduces an HDRI (which a mirror would reflect)."""
    bpa = render.bpa
    bpa.clear()
    bpa.initialize(
        transparent=False, background_color=render.SCENE_WORLD_GREY, samples=8
    )
    world = bpa.bpy.context.scene.world
    env_nodes = [n for n in world.node_tree.nodes if n.type == "TEX_ENVIRONMENT"]
    assert env_nodes == [], "scene world must be HDRI-free (no environment texture)"


def test_scene_uses_no_emissive_lights():
    """The studio is lit by real AREA lights, not emissive meshes (which popped as
    a white quad in mirror reflections). Build the scene and assert there are area
    lights and NO mesh material has a non-zero emission strength."""
    bpa = render.bpa
    bpa.clear()
    bpa.initialize(
        transparent=False, background_color=render.SCENE_WORLD_GREY, samples=8
    )
    builder = bpa.Builder()
    panel = builder.new_plane(name="Panel")
    builder.add_material(panel, basecolor=(255, 255, 255), roughness=0.0, metallic=1.0)
    bpa.transform(panel, rotation=(90, 0, 0), position=(0, 0, render.SCENE_PANEL_LIFT))
    render._build_studio_room(builder)
    render._build_scene_props(builder)
    renderer = bpa.Renderer()
    for position, aim_at, energy, size, casts in render.SCENE_LIGHTS:
        renderer.add_area_light(
            position=position,
            aim_at=aim_at,
            energy=int(energy),
            size=size,
            use_shadow=casts,
        )
    lights = [o for o in bpa.bpy.data.objects if o.type == "LIGHT"]
    assert len(lights) >= 2 and all(light.data.type == "AREA" for light in lights)
    # No mesh emits light (no emissive softbox geometry to pop in a reflection).
    for mat in bpa.bpy.data.materials:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                assert node.inputs["Emission Strength"].default_value == 0.0


def test_scene_no_blown_white_quad(tmp_path):
    """Regression for the white-polygon artifact: a mirror render must contain no
    blown-out pure-white region (the old emissive softbox reflected as a hard white
    quad). Real area lights do not appear as geometry in the reflection."""
    import cv2

    out = tmp_path / "m"
    out.mkdir()
    renders = render.program_to_material(
        _MIRROR, str(out), width=256, height=256, scene=True, single_view=True
    )
    img = cv2.imread(str(renders[0]), cv2.IMREAD_GRAYSCALE).astype(float)
    near_white = (img >= 253).mean()
    assert near_white < 0.003, f"a blown-white quad appears: {near_white:.4f}"


def _scene_floor_shadow_blobs(img: np.ndarray) -> int:
    """Count distinct shadow regions on the lower (floor) half of a scene render,
    ignoring specks. Used to assert a single key light → a single shadow set."""
    import cv2

    h = img.shape[0]
    floor = img[int(h * 0.6) :, :]
    median = float(np.median(floor))
    dark = (floor < median - 12).astype(np.uint8)
    n, labels = cv2.connectedComponents(dark)
    min_size = floor.size * 0.004
    return sum(1 for i in range(1, n) if int((labels == i).sum()) > min_size)


def test_scene_single_key_light_one_shadow(tmp_path):
    """Only the key light casts shadows (the fills are shadowless), so an opaque
    panel with no props produces a single shadow blob on the floor — not the
    multiple, conflicting shadows older multi-light setups produced."""
    import cv2

    opaque = (
        "View(0, 0, 1, 1)\nMaterial(\n  Layer(1).basecolor(180, 90, 70)"
        ".roughness(0.9).metallic(0)\n)"
    )
    out = tmp_path / "s"
    out.mkdir()

    # Drop the props so the only floor shadow is the panel's (the room stays).
    orig_props = render._build_scene_props
    render._build_scene_props = lambda builder: None
    try:
        renders = render.program_to_material(
            opaque,
            str(out),
            width=160,
            height=160,
            scene=True,
            single_view=True,
        )
    finally:
        render._build_scene_props = orig_props
    img = cv2.imread(str(renders[0]), cv2.IMREAD_GRAYSCALE).astype(float)
    assert _scene_floor_shadow_blobs(img) == 1


def _front_prop_color_frac(img: np.ndarray) -> float:
    """Fraction of pixels matching the front green tetrahedron (a prop that exists
    ONLY in front of the panel). It should appear in a mirror's reflection but
    never directly in the camera frame (it is outside the frustum). BGR input."""
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return float(((g > r + 20) & (g > b + 15)).mean())


def test_mirror_reflects_front_props(tmp_path):
    """A mirror panel must reflect the real front props — including the green
    tetrahedron that exists only in front of the panel — so its face shows that
    color. A consistent world: the props are real, fully-shadowed objects."""
    img = _render(_MIRROR, tmp_path, scene=True)  # BGR
    assert _front_prop_color_frac(img) > 0.01, "mirror does not reflect front props"


def test_front_props_outside_camera_frustum(tmp_path):
    """The front props are real objects (cast/receive shadow, camera-visible flag
    on) but placed outside the camera frustum, so a MATTE panel — which does not
    reflect them — must render almost identically whether the front props are
    present or not. (This is what replaces the old reflection-only ray-visibility
    trick: consistency, verified by the props having no effect on the direct view.)
    Only the FRONT props are toggled; the room and back props stay."""
    import cv2

    matte = (
        "View(0, 0, 1, 1)\nMaterial(\n  Layer(1).basecolor(180, 180, 180)"
        ".roughness(1).metallic(0)\n)"
    )

    a = tmp_path / "with"
    a.mkdir()
    renders_a = render.program_to_material(
        matte, str(a), width=128, height=128, scene=True, single_view=True
    )
    img_with = cv2.imread(str(renders_a[0]), cv2.IMREAD_COLOR).astype(float)

    # Re-render with the FRONT props disabled (the room + back props stay).
    orig_front = render._build_front_props
    render._build_front_props = lambda builder: None
    b = tmp_path / "without"
    b.mkdir()
    try:
        renders_b = render.program_to_material(
            matte, str(b), width=128, height=128, scene=True, single_view=True
        )
    finally:
        render._build_front_props = orig_front
    img_without = cv2.imread(str(renders_b[0]), cv2.IMREAD_COLOR).astype(float)

    # The front props are outside the frustum, so a matte (non-reflective) frame is
    # essentially unchanged by them (small tolerance for path-tracing noise).
    assert float(np.abs(img_with - img_without).mean()) < 1.5


def test_set_ray_visibility_toggles_attrs():
    bpa = render.bpa
    bpa.clear()
    cube = bpa.Builder.new_cube()
    bpa.set_ray_visibility(
        cube,
        camera=False,
        diffuse=False,
        glossy=True,
        transmission=False,
        shadow=False,
    )
    assert cube.visible_camera is False
    assert cube.visible_glossy is True
    assert cube.visible_diffuse is False
    assert cube.visible_transmission is False
    assert cube.visible_shadow is False


def test_new_cone_and_torus_are_meshes():
    bpa = render.bpa
    bpa.clear()
    tetra = bpa.Builder.new_cone(vertices=3)  # a 4-vertex tetrahedron
    torus = bpa.Builder.new_torus()
    assert bpa.is_mesh(tetra) and len(tetra.data.vertices) == 4
    assert bpa.is_mesh(torus) and len(torus.data.vertices) > 0


def test_add_area_light_aims_and_sets_shadow():
    """`add_area_light` must aim its emitting face (local -Z) at ``aim_at`` and
    honour ``use_shadow``/``size`` (the softbox knobs the scene relies on)."""
    import math

    bpa = render.bpa
    bpa.clear()
    light = bpa.Renderer().add_area_light(
        position=(0, -3, 3),
        aim_at=(0, 0, 1),
        energy=200,
        size=2.0,
        use_shadow=False,
    )
    assert light.type == "LIGHT"
    assert light.data.use_shadow is False
    assert math.isclose(light.data.size, 2.0, rel_tol=1e-6)
    # The light's -Z axis (its emitting direction) should point toward the target.
    # matrix_world is lazily evaluated, so flush the depsgraph first.
    render.bpa.bpy.context.view_layer.update()
    forward = light.matrix_world.to_quaternion() @ render.bpa.Vector((0, 0, -1))
    target_dir = render.bpa.Vector((0, 0, 1)) - render.bpa.Vector((0, -3, 3))
    target_dir.normalize()
    assert forward.dot(target_dir) > 0.95
