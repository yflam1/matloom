"""Tests for matloom.engine.main: LayeredMaterial (de)serialization and export."""

from __future__ import annotations

import numpy as np
import pytest

from matloom.engine.expr import Constant
from matloom.engine.layer import Color, Emissive, Layer
from matloom.engine.main import LayeredMaterial, _cli

# |---------------------|
# |   Basic structure   |
# |---------------------|


def test_empty_material_has_no_layers():
    m = LayeredMaterial()
    assert m.n_layers == 0


def test_add_layer_increments_count():
    m = LayeredMaterial()
    m.add_layer(Layer())
    m.add_layer(Layer())
    assert m.n_layers == 2


def test_empty_material_serializes_to_empty():
    out = LayeredMaterial().serialize()
    # A code-constructed material never had `set_view` called, so no `View(...)`
    # preamble is emitted (set-tracking: an unauthored default view is omitted).
    assert "View(" not in out
    assert out == "Material()"


def test_export_empty_material_raises():
    with pytest.raises(ValueError, match="No layers"):
        LayeredMaterial().export()


# |-----------------|
# |   Serialize     |
# |-----------------|


def test_serialize_single_layer_structure():
    # A default layer (nothing explicitly set) serializes as `Layer()` with no
    # channel lines: only authored channels are emitted.
    m = LayeredMaterial()
    m.add_layer(Layer())
    s = m.serialize()
    # No `set_view` was called, so no `View(...)` preamble (the unauthored
    # default view is omitted, mirroring the omitted untouched channels below).
    assert "View(" not in s
    assert s.startswith("Material(")
    assert s.rstrip().endswith(")")
    assert "Layer(" in s
    for chan in (
        ".basecolor(",
        ".metallic(",
        ".roughness(",
        ".sheen(",
        ".coat(",
        ".transmission(",
        ".ior(",
        ".subsurface(",
        ".anisotropy(",
        ".emissive(",
        ".height(",
    ):
        assert chan not in s

    # A layer with channels set emits exactly those, in canonical order.
    m2 = LayeredMaterial.deserialize(
        "Material(Layer(0.5).basecolor(10, 20, 30).metallic(0.2).roughness(0.7))"
    )
    s2 = m2.serialize()
    assert ".basecolor(10, 20, 30)" in s2
    assert ".metallic(0.2)" in s2
    assert ".roughness(0.7)" in s2
    for chan in (
        ".sheen(",
        ".coat(",
        ".transmission(",
        ".ior(",
        ".subsurface(",
        ".anisotropy(",
        ".emissive(",
        ".height(",
    ):
        assert chan not in s2


def test_serialize_openpbr_channel_defaults():
    # A default layer emits NONE of the OpenPBR channels (set-tracking omits
    # untouched channels); the inert values live on the Layer as defaults.
    s = LayeredMaterial.deserialize("Material(Layer(1))").serialize()
    for chan in (
        ".sheen(0)",
        ".coat(0)",
        ".transmission(0)",
        ".ior(1.5)",
        ".subsurface(0)",
        ".anisotropy(0)",
    ):
        assert chan not in s

    # Explicitly-set-to-default channels ARE preserved on round-trip (the point
    # of set-tracking: an authored `.sheen(0)` is kept, not dropped).
    src = (
        "Material(Layer(1)"
        ".sheen(0).coat(0).transmission(0).ior(1.5)"
        ".subsurface(0).anisotropy(0))"
    )
    rt = LayeredMaterial.deserialize(src).serialize()
    for chan in (
        ".sheen(0)",
        ".coat(0)",
        ".transmission(0)",
        ".ior(1.5)",
        ".subsurface(0)",
        ".anisotropy(0)",
    ):
        assert chan in rt


def test_openpbr_channels_roundtrip_stably():
    src = (
        "Material(Layer(1).sheen(0.8).coat(0.2).transmission(0.5)"
        ".ior(1.45).subsurface(0.3).anisotropy(0.6))"
    )
    once = LayeredMaterial.deserialize(src).serialize()
    twice = LayeredMaterial.deserialize(once).serialize()
    assert once == twice
    assert ".sheen(0.8)" in once
    assert ".ior(1.45)" in once


def test_serialize_includes_height_expression():
    m = LayeredMaterial()
    m.add_layer(Layer(height=Constant(9.0)))
    assert ".height(9)" in m.serialize()


def test_height_roundtrips_stably():
    src = (
        "Material(Layer(1).basecolor(0, 0, 0).metallic(0).roughness(0.5)"
        ".emissive(0, 0, 0, 0).height(fBm(base_freq=4, seed=3)))"
    )
    once = LayeredMaterial.deserialize(src).serialize()
    twice = LayeredMaterial.deserialize(once).serialize()
    assert once == twice
    assert ".height(fBm(base_freq=4, seed=3))" in once


def test_missing_height_defaults_to_zero():
    # Old strings without `.height` still parse (height defaults to 0).
    src = (
        "Material(Layer(1).basecolor(0, 0, 0).metallic(0)"
        ".roughness(0.5).emissive(0, 0, 0, 0))"
    )
    m = LayeredMaterial.deserialize(src)
    assert m._layers[0].height(0.3, 0.7) == 0.0


# |-------------|
# |    View     |
# |-------------|


def test_unauthored_view_is_omitted():
    """A material deserialized from a source with no `View(...)` line omits the
    `View(...)` preamble on serialize (set-tracking: an unauthored default view
    is not re-emitted). The view still defaults to the unit square for sampling."""
    m = LayeredMaterial.deserialize("Material(Layer(1))")
    assert m._view == (0.0, 0.0, 1.0, 1.0)
    assert not m._view_set
    out = m.serialize()
    assert "View(" not in out
    assert out.startswith("Material(")


def test_unauthored_view_roundtrips_stably():
    """Absence of `View(...)` round-trips: re-deserializing the omitted output
    still omits it (byte-for-byte stable)."""
    src = "Material(Layer(1).height(3))"
    once = LayeredMaterial.deserialize(src).serialize()
    assert "View(" not in once
    assert LayeredMaterial.deserialize(once).serialize() == once


def test_set_view_marks_authored_and_emits():
    """Calling `set_view` (even to the default unit square) marks the view
    authored, so serialize emits the `View(...)` line. This is how the
    procedural generator and the deserialize path mark an explicit window."""
    m = LayeredMaterial()
    m.set_view(0.0, 0.0, 1.0, 1.0)
    assert m._view_set
    assert m.serialize().startswith("View(0, 0, 1, 1)")


def test_view_roundtrips_stably():
    src = "View(0, 0, 2, 1.5)\nMaterial(Layer(1).height(3))"
    m = LayeredMaterial.deserialize(src)
    assert m._view == (0.0, 0.0, 2.0, 1.5)
    once = m.serialize()
    assert once.startswith("View(0, 0, 2, 1.5)")
    assert LayeredMaterial.deserialize(once).serialize() == once


def test_view_before_define_and_material():
    src = (
        "View(0, 0, 4, 4)\n"
        "Define(box, Fill(Rect(0.2, 0.2, 0.3, 0.3)))\n"
        "Material(Layer(box))"
    )
    m = LayeredMaterial.deserialize(src)
    assert m._view == (0.0, 0.0, 4.0, 4.0)
    out = m.serialize()
    assert out.index("View(") < out.index("Define(") < out.index("Material(")


def test_view_wrong_arity_raises():
    # Too few args.
    with pytest.raises(ValueError, match="4 arguments"):
        LayeredMaterial.deserialize("View(0, 0, 1)\nMaterial(Layer(1))")
    # The old 5-arg form (with a mm/unit scale) is no longer accepted.
    with pytest.raises(ValueError, match="4 arguments"):
        LayeredMaterial.deserialize("View(0, 0, 1, 1, 100)\nMaterial(Layer(1))")


def test_export_produces_relief_maps(tmp_path):
    import cv2

    # A material with relief exports a non-flat normal map (and AO): sanity that
    # export() wires the composited height through the relief derivation. The
    # lateral-scale → slope relationship itself is unit-tested in test_relief.py.
    src = "Material(Layer(1).height(1), Layer(Fill(Rect(0.25,0.25,0.5,0.5))).height(9))"
    m = LayeredMaterial.deserialize(src)
    out = m.export(tmp_path / "relief", width=64, height=64)
    n = cv2.imread(str(out / "normal.png"), cv2.IMREAD_UNCHANGED)
    # The rect edge tilts the normal away from flat (128,128,255) somewhere.
    assert int(np.abs(n[:, :, 2].astype(int) - 128).max()) > 0


def test_export_integer_literal_material_with_negative_height(tmp_path):
    import cv2

    # A full-coverage layer written with integer literals (int alpha `1`, int
    # constants in the height) whose height dips below the substrate must export
    # the full relief pipeline without raising a dtype/cast error — correct math
    # should never fault inside the engine. The .hdr fallback warns (it clamps
    # negatives), which we assert here too.
    src = "Material(Layer(1).height((X() * 8 - 4)))"
    m = LayeredMaterial.deserialize(src)
    with pytest.warns(RuntimeWarning, match="Radiance|below-substrate"):
        out = m.export(tmp_path / "int_neg", width=32, height=32)
    height = cv2.imread(str(out / "height.hdr"), cv2.IMREAD_UNCHANGED)
    assert height is not None
    for name in ("normal.png", "ao.png"):
        assert (out / name).exists()


# |---------------------|
# |   Export CLI        |
# |---------------------|


def test_cli_exports_full_map_set(tmp_path, capsys):
    # The CLI deserializes --program and writes the full map set to --output-dir,
    # printing the directory it wrote. The program string is exactly the format
    # `matloom-generate` emits.
    out_dir = tmp_path / "cli_mat"
    _cli(
        [
            "--program",
            "Material(Layer(1).basecolor(200,150,100).roughness(0.5).height(Y()))",
            "--output-dir",
            str(out_dir),
            "--width",
            "16",
            "--height",
            "16",
        ]
    )
    printed = capsys.readouterr().out.strip()
    assert printed == str(out_dir)
    for name in (
        "basecolor.png",
        "metallic.png",
        "roughness.png",
        "sheen.png",
        "coat.png",
        "transmission.png",
        "subsurface.png",
        "anisotropy.png",
        "emissive.hdr",
        "ior.hdr",
        "height.hdr",
        "normal.png",
        "ao.png",
    ):
        assert (out_dir / name).exists()


def test_cli_no_y_up_flips_sampling(tmp_path):
    import cv2

    # --y-up / --no-y-up flip the Y sampling direction, so a Y-dependent basecolor
    # exports vertically mirrored maps.
    prog = "Material(Layer(1).basecolor((Y() * 255), 0, 0))"
    up = tmp_path / "up"
    down = tmp_path / "down"
    _cli(
        [
            "--program",
            prog,
            "--output-dir",
            str(up),
            "--width",
            "8",
            "--height",
            "8",
        ]
    )
    _cli(
        [
            "--program",
            prog,
            "--output-dir",
            str(down),
            "--no-y-up",
            "--width",
            "8",
            "--height",
            "8",
        ]
    )
    a = cv2.imread(str(up / "basecolor.png"), cv2.IMREAD_UNCHANGED)
    b = cv2.imread(str(down / "basecolor.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(a, b[::-1])


def test_cli_requires_program():
    with pytest.raises(SystemExit):
        _cli(["--width", "16"])


def test_serialize_color_uses_0_255_convention():
    # A color imported in 0-255 form (stored as `<int> / 255`) serializes back
    # to bare integers; this is the cross-engine interchange contract.
    src = (
        "Material(Layer(1).basecolor(255, 128, 0).metallic(0)"
        ".roughness(0.5).emissive(0, 0, 0, 0))"
    )
    s = LayeredMaterial.deserialize(src).serialize()
    assert ".basecolor(255, 128, 0)" in s


# |--------------------------|
# |   Deserialize roundtrip  |
# |--------------------------|


def test_deserialize_requires_material_prefix():
    with pytest.raises(ValueError, match="Material"):
        LayeredMaterial.deserialize("Layer(Solid)")


def test_deserialize_rejects_trailing_text():
    with pytest.raises(ValueError, match="Unexpected text"):
        LayeredMaterial.deserialize("Material()trailing")


def test_deserialize_empty_material():
    m = LayeredMaterial.deserialize("Material()")
    assert m.n_layers == 0


def test_serialize_deserialize_roundtrip_is_stable():
    m = LayeredMaterial()
    m.add_layer(Layer(basecolor=Color.from_int(r=200, g=100, b=50)))
    m.add_layer(
        Layer(
            alpha=Constant(0.5),
            metallic=Constant(1.0),
            roughness=Constant(0.3),
            emissive=Emissive(r=Constant(1.0), strength=Constant(2.0)),
        )
    )
    once = m.serialize()
    twice = LayeredMaterial.deserialize(once).serialize()
    assert once == twice


def test_deserialize_preserves_layer_count():
    src = (
        "Material(\n"
        "  Layer(1).basecolor(255, 0, 0).metallic(0).roughness(0.5)"
        ".emissive(0, 0, 0, 0),\n"
        "  Layer(0.5).basecolor(0, 255, 0).metallic(1).roughness(0.2)"
        ".emissive(0, 0, 0, 0)\n"
        ")"
    )
    m = LayeredMaterial.deserialize(src)
    assert m.n_layers == 2


def test_deserialize_flattened_single_line():
    # The README notes a hand-flattened single-line string imports fine.
    src = "Material(Layer(1).basecolor(255, 255, 255))"
    m = LayeredMaterial.deserialize(src)
    assert m.n_layers == 1


def test_layer_too_many_args_raises():
    with pytest.raises(ValueError, match="at most one argument"):
        LayeredMaterial.deserialize("Material(Layer(1, 2))")


def test_unknown_layer_property_raises():
    with pytest.raises(ValueError, match="Unknown layer property"):
        LayeredMaterial.deserialize("Material(Layer(1).bogus(2))")


def test_basecolor_wrong_arg_count_raises():
    with pytest.raises(ValueError, match="basecolor"):
        LayeredMaterial.deserialize("Material(Layer(1).basecolor(1, 2))")


# |----------------------|
# |   Define / preamble  |
# |----------------------|

_MATERIAL_WITH_DEF = (
    "Define(star, Fill(Path(0.4, 0.4, LineTo(0.6, 0.4), LineTo(0.5, 0.7))))\n"
    "Material(\n"
    "  Layer(Rotate(star, 30))\n"
    "    .basecolor(200, 120, 60)\n"
    "    .metallic(0)\n"
    "    .roughness(0.5)\n"
    "    .emissive(0, 0, 0, 0)\n"
    "    .height(0)\n"
    ")"
)


def test_define_roundtrips_stably():
    once = LayeredMaterial.deserialize(_MATERIAL_WITH_DEF).serialize()
    twice = LayeredMaterial.deserialize(once).serialize()
    assert once == twice
    # The source has no `View(...)` line, so the unauthored default view is
    # omitted and the output starts with the `Define(...)` preamble.
    assert "View(" not in once
    assert once.startswith("Define(star, ")
    assert "Rotate(star, 30)" in once  # reference kept, not inlined


def test_define_emits_preamble_before_material():
    s = LayeredMaterial.deserialize(_MATERIAL_WITH_DEF).serialize()
    assert s.index("Define(") < s.index("Material(")


def test_later_define_references_earlier():
    src = (
        "Define(box, Fill(Rect(0.2, 0.2, 0.3, 0.3)))\n"
        "Define(shifted, Translate(box, 0.4, 0.4))\n"
        "Material(Layer(shifted).basecolor(255, 255, 255)"
        ".metallic(0).roughness(0.5).emissive(0, 0, 0, 0))"
    )
    m = LayeredMaterial.deserialize(src)
    out = m.serialize()
    assert "Define(box, " in out
    assert "Define(shifted, Translate(box, 0.4, 0.4))" in out


def test_undefined_reference_raises():
    src = "Material(Layer(unknowndef).basecolor(0, 0, 0).metallic(0).roughness(0.5).emissive(0, 0, 0, 0))"
    with pytest.raises(ValueError, match="Unknown name"):
        LayeredMaterial.deserialize(src)


def test_define_bad_arity_raises():
    with pytest.raises(ValueError, match="exactly two arguments"):
        LayeredMaterial.deserialize("Define(a)\nMaterial()")


# |----------------|
# |   Coordinates  |
# |----------------|


def test_axis_coordinates_are_pixel_centered():
    m = LayeredMaterial()
    coords = m._generate_axis_coordinates(4, 0.0, 1.0)
    # Pixel centers for 4 samples across [0,1]: 0.125, 0.375, 0.625, 0.875.
    np.testing.assert_allclose(coords, [0.125, 0.375, 0.625, 0.875], atol=1e-6)


def test_axis_coordinates_reverse_for_y_up():
    m = LayeredMaterial()
    fwd = m._generate_axis_coordinates(4, 0.0, 1.0)
    rev = m._generate_axis_coordinates(4, 0.0, 1.0, reverse=True)
    np.testing.assert_allclose(rev, fwd[::-1], atol=1e-6)


# |-------------|
# |   Export    |
# |-------------|


def test_export_writes_texture_maps(tmp_path):
    m = LayeredMaterial()
    m.add_layer(Layer(basecolor=Color.from_int(r=255, g=0, b=0)))
    out = m.export(tmp_path, width=8, height=8)
    assert (out / "basecolor.png").exists()
    assert (out / "metallic.png").exists()
    assert (out / "roughness.png").exists()
    for name in ("sheen", "coat", "transmission", "subsurface", "anisotropy"):
        assert (out / f"{name}.png").exists()
    assert (out / "normal.png").exists()
    assert (out / "ao.png").exists()
    # Emissive, ior and height are .hdr unless OPENCV_IO_ENABLE_OPENEXR=1.
    assert (out / "emissive.hdr").exists() or (out / "emissive.exr").exists()
    assert (out / "ior.hdr").exists() or (out / "ior.exr").exists()
    assert (out / "height.hdr").exists() or (out / "height.exr").exists()


def test_export_flat_material_has_flat_normal(tmp_path):
    import cv2

    m = LayeredMaterial()
    m.add_layer(Layer())  # height 0 everywhere
    out = m.export(tmp_path, width=8, height=8)
    n = cv2.imread(str(out / "normal.png"), cv2.IMREAD_UNCHANGED)
    # A flat surface → OpenGL normal (128, 128, 255); OpenCV is BGR.
    assert np.all(n[:, :, 0] == 255)  # B = nz
    assert np.all(n[:, :, 1] == 128)  # G = ny
    assert np.all(n[:, :, 2] == 128)  # R = nx


def test_export_height_preserves_absolute_mm(tmp_path):
    import cv2

    # Base 1mm everywhere; a centered rect rises to 9mm. With max compositing the
    # height map holds the true mm values (1 outside, 9 inside the rect).
    src = (
        "Material(Layer(1).height(1), "
        "Layer(Fill(Rect(0.25, 0.25, 0.5, 0.5))).height(9))"
    )
    m = LayeredMaterial.deserialize(src)
    out = m.export(tmp_path, width=16, height=16)
    ext = "exr" if (out / "height.exr").exists() else "hdr"
    h = cv2.imread(str(out / f"height.{ext}"), cv2.IMREAD_UNCHANGED)
    assert h[:, :, 0].max() == pytest.approx(9.0, abs=1e-3)
    assert h[:, :, 0].min() == pytest.approx(1.0, abs=1e-3)


def test_export_basecolor_is_red(tmp_path):
    import cv2

    m = LayeredMaterial()
    m.add_layer(Layer(basecolor=Color.from_int(r=255, g=0, b=0)))
    out = m.export(tmp_path, width=8, height=8)
    img = cv2.imread(str(out / "basecolor.png"), cv2.IMREAD_UNCHANGED)
    assert img.shape == (8, 8, 4)
    # OpenCV is BGRA: red channel is index 2.
    assert np.all(img[:, :, 2] == 255)
    assert np.all(img[:, :, 0] == 0)
    assert np.all(img[:, :, 3] == 255)  # fully opaque


def test_export_respects_dimensions(tmp_path):
    import cv2

    m = LayeredMaterial()
    m.add_layer(Layer())
    out = m.export(tmp_path, width=16, height=8)
    img = cv2.imread(str(out / "basecolor.png"), cv2.IMREAD_UNCHANGED)
    assert img.shape == (8, 16, 4)


def test_resolve_alpha_over_operator():
    # Two opaque layers: top fully covers, so final coverage is 1 and the top
    # layer takes all the weight.
    m = LayeredMaterial()
    m.add_layer(Layer(alpha=Constant(1.0)))  # bottom
    m.add_layer(Layer(alpha=Constant(1.0)))  # top
    xs = np.array([0.5], dtype=np.float64)
    ys = np.array([0.5], dtype=np.float64)
    alpha_grids = [layer.alpha.eval_grid(xs, ys) for layer in m._layers]
    alpha, Vs = m._resolve_alpha(alpha_grids)
    assert alpha[0, 0] == pytest.approx(1.0)
    assert Vs[1][0, 0] == pytest.approx(1.0)  # top
    assert Vs[0][0, 0] == pytest.approx(0.0)  # fully occluded bottom


def test_resolve_alpha_half_transparent_top():
    m = LayeredMaterial()
    m.add_layer(Layer(alpha=Constant(1.0)))  # bottom opaque
    m.add_layer(Layer(alpha=Constant(0.5)))  # top half
    xs = np.array([0.5], dtype=np.float64)
    ys = np.array([0.5], dtype=np.float64)
    alpha_grids = [layer.alpha.eval_grid(xs, ys) for layer in m._layers]
    alpha, Vs = m._resolve_alpha(alpha_grids)
    assert alpha[0, 0] == pytest.approx(1.0)
    assert Vs[1][0, 0] == pytest.approx(0.5)  # top contributes half
    assert Vs[0][0, 0] == pytest.approx(0.5)  # bottom shows through other half
