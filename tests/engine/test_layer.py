"""Tests for matloom.engine.layer: Color, Emissive, and Layer data classes."""

from __future__ import annotations

import pytest

from matloom.engine.expr import Constant, X
from matloom.engine.layer import Color, Emissive, Layer
from matloom.engine.util import srgb_to_linear

# |-----------|
# |   Color   |
# |-----------|


def test_color_defaults_to_black():
    c = Color()
    assert c.raw_r() == 0.0
    assert c.raw_g() == 0.0
    assert c.raw_b() == 0.0


def test_color_channels_are_srgb_to_linear_of_clamped_raw():
    c = Color(r=Constant(0.5))
    # The exposed channel clamps to [0,1] then converts sRGB -> linear.
    assert c.r(0.0, 0.0) == pytest.approx(srgb_to_linear(0.5))


def test_color_channel_clamps_out_of_range():
    c = Color(r=Constant(2.0))
    assert c.r() == pytest.approx(srgb_to_linear(1.0))
    c2 = Color(r=Constant(-1.0))
    assert c2.r() == pytest.approx(srgb_to_linear(0.0))


def test_color_from_int():
    c = Color.from_int(r=255, g=128, b=0)
    assert c.raw_r() == pytest.approx(1.0)
    assert c.raw_g() == pytest.approx(128 / 255)
    assert c.raw_b() == pytest.approx(0.0)


def test_color_from_int_rejects_out_of_range():
    with pytest.raises(ValueError):
        Color.from_int(r=256)
    with pytest.raises(ValueError):
        Color.from_int(r=-1)


# |--------------|
# |   Emissive   |
# |--------------|


def test_emissive_strength_defaults_zero():
    e = Emissive()
    assert e.raw_strength() == 0.0
    assert e.strength() == 0.0


def test_emissive_strength_clamped_to_non_negative():
    e = Emissive(strength=Constant(-3.0))
    assert e.strength() == pytest.approx(0.0)


def test_emissive_strength_allows_large_values():
    e = Emissive(strength=Constant(100.0))
    assert e.strength() == pytest.approx(100.0)


def test_emissive_inherits_color_channels():
    e = Emissive(r=Constant(1.0))
    assert e.r() == pytest.approx(srgb_to_linear(1.0))


# |-----------|
# |   Layer   |
# |-----------|


def test_layer_defaults():
    layer = Layer()
    assert layer.alpha() == pytest.approx(1.0)
    assert layer.metallic() == pytest.approx(0.0)
    assert layer.roughness() == pytest.approx(0.0)
    assert layer.height() == pytest.approx(0.0)
    # OpenPBR-aligned channels default to inert values (weights 0, ior 1.5).
    assert layer.sheen() == pytest.approx(0.0)
    assert layer.coat() == pytest.approx(0.0)
    assert layer.transmission() == pytest.approx(0.0)
    assert layer.ior() == pytest.approx(1.5)
    assert layer.subsurface() == pytest.approx(0.0)
    assert layer.anisotropy() == pytest.approx(0.0)
    assert isinstance(layer.basecolor, Color)
    assert isinstance(layer.emissive, Emissive)


def test_layer_openpbr_weights_clamped():
    # sheen/coat/transmission/subsurface/anisotropy are [0, 1] like metallic.
    layer = Layer(
        sheen=Constant(2.0),
        coat=Constant(-1.0),
        transmission=Constant(5.0),
        subsurface=Constant(-0.5),
        anisotropy=Constant(3.0),
    )
    assert layer.sheen() == pytest.approx(1.0)
    assert layer.coat() == pytest.approx(0.0)
    assert layer.transmission() == pytest.approx(1.0)
    assert layer.subsurface() == pytest.approx(0.0)
    assert layer.anisotropy() == pytest.approx(1.0)


def test_layer_ior_clamped_to_at_least_one():
    # IOR is a refractive index >= 1 (not a [0, 1] weight); values below 1 snap
    # to 1, larger values pass through unchanged.
    assert Layer(ior=Constant(0.2)).ior() == pytest.approx(1.0)
    assert Layer(ior=Constant(2.4)).ior() == pytest.approx(2.4)


def test_layer_height_is_unclamped():
    # Height may be any real value, including negative.
    assert Layer(height=Constant(9.0)).height() == pytest.approx(9.0)
    assert Layer(height=Constant(-3.0)).height() == pytest.approx(-3.0)


def test_layer_alpha_clamped():
    layer = Layer(alpha=Constant(2.0))
    assert layer.alpha() == pytest.approx(1.0)
    layer2 = Layer(alpha=Constant(-1.0))
    assert layer2.alpha() == pytest.approx(0.0)


def test_layer_metallic_roughness_clamped():
    layer = Layer(metallic=Constant(5.0), roughness=Constant(-2.0))
    assert layer.metallic() == pytest.approx(1.0)
    assert layer.roughness() == pytest.approx(0.0)


def test_layer_accepts_coordinate_expression_for_alpha():
    layer = Layer(alpha=X())
    assert layer.alpha(0.3) == pytest.approx(0.3)
    assert layer.alpha(1.5) == pytest.approx(1.0)  # clamped


def test_layer_repr_uses_aliases():
    # _CustomBaseModel renames fields back to their public names in repr.
    text = repr(Layer())
    assert "alpha" in text
    assert "raw_alpha" not in text
