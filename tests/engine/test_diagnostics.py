"""Tests for matloom.engine.diagnostics — objective appearance descriptors.

The module no longer classifies a material as "bad" with thresholds; it computes
interpretable scalars from the unlit composited channel maps and lets a critic
decide. These tests assert scalar *values* and *orderings* (flat < textured,
isotropic < directional), not pass/fail codes. ``describe`` returns raw floats;
``describe_program`` wraps them in :class:`Descriptor` (label/formatted are
derived). No torch, no Blender; cv2 is pulled in transitively via
matloom.engine.main and is in requirements-test.txt.
"""

from __future__ import annotations

import numpy as np

from matloom.engine.diagnostics import (
    DESCRIPTOR_DEFINITIONS,
    Descriptor,
    describe,
    describe_program,
)
from matloom.engine.main import LayeredMaterial

_SMALL = {"width": 64, "height": 64}


def _desc(program: str) -> dict[str, float]:
    """Raw scalar dict (floats) for value comparison."""
    return describe(LayeredMaterial.deserialize(program), **_SMALL)


_FLAT = (
    "View(0, 0, 1, 1)\nMaterial(\n  Layer(1).basecolor(120, 120, 120).roughness(0.8)\n)"
)
_BRICK = (
    "View(0, 0, 6, 6)\n"
    "Define(brick, Bricks(brick_width=1, brick_height=0.45, mortar=0.06))\n"
    "Define(tone, fBm(octaves=4, base_freq=2.5, to_01=True, seed=7))\n"
    "Material(\n"
    "  Layer(1).basecolor(175, 170, 160).roughness(0.95),\n"
    "  Layer(brick).basecolor((150 + (tone * 50)), 55, 45).height((0.12 + (tone * 0.05)))\n"
    ")"
)
# Round isotropic pebbles on sand: low contrast, coarse, isotropic.
_SAND = (
    "View(0, 0, 3, 3)\n"
    "Define(sand, fBm(octaves=5, base_freq=20, to_01=True, seed=2))\n"
    "Define(pebble, Threshold(Worley(base_freq=2.5, to_01=True, seed=9),"
    " below_at=0.15, below_to=1, above_at=0.35, above_to=0))\n"
    "Material(\n"
    "  Layer(1).basecolor((200 + (sand * 30)), (175 + (sand * 25)), (140 + (sand * 20)))"
    ".roughness(0.85),\n"
    "  Layer(pebble).basecolor(120, 110, 95).height((pebble * 0.05))\n"
    ")"
)


# --------------------------------------------------------------------------- #
# Descriptor type + labels
# --------------------------------------------------------------------------- #
def test_descriptor_label_uppercases_acronyms_and_capitalizes_first():
    d = Descriptor(name="albedo_rms_contrast", value=0.3, definition="x")
    assert d.label == "Albedo RMS contrast"
    assert d.formatted == "0.300"
    assert Descriptor(name="ao_mean", value=1.0, definition="x").label == "AO mean"
    assert Descriptor(name="ior_mean", value=1.5, definition="x").label == "IOR mean"
    assert (
        Descriptor(name="albedo_luminance_mean", value=0.0, definition="x").label
        == "Albedo luminance mean"
    )
    # Formatting: 0 -> "0"; >=100 -> 1 decimal; 10-100 -> 2 decimals; <10 -> 3.
    assert Descriptor(name="x", value=0.0, definition="x").formatted == "0"
    assert Descriptor(name="x", value=132.4, definition="x").formatted == "132.4"
    assert Descriptor(name="x", value=42.5, definition="x").formatted == "42.50"


def test_describe_program_returns_descriptors_with_value_and_definition():
    d = describe_program(_BRICK, **_SMALL)
    first = d["albedo_luminance_mean"]
    assert isinstance(first, Descriptor)
    assert first.name == "albedo_luminance_mean"
    assert first.definition == DESCRIPTOR_DEFINITIONS["albedo_luminance_mean"]
    assert isinstance(first.value, float)


def test_describe_program_wraps_describe_values():
    p = _BRICK
    wrapped = {k: v.value for k, v in describe_program(p, **_SMALL).items()}
    assert wrapped == describe(LayeredMaterial.deserialize(p), **_SMALL)


def test_describe_program_preserves_definition_order():
    assert list(describe_program(_FLAT, **_SMALL).keys()) == list(
        DESCRIPTOR_DEFINITIONS
    )


def test_descriptors_keys_exactly_match_definitions():
    assert set(_desc(_FLAT)) == set(DESCRIPTOR_DEFINITIONS)


# --------------------------------------------------------------------------- #
# Scalar values + orderings (raw floats via describe)
# --------------------------------------------------------------------------- #
def test_flat_uniform_material_reads_flat():
    d = _desc(_FLAT)
    assert abs(d["albedo_luminance_mean"] - 120.0) < 1.0
    assert d["albedo_rms_contrast"] < 1e-6
    assert d["albedo_saturation"] < 1e-6
    assert d["albedo_edge_density"] < 1e-6
    assert d["albedo_color_span"] < 1e-6
    assert d["albedo_dominant_freq"] == 0  # flat: no pattern, not the argmax bin
    assert d["height_dominant_freq"] == 0
    assert d["height_range"] < 1e-6
    assert d["normal_mean_tilt_deg"] < 1e-6
    assert abs(d["ao_mean"] - 1.0) < 1e-6  # no relief -> no occlusion
    assert d["metallic_mean"] == 0.0
    assert d["transmission_mean"] == 0.0
    assert abs(d["ior_mean"] - 1.5) < 1e-6  # default IOR
    assert d["emissive_mean"] == 0.0


def test_brick_has_contrast_relief_and_direction():
    d = _desc(_BRICK)
    assert d["albedo_rms_contrast"] > 0.2  # mortar vs brick tone
    assert d["albedo_saturation"] > 0.3  # reddish bricks
    assert d["albedo_dominant_freq"] >= 3  # several bricks across the view
    assert d["albedo_anisotropy"] > 0.3  # running bond is directional
    assert d["albedo_color_span"] > 50
    assert d["height_range"] > 0.05  # brick relief
    assert d["height_anisotropy"] > 0.3  # ridges follow the brick lattice
    assert d["normal_mean_tilt_deg"] > 5.0
    assert d["ao_mean"] < 1.0  # some crevice darkening from relief


def test_sand_is_coarser_and_more_isotropic_than_brick():
    sand = _desc(_SAND)
    brick = _desc(_BRICK)
    # Round Worley pebbles are isotropic; a brick lattice is directional.
    assert sand["albedo_anisotropy"] < brick["albedo_anisotropy"]
    assert sand["height_anisotropy"] < brick["height_anisotropy"]
    # Fewer, coarser features across a 3-unit view than bricks across 6.
    assert sand["albedo_dominant_freq"] < brick["albedo_dominant_freq"]
    # Sand is low-contrast and unsaturated vs the mortar/brick split.
    assert sand["albedo_rms_contrast"] < brick["albedo_rms_contrast"]
    assert sand["albedo_saturation"] < brick["albedo_saturation"]


def test_transmissive_material_reports_transmission():
    p = (
        "View(0, 0, 1, 1)\nMaterial(\n"
        "  Layer(1).basecolor(220, 232, 238).roughness(0.4).transmission(0.92).ior(1.5)\n)"
    )
    d = _desc(p)
    assert d["transmission_mean"] > 0.8
    assert d["ior_mean"] > 1.0


def test_metallic_material_reports_metallic():
    p = (
        "View(0, 0, 1, 1)\nMaterial(\n"
        "  Layer(1).basecolor(180, 182, 188).metallic(1).roughness(0.3)\n)"
    )
    d = _desc(p)
    assert d["metallic_mean"] > 0.9
    assert d["metallic_coverage"] > 0.9


def test_emissive_material_reports_emissive():
    p = (
        "View(0, 0, 1, 1)\nMaterial(\n"
        "  Layer(1).basecolor(20, 20, 20).emissive(255, 200, 100, 2.0)\n)"
    )
    d = _desc(p)
    assert d["emissive_mean"] > 0.1


def test_descriptors_are_deterministic():
    assert describe_program(_BRICK, **_SMALL) == describe_program(_BRICK, **_SMALL)


def test_descriptors_robust_to_nan_expression():
    # Sqrt(-1) -> NaN, which survives the layer's np.clip (clip passes NaN
    # through). The descriptors must stay finite so the critic never sees "nan".
    p = (
        "View(0, 0, 1, 1)\nMaterial(\n"
        "  Layer(1).basecolor(Sqrt(-1), 0, 0).roughness(0.5)\n)"
    )
    d = _desc(p)
    assert all(np.isfinite(v) for v in d.values())
    assert d["albedo_luminance_mean"] == 0.0  # degenerate texel read as black
