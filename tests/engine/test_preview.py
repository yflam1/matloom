"""Tests for matloom.engine.preview — the no-Blender fixed-light renderer shared
by the text2dsl critic loop and polish. Pure numpy + engine (no torch, no
Blender). These pin the two correctness fixes from the channel-coverage audit:
the emissive HDR axis is not flattened, and coverage alpha composites over the
background instead of rendering black."""

import numpy as np
import pytest

from matloom.engine.expr import Constant
from matloom.engine.layer import Color, Emissive, Layer
from matloom.engine.main import LayeredMaterial
from matloom.engine.noise import fBm
from matloom.engine.preview import _BG, _composite, fixed_light_preview


def _emissive(strength: float) -> LayeredMaterial:
    """A single opaque layer with a fixed emissive color and a given strength."""
    e = Emissive.from_int(r=255, g=200, b=100)
    e.raw_strength = Constant(strength)
    m = LayeredMaterial()
    m.add_layer(
        Layer(
            alpha=Constant(1.0),
            basecolor=Color.from_int(r=20, g=20, b=20),
            emissive=e,
        )
    )
    return m


def _solid(alpha: float) -> LayeredMaterial:
    """A single layer with a flat color and a uniform coverage alpha."""
    m = LayeredMaterial()
    m.add_layer(
        Layer(
            alpha=Constant(alpha),
            basecolor=Color.from_int(r=200, g=50, b=50),
        )
    )
    return m


# --------------------------------------------------------------------------- #
# Emissive HDR axis (audit fix: the strength axis must not be flattened)
# --------------------------------------------------------------------------- #
def test_emissive_strength_axis_is_not_flattened():
    # A hard clip before tone-map would make strength=5 render equal to
    # strength=1 (both clipped to 1). The soft-knee tonemap preserves the
    # ordering so a brighter emitter renders brighter, giving the critic/CMA-ES
    # a gradient on the strength axis.
    dim = fixed_light_preview(_emissive(1.0), 32, 32)
    bright = fixed_light_preview(_emissive(5.0), 32, 32)
    assert bright.max() > dim.max()
    assert bright.mean() > dim.mean()


def test_emissive_strength_monotonic():
    # The soft-knee tonemap (x / (1 + x)) is monotonic, so preview brightness
    # increases with strength across the range.
    means = [
        fixed_light_preview(_emissive(s), 32, 32).mean() for s in (0.0, 0.5, 1.0, 5.0)
    ]
    assert all(a < b for b, a in zip(means[1:], means))


# --------------------------------------------------------------------------- #
# Coverage alpha (audit fix: gaps composite over the background, not black)
# --------------------------------------------------------------------------- #
def test_alpha_zero_renders_background_not_black():
    # Where final alpha=0 the material is absent; the preview must composite
    # over the mid-gray background, not render black holes.
    img = fixed_light_preview(_solid(0.0), 32, 32)
    bg = int(_BG * 255 + 0.5)
    assert tuple(img[16, 16].tolist()) == (bg, bg, bg)
    assert img.min() > 0  # no black texels


def test_alpha_full_coverage_is_the_material():
    # alpha=1 is the designed opaque-substrate class: the material is shown
    # unchanged (no background leak), so no regression vs. the pre-fix behavior.
    img = fixed_light_preview(_solid(1.0), 32, 32)
    assert tuple(img[16, 16].tolist()) != (int(_BG * 255 + 0.5),) * 3


def test_alpha_partial_blends_material_over_background():
    # 0 < alpha < 1 must blend the material color with the background by alpha,
    # not show the albedo at full opacity.
    full = fixed_light_preview(_solid(1.0), 32, 32)[16, 16].astype(float)
    half = fixed_light_preview(_solid(0.5), 32, 32)[16, 16].astype(float)
    bg = float(int(_BG * 255 + 0.5))
    # The half-coverage pixel sits between the full material and the background,
    # and is not equal to the full-coverage pixel.
    assert not np.allclose(half, full)
    assert all(
        min(full[i], bg) - 1 <= half[i] <= max(full[i], bg) + 1 for i in range(3)
    )


def test_preview_shape_dtype_and_finite():
    img = fixed_light_preview(_solid(1.0), 48, 32)
    assert img.shape == (32, 48, 3) and img.dtype == np.uint8
    assert np.isfinite(img.astype(float)).all()


# --------------------------------------------------------------------------- #
# Per-render eval_grid memo (Ref boundary): memo on/off must be bit-identical
# --------------------------------------------------------------------------- #

# No Define/Ref at all — the memo must be a complete no-op for plain programs.
_PLAIN_PROGRAM = """\
View(0, 0, 6, 6)
Material(
  Layer(1)
    .basecolor(120, 80, 60)
    .roughness((fBm(octaves=4, base_freq=12, to_01=True, seed=9) * 0.5))
    .height((Worley(seed=3, base_freq=20, to_01=True) * 0.02))
)
"""

# Corpus-style Define-heavy program: `tone` feeds four channels, `grain` one.
_DEFINE_PROGRAM = """\
View(0, 0, 6, 6)
Define(tone, fBm(octaves=4, base_freq=3, to_01=True, seed=7))
Define(grain, fBm(octaves=5, base_freq=25, to_01=True, seed=11))
Material(
  Layer(1)
    .basecolor(190, 185, 175)
    .roughness(0.95)
    .height((fBm(octaves=3, base_freq=12, to_01=True, seed=2) * 0.015)),
  Layer(1)
    .basecolor((130 + (tone * 50)), (45 + (tone * 25)), (35 + (tone * 18)))
    .roughness((0.7 + (grain * 0.2)))
    .height((0.035 + (tone * 0.04)))
)
"""

# Exercises the coords key path too: `spin` is a Define'd Rotate referenced
# whole (Ref hit at coords=None), while each Rotate(tone, 20) builds fresh
# coords grids around a shared Define (sound misses, never false hits).
_ROTATE_PROGRAM = """\
View(0, 0, 6, 6)
Define(tone, fBm(octaves=4, base_freq=3, to_01=True, seed=7))
Define(spin, Rotate(Worley(distance="manhattan", combination="F2", seed=4, base_freq=9), 30))
Material(
  Layer(1)
    .basecolor(140, 60, 40)
    .roughness((0.4 + (tone * 0.3)))
    .height((spin * 0.05)),
  Layer(1)
    .basecolor((30 + (tone * 60)), (30 + (tone * 20)), 30)
    .metallic(tone)
    .roughness(spin)
)
"""


def _assert_channels_equal(a, b):
    assert set(a) == set(b)
    for ch, grid in a.items():
        assert grid.dtype == b[ch].dtype, ch
        np.testing.assert_array_equal(grid, b[ch])


@pytest.mark.parametrize("size", [32, 64])
@pytest.mark.parametrize(
    "program",
    [_PLAIN_PROGRAM, _DEFINE_PROGRAM, _ROTATE_PROGRAM],
    ids=["plain", "define_heavy", "define_rotate"],
)
def test_composite_memo_on_off_bit_identical(program, size):
    # The memo dedupes Ref-shared subtrees, so memo-on grids are produced by
    # different code paths than memo-off; every channel (dtype included) must
    # still match exactly, and an interleaved memo-on repeat must reproduce
    # the first memo-on render (no cross-call state).
    material = LayeredMaterial.deserialize(program)
    memo_on = _composite(material, size, size)
    memo_off = _composite(material, size, size, memo=False)
    repeat_on = _composite(material, size, size)
    _assert_channels_equal(memo_on, memo_off)
    _assert_channels_equal(memo_on, repeat_on)


def test_fixed_light_preview_memo_kwarg_bit_identical():
    material = LayeredMaterial.deserialize(_DEFINE_PROGRAM)
    on = fixed_light_preview(material, 64, 64)
    off = fixed_light_preview(material, 64, 64, memo=False)
    assert on.tobytes() == off.tobytes()


def test_preview_renders_stable_across_calls():
    # The memo lives exactly one _composite call: successive renders (with a
    # different size interleaved) leak nothing between calls.
    material = LayeredMaterial.deserialize(_DEFINE_PROGRAM)
    first = fixed_light_preview(material, 64, 64).tobytes()
    fixed_light_preview(material, 32, 32)
    assert fixed_light_preview(material, 64, 64).tobytes() == first


def test_composite_memo_bit_identical_on_duplicated_subtree():
    # Code-built sharing WITHOUT Refs: the memo neither dedupes (Ref-only
    # hook) nor corrupts — the same fBm object feeding three channels renders
    # bit-identically with the memo on and off.
    shared = fBm(octaves=4, base_freq=6, to_01=True, seed=9)
    material = LayeredMaterial()
    material.add_layer(
        Layer(
            alpha=Constant(1.0),
            roughness=shared,
            metallic=shared,
            height=shared * 0.05,
        )
    )
    _assert_channels_equal(
        _composite(material, 64, 64), _composite(material, 64, 64, memo=False)
    )
