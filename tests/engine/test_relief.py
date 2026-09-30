"""Tests for matloom.engine.relief: height compositing and derived normal/AO."""

from __future__ import annotations

import numpy as np
import pytest

from matloom.engine.relief import composite_height, height_to_ao, height_to_normal

# |--------------------|
# |  composite_height  |
# |--------------------|


def test_composite_height_takes_max_over_present_layers():
    # Two cells. Cell 0: both layers present (α>0) → max(1, 9) = 9.
    # Cell 1: top layer absent (α=0), so it is excluded → only the 1 remains.
    alpha = [np.array([[1.0, 1.0]]), np.array([[1.0, 0.0]])]
    height = [np.array([[1.0, 1.0]]), np.array([[9.0, 9.0]])]
    h = composite_height(alpha, height)
    np.testing.assert_array_equal(h, [[9.0, 1.0]])


def test_composite_height_alpha_is_coverage_not_scale():
    # Alpha is a coverage mask, not a multiplier: a partially-transparent layer
    # (α=0.5) contributes its FULL height where present, not a scaled-down value
    # (the old `α·h` would have given 5).
    h = composite_height([np.array([[0.5]])], [np.array([[10.0]])])
    assert h[0, 0] == 10.0


def test_composite_height_absent_layer_contributes_nothing():
    # A present layer's height wins over a taller-but-absent layer's: the top
    # layer (α=0) is excluded even though its height (100) exceeds the bottom's.
    alpha = [np.array([[1.0]]), np.array([[0.0]])]
    height = [np.array([[1.0]]), np.array([[100.0]])]
    h = composite_height(alpha, height)
    assert h[0, 0] == 1.0


def test_composite_height_preserves_negative_height():
    # Height may be any real value (incl. negative — carving below the
    # substrate); a present layer is never floored to 0.
    h = composite_height([np.array([[1.0]])], [np.array([[-5.0]])])
    assert h[0, 0] == -5.0


def test_composite_height_substrate_is_zero_where_no_layer_present():
    # Where no layer contributes (α = 0 everywhere), the surface is the
    # substrate (0), not −inf.
    h = composite_height([np.array([[0.0]])], [np.array([[-5.0]])])
    assert h[0, 0] == 0.0


def test_composite_height_zero_contribution_is_positive_zero():
    # An absent texel (α=0) falls to the substrate; the output must be +0.0, never
    # −0.0, so it is bit-identical to the substrate (and to the TypeScript port,
    # whose strict ``Object.is`` / ``=== 0`` test normalizes −0 → +0).
    h = composite_height([np.array([[0.0]])], [np.array([[-5.0]])])
    assert not np.signbit(h[0, 0])


def test_composite_height_drops_non_finite_contribution():
    # A height expression can hit ``Log(0)`` (−inf) or ``Sqrt(-1)`` (NaN). Such a
    # texel must be treated as absent for that layer, never poisoning the
    # composite, so the finite layer below it still wins.
    alpha = [np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]])]
    height = [np.array([[np.inf, np.nan]]), np.array([[2.0, 3.0]])]
    h = composite_height(alpha, height)
    np.testing.assert_array_equal(h, [[2.0, 3.0]])


def test_composite_height_all_non_finite_falls_to_substrate():
    # If the only contribution at a texel is non-finite, it falls to the
    # substrate (0) rather than emitting ±inf / NaN into the height map.
    h = composite_height([np.array([[1.0]])], [np.array([[-np.inf]])])
    assert h[0, 0] == 0.0
    h = composite_height([np.array([[1.0]])], [np.array([[np.nan]])])
    assert h[0, 0] == 0.0


def test_composite_height_accepts_integer_grids():
    # Integer grids arise from bare literals (e.g. ``Layer(1)`` → int alpha,
    # ``.height(9)`` → int height). Correct math must never raise a cast error;
    # the result is the float max field.
    alpha = [
        np.array([[1, 1]], dtype=np.int64),
        np.array([[1, 0]], dtype=np.int64),
    ]
    height = [
        np.array([[1, 1]], dtype=np.int64),
        np.array([[9, 9]], dtype=np.int64),
    ]
    h = composite_height(alpha, height)
    assert np.issubdtype(h.dtype, np.floating)
    np.testing.assert_array_equal(h, [[9.0, 1.0]])


# |------------------|
# |  height_to_normal |
# |------------------|


def test_flat_height_gives_up_normal():
    h = np.full((2, 3), 2.0)
    nrm = height_to_normal(h, 1.0, 1.0, y_up=True)
    np.testing.assert_allclose(nrm[..., 0], 0.0, atol=1e-9)  # nx
    np.testing.assert_allclose(nrm[..., 1], 0.0, atol=1e-9)  # ny
    np.testing.assert_allclose(nrm[..., 2], 1.0, atol=1e-9)  # nz


def test_ramp_in_x_tilts_normal_toward_minus_x():
    # height increases left→right; ∂H/∂X > 0 → nx = -∂H/∂X < 0.
    h = np.array([[0.0, 1.0, 2.0, 3.0]])
    nrm = height_to_normal(h, 1.0, 1.0, y_up=True)
    assert nrm[0, 1, 0] < 0.0  # nx
    assert nrm[0, 1, 2] > 0.0  # nz


def test_smaller_spacing_steepens_normal():
    # Lateral scale now comes purely from the per-texel spacing (dx): the same
    # height ramp over a finer spacing is a steeper surface, so |nx| grows.
    h = np.array([[0.0, 1.0, 2.0, 3.0]])
    coarse = height_to_normal(h, 1.0, 1.0, y_up=True)
    fine = height_to_normal(h, 0.5, 0.5, y_up=True)
    assert abs(fine[0, 1, 0]) > abs(coarse[0, 1, 0])  # nx


def test_y_up_flips_sign_of_y_derivative():
    h = np.array([[0.0, 0.0], [1.0, 1.0]])  # bottom row higher
    up = height_to_normal(h, 1.0, 1.0, y_up=True)
    down = height_to_normal(h, 1.0, 1.0, y_up=False)
    assert up[0, 0, 1] == pytest.approx(-down[0, 0, 1], abs=1e-9)


def test_normal_is_unit_length():
    h = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    nrm = height_to_normal(h, 0.5, 0.5, y_up=True)
    lengths = np.sqrt((nrm**2).sum(axis=-1))
    np.testing.assert_allclose(lengths, 1.0, atol=1e-6)


# |---------------|
# |  height_to_ao  |
# |---------------|


def test_flat_height_is_fully_lit():
    h = np.full((5, 5), 3.0)
    ao = height_to_ao(h, 1.0, radius=2, strength=1.0)
    np.testing.assert_allclose(ao, 1.0, atol=1e-9)


def test_pit_is_darkened_peak_is_not():
    g = np.ones((5, 5))
    g[2, 2] = -4.0  # deep pit at the center
    ao = height_to_ao(g, 1.0, radius=1, strength=1.0)
    assert ao[2, 2] < 1.0  # cavity darkened

    g2 = np.ones((5, 5))
    g2[2, 2] = 5.0  # tall peak
    ao2 = height_to_ao(g2, 1.0, radius=1, strength=1.0)
    assert ao2[2, 2] == 1.0  # peaks are not occluded


def test_radius_y_defaults_to_radius():
    # Omitting radius_y is the isotropic case: identical to passing radius twice.
    g = np.ones((7, 7))
    g[3, 3] = -3.0
    iso = height_to_ao(g, 1.0, radius=2, strength=1.0)
    explicit = height_to_ao(g, 1.0, radius=2, strength=1.0, radius_y=2)
    np.testing.assert_array_equal(iso, explicit)


def test_anisotropic_radius_blurs_axes_independently():
    # A full low row is constant along X, so an X-only blur (radius_y=0) leaves
    # it unchanged → nothing darkens; a Y-only blur (radius_x=0) smears it across
    # rows → the row reads as a cavity. This isolates each axis cleanly.
    g = np.ones((7, 7))
    g[3, :] = -3.0  # an entire low row

    ao_x = height_to_ao(g, 1.0, radius=2, strength=1.0, radius_y=0)
    np.testing.assert_array_equal(ao_x, 1.0)  # X-blur is identity on a constant row

    ao_y = height_to_ao(g, 1.0, radius=0, strength=1.0, radius_y=2)
    assert ao_y[3, 3] < 1.0  # the low row is darkened by the Y-blur
    assert ao_y[0, 3] == 1.0  # far rows stay lit


def test_world_fixed_radius_keeps_ao_invariant_under_crop():
    # Crop invariance: a step edge sampled at two resolutions. "Cropping" halves
    # the world texel size (dx) and the rendered span, so the zoom is 2× and the
    # AO blur radius doubles to hold the world radius (radius·dx) constant. With
    # the radius fixed in *world* units the darkening converges (a small residual
    # from the discrete window's tap asymmetry remains); with the old fixed-texel
    # radius it instead ran away to fully black — the drift the user reported.
    coarse = np.ones((1, 32))
    coarse[0, 16:] = -4.0  # a step down halfway across
    fine = np.ones((1, 64))  # same world step, sampled twice as densely
    fine[0, 32:] = -4.0

    dx_coarse, dx_fine = 1.0, 0.5  # halving dx is a 2× crop
    base_radius = 4
    ao_coarse = height_to_ao(coarse, dx_coarse, radius=base_radius, strength=1.0)
    # World-fixed: radius scales with the zoom (2×) so radius·dx is unchanged.
    ao_fine = height_to_ao(fine, dx_fine, radius=base_radius * 2, strength=1.0)
    # Old fixed-texel behavior: same radius at the finer dx.
    ao_fine_fixed = height_to_ao(fine, dx_fine, radius=base_radius, strength=1.0)

    # World-fixed stays close to the coarse darkening; fixed-texel runs to black.
    assert abs(ao_fine.min() - ao_coarse.min()) < 0.05
    assert ao_fine_fixed.min() == 0.0
    assert ao_coarse.min() > 0.3


def test_integer_radius_is_bit_exact_to_direct_path():
    # Fractional support must be a pure superset: an integer radius must produce
    # exactly the same bytes as before the fractional lerp existed (the lerp
    # short-circuits at frac == 0). Two integer radii vs. their float spellings.
    g = np.ones((7, 7))
    g[3, 3] = -3.0
    int_r = height_to_ao(g, 1.0, radius=2, strength=1.0)
    float_r = height_to_ao(g, 1.0, radius=2.0, strength=1.0)
    np.testing.assert_array_equal(int_r, float_r)


def test_fractional_box_blur_axis_is_linear_blend_of_enclosing_integers():
    # The fractional radius is, per axis, the linear blend of the two enclosing
    # integer-radius blurs (the separable 2D blur composes two of these). Verify
    # the 1D axis primitive directly — the property the AO darkening inherits
    # its continuity from.
    from matloom.engine.relief import _box_blur_axis_frac, _box_blur_axis_int

    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 8)).astype(np.float64)
    r_lo, frac = 2, 0.5
    out = _box_blur_axis_frac(a, r_lo + frac, axis=1)
    blend = 0.5 * _box_blur_axis_int(a, r_lo, axis=1) + 0.5 * _box_blur_axis_int(
        a, r_lo + 1, axis=1
    )
    np.testing.assert_allclose(out, blend, atol=1e-12)


def test_fractional_radius_ao_is_continuous_in_radius():
    # A 0.5-texel radius step changes the AO by less than the full integer step
    # would (no `floor` step): the fractional result sits between the two
    # enclosing integers and is smooth in the radius. Use a shallow pit so the
    # cavity darkening stays clear of the [0,1] saturation that would flatten
    # all radii to the same value.
    g = np.ones((9, 9))
    g[4, 4] = -0.5
    ao2 = height_to_ao(g, 1.0, radius=2.0, strength=1.0)
    ao25 = height_to_ao(g, 1.0, radius=2.5, strength=1.0)
    ao3 = height_to_ao(g, 1.0, radius=3.0, strength=1.0)
    # 2.5 lies between 2 and 3 at the darkest texel.
    lo, hi = min(ao2[4, 4], ao3[4, 4]), max(ao2[4, 4], ao3[4, 4])
    assert lo - 1e-9 <= ao25[4, 4] <= hi + 1e-9
    # And the 0.5 step is smaller than the full integer step.
    step_half = abs(ao25[4, 4] - ao2[4, 4])
    step_full = abs(ao3[4, 4] - ao2[4, 4])
    assert step_half < step_full
