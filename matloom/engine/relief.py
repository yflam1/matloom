"""
relief.py — derive surface relief maps from a composited per-layer height field.

A layer's ``height`` channel is an absolute displacement in **coordinate units**
(the same units as the X/Y plane). The layers are combined with a *max* operator
(not the alpha-weighted blend the other channels use), and alpha acts as a pure
*coverage mask* — it gates *where* a layer contributes height but does not scale
the magnitude:

    H(x, y) = maxᵢ { hᵢ(x, y) : αᵢ(x, y) > 0 }

so an author types absolute heights (bricks 9, mortar 1) and the tallest *present*
layer wins at each texel, regardless of its opacity. Heights may be any real value,
including negative (carving *below* the substrate), so there is no substrate floor;
where no layer is present the surface is 0.

From that height field we derive, with no extra authoring:

- a **tangent-space normal map** (OpenGL / +Y-up convention — what Blender's
  Normal Map node ``TANGENT`` mode and Babylon expect), and
- an approximate **ambient-occlusion (AO)** map that darkens cavities.

Height shares the coordinate-unit space of the X/Y plane, so the caller passes the
per-texel lateral spacing directly (``dx`` / ``dy`` = region span / resolution);
heights and lateral distances already share a unit and the normals come out
dimensionless.

This module is the single source of the derivation math; ``frontend/src/engine/
relief.ts`` is a byte-for-byte port (verified by the cross-engine parity tests).
"""

import numpy as np

# AO tuning, shared with the TypeScript port. ``radius`` is the box-blur half
# width in texels; ``strength`` scales the darkening. Kept as module constants so
# both engines agree without threading extra parameters through the pipeline.
AO_RADIUS = 8
AO_STRENGTH = 1.0


def composite_height(
    alpha_grids: list[np.ndarray], height_grids: list[np.ndarray]
) -> np.ndarray:
    """
    Combine per-layer height fields with the max operator.

    Args:
        alpha_grids: Per-layer clamped alpha grids (bottom-most first), as
            already resolved during alpha compositing.
        height_grids: Per-layer height grids (coordinate units), same order
            and shape.

    Returns:
        The composited height field in coordinate units (any real value).
    """
    if not alpha_grids:
        raise ValueError("composite_height requires at least one layer")
    # −inf is the identity for ``max``: a texel stays at it until a layer
    # contributes ("no contribution yet"). Force a floating accumulator (with a
    # float32 floor) so integer height/alpha grids — e.g. a bare ``Layer(1)`` —
    # cannot raise a cast error from the in-place ``np.maximum`` below.
    dtype = np.result_type(alpha_grids[0].dtype, height_grids[0].dtype, np.float32)
    out = np.full(alpha_grids[0].shape, -np.inf, dtype=dtype)
    for alpha, height in zip(alpha_grids, height_grids):
        # Alpha is a *coverage mask* for height, not a multiplier: a layer
        # contributes its full ``height`` wherever it is present (``alpha > 0``)
        # and nothing where it is absent — opacity no longer scales the relief.
        # A height expression can still evaluate to a non-finite value (e.g.
        # ``Log(0)`` → −inf, ``Sqrt(-1)`` → NaN); treat such a texel as absent for
        # this layer so one poisoned layer cannot corrupt the composite (and, in
        # turn, the derived normal / AO). Mirrors the TypeScript port's per-cell
        # ``alpha > 0 && isFinite(height) && height > out`` test.
        present = (alpha > 0.0) & np.isfinite(height)
        np.maximum(out, np.where(present, height, -np.inf), out=out)
    # Texels no finite layer reached fall to the substrate (0). Folding the
    # mask with ``out == 0`` also rewrites −0.0 (e.g. ``0 · -5``) to +0.0 so a
    # zero-contribution texel is bit-identical to the substrate and to the port.
    out = np.where(np.isfinite(out), out, 0.0)
    out[out == 0.0] = 0.0
    return out


def height_to_normal(
    height: np.ndarray, dx: float, dy: float, y_up: bool = True
) -> np.ndarray:
    """
    Derive an OpenGL (+Y-up) tangent-space normal from a height field.

    Uses central differences with clamped edges. For a surface ``z = H(X, Y)``
    the unnormalized normal is ``(-∂H/∂X, -∂H/∂Y, 1)``. Row 0 of the grid is the
    top (max Y) when ``y_up``, so the row-direction derivative is negated to
    recover ``∂H/∂Y`` in world space.

    Args:
        height: ``(rows, cols)`` height field in coordinate units.
        dx: World spacing between adjacent columns, in coordinate units.
        dy: World spacing between adjacent rows, in coordinate units.
        y_up: Whether row 0 is max Y (True) or min Y (False).

    Returns:
        ``(rows, cols, 3)`` array of normal components ``(nx, ny, nz)`` in
        ``[-1, 1]``; a flat surface yields ``(0, 0, 1)``.
    """
    dHdx = _central_diff(height, axis=1, spacing=dx)
    dHdy_rows = _central_diff(height, axis=0, spacing=dy)
    # Convert the row-direction derivative to a world-Y derivative.
    dHdy = -dHdy_rows if y_up else dHdy_rows

    nx = -dHdx
    ny = -dHdy
    nz = np.ones_like(height)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack((nx * inv_len, ny * inv_len, nz * inv_len), axis=-1)


def height_to_ao(
    height: np.ndarray,
    dx: float,
    radius: float = AO_RADIUS,
    strength: float = AO_STRENGTH,
    radius_y: float | None = None,
) -> np.ndarray:
    """
    Approximate ambient occlusion from a height field (cavity darkening).

    A texel sitting below the local mean of its neighborhood reads as a cavity
    and is darkened; slopes and peaks are left bright. The depth below the mean
    is normalized by the world radius of the blur (``radius·dx``) so the result
    is scale-consistent.

    The blur radius is given separately per axis (``radius`` columns,
    ``radius_y`` rows) so a caller can hold the *world* size of the neighborhood
    fixed while the texel size changes — e.g. under a crop, where the rendered
    region shrinks but the resolution does not. Keeping the world radius constant
    makes AO crop-invariant: the cavity darkening depends only on the surface
    geometry, not on how finely it happens to be sampled. ``radius_y`` defaults
    to ``radius``, so an isotropic caller is unchanged.

    The radius may be fractional: a non-integer radius is the linear blend of the
    two enclosing integer-radius blurs (separable: lerp each axis independently).
    At an integer radius the blend collapses to the integer blur exactly, so
    integer-radius behavior is bit-identical to the direct path — fractional
    support is a pure superset, never a behavior change for existing callers.
    This keeps the cavity darkening continuous in the crop zoom (no 1-texel
    steps from a ``floor`` rounding).

    Args:
        height: ``(rows, cols)`` height field in coordinate units.
        dx: World spacing between adjacent columns, in coordinate units.
        radius: Box-blur half width in texels along the column (X) axis; may be
            fractional.
        strength: Multiplier on the darkening.
        radius_y: Box-blur half width in texels along the row (Y) axis; defaults
            to ``radius``; may be fractional.

    Returns:
        ``(rows, cols)`` AO field in ``[0, 1]`` (1 = fully lit).
    """
    if radius_y is None:
        radius_y = radius
    mean = _box_blur(height, radius, radius_y)
    norm = max(radius, 1) * dx
    cavity = np.clip((mean - height) / norm, 0.0, 1.0)
    return np.clip(1.0 - strength * cavity, 0.0, 1.0)


def _central_diff(a: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    """
    Clamped central difference along ``axis``: interior cells use
    ``(a[k+1] - a[k-1]) / (2·spacing)``; the two edges use a one-sided
    ``(a[1] - a[0]) / spacing`` / ``(a[-1] - a[-2]) / spacing``. A length-1 axis
    yields zeros. Matches the per-cell loop in the TypeScript port exactly.
    """
    a = np.moveaxis(a, axis, -1)
    out = np.zeros_like(a)
    n = a.shape[-1]
    if n >= 2:
        out[..., 1:-1] = (a[..., 2:] - a[..., :-2]) / (2.0 * spacing)
        out[..., 0] = (a[..., 1] - a[..., 0]) / spacing
        out[..., -1] = (a[..., -1] - a[..., -2]) / spacing
    return np.moveaxis(out, -1, axis)


def _box_blur(
    a: np.ndarray, radius: float, radius_y: float | None = None
) -> np.ndarray:
    """
    Separable, edge-clamped box blur with a ``2·radius + 1`` (columns) by
    ``2·radius_y + 1`` (rows) window. Taps are summed in ascending order
    (``k = -radius … +radius``) to match the TS port's summation, keeping
    cross-engine divergence at the float32-rounding level. ``radius_y`` defaults
    to ``radius``.

    The radius may be fractional: a non-integer radius is the linear blend of the
    two enclosing integer-radius blurs (separable: lerp each axis independently).
    At an integer radius the blend collapses to the integer blur exactly
    (``1.0·x + 0.0·y = x`` in IEEE 754), so integer-radius behavior is
    bit-identical to the direct path.
    """
    if radius_y is None:
        radius_y = radius
    if radius <= 0 and radius_y <= 0:
        return a.astype(a.dtype, copy=True)
    out = _box_blur_axis_frac(a, radius, axis=1)
    out = _box_blur_axis_frac(out, radius_y, axis=0)
    return out


def _box_blur_axis_frac(a: np.ndarray, radius: float, axis: int) -> np.ndarray:
    """Fractional-radius box blur along one axis: lerp the floor/ceil integer
    blurs by the fractional part. Integer radius → the integer blur exactly."""
    r_lo = int(np.floor(radius))
    frac = float(radius - r_lo)
    if frac == 0.0:
        return _box_blur_axis_int(a, r_lo, axis=axis)
    lo = _box_blur_axis_int(a, r_lo, axis=axis)
    hi = _box_blur_axis_int(a, r_lo + 1, axis=axis)
    # np.float32 scalars keep the multiply in float32 to mirror the TS port's
    # Float32Array arithmetic (the relief parity tolerance is single-precision).
    w_lo = np.float32(1.0 - frac)
    w_hi = np.float32(frac)
    return w_lo * lo + w_hi * hi


def _box_blur_axis_int(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        return a.astype(a.dtype, copy=True)
    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    padded = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(radius, radius)], mode="edge")
    acc = np.zeros_like(a)
    for k in range(-radius, radius + 1):
        start = radius + k
        acc = acc + padded[..., start : start + n]
    acc = acc / (2 * radius + 1)
    return np.moveaxis(acc, -1, axis)
