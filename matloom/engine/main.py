import argparse
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from pydantic import Field, validate_call

from matloom.engine.expr import Constant, DivisionExpression2D, Expression2D
from matloom.engine.layer import Color, Emissive, Layer
from matloom.engine.parser import parse_expr
from matloom.engine.relief import (
    composite_height,
    height_to_ao,
    height_to_normal,
)
from matloom.engine.serialize import serialize_expr
from matloom.engine.util import linear_to_srgb
from matloom.utils.anybase import AnyBase
from matloom.utils.file import get_available_path

# Default viewing window (x1, y1, x2, y2). Height shares the coordinate-unit
# space of the X/Y plane, so there is no separate physical scale.
DEFAULT_VIEW = (0.0, 0.0, 1.0, 1.0)

# [0, 1] scalar channels exported as uint8 PNGs via the alpha-weighted over
# blend (in chain/export order). `ior` (>= 1) and `emissive`/`height` (float)
# are handled separately.
_SCALAR_CHANNELS = (
    "metallic",
    "roughness",
    "sheen",
    "coat",
    "transmission",
    "subsurface",
    "anisotropy",
)


class LayeredMaterial(AnyBase):
    @validate_call
    def __init__(self, dtype: type[np.floating] = np.float32) -> None:
        self._dtype = dtype
        self._layers: list[Layer] = []
        # Ordered (name, expression) definitions emitted as a `Define(...)`
        # preamble. Later definitions may reference earlier ones.
        self._defs: list[tuple[str, Expression2D]] = []
        # Viewing window: (x1, y1, x2, y2). Part of the material (affects the
        # sampled look), emitted as a `View(...)` preamble. ``_view_set`` tracks
        # whether the view was explicitly authored (via :meth:`set_view` or a
        # deserialized `View(...)` line); an unauthored default view is omitted
        # on serialize, mirroring the channel set-tracking in ``_serialize_layer``.
        self._view: tuple[float, float, float, float] = DEFAULT_VIEW
        self._view_set: bool = False

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @validate_call
    def set_view(
        self,
        x1: float = 0.0,
        y1: float = 0.0,
        x2: float = 1.0,
        y2: float = 1.0,
    ) -> None:
        """Set the viewing window (region). Marking the view as explicitly
        authored so :meth:`serialize` emits a ``View(...)`` line (an unauthored
        default view is omitted)."""
        self._view = (x1, y1, x2, y2)
        self._view_set = True

    @validate_call
    def add_layer(self, layer: Layer) -> None:
        self._layers.append(layer)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def add_def(self, name: str, expr: Expression2D) -> None:
        self._defs.append((name, expr))

    def serialize(self) -> str:
        """
        Serialize the layer stack to the textual material format (pretty-printed
        with a 2-space indent), preceded by a `View(...)` line (only when the
        view was explicitly set via :meth:`set_view` or a deserialized
        `View(...)` preamble; an unauthored default view is omitted) and any
        `Define(name, expr)` lines:

            View(<x1>, <y1>, <x2>, <y2>)
            Define(<name>, <expr>)
            Material(
              Layer(<alpha>)
                .basecolor(<r>, <g>, <b>)
                .metallic(<m>)
                .roughness(<rough>)
                .emissive(<r>, <g>, <b>, <strength>)
                .height(<height>),
              …
            )

        The first `Layer` is the bottom-most (`self._layers[0]`). Basecolor
        and emissive RGB are emitted in the 0–255 convention; alpha, metallic,
        roughness and emissive strength are in the native [0, 1] / [0, ∞)
        domain, and height is in coordinate units. Only channels that were
        explicitly set are emitted (an untouched channel is omitted, not
        re-emitted at its default; an untouched alpha yields `Layer()`), so a
        round-trip preserves the author's exact channel set. Likewise,
        `View(...)` is emitted only when it was explicitly set (an unauthored
        default view is omitted), so a round-trip preserves the author's exact
        preamble. The string round-trips through :meth:`deserialize` and is
        byte-for-byte interchangeable with the frontend viewer's import/export.
        """
        if not self._layers:
            material = "Material()"
        else:
            layers = ",\n".join(
                _indent_lines(_serialize_layer(layer), "  ") for layer in self._layers
            )
            material = f"Material(\n{layers}\n)"
        lines: list[str] = []
        if self._view_set:
            lines.append(f"View({', '.join(_fmt_num(v) for v in self._view)})")
        if self._defs:
            lines.append(
                "\n".join(
                    f"Define({name}, {serialize_expr(expr)})"
                    for name, expr in self._defs
                )
            )
        lines.append(material)
        return "\n".join(lines)

    @classmethod
    def deserialize(cls, text: str) -> "LayeredMaterial":
        """Reconstruct a :class:`LayeredMaterial` from a material string."""
        material = cls()
        view, rest = _split_view(text)
        if view is not None:
            material.set_view(*view)
        defs, material_src = _split_defs(rest)
        env: dict[str, Expression2D] = {}
        for name, expr_src in defs:
            expr = parse_expr(expr_src, env)
            material.add_def(name, expr)
            env[name] = expr
        for spec in _parse_material(material_src):
            material.add_layer(_build_layer(spec, env))
        return material

    @validate_call
    def export(
        self,
        output_dir: Path | None = None,
        *,
        x1: float | None = None,
        y1: float | None = None,
        x2: float | None = None,
        y2: float | None = None,
        width: Annotated[int, Field(ge=1)] = 512,
        height: Annotated[int, Field(ge=1)] = 512,
        y_up: bool = True,
    ) -> Path:
        if self.n_layers == 0:
            raise ValueError("No layers to export")

        # Fall back to the material's own view (set via `View(...)`) for any
        # region argument the caller leaves unspecified.
        vx1, vy1, vx2, vy2 = self._view
        x1 = vx1 if x1 is None else x1
        y1 = vy1 if y1 is None else y1
        x2 = vx2 if x2 is None else x2
        y2 = vy2 if y2 is None else y2

        # Ensure output directory exists
        output_dir = self._maybe_create_output_dir(output_dir)

        xs = self._generate_axis_coordinates(width, x1, x2)
        ys = self._generate_axis_coordinates(height, y1, y2, reverse=y_up)
        alpha_grids = [
            layer.alpha.eval_grid(xs, ys).astype(self._dtype) for layer in self._layers
        ]
        alpha, Vs = self._resolve_alpha(alpha_grids)
        self._export_basecolor(xs, ys, alpha, Vs, output_dir)
        for name in _SCALAR_CHANNELS:
            self._export_scalar(xs, ys, alpha, Vs, output_dir, name)
        self._export_ior(xs, ys, alpha, Vs, output_dir)
        self._export_emissive(xs, ys, alpha, Vs, output_dir)

        # Relief (height → normal + AO). Height shares the coordinate-unit space
        # of the X/Y plane, so the per-texel lateral spacing is just the region
        # span divided by the resolution — height and lateral distance already
        # share a unit and the derived slopes are dimensionless.
        dx = abs(x2 - x1) / width
        dy = abs(y2 - y1) / height
        self._export_relief(xs, ys, alpha_grids, dx, dy, y_up, output_dir)

        # Return output_dir as it may have been auto-generated or modified
        return output_dir

    def _maybe_create_output_dir(self, output_dir: Path | None) -> Path:
        if output_dir is None:
            now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            output_dir = Path("outputs") / self.cname_ / now
            output_dir = get_available_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @validate_call
    def _generate_axis_coordinates(
        self,
        length: Annotated[int, Field(ge=1)],
        v1: float,
        v2: float,
        reverse: bool = False,
    ) -> np.ndarray:
        """
        Generates a 1D array of pixel-centered coordinates between v1 and v2.

        Args:
            length (int): The number of pixels / grid points along this axis
                (e.g., width or height).
            v1 (float): The starting coordinate value (e.g., x_min or y_min).
            v2 (float): The ending coordinate value (e.g., x_max or y_max).
            reverse (bool, optional): If True, reverses the array sequence.
                Useful for Y-up coordinate systems. Defaults to False.

        Returns:
            np.ndarray: A 1D array of centered coordinates.
        """
        # 1. Reverse by swapping endpoints so the descending sequence is
        #    written directly into the contiguous, pre-allocated buffer (zero
        #    copy, no negative-stride view).
        if reverse:
            v1, v2 = v2, v1

        # 2. Allocate the array once
        coords = np.arange(length, dtype=self._dtype)

        # 3. Perform all operations in-place
        coords += 0.5
        coords *= (v2 - v1) / length
        coords += v1

        return coords

    def _resolve_alpha(
        self, alpha_grids: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Over-operator coverage and per-layer visibility weights from the
        per-layer alpha grids (bottom-most first)."""
        ys_size, xs_size = alpha_grids[0].shape
        Vs = np.empty((self.n_layers, ys_size, xs_size), dtype=self._dtype)
        running = np.ones((ys_size, xs_size), dtype=self._dtype)
        for i in range(self.n_layers - 1, -1, -1):
            Vs[i] = alpha_grids[i] * running
            running *= 1.0 - alpha_grids[i]
        final_alpha = np.subtract(1.0, running)
        return final_alpha, Vs

    def _weighted_blend(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        alpha: np.ndarray,
        Vs: np.ndarray,
        samplers: list[Expression2D],
    ) -> np.ndarray:
        blended = np.zeros((ys.size, xs.size), dtype=self._dtype)
        for i, s in enumerate(samplers):
            blended += Vs[i] * s.eval_grid(xs, ys).astype(self._dtype)
        mask = alpha > 0.0
        np.divide(blended, alpha, out=blended, where=mask)
        blended[~mask] = 0.0
        return blended

    def _export_basecolor(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        alpha: np.ndarray,
        Vs: np.ndarray,
        output_dir: Path,
    ) -> None:
        out = np.empty((ys.size, xs.size, 4), dtype=self._dtype)
        for ch, attr in enumerate(("b", "g", "r")):
            samplers = [getattr(layer.basecolor, attr) for layer in self._layers]
            out[:, :, ch] = linear_to_srgb(
                self._weighted_blend(xs, ys, alpha, Vs, samplers)
            )
        out[:, :, 3] = alpha
        cv2.imwrite(str(output_dir / "basecolor.png"), self._to_uint8(out))

    def _export_scalar(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        alpha: np.ndarray,
        Vs: np.ndarray,
        output_dir: Path,
        name: str,
    ) -> None:
        """Alpha-weighted blend of a [0, 1] scalar channel (metallic, roughness,
        sheen, coat, transmission, subsurface, anisotropy), written as a uint8
        PNG. All such channels composite with the over-operator weights exactly
        like metallic/roughness."""
        out = self._weighted_blend(
            xs, ys, alpha, Vs, [getattr(layer, name) for layer in self._layers]
        )
        cv2.imwrite(str(output_dir / f"{name}.png"), self._to_uint8(out))

    def _export_ior(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        alpha: np.ndarray,
        Vs: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Index of refraction (>= 1, not a [0, 1] channel), written as a float
        map so the true value survives. .exr when OpenEXR is enabled, else .hdr
        (Radiance RGBE encodes the positive IOR range fine)."""
        out = self._weighted_blend(
            xs, ys, alpha, Vs, [layer.ior for layer in self._layers]
        )
        ext = "exr" if os.environ.get("OPENCV_IO_ENABLE_OPENEXR", "0") == "1" else "hdr"
        img = np.repeat(out[:, :, None], 3, axis=2).astype(np.float32)
        cv2.imwrite(str(output_dir / f"ior.{ext}"), img)

    def _export_emissive(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        alpha: np.ndarray,
        Vs: np.ndarray,
        output_dir: Path,
    ) -> None:
        out = np.empty((ys.size, xs.size, 3), dtype=self._dtype)
        for ch, attr in enumerate(("b", "g", "r")):
            samplers = [
                getattr(layer.emissive, attr) * layer.emissive.strength
                for layer in self._layers
            ]
            out[:, :, ch] = self._weighted_blend(xs, ys, alpha, Vs, samplers)

        # Use .exr if OpenEXR support is enabled, otherwise fall back to .hdr
        ext = "exr" if os.environ.get("OPENCV_IO_ENABLE_OPENEXR", "0") == "1" else "hdr"
        cv2.imwrite(str(output_dir / f"emissive.{ext}"), out)

    def _export_relief(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        alpha_grids: list[np.ndarray],
        dx: float,
        dy: float,
        y_up: bool,
        output_dir: Path,
    ) -> None:
        """Composite the per-layer height fields and write the height map plus the
        derived normal and ambient-occlusion maps."""
        height_grids = [
            layer.height.eval_grid(xs, ys).astype(self._dtype) for layer in self._layers
        ]
        h = composite_height(alpha_grids, height_grids)

        # Absolute height in coordinate units, stored as float to preserve the
        # true values.
        # .exr keeps full precision and negatives; .hdr is the fallback when
        # OpenEXR is off, but Radiance RGBE cannot encode negative values — they
        # clamp to 0 — so warn when below-substrate heights would be lost.
        ext = "exr" if os.environ.get("OPENCV_IO_ENABLE_OPENEXR", "0") == "1" else "hdr"
        if ext == "hdr" and h.min() < 0.0:
            warnings.warn(
                "height has negative (below-substrate) values, but the .hdr "
                "(Radiance RGBE) fallback clamps them to 0; set "
                "OPENCV_IO_ENABLE_OPENEXR=1 to write a float .exr that preserves "
                "them.",
                RuntimeWarning,
                stacklevel=2,
            )
        height_img = np.repeat(h[:, :, None], 3, axis=2).astype(np.float32)
        cv2.imwrite(str(output_dir / f"height.{ext}"), height_img)

        # Tangent-space normal (OpenGL / +Y-up). cv2 writes BGR, so reverse the
        # encoded RGB (nx, ny, nz) → (nz, ny, nx).
        nrm = height_to_normal(h, dx, dy, y_up)
        enc = np.ascontiguousarray((nrm * 0.5 + 0.5)[..., ::-1])
        cv2.imwrite(str(output_dir / "normal.png"), self._to_uint8(enc))

        # Approximate ambient occlusion, grayscale.
        ao = height_to_ao(h, dx)
        cv2.imwrite(str(output_dir / "ao.png"), self._to_uint8(ao))

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        np.clip(arr, 0.0, 1.0, out=arr)
        arr *= 255.0
        arr += 0.5
        return arr.astype(np.uint8, copy=False)


# |======================================================|
# |     Material <-> string (de)serialization helpers    |
# |======================================================|


def _color_to_format(raw: Expression2D) -> str:
    """
    Serialize a [0, 1] color channel into the 0–255 format convention. The
    importer stores a 0–255 value as `<expr> / 255`, so unwrap that exact
    shape back to `<expr>`; anything else is scaled up by 255.
    """
    if (
        isinstance(raw, DivisionExpression2D)
        and isinstance(raw._divisor, Constant)
        and raw._divisor._value == 255
    ):
        return serialize_expr(raw._dividend)
    return f"({serialize_expr(raw)} * 255)"


def _serialize_layer(layer: Layer) -> str:
    """Serialize one layer, emitting only the channels that were explicitly set
    at construction (tracked by pydantic's ``model_fields_set``). A channel the
    author never wrote is omitted rather than re-emitted at its default, so a
    round-trip preserves the author's exact channel set (an explicit
    ``.roughness(0)`` is kept; an untouched roughness is dropped). ``alpha`` is
    positional in ``Layer(...)``: an untouched alpha serializes as ``Layer()``.
    The TS port mirrors this in ``frontend/src/app/material-io.ts``."""
    set_fields = layer.model_fields_set
    bc, em = layer.basecolor, layer.emissive
    lines: list[str] = []
    if "raw_alpha" in set_fields:
        lines.append(f"Layer({serialize_expr(layer.raw_alpha)})")
    else:
        lines.append("Layer()")
    if "basecolor" in set_fields:
        lines.append(
            f"  .basecolor({_color_to_format(bc.raw_r)}, "
            f"{_color_to_format(bc.raw_g)}, {_color_to_format(bc.raw_b)})"
        )
    if "raw_metallic" in set_fields:
        lines.append(f"  .metallic({serialize_expr(layer.raw_metallic)})")
    if "raw_roughness" in set_fields:
        lines.append(f"  .roughness({serialize_expr(layer.raw_roughness)})")
    if "raw_sheen" in set_fields:
        lines.append(f"  .sheen({serialize_expr(layer.raw_sheen)})")
    if "raw_coat" in set_fields:
        lines.append(f"  .coat({serialize_expr(layer.raw_coat)})")
    if "raw_transmission" in set_fields:
        lines.append(f"  .transmission({serialize_expr(layer.raw_transmission)})")
    if "raw_ior" in set_fields:
        lines.append(f"  .ior({serialize_expr(layer.raw_ior)})")
    if "raw_subsurface" in set_fields:
        lines.append(f"  .subsurface({serialize_expr(layer.raw_subsurface)})")
    if "raw_anisotropy" in set_fields:
        lines.append(f"  .anisotropy({serialize_expr(layer.raw_anisotropy)})")
    if "emissive" in set_fields:
        lines.append(
            f"  .emissive({_color_to_format(em.raw_r)}, "
            f"{_color_to_format(em.raw_g)}, {_color_to_format(em.raw_b)}, "
            f"{serialize_expr(em.raw_strength)})"
        )
    if "raw_height" in set_fields:
        lines.append(f"  .height({serialize_expr(layer.raw_height)})")
    return "\n".join(lines)


def _indent_lines(text: str, pad: str) -> str:
    return "\n".join(pad + line for line in text.split("\n"))


def _build_layer(spec: dict[str, object], env: dict[str, Expression2D]) -> Layer:
    kwargs: dict[str, object] = {}
    if spec["alpha"] is not None:
        kwargs["alpha"] = parse_expr(spec["alpha"], env)  # type: ignore[arg-type]
    if spec["basecolor"] is not None:
        r, g, b = spec["basecolor"]  # type: ignore[misc]
        kwargs["basecolor"] = Color(
            r=parse_expr(r, env) / 255.0,
            g=parse_expr(g, env) / 255.0,
            b=parse_expr(b, env) / 255.0,
        )
    if spec["metallic"] is not None:
        kwargs["metallic"] = parse_expr(spec["metallic"], env)  # type: ignore[arg-type]
    if spec["roughness"] is not None:
        kwargs["roughness"] = parse_expr(spec["roughness"], env)  # type: ignore[arg-type]
    for _ch in (
        "sheen",
        "coat",
        "transmission",
        "ior",
        "subsurface",
        "anisotropy",
    ):
        if spec[_ch] is not None:
            kwargs[_ch] = parse_expr(spec[_ch], env)  # type: ignore[arg-type]
    if spec["emissive"] is not None:
        r, g, b, s = spec["emissive"]  # type: ignore[misc]
        kwargs["emissive"] = Emissive(
            r=parse_expr(r, env) / 255.0,
            g=parse_expr(g, env) / 255.0,
            b=parse_expr(b, env) / 255.0,
            strength=parse_expr(s, env),
        )
    if spec["height"] is not None:
        kwargs["height"] = parse_expr(spec["height"], env)  # type: ignore[arg-type]
    return Layer(**kwargs)


def _fmt_num(v: float) -> str:
    """Format a View number to match the frontend's `String(v)` (integers
    without a trailing ``.0``; other floats in shortest round-trip form)."""
    return str(int(v)) if float(v).is_integer() else repr(v)


def _split_view(
    text: str,
) -> tuple[tuple[float, float, float, float] | None, str]:
    """Split a leading ``View(x1, y1, x2, y2)`` statement (if any) from the rest.
    Returns the parsed 4-tuple (or None) and the remaining text."""
    src = text.strip()
    if not re.match(r"^View\s*\(", src):
        return None, src
    open_idx = src.index("(")
    close = _match_paren(src, open_idx)
    parts = _split_top_level(src[open_idx + 1 : close])
    if len(parts) != 4:
        raise ValueError("View(...) takes exactly 4 arguments (x1, y1, x2, y2)")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise ValueError(f"Invalid View argument in: {parts}")
    return vals, src[close + 1 :].strip()  # type: ignore[return-value]


def _split_defs(text: str) -> tuple[list[tuple[str, str]], str]:
    """
    Split leading ``Define(name, expr)`` statements from the ``Material(...)``
    body. Returns the ordered (name, expr-source) pairs and the remaining text.
    """
    src = text.strip()
    defs: list[tuple[str, str]] = []
    while re.match(r"^Define\s*\(", src):
        open_idx = src.index("(")
        close = _match_paren(src, open_idx)
        parts = _split_top_level(src[open_idx + 1 : close])
        if len(parts) != 2:
            raise ValueError("Define(name, expr) takes exactly two arguments")
        name = parts[0]
        if not name.isidentifier():
            raise ValueError(f"Invalid Define name: {name!r}")
        defs.append((name, parts[1]))
        src = src[close + 1 :].strip()
    return defs, src


def _parse_material(text: str) -> list[dict[str, object]]:
    src = text.strip()
    if not re.match(r"^Material\s*\(", src):
        raise ValueError("Expected the material to start with 'Material('")
    open_idx = src.index("(")
    close = _match_paren(src, open_idx)
    if src[close + 1 :].strip() != "":
        raise ValueError("Unexpected text after the closing ')'")
    return [_parse_layer_chunk(c) for c in _split_top_level(src[open_idx + 1 : close])]


def _parse_layer_chunk(chunk: str) -> dict[str, object]:
    text = chunk.strip()
    if not re.match(r"^Layer\s*\(", text):
        raise ValueError(f"Expected 'Layer(' in: {text}")
    result: dict[str, object] = {
        "alpha": None,
        "basecolor": None,
        "metallic": None,
        "roughness": None,
        "sheen": None,
        "coat": None,
        "transmission": None,
        "ior": None,
        "subsurface": None,
        "anisotropy": None,
        "emissive": None,
        "height": None,
    }

    pos = text.index("(")
    close = _match_paren(text, pos)
    layer_args = _split_top_level(text[pos + 1 : close])
    if len(layer_args) > 1:
        raise ValueError("Layer() takes at most one argument (alpha)")
    if len(layer_args) == 1:
        result["alpha"] = layer_args[0]
    pos = close + 1

    n = len(text)
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        if text[pos] != ".":
            raise ValueError(f"Expected '.' but found {text[pos]!r}")
        pos += 1
        name_start = pos
        while pos < n and (text[pos].isalpha() or text[pos] == "_"):
            pos += 1
        method = text[name_start:pos]
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n or text[pos] != "(":
            raise ValueError(f"Expected '(' after .{method}")
        close = _match_paren(text, pos)
        _apply_method(result, method, _split_top_level(text[pos + 1 : close]))
        pos = close + 1
    return result


def _apply_method(result: dict[str, object], method: str, args: list[str]) -> None:
    def expect(count: int) -> None:
        if len(args) != count:
            raise ValueError(f"{method}() expects {count} argument(s), got {len(args)}")

    if method == "alpha":
        expect(1)
        result["alpha"] = args[0]
    elif method == "basecolor":
        expect(3)
        result["basecolor"] = (args[0], args[1], args[2])
    elif method == "metallic":
        expect(1)
        result["metallic"] = args[0]
    elif method == "roughness":
        expect(1)
        result["roughness"] = args[0]
    elif method in (
        "sheen",
        "coat",
        "transmission",
        "ior",
        "subsurface",
        "anisotropy",
    ):
        expect(1)
        result[method] = args[0]
    elif method == "emissive":
        expect(4)
        result["emissive"] = (args[0], args[1], args[2], args[3])
    elif method == "height":
        expect(1)
        result["height"] = args[0]
    else:
        raise ValueError(f"Unknown layer property: {method}")


def _match_paren(text: str, open_idx: int) -> int:
    """Index of the ')' matching the '(' at `open_idx` (ignores strings)."""
    depth = 0
    quote = ""
    for i in range(open_idx, len(text)):
        c = text[i]
        if quote:
            if c == quote:
                quote = ""
            continue
        if c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unbalanced parentheses")


def _split_top_level(s: str) -> list[str]:
    """Split at top-level commas (outside parens and strings); trims fragments."""
    out: list[str] = []
    depth = 0
    quote = ""
    start = 0
    for i, c in enumerate(s):
        if quote:
            if c == quote:
                quote = ""
            continue
        if c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
    tail = s[start:].strip()
    if not out and tail == "":
        return []
    out.append(tail)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matloom-export",
        description=(
            "Export the texture maps of a layered-material DSL program. "
            "The --program string is the same format printed by "
            "`python -m matloom.engine.generate`."
        ),
    )
    p.add_argument(
        "--program",
        required=True,
        help="the material DSL program to export (e.g. the output of matloom-generate)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory to write the maps into (default: auto-generated under outputs/)",
    )
    p.add_argument("--width", type=int, default=512, help="output width in pixels")
    p.add_argument("--height", type=int, default=512, help="output height in pixels")
    p.add_argument(
        "--y-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample Y increasing upward (use --no-y-up to flip)",
    )
    return p


def _cli(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    material = LayeredMaterial.deserialize(args.program)
    out = material.export(
        args.output_dir,
        width=args.width,
        height=args.height,
        y_up=args.y_up,
    )
    print(out)


if __name__ == "__main__":
    _cli()
