from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BaseModel, Field, computed_field, validate_call
from pydantic.fields import FieldInfo
from typing_extensions import Self

from matloom.engine.expr import Constant, Expression2D
from matloom.engine.util import Threshold, sRGB2Linear


def _raw(default_factory: Callable[[], Any], name: str) -> FieldInfo:
    return Field(
        default_factory=default_factory,
        alias=name,
        json_schema_extra={"repr_name": name},
    )


class _CustomBaseModel(BaseModel, arbitrary_types_allowed=True):
    def __repr_args__(self):
        fields = type(self).model_fields
        for k, v in super().__repr_args__():
            extra = (fields[k].json_schema_extra or {}) if k in fields else {}
            yield extra.get("repr_name", k), v

    def __str__(self) -> str:
        return self.__repr__()


class Color(_CustomBaseModel):
    raw_r: Annotated[Expression2D, _raw(lambda: Constant(), "r")]
    raw_g: Annotated[Expression2D, _raw(lambda: Constant(), "g")]
    raw_b: Annotated[Expression2D, _raw(lambda: Constant(), "b")]

    @computed_field(repr=False)
    @property
    def r(self) -> Expression2D:
        return sRGB2Linear(Threshold(self.raw_r, below_at=0.0, above_at=1.0))

    @computed_field(repr=False)
    @property
    def g(self) -> Expression2D:
        return sRGB2Linear(Threshold(self.raw_g, below_at=0.0, above_at=1.0))

    @computed_field(repr=False)
    @property
    def b(self) -> Expression2D:
        return sRGB2Linear(Threshold(self.raw_b, below_at=0.0, above_at=1.0))

    @classmethod
    @validate_call
    def from_int(
        cls,
        *,
        r: Annotated[int, Field(ge=0, le=255)] = 0,
        g: Annotated[int, Field(ge=0, le=255)] = 0,
        b: Annotated[int, Field(ge=0, le=255)] = 0,
    ) -> Self:
        return cls(
            r=Constant(r) / 255.0,
            g=Constant(g) / 255.0,
            b=Constant(b) / 255.0,
        )


class Emissive(Color):
    raw_strength: Annotated[Expression2D, _raw(lambda: Constant(), "strength")]

    @computed_field(repr=False)
    @property
    def strength(self) -> Expression2D:
        return Threshold(self.raw_strength, below_at=0.0)


class Layer(_CustomBaseModel):
    raw_alpha: Annotated[Expression2D, _raw(lambda: Constant(1.0), "alpha")]
    basecolor: Annotated[Color, Field(default_factory=lambda: Color())]
    raw_metallic: Annotated[Expression2D, _raw(lambda: Constant(), "metallic")]
    raw_roughness: Annotated[Expression2D, _raw(lambda: Constant(), "roughness")]
    # OpenPBR-aligned reflectance channels (each a [0, 1] scalar like metallic,
    # except `ior` which is a refractive index >= 1). They default to the inert
    # values (weights 0, ior 1.5) so existing materials render identically.
    raw_sheen: Annotated[Expression2D, _raw(lambda: Constant(), "sheen")]
    raw_coat: Annotated[Expression2D, _raw(lambda: Constant(), "coat")]
    raw_transmission: Annotated[Expression2D, _raw(lambda: Constant(), "transmission")]
    raw_ior: Annotated[Expression2D, _raw(lambda: Constant(1.5), "ior")]
    raw_subsurface: Annotated[Expression2D, _raw(lambda: Constant(), "subsurface")]
    raw_anisotropy: Annotated[Expression2D, _raw(lambda: Constant(), "anisotropy")]
    emissive: Annotated[Emissive, Field(default_factory=lambda: Emissive())]
    raw_height: Annotated[Expression2D, _raw(lambda: Constant(), "height")]

    @computed_field(repr=False)
    @property
    def alpha(self) -> Expression2D:
        return Threshold(self.raw_alpha, below_at=0.0, above_at=1.0)

    @computed_field(repr=False)
    @property
    def metallic(self) -> Expression2D:
        return Threshold(self.raw_metallic, below_at=0.0, above_at=1.0)

    @computed_field(repr=False)
    @property
    def roughness(self) -> Expression2D:
        return Threshold(self.raw_roughness, below_at=0.0, above_at=1.0)

    @computed_field(repr=False)
    @property
    def sheen(self) -> Expression2D:
        # Microfiber sheen weight (cloth/velvet/fabric fuzz lobe).
        return Threshold(self.raw_sheen, below_at=0.0, above_at=1.0)

    @computed_field(repr=False)
    @property
    def coat(self) -> Expression2D:
        # Clearcoat weight (lacquer, car paint, glaze) — a glossy layer on top.
        return Threshold(self.raw_coat, below_at=0.0, above_at=1.0)

    @computed_field(repr=False)
    @property
    def transmission(self) -> Expression2D:
        # Transmission weight (glass, ice, gel): how much light passes through.
        return Threshold(self.raw_transmission, below_at=0.0, above_at=1.0)

    @computed_field(repr=False)
    @property
    def ior(self) -> Expression2D:
        # Index of refraction (>= 1; default 1.5). Governs dielectric specular
        # and how transmission bends light; clamped to the physical >= 1 range.
        return Threshold(self.raw_ior, below_at=1.0)

    @computed_field(repr=False)
    @property
    def subsurface(self) -> Expression2D:
        # Subsurface-scattering weight (skin, wax, jade, marble). The scatter
        # color reuses the base color (matching Blender 4.x Principled BSDF).
        return Threshold(self.raw_subsurface, below_at=0.0, above_at=1.0)

    @computed_field(repr=False)
    @property
    def anisotropy(self) -> Expression2D:
        # Specular anisotropy strength (brushed metal, satin). Direction is the
        # tangent (X) axis; only the magnitude is authored, to keep it minimal.
        return Threshold(self.raw_anisotropy, below_at=0.0, above_at=1.0)

    @computed_field(repr=False)
    @property
    def height(self) -> Expression2D:
        # Absolute displacement in coordinate units (same space as the X/Y
        # plane); left unclamped (it may be any real value, including negative).
        # The per-layer fields are composited with the max operator (alpha as a
        # coverage mask) in matloom.engine.relief.composite_height.
        return self.raw_height
