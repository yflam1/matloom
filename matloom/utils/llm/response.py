from typing import Annotated, get_args, get_origin

from pydantic import BaseModel, Field, validate_call
from typing_extensions import Self

ZERO_SHOT_COT_REASONING = '"Let\'s think step by step. <<FILL_IN>>"'


class ResponseModelParsingError(RuntimeError):
    pass


class ResponseModel(BaseModel, validate_assignment=True):
    @classmethod
    def to_str(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def from_str(cls, text: str) -> Self:
        raise NotImplementedError()


class JsonResponseModel(ResponseModel):
    @classmethod
    @validate_call
    def to_str(
        cls,
        _indent: Annotated[int, Field(ge=0)] = 2,
        _depth: Annotated[int, Field(ge=1)] = 1,
        **substitutions: str,
    ) -> str:
        def get_fill(
            ann,
            depth: int,
            name: str | None = None,
            template: str | None = None,
        ) -> str:
            def check_ann(ann, cls: type) -> bool:
                origin = get_origin(ann)
                if origin is not None:
                    return isinstance(origin, type) and issubclass(origin, cls)
                return isinstance(ann, type) and issubclass(ann, cls)

            if name is not None and name in substitutions:
                return substitutions[name]
            if check_ann(ann, dict):
                raise RuntimeError("Use JsonResponseModel for dict fields")
            args = get_args(ann)
            if check_ann(ann, list):
                if len(args) == 0:
                    return "[...]"
                fill = "["
                for i, arg in enumerate(args):
                    fill += get_fill(arg, depth + 1, template=template)
                    fill += ", " if len(args) == 1 or i < len(args) - 1 else ""
                if len(args) == 1:
                    fill += "..."
                fill += "]"
                return fill
            elif issubclass(ann, JsonResponseModel):
                return ann.to_str(_depth=depth + 1, _indent=_indent, **substitutions)
            if template is not None:
                return template
            if issubclass(ann, bool):
                return "true/false"
            elif issubclass(ann, int):
                return "<<FILL_IN_INTEGER>>"
            elif issubclass(ann, float):
                return "<<FILL_IN_FLOAT>>"
            elif issubclass(ann, str):
                return '"<<FILL_IN_STRING>>"'
            else:
                raise TypeError(f"Unsupported data type: {ann}")

        output = "{\n"
        for i, (name, field) in enumerate(cls.model_fields.items()):
            ann = field.annotation
            args = get_args(ann)
            while len(args) == 2:
                ann = args[1] if args[0] is type(None) else args[0]
                args = get_args(ann)
            output += " " * (_indent * _depth)
            output += f'"{name}": '
            output += get_fill(
                ann,
                _depth,
                name=name,
                template=field.json_schema_extra.get("template", None)
                if field.json_schema_extra is not None
                else None,
            )
            output += "," if i < len(cls.model_fields) - 1 else ""
            if field.description is not None:
                output += f"  # {field.description}"
            output += "\n"
        output += " " * (_indent * (_depth - 1)) + "}"
        return output

    @classmethod
    @validate_call
    def from_str(cls, text: str) -> Self:
        from .parser import JsonParser

        parsed = JsonParser()(text)
        if parsed is None:
            raise ResponseModelParsingError("No JSON object could be extracted")
        try:
            return cls.model_validate(parsed)
        except Exception as e:
            raise ResponseModelParsingError(
                f"Failed to parse response into {cls.__name__}: {e!s}"
            ) from e
