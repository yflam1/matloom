"""Tests for matloom.utils.dtypes and matloom.utils.anybase."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from matloom.utils.anybase import AnyBase
from matloom.utils.dtypes import NonZeroFloat


class _Sample(AnyBase):
    pass


def test_anybase_cname_returns_class_name():
    assert _Sample().cname_ == "_Sample"


def test_anybase_subclass_cname():
    class Inner(AnyBase):
        pass

    assert Inner().cname_ == "Inner"


# NonZeroFloat is an Annotated type with an AfterValidator; exercise it through
# a pydantic model the way the engine does.
class _Model(BaseModel):
    value: NonZeroFloat


def test_nonzerofloat_accepts_nonzero():
    assert _Model(value=1.5).value == 1.5
    assert _Model(value=-2.0).value == -2.0


def test_nonzerofloat_rejects_zero():
    with pytest.raises(Exception, match="cannot be exactly zero"):
        _Model(value=0.0)
