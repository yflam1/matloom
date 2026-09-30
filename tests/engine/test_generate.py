"""Tests for matloom.engine.generate: the grammar-free random material generator."""

from __future__ import annotations

import pytest

from matloom.engine.expr import Constant
from matloom.engine.generate import (
    GeneratorConfig,
    MaterialGenerator,
    generate_material,
)
from matloom.engine.main import LayeredMaterial

# A spread of seeds reused across the property-style tests below.
_SEEDS = list(range(12))


# |-------------------|
# |   Return types    |
# |-------------------|


def test_generate_returns_material_by_default():
    out = generate_material(seed=0)
    assert isinstance(out, LayeredMaterial)


def test_generate_as_string_returns_serialized_dsl():
    out = generate_material(seed=0, as_string=True)
    assert isinstance(out, str)
    assert out.startswith("View(")
    assert "Material(" in out


def test_as_string_matches_material_serialize():
    material = generate_material(seed=5)
    string = generate_material(seed=5, as_string=True)
    assert material.serialize() == string


# |-------------------|
# |   Determinism     |
# |-------------------|


def test_same_seed_is_deterministic():
    a = generate_material(seed=123, as_string=True)
    b = generate_material(seed=123, as_string=True)
    assert a == b


def test_different_seeds_differ():
    a = generate_material(seed=1, as_string=True)
    b = generate_material(seed=2, as_string=True)
    assert a != b


# |----------------------------|
# |   Round-trip / validity    |
# |----------------------------|


@pytest.mark.parametrize("seed", _SEEDS)
def test_generated_material_roundtrips_stably(seed):
    # Building objects through the real constructors guarantees the output is a
    # valid program: it must deserialize and re-serialize identically.
    once = generate_material(seed=seed, as_string=True)
    twice = LayeredMaterial.deserialize(once).serialize()
    assert once == twice


@pytest.mark.parametrize("seed", _SEEDS)
def test_generated_material_deserializes_to_same_shape(seed):
    cfg = GeneratorConfig(num_layers=3, num_defs=2)
    src = generate_material(cfg, seed=seed, as_string=True)
    m = LayeredMaterial.deserialize(src)
    assert m.n_layers == 3
    assert len(m._defs) == 2


# |-------------------|
# |   Layer counts    |
# |-------------------|


def test_num_layers_is_exact():
    m = generate_material(GeneratorConfig(num_layers=4), seed=0)
    assert m.n_layers == 4


@pytest.mark.parametrize("seed", _SEEDS)
def test_layer_count_within_range(seed):
    cfg = GeneratorConfig(min_layers=2, max_layers=5)
    m = generate_material(cfg, seed=seed)
    assert 2 <= m.n_layers <= 5


def test_default_layer_count_within_1_to_3():
    for seed in range(40):
        m = generate_material(seed=seed)
        assert 1 <= m.n_layers <= 3


# |-------------------|
# |   Define preamble |
# |-------------------|


def test_num_defs_is_exact_and_named_in_order():
    m = generate_material(GeneratorConfig(num_defs=3), seed=0)
    assert [name for name, _ in m._defs] == ["def1", "def2", "def3"]


def test_zero_defs_emits_no_define():
    src = generate_material(GeneratorConfig(num_defs=0), seed=0, as_string=True)
    assert "Define(" not in src


@pytest.mark.parametrize("seed", _SEEDS)
def test_def_count_within_range(seed):
    cfg = GeneratorConfig(min_defs=1, max_defs=4)
    m = generate_material(cfg, seed=seed)
    assert 1 <= len(m._defs) <= 4


def test_defines_only_reference_earlier_defines():
    # Many defines + deep trees maximize the chance of a forward reference; if
    # one leaked, deserialize() would raise "unknown name". Stability is the
    # assertion.
    cfg = GeneratorConfig(num_defs=5, num_layers=2, max_depth=5)
    for seed in range(20):
        src = generate_material(cfg, seed=seed, as_string=True)
        LayeredMaterial.deserialize(src)  # must not raise


# |-------------------|
# |   Opaque base     |
# |-------------------|


def test_opaque_base_forces_bottom_alpha_to_one():
    m = generate_material(GeneratorConfig(opaque_base=True), seed=0)
    bottom = m._layers[0]
    assert isinstance(bottom.raw_alpha, Constant)
    assert bottom.raw_alpha._value == 1.0


def test_opaque_base_only_affects_bottom_layer():
    # With several layers, only index 0 is forced opaque; upper layers get a
    # generated (generally non-constant-1) alpha.
    cfg = GeneratorConfig(num_layers=3, opaque_base=True, max_depth=4)
    forced = 0
    for seed in range(30):
        m = generate_material(cfg, seed=seed)
        assert isinstance(m._layers[0].raw_alpha, Constant)
        assert m._layers[0].raw_alpha._value == 1.0
        upper = m._layers[1].raw_alpha
        if not (isinstance(upper, Constant) and upper._value == 1.0):
            forced += 1
    # At least some upper-layer alphas are not the forced constant 1.
    assert forced > 0


def test_non_opaque_base_still_roundtrips():
    cfg = GeneratorConfig(opaque_base=False)
    for seed in range(10):
        src = generate_material(cfg, seed=seed, as_string=True)
        assert LayeredMaterial.deserialize(src).serialize() == src


# |-------------------|
# |   Depth control   |
# |-------------------|


def test_max_depth_zero_produces_only_terminals():
    # Depth 0 still yields valid, round-trippable programs (terminal nodes only,
    # no nested unary/binary/transform/threshold wrapping a generated child).
    cfg = GeneratorConfig(max_depth=0, num_layers=2, num_defs=1)
    for seed in range(15):
        src = generate_material(cfg, seed=seed, as_string=True)
        assert LayeredMaterial.deserialize(src).serialize() == src


# |-------------------------|
# |   View region sanity    |
# |-------------------------|


@pytest.mark.parametrize("seed", _SEEDS)
def test_view_region_is_non_degenerate(seed):
    m = generate_material(seed=seed)
    x1, y1, x2, y2 = m._view
    assert x2 > x1
    assert y2 > y1


@pytest.mark.parametrize("seed", _SEEDS)
def test_random_view_disabled_pins_unit_square(seed):
    m = generate_material(GeneratorConfig(random_view=False), seed=seed)
    assert m._view == (0.0, 0.0, 1.0, 1.0)
    assert m.serialize().startswith("View(0, 0, 1, 1)")


# |---------------------------|
# |   Type-faithful sampling  |
# |---------------------------|


def test_numeric_samplers_respect_domains():
    gen = MaterialGenerator(
        GeneratorConfig(real_lo=-3.0, real_hi=4.0, pos_real_min=0.01, pos_real_max=2.0),
        seed=0,
    )
    for _ in range(2000):
        assert -3.0 <= gen._real() <= 4.0
        assert 0.01 <= gen._pos_real() <= 2.0
        assert 0.0 <= gen._non_neg_real() <= gen._cfg.non_neg_real_max
        assert gen._nonzero_real() != 0.0
        assert 0.0 <= gen._zero_to_one() <= 1.0
        assert 0.0 <= gen._degrees() <= 360.0
        assert 0 <= gen._byte() <= 255
        assert 0 <= gen._seed() <= 2**31 - 1


# |-------------------|
# |   Config errors   |
# |-------------------|


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_layers": 3, "max_layers": 2}, "min_layers"),
        ({"num_layers": 0}, "num_layers"),
        ({"min_layers": 0}, "positive"),
        ({"min_defs": 2, "max_defs": 1}, "min_defs"),
        ({"num_defs": -1}, "num_defs"),
        ({"max_depth": -1}, "max_depth"),
        ({"leaf_bias": 1.5}, "leaf_bias"),
        ({"real_lo": 5.0, "real_hi": 1.0}, "real_lo"),
        ({"pos_real_min": 0.0}, "pos_real_min"),
        ({"max_octaves": 0}, "max_octaves"),
        ({"max_path_segments": 0}, "max_path_segments"),
    ],
)
def test_invalid_config_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MaterialGenerator(GeneratorConfig(**kwargs))


# |-------------------|
# |   Export smoke    |
# |-------------------|


def test_generated_material_exports(tmp_path):
    # End-to-end: a generated material must evaluate and write its texture maps
    # without raising. Kept small (low depth, tiny grid) for speed.
    cfg = GeneratorConfig(num_layers=2, num_defs=1, max_depth=2)
    m = generate_material(cfg, seed=3)
    out = m.export(tmp_path, width=8, height=8)
    assert (out / "basecolor.png").exists()
