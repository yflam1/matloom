"""Tests for matloom.bnf: the BNF and ExtendedBNF grammar engines."""

from __future__ import annotations

import random

import pytest

from matloom.bnf import BNF, EbnfParseError, ExtendedBNF

# |---------------------|
# |   BNF (::= form)    |
# |---------------------|


def test_bnf_generates_terminal():
    g = BNF('<greeting> ::= "hello"')
    assert list(g.generate("greeting", samples=1)) == ["hello"]


def test_bnf_alternation_only_yields_known_options():
    g = BNF('<color> ::= "red" | "green" | "blue"')
    random.seed(0)
    samples = list(g.generate("color", samples=20))
    assert set(samples) <= {"red", "green", "blue"}


def test_bnf_nonterminal_expansion():
    g = BNF('<phrase> ::= <word> <word>\n<word> ::= "ab"')
    assert list(g.generate("phrase", samples=1)) == ["abab"]


def test_bnf_separator_inserted_between_tokens():
    g = BNF('<pair> ::= "a" "b"')
    assert list(g.generate("pair", samples=1, sep="-")) == ["a-b"]


def test_bnf_missing_rule_raises():
    g = BNF('<x> ::= "y"')
    with pytest.raises(ValueError, match="not found"):
        list(g.generate("nonexistent"))


def test_bnf_invalid_rule_name_raises():
    with pytest.raises(ValueError, match="Invalid rule name"):
        BNF('badname ::= "x"')


def test_bnf_comments_are_stripped():
    g = BNF('<x> ::= "y"  # this is a comment')
    assert list(g.generate("x", samples=1)) == ["y"]


def test_bnf_validate_accepts_valid_string():
    g = BNF('<color> ::= "red" | "blue"')
    assert g.validate("red", start="color")


def test_bnf_validate_rejects_invalid_string():
    g = BNF('<color> ::= "red" | "blue"')
    assert not g.validate("green", start="color")


def test_bnf_empty_grammar_string_rejected():
    with pytest.raises(ValueError):
        BNF("")


# |--------------------------------|
# |   ExtendedBNF (ISO 14977)      |
# |--------------------------------|


def test_ebnf_generates_terminal():
    g = ExtendedBNF('greeting = "hello" ;')
    assert list(g.generate("greeting", samples=1)) == ["hello"]


def test_ebnf_concatenation():
    g = ExtendedBNF('pair = "a" , "b" ;')
    assert list(g.generate("pair", samples=1)) == ["ab"]


def test_ebnf_alternation_only_known_options():
    g = ExtendedBNF('color = "red" | "green" | "blue" ;')
    random.seed(1)
    samples = list(g.generate("color", samples=20))
    assert set(samples) <= {"red", "green", "blue"}


def test_ebnf_optional_can_be_empty_or_present():
    g = ExtendedBNF('x = "a" , [ "b" ] ;')
    random.seed(2)
    samples = set(g.generate("x", samples=30))
    assert samples <= {"a", "ab"}
    assert "a" in samples or "ab" in samples


def test_ebnf_repetition_factor_repeats_exactly():
    g = ExtendedBNF('x = 3 * "a" ;')
    assert list(g.generate("x", samples=1)) == ["aaa"]


def test_ebnf_repetition_braces_respects_max_repeat():
    g = ExtendedBNF('x = { "a" } ;')
    random.seed(3)
    for s in g.generate("x", samples=20, max_repeat=2):
        assert len(s) <= 2  # at most max_repeat copies of "a"


def test_ebnf_nonterminal_expansion():
    g = ExtendedBNF('phrase = word , word ; word = "ab" ;')
    assert list(g.generate("phrase", samples=1)) == ["abab"]


def test_ebnf_comments_ignored():
    g = ExtendedBNF('x = (* a comment *) "y" ;')
    assert list(g.generate("x", samples=1)) == ["y"]


def test_ebnf_nested_comments():
    g = ExtendedBNF('x = (* outer (* inner *) still comment *) "y" ;')
    assert list(g.generate("x", samples=1)) == ["y"]


def test_ebnf_validate_accepts_valid():
    g = ExtendedBNF('color = "red" | "blue" ;')
    assert g.validate("red", start="color")


def test_ebnf_validate_rejects_invalid():
    g = ExtendedBNF('color = "red" | "blue" ;')
    assert not g.validate("green", start="color")


def test_ebnf_validate_repetition():
    g = ExtendedBNF('x = { "a" } ;')
    assert g.validate("aaaa", start="x")
    assert g.validate("", start="x")


def test_ebnf_unterminated_comment_raises():
    with pytest.raises(EbnfParseError, match="Unterminated comment"):
        ExtendedBNF('x = (* unclosed "y" ;')


def test_ebnf_unterminated_string_raises():
    with pytest.raises(EbnfParseError, match="Unterminated double-quoted"):
        ExtendedBNF('x = "unclosed ;')


def test_ebnf_missing_semicolon_raises():
    with pytest.raises(EbnfParseError):
        ExtendedBNF('x = "y"')


def test_ebnf_missing_rule_raises():
    g = ExtendedBNF('x = "y" ;')
    with pytest.raises(ValueError, match="not found"):
        list(g.generate("missing"))


def test_ebnf_max_depth_limits_recursion():
    # A left-recursive grammar would loop forever without max_depth.
    g = ExtendedBNF('x = "a" , x | "b" ;')
    random.seed(4)
    for s in g.generate("x", samples=10, max_depth=5):
        assert isinstance(s, str)


def test_ebnf_special_sequence_generation_unsupported():
    g = ExtendedBNF("x = ? any text ? ;")
    with pytest.raises(NotImplementedError):
        list(g.generate("x"))

