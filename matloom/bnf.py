from __future__ import annotations

import enum
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import auto
from typing import Annotated

from pydantic import Field, StringConstraints, validate_call

NonEmptyStr = Annotated[str, StringConstraints(strict=True, min_length=1)]


class BNF:
    @validate_call
    def __init__(self, bnf: NonEmptyStr) -> None:
        self._rules = self._load_bnf(bnf)
        self._token_pattern = re.compile(r'(<[^>]+>|"[^"]+")')

    @validate_call
    def generate(
        self,
        start: NonEmptyStr,
        samples: Annotated[int, Field(ge=1)] = 1,
        sep: str = "",
        max_depth: Annotated[int, Field(ge=1)] | None = None,
    ) -> Iterator[str]:
        for _ in range(samples):
            yield self._generate(start, sep=sep, max_depth=max_depth)

    def _remove_comments_outside_quotes(self, line: str) -> str:
        in_quote = False
        i = 0
        while i < len(line):
            if line[i] == '"':
                in_quote = not in_quote
                i += 1
                continue
            if not in_quote and (line[i] == "#" or line[i : i + 2] == "//"):
                return line[:i].strip()
            i += 1
        return line

    @validate_call
    def validate(self, text: str, start: NonEmptyStr) -> bool:
        for remainder in self._parse_for_remainders(text.strip(), start):
            print(remainder)
            if remainder == "":
                return True
        return False

    def _preprocess_line(self, line: str) -> str:
        line = self._remove_comments_outside_quotes(line.strip())
        line = line.removesuffix(";")
        return line.strip()

    def _get_options(self, fragments: list[str]) -> list[str]:
        rule = "".join(fragments)
        options = [opt.strip() for opt in rule.split("|")]
        return options

    def _load_bnf(self, bnf: str) -> dict[str, list[str]]:
        rules: dict[str, list[str]] = {}
        lines = bnf.strip().splitlines()
        name, fragments = None, []
        for line in lines:
            if (line := self._preprocess_line(line)) == "":
                continue
            if "::=" in line:
                # Handle previous rule
                if len(fragments) > 0:
                    rules[name] = self._get_options(fragments)
                    fragments.clear()

                name, content = line.split("::=", maxsplit=1)
                name = name.strip()
                if name.startswith("<") and name.endswith(">"):
                    name = name[1:-1]
                else:
                    raise ValueError(f"Invalid rule name: {name}")
                fragments.append(content.strip())
            else:
                fragments.append(line)
        # Handle last rule
        if len(fragments) > 0:
            rules[name] = self._get_options(fragments)
        return rules

    def _generate(
        self,
        start: str,
        sep: str = "",
        max_depth: int | None = None,
        _depth: int = 0,
    ) -> str:
        if max_depth is not None and _depth > max_depth:
            return ""
        if start not in self._rules:
            raise ValueError(f"Rule <{start}> not found in grammar")
        result = []
        choice = random.choice(self._rules[start])
        tokens = self._token_pattern.split(choice)
        for token in tokens:
            assert isinstance(token, str)
            token = token.strip()
            if token == "":
                continue
            if token.startswith("<") and token.endswith(">"):
                result.append(
                    self._generate(
                        token[1:-1],
                        sep=sep,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                    )
                )
            elif token.startswith('"') and token.endswith('"'):
                result.append(token[1:-1])
            else:
                result.append(token)
        return sep.join(result)

    def _parse_for_remainders(self, text: str, start: str) -> Iterator[str]:
        if start not in self._rules:
            raise ValueError(f"Rule <{start}> not found in grammar")
        for option in self._rules[start]:
            possible_remainders = {text}
            tokens = [t.strip() for t in self._token_pattern.split(option) if t.strip()]
            is_valid = True
            for token in tokens:
                next_remainders: set[str] = set()
                for remainder in possible_remainders:
                    if token.startswith('"') and token.endswith('"'):
                        literal = token[1:-1]
                        remainder = remainder.lstrip()
                        if remainder.startswith(literal):
                            next_remainders.add(remainder[len(literal) :])
                    elif token.startswith("<") and token.endswith(">"):
                        next_remainders.update(
                            self._parse_for_remainders(remainder.lstrip(), token[1:-1])
                        )
                    else:
                        if remainder.lstrip().startswith(token):
                            next_remainders.add(remainder.lstrip()[len(token) :])
                possible_remainders = next_remainders
                if len(possible_remainders) == 0:
                    is_valid = False
                    break
            if is_valid:
                yield from possible_remainders


# |==========================================|
# |     Extended BNF (ISO 14977) support     |
# |==========================================|


class EbnfParseError(Exception):
    """Raised when EBNF grammar text cannot be parsed."""

    def __init__(self, msg: str, pos: int) -> None:
        super().__init__(f"{msg} at position {pos}")
        self.pos = pos


# |------------------------|
# |     AST node types     |
# |------------------------|


@dataclass(frozen=True, slots=True)
class Terminal:
    value: str


@dataclass(frozen=True, slots=True)
class Nonterminal:
    name: str


@dataclass(frozen=True, slots=True)
class SpecialSequence:
    text: str


@dataclass(frozen=True, slots=True)
class Concatenation:
    items: list[EbnfNode]


@dataclass(frozen=True, slots=True)
class Alternation:
    choices: list[EbnfNode]


@dataclass(frozen=True, slots=True)
class Optional_:
    child: EbnfNode


@dataclass(frozen=True, slots=True)
class Repetition:
    child: EbnfNode


@dataclass(frozen=True, slots=True)
class Group:
    child: EbnfNode


@dataclass(frozen=True, slots=True)
class Exception_:
    base: EbnfNode
    excluded: EbnfNode


@dataclass(frozen=True, slots=True)
class RepetitionFactor:
    count: int
    child: EbnfNode


EbnfNode = (
    Terminal
    | Nonterminal
    | SpecialSequence
    | Concatenation
    | Alternation
    | Optional_
    | Repetition
    | Group
    | Exception_
    | RepetitionFactor
)


# |-------------------|
# |     Tokenizer     |
# |-------------------|


class _TokenType(enum.Enum):
    IDENTIFIER = auto()
    TERMINAL_DQ = auto()
    TERMINAL_SQ = auto()
    EQUALS = auto()
    SEMICOLON = auto()
    PIPE = auto()
    COMMA = auto()
    MINUS = auto()
    STAR = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    SPECIAL_SEQ = auto()
    INTEGER = auto()
    EOF = auto()


@dataclass(frozen=True, slots=True)
class _Token:
    type: _TokenType
    value: str
    pos: int


_SINGLE_CHAR_TOKENS: dict[str, _TokenType] = {
    "=": _TokenType.EQUALS,
    ";": _TokenType.SEMICOLON,
    "|": _TokenType.PIPE,
    ",": _TokenType.COMMA,
    "-": _TokenType.MINUS,
    "*": _TokenType.STAR,
    "(": _TokenType.LPAREN,
    ")": _TokenType.RPAREN,
    "[": _TokenType.LBRACKET,
    "]": _TokenType.RBRACKET,
    "{": _TokenType.LBRACE,
    "}": _TokenType.RBRACE,
}


def _tokenize(text: str) -> list[_Token]:
    text = text.replace("\n", " ")
    tokens: list[_Token] = []
    i = 0
    n = len(text)

    while i < n:
        # Skip whitespace
        if text[i].isspace():
            i += 1
            continue

        # Skip nestable (* ... *) comments
        if i + 1 < n and text[i] == "(" and text[i + 1] == "*":
            depth = 1
            i += 2
            while i < n and depth > 0:
                if i + 1 < n and text[i] == "(" and text[i + 1] == "*":
                    depth += 1
                    i += 2
                elif i + 1 < n and text[i] == "*" and text[i + 1] == ")":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth > 0:
                raise EbnfParseError("Unterminated comment", i)
            continue

        # Single-character tokens (but not '(' which starts a comment)
        if text[i] in _SINGLE_CHAR_TOKENS:
            tokens.append(_Token(_SINGLE_CHAR_TOKENS[text[i]], text[i], i))
            i += 1
            continue

        # Double-quoted terminal
        if text[i] == '"':
            start = i
            i += 1
            while i < n and text[i] != '"':
                i += 1
            if i >= n:
                raise EbnfParseError("Unterminated double-quoted string", start)
            tokens.append(_Token(_TokenType.TERMINAL_DQ, text[start + 1 : i], start))
            i += 1
            continue

        # Single-quoted terminal
        if text[i] == "'":
            start = i
            i += 1
            while i < n and text[i] != "'":
                i += 1
            if i >= n:
                raise EbnfParseError("Unterminated single-quoted string", start)
            tokens.append(_Token(_TokenType.TERMINAL_SQ, text[start + 1 : i], start))
            i += 1
            continue

        # Special sequence ? ... ?
        if text[i] == "?":
            start = i
            i += 1
            while i < n and text[i] != "?":
                i += 1
            if i >= n:
                raise EbnfParseError("Unterminated special sequence", start)
            tokens.append(
                _Token(_TokenType.SPECIAL_SEQ, text[start + 1 : i].strip(), start)
            )
            i += 1
            continue

        # Identifier or integer
        if text[i].isalpha() or text[i] == "_":
            start = i
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            tokens.append(_Token(_TokenType.IDENTIFIER, text[start:i], start))
            continue

        if text[i].isdigit():
            start = i
            while i < n and text[i].isdigit():
                i += 1
            tokens.append(_Token(_TokenType.INTEGER, text[start:i], start))
            continue

        raise EbnfParseError(f"Unexpected character {text[i]!r}", i)

    tokens.append(_Token(_TokenType.EOF, "", n))
    return tokens


# |----------------------------------|
# |     Recursive descent parser     |
# |----------------------------------|


class _EbnfParser:
    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> _Token:
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _at(self, tt: _TokenType) -> bool:
        return self._peek().type == tt

    def _expect(self, tt: _TokenType) -> _Token:
        tok = self._advance()
        if tok.type != tt:
            raise EbnfParseError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})",
                tok.pos,
            )
        return tok

    def parse_grammar(self) -> dict[str, EbnfNode]:
        rules: dict[str, EbnfNode] = {}
        while not self._at(_TokenType.EOF):
            name, node = self._parse_rule()
            rules[name] = node
        return rules

    def _parse_rule(self) -> tuple[str, EbnfNode]:
        name_tok = self._expect(_TokenType.IDENTIFIER)
        self._expect(_TokenType.EQUALS)
        body = self._parse_expression()
        self._expect(_TokenType.SEMICOLON)
        return (name_tok.value, body)

    def _parse_expression(self) -> EbnfNode:
        """Alternation (lowest precedence)."""
        left = self._parse_term()
        choices = [left]
        while self._at(_TokenType.PIPE):
            self._advance()
            choices.append(self._parse_term())
        if len(choices) == 1:
            return choices[0]
        return Alternation(choices=choices)

    def _parse_term(self) -> EbnfNode:
        """Concatenation (binds tighter than alternation)."""
        left = self._parse_exception()
        items = [left]
        while self._at(_TokenType.COMMA):
            self._advance()
            items.append(self._parse_exception())
        if len(items) == 1:
            return items[0]
        return Concatenation(items=items)

    def _parse_exception(self) -> EbnfNode:
        """Exception (binds tighter than concatenation)."""
        base = self._parse_factor()
        if self._at(_TokenType.MINUS):
            self._advance()
            excluded = self._parse_factor()
            return Exception_(base=base, excluded=excluded)
        return base

    def _parse_factor(self) -> EbnfNode:
        """Repetition factor: N * primary."""
        # Lookahead: only treat as repetition factor if followed by *
        if self._at(_TokenType.INTEGER) and (
            self._pos + 1 < len(self._tokens)
            and self._tokens[self._pos + 1].type == _TokenType.STAR
        ):
            count_tok = self._advance()
            self._advance()  # consume *
            child = self._parse_primary()
            return RepetitionFactor(count=int(count_tok.value), child=child)
        return self._parse_primary()

    def _parse_primary(self) -> EbnfNode:
        tok = self._peek()

        if tok.type in (_TokenType.TERMINAL_DQ, _TokenType.TERMINAL_SQ):
            self._advance()
            return Terminal(value=tok.value)

        if tok.type == _TokenType.IDENTIFIER:
            self._advance()
            return Nonterminal(name=tok.value)

        if tok.type == _TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(_TokenType.RPAREN)
            return Group(child=expr)

        if tok.type == _TokenType.LBRACKET:
            self._advance()
            expr = self._parse_expression()
            self._expect(_TokenType.RBRACKET)
            return Optional_(child=expr)

        if tok.type == _TokenType.LBRACE:
            self._advance()
            expr = self._parse_expression()
            self._expect(_TokenType.RBRACE)
            return Repetition(child=expr)

        if tok.type == _TokenType.SPECIAL_SEQ:
            self._advance()
            return SpecialSequence(text=tok.value)

        raise EbnfParseError(
            f"Unexpected token {tok.type.name} ({tok.value!r})", tok.pos
        )


# |---------------------------|
# |     ExtendedBNF class     |
# |---------------------------|


class ExtendedBNF:
    @validate_call
    def __init__(self, ebnf: NonEmptyStr) -> None:
        tokens = _tokenize(ebnf)
        parser = _EbnfParser(tokens)
        self._rules: dict[str, EbnfNode] = parser.parse_grammar()

    def _get_rule(self, name: str) -> EbnfNode:
        if name not in self._rules:
            raise ValueError(f"Rule '{name}' not found in grammar")
        return self._rules[name]

    # |--------------------|
    # |     Generation     |
    # |--------------------|

    @validate_call
    def generate(
        self,
        start: NonEmptyStr,
        samples: Annotated[int, Field(ge=1)] = 1,
        sep: str = "",
        max_depth: Annotated[int, Field(ge=1)] | None = None,
        max_repeat: Annotated[int, Field(ge=0)] = 3,
    ) -> Iterator[str]:
        for _ in range(samples):
            yield self._generate_node(
                self._get_rule(start),
                sep=sep,
                max_depth=max_depth,
                max_repeat=max_repeat,
                _depth=0,
            )

    def _generate_node(
        self,
        node: EbnfNode,
        *,
        sep: str,
        max_depth: int | None,
        max_repeat: int,
        _depth: int,
    ) -> str:
        kw = dict(sep=sep, max_depth=max_depth, max_repeat=max_repeat, _depth=_depth)

        if isinstance(node, Terminal):
            return node.value

        if isinstance(node, Nonterminal):
            if max_depth is not None and _depth >= max_depth:
                return ""
            return self._generate_node(
                self._get_rule(node.name), **{**kw, "_depth": _depth + 1}
            )

        if isinstance(node, Concatenation):
            parts = [self._generate_node(item, **kw) for item in node.items]
            return sep.join(parts)

        if isinstance(node, Alternation):
            chosen = random.choice(node.choices)
            return self._generate_node(chosen, **kw)

        if isinstance(node, Optional_):
            if random.random() < 0.5:
                return ""
            return self._generate_node(node.child, **kw)

        if isinstance(node, Repetition):
            count = random.randint(0, max_repeat)
            parts = [self._generate_node(node.child, **kw) for _ in range(count)]
            return sep.join(parts)

        if isinstance(node, Group):
            return self._generate_node(node.child, **kw)

        if isinstance(node, Exception_):
            for _ in range(100):
                candidate = self._generate_node(node.base, **kw)
                # Check if candidate matches the excluded pattern
                matched = False
                for rem in self._validate_node(candidate, node.excluded):
                    if rem == "":
                        matched = True
                        break
                if not matched:
                    return candidate
            raise RuntimeError(
                "Could not generate value for exception after 100 retries"
            )

        if isinstance(node, RepetitionFactor):
            parts = [self._generate_node(node.child, **kw) for _ in range(node.count)]
            return sep.join(parts)

        if isinstance(node, SpecialSequence):
            raise NotImplementedError(
                f"Cannot generate from special sequence: ?{node.text}?"
            )

        raise TypeError(f"Unknown node type: {type(node)}")  # pragma: no cover

    # |--------------------|
    # |     Validation     |
    # |--------------------|

    @validate_call
    def validate(self, text: str, start: NonEmptyStr) -> bool:
        for remainder in self._validate_node(text.strip(), self._get_rule(start)):
            if remainder == "":
                return True
        return False

    def _validate_node(self, text: str, node: EbnfNode) -> Iterator[str]:
        if isinstance(node, Terminal):
            stripped = text.lstrip()
            if stripped.startswith(node.value):
                yield stripped[len(node.value) :]

        elif isinstance(node, Nonterminal):
            yield from self._validate_node(text, self._get_rule(node.name))

        elif isinstance(node, Concatenation):
            possible: set[str] = {text}
            for item in node.items:
                next_possible: set[str] = set()
                for rem in possible:
                    next_possible.update(self._validate_node(rem, item))
                possible = next_possible
                if not possible:
                    return
            yield from possible

        elif isinstance(node, Alternation):
            for choice in node.choices:
                yield from self._validate_node(text, choice)

        elif isinstance(node, Optional_):
            yield text  # skip case
            yield from self._validate_node(text, node.child)

        elif isinstance(node, Repetition):
            # Fixed-point iteration with seen set to prevent infinite loops
            seen: set[str] = set()
            yield text  # zero repetitions
            seen.add(text)
            current: set[str] = {text}
            while current:
                next_round: set[str] = set()
                for rem in current:
                    for new_rem in self._validate_node(rem, node.child):
                        if new_rem not in seen:
                            seen.add(new_rem)
                            next_round.add(new_rem)
                            yield new_rem
                current = next_round

        elif isinstance(node, Group):
            yield from self._validate_node(text, node.child)

        elif isinstance(node, Exception_):
            for remainder in self._validate_node(text, node.base):
                consumed = text[: len(text) - len(remainder)]
                # Check if consumed text matches the excluded pattern
                excluded_match = False
                for exc_rem in self._validate_node(consumed, node.excluded):
                    if exc_rem == "":
                        excluded_match = True
                        break
                if not excluded_match:
                    yield remainder

        elif isinstance(node, RepetitionFactor):
            # Expand to concatenation of count copies
            expanded = Concatenation(items=[node.child] * node.count)
            yield from self._validate_node(text, expanded)

        elif isinstance(node, SpecialSequence):
            # Match any non-empty prefix
            for i in range(1, len(text) + 1):
                yield text[i:]

        else:
            raise TypeError(f"Unknown node type: {type(node)}")  # pragma: no cover


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--ebnf",
        action="store_true",
        help="Use ExtendedBNF (ISO 14977 EBNF) parser",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gen", type=int, default=5)
    group.add_argument("--val", type=str, default=None)
    parser.add_argument("--start", type=str, default="program")
    # A recursive grammar (e.g. the material DSL, whose `expr` nests without
    # bound) recurses until the Python stack overflows with a RecursionError
    # unless generation is depth-limited. Cap it by default; pass
    # `--max-depth 0` to lift the cap (only safe for non-recursive grammars).
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-repeat", type=int, default=3)
    args = parser.parse_args()

    with open(args.path) as f:
        content = f.read()
    if args.path.endswith(".md"):
        tag = "ebnf" if args.ebnf else "bnf"
        blocks = re.findall(rf"```{tag}(.*?)```", content, re.DOTALL)
        if len(blocks) == 0:
            raise ValueError(f"No {tag.upper()} block found in the .md file")
        content = blocks[0]
        assert isinstance(content, str)

    grammar = ExtendedBNF(content) if args.ebnf else BNF(content)
    if args.val is not None:
        is_valid = grammar.validate(args.val, start=args.start)
        print(f"Validation result: {is_valid}")
    else:
        max_depth = args.max_depth if args.max_depth > 0 else None
        gen_kwargs = {"max_depth": max_depth}
        if args.ebnf:
            gen_kwargs["max_repeat"] = args.max_repeat
        samples = list(grammar.generate(args.start, samples=args.gen, **gen_kwargs))
        print(("\n" + "-" * 20 + "\n").join(samples))
