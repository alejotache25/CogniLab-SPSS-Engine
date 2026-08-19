"""
Token definitions for the SPSS syntax lexer.

Token types represent the atomic units of SPSS syntax:
  - Commands, subcommands, keywords
  - Variables (case-preserving), strings, numbers
  - Operators (+, -, *, /, =, <>, <=, >=, EQ, NE, GT, LT, GE, LE)
  - Delimiters: slash, comma, parenthesis, dot (terminator)
  - Raw data block (BEGIN DATA ... END DATA)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class TokenType(Enum):
    """Enumeration of all token types produced by the lexer."""
    # Structural
    COMMAND = auto()        # First keyword of a command (uppercased)
    SUBCOMMAND = auto()     # Token after / (uppercased keyword)
    KEYWORD = auto()        # Reserved keyword: TO, ALL, BY, WITH, PAIRED, etc.
    TERMINATOR = auto()     # Command terminator: . (dot at end of command)

    # Identifiers and literals
    VARIABLE = auto()      # Variable name (case preserved)
    NUMBER = auto()        # Numeric literal
    STRING = auto()        # String literal (quotes stripped, embedded quotes resolved)

    # Operators (symbol)
    PLUS = auto()           # +
    MINUS = auto()          # -
    STAR = auto()           # * (multiply)
    SLASH = auto()          # / (but subcommand slash is handled separately)
    EQUALS = auto()         # =
    LT_GT = auto()          # <>
    LE_OP = auto()          # <=
    GE_OP = auto()          # >=
    LT_OP = auto()          # < (not used if we always get <= or <>)
    GT_OP = auto()          # > (not used if we always get >= or <>)

    # Operators (keyword)
    OP_EQ = auto()          # EQ
    OP_NE = auto()          # NE
    OP_GT = auto()          # GT
    OP_LT = auto()          # LT
    OP_GE = auto()          # GE
    OP_LE = auto()          # LE
    OP_AND = auto()         # AND
    OP_OR = auto()          # OR
    OP_NOT = auto()         # NOT

    # Delimiters
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    COMMA = auto()          # ,
    SUBCMD_SLASH = auto()   # / (subcommand separator)

    # Special
    RAW_DATA = auto()       # Raw data block from BEGIN DATA ... END DATA
    FORMAT_SPEC = auto()    # Format like (A), (F8.2), (A10) — parens-enclosed format
    EOF = auto()            # End of input


# Reserved keywords (always uppercased)
RESERVED_KEYWORDS: set[str] = {
    "TO", "ALL", "BY", "WITH", "PAIRED",
    "AND", "OR", "NOT",
    "EQ", "NE", "GT", "LT", "GE", "LE",
    "THRU", "LOWEST", "HIGHEST", "ELSE", "COPY", "SYSMIS", "MISSING",
    "LO", "HI", "INTO",
}

# Keyword operator mapping
KEYWORD_OPERATORS: dict[str, TokenType] = {
    "EQ": TokenType.OP_EQ,
    "NE": TokenType.OP_NE,
    "GT": TokenType.OP_GT,
    "LT": TokenType.OP_LT,
    "GE": TokenType.OP_GE,
    "LE": TokenType.OP_LE,
    "AND": TokenType.OP_AND,
    "OR": TokenType.OP_OR,
    "NOT": TokenType.OP_NOT,
}

# Commands that trigger a raw data block
DATA_BLOCK_START = "BEGIN DATA"
DATA_BLOCK_END = "END DATA"


@dataclass(frozen=True)
class Token:
    """An immutable token produced by the lexer."""
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return (f"Token({self.type.name}, {self.value!r}, "
                f"line={self.line}, col={self.column})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Token):
            return NotImplemented
        return (self.type == other.type and
                self.value == other.value and
                self.line == other.line and
                self.column == other.column)