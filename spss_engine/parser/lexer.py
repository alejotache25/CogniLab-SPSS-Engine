"""
Lexer for SPSS syntax.

Tokenizes SPSS command syntax into a stream of Token objects.

Key rules (from CSR Universals):
  - Commands, subcommands, keywords are uppercased.
  - Variable names preserve case.
  - Strings: single or double quoted; embedded quotes via doubling ('Client''s').
  - BEGIN DATA ... END DATA is captured as a single RAW_DATA token.
  - Command terminator: . (dot), must be last nonblank char.
  - Subcommands preceded by /.
  - Operators: + - * / = <> <= >= and keyword forms EQ NE GT LT GE LE.
  - Numbers, parentheses, commas are individual tokens.
"""

from __future__ import annotations
import re
from typing import List, Optional

from spss_engine.parser.tokens import (
    Token, TokenType, RESERVED_KEYWORDS, KEYWORD_OPERATORS,
    DATA_BLOCK_START, DATA_BLOCK_END,
)
from spss_engine.utils.errors import SPSSSyntaxError


# Characters that start an identifier (variable name or command/keyword)
# Note: $ and # are only valid at position 0 for system/scratch variables
_IDENT_START = re.compile(r"[A-Za-z@#$]")
# Identifier characters: letters, digits, @, #, $, _
# NOTE: period (.) is NOT included here to avoid consuming the command
# terminator. SPSS allows . within variable names, but a trailing . is always
# a command terminator. This is a simplification for Phase 1.
_IDENT_CHAR = re.compile(r"[A-Za-z0-9@#$_]")
# A number: optional sign, digits, optional decimal part, optional exponent
_NUMBER_RE = re.compile(r"\d+\.\d+([eE][+-]?\d+)?|\d+([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?")
# Column range like 1-3 or 7-8
_COL_RANGE_RE = re.compile(r"\d+-\d+")

# Two-word commands (for context tracking in the lexer)
_TWO_WORD_COMMANDS_SET: set[str] = {
    "DATA LIST", "SELECT IF", "DO IF", "END IF", "END DATA",
    "SPLIT FILE", "SORT CASES", "MISSING VALUES",
    "VARIABLE LABELS", "VALUE LABELS", "NPAR TESTS",
}

# Commands where = introduces an expression (so / is division, not subcommand)
_EXPRESSION_COMMANDS: set[str] = {"COMPUTE", "IF", "SELECT IF", "DO IF"}


class Lexer:
    """Tokenizes SPSS syntax string into a list of Token objects."""

    def __init__(self, text: str) -> None:
        self._text: str = text
        self._pos: int = 0
        self._line: int = 1
        self._col: int = 1
        self._tokens: List[Token] = []
        # Track whether the next identifier token is a command name
        # (first token after a terminator or at start)
        self._expecting_command: bool = True
        # Track the current command name (for context-sensitive lexing)
        self._current_command: str = ""
        # Track whether we're inside an expression context (after = in COMPUTE etc.)
        self._in_expression: bool = False

    def tokenize(self) -> List[Token]:
        """Tokenize the input text and return a list of tokens."""
        while self._pos < len(self._text):
            ch = self._text[self._pos]

            # Track line/column
            if ch == "\n":
                self._line += 1
                self._col = 1
                self._pos += 1
                continue

            # Skip whitespace
            if ch.isspace():
                self._pos += 1
                self._col += 1
                continue

            # Skip comments: * starts a comment line to end of line
            if ch == "*":
                # Only treat as comment if at start of a command position
                # (i.e., expecting_command is True and previous char was newline or start)
                if self._expecting_command:
                    self._skip_comment_line()
                    continue
                # Otherwise * is multiply
                self._add_token(TokenType.STAR, "*")
                self._pos += 1
                self._col += 1
                continue

            # COMMENT keyword
            if (self._expecting_command and
                    self._peek_keyword() == "COMMENT"):
                self._skip_comment_line()
                continue

            # BEGIN DATA ... END DATA block
            if self._expecting_command and self._peek_keyword() == "BEGIN DATA":
                self._lex_data_block()
                continue

            # Subcommand slash
            if ch == "/":
                # Distinguish subcommand slash from division
                # In SPSS, / at the start of a token (after whitespace) is subcommand
                # In expressions, * is multiply. / outside expressions is subcommand.
                if self._is_subcommand_context():
                    self._add_token(TokenType.SUBCMD_SLASH, "/")
                    self._pos += 1
                    self._col += 1
                    continue
                else:
                    self._add_token(TokenType.SLASH, "/")
                    self._pos += 1
                    self._col += 1
                    continue

            # Command terminator: . (dot)
            # A dot is a terminator when it's at the end of a command line
            # and not part of a number or format spec
            if ch == ".":
                # Check if it's part of a number (e.g., .5 or 3.14)
                if self._pos + 1 < len(self._text) and self._text[self._pos + 1].isdigit():
                    self._lex_number()
                    continue
                # Check if it's a command terminator (dot followed by whitespace/newline/EOF)
                # In interactive mode, . at end of line is the terminator
                rest = self._text[self._pos + 1:].lstrip()
                if rest == "" or rest[0] == "\n":
                    # It's a terminator
                    self._add_token(TokenType.TERMINATOR, ".")
                    self._pos += 1
                    self._col += 1
                    self._expecting_command = True
                    self._current_command = ""
                    self._in_expression = False
                    continue
                # Check: is next non-space char a command keyword?
                # In interactive mode, . followed by a new command name is a terminator
                # Heuristic: if next non-space starts with a letter, it's a terminator
                # (because numbers after dot would be part of the number)
                if rest and rest[0].isalpha():
                    self._add_token(TokenType.TERMINATOR, ".")
                    self._pos += 1
                    self._col += 1
                    self._expecting_command = True
                    self._current_command = ""
                    self._in_expression = False
                    continue
                # Otherwise treat as part of something else (format spec dot like F8.2)
                # This shouldn't happen in normal flow since formats are in parens
                self._add_token(TokenType.TERMINATOR, ".")
                self._pos += 1
                self._col += 1
                self._expecting_command = True
                self._current_command = ""
                self._in_expression = False
                continue

            # String literals (single or double quoted)
            if ch == "'" or ch == '"':
                self._lex_string(ch)
                continue

            # Parentheses
            if ch == "(":
                self._add_token(TokenType.LPAREN, "(")
                self._pos += 1
                self._col += 1
                continue
            if ch == ")":
                self._add_token(TokenType.RPAREN, ")")
                self._pos += 1
                self._col += 1
                continue

            # Comma
            if ch == ",":
                self._add_token(TokenType.COMMA, ",")
                self._pos += 1
                self._col += 1
                continue

            # Operators
            if ch == "+":
                self._add_token(TokenType.PLUS, "+")
                self._pos += 1
                self._col += 1
                continue
            if ch == "-":
                # Could be minus operator or part of a number
                # If previous token is a number/variable/RPAREN, it's an operator
                # If we're at a position where a number is expected, check if it's a number
                if self._prev_token_is_operand():
                    self._add_token(TokenType.MINUS, "-")
                    self._pos += 1
                    self._col += 1
                    continue
                # Try to lex as number (e.g., -5)
                if self._pos + 1 < len(self._text) and self._text[self._pos + 1].isdigit():
                    self._lex_number()
                    continue
                self._add_token(TokenType.MINUS, "-")
                self._pos += 1
                self._col += 1
                continue
            if ch == "*":
                self._add_token(TokenType.STAR, "*")
                self._pos += 1
                self._col += 1
                continue
            if ch == "=":
                self._add_token(TokenType.EQUALS, "=")
                self._pos += 1
                self._col += 1
                self._expecting_command = False
                # If the current command is an expression command (COMPUTE, IF, etc.),
                # set expression context so / is treated as division
                if self._current_command in _EXPRESSION_COMMANDS:
                    self._in_expression = True
                continue
            # Two-char operators: <>, <=, >=, <, >
            if ch == "<":
                if self._pos + 1 < len(self._text) and self._text[self._pos + 1] == ">":
                    self._add_token(TokenType.LT_GT, "<>")
                    self._pos += 2
                    self._col += 2
                    continue
                if self._pos + 1 < len(self._text) and self._text[self._pos + 1] == "=":
                    self._add_token(TokenType.LE_OP, "<=")
                    self._pos += 2
                    self._col += 2
                    continue
                self._add_token(TokenType.LT_OP, "<")
                self._pos += 1
                self._col += 1
                continue
            if ch == ">":
                if self._pos + 1 < len(self._text) and self._text[self._pos + 1] == "=":
                    self._add_token(TokenType.GE_OP, ">=")
                    self._pos += 2
                    self._col += 2
                    continue
                self._add_token(TokenType.GT_OP, ">")
                self._pos += 1
                self._col += 1
                continue

            # Numbers
            if ch.isdigit():
                self._lex_number()
                continue

            # Identifiers (commands, keywords, variables)
            if _IDENT_START.match(ch):
                self._lex_identifier()
                continue

            # Unknown character — raise error
            raise SPSSSyntaxError(
                f"Unexpected character: {ch!r}",
                line=self._line,
            )

        self._add_token(TokenType.EOF, "", self._line, self._col)
        return self._tokens

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _add_token(self, ttype: TokenType, value: str,
                   line: Optional[int] = None,
                   col: Optional[int] = None) -> None:
        self._tokens.append(Token(
            type=ttype,
            value=value,
            line=line if line is not None else self._line,
            column=col if col is not None else self._col,
        ))

    def _peek_keyword(self) -> str:
        """Peek at the next word and return it uppercased (without consuming)."""
        start = self._pos
        # Skip optional leading spaces for keyword matching
        while start < len(self._text) and self._text[start] == " ":
            start += 1
        end = start
        while end < len(self._text) and _IDENT_CHAR.match(self._text[end]):
            end += 1
        word = self._text[start:end].upper()
        # Check for two-word keywords: BEGIN DATA, END DATA
        if word == "BEGIN":
            # Check if next word is DATA
            rest_start = end
            while rest_start < len(self._text) and self._text[rest_start] == " ":
                rest_start += 1
            rest = self._text[rest_start:rest_start + 4].upper()
            if rest == "DATA":
                return "BEGIN DATA"
        if word == "END":
            rest_start = end
            while rest_start < len(self._text) and self._text[rest_start] == " ":
                rest_start += 1
            rest = self._text[rest_start:rest_start + 4].upper()
            if rest == "DATA":
                return "END DATA"
        return word

    def _is_subcommand_context(self) -> bool:
        """Determine if / is a subcommand separator or division operator.

        In SPSS, / is a subcommand separator when not inside an expression.
        After = in COMPUTE/IF, / is division. Inside parens, / is division.
        Otherwise (command/subcommand context), / is a subcommand separator.
        """
        # If we're inside an expression context (COMPUTE target = expr),
        # / is division, not subcommand
        if self._in_expression:
            return False
        # Inside parens -> division
        paren_depth = sum(1 for t in self._tokens if t.type == TokenType.LPAREN) - \
                      sum(1 for t in self._tokens if t.type == TokenType.RPAREN)
        if paren_depth > 0:
            return False
        return True

    def _prev_token_is_operand(self) -> bool:
        """Check if the previous token is an operand (number, variable, RPAREN)."""
        if not self._tokens:
            return False
        prev = self._tokens[-1]
        return prev.type in (TokenType.NUMBER, TokenType.VARIABLE,
                             TokenType.RPAREN, TokenType.STRING)

    def _skip_comment_line(self) -> None:
        """Skip a comment line (from * or COMMENT to end of line)."""
        while self._pos < len(self._text) and self._text[self._pos] != "\n":
            self._pos += 1
            self._col += 1
        # The newline will be handled by the main loop

    def _lex_number(self) -> None:
        """Lex a numeric literal."""
        match = _NUMBER_RE.match(self._text[self._pos:])
        if match:
            value = match.group(0)
            self._add_token(TokenType.NUMBER, value)
            self._pos += match.end()
            self._col += match.end()
        else:
            # Fallback: just grab digits
            end = self._pos
            while end < len(self._text) and self._text[end].isdigit():
                end += 1
            value = self._text[self._pos:end]
            self._add_token(TokenType.NUMBER, value)
            self._col += (end - self._pos)
            self._pos = end

    def _lex_string(self, quote_char: str) -> None:
        """Lex a string literal, handling embedded quotes (doubled)."""
        start_line = self._line
        start_col = self._col
        self._pos += 1  # skip opening quote
        self._col += 1
        chars: list[str] = []
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch == quote_char:
                # Check for doubled quote (embedded)
                if self._pos + 1 < len(self._text) and self._text[self._pos + 1] == quote_char:
                    chars.append(quote_char)
                    self._pos += 2
                    self._col += 2
                    continue
                else:
                    # End of string
                    self._pos += 1
                    self._col += 1
                    break
            elif ch == "\n":
                # Strings can span lines in some contexts, but we'll preserve newline
                chars.append(ch)
                self._pos += 1
                self._line += 1
                self._col = 1
            else:
                chars.append(ch)
                self._pos += 1
                self._col += 1
        value = "".join(chars)
        self._add_token(TokenType.STRING, value, start_line, start_col)
        # After a string, we're not expecting a command
        self._expecting_command = False

    def _lex_identifier(self) -> None:
        """Lex an identifier: command, keyword, or variable name."""
        start = self._pos
        start_line = self._line
        start_col = self._col
        while self._pos < len(self._text) and _IDENT_CHAR.match(self._text[self._pos]):
            self._pos += 1
            self._col += 1
        word = self._text[start:self._pos]
        upper_word = word.upper()

        # Handle hyphenated command names: T-TEST, NPAR TESTS, etc.
        # When we're expecting a command and the next char is '-' followed by
        # an identifier character, it's part of the command name
        if self._expecting_command and self._pos < len(self._text) and \
                self._text[self._pos] == "-":
            # Look ahead: is there an identifier after the hyphen?
            next_pos = self._pos + 1
            if next_pos < len(self._text) and _IDENT_START.match(self._text[next_pos]):
                # Consume the hyphen and continue lexing
                self._pos += 1  # consume -
                self._col += 1
                while self._pos < len(self._text) and _IDENT_CHAR.match(self._text[self._pos]):
                    self._pos += 1
                    self._col += 1
                word = self._text[start:self._pos]
                upper_word = word.upper()

        # Check if this is a keyword operator (EQ, NE, GT, etc.)
        # But NOT when we're expecting a command (so MISSING VALUES etc. work)
        if upper_word in KEYWORD_OPERATORS and not self._expecting_command:
            self._add_token(KEYWORD_OPERATORS[upper_word], upper_word,
                            start_line, start_col)
            self._expecting_command = False
            return

        # Check if this is a reserved keyword (TO, ALL, BY, WITH, etc.)
        # But NOT when we're expecting a command (so MISSING VALUES etc. work)
        if upper_word in RESERVED_KEYWORDS and not self._expecting_command:
            self._add_token(TokenType.KEYWORD, upper_word, start_line, start_col)
            self._expecting_command = False
            return

        # If we're expecting a command name, this is a command
        if self._expecting_command:
            self._add_token(TokenType.COMMAND, upper_word, start_line, start_col)
            self._expecting_command = False
            # Track the current command for context-sensitive lexing
            self._current_command = upper_word
            return

        # Check if previous token was a subcommand slash
        if self._tokens and self._tokens[-1].type == TokenType.SUBCMD_SLASH:
            # In DATA LIST, the first token after / is a variable name,
            # not a subcommand name. DATA LIST uses / to separate record
            # groups, and the token after / is the first variable.
            if self._current_command == "DATA":
                self._add_token(TokenType.VARIABLE, word, start_line, start_col)
                return
            # In other commands, the token after / is a subcommand name
            self._add_token(TokenType.SUBCOMMAND, upper_word, start_line, start_col)
            return

        # Otherwise it's a variable name (case preserved)
        self._add_token(TokenType.VARIABLE, word, start_line, start_col)

    def _lex_data_block(self) -> None:
        """Lex a BEGIN DATA ... END DATA block as a single RAW_DATA token."""
        start_line = self._line
        start_col = self._col

        # Consume "BEGIN" keyword
        while self._pos < len(self._text) and _IDENT_CHAR.match(self._text[self._pos]):
            self._pos += 1
            self._col += 1

        # Skip spaces between BEGIN and DATA
        while self._pos < len(self._text) and self._text[self._pos] == " ":
            self._pos += 1
            self._col += 1

        # Consume "DATA" keyword
        while self._pos < len(self._text) and _IDENT_CHAR.match(self._text[self._pos]):
            self._pos += 1
            self._col += 1

        # Skip optional period terminator after BEGIN DATA
        # (CSR says: "It is best to omit the terminator on BEGIN DATA")
        # Skip to end of line (rest of BEGIN DATA line)
        while self._pos < len(self._text) and self._text[self._pos] != "\n":
            self._pos += 1
        if self._pos < len(self._text) and self._text[self._pos] == "\n":
            self._line += 1
            self._col = 1
            self._pos += 1

        # Now collect lines until we find END DATA
        data_lines: list[str] = []
        while self._pos < len(self._text):
            # Check if this line starts with "END DATA"
            # Skip leading whitespace
            check_pos = self._pos
            while check_pos < len(self._text) and self._text[check_pos] in " \t":
                check_pos += 1
            # Check for "END" keyword
            word_end = check_pos
            while word_end < len(self._text) and _IDENT_CHAR.match(self._text[word_end]):
                word_end += 1
            word = self._text[check_pos:word_end].upper()
            if word == "END":
                # Check if next word is DATA
                ws_pos = word_end
                while ws_pos < len(self._text) and self._text[ws_pos] == " ":
                    ws_pos += 1
                data_end = ws_pos
                while data_end < len(self._text) and _IDENT_CHAR.match(self._text[data_end]):
                    data_end += 1
                if self._text[ws_pos:data_end].upper() == "DATA":
                    # Found END DATA — consume to end of line
                    self._pos = data_end
                    self._col += (data_end - self._pos)
                    # Skip rest of line
                    while self._pos < len(self._text) and self._text[self._pos] != "\n":
                        self._pos += 1
                    if self._pos < len(self._text) and self._text[self._pos] == "\n":
                        self._line += 1
                        self._col = 1
                        self._pos += 1
                    break

            # Collect this data line
            line_end = self._pos
            while line_end < len(self._text) and self._text[line_end] != "\n":
                line_end += 1
            data_lines.append(self._text[self._pos:line_end])
            if line_end < len(self._text) and self._text[line_end] == "\n":
                self._line += 1
                self._col = 1
            self._pos = line_end
            if self._pos < len(self._text) and self._text[self._pos] == "\n":
                self._pos += 1

        raw_data = "\n".join(data_lines)
        self._add_token(TokenType.RAW_DATA, raw_data, start_line, start_col)
        self._expecting_command = True