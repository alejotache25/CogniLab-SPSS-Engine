"""
Recursive-descent parser for SPSS syntax.

Consumes tokens from the Lexer and produces a list of CommandNode objects.

Grammar (simplified):

  program     := command*
  command     := COMMAND command_body TERMINATOR
  command_body := subcommand* (varlist | expression)
  subcommand  := SUBCMD_SLASH SUBCOMMAND subcommand_body
  subcommand_body := EQUALS? (varlist | keyword_list | expression)

  expression  := or_expr
  or_expr     := and_expr (OR and_expr)*
  and_expr    := not_expr (AND not_expr)*
  not_expr    := NOT? rel_expr
  rel_expr    := add_expr ((EQ|NE|GT|LT|GE|LE|<>|<=|>=|<|>) add_expr)?
  add_expr    := mul_expr ((PLUS|MINUS) mul_expr)*
  mul_expr    := unary ((STAR|SLASH) unary)*
  unary       := MINUS? primary
  primary     := NUMBER | STRING | VARIABLE | function_call | LPAREN expression RPAREN
  function_call := VARIABLE (DOT NUMBER)? LPAREN arg_list RPAREN
  arg_list    := expression (COMMA expression)*
"""

from __future__ import annotations
from typing import List, Optional, Tuple

from spss_engine.parser.tokens import Token, TokenType
from spss_engine.parser.lexer import Lexer
from spss_engine.parser.ast_nodes import (
    CommandNode, SubcommandNode, VarDefNode,
    NumberNode, StringNode, VariableNode, VarRangeNode,
    BinaryOpNode, UnaryOpNode, FunctionCallNode,
    ExpressionNode,
)
from spss_engine.utils.errors import SPSSSyntaxError


# Commands that have special parsing (not standard subcommand form)
_SPECIAL_COMMANDS: set[str] = {"DATA LIST", "BEGIN DATA", "END DATA", "COMPUTE",
                                "RECODE", "SELECT IF", "IF", "DO IF", "ELSE",
                                "END IF", "MISSING VALUES", "VARIABLE LABELS",
                                "VALUE LABELS", "STRING", "FORMATS"}

# Two-word commands
_TWO_WORD_COMMANDS: dict[str, str] = {
    "DATA LIST": "DATA LIST",
    "SELECT IF": "SELECT IF",
    "DO IF": "DO IF",
    "END IF": "END IF",
    "END DATA": "END DATA",
    "SPLIT FILE": "SPLIT FILE",
    "SORT CASES": "SORT CASES",
    "MISSING VALUES": "MISSING VALUES",
    "VARIABLE LABELS": "VARIABLE LABELS",
    "VALUE LABELS": "VALUE LABELS",
    "NPAR TESTS": "NPAR TESTS",
}

# Relational operators mapping
_REL_OPS: dict[TokenType, str] = {
    TokenType.OP_EQ: "=",
    TokenType.OP_NE: "<>",
    TokenType.OP_GT: ">",
    TokenType.OP_LT: "<",
    TokenType.OP_GE: ">=",
    TokenType.OP_LE: "<=",
    TokenType.LT_GT: "<>",
    TokenType.LE_OP: "<=",
    TokenType.GE_OP: ">=",
    TokenType.LT_OP: "<",
    TokenType.GT_OP: ">",
    TokenType.EQUALS: "=",
}


class Parser:
    """Recursive-descent parser producing a list of CommandNode objects."""

    def __init__(self) -> None:
        self._tokens: List[Token] = []
        self._pos: int = 0

    def parse(self, text: str) -> List[CommandNode]:
        """Parse SPSS syntax text into a list of CommandNode objects."""
        lexer = Lexer(text)
        self._tokens = lexer.tokenize()
        self._pos = 0
        commands: List[CommandNode] = []
        while self._peek().type != TokenType.EOF:
            cmd = self._parse_command()
            if cmd is not None:
                commands.append(cmd)
        return commands

    def parse_tokens(self, tokens: List[Token]) -> List[CommandNode]:
        """Parse a pre-tokenized list of tokens."""
        self._tokens = tokens
        self._pos = 0
        commands: List[CommandNode] = []
        while self._peek().type != TokenType.EOF:
            cmd = self._parse_command()
            if cmd is not None:
                commands.append(cmd)
        return commands

    # ------------------------------------------------------------------
    # Token navigation helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1] if self._tokens else Token(TokenType.EOF, "", 0, 0)
        return self._tokens[idx]

    def _advance(self) -> Token:
        tok = self._peek()
        if self._pos < len(self._tokens):
            self._pos += 1
        return tok

    def _expect(self, ttype: TokenType, context: str = "") -> Token:
        tok = self._peek()
        if tok.type != ttype:
            ctx = f" ({context})" if context else ""
            raise SPSSSyntaxError(
                f"Expected {ttype.name} but got {tok.type.name} ({tok.value!r}){ctx}",
                line=tok.line,
            )
        return self._advance()

    def _match(self, *types: TokenType) -> bool:
        return self._peek().type in types

    def _match_kw(self, keyword: str) -> bool:
        tok = self._peek()
        return tok.type == TokenType.KEYWORD and tok.value == keyword.upper()

    def _match_command(self, name: str) -> bool:
        tok = self._peek()
        return tok.type == TokenType.COMMAND and tok.value == name.upper()

    # ------------------------------------------------------------------
    # Command parsing
    # ------------------------------------------------------------------

    def _parse_command(self) -> Optional[CommandNode]:
        """Parse a single command."""
        tok = self._peek()

        # Skip stray terminators
        if tok.type == TokenType.TERMINATOR:
            self._advance()
            return None

        # Raw data block (should be consumed by DATA LIST, but handle stray)
        if tok.type == TokenType.RAW_DATA:
            self._advance()
            return None

        if tok.type != TokenType.COMMAND:
            raise SPSSSyntaxError(
                f"Expected command name but got {tok.type.name} ({tok.value!r})",
                line=tok.line,
            )

        cmd_name = tok.value
        line = tok.line
        self._advance()

        # Handle two-word commands where second word is a VARIABLE token
        # DATA LIST, MISSING VALUES, VARIABLE LABELS, VALUE LABELS,
        # SELECT IF, SPLIT FILE, SORT CASES
        next_tok = self._peek()
        if next_tok.type == TokenType.VARIABLE:
            two_word = f"{cmd_name} {next_tok.value.upper()}"
            if two_word in _TWO_WORD_COMMANDS:
                self._advance()  # consume second word
                cmd_name = two_word
                if cmd_name == "MISSING VALUES":
                    return self._parse_missing_values(line)
                if cmd_name == "VARIABLE LABELS":
                    return self._parse_variable_labels(line)
                if cmd_name == "VALUE LABELS":
                    return self._parse_value_labels(line)
                if cmd_name == "SELECT IF":
                    return self._parse_select_if(line)
                if cmd_name == "SPLIT FILE":
                    return self._parse_split_file(line)
                if cmd_name == "SORT CASES":
                    return self._parse_sort_cases(line)

        if cmd_name == "DATA":
            # DATA LIST — "LIST" was already consumed as second word above
            # But if LIST was consumed via two_word, we already returned
            # If we're here, it means "DATA" was followed by something else
            # or "DATA LIST" wasn't in _TWO_WORD_COMMANDS...
            # Actually "DATA LIST" IS in _TWO_WORD_COMMANDS, so we should have
            # returned already. But "LIST" is tokenized as VARIABLE...
            # Let me check: "DATA LIST" → COMMAND(DATA) VARIABLE(LIST)
            # two_word = "DATA LIST" which is in _TWO_WORD_COMMANDS
            # So we should have consumed LIST and returned _parse_data_list
            # But we didn't return because... let me check the code flow
            # Actually, we did consume it and set cmd_name = "DATA LIST"
            # but we didn't call _parse_data_list!
            # Let me fix: check for "DATA LIST" before the two_word check
            pass

        if cmd_name == "DATA LIST":
            return self._parse_data_list(line)

        if cmd_name == "COMPUTE":
            return self._parse_compute(line)
        if cmd_name == "RECODE":
            return self._parse_recode(line)
        if cmd_name == "STRING":
            return self._parse_string_cmd(line)
        if cmd_name == "FORMATS":
            return self._parse_formats(line)
        if cmd_name == "FILTER":
            return self._parse_filter(line)
        if cmd_name == "WEIGHT":
            return self._parse_weight(line)

        # Default: generic command with subcommands
        return self._parse_generic_command(cmd_name, line)

    def _match_command_or_kw(self, name: str) -> bool:
        """Check if next token matches a command or keyword with given name."""
        tok = self._peek()
        return tok.value.upper() == name.upper() and tok.type in (
            TokenType.COMMAND, TokenType.VARIABLE, TokenType.KEYWORD
        )

    # ------------------------------------------------------------------
    # DATA LIST command
    # ------------------------------------------------------------------

    def _parse_data_list(self, line: int) -> CommandNode:
        """Parse DATA LIST command."""
        cmd = CommandNode(name="DATA LIST", line=line)
        cmd.data_format = "FIXED"  # default

        # Parse optional subcommands before the first slash
        while not self._match(TokenType.SUBCMD_SLASH):
            tok = self._peek()
            if tok.type == TokenType.TERMINATOR:
                self._advance()
                # Check for following RAW_DATA token
                if self._peek().type == TokenType.RAW_DATA:
                    cmd.raw_data = self._advance().value
                return cmd
            if tok.type == TokenType.EOF:
                return cmd
            if tok.type == TokenType.RAW_DATA:
                cmd.raw_data = self._advance().value
                continue

            # Parse optional keywords: FIXED, FREE, LIST, FILE=, RECORDS=, etc.
            upper = tok.value.upper()
            if upper in ("FIXED", "FREE", "LIST"):
                cmd.data_format = upper
                self._advance()
                continue
            if upper == "FILE":
                self._advance()
                if self._match(TokenType.EQUALS):
                    self._advance()
                if self._peek().type == TokenType.STRING:
                    cmd.file_path = self._advance().value
                continue
            if upper == "RECORDS":
                self._advance()
                if self._match(TokenType.EQUALS):
                    self._advance()
                if self._peek().type == TokenType.NUMBER:
                    self._advance()  # skip records count
                continue
            # Unknown pre-slash token, skip it
            self._advance()

        # Parse variable definitions (after first /)
        while self._match(TokenType.SUBCMD_SLASH):
            self._advance()  # consume /
            # Optional record number (e.g., /1 or /2)
            if self._peek().type == TokenType.NUMBER:
                self._advance()  # skip record number
            # Parse variable definitions until terminator or next /
            self._parse_var_defs(cmd)

        # Consume terminator
        if self._match(TokenType.TERMINATOR):
            self._advance()

        # Check for following RAW_DATA token
        if self._peek().type == TokenType.RAW_DATA:
            cmd.raw_data = self._advance().value

        return cmd

    def _parse_var_defs(self, cmd: CommandNode) -> None:
        """Parse variable definitions in DATA LIST."""
        while not self._match(TokenType.TERMINATOR, TokenType.SUBCMD_SLASH,
                               TokenType.EOF):
            tok = self._peek()
            if tok.type == TokenType.RAW_DATA:
                cmd.raw_data = self._advance().value
                continue
            if tok.type == TokenType.VARIABLE:
                var_name = self._advance().value
                # Check for TO keyword: OPINION1 TO OPINION5
                if self._match_kw("TO"):
                    self._advance()  # consume TO
                    if self._peek().type == TokenType.VARIABLE:
                        end_var = self._advance().value
                        # Parse column range for TO group
                        start_col, end_col = self._parse_col_range_optional()
                        fmt_type, fmt_spec, width = self._parse_format_optional()
                        # Expand TO variables
                        self._expand_to_vars(cmd, var_name, end_var,
                                            start_col, end_col,
                                            fmt_type, fmt_spec, width)
                        continue
                # Regular variable definition
                start_col, end_col = self._parse_col_range_optional()
                fmt_type, fmt_spec, width = self._parse_format_optional()
                vd = VarDefNode(
                    name=var_name,
                    start_col=start_col,
                    end_col=end_col,
                    var_type=fmt_type if fmt_type else "numeric",
                    format_spec=fmt_spec,
                    width=width,
                )
                cmd.var_defs.append(vd)
            else:
                # Unexpected token in var defs, skip
                self._advance()

    def _parse_col_range_optional(self) -> Tuple[Optional[int], Optional[int]]:
        """Parse optional column range (e.g., 1-3 or 5)."""
        tok = self._peek()
        if tok.type == TokenType.NUMBER:
            start = int(float(tok.value))
            self._advance()
            # Check for range: NUMBER MINUS NUMBER (1-3)
            if self._match(TokenType.MINUS):
                self._advance()
                if self._peek().type == TokenType.NUMBER:
                    end = int(float(self._advance().value))
                    return start, end
            # Single column
            return start, start
        return None, None

    def _parse_format_optional(self) -> Tuple[str, Optional[str], int]:
        """Parse optional format spec in parens: (A), (A10), (F8.2), etc.

        Returns (var_type, format_spec, width).
        """
        if not self._match(TokenType.LPAREN):
            return "numeric", None, 8
        self._advance()  # consume (
        # The format spec is typically a VARIABLE-like token or a sequence
        # e.g., A, A10, F8.2, etc.
        # In our lexer, A would be tokenized as VARIABLE "A"
        # But F8.2 might be tokenized differently — let's handle it
        fmt_parts: list[str] = []
        while not self._match(TokenType.RPAREN):
            tok = self._peek()
            if tok.type in (TokenType.VARIABLE, TokenType.NUMBER,
                            TokenType.COMMAND, TokenType.KEYWORD,
                            TokenType.SUBCOMMAND):
                fmt_parts.append(tok.value)
                self._advance()
            elif tok.type == TokenType.MINUS:
                # Could be part of format (rare)
                self._advance()
            elif tok.type == TokenType.STAR:
                fmt_parts.append("*")
                self._advance()
            else:
                break
        if self._match(TokenType.RPAREN):
            self._advance()
        fmt_str = "".join(fmt_parts).strip()
        # Parse format string
        if fmt_str.upper().startswith("A"):
            # String format: A or A10
            width_str = fmt_str[1:]
            width = int(width_str) if width_str.isdigit() else 1
            return "string", fmt_str, width
        elif fmt_str.upper().startswith("F") or fmt_str.upper().startswith("N"):
            # Numeric format: F8.2, N3, etc.
            return "numeric", fmt_str, 8
        else:
            return "numeric", fmt_str, 8

    def _expand_to_vars(self, cmd: CommandNode, start_name: str,
                        end_name: str,
                        start_col: Optional[int], end_col: Optional[int],
                        fmt_type: str, fmt_spec: Optional[str],
                        width: int) -> None:
        """Expand OPINION1 TO OPINION5 into individual VarDefNodes."""
        # Extract prefix and start/end numbers
        import re
        m1 = re.match(r"^(.*?)(\d+)$", start_name)
        m2 = re.match(r"^(.*?)(\d+)$", end_name)
        if not m1 or not m2:
            # Can't expand, just add both as-is
            cmd.var_defs.append(VarDefNode(
                name=start_name, start_col=start_col, end_col=end_col,
                var_type=fmt_type, format_spec=fmt_spec, width=width))
            cmd.var_defs.append(VarDefNode(
                name=end_name, start_col=start_col, end_col=end_col,
                var_type=fmt_type, format_spec=fmt_spec, width=width))
            return
        prefix = m1.group(1)
        n1 = int(m1.group(2))
        n2 = int(m2.group(2))
        # Compute column width per variable
        if start_col is not None and end_col is not None:
            total_cols = end_col - start_col + 1
            per_var = total_cols // (n2 - n1 + 1)
        else:
            per_var = 1
            start_col = 0
        for i, n in enumerate(range(n1, n2 + 1)):
            var_name = f"{prefix}{n}"
            col_start = start_col + i * per_var if start_col else None
            col_end = (start_col + (i + 1) * per_var - 1) if start_col else None
            cmd.var_defs.append(VarDefNode(
                name=var_name,
                start_col=col_start,
                end_col=col_end,
                var_type=fmt_type,
                format_spec=fmt_spec,
                width=width,
            ))

    # ------------------------------------------------------------------
    # COMPUTE command
    # ------------------------------------------------------------------

    def _parse_compute(self, line: int) -> CommandNode:
        """Parse COMPUTE target = expression."""
        cmd = CommandNode(name="COMPUTE", line=line)

        # Target variable
        target_tok = self._peek()
        if target_tok.type != TokenType.VARIABLE:
            raise SPSSSyntaxError(
                f"COMPUTE expects target variable, got {target_tok.type.name}",
                line=target_tok.line,
            )
        cmd.target_var = self._advance().value

        # Equals sign
        self._expect(TokenType.EQUALS, "COMPUTE")

        # Expression
        cmd.target_expression = self._parse_expression()

        # Terminator
        if self._match(TokenType.TERMINATOR):
            self._advance()

        return cmd

    # ------------------------------------------------------------------
    # RECODE command (basic parsing for Phase 1)
    # ------------------------------------------------------------------

    def _parse_recode(self, line: int) -> CommandNode:
        """Parse RECODE varlist (val1=val2) ... INTO newvar."""
        cmd = CommandNode(name="RECODE", line=line)

        # Variable list (until first LPAREN)
        while not self._match(TokenType.LPAREN, TokenType.TERMINATOR,
                               TokenType.EOF):
            tok = self._peek()
            if tok.type == TokenType.VARIABLE:
                # Check for TO
                if self._match_kw("TO") and self._peek(1).type == TokenType.VARIABLE:
                    self._advance()  # consume var
                    self._advance()  # consume TO
                    end_var = self._advance().value
                    sc = SubcommandNode(name="VARS",
                                       variables=[tok.value, "TO", end_var])
                    cmd.subcommands.append(sc)
                else:
                    sc = SubcommandNode(name="VARS", variables=[tok.value])
                    cmd.subcommands.append(sc)
                    self._advance()
            elif self._match_kw("INTO"):
                self._advance()
                if self._peek().type == TokenType.VARIABLE:
                    cmd.target_var = self._advance().value
            else:
                self._advance()

        # Parse recode groups (parenthesized value mappings)
        groups: list[list[str]] = []
        while self._match(TokenType.LPAREN):
            self._advance()
            group: list[str] = []
            while not self._match(TokenType.RPAREN, TokenType.TERMINATOR,
                                   TokenType.EOF):
                tok = self._peek()
                if tok.type == TokenType.EQUALS:
                    self._advance()
                    group.append("=")
                elif tok.type == TokenType.VARIABLE:
                    group.append(tok.value)
                    self._advance()
                elif tok.type == TokenType.NUMBER:
                    group.append(tok.value)
                    self._advance()
                elif tok.type == TokenType.STRING:
                    group.append(tok.value)
                    self._advance()
                elif tok.type == TokenType.KEYWORD:
                    group.append(tok.value)
                    self._advance()
                elif tok.type == TokenType.PLUS or tok.type == TokenType.MINUS:
                    group.append(tok.value)
                    self._advance()
                else:
                    self._advance()
            if self._match(TokenType.RPAREN):
                self._advance()
            groups.append(group)

        # Store groups in raw_tokens
        cmd.subcommands.append(SubcommandNode(name="GROUPS", raw_tokens=groups))

        # Check for INTO newvar (after groups)
        if self._match_kw("INTO"):
            self._advance()
            if self._peek().type == TokenType.VARIABLE:
                cmd.target_var = self._advance().value

        if self._match(TokenType.TERMINATOR):
            self._advance()

        return cmd

    # ------------------------------------------------------------------
    # SELECT IF command
    # ------------------------------------------------------------------

    def _parse_select_if(self, line: int) -> CommandNode:
        """Parse SELECT IF expression."""
        cmd = CommandNode(name="SELECT IF", line=line)
        # Optional parentheses around condition
        had_paren = False
        if self._match(TokenType.LPAREN):
            self._advance()
            had_paren = True
        sc = SubcommandNode(name="CONDITION")
        sc.expression = self._parse_expression()
        cmd.subcommands.append(sc)
        if had_paren and self._match(TokenType.RPAREN):
            self._advance()
        if self._match(TokenType.TERMINATOR):
            self._advance()
        return cmd

    # ------------------------------------------------------------------
    # MISSING VALUES command (immediate)
    # ------------------------------------------------------------------

    def _parse_missing_values(self, line: int) -> CommandNode:
        """Parse MISSING VALUES varlist (values)."""
        cmd = CommandNode(name="MISSING VALUES", line=line)

        # Variables until LPAREN
        vars_list: list[str] = []
        while not self._match(TokenType.LPAREN, TokenType.TERMINATOR,
                               TokenType.EOF):
            tok = self._peek()
            if tok.type == TokenType.VARIABLE:
                vars_list.append(tok.value)
                self._advance()
                # Check for TO
                if self._match_kw("TO"):
                    self._advance()
                    if self._peek().type == TokenType.VARIABLE:
                        vars_list.append("TO")
                        vars_list.append(self._advance().value)
            elif tok.type == TokenType.KEYWORD and tok.value == "ALL":
                vars_list.append("ALL")
                self._advance()
            else:
                self._advance()

        # Parse missing value spec in parens
        missing_vals: list[str] = []
        if self._match(TokenType.LPAREN):
            self._advance()
            while not self._match(TokenType.RPAREN, TokenType.EOF):
                tok = self._peek()
                if tok.type == TokenType.NUMBER:
                    missing_vals.append(tok.value)
                    self._advance()
                elif tok.type == TokenType.STRING:
                    missing_vals.append(tok.value)
                    self._advance()
                elif tok.type == TokenType.COMMA:
                    self._advance()
                elif tok.type == TokenType.KEYWORD:
                    missing_vals.append(tok.value)
                    self._advance()
                else:
                    self._advance()
            if self._match(TokenType.RPAREN):
                self._advance()

        sc = SubcommandNode(name="VALUES", variables=vars_list,
                           raw_tokens=missing_vals)
        cmd.subcommands.append(sc)

        if self._match(TokenType.TERMINATOR):
            self._advance()
        return cmd

    # ------------------------------------------------------------------
    # VARIABLE LABELS command (immediate)
    # ------------------------------------------------------------------

    def _parse_variable_labels(self, line: int) -> CommandNode:
        """Parse VARIABLE LABELS var 'label' var 'label' ..."""
        cmd = CommandNode(name="VARIABLE LABELS", line=line)

        while not self._match(TokenType.TERMINATOR, TokenType.EOF):
            tok = self._peek()
            if tok.type == TokenType.VARIABLE:
                var_name = self._advance().value
                # Next should be a string label
                if self._peek().type == TokenType.STRING:
                    label = self._advance().value
                    sc = SubcommandNode(name="LABEL",
                                       variables=[var_name],
                                       raw_tokens=[label])
                    cmd.subcommands.append(sc)
                else:
                    # Unexpected, skip
                    self._advance()
            else:
                self._advance()

        if self._match(TokenType.TERMINATOR):
            self._advance()
        return cmd

    # ------------------------------------------------------------------
    # VALUE LABELS command (immediate)
    # ------------------------------------------------------------------

    def _parse_value_labels(self, line: int) -> CommandNode:
        """Parse VALUE LABELS varlist val 'label' val 'label' ..."""
        cmd = CommandNode(name="VALUE LABELS", line=line)

        # Variables until we hit a NUMBER or STRING (value spec)
        vars_list: list[str] = []
        while not self._match(TokenType.TERMINATOR, TokenType.EOF):
            tok = self._peek()
            if tok.type == TokenType.VARIABLE:
                # Could be variable or start of value specs
                # Heuristic: if next token is a NUMBER or STRING, this var ends vars list
                next_tok = self._peek(1)
                if next_tok.type == TokenType.STRING:
                    # This is the last variable, followed by value-label pairs
                    vars_list.append(self._advance().value)
                    break
                if next_tok.type == TokenType.NUMBER:
                    vars_list.append(self._advance().value)
                    break
                vars_list.append(self._advance().value)
            elif tok.type == TokenType.KEYWORD and tok.value == "ALL":
                vars_list.append("ALL")
                self._advance()
                break
            else:
                break

        # Parse value-label pairs
        pairs: list[tuple[str, str]] = []
        while not self._match(TokenType.TERMINATOR, TokenType.EOF):
            tok = self._peek()
            if tok.type == TokenType.NUMBER:
                val = self._advance().value
                if self._peek().type == TokenType.STRING:
                    label = self._advance().value
                    pairs.append((val, label))
            elif tok.type == TokenType.STRING:
                val = self._advance().value
                if self._peek().type == TokenType.STRING:
                    label = self._advance().value
                    pairs.append((val, label))
            else:
                self._advance()

        sc = SubcommandNode(name="LABELS", variables=vars_list,
                           raw_tokens=pairs)
        cmd.subcommands.append(sc)

        if self._match(TokenType.TERMINATOR):
            self._advance()
        return cmd

    # ------------------------------------------------------------------
    # STRING command (declare string variable)
    # ------------------------------------------------------------------

    def _parse_string_cmd(self, line: int) -> CommandNode:
        """Parse STRING varname (A10) varname (A20) ..."""
        cmd = CommandNode(name="STRING", line=line)

        while not self._match(TokenType.TERMINATOR, TokenType.EOF):
            if self._peek().type == TokenType.VARIABLE:
                var_name = self._advance().value
                fmt_type, fmt_spec, width = self._parse_format_optional()
                vd = VarDefNode(name=var_name, var_type="string",
                               format_spec=fmt_spec, width=width)
                cmd.var_defs.append(vd)
            else:
                self._advance()

        if self._match(TokenType.TERMINATOR):
            self._advance()
        return cmd

    # ------------------------------------------------------------------
    # FORMATS command (immediate)
    # ------------------------------------------------------------------

    def _parse_formats(self, line: int) -> CommandNode:
        """Parse FORMATS var (format) var (format) ..."""
        cmd = CommandNode(name="FORMATS", line=line)

        while not self._match(TokenType.TERMINATOR, TokenType.EOF):
            if self._peek().type == TokenType.VARIABLE:
                var_name = self._advance().value
                fmt_type, fmt_spec, width = self._parse_format_optional()
                sc = SubcommandNode(name="FORMAT",
                                   variables=[var_name],
                                   raw_tokens=[fmt_spec or "F8.2"])
                cmd.subcommands.append(sc)
            else:
                self._advance()

        if self._match(TokenType.TERMINATOR):
            self._advance()
        return cmd

    # ------------------------------------------------------------------
    # FILTER command (immediate)
    # ------------------------------------------------------------------

    def _parse_filter(self, line: int) -> CommandNode:
        """Parse FILTER BY var / FILTER OFF."""
        cmd = CommandNode(name="FILTER", line=line)

        if self._match_kw("OFF"):
            self._advance()
            cmd.subcommands.append(SubcommandNode(name="OFF"))
        elif self._match_kw("BY"):
            self._advance()
            if self._peek().type == TokenType.VARIABLE:
                var = self._advance().value
                cmd.subcommands.append(SubcommandNode(name="BY",
                                                    variables=[var]))
        else:
            # FILTER var. (BY is optional)
            if self._peek().type == TokenType.VARIABLE:
                var = self._advance().value
                cmd.subcommands.append(SubcommandNode(name="BY",
                                                    variables=[var]))

        if self._match(TokenType.TERMINATOR):
            self._advance()
        return cmd

    # ------------------------------------------------------------------
    # WEIGHT command (immediate)
    # ------------------------------------------------------------------

    def _parse_weight(self, line: int) -> CommandNode:
        """Parse WEIGHT BY var / WEIGHT OFF."""
        cmd = CommandNode(name="WEIGHT", line=line)

        if self._match_kw("OFF"):
            self._advance()
            cmd.subcommands.append(SubcommandNode(name="OFF"))
        elif self._match_kw("BY"):
            self._advance()
            if self._peek().type == TokenType.VARIABLE:
                var = self._advance().value
                cmd.subcommands.append(SubcommandNode(name="BY",
                                                    variables=[var]))

        if self._match(TokenType.TERMINATOR):
            self._advance()
        return cmd

    # ------------------------------------------------------------------
    # SPLIT FILE command (immediate)
    # ------------------------------------------------------------------

    def _parse_split_file(self, line: int) -> CommandNode:
        """Parse SPLIT FILE BY vars / SPLIT FILE OFF / SPLIT FILE LAYERED BY vars."""
        cmd = CommandNode(name="SPLIT FILE", line=line)

        if self._match_kw("OFF"):
            self._advance()
            cmd.subcommands.append(SubcommandNode(name="OFF"))
        else:
            # LAYERED keyword?
            layered = False
            if self._peek().type == TokenType.VARIABLE and \
               self._peek().value.upper() == "LAYERED":
                layered = True
                self._advance()
            if self._match_kw("BY"):
                self._advance()
            vars_list: list[str] = []
            while self._peek().type == TokenType.VARIABLE:
                vars_list.append(self._advance().value)
            sc = SubcommandNode(name="BY", variables=vars_list)
            if layered:
                sc.raw_tokens = ["LAYERED"]
            cmd.subcommands.append(sc)

        if self._match(TokenType.TERMINATOR):
            self._advance()
        return cmd

    # ------------------------------------------------------------------
    # SORT CASES command (reads data)
    # ------------------------------------------------------------------

    def _parse_sort_cases(self, line: int) -> CommandNode:
        """Parse SORT CASES BY var (A) var (D) ..."""
        cmd = CommandNode(name="SORT CASES", line=line)

        if self._match_kw("BY"):
            self._advance()
        vars_list: list[str] = []
        directions: list[str] = []
        while not self._match(TokenType.TERMINATOR, TokenType.EOF):
            tok = self._peek()
            if tok.type == TokenType.VARIABLE:
                vars_list.append(self._advance().value)
            elif tok.type == TokenType.LPAREN:
                self._advance()
                if self._peek().type in (TokenType.VARIABLE, TokenType.COMMAND,
                                          TokenType.KEYWORD):
                    d = self._advance().value.upper()
                    directions.append(d)
                if self._match(TokenType.RPAREN):
                    self._advance()
            else:
                self._advance()

        sc = SubcommandNode(name="BY", variables=vars_list,
                           raw_tokens=directions)
        cmd.subcommands.append(sc)

        if self._match(TokenType.TERMINATOR):
            self._advance()
        return cmd

    # ------------------------------------------------------------------
    # Generic command parser (for FREQUENCIES, DESCRIPTIVES, CROSSTABS, etc.)
    # ------------------------------------------------------------------

    def _parse_generic_command(self, cmd_name: str, line: int) -> CommandNode:
        """Parse a generic command with subcommands.

        Format: COMMAND [VARIABLES=varlist] /SUBCMD=specs ... .
        Also handles: COMMAND varlist /SUBCMD=specs ...
        """
        cmd = CommandNode(name=cmd_name, line=line)

        # Check if the main spec is a subcommand with = sign
        # Pattern: VARIABLE EQUALS varlist (e.g., VARIABLES=AGE SEX)
        if (self._peek().type == TokenType.VARIABLE and
                self._peek(1).type == TokenType.EQUALS):
            sub_name = self._advance().value
            self._advance()  # consume =
            sc = SubcommandNode(name=sub_name.upper())
            self._parse_subcommand_body(sc)
            cmd.subcommands.append(sc)
        else:
            # Parse main variable list or keyword spec (before first /)
            main_sc = SubcommandNode(name="_MAIN")
            self._parse_subcommand_body(main_sc)
            if main_sc.variables or main_sc.keywords or main_sc.raw_tokens:
                cmd.subcommands.append(main_sc)

        # Parse subcommands (after /)
        while self._match(TokenType.SUBCMD_SLASH):
            self._advance()
            if self._peek().type in (TokenType.SUBCOMMAND, TokenType.VARIABLE,
                                      TokenType.KEYWORD):
                sub_name = self._advance().value
                sc = SubcommandNode(name=sub_name.upper())
                # Optional equals
                if self._match(TokenType.EQUALS):
                    self._advance()
                self._parse_subcommand_body(sc)
                cmd.subcommands.append(sc)
            else:
                # Empty subcommand or unexpected token
                self._advance()

        # Consume terminator
        if self._match(TokenType.TERMINATOR):
            self._advance()

        return cmd

    def _parse_subcommand_body(self, sc: SubcommandNode) -> None:
        """Parse the body of a subcommand (variables, keywords, expressions)."""
        while not self._match(TokenType.SUBCMD_SLASH, TokenType.TERMINATOR,
                               TokenType.EOF):
            tok = self._peek()
            if tok.type == TokenType.VARIABLE:
                # Check for TO keyword
                if self._match_kw("TO") and self._peek(1).type == TokenType.VARIABLE:
                    self._advance()
                    self._advance()
                    end_var = self._advance().value
                    sc.variables.extend([tok.value, "TO", end_var])
                else:
                    val = self._advance().value
                    sc.variables.append(val)
                    # Also add to keywords for subcommands like STATISTICS=MEAN STDDEV
                    # where MEAN is tokenized as VARIABLE but is really a keyword
                    sc.keywords.append(val.upper())
            elif tok.type == TokenType.KEYWORD:
                if tok.value == "ALL":
                    sc.variables.append("ALL")
                else:
                    sc.keywords.append(tok.value)
                self._advance()
            elif tok.type == TokenType.NUMBER:
                sc.keywords.append(tok.value)
                self._advance()
            elif tok.type == TokenType.STRING:
                sc.keywords.append(tok.value)
                self._advance()
            elif tok.type == TokenType.LPAREN:
                # Parenthesized spec
                self._advance()
                paren_items: list[str] = []
                while not self._match(TokenType.RPAREN, TokenType.EOF):
                    paren_items.append(self._peek().value)
                    self._advance()
                if self._match(TokenType.RPAREN):
                    self._advance()
                sc.raw_tokens.extend(paren_items)
            elif tok.type == TokenType.EQUALS:
                self._advance()
            elif tok.type == TokenType.COMMA:
                self._advance()
            else:
                self._advance()

    # ------------------------------------------------------------------
    # Expression parser (recursive descent)
    # ------------------------------------------------------------------

    def _parse_expression(self) -> ExpressionNode:
        """Parse a full expression (OR level)."""
        return self._parse_or()

    def _parse_or(self) -> ExpressionNode:
        left = self._parse_and()
        while self._match(TokenType.OP_OR):
            op = self._advance().value
            right = self._parse_and()
            left = BinaryOpNode(op=op, left=left, right=right,
                                line=left.line if hasattr(left, "line") else 0)
        return left

    def _parse_and(self) -> ExpressionNode:
        left = self._parse_not()
        while self._match(TokenType.OP_AND):
            op = self._advance().value
            right = self._parse_not()
            left = BinaryOpNode(op=op, left=left, right=right,
                                line=left.line if hasattr(left, "line") else 0)
        return left

    def _parse_not(self) -> ExpressionNode:
        if self._match(TokenType.OP_NOT):
            op = self._advance().value
            operand = self._parse_not()
            return UnaryOpNode(op=op, operand=operand,
                               line=operand.line if hasattr(operand, "line") else 0)
        return self._parse_rel()

    def _parse_rel(self) -> ExpressionNode:
        left = self._parse_add()
        if self._peek().type in _REL_OPS:
            tok = self._advance()
            op = _REL_OPS[tok.type]
            right = self._parse_add()
            return BinaryOpNode(op=op, left=left, right=right, line=tok.line)
        return left

    def _parse_add(self) -> ExpressionNode:
        left = self._parse_mul()
        while self._match(TokenType.PLUS, TokenType.MINUS):
            tok = self._advance()
            op = tok.value
            right = self._parse_mul()
            left = BinaryOpNode(op=op, left=left, right=right, line=tok.line)
        return left

    def _parse_mul(self) -> ExpressionNode:
        left = self._parse_unary()
        while self._match(TokenType.STAR, TokenType.SLASH):
            tok = self._advance()
            op = tok.value
            right = self._parse_unary()
            left = BinaryOpNode(op=op, left=left, right=right, line=tok.line)
        return left

    def _parse_unary(self) -> ExpressionNode:
        if self._match(TokenType.MINUS):
            tok = self._advance()
            operand = self._parse_unary()
            return UnaryOpNode(op="-", operand=operand, line=tok.line)
        return self._parse_primary()

    def _parse_primary(self) -> ExpressionNode:
        tok = self._peek()

        if tok.type == TokenType.NUMBER:
            self._advance()
            return NumberNode(value=float(tok.value), line=tok.line)

        if tok.type == TokenType.STRING:
            self._advance()
            return StringNode(value=tok.value, line=tok.line)

        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN, "expression")
            return expr

        if tok.type == TokenType.VARIABLE:
            self._advance()
            # Check for min_valid suffix FIRST: VARIABLE NUMBER LPAREN
            # This handles MEAN.2(args) where .2 is tokenized as NUMBER
            if (self._peek().type == TokenType.NUMBER and
                    self._peek(1).type == TokenType.LPAREN):
                num_tok = self._advance()  # consume NUMBER
                # Extract the min_valid: ".2" → 2, "2" → 2
                num_str = num_tok.value.lstrip(".")
                min_val = int(num_str) if num_str.isdigit() else int(float(num_tok.value))
                self._advance()  # consume LPAREN
                args2: List[ExpressionNode] = []
                if not self._match(TokenType.RPAREN):
                    args2.append(self._parse_expression())
                    while self._match(TokenType.COMMA):
                        self._advance()
                        args2.append(self._parse_expression())
                self._expect(TokenType.RPAREN, "function arguments")
                return FunctionCallNode(
                    name=tok.value.upper(), args=args2,
                    min_valid=min_val, line=tok.line,
                )
            # Check if this is a function call: VAR ( args )
            # In SPSS, function names are not reserved keywords, so MEAN, SUM, etc.
            # are tokenized as VARIABLE tokens
            if self._match(TokenType.LPAREN):
                self._advance()  # consume (
                args: List[ExpressionNode] = []
                if not self._match(TokenType.RPAREN):
                    args.append(self._parse_expression())
                    while self._match(TokenType.COMMA):
                        self._advance()
                        args.append(self._parse_expression())
                self._expect(TokenType.RPAREN, "function arguments")
                func_name = tok.value.upper()
                return FunctionCallNode(name=func_name, args=args, line=tok.line)
            # Regular variable reference
            return VariableNode(name=tok.value, line=tok.line)

        # Keywords that can appear in expressions (TO in var lists, etc.)
        if tok.type == TokenType.KEYWORD:
            # Handle SYSMIS, MISSING as functions or constants
            if tok.value in ("SYSMIS", "MISSING"):
                self._advance()
                if self._match(TokenType.LPAREN):
                    self._advance()
                    arg = self._parse_expression()
                    self._expect(TokenType.RPAREN, "function arguments")
                    return FunctionCallNode(name=tok.value,
                                            args=[arg], line=tok.line)
                # As a constant
                return VariableNode(name=tok.value, line=tok.line)

        raise SPSSSyntaxError(
            f"Unexpected token in expression: {tok.type.name} ({tok.value!r})",
            line=tok.line,
        )