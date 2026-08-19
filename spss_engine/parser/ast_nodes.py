"""
AST node definitions for SPSS syntax.

The parser produces a list of CommandNode objects, each containing
SubcommandNode objects, which contain expression nodes and varlists.

Node hierarchy:
  CommandNode
    └── SubcommandNode
          ├── varlist: list[str | VarRangeNode]
          ├── expression: ExpressionNode (for COMPUTE)
          └── args: list[...] (subcommand-specific)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


# ----------------------------------------------------------------------
# Expression AST nodes (for COMPUTE, IF conditions, SELECT IF, etc.)
# ----------------------------------------------------------------------

@dataclass
class NumberNode:
    """A numeric literal."""
    value: float
    line: int = 0

    def __repr__(self) -> str:
        return f"Number({self.value})"


@dataclass
class StringNode:
    """A string literal."""
    value: str
    line: int = 0

    def __repr__(self) -> str:
        return f"String({self.value!r})"


@dataclass
class VariableNode:
    """A variable reference."""
    name: str
    line: int = 0

    def __repr__(self) -> str:
        return f"Var({self.name})"


@dataclass
class VarRangeNode:
    """A variable range using TO: var1 TO var2."""
    start: str
    end: str
    line: int = 0

    def __repr__(self) -> str:
        return f"VarRange({self.start} TO {self.end})"


@dataclass
class BinaryOpNode:
    """A binary operation: left op right."""
    op: str
    left: "ExpressionNode"
    right: "ExpressionNode"
    line: int = 0

    def __repr__(self) -> str:
        return f"BinOp({self.left} {self.op} {self.right})"


@dataclass
class UnaryOpNode:
    """A unary operation: op operand (e.g., NOT, unary minus)."""
    op: str
    operand: "ExpressionNode"
    line: int = 0

    def __repr__(self) -> str:
        return f"UnaryOp({self.op} {self.operand})"


@dataclass
class FunctionCallNode:
    """A function call: NAME(args) or NAME.n(args) for statistical functions.

    The `min_valid` field holds the optional .n suffix (e.g., MEAN.2 → min_valid=2).
    """
    name: str
    args: List["ExpressionNode"]
    min_valid: Optional[int] = None
    line: int = 0

    def __repr__(self) -> str:
        suffix = f".{self.min_valid}" if self.min_valid is not None else ""
        return f"Func({self.name}{suffix}({', '.join(map(repr, self.args))}))"


# Union type for expression nodes
ExpressionNode = Union[
    NumberNode, StringNode, VariableNode, VarRangeNode,
    BinaryOpNode, UnaryOpNode, FunctionCallNode,
]


# ----------------------------------------------------------------------
# Command-level AST nodes
# ----------------------------------------------------------------------

@dataclass
class SubcommandNode:
    """A subcommand within a command (e.g., /VARIABLES=AGE SEX)."""
    name: str                       # Uppercased subcommand name (e.g., "VARIABLES")
    variables: List[str] = field(default_factory=list)  # Variable names
    keywords: List[str] = field(default_factory=list)   # Keywords (e.g., MEAN, STDDEV)
    expression: Optional[ExpressionNode] = None  # For COMPUTE target expression
    raw_tokens: List[Any] = field(default_factory=list)  # Raw tokens for flexibility

    def __repr__(self) -> str:
        return (f"Subcmd({self.name}, vars={self.variables}, "
                f"kw={self.keywords})")


@dataclass
class CommandNode:
    """A parsed SPSS command (e.g., FREQUENCIES, COMPUTE, DATA LIST)."""
    name: str                           # Uppercased command name
    subcommands: List[SubcommandNode] = field(default_factory=list)
    # For DATA LIST: variable definitions
    var_defs: List["VarDefNode"] = field(default_factory=list)
    # For DATA LIST: format type (FIXED, FREE, LIST)
    data_format: Optional[str] = None
    # For DATA LIST: file path
    file_path: Optional[str] = None
    # For DATA LIST: raw data from BEGIN DATA
    raw_data: Optional[str] = None
    # For COMPUTE: target variable and expression
    target_var: Optional[str] = None
    target_expression: Optional[ExpressionNode] = None
    # Line number where the command starts
    line: int = 0

    def get_subcommand(self, name: str) -> Optional[SubcommandNode]:
        """Find a subcommand by name (case-insensitive)."""
        upper = name.upper()
        for sc in self.subcommands:
            if sc.name.upper() == upper:
                return sc
        return None

    def __repr__(self) -> str:
        return f"Cmd({self.name}, subcmds={self.subcommands})"


@dataclass
class VarDefNode:
    """A variable definition from DATA LIST.

    For FIXED format: name, start_col, end_col, var_type, format_spec
    For FREE/LIST format: name, var_type, format_spec (no columns)
    """
    name: str                           # Variable name (case preserved)
    start_col: Optional[int] = None     # Start column (FIXED only)
    end_col: Optional[int] = None       # End column (FIXED only)
    var_type: str = "numeric"           # "numeric" or "string"
    format_spec: Optional[str] = None   # Format like "F8.2" or "A10"
    width: int = 8                       # Display width

    def __repr__(self) -> str:
        if self.start_col is not None:
            return (f"VarDef({self.name}, cols={self.start_col}-{self.end_col}, "
                    f"type={self.var_type})")
        return f"VarDef({self.name}, type={self.var_type})"