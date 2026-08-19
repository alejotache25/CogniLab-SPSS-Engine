"""
Expression evaluator for SPSS COMPUTE and IF conditions.

Evaluates AST expression nodes against a Dataset and case index.

Supported expression nodes:
  - NumberNode, StringNode, VariableNode
  - BinaryOpNode (+, -, *, /, =, <>, <, >, <=, >=, AND, OR)
  - UnaryOpNode (NOT, unary minus)
  - FunctionCallNode (MEAN, SUM, MIN, MAX, SD, VAR, ABS, SQRT, RND, TRUNC,
    LENGTH, RTRIM, LTRIM, SUBSTR, CONCAT, INDEX, MISSING, SYSMIS, NMISS,
    NVALID, VALUE, etc.)

Missing value handling per CSR:
  - Arithmetic with missing → system-missing
  - Statistical functions: require min_valid args (default 1 for MEAN/SUM/MIN/MAX,
    default 2 for SD/VAR/CFVAR)
  - 0 * missing = 0, 0 / missing = 0, MOD(0, missing) = 0
"""

from __future__ import annotations
import math
import numpy as np
from typing import Any, List, Optional, Union

from spss_engine.parser.ast_nodes import (
    NumberNode, StringNode, VariableNode, VarRangeNode,
    BinaryOpNode, UnaryOpNode, FunctionCallNode,
    ExpressionNode,
)
from spss_engine.data.dataset import Dataset
from spss_engine.utils.errors import SPSSRuntimeError


# Default min_valid for statistical functions
_DEFAULT_MIN_VALID: dict[str, int] = {
    "SUM": 1, "MEAN": 1, "MIN": 1, "MAX": 1,
    "SD": 2, "VARIANCE": 2, "CFVAR": 2,
}


class ExpressionEvaluator:
    """Evaluates SPSS expression AST nodes against a Dataset."""

    def __init__(self, dataset: Dataset) -> None:
        self._ds: Dataset = dataset

    def evaluate(self, expr: ExpressionNode,
                 case_idx: Optional[int] = None) -> Any:
        """Evaluate an expression for a specific case.

        Args:
            expr: The expression AST node.
            case_idx: Row index in the DataFrame (0-based). If None,
                      evaluates in a context without case data (for constants).

        Returns:
            The evaluated value (float, str, or NaN/None for missing).
        """
        return self._eval(expr, case_idx)

    def _eval(self, expr: ExpressionNode, case_idx: Optional[int]) -> Any:
        """Dispatch evaluation based on node type."""
        if isinstance(expr, NumberNode):
            return float(expr.value)

        if isinstance(expr, StringNode):
            return expr.value

        if isinstance(expr, VariableNode):
            return self._eval_variable(expr.name, case_idx)

        if isinstance(expr, BinaryOpNode):
            return self._eval_binary(expr, case_idx)

        if isinstance(expr, UnaryOpNode):
            return self._eval_unary(expr, case_idx)

        if isinstance(expr, FunctionCallNode):
            return self._eval_function(expr, case_idx)

        raise SPSSRuntimeError(f"Unknown expression node: {type(expr).__name__}")

    def _eval_variable(self, name: str,
                         case_idx: Optional[int]) -> Any:
        """Get the value of a variable for a specific case."""
        if not self._ds.has_variable(name):
            raise SPSSRuntimeError(f"Variable not found: {name}")
        if case_idx is None:
            return np.nan
        df = self._ds.df
        if case_idx >= len(df):
            return np.nan
        val = df.iloc[case_idx][name]
        # Convert NaN to nan for numerics, keep string as-is
        var = self._ds.get_variable(name)
        if var.is_numeric:
            if pd.isna(val):
                return np.nan
            return float(val)
        # String: convert None to ""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return str(val)

    def _eval_binary(self, expr: BinaryOpNode,
                      case_idx: Optional[int]) -> Any:
        """Evaluate a binary operation."""
        left = self._eval(expr.left, case_idx)
        right = self._eval(expr.right, case_idx)
        op = expr.op

        # Arithmetic operators
        if op == "+":
            # String concatenation if either is string
            if isinstance(left, str) or isinstance(right, str):
                ls = left if isinstance(left, str) else ""
                rs = right if isinstance(right, str) else ""
                return ls + rs
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return float(left) + float(right)

        if op == "-":
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return float(left) - float(right)

        if op == "*":
            # 0 * missing = 0 (per CSR)
            if (not _is_missing(left) and float(left) == 0 and
                    _is_missing(right)):
                return 0.0
            if (not _is_missing(right) and float(right) == 0 and
                    _is_missing(left)):
                return 0.0
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return float(left) * float(right)

        if op == "/":
            # 0 / missing = 0 (per CSR)
            if (not _is_missing(left) and float(left) == 0 and
                    _is_missing(right)):
                return 0.0
            if _is_missing(left) or _is_missing(right):
                return np.nan
            if float(right) == 0:
                return np.nan  # division by zero
            return float(left) / float(right)

        # Relational operators
        if op == "=" or op == "==":
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return 1.0 if left == right else 0.0

        if op == "<>" or op == "!=":
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return 1.0 if left != right else 0.0

        if op == "<":
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return 1.0 if float(left) < float(right) else 0.0

        if op == ">":
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return 1.0 if float(left) > float(right) else 0.0

        if op == "<=":
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return 1.0 if float(left) <= float(right) else 0.0

        if op == ">=":
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return 1.0 if float(left) >= float(right) else 0.0

        # Logical operators
        if op == "AND":
            # Per CSR: AND can be false if one is false, even if other is missing
            if _to_bool(left) is False or _to_bool(right) is False:
                return 0.0
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return 1.0 if _to_bool(left) and _to_bool(right) else 0.0

        if op == "OR":
            # Per CSR: OR can be true if one is true, even if other is missing
            if _to_bool(left) is True or _to_bool(right) is True:
                return 1.0
            if _is_missing(left) or _is_missing(right):
                return np.nan
            return 0.0

        raise SPSSRuntimeError(f"Unknown operator: {op}")

    def _eval_unary(self, expr: UnaryOpNode,
                     case_idx: Optional[int]) -> Any:
        """Evaluate a unary operation."""
        operand = self._eval(expr.operand, case_idx)
        if expr.op == "-":
            if _is_missing(operand):
                return np.nan
            return -float(operand)
        if expr.op == "NOT":
            b = _to_bool(operand)
            if b is None:
                return np.nan
            return 0.0 if b else 1.0
        raise SPSSRuntimeError(f"Unknown unary operator: {expr.op}")

    def _eval_function(self, expr: FunctionCallNode,
                         case_idx: Optional[int]) -> Any:
        """Evaluate a function call."""
        name = expr.name.upper()
        args = expr.args

        # Missing value functions
        if name == "MISSING":
            val = self._eval(args[0], case_idx)
            var_node = args[0]
            if isinstance(var_node, VariableNode):
                var = self._ds.get_variable(var_node.name)
                return 1.0 if var.is_missing(val) else 0.0
            return 1.0 if _is_missing(val) else 0.0

        if name == "SYSMIS":
            val = self._eval(args[0], case_idx)
            var_node = args[0]
            if isinstance(var_node, VariableNode):
                var = self._ds.get_variable(var_node.name)
                return 1.0 if var.is_system_missing(val) else 0.0
            return 1.0 if _is_missing(val) else 0.0

        if name == "NMISS":
            vals = self._eval_all(args, case_idx)
            return float(sum(1 for v in vals if _is_missing(v)))

        if name == "NVALID":
            vals = self._eval_all(args, case_idx)
            return float(sum(1 for v in vals if not _is_missing(v)))

        if name == "VALUE":
            val = self._eval(args[0], case_idx)
            # VALUE returns the value ignoring user-missing, but system-missing
            # is still missing
            var_node = args[0]
            if isinstance(var_node, VariableNode):
                var = self._ds.get_variable(var_node.name)
                if var.is_system_missing(val):
                    return np.nan
                return val
            return val

        # Statistical functions (MEAN, SUM, MIN, MAX, SD, VARIANCE, CFVAR)
        if name in _DEFAULT_MIN_VALID:
            vals = self._eval_all(args, case_idx)
            valid_vals = [float(v) for v in vals if not _is_missing(v)]
            min_valid = expr.min_valid if expr.min_valid is not None else _DEFAULT_MIN_VALID[name]
            if len(valid_vals) < min_valid:
                return np.nan
            if name == "SUM":
                return float(sum(valid_vals))
            if name == "MEAN":
                return float(np.mean(valid_vals))
            if name == "MIN":
                return float(min(valid_vals))
            if name == "MAX":
                return float(max(valid_vals))
            if name == "SD":
                if len(valid_vals) < 2:
                    return np.nan
                return float(np.std(valid_vals, ddof=1))
            if name == "VARIANCE":
                if len(valid_vals) < 2:
                    return np.nan
                return float(np.var(valid_vals, ddof=1))
            if name == "CFVAR":
                mean_val = float(np.mean(valid_vals))
                if abs(mean_val) < 1e-10:
                    return np.nan
                return float(np.std(valid_vals, ddof=1) / mean_val)

        # Math functions
        if name == "ABS":
            val = self._eval(args[0], case_idx)
            if _is_missing(val):
                return np.nan
            return float(abs(float(val)))

        if name == "SQRT":
            val = self._eval(args[0], case_idx)
            if _is_missing(val) or float(val) < 0:
                return np.nan
            return float(math.sqrt(float(val)))

        if name == "RND":
            val = self._eval(args[0], case_idx)
            if _is_missing(val):
                return np.nan
            return float(round(float(val)))

        if name == "TRUNC":
            val = self._eval(args[0], case_idx)
            if _is_missing(val):
                return np.nan
            return float(math.trunc(float(val)))

        if name == "EXP":
            val = self._eval(args[0], case_idx)
            if _is_missing(val):
                return np.nan
            return float(math.exp(float(val)))

        if name == "LN":
            val = self._eval(args[0], case_idx)
            if _is_missing(val) or float(val) <= 0:
                return np.nan
            return float(math.log(float(val)))

        if name == "LG10":
            val = self._eval(args[0], case_idx)
            if _is_missing(val) or float(val) <= 0:
                return np.nan
            return float(math.log10(float(val)))

        if name == "MOD":
            x = self._eval(args[0], case_idx)
            y = self._eval(args[1], case_idx)
            # MOD(0, missing) = 0 (per CSR)
            if (not _is_missing(x) and float(x) == 0 and
                    _is_missing(y)):
                return 0.0
            if _is_missing(x) or _is_missing(y) or float(y) == 0:
                return np.nan
            return float(math.fmod(float(x), float(y)))

        if name == "UPCASE":
            val = self._eval(args[0], case_idx)
            if val is None:
                return ""
            return str(val).upper()

        if name == "DOWNCASE":
            val = self._eval(args[0], case_idx)
            if val is None:
                return ""
            return str(val).lower()

        # String functions
        if name == "LENGTH":
            val = self._eval(args[0], case_idx)
            if val is None:
                return 0.0
            return float(len(str(val)))

        if name == "RTRIM":
            val = self._eval(args[0], case_idx)
            if val is None:
                return ""
            return str(val).rstrip()

        if name == "LTRIM":
            val = self._eval(args[0], case_idx)
            if val is None:
                return ""
            return str(val).lstrip()

        if name == "SUBSTR":
            val = self._eval(args[0], case_idx)
            if val is None:
                return ""
            s = str(val)
            start = int(float(self._eval(args[1], case_idx)))
            if start < 1:
                start = 1
            start -= 1  # SPSS is 1-indexed
            if len(args) >= 3:
                length = int(float(self._eval(args[2], case_idx)))
                return s[start:start + length]
            return s[start:]

        if name == "CONCAT":
            parts: list[str] = []
            for arg in args:
                val = self._eval(arg, case_idx)
                if val is None:
                    parts.append("")
                else:
                    parts.append(str(val))
            return "".join(parts)

        if name == "INDEX":
            s = self._eval(args[0], case_idx)
            substr = self._eval(args[1], case_idx)
            if s is None or substr is None:
                return 0.0
            pos = str(s).find(str(substr))
            return float(pos + 1) if pos >= 0 else 0.0

        if name == "REPLACE":
            s = self._eval(args[0], case_idx)
            old = self._eval(args[1], case_idx)
            new = self._eval(args[2], case_idx)
            if s is None or old is None or new is None:
                return ""
            return str(s).replace(str(old), str(new))

        # Lag function
        if name == "LAG":
            val = self._eval(args[0], case_idx)
            lag_n = 1
            if len(args) >= 2:
                lag_n = int(float(self._eval(args[1], case_idx)))
            if case_idx is None or case_idx < lag_n:
                return np.nan
            return self._eval_variable(
                args[0].name if isinstance(args[0], VariableNode) else "",
                case_idx - lag_n)

        # ANY function: ANY(x, x1, x2, ...) returns 1 if x matches any
        if name == "ANY":
            x = self._eval(args[0], case_idx)
            if _is_missing(x):
                return np.nan
            for arg in args[1:]:
                val = self._eval(arg, case_idx)
                if not _is_missing(val) and x == val:
                    return 1.0
            return 0.0

        # RANGE function: RANGE(x, lo, hi) returns 1 if lo <= x <= hi
        if name == "RANGE":
            x = self._eval(args[0], case_idx)
            lo = self._eval(args[1], case_idx)
            hi = self._eval(args[2], case_idx)
            if _is_missing(x) or _is_missing(lo) or _is_missing(hi):
                return np.nan
            return 1.0 if float(lo) <= float(x) <= float(hi) else 0.0

        raise SPSSRuntimeError(f"Unknown function: {name}")

    def _eval_all(self, args: List[ExpressionNode],
                  case_idx: Optional[int]) -> List[Any]:
        """Evaluate all arguments, expanding TO ranges."""
        results: list[Any] = []
        for arg in args:
            if isinstance(arg, VarRangeNode):
                # Expand TO range
                var_names = self._ds._expand_to(arg.start, arg.end)
                for vn in var_names:
                    results.append(self._eval_variable(vn, case_idx))
            else:
                results.append(self._eval(arg, case_idx))
        return results


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    """Check if a value is system-missing (NaN or None)."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value == "":
        return False  # Empty string is NOT system-missing in SPSS
    return False


def _to_bool(value: Any) -> Optional[bool]:
    """Convert a value to boolean for logical operations.

    Returns None if the value is missing (indeterminate).
    """
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return len(value) > 0
    return float(value) != 0.0


# Import pandas at module level for isna check
import pandas as pd  # noqa: E402