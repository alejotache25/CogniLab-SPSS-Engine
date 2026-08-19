"""
COMPUTE command implementation.

COMPUTE target = expression.

The expression is evaluated for each case (row) in the dataset and the
result is stored in the target variable. If the target variable does not
exist, it is created as numeric (F8.2). String target variables must be
declared beforehand with STRING.

Missing value propagation rules (from csr_missing_values.txt):
  - 0 * missing = 0, 0 / missing = 0, MOD(0, missing) = 0
  - Arithmetic with any missing → system-missing (NaN)
  - Statistical functions (MEAN, SUM, etc.) require min_valid args
  - MISSING(x) → 1 if x is system or user-missing
  - SYSMIS(x) → 1 if x is system-missing
  - NMISS(...) → count of missing args
  - NVALID(...) → count of valid args
  - VALUE(x) → treats user-missing as valid

The evaluator (ExpressionEvaluator in utils/expressions.py) handles all
function dispatch. This module orchestrates the per-case evaluation loop
and the creation of the target variable.
"""

from __future__ import annotations
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from spss_engine.data.dataset import Dataset
from spss_engine.data.variable import Variable
from spss_engine.parser.ast_nodes import CommandNode, ExpressionNode
from spss_engine.utils.expressions import ExpressionEvaluator
from spss_engine.utils.errors import SPSSRuntimeError

logger = logging.getLogger(__name__)


def execute_compute(cmd: CommandNode, ds: Dataset) -> None:
    """Execute a COMPUTE command against the dataset.

    Args:
        cmd: Parsed COMPUTE CommandNode with target_var and
            target_expression set.
        ds: The active Dataset.

    Raises:
        SPSSRuntimeError: if target variable is missing, expression
            references a non-existent variable, or string target is
            undeclared.
    """
    if cmd.target_var is None:
        raise SPSSRuntimeError("COMPUTE requires a target variable",
                               command="COMPUTE")
    if cmd.target_expression is None:
        raise SPSSRuntimeError("COMPUTE requires an expression",
                               command="COMPUTE")

    target = cmd.target_var
    expr = cmd.target_expression

    # Evaluate the expression once on a dummy case to determine the type
    # (numeric or string). This also catches "Variable not found" early.
    evaluator = ExpressionEvaluator(ds)
    # Probe: evaluate on first valid case (or case 0)
    probe_idx: Optional[int] = 0 if ds.n_cases > 0 else None
    try:
        probe_val = evaluator.evaluate(expr, probe_idx)
    except SPSSRuntimeError:
        raise
    is_string_result = isinstance(probe_val, str)

    # Create the target variable if it doesn't exist
    if not ds.has_variable(target):
        if is_string_result:
            raise SPSSRuntimeError(
                f"COMPUTE target variable not found (string targets must be "
                f"declared with STRING first): {target}",
                command="COMPUTE")
        ds.add_variable(Variable(name=target, var_type="numeric",
                                  width=8, format="F8.2"))
        # Add column to DataFrame
        if ds.df is not None:
            ds.df[target] = np.nan
        else:
            ds.df = pd.DataFrame({target: []})

    # Verify type compatibility
    target_var = ds.get_variable(target)
    if is_string_result and target_var.is_numeric:
        raise SPSSRuntimeError(
            f"COMPUTE: cannot assign string value to numeric variable {target}",
            command="COMPUTE")
    if not is_string_result and target_var.is_string:
        raise SPSSRuntimeError(
            f"COMPUTE: cannot assign numeric value to string variable {target}",
            command="COMPUTE")

    # Evaluate expression for every case
    n = ds.n_cases
    if n == 0:
        return

    df = ds.df
    results: list[Any] = []
    for i in range(n):
        val = evaluator.evaluate(expr, i)
        if is_string_result:
            if val is None:
                results.append("")
            else:
                results.append(str(val))
        else:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                results.append(np.nan)
            else:
                try:
                    results.append(float(val))
                except (TypeError, ValueError):
                    results.append(np.nan)

    # Assign results to the target column
    if is_string_result:
        df[target] = results
    else:
        df[target] = pd.Series(results, dtype=float)