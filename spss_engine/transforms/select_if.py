"""
SELECT IF command implementation.

SELECT IF expression — filters cases permanently from the dataset.

The expression is evaluated for each case. If the result is 0, missing,
or False, the case is removed from the DataFrame. This is a permanent
deletion (unlike FILTER, which is reversible).

SELECT IF is a pending transform: it is enqueued and executed during the
next data pass (when a data-reading command like FREQUENCIES runs, or
EXECUTE is called).
"""

from __future__ import annotations
import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from spss_engine.data.dataset import Dataset
from spss_engine.parser.ast_nodes import CommandNode, SubcommandNode, \
    ExpressionNode
from spss_engine.utils.expressions import ExpressionEvaluator, _is_missing
from spss_engine.utils.errors import SPSSRuntimeError

logger = logging.getLogger(__name__)


def execute_select_if(cmd: CommandNode, ds: Dataset) -> None:
    """Execute a SELECT IF command: permanently remove cases where the
    expression evaluates to 0, missing, or False.

    Args:
        cmd: Parsed SELECT IF CommandNode. The expression is in the
            first subcommand named "CONDITION".
        ds: The active Dataset.
    """
    expr: Optional[ExpressionNode] = None
    for sc in cmd.subcommands:
        if sc.name == "CONDITION" and sc.expression is not None:
            expr = sc.expression
            break
    if expr is None:
        raise SPSSRuntimeError("SELECT IF requires a condition expression",
                               command="SELECT IF")

    if ds.df is None or ds.n_cases == 0:
        return

    evaluator = ExpressionEvaluator(ds)
    n = ds.n_cases
    keep_mask: list[bool] = []
    for i in range(n):
        val = evaluator.evaluate(expr, i)
        # Keep case if val is truthy (non-zero, non-missing)
        if _is_missing(val):
            keep_mask.append(False)
        elif isinstance(val, str):
            keep_mask.append(len(val) > 0)
        else:
            try:
                keep_mask.append(float(val) != 0.0)
            except (TypeError, ValueError):
                keep_mask.append(False)

    # Apply the filter permanently
    new_df = ds.df[pd.Series(keep_mask, index=ds.df.index)]
    ds.df = new_df.reset_index(drop=True)
    ds._n_cases = len(new_df)