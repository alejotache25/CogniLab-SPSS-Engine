"""
FILTER command implementation.

FILTER BY var   — sets a filter variable; cases where var is 0 or missing
                  are excluded from subsequent procedures.
FILTER OFF      — clears the filter.

The filter is reversible: it does not delete cases, only marks them as
filtered. Cases with missing values in the filter variable are excluded.

This module delegates to Dataset.set_filter / Dataset.clear_filter, which
are already implemented in the data layer. The command_registry handler
calls execute_filter() here.
"""

from __future__ import annotations
import logging
from typing import Any

from spss_engine.data.dataset import Dataset
from spss_engine.parser.ast_nodes import CommandNode, SubcommandNode
from spss_engine.utils.errors import SPSSRuntimeError

logger = logging.getLogger(__name__)


def execute_filter(cmd: CommandNode, ds: Dataset) -> None:
    """Execute a FILTER command.

    Handles:
      - FILTER BY varname
      - FILTER OFF
    """
    for sc in cmd.subcommands:
        if sc.name == "OFF":
            ds.clear_filter()
            return
        if sc.name == "BY" and sc.variables:
            var_name = sc.variables[0]
            if not ds.has_variable(var_name):
                raise SPSSRuntimeError(
                    f"Filter variable not found: {var_name}",
                    command="FILTER")
            ds.set_filter(var_name)
            return
    # If no subcommand matched, treat as FILTER OFF
    ds.clear_filter()