"""
RELIABILITY procedure implementation.

RELIABILITY
  /VARIABLES=varlist
  /SCALE(name)
  /SUMMARY=TOTAL

Computes Cronbach's alpha for a set of items.

N=0: empty PivotTable + warning, no crash.
"""

from __future__ import annotations
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from spss_engine.data.dataset import Dataset
from spss_engine.data.variable import Variable
from spss_engine.parser.ast_nodes import CommandNode, SubcommandNode
from spss_engine.output.pivot_table import (
    PivotTable, Dimension, DimensionPlace, Category, CellText,
    FormatSpec, FormatType,
)
from spss_engine.transforms.split_file import get_split_groups
from spss_engine.utils.errors import SPSSRuntimeError

logger = logging.getLogger(__name__)


def execute_reliability(cmd: CommandNode, ds: Dataset,
                          executor: Any = None) -> List[PivotTable]:
    """Execute a RELIABILITY command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    # Parse /VARIABLES
    var_names: List[str] = []
    for sc in cmd.subcommands:
        if sc.name == "VARIABLES":
            var_names = ds.get_varlist(sc.variables) if sc.variables else []
            break

    if not var_names:
        raise SPSSRuntimeError("RELIABILITY requires /VARIABLES",
                                command="RELIABILITY")

    # Parse /SCALE(name)
    scale_name: str = "ALL"
    for sc in cmd.subcommands:
        if sc.name == "SCALE":
            if sc.raw_tokens:
                scale_name = str(sc.raw_tokens[0])
            break

    # Parse /SUMMARY
    do_total = False
    for sc in cmd.subcommands:
        if sc.name == "SUMMARY":
            for kw in sc.keywords:
                if isinstance(kw, str) and kw.upper() == "TOTAL":
                    do_total = True
                    break

    split_groups = get_split_groups(ds)
    for group_key, group_df in split_groups:
        avail_vars = [v for v in var_names if v in group_df.columns]
        if not avail_vars:
            empty = PivotTable(title="Reliability Statistics")
            empty.notes.append("No valid variables found.")
            tables.append(empty)
            continue

        # Listwise deletion
        valid = group_df[avail_vars].dropna()
        n = len(valid)
        k = len(avail_vars)

        if n == 0 or k < 2:
            empty = PivotTable(title="Reliability Statistics")
            if group_key:
                from spss_engine.transforms.split_file import split_group_label
                label = split_group_label(ds, group_key)
                empty.title = f"Reliability Statistics [{label}]"
            empty.notes.append("No valid cases or too few items.")
            tables.append(empty)
            continue

        data = valid[avail_vars].values.astype(float)

        # Cronbach's alpha
        # alpha = (k / (k-1)) * (1 - sum(item_var) / total_var)
        item_vars = np.var(data, axis=0, ddof=1)
        total_var = np.var(data.sum(axis=1), ddof=1)

        if total_var > 0:
            alpha = (k / (k - 1)) * (1 - np.sum(item_vars) / total_var)
        else:
            alpha = float("nan")

        # Reliability statistics table
        title = "Reliability Statistics"
        if group_key:
            from spss_engine.transforms.split_file import split_group_label
            label = split_group_label(ds, group_key)
            title = f"Reliability Statistics [{label}]"

        rel_table = PivotTable(title=title)
        rel_table.simple_pivot_table(
            rowdim=" ",
            rowlabels=["Cronbach's Alpha"],
            coldim=" ",
            collabels=["Cronbach's Alpha", "N of Items"],
            cells=[alpha if not math.isnan(alpha) else None, float(k)],
        )
        tables.append(rel_table)

        # Item statistics
        item_title = "Item Statistics"
        if group_key:
            from spss_engine.transforms.split_file import split_group_label
            label = split_group_label(ds, group_key)
            item_title = f"Item Statistics [{label}]"

        item_table = PivotTable(title=item_title)
        col_labels = ["Mean", "Std. Deviation", "N"]
        row_labels = list(avail_vars)
        cells: List[Any] = []
        for i, iv in enumerate(avail_vars):
            col_vals = data[:, i]
            im = float(np.mean(col_vals))
            isd = float(np.std(col_vals, ddof=1)) if n > 1 else float("nan")
            cells.extend([im, isd, float(n)])

        item_table.simple_pivot_table(
            rowdim="Items",
            rowlabels=row_labels,
            coldim="Statistics",
            collabels=col_labels,
            cells=cells,
        )
        tables.append(item_table)

        if do_total:
            # Summary: total scale statistics
            total_title = "Scale Statistics"
            if group_key:
                from spss_engine.transforms.split_file import split_group_label
                label = split_group_label(ds, group_key)
                total_title = f"Scale Statistics [{label}]"

            total_table = PivotTable(title=total_title)
            total_scores = data.sum(axis=1)
            tm = float(np.mean(total_scores))
            tsd = float(np.std(total_scores, ddof=1)) if n > 1 else float("nan")
            tv = float(np.var(total_scores, ddof=1)) if n > 1 else float("nan")

            total_table.simple_pivot_table(
                rowdim=" ",
                rowlabels=["Scale"],
                coldim="Statistics",
                collabels=["Mean", "Variance", "Std. Deviation", "N"],
                cells=[tm, tv, tsd, float(n)],
            )
            tables.append(total_table)

    if executor is not None:
        for t in tables:
            executor.add_table(t)

    return tables