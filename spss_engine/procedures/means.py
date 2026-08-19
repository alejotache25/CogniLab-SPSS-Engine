"""
MEANS procedure implementation.

MEANS TABLES=var BY var BY var
  /CELLS=MEAN STDDEV COUNT

Produces layered means table with optional statistics per cell.

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


def execute_means(cmd: CommandNode, ds: Dataset,
                    executor: Any = None) -> List[PivotTable]:
    """Execute a MEANS command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    # Parse TABLES=var BY var BY var
    dep_vars: List[str] = []
    by_vars: List[str] = []
    for sc in cmd.subcommands:
        if sc.name in ("TABLES", "_MAIN"):
            # Find BY keywords
            for i, kw in enumerate(sc.keywords):
                if isinstance(kw, str) and kw.upper() == "BY":
                    # Dependent vars are variables before BY
                    # Independent vars are variables after BY
                    pass
            # Try to split variables list by BY
            by_indices: List[int] = []
            for i, kw in enumerate(sc.keywords):
                if isinstance(kw, str) and kw.upper() == "BY":
                    by_indices.append(i)

            if by_indices:
                # First segment: dependent vars
                dep_vars = list(sc.variables[:len(sc.variables) - len(by_indices)])
                # Remaining: independent (BY) vars
                # Actually the variables list includes all vars in order
                # BY keywords separate them
                all_vars = sc.variables
                by_positions: List[int] = []
                for i, kw in enumerate(sc.keywords):
                    if isinstance(kw, str) and kw.upper() == "BY":
                        by_positions.append(i)

                # Dependent vars come first
                n_dep = len(all_vars) - len(by_positions)
                dep_vars = all_vars[:n_dep]
                by_vars = all_vars[n_dep:]
            else:
                dep_vars = list(sc.variables)
            break

    if not dep_vars:
        raise SPSSRuntimeError("MEANS requires TABLES with dependent variables",
                                command="MEANS")

    # Parse /CELLS
    do_mean = True
    do_stddev = False
    do_count = False
    for sc in cmd.subcommands:
        if sc.name == "CELLS":
            for kw in sc.keywords:
                if isinstance(kw, str):
                    ku = kw.upper()
                    if ku == "MEAN":
                        do_mean = True
                    elif ku == "STDDEV":
                        do_stddev = True
                    elif ku == "COUNT":
                        do_count = True
                    elif ku == "ALL":
                        do_mean = do_stddev = do_count = True

    # Build column labels based on requested cells
    cell_labels: List[str] = []
    if do_mean:
        cell_labels.append("Mean")
    if do_stddev:
        cell_labels.append("Std. Deviation")
    if do_count:
        cell_labels.append("N")

    if not cell_labels:
        cell_labels = ["Mean"]

    split_groups = get_split_groups(ds)
    for group_key, group_df in split_groups:
        for dep_var in dep_vars:
            if dep_var not in group_df.columns:
                continue

            table = _build_means_table(dep_var, by_vars, group_df,
                                          cell_labels, do_mean,
                                          do_stddev, do_count,
                                          group_key, ds)
            tables.append(table)

    if executor is not None:
        for t in tables:
            executor.add_table(t)

    return tables


def _build_means_table(dep_var: str, by_vars: List[str],
                         df: pd.DataFrame,
                         cell_labels: List[str],
                         do_mean: bool, do_stddev: bool, do_count: bool,
                         group_key: tuple,
                         ds: Dataset) -> PivotTable:
    """Build a means table."""
    title = f"Means: {dep_var}"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Means: {dep_var} [{label}]"

    table = PivotTable(title=title)

    col = df[dep_var]
    n = col.notna().sum()

    if n == 0:
        table.notes.append(f"No valid cases for {dep_var}.")
        return table

    if not by_vars:
        # Just overall stats
        vals = col.dropna().values.astype(float)
        cells: List[Any] = []
        if do_mean:
            cells.append(float(np.mean(vals)))
        if do_stddev:
            cells.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else None)
        if do_count:
            cells.append(float(len(vals)))

        table.simple_pivot_table(
            rowdim=" ",
            rowlabels=["Total"],
            coldim="Statistics",
            collabels=cell_labels,
            cells=cells,
        )
        return table

    # Group by first BY variable
    first_by = by_vars[0]
    if first_by not in df.columns:
        table.notes.append(f"BY variable not found: {first_by}")
        return table

    row_labels: List[str] = []
    cells: List[Any] = []

    for fval, group in df.groupby(first_by, sort=True):
        vals = group[dep_var].dropna().values.astype(float)
        gn = len(vals)

        if len(by_vars) == 1:
            # Single BY: rows = groups of first BY var
            label = str(fval)
            row_labels.append(label)
            if gn == 0:
                cells.extend([None] * len(cell_labels))
            else:
                if do_mean:
                    cells.append(float(np.mean(vals)))
                if do_stddev:
                    cells.append(float(np.std(vals, ddof=1)) if gn > 1 else None)
                if do_count:
                    cells.append(float(gn))
        else:
            # Nested BY: for each sub-group
            second_by = by_vars[1]
            if second_by not in df.columns:
                continue
            for sval, subgroup in group.groupby(second_by, sort=True):
                svals = subgroup[dep_var].dropna().values.astype(float)
                sn = len(svals)
                label = f"{fval} * {sval}"
                row_labels.append(label)
                if sn == 0:
                    cells.extend([None] * len(cell_labels))
                else:
                    if do_mean:
                        cells.append(float(np.mean(svals)))
                    if do_stddev:
                        cells.append(float(np.std(svals, ddof=1)) if sn > 1 else None)
                    if do_count:
                        cells.append(float(sn))

    # Add Total row
    vals_all = col.dropna().values.astype(float)
    n_all = len(vals_all)
    row_labels.append("Total")
    if n_all == 0:
        cells.extend([None] * len(cell_labels))
    else:
        if do_mean:
            cells.append(float(np.mean(vals_all)))
        if do_stddev:
            cells.append(float(np.std(vals_all, ddof=1)) if n_all > 1 else None)
        if do_count:
            cells.append(float(n_all))

    table.simple_pivot_table(
        rowdim=first_by if len(by_vars) == 1 else "Groups",
        rowlabels=row_labels,
        coldim="Statistics",
        collabels=cell_labels,
        cells=cells,
    )
    return table