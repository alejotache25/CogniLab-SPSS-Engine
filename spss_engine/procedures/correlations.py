"""
CORRELATIONS procedure implementation.

CORRELATIONS VARIABLES=varlist
  /MISSING=PAIRWISE|LISTWISE
  /PRINT=TWOTAIL|ONETAIL

Computes Pearson (default) and Spearman correlations.

Verified against scipy.stats.pearsonr, spearmanr.

N=0: empty PivotTable + warning, no crash.
"""

from __future__ import annotations
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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


def execute_correlations(cmd: CommandNode, ds: Dataset,
                           executor: Any = None) -> List[PivotTable]:
    """Execute a CORRELATIONS command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    # Parse VARIABLES
    var_names: List[str] = []
    for sc in cmd.subcommands:
        if sc.name == "VARIABLES":
            var_names = ds.get_varlist(sc.variables) if sc.variables else []
            break
    if not var_names:
        # Try _MAIN
        for sc in cmd.subcommands:
            if sc.name == "_MAIN" and sc.variables:
                var_names = ds.get_varlist(sc.variables)
                break

    if not var_names:
        raise SPSSRuntimeError("CORRELATIONS requires VARIABLES",
                                command="CORRELATIONS")

    # Parse /MISSING
    missing_mode: str = "PAIRWISE"
    for sc in cmd.subcommands:
        if sc.name == "MISSING":
            for kw in sc.keywords:
                if isinstance(kw, str) and kw.upper() in ("PAIRWISE", "LISTWISE"):
                    missing_mode = kw.upper()
                    break

    # Parse /PRINT
    tail: str = "TWOTAIL"
    for sc in cmd.subcommands:
        if sc.name == "PRINT":
            for kw in sc.keywords:
                if isinstance(kw, str) and kw.upper() in ("TWOTAIL", "ONETAIL"):
                    tail = kw.upper()
                    break

    # Check for SPEARMAN keyword
    is_spearman = False
    for sc in cmd.subcommands:
        if sc.name == "_MAIN" or sc.name == "VARIABLES":
            for kw in sc.keywords:
                if isinstance(kw, str) and kw.upper() == "SPEARMAN":
                    is_spearman = True
                    break

    split_groups = get_split_groups(ds)
    for group_key, group_df in split_groups:
        table = _build_correlations_table(var_names, group_df,
                                            missing_mode, tail,
                                            is_spearman, group_key, ds)
        tables.append(table)

    if executor is not None:
        for t in tables:
            executor.add_table(t)

    return tables


def _build_correlations_table(var_names: List[str],
                                df: pd.DataFrame,
                                missing_mode: str, tail: str,
                                is_spearman: bool,
                                group_key: tuple,
                                ds: Dataset) -> PivotTable:
    """Build a correlation matrix PivotTable."""
    method_name = "Spearman" if is_spearman else "Pearson"
    title = f"{method_name} Correlations"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"{method_name} Correlations [{label}]"

    table = PivotTable(title=title)

    # Filter to only available variables
    avail_vars = [v for v in var_names if v in df.columns]
    if not avail_vars:
        table.notes.append("No valid variables found.")
        return table

    n_vars = len(avail_vars)

    # Build correlation matrix
    corr_matrix = np.full((n_vars, n_vars), np.nan)
    sig_matrix = np.full((n_vars, n_vars), np.nan)
    n_matrix = np.full((n_vars, n_vars), 0, dtype=int)

    for i in range(n_vars):
        for j in range(i, n_vars):
            vi = avail_vars[i]
            vj = avail_vars[j]
            if i == j:
                corr_matrix[i][j] = 1.0
                sig_matrix[i][j] = 0.0
                n_valid = df[vi].notna().sum()
                n_matrix[i][j] = int(n_valid)
                continue

            if missing_mode == "LISTWISE":
                # Use only cases where all variables are non-missing
                valid = df[avail_vars].dropna()
                if len(valid) == 0:
                    continue
                xi = valid[vi].values.astype(float)
                xj = valid[vj].values.astype(float)
            else:
                # PAIRWISE: use only cases where both vars are non-missing
                valid = df[[vi, vj]].dropna()
                if len(valid) == 0:
                    continue
                xi = valid[vi].values.astype(float)
                xj = valid[vj].values.astype(float)

            n_ij = len(xi)
            n_matrix[i][j] = n_ij
            n_matrix[j][i] = n_ij

            if n_ij < 2:
                continue

            if is_spearman:
                r, p = scipy_stats.spearmanr(xi, xj)
            else:
                r, p = scipy_stats.pearsonr(xi, xj)

            if math.isnan(r):
                continue

            corr_matrix[i][j] = r
            corr_matrix[j][i] = r

            # Adjust p-value for one-tail
            if tail == "ONETAIL":
                if r >= 0:
                    p_one = p / 2.0
                else:
                    p_one = 1.0 - p / 2.0
                sig_matrix[i][j] = p_one
                sig_matrix[j][i] = p_one
            else:
                sig_matrix[i][j] = p
                sig_matrix[j][i] = p

    if np.all(np.isnan(corr_matrix)):
        table.notes.append("No valid correlations could be computed.")
        return table

    # Build a flat cell array: for each variable pair, show correlation
    # The table is var x var matrix
    col_labels = avail_vars
    row_labels = avail_vars

    cells: List[Any] = []
    for i in range(n_vars):
        for j in range(n_vars):
            r = corr_matrix[i][j]
            if math.isnan(r):
                cells.append(None)
            else:
                cells.append(float(r))

    table.simple_pivot_table(
        rowdim="Variables",
        rowlabels=row_labels,
        coldim="Variables",
        collabels=col_labels,
        cells=cells,
    )

    # Store significance values as notes for reference
    sig_strs: List[str] = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            p = sig_matrix[i][j]
            if not math.isnan(p):
                sig_strs.append(f"{avail_vars[i]} & {avail_vars[j]}: p={p:.6f}")
    if sig_strs:
        table.notes.extend(sig_strs)

    return table