"""
FREQUENCIES procedure implementation.

Produces:
  - Frequency table: Frequency, Percent, Valid Percent, Cumulative Percent
  - Statistics: MEAN, STDDEV, MIN, MAX, MEDIAN, MODE, RANGE, SUM, VARIANCE,
    SKEWNESS, SESKEW, KURTOSIS, SEKURT, SEMEAN, CV
  - Percentiles (HAVERAGE method)
  - /MISSING=INCLUDE|TABLE (TABLE excludes user-missing from stats but
    shows them in the table; INCLUDE treats user-missing as valid)
  - /HISTOGRAM (produces histogram bin data)

Output: PivotTable objects added to the executor's tables list.

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
from spss_engine.utils.percentile import percentile_haverage
from spss_engine.utils.errors import SPSSRuntimeError
from spss_engine.procedures.base import (
    mean, std, variance, minimum, maximum, range_stat, sum_stat,
    skewness, kurtosis, semean, seskew, sekurt, mode_stat, cv_stat,
    valid_values,
)

logger = logging.getLogger(__name__)

# All supported statistics keywords
_ALL_STATS = {
    "MEAN", "STDDEV", "MINIMUM", "MAXIMUM", "SEMEAN", "VARIANCE",
    "SKEWNESS", "SESKEW", "KURTOSIS", "SEKURT", "RANGE", "MODE",
    "MEDIAN", "SUM", "CV", "ALL", "NONE", "DEFAULT",
}

_DEFAULT_STATS = {"MEAN", "STDDEV", "MINIMUM", "MAXIMUM"}


def execute_frequencies(cmd: CommandNode, ds: Dataset,
                        executor: Any = None) -> List[PivotTable]:
    """Execute a FREQUENCIES command.

    Returns a list of PivotTable objects (frequency table + statistics).
    Also adds them to the executor's tables list if provided.
    """
    tables: List[PivotTable] = []

    # Parse VARIABLES subcommand
    var_names = _parse_variables(cmd, ds)
    if not var_names:
        raise SPSSRuntimeError("FREQUENCIES requires VARIABLES",
                                command="FREQUENCIES")

    # Parse /STATISTICS
    stats = _parse_statistics(cmd)

    # Parse /MISSING
    missing_mode = _parse_missing(cmd)  # "TABLE" or "INCLUDE"

    # Parse /PERCENTILES
    percentiles = _parse_percentiles(cmd)

    # Parse /HISTOGRAM
    histogram = _parse_histogram(cmd)

    # Iterate over split groups
    split_groups = get_split_groups(ds)
    for group_key, group_df in split_groups:
        for var_name in var_names:
            if var_name not in group_df.columns:
                continue
            var = ds.get_variable(var_name)
            col = group_df[var_name]

            # Build frequency table
            freq_table = _build_frequency_table(var_name, var, col,
                                                  missing_mode, group_key, ds)
            tables.append(freq_table)

            # Build statistics table (only for numeric vars)
            # Generate when stats requested OR percentiles requested
            if var.is_numeric and (stats and stats != {"NONE"} or percentiles):
                stat_table = _build_statistics_table(var_name, var, col,
                                                       stats, percentiles,
                                                       group_key, ds)
                if stat_table is not None:
                    tables.append(stat_table)

            # Build histogram data
            if histogram and var.is_numeric:
                hist_table = _build_histogram_table(var_name, col,
                                                     group_key, ds)
                if hist_table is not None:
                    tables.append(hist_table)

    # Add tables to executor
    if executor is not None:
        for t in tables:
            executor.add_table(t)

    return tables


def _parse_variables(cmd: CommandNode, ds: Dataset) -> List[str]:
    """Parse the VARIABLES subcommand or main spec."""
    for sc in cmd.subcommands:
        if sc.name in ("VARIABLES", "_MAIN") and sc.variables:
            return ds.get_varlist(sc.variables)
    # Fallback: try _MAIN
    for sc in cmd.subcommands:
        if sc.variables:
            return ds.get_varlist(sc.variables)
    return []


def _parse_statistics(cmd: CommandNode) -> set[str]:
    """Parse /STATISTICS subcommand. Returns set of stat keywords."""
    for sc in cmd.subcommands:
        if sc.name == "STATISTICS":
            kws = {k.upper() for k in sc.keywords if isinstance(k, str)}
            if "ALL" in kws:
                return _ALL_STATS - {"ALL", "NONE", "DEFAULT"}
            if "DEFAULT" in kws:
                return set(_DEFAULT_STATS)
            if "NONE" in kws:
                return {"NONE"}
            return kws
    return set()  # no /STATISTICS → no stats table


def _parse_missing(cmd: CommandNode) -> str:
    """Parse /MISSING subcommand. Returns 'TABLE' or 'INCLUDE'."""
    for sc in cmd.subcommands:
        if sc.name == "MISSING":
            for k in sc.keywords:
                if isinstance(k, str) and k.upper() == "INCLUDE":
                    return "INCLUDE"
            return "TABLE"
    return "TABLE"


def _parse_percentiles(cmd: CommandNode) -> List[float]:
    """Parse /PERCENTILES subcommand."""
    for sc in cmd.subcommands:
        if sc.name in ("PERCENTILES", "PERCENTILE"):
            result: List[float] = []
            for k in sc.keywords:
                try:
                    result.append(float(k))
                except (ValueError, TypeError):
                    pass
            return result
    return []


def _parse_histogram(cmd: CommandNode) -> bool:
    """Parse /HISTOGRAM subcommand (presence-based)."""
    for sc in cmd.subcommands:
        if sc.name == "HISTOGRAM":
            return True
    return False


def _build_frequency_table(var_name: str, var: Variable,
                            col: pd.Series, missing_mode: str,
                            group_key: tuple,
                            ds: Dataset) -> PivotTable:
    """Build the frequency table for a variable."""
    title = "Frequencies"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Frequencies [{label}]"

    table = PivotTable(title=title)

    # Get value counts
    if var.is_numeric:
        # For numeric: treat NaN as system-missing
        valid_mask = col.notna()
        valid_vals = col[valid_mask]
        # User-missing values
        user_missing_mask = valid_vals.apply(lambda v: var.is_user_missing(v))
        user_missing_vals = valid_vals[user_missing_mask]
        truly_valid = valid_vals[~user_missing_mask]
    else:
        # String
        valid_vals = col[col.notna() & (col != "")]
        truly_valid = valid_vals
        user_missing_vals = pd.Series([], dtype=object)

    # Count frequencies
    total_n = len(col)
    valid_n = len(truly_valid)
    missing_n = total_n - valid_n

    if total_n == 0:
        # N=0: empty table + warning
        table.notes.append("No cases to process.")
        return table

    # Build rows: each distinct value
    if var.is_numeric:
        val_counts = truly_valid.value_counts().sort_index()
    else:
        val_counts = truly_valid.value_counts().sort_index()

    # Labels for row dimension
    row_labels: List[str] = []
    freqs: List[float] = []
    percents: List[float] = []
    valid_pcts: List[float] = []
    cum_pcts: List[float] = []

    cum_valid = 0.0
    for val, count in val_counts.items():
        # Format value label
        if var.is_numeric:
            lbl = _format_value(val, var)
        else:
            lbl = str(val)
        row_labels.append(lbl)
        freqs.append(float(count))
        pct = (count / total_n) * 100.0 if total_n > 0 else 0.0
        percents.append(pct)
        cum_valid += count
        vp = (cum_valid / valid_n) * 100.0 if valid_n > 0 else 0.0
        valid_pcts.append(vp)
        cum_pcts.append(vp)

    # Add user-missing rows (shown in table but not in valid percent)
    if missing_mode == "TABLE" and len(user_missing_vals) > 0:
        um_counts = user_missing_vals.value_counts().sort_index()
        for val, count in um_counts.items():
            lbl = _format_value(val, var)
            row_labels.append(lbl)
            freqs.append(float(count))
            pct = (count / total_n) * 100.0 if total_n > 0 else 0.0
            percents.append(pct)
            valid_pcts.append(float("nan"))  # not shown for missing
            cum_pcts.append(float("nan"))

    # Add system-missing row
    sysmis_count = col.isna().sum() if var.is_numeric else \
        ((col.isna() | (col == "")).sum())
    if sysmis_count > 0:
        row_labels.append("Missing")
        freqs.append(float(sysmis_count))
        pct = (sysmis_count / total_n) * 100.0 if total_n > 0 else 0.0
        percents.append(pct)
        valid_pct_val = float("nan")
        cum_pct_val = float("nan")
        valid_pcts.append(valid_pct_val)
        cum_pcts.append(cum_pct_val)

    # Add Total row
    row_labels.append("Total")
    freqs.append(float(total_n))
    percents.append(100.0)
    valid_pcts.append(100.0 if valid_n > 0 else float("nan"))
    cum_pcts.append(float("nan"))

    # Build pivot table
    table.simple_pivot_table(
        rowdim=var_name,
        rowlabels=row_labels,
        coldim="",
        collabels=["Frequency", "Percent", "Valid Percent",
                    "Cumulative Percent"],
        cells=_flatten_cells(freqs, percents, valid_pcts, cum_pcts),
    )
    return table


def _format_value(val: Any, var: Variable) -> str:
    """Format a value for display in the frequency table."""
    if var.is_numeric:
        f = float(val)
        if math.isnan(f):
            return "."
        if f == int(f):
            return f"{int(f)}"
        return f"{f:.4f}".rstrip("0").rstrip(".")
    return str(val)


def _flatten_cells(freqs, percents, valid_pcts, cum_pcts) -> List[Any]:
    """Flatten 4 column arrays into row-major cell list."""
    cells: List[Any] = []
    n = len(freqs)
    for i in range(n):
        cells.append(freqs[i])
        cells.append(percents[i])
        cells.append(valid_pcts[i])
        cells.append(cum_pcts[i])
    return cells


def _build_statistics_table(var_name: str, var: Variable,
                             col: pd.Series, stats: set[str],
                             percentiles: List[float],
                             group_key: tuple,
                             ds: Dataset) -> Optional[PivotTable]:
    """Build the statistics table for a numeric variable."""
    title = "Statistics"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Statistics [{label}]"

    table = PivotTable(title=title)

    # Get valid values (exclude all missing for stats)
    vals = col.dropna() if var.is_numeric else col[col.notna()]
    v_list = [float(x) for x in vals if not (isinstance(x, float) and
                                             math.isnan(x))]
    n = len(v_list)

    if n == 0:
        table.notes.append(f"No valid cases for {var_name}.")
        return table

    # Build stat label/value pairs
    stat_labels: List[str] = []
    stat_values: List[Any] = []

    if "N" in stats or "ALL" in stats or not stats:
        stat_labels.append("N Valid")
        stat_values.append(float(n))

    if "MEAN" in stats:
        stat_labels.append("Mean")
        stat_values.append(mean(v_list))
    if "STDDEV" in stats:
        stat_labels.append("Std. Deviation")
        stat_values.append(std(v_list))
    if "VARIANCE" in stats:
        stat_labels.append("Variance")
        stat_values.append(variance(v_list))
    if "MINIMUM" in stats:
        stat_labels.append("Minimum")
        stat_values.append(minimum(v_list))
    if "MAXIMUM" in stats:
        stat_labels.append("Maximum")
        stat_values.append(maximum(v_list))
    if "RANGE" in stats:
        stat_labels.append("Range")
        stat_values.append(range_stat(v_list))
    if "SUM" in stats:
        stat_labels.append("Sum")
        stat_values.append(sum_stat(v_list))
    if "SEMEAN" in stats:
        stat_labels.append("Std. Error of Mean")
        stat_values.append(semean(v_list))
    if "SKEWNESS" in stats:
        stat_labels.append("Skewness")
        stat_values.append(skewness(v_list))
    if "SESKEW" in stats:
        stat_labels.append("Std. Error of Skewness")
        stat_values.append(seskew(v_list))
    if "KURTOSIS" in stats:
        stat_labels.append("Kurtosis")
        stat_values.append(kurtosis(v_list))
    if "SEKURT" in stats:
        stat_labels.append("Std. Error of Kurtosis")
        stat_values.append(sekurt(v_list))
    if "MEDIAN" in stats:
        stat_labels.append("Median")
        stat_values.append(percentile_haverage(v_list, 50.0))
    if "MODE" in stats:
        stat_labels.append("Mode")
        stat_values.append(mode_stat(v_list))
    if "CV" in stats:
        stat_labels.append("Coefficient of Variation")
        stat_values.append(cv_stat(v_list))

    # Add percentiles
    for p in percentiles:
        stat_labels.append(f"{int(p) if p == int(p) else p}th Percentile")
        stat_values.append(percentile_haverage(v_list, p))

    if not stat_labels:
        return None

    table.simple_pivot_table(
        rowdim="Statistic",
        rowlabels=stat_labels,
        coldim=var_name,
        collabels=["Value"],
        cells=stat_values,
    )
    return table


def _build_histogram_table(var_name: str, col: pd.Series,
                            group_key: tuple,
                            ds: Dataset) -> Optional[PivotTable]:
    """Build histogram bin data as a simple table."""
    vals = col.dropna()
    v_list = [float(x) for x in vals if not (isinstance(x, float) and
                                             math.isnan(x))]
    if len(v_list) == 0:
        return None

    title = f"Histogram: {var_name}"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Histogram: {var_name} [{label}]"

    table = PivotTable(title=title)
    # Simple: use numpy histogram with ~10 bins
    arr = np.array(v_list)
    counts, edges = np.histogram(arr, bins=10)
    bin_labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}"
                   for i in range(len(counts))]
    table.simple_pivot_table(
        rowdim="Bin",
        rowlabels=bin_labels,
        coldim="Count",
        collabels=["Frequency"],
        cells=[float(c) for c in counts],
    )
    return table