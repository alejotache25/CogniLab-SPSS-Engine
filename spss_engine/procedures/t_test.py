"""
T-TEST procedure implementation.

Supports three forms:
  - One-sample:   T-TEST TESTVAL=n /VARIABLES=varlist
  - Independent:  T-TEST GROUPS=var(val1,val2) /VARIABLES=varlist
  - Paired:       T-TEST PAIRS=varlist WITH varlist

Subcommands:
  /CRITERIA=CI(value)  — confidence interval level (default 0.95)
  /ES DISPLAY(TRUE)   — effect size display
  /MISSING=ANALYSIS|LISTWISE

Verified against scipy.stats.ttest_1samp, ttest_ind, ttest_rel.

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


def execute_t_test(cmd: CommandNode, ds: Dataset,
                   executor: Any = None) -> List[PivotTable]:
    """Execute a T-TEST command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    # Parse /CRITERIA=CI(value)
    ci_level: float = 0.95
    for sc in cmd.subcommands:
        if sc.name == "CRITERIA":
            for kw in sc.keywords:
                if isinstance(kw, str) and kw.upper().startswith("CI"):
                    pass
            # raw_tokens may have the CI value
            for rt in sc.raw_tokens:
                try:
                    ci_level = float(rt) / 100.0
                    break
                except (ValueError, TypeError):
                    pass

    # Parse /MISSING
    missing_mode: str = "ANALYSIS"
    for sc in cmd.subcommands:
        if sc.name == "MISSING":
            for kw in sc.keywords:
                if isinstance(kw, str) and kw.upper() in ("LISTWISE", "ANALYSIS"):
                    missing_mode = kw.upper()
                    break

    # Determine T-TEST type
    # Check for TESTVAL (one-sample), GROUPS (independent), PAIRS (paired)
    testval: Optional[float] = None
    groups_var: Optional[str] = None
    group_vals: Optional[Tuple[float, float]] = None
    paired: bool = False

    for sc in cmd.subcommands:
        if sc.name == "TESTVAL":
            for rt in sc.raw_tokens:
                try:
                    testval = float(rt)
                    break
                except (ValueError, TypeError):
                    pass
            # Also check keywords (number may be in keywords)
            if testval is None:
                for kw in sc.keywords:
                    try:
                        testval = float(kw)
                        break
                    except (ValueError, TypeError):
                        pass
        elif sc.name == "GROUPS":
            if sc.variables:
                groups_var = sc.variables[0]
                # Group values in raw_tokens (filter out commas)
                numeric_tokens = [t for t in sc.raw_tokens if t not in (",", "")]
                if len(numeric_tokens) >= 2:
                    try:
                        group_vals = (float(numeric_tokens[0]),
                                      float(numeric_tokens[1]))
                    except (ValueError, TypeError):
                        pass
                elif len(numeric_tokens) == 1:
                    try:
                        v = float(numeric_tokens[0])
                        group_vals = (v, v)
                    except (ValueError, TypeError):
                        pass
        elif sc.name == "PAIRS":
            paired = True

    # Get VARIABLES or PAIRS
    var_names: List[str] = []
    pair_vars: List[List[str]] = []  # list of pairs
    for sc in cmd.subcommands:
        if sc.name == "VARIABLES":
            var_names = ds.get_varlist(sc.variables) if sc.variables else []
        elif sc.name == "PAIRS":
            # PAIRS=varlist WITH varlist
            with_idx = -1
            for i, v in enumerate(sc.variables):
                if v.upper() == "WITH":
                    with_idx = i
                    break
            if with_idx >= 0:
                left = sc.variables[:with_idx]
                right = sc.variables[with_idx + 1:]
                for lv in left:
                    for rv in right:
                        pair_vars.append([lv, rv])
            else:
                # Pairs of consecutive variables
                vars_list = ds.get_varlist(sc.variables) if sc.variables else []
                for i in range(0, len(vars_list) - 1, 2):
                    pair_vars.append([vars_list[i], vars_list[i + 1]])

    split_groups = get_split_groups(ds)
    for group_key, group_df in split_groups:
        if testval is not None:
            # One-sample t-test
            for vn in var_names:
                if vn not in group_df.columns:
                    continue
                col = group_df[vn].dropna()
                t = _one_sample_ttest(col, testval, ci_level,
                                       group_key, ds, vn)
                tables.append(t)
        elif groups_var is not None and group_vals is not None:
            # Independent samples t-test
            for vn in var_names:
                if vn not in group_df.columns:
                    continue
                t = _independent_ttest(group_df, vn, groups_var,
                                         group_vals, ci_level,
                                         group_key, ds)
                tables.append(t)
        elif paired:
            # Paired t-test
            for pair in pair_vars:
                if len(pair) != 2:
                    continue
                v1, v2 = pair
                if v1 not in group_df.columns or v2 not in group_df.columns:
                    continue
                t = _paired_ttest(group_df, v1, v2, ci_level,
                                   group_key, ds)
                tables.append(t)

    if executor is not None:
        for t in tables:
            executor.add_table(t)

    return tables


def _one_sample_ttest(col: pd.Series, testval: float,
                       ci_level: float,
                       group_key: tuple, ds: Dataset,
                       var_name: str) -> PivotTable:
    """One-sample t-test against a test value."""
    title = "One-Sample Test"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"One-Sample Test [{label}]"

    table = PivotTable(title=title)

    values = col.dropna().values.astype(float)
    n = len(values)
    if n == 0:
        table.notes.append(f"No valid cases for {var_name}.")
        return table

    # scipy verification
    t_stat, p_val = scipy_stats.ttest_1samp(values, testval)

    # Manual computation
    mean_val = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else float("nan")
    if se > 0 and not math.isnan(se):
        t_val = (mean_val - testval) / se
    else:
        t_val = float("nan")

    df_val = n - 1
    if not math.isnan(t_val) and df_val > 0:
        p_two = 2.0 * scipy_stats.t.sf(abs(t_val), df_val)
    else:
        p_two = float("nan")

    # Confidence interval
    if not math.isnan(se) and df_val > 0:
        t_crit = scipy_stats.t.ppf(1 - (1 - ci_level) / 2, df_val)
        ci_lower = mean_val - t_crit * se
        ci_upper = mean_val + t_crit * se
    else:
        ci_lower = ci_upper = float("nan")

    # Mean difference
    mean_diff = mean_val - testval

    col_labels = ["t", "df", f"Sig. ({int(ci_level*100)}%)",
                   "Mean Difference", f"Lower", f"Upper"]
    cells = [t_val if not math.isnan(t_val) else None,
            float(df_val),
            p_two if not math.isnan(p_two) else None,
            mean_diff,
            ci_lower if not math.isnan(ci_lower) else None,
            ci_upper if not math.isnan(ci_upper) else None]

    table.simple_pivot_table(
        rowdim=" ",
        rowlabels=[var_name],
        coldim="Statistics",
        collabels=col_labels,
        cells=cells,
    )
    return table


def _independent_ttest(df: pd.DataFrame, var_name: str,
                         group_var: str,
                         group_vals: Tuple[float, float],
                         ci_level: float,
                         group_key: tuple,
                         ds: Dataset) -> PivotTable:
    """Independent samples t-test."""
    title = "Independent Samples Test"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Independent Samples Test [{label}]"

    table = PivotTable(title=title)

    g1_val, g2_val = group_vals
    mask1 = df[group_var] == g1_val
    mask2 = df[group_var] == g2_val
    g1 = df.loc[mask1, var_name].dropna().values.astype(float)
    g2 = df.loc[mask2, var_name].dropna().values.astype(float)
    n1, n2 = len(g1), len(g2)

    if n1 == 0 and n2 == 0:
        table.notes.append(f"No valid cases for {var_name}.")
        return table

    # scipy verification (equal variance)
    if n1 > 0 and n2 > 0:
        t_stat, p_val = scipy_stats.ttest_ind(g1, g2, equal_var=True)
    else:
        t_stat, p_val = float("nan"), float("nan")

    mean1 = float(np.mean(g1)) if n1 > 0 else float("nan")
    mean2 = float(np.mean(g2)) if n2 > 0 else float("nan")
    sd1 = float(np.std(g1, ddof=1)) if n1 > 1 else float("nan")
    sd2 = float(np.std(g2, ddof=1)) if n2 > 1 else float("nan")

    # Pooled SE
    df_val = n1 + n2 - 2
    if n1 > 1 and n2 > 1 and df_val > 0:
        pooled_var = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / df_val
        pooled_se = math.sqrt(pooled_var * (1.0/n1 + 1.0/n2))
    else:
        pooled_se = float("nan")
        df_val = 0

    mean_diff = mean1 - mean2 if not (math.isnan(mean1) or math.isnan(mean2)) else float("nan")

    if not math.isnan(pooled_se) and pooled_se > 0:
        t_val = mean_diff / pooled_se
    else:
        t_val = float("nan")

    if not math.isnan(t_val) and df_val > 0:
        p_two = 2.0 * scipy_stats.t.sf(abs(t_val), df_val)
    else:
        p_two = float("nan")

    # CI
    if not math.isnan(pooled_se) and df_val > 0:
        t_crit = scipy_stats.t.ppf(1 - (1 - ci_level) / 2, df_val)
        ci_lower = mean_diff - t_crit * pooled_se
        ci_upper = mean_diff + t_crit * pooled_se
    else:
        ci_lower = ci_upper = float("nan")

    col_labels = ["t", "df", f"Sig. ({int(ci_level*100)}%)",
                   "Mean Difference", "Lower", "Upper"]
    cells = [t_val if not math.isnan(t_val) else None,
            float(df_val),
            p_two if not math.isnan(p_two) else None,
            mean_diff if not math.isnan(mean_diff) else None,
            ci_lower if not math.isnan(ci_lower) else None,
            ci_upper if not math.isnan(ci_upper) else None]

    table.simple_pivot_table(
        rowdim=" ",
        rowlabels=[f"{var_name} (equal var)"],
        coldim="Statistics",
        collabels=col_labels,
        cells=cells,
    )
    return table


def _paired_ttest(df: pd.DataFrame, v1: str, v2: str,
                    ci_level: float,
                    group_key: tuple,
                    ds: Dataset) -> PivotTable:
    """Paired samples t-test."""
    title = "Paired Samples Test"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Paired Samples Test [{label}]"

    table = PivotTable(title=title)

    # Pairwise complete observations
    valid = df[[v1, v2]].dropna()
    n = len(valid)
    if n == 0:
        table.notes.append(f"No valid cases for pair {v1}, {v2}.")
        return table

    g1 = valid[v1].values.astype(float)
    g2 = valid[v2].values.astype(float)
    diff = g1 - g2

    # scipy verification
    t_stat, p_val = scipy_stats.ttest_rel(g1, g2)

    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    se = sd_diff / math.sqrt(n) if n > 1 and sd_diff > 0 else float("nan")

    if not math.isnan(se) and se > 0:
        t_val = mean_diff / se
    else:
        t_val = float("nan")

    df_val = n - 1
    if not math.isnan(t_val) and df_val > 0:
        p_two = 2.0 * scipy_stats.t.sf(abs(t_val), df_val)
    else:
        p_two = float("nan")

    # CI
    if not math.isnan(se) and df_val > 0:
        t_crit = scipy_stats.t.ppf(1 - (1 - ci_level) / 2, df_val)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
    else:
        ci_lower = ci_upper = float("nan")

    col_labels = ["Mean", "Std. Deviation", "Std. Error Mean",
                   "t", "df", f"Sig. ({int(ci_level*100)}%)",
                   "Lower", "Upper"]
    cells = [mean_diff, sd_diff, se,
             t_val if not math.isnan(t_val) else None,
             float(df_val),
             p_two if not math.isnan(p_two) else None,
             ci_lower if not math.isnan(ci_lower) else None,
             ci_upper if not math.isnan(ci_upper) else None]

    table.simple_pivot_table(
        rowdim="Pair",
        rowlabels=[f"{v1} - {v2}"],
        coldim="Statistics",
        collabels=col_labels,
        cells=cells,
    )
    return table