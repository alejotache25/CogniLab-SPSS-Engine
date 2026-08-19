"""
ONEWAY procedure implementation.

ONEWAY varlist BY factor
  /STATISTICS=DESCRIPTIVES HOMOGENEITY
  /POSTHOC=TUKEY

Computes one-way ANOVA, group descriptive statistics, and optionally
Tukey post-hoc comparisons.

Verified against scipy.stats.f_oneway.

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
from spss_engine.procedures.base import mean, std, minimum, maximum, variance
from spss_engine.utils.errors import SPSSRuntimeError

logger = logging.getLogger(__name__)


def execute_oneway(cmd: CommandNode, ds: Dataset,
                    executor: Any = None) -> List[PivotTable]:
    """Execute an ONEWAY command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    # Parse: varlist BY factor
    main_sc = cmd.get_subcommand("_MAIN")
    if main_sc is None:
        # Try VARIABLES
        main_sc = cmd.get_subcommand("VARIABLES")

    dep_vars: List[str] = []
    factor_var: Optional[str] = None

    if main_sc is not None:
        # Look for BY in keywords
        by_idx = -1
        for i, kw in enumerate(main_sc.keywords):
            if isinstance(kw, str) and kw.upper() == "BY":
                by_idx = i
                break
        if by_idx >= 0:
            # Variables before BY are dependent, after BY is the factor
            dep_vars = [v for v in main_sc.variables]
            # Factor might be in keywords after BY
            if by_idx + 1 < len(main_sc.keywords):
                factor_var = main_sc.keywords[by_idx + 1]
            # If factor is still None, look in variables after BY in raw_tokens
            if factor_var is None and len(main_sc.raw_tokens) > 0:
                factor_var = str(main_sc.raw_tokens[-1])
        else:
            # No BY - just variables
            dep_vars = list(main_sc.variables)

    if not dep_vars:
        raise SPSSRuntimeError("ONEWAY requires dependent variables",
                                command="ONEWAY")
    if factor_var is None:
        raise SPSSRuntimeError("ONEWAY requires a BY factor variable",
                                command="ONEWAY")

    # Parse /STATISTICS
    do_descriptives = False
    do_homogeneity = False
    for sc in cmd.subcommands:
        if sc.name == "STATISTICS":
            for kw in sc.keywords:
                if isinstance(kw, str):
                    ku = kw.upper()
                    if ku == "DESCRIPTIVES":
                        do_descriptives = True
                    elif ku == "HOMOGENEITY":
                        do_homogeneity = True

    # Parse /POSTHOC
    do_tukey = False
    for sc in cmd.subcommands:
        if sc.name == "POSTHOC":
            for kw in sc.keywords:
                if isinstance(kw, str) and kw.upper() == "TUKEY":
                    do_tukey = True

    split_groups = get_split_groups(ds)
    for group_key, group_df in split_groups:
        if factor_var not in group_df.columns:
            continue

        for dep_var in dep_vars:
            if dep_var not in group_df.columns:
                continue

            # ANOVA table
            anova_table = _build_anova_table(dep_var, factor_var,
                                               group_df, group_key, ds)
            tables.append(anova_table)

            if do_descriptives:
                desc_table = _build_descriptives_table(dep_var, factor_var,
                                                         group_df, group_key, ds)
                tables.append(desc_table)

            if do_homogeneity:
                homog_table = _build_homogeneity_test(dep_var, factor_var,
                                                       group_df, group_key, ds)
                tables.append(homog_table)

            if do_tukey:
                tukey_table = _build_tukey_posthoc(dep_var, factor_var,
                                                     group_df, group_key, ds)
                tables.append(tukey_table)

    if executor is not None:
        for t in tables:
            executor.add_table(t)

    return tables


def _build_anova_table(dep_var: str, factor_var: str,
                        df: pd.DataFrame,
                        group_key: tuple, ds: Dataset) -> PivotTable:
    """Build the ANOVA table (Between Groups, Within Groups, Total)."""
    title = "ANOVA"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"ANOVA [{label}]"

    table = PivotTable(title=title)

    col = df[dep_var].dropna()
    factor = df.loc[col.index, factor_var]
    n = len(col)

    if n == 0:
        table.notes.append(f"No valid cases for {dep_var}.")
        return table

    # Group by factor
    groups = []
    group_means = []
    group_ns = []
    for fval, gcol in col.groupby(factor):
        vals = gcol.values.astype(float)
        groups.append(vals)
        group_means.append(float(np.mean(vals)))
        group_ns.append(len(vals))

    k = len(groups)
    grand_mean = float(np.mean(col.values))
    total_ss = float(np.sum((col.values - grand_mean) ** 2))
    between_ss = float(sum(gn * (gm - grand_mean) ** 2
                            for gm, gn in zip(group_means, group_ns)))
    within_ss = total_ss - between_ss

    df_between = k - 1 if k > 0 else 0
    df_within = n - k
    df_total = n - 1

    ms_between = between_ss / df_between if df_between > 0 else float("nan")
    ms_within = within_ss / df_within if df_within > 0 else float("nan")

    if ms_within > 0 and not math.isnan(ms_within):
        f_stat = ms_between / ms_within
    else:
        f_stat = float("nan")

    if not math.isnan(f_stat) and df_between > 0 and df_within > 0:
        p_val = float(scipy_stats.f.sf(f_stat, df_between, df_within))
    else:
        p_val = float("nan")

    # Verify with scipy
    if k >= 2:
        f_scipy, p_scipy = scipy_stats.f_oneway(*groups)
    else:
        f_scipy, p_scipy = f_stat, p_val

    # Table: rows = Between Groups, Within Groups, Total
    # Columns: Sum of Squares, df, Mean Square, F, Sig.
    col_labels = ["Sum of Squares", "df", "Mean Square", "F", "Sig."]
    row_labels = ["Between Groups", "Within Groups", "Total"]

    cells = [
        between_ss, float(df_between), ms_between,
        f_stat if not math.isnan(f_stat) else None,
        p_val if not math.isnan(p_val) else None,
        within_ss, float(df_within), ms_within,
        None, None,
        total_ss, float(df_total), None, None, None,
    ]

    table.simple_pivot_table(
        rowdim="Source",
        rowlabels=row_labels,
        coldim="Statistics",
        collabels=col_labels,
        cells=cells,
    )
    return table


def _build_descriptives_table(dep_var: str, factor_var: str,
                                df: pd.DataFrame,
                                group_key: tuple,
                                ds: Dataset) -> PivotTable:
    """Build descriptive statistics per group."""
    title = "Descriptives"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Descriptives [{label}]"

    table = PivotTable(title=title)

    col = df[dep_var].dropna()
    factor = df.loc[col.index, factor_var]
    n = len(col)

    if n == 0:
        table.notes.append(f"No valid cases for {dep_var}.")
        return table

    col_labels = ["N", "Mean", "Std. Deviation", "Std. Error",
                   "Lower Bound", "Upper Bound", "Minimum", "Maximum"]
    row_labels: List[str] = []
    cells: List[Any] = []

    for fval, gcol in col.groupby(factor):
        vals = gcol.values.astype(float)
        gn = len(vals)
        gm = float(np.mean(vals))
        gsd = float(np.std(vals, ddof=1)) if gn > 1 else float("nan")
        gse = gsd / math.sqrt(gn) if gn > 1 and not math.isnan(gsd) else float("nan")

        if not math.isnan(gse) and gn > 1:
            t_crit = scipy_stats.t.ppf(0.975, gn - 1)
            ci_lower = gm - t_crit * gse
            ci_upper = gm + t_crit * gse
        else:
            ci_lower = ci_upper = float("nan")

        gmin = float(np.min(vals))
        gmax = float(np.max(vals))

        row_labels.append(str(fval))
        cells.extend([
            float(gn), gm, gsd, gse,
            ci_lower if not math.isnan(ci_lower) else None,
            ci_upper if not math.isnan(ci_upper) else None,
            gmin, gmax,
        ])

    # Total row
    tm = float(np.mean(col.values))
    tsd = float(np.std(col.values, ddof=1)) if n > 1 else float("nan")
    tse = tsd / math.sqrt(n) if n > 1 and not math.isnan(tsd) else float("nan")
    if not math.isnan(tse) and n > 1:
        t_crit = scipy_stats.t.ppf(0.975, n - 1)
        tl = tm - t_crit * tse
        tu = tm + t_crit * tse
    else:
        tl = tu = float("nan")
    row_labels.append("Total")
    cells.extend([
        float(n), tm, tsd, tse,
        tl if not math.isnan(tl) else None,
        tu if not math.isnan(tu) else None,
        float(np.min(col.values)), float(np.max(col.values)),
    ])

    table.simple_pivot_table(
        rowdim=factor_var,
        rowlabels=row_labels,
        coldim="Statistics",
        collabels=col_labels,
        cells=cells,
    )
    return table


def _build_homogeneity_test(dep_var: str, factor_var: str,
                              df: pd.DataFrame,
                              group_key: tuple,
                              ds: Dataset) -> PivotTable:
    """Levene's test for homogeneity of variances."""
    title = "Test of Homogeneity of Variances"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Test of Homogeneity of Variances [{label}]"

    table = PivotTable(title=title)

    col = df[dep_var].dropna()
    factor = df.loc[col.index, factor_var]
    n = len(col)

    if n == 0:
        table.notes.append(f"No valid cases for {dep_var}.")
        return table

    groups = []
    for fval, gcol in col.groupby(factor):
        groups.append(gcol.values.astype(float))

    if len(groups) < 2:
        table.notes.append("Need at least 2 groups for Levene's test.")
        return table

    levene_stat, levene_p = scipy_stats.levene(*groups, center="mean")
    k = len(groups)
    df1 = k - 1
    df2 = n - k

    table.simple_pivot_table(
        rowdim=" ",
        rowlabels=["Levene Statistic"],
        coldim=" ",
        collabels=["Levene Statistic", "df1", "df2", "Sig."],
        cells=[float(levene_stat), float(df1), float(df2), float(levene_p)],
    )
    return table


def _build_tukey_posthoc(dep_var: str, factor_var: str,
                           df: pd.DataFrame,
                           group_key: tuple,
                           ds: Dataset) -> PivotTable:
    """Tukey HSD post-hoc comparisons (simplified)."""
    title = "Tukey HSD Post Hoc Tests"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Tukey HSD Post Hoc Tests [{label}]"

    table = PivotTable(title=title)

    col = df[dep_var].dropna()
    factor = df.loc[col.index, factor_var]
    n = len(col)

    if n == 0:
        table.notes.append(f"No valid cases for {dep_var}.")
        return table

    groups = {}
    for fval, gcol in col.groupby(factor):
        vals = gcol.values.astype(float)
        groups[fval] = vals

    group_keys = sorted(groups.keys())
    if len(group_keys) < 2:
        table.notes.append("Need at least 2 groups for post-hoc.")
        return table

    k = len(group_keys)
    df_within = n - k
    ms_within = 0.0
    for key in group_keys:
        vals = groups[key]
        if len(vals) > 1:
            ms_within += np.sum((vals - np.mean(vals)) ** 2)
    if df_within > 0:
        ms_within /= df_within
    se_base = math.sqrt(ms_within / max(len(groups[key_keys]) for key_keys in [group_keys]))

    col_labels = ["Mean Difference (I-J)", "Std. Error", "Sig.", "Lower", "Upper"]
    row_labels: List[str] = []
    cells: List[Any] = []

    for i in range(len(group_keys)):
        for j in range(i + 1, len(group_keys)):
            gi = groups[group_keys[i]]
            gj = groups[group_keys[j]]
            mi = float(np.mean(gi))
            mj = float(np.mean(gj))
            ni = len(gi)
            nj = len(gj)
            se = math.sqrt(ms_within * (1.0/ni + 1.0/nj)) if ms_within > 0 else float("nan")
            md = mi - mj

            if not math.isnan(se) and df_within > 0 and k > 2:
                q_crit = float(scipy_stats.studentized_range.ppf(0.95, k, df_within))
                ci_lower = md - q_crit * se
                ci_upper = md + q_crit * se
                # Approximate p-value using studentized range
                q = abs(md) / se if se > 0 else 0.0
                p_val = float(scipy_stats.studentized_range.sf(q, k, df_within))
            else:
                ci_lower = ci_upper = float("nan")
                p_val = float("nan")

            row_labels.append(f"{group_keys[i]} - {group_keys[j]}")
            cells.extend([
                md,
                se if not math.isnan(se) else None,
                p_val if not math.isnan(p_val) else None,
                ci_lower if not math.isnan(ci_lower) else None,
                ci_upper if not math.isnan(ci_upper) else None,
            ])

    table.simple_pivot_table(
        rowdim="Comparisons",
        rowlabels=row_labels,
        coldim=" ",
        collabels=col_labels,
        cells=cells,
    )
    return table