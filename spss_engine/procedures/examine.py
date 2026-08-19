"""
EXAMINE procedure implementation.

EXAMINE VARIABLES=varlist
  BY factor
  /PLOT=BOXPLOT
  /STATISTICS=DESCRIPTIVES

Produces descriptive statistics and tests of normality
(K-S and Shapiro-Wilk).

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
from spss_engine.procedures.base import (
    mean, std, variance, minimum, maximum, skewness, kurtosis, semean,
)
from spss_engine.utils.errors import SPSSRuntimeError

logger = logging.getLogger(__name__)


def execute_examine(cmd: CommandNode, ds: Dataset,
                      executor: Any = None) -> List[PivotTable]:
    """Execute an EXAMINE command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    # Parse VARIABLES=varlist (optionally BY factor)
    var_names: List[str] = []
    by_var: Optional[str] = None
    for sc in cmd.subcommands:
        if sc.name in ("VARIABLES", "_MAIN"):
            # Find BY keyword
            for i, kw in enumerate(sc.keywords):
                if isinstance(kw, str) and kw.upper() == "BY":
                    # After BY is the factor variable
                    if i + 1 < len(sc.keywords):
                        by_var = sc.keywords[i + 1]
                    break
            var_names = ds.get_varlist(sc.variables) if sc.variables else []
            break

    if not var_names:
        raise SPSSRuntimeError("EXAMINE requires VARIABLES",
                                command="EXAMINE")

    # Parse /STATISTICS
    do_descriptives = True
    for sc in cmd.subcommands:
        if sc.name == "STATISTICS":
            for kw in sc.keywords:
                if isinstance(kw, str) and kw.upper() == "DESCRIPTIVES":
                    do_descriptives = True

    split_groups = get_split_groups(ds)
    for group_key, group_df in split_groups:
        for var_name in var_names:
            if var_name not in group_df.columns:
                continue

            if by_var and by_var in group_df.columns:
                # Group by BY variable
                for fval, subgroup in group_df.groupby(by_var, sort=True):
                    vals = subgroup[var_name].dropna().values.astype(float)
                    desc_table = _build_descriptives(var_name, fval, vals,
                                                       group_key, ds)
                    tables.append(desc_table)
                    norm_table = _build_normality_tests(var_name, fval, vals,
                                                          group_key, ds)
                    tables.append(norm_table)
            else:
                vals = group_df[var_name].dropna().values.astype(float)
                desc_table = _build_descriptives(var_name, None, vals,
                                                   group_key, ds)
                tables.append(desc_table)
                norm_table = _build_normality_tests(var_name, None, vals,
                                                       group_key, ds)
                tables.append(norm_table)

    if executor is not None:
        for t in tables:
            executor.add_table(t)

    return tables


def _build_descriptives(var_name: str, group_label: Optional[Any],
                         vals: np.ndarray,
                         group_key: tuple,
                         ds: Dataset) -> PivotTable:
    """Build descriptive statistics table."""
    title = "Descriptives"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Descriptives [{label}]"
    if group_label is not None:
        title = f"Descriptives: {var_name} = {group_label}"

    table = PivotTable(title=title)
    n = len(vals)

    if n == 0:
        table.notes.append(f"No valid cases for {var_name}.")
        return table

    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 and not math.isnan(sd) else float("nan")
    mn = float(np.min(vals))
    mx = float(np.max(vals))
    rng = mx - mn
    med = float(np.median(vals))
    sk = skewness(vals) if n >= 3 else float("nan")
    kt = kurtosis(vals) if n >= 4 else float("nan")

    # CI for mean
    if not math.isnan(se) and n > 1:
        t_crit = scipy_stats.t.ppf(0.975, n - 1)
        ci_lower = m - t_crit * se
        ci_upper = m + t_crit * se
    else:
        ci_lower = ci_upper = float("nan")

    col_labels = ["Statistic", "Value"]
    row_labels = [
        "Mean", "Std. Deviation", "Std. Error", "95% CI Lower",
        "95% CI Upper", "Minimum", "Maximum", "Range", "Median",
        "Skewness", "Kurtosis", "N",
    ]
    cells = [
        m, sd if not math.isnan(sd) else None,
        se if not math.isnan(se) else None,
        ci_lower if not math.isnan(ci_lower) else None,
        ci_upper if not math.isnan(ci_upper) else None,
        mn, mx, rng, med,
        sk if not math.isnan(sk) else None,
        kt if not math.isnan(kt) else None,
        float(n),
    ]

    table.simple_pivot_table(
        rowdim="Statistic",
        rowlabels=row_labels,
        coldim=" ",
        collabels=["Value"],
        cells=cells,
    )
    return table


def _build_normality_tests(var_name: str,
                             group_label: Optional[Any],
                             vals: np.ndarray,
                             group_key: tuple,
                             ds: Dataset) -> PivotTable:
    """Build Tests of Normality table (K-S and Shapiro-Wilk)."""
    title = "Tests of Normality"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Tests of Normality [{label}]"

    table = PivotTable(title=title)
    n = len(vals)

    if n == 0:
        table.notes.append(f"No valid cases for {var_name}.")
        return table

    # Kolmogorov-Smirnov (Lilliefors correction approximation)
    # scipy stats.kstest against normal with estimated parameters
    ks_stat: Any = None
    ks_p: Any = None
    if n >= 3:
        try:
            ks_stat, ks_p = scipy_stats.kstest(vals, "norm",
                                                 args=(np.mean(vals),
                                                       np.std(vals, ddof=1)))
            ks_stat = float(ks_stat)
            ks_p = float(ks_p)
        except Exception:
            ks_stat = ks_p = None

    # Shapiro-Wilk
    sw_stat: Any = None
    sw_p: Any = None
    if 3 <= n <= 5000:
        try:
            sw_stat, sw_p = scipy_stats.shapiro(vals)
            sw_stat = float(sw_stat)
            sw_p = float(sw_p)
        except Exception:
            sw_stat = sw_p = None

    label_str = str(group_label) if group_label is not None else var_name

    col_labels = ["Kolmogorov-Smirnov Statistic", "K-S df", "K-S Sig.",
                   "Shapiro-Wilk Statistic", "Shapiro-Wilk df", "Shapiro-Wilk Sig."]
    row_labels = [label_str]

    cells = [
        ks_stat if ks_stat is not None else None,
        float(n),
        ks_p if ks_p is not None else None,
        sw_stat if sw_stat is not None else None,
        float(n),
        sw_p if sw_p is not None else None,
    ]

    table.simple_pivot_table(
        rowdim="Variable",
        rowlabels=row_labels,
        coldim="Tests",
        collabels=col_labels,
        cells=cells,
    )
    return table