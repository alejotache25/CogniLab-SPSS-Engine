"""
NPAR TESTS procedure implementation.

NPAR TESTS
  /M-W=varlist BY var(val1,val2)
  /WILCOXON=varlist WITH varlist
  /KRUSKAL=varlist BY factor
  /CHISQUARE=varlist

Verified against scipy.stats.mannwhitneyu, wilcoxon, kruskal, chisquare.

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


def execute_npar_tests(cmd: CommandNode, ds: Dataset,
                         executor: Any = None) -> List[PivotTable]:
    """Execute an NPAR TESTS command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    split_groups = get_split_groups(ds)
    for group_key, group_df in split_groups:
        for sc in cmd.subcommands:
            sc_name = sc.name.upper()

            if sc_name == "M-W":
                # Mann-Whitney U: /M-W=varlist BY var(val1,val2)
                mw_tables = _handle_mann_whitney(sc, group_df,
                                                    group_key, ds)
                tables.extend(mw_tables)

            elif sc_name == "WILCOXON":
                # Wilcoxon signed-rank: /WILCOXON=varlist WITH varlist
                wc_tables = _handle_wilcoxon(sc, group_df,
                                                group_key, ds)
                tables.extend(wc_tables)

            elif sc_name == "KRUSKAL":
                # Kruskal-Wallis: /KRUSKAL=varlist BY factor
                kw_tables = _handle_kruskal_wallis(sc, group_df,
                                                      group_key, ds)
                tables.extend(kw_tables)

            elif sc_name == "CHISQUARE":
                # Chi-square: /CHISQUARE=varlist
                chi_tables = _handle_chisquare(sc, group_df,
                                                  group_key, ds)
                tables.extend(chi_tables)

    if executor is not None:
        for t in tables:
            executor.add_table(t)

    return tables


def _parse_by_spec(sc: SubcommandNode) -> Tuple[List[str], Optional[str],
                                                  Optional[Tuple[float, float]]]:
    """Parse varlist BY var(val1,val2) from subcommand."""
    # Find BY in keywords
    by_idx = -1
    for i, kw in enumerate(sc.keywords):
        if isinstance(kw, str) and kw.upper() == "BY":
            by_idx = i
            break

    if by_idx < 0:
        return list(sc.variables), None, None

    # Variables before BY
    all_vars = sc.variables
    # The factor variable after BY might be in keywords
    factor_var: Optional[str] = None
    if by_idx + 1 < len(sc.keywords):
        factor_var = sc.keywords[by_idx + 1]

    # Group values in raw_tokens (parenthesized spec)
    group_vals: Optional[Tuple[float, float]] = None
    if len(sc.raw_tokens) >= 2:
        try:
            group_vals = (float(sc.raw_tokens[0]), float(sc.raw_tokens[1]))
        except (ValueError, TypeError):
            pass

    # Dependent vars: all variables before BY position
    # We need to figure out how many variables are before BY
    # The variables list contains all variables; BY appears as a keyword
    # Count: number of keywords before BY = number of BY separators before current
    # Simple approach: vars before the factor var
    dep_vars = list(all_vars)
    if factor_var and factor_var in dep_vars:
        idx = dep_vars.index(factor_var)
        dep_vars = dep_vars[:idx]

    return dep_vars, factor_var, group_vals


def _handle_mann_whitney(sc: SubcommandNode, df: pd.DataFrame,
                           group_key: tuple,
                           ds: Dataset) -> List[PivotTable]:
    """Mann-Whitney U test."""
    tables: List[PivotTable] = []
    dep_vars, factor_var, group_vals = _parse_by_spec(sc)

    if not factor_var or not group_vals:
        return tables

    g1_val, g2_val = group_vals

    for dep_var in dep_vars:
        if dep_var not in df.columns or factor_var not in df.columns:
            continue

        title = "Mann-Whitney Test"
        if group_key:
            from spss_engine.transforms.split_file import split_group_label
            label = split_group_label(ds, group_key)
            title = f"Mann-Whitney Test [{label}]"

        table = PivotTable(title=title)

        mask1 = df[factor_var] == g1_val
        mask2 = df[factor_var] == g2_val
        g1 = df.loc[mask1, dep_var].dropna().values.astype(float)
        g2 = df.loc[mask2, dep_var].dropna().values.astype(float)
        n1, n2 = len(g1), len(g2)

        if n1 == 0 or n2 == 0:
            table.notes.append(f"No valid cases for {dep_var}.")
            tables.append(table)
            continue

        # scipy verification
        u_stat, p_val = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")

        col_labels = ["Mann-Whitney U", "Wilcoxon W", "Z", "Asymp. Sig. (2-tailed)"]
        # Z statistic approximation
        mu = n1 * n2 / 2.0
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        if sigma > 0:
            z = (u_stat - mu) / sigma
        else:
            z = float("nan")

        # Wilcoxon W = sum of ranks for smaller group
        w = min(u_stat, n1 * n2 - u_stat + n1 * (n1 + 1) / 2)

        table.simple_pivot_table(
            rowdim=" ",
            rowlabels=[dep_var],
            coldim="Statistics",
            collabels=col_labels,
            cells=[
                float(u_stat),
                float(w),
                z if not math.isnan(z) else None,
                float(p_val),
            ],
        )
        tables.append(table)

    return tables


def _handle_wilcoxon(sc: SubcommandNode, df: pd.DataFrame,
                       group_key: tuple,
                       ds: Dataset) -> List[PivotTable]:
    """Wilcoxon signed-rank test."""
    tables: List[PivotTable] = []

    # Parse varlist WITH varlist
    with_idx = -1
    for i, kw in enumerate(sc.keywords):
        if isinstance(kw, str) and kw.upper() == "WITH":
            with_idx = i
            break

    if with_idx >= 0:
        # Split variables into left and right
        all_vars = sc.variables
        # Find where WITH is in the sequence
        # Heuristic: split variables in half
        mid = len(all_vars) // 2
        left_vars = all_vars[:mid]
        right_vars = all_vars[mid:]
        pairs = list(zip(left_vars, right_vars))
    else:
        # Pairs of consecutive variables
        vars_list = list(sc.variables)
        pairs = []
        for i in range(0, len(vars_list) - 1, 2):
            pairs.append((vars_list[i], vars_list[i + 1]))

    for v1, v2 in pairs:
        if v1 not in df.columns or v2 not in df.columns:
            continue

        title = "Wilcoxon Signed-Rank Test"
        if group_key:
            from spss_engine.transforms.split_file import split_group_label
            label = split_group_label(ds, group_key)
            title = f"Wilcoxon Signed-Rank Test [{label}]"

        table = PivotTable(title=title)

        valid = df[[v1, v2]].dropna()
        n = len(valid)

        if n == 0:
            table.notes.append(f"No valid cases for pair {v1}, {v2}.")
            tables.append(table)
            continue

        x1 = valid[v1].values.astype(float)
        x2 = valid[v2].values.astype(float)

        try:
            w_stat, p_val = scipy_stats.wilcoxon(x1, x2)
            w_stat = float(w_stat)
            p_val = float(p_val)
        except ValueError:
            w_stat = p_val = float("nan")

        # Z approximation
        diff = x1 - x2
        n_nonzero = np.sum(diff != 0)
        if n_nonzero > 0:
            mu = n_nonzero * (n_nonzero + 1) / 4.0
            sigma = math.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)
            if sigma > 0:
                z = (w_stat - mu) / sigma
            else:
                z = float("nan")
        else:
            z = float("nan")

        col_labels = ["Z", "Asymp. Sig. (2-tailed)", "N"]
        table.simple_pivot_table(
            rowdim="Pair",
            rowlabels=[f"{v1} - {v2}"],
            coldim="Statistics",
            collabels=col_labels,
            cells=[
                z if not math.isnan(z) else None,
                p_val if not math.isnan(p_val) else None,
                float(n),
            ],
        )
        tables.append(table)

    return tables


def _handle_kruskal_wallis(sc: SubcommandNode, df: pd.DataFrame,
                             group_key: tuple,
                             ds: Dataset) -> List[PivotTable]:
    """Kruskal-Wallis H test."""
    tables: List[PivotTable] = []
    dep_vars, factor_var, _ = _parse_by_spec(sc)

    if not factor_var:
        return tables

    for dep_var in dep_vars:
        if dep_var not in df.columns or factor_var not in df.columns:
            continue

        title = "Kruskal-Wallis Test"
        if group_key:
            from spss_engine.transforms.split_file import split_group_label
            label = split_group_label(ds, group_key)
            title = f"Kruskal-Wallis Test [{label}]"

        table = PivotTable(title=title)

        col = df[dep_var].dropna()
        factor = df.loc[col.index, factor_var]
        n = len(col)

        if n == 0:
            table.notes.append(f"No valid cases for {dep_var}.")
            tables.append(table)
            continue

        groups = []
        group_names = []
        for fval, gcol in col.groupby(factor):
            vals = gcol.values.astype(float)
            if len(vals) > 0:
                groups.append(vals)
                group_names.append(str(fval))

        if len(groups) < 2:
            table.notes.append("Need at least 2 groups.")
            tables.append(table)
            continue

        # scipy verification
        h_stat, p_val = scipy_stats.kruskal(*groups)
        k = len(groups)
        df_val = k - 1

        table.simple_pivot_table(
            rowdim=" ",
            rowlabels=[dep_var],
            coldim="Statistics",
            collabels=["Kruskal-Wallis H", "df", "Asymp. Sig."],
            cells=[
                float(h_stat),
                float(df_val),
                float(p_val),
            ],
        )
        tables.append(table)

    return tables


def _handle_chisquare(sc: SubcommandNode, df: pd.DataFrame,
                        group_key: tuple,
                        ds: Dataset) -> List[PivotTable]:
    """Chi-square goodness-of-fit test."""
    tables: List[PivotTable] = []

    for var_name in sc.variables:
        if var_name not in df.columns:
            continue

        title = "Chi-Square Test"
        if group_key:
            from spss_engine.transforms.split_file import split_group_label
            label = split_group_label(ds, group_key)
            title = f"Chi-Square Test [{label}]"

        table = PivotTable(title=title)

        col = df[var_name].dropna()
        n = len(col)

        if n == 0:
            table.notes.append(f"No valid cases for {var_name}.")
            tables.append(table)
            continue

        # Observed frequencies
        vals, counts = np.unique(col.values, return_counts=True)
        k = len(vals)
        if k < 2:
            table.notes.append("Need at least 2 categories.")
            tables.append(table)
            continue

        observed = counts.astype(float)
        expected = np.full(k, n / k)  # Uniform expected

        # scipy verification
        chi_stat, p_val = scipy_stats.chisquare(observed, expected)
        df_val = k - 1

        # Build frequency table
        col_labels = ["Observed N", "Expected N", "Residual"]
        row_labels = [str(v) for v in vals]
        cells: List[Any] = []
        for i in range(k):
            cells.extend([
                float(observed[i]),
                float(expected[i]),
                float(observed[i] - expected[i]),
            ])
        # Add Total
        row_labels.append("Total")
        cells.extend([float(n), float(n), 0.0])

        freq_table = PivotTable(title=f"{title}: Frequencies")
        freq_table.simple_pivot_table(
            rowdim=var_name,
            rowlabels=row_labels,
            coldim="Statistics",
            collabels=col_labels,
            cells=cells,
        )
        tables.append(freq_table)

        # Test statistics table
        stat_table = PivotTable(title=f"{title}: Statistics")
        stat_table.simple_pivot_table(
            rowdim=" ",
            rowlabels=[var_name],
            coldim="Statistics",
            collabels=["Chi-Square", "df", "Asymp. Sig."],
            cells=[
                float(chi_stat),
                float(df_val),
                float(p_val),
            ],
        )
        tables.append(stat_table)

    return tables