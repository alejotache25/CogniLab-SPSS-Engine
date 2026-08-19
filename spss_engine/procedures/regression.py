"""
REGRESSION procedure implementation.

REGRESSION
  /DEPENDENT=var
  /METHOD=ENTER varlist

Computes OLS regression: Model Summary (R, R², Adj R², Std Error),
ANOVA table, and Coefficients table.

Verified against numpy.linalg.lstsq.

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


def execute_regression(cmd: CommandNode, ds: Dataset,
                         executor: Any = None) -> List[PivotTable]:
    """Execute a REGRESSION command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    # Parse /DEPENDENT
    dep_var: Optional[str] = None
    for sc in cmd.subcommands:
        if sc.name == "DEPENDENT" and sc.variables:
            dep_var = sc.variables[0]
            break

    if dep_var is None:
        raise SPSSRuntimeError("REGRESSION requires /DEPENDENT",
                                command="REGRESSION")

    # Parse /METHOD=ENTER varlist
    indep_vars: List[str] = []
    for sc in cmd.subcommands:
        if sc.name == "METHOD":
            # Check for ENTER keyword
            for kw in sc.keywords:
                if isinstance(kw, str) and kw.upper() == "ENTER":
                    break
            indep_vars = ds.get_varlist(sc.variables) if sc.variables else []
            break

    if not indep_vars:
        raise SPSSRuntimeError("REGRESSION requires /METHOD=ENTER with predictors",
                                command="REGRESSION")

    split_groups = get_split_groups(ds)
    for group_key, group_df in split_groups:
        if dep_var not in group_df.columns:
            continue

        # Prepare data: listwise deletion
        all_vars = [dep_var] + indep_vars
        all_vars = [v for v in all_vars if v in group_df.columns]
        valid = group_df[all_vars].dropna()

        n = len(valid)
        if n == 0:
            empty = PivotTable(title="Regression")
            empty.notes.append("No valid cases for regression.")
            tables.append(empty)
            continue

        y = valid[dep_var].values.astype(float)
        X = valid[indep_vars].values.astype(float)
        n_obs = len(y)
        n_pred = len(indep_vars)

        # Design matrix with intercept
        X_design = np.column_stack([np.ones(n_obs), X])
        n_params = n_pred + 1

        # Solve OLS via numpy.linalg.lstsq (verified)
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_design, y, rcond=None)

        y_pred = X_design @ coeffs
        residuals = y - y_pred

        # R-squared
        ss_total = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum(residuals ** 2))
        ss_reg = ss_total - ss_res

        if ss_total > 0:
            r_squared = 1.0 - ss_res / ss_total
        else:
            r_squared = float("nan")

        r_val = math.sqrt(r_squared) if r_squared >= 0 else 0.0
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - n_params) \
            if n_obs > n_params else float("nan")
        std_error = math.sqrt(ss_res / (n_obs - n_params)) if n_obs > n_params else float("nan")

        # F statistic
        df_reg = n_pred
        df_res = n_obs - n_params
        if df_res > 0 and ss_total > 0:
            ms_reg = ss_reg / df_reg
            ms_res = ss_res / df_res
            f_stat = ms_reg / ms_res if ms_res > 0 else float("nan")
            if not math.isnan(f_stat) and df_reg > 0 and df_res > 0:
                p_val = float(scipy_stats.f.sf(f_stat, df_reg, df_res))
            else:
                p_val = float("nan")
        else:
            ms_reg = ms_res = f_stat = p_val = float("nan")

        # Model Summary table
        model_table = PivotTable(title="Model Summary")
        if group_key:
            from spss_engine.transforms.split_file import split_group_label
            label = split_group_label(ds, group_key)
            model_table.title = f"Model Summary [{label}]"

        model_table.simple_pivot_table(
            rowdim="Model",
            rowlabels=["1"],
            coldim="Statistics",
            collabels=["R", "R Square", "Adjusted R Square",
                         "Std. Error of the Estimate"],
            cells=[r_val, r_squared, adj_r_squared if not math.isnan(adj_r_squared) else None,
                    std_error if not math.isnan(std_error) else None],
        )
        tables.append(model_table)

        # ANOVA table
        anova_table = PivotTable(title="ANOVA")
        if group_key:
            from spss_engine.transforms.split_file import split_group_label
            label = split_group_label(ds, group_key)
            anova_table.title = f"ANOVA [{label}]"

        anova_table.simple_pivot_table(
            rowdim="Model",
            rowlabels=["Regression", "Residual", "Total"],
            coldim="Statistics",
            collabels=["Sum of Squares", "df", "Mean Square", "F", "Sig."],
            cells=[
                ss_reg, float(df_reg),
                ms_reg if not math.isnan(ms_reg) else None,
                f_stat if not math.isnan(f_stat) else None,
                p_val if not math.isnan(p_val) else None,
                ss_res, float(df_res),
                ms_res if not math.isnan(ms_res) else None,
                None, None,
                ss_total, float(n_obs - 1),
                None, None, None,
            ],
        )
        tables.append(anova_table)

        # Coefficients table
        coef_table = PivotTable(title="Coefficients")
        if group_key:
            from spss_engine.transforms.split_file import split_group_label
            label = split_group_label(ds, group_key)
            coef_table.title = f"Coefficients [{label}]"

        # Standard errors of coefficients
        if df_res > 0 and rank == n_params:
            try:
                xtx_inv = np.linalg.inv(X_design.T @ X_design)
                se_coeffs = np.sqrt(np.diag(xtx_inv) * (ss_res / df_res))
                t_stats = coeffs / se_coeffs
                p_coeffs = [float(scipy_stats.t.sf(abs(t), df_res) * 2)
                             for t in t_stats]
            except (np.linalg.LinAlgError, ZeroDivisionError):
                se_coeffs = np.full(n_params, np.nan)
                t_stats = np.full(n_params, np.nan)
                p_coeffs = [float("nan")] * n_params
        else:
            se_coeffs = np.full(n_params, np.nan)
            t_stats = np.full(n_params, np.nan)
            p_coeffs = [float("nan")] * n_params

        # 95% CI for coefficients
        ci_lower = []
        ci_upper = []
        if df_res > 0:
            t_crit = scipy_stats.t.ppf(0.975, df_res)
            for i in range(n_params):
                if not math.isnan(se_coeffs[i]):
                    ci_lower.append(coeffs[i] - t_crit * se_coeffs[i])
                    ci_upper.append(coeffs[i] + t_crit * se_coeffs[i])
                else:
                    ci_lower.append(None)
                    ci_upper.append(None)
        else:
            ci_lower = [None] * n_params
            ci_upper = [None] * n_params

        # Standardized coefficients (beta)
        std_y = np.std(y, ddof=1) if n_obs > 1 else float("nan")
        betas: List[Any] = []
        for j, iv in enumerate(indep_vars):
            std_xj = np.std(valid[iv].values.astype(float), ddof=1) if n_obs > 1 else float("nan")
            if std_y > 0 and std_xj > 0 and not math.isnan(std_y) and not math.isnan(std_xj):
                betas.append(coeffs[j + 1] * std_xj / std_y)
            else:
                betas.append(None)

        row_labels = ["(Constant)"] + indep_vars
        col_labels = ["B", "Std. Error", "Beta", "t", "Sig.",
                        "Lower 95%", "Upper 95%"]

        cells_coef: List[Any] = []
        for i, lbl in enumerate(row_labels):
            cells_coef.extend([
                float(coeffs[i]),
                se_coeffs[i] if not math.isnan(se_coeffs[i]) else None,
                betas[i - 1] if i > 0 and betas[i - 1] is not None else None,
                t_stats[i] if not math.isnan(t_stats[i]) else None,
                p_coeffs[i] if not math.isnan(p_coeffs[i]) else None,
                ci_lower[i],
                ci_upper[i],
            ])

        coef_table.simple_pivot_table(
            rowdim="Variables",
            rowlabels=row_labels,
            coldim="Statistics",
            collabels=col_labels,
            cells=cells_coef,
        )
        tables.append(coef_table)

    if executor is not None:
        for t in tables:
            executor.add_table(t)

    return tables