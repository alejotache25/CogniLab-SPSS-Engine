"""
CROSSTABS procedure for SPSS engine.

Produces contingency tables (crosstabulation) between categorical variables.

Subcommands:
  TABLES=varlist BY varlist [BY varlist...]   (required, general mode)
  /CELLS=COUNT|ROW|COLUMN|TOTAL|EXPECTED|RESID|SRESID|ASRESID|ALL|NONE
  /STATISTICS=CHISQ|PHI|CC|LAMBDA|UC|BTAU|CTAU|GAMMA|D|ETA|CORR|KAPPA|MCNEMAR|RISK|ALL|NONE
  /MISSING=TABLE|INCLUDE|REPORT
  /FORMAT=AVALUE|DVALUE|TABLES|NOTABLES
  /COUNT=CELL|CASE|ASIS|ROUND|TRUNCATE
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from spss_engine.data.dataset import Dataset
from spss_engine.output.pivot_table import PivotTable, CellText, FormatSpec
from spss_engine.parser.ast_nodes import CommandNode
from spss_engine.utils.errors import SPSSRuntimeError


def execute_crosstabs(cmd: CommandNode, ds: Dataset) -> List[PivotTable]:
    """Execute CROSSTABS command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    # Parse TABLES subcommand: varlist BY varlist [BY varlist...]
    row_vars: List[str] = []
    col_vars: List[str] = []
    layer_vars: List[str] = []

    for sc in cmd.subcommands:
        if sc.name.upper() in ("TABLES", "_MAIN"):
            # Build combined list preserving order: variables + BY keywords
            # The parser puts vars in sc.variables and BY in sc.keywords
            # We need to reconstruct the original order
            combined: List[str] = []
            # Use keywords which has all items in order (including BY)
            all_items = sc.keywords if sc.keywords else sc.variables
            for v in all_items:
                combined.append(v)
            # Split by BY keyword
            groups: List[List[str]] = []
            current: List[str] = []
            for v in combined:
                if v.upper() == "BY":
                    groups.append(current)
                    current = []
                else:
                    current.append(v)
            groups.append(current)

            if len(groups) >= 1:
                row_vars = ds.get_varlist(groups[0])
            if len(groups) >= 2:
                col_vars = ds.get_varlist(groups[1])
            if len(groups) >= 3:
                layer_vars = ds.get_varlist(groups[2])
            break

    if not row_vars or not col_vars:
        raise SPSSRuntimeError("CROSSTABS requires TABLES=varlist BY varlist")

    # Parse CELLS
    cells_req: Set[str] = set()
    for sc in cmd.subcommands:
        if sc.name.upper() == "CELLS":
            for kw in sc.keywords:
                cells_req.add(kw.upper())
    if "ALL" in cells_req:
        cells_req = {"COUNT", "ROW", "COLUMN", "TOTAL", "EXPECTED",
                     "RESID", "SRESID", "ASRESID"}
    if not cells_req:
        cells_req = {"COUNT"}

    # Parse STATISTICS
    stats_req: Set[str] = set()
    for sc in cmd.subcommands:
        if sc.name.upper() == "STATISTICS":
            for kw in sc.keywords:
                stats_req.add(kw.upper())
    if "ALL" in stats_req:
        stats_req = {"CHISQ", "PHI", "CC", "LAMBDA", "UC", "BTAU",
                     "CTAU", "GAMMA", "D", "ETA", "CORR"}

    # Parse MISSING
    missing_mode = "TABLE"
    for sc in cmd.subcommands:
        if sc.name.upper() == "MISSING":
            for kw in sc.keywords:
                missing_mode = kw.upper()

    # Get filtered data
    df = ds.get_filtered_df()
    if len(df) == 0:
        table = PivotTable(title="Crosstabulation")
        table.simple_pivot_table(
            rowdim=row_vars[0] if row_vars else "",
            rowlabels=[],
            coldim=col_vars[0] if col_vars else "",
            collabels=[],
            cells=[],
        )
        tables.append(table)
        return tables

    # For each row_var x col_var combination
    for rv in row_vars:
        for cv in col_vars:
            if not ds.has_variable(rv) or not ds.has_variable(cv):
                raise SPSSRuntimeError(f"Variable not found: {rv} or {cv}")

            # Drop missing values
            if missing_mode == "TABLE":
                temp_df = df[[rv, cv]].dropna()
            else:
                temp_df = df[[rv, cv]]

            if len(temp_df) == 0:
                continue

            # Build contingency table
            ct = pd.crosstab(temp_df[rv], temp_df[cv])
            row_cats = [str(x) for x in ct.index]
            col_cats = [str(x) for x in ct.columns]

            # Build pivot table
            table = PivotTable(title=f"{rv} * {cv} Crosstabulation")
            cell_values: List[Any] = []

            for ri, row_val in enumerate(ct.index):
                for ci, col_val in enumerate(ct.columns):
                    count = int(ct.iloc[ri, ci])
                    cell_values.append(float(count))

                    if "ROW" in cells_req:
                        row_total = ct.iloc[ri].sum()
                        cell_values.append(float(count / row_total * 100) if row_total > 0 else 0.0)
                    if "COLUMN" in cells_req:
                        col_total = ct.iloc[:, ci].sum()
                        cell_values.append(float(count / col_total * 100) if col_total > 0 else 0.0)
                    if "TOTAL" in cells_req:
                        grand_total = ct.values.sum()
                        cell_values.append(float(count / grand_total * 100) if grand_total > 0 else 0.0)
                    if "EXPECTED" in cells_req:
                        row_total = ct.iloc[ri].sum()
                        col_total = ct.iloc[:, ci].sum()
                        grand_total = ct.values.sum()
                        expected = (row_total * col_total / grand_total) if grand_total > 0 else 0.0
                        cell_values.append(float(expected))
                    if "RESID" in cells_req:
                        row_total = ct.iloc[ri].sum()
                        col_total = ct.iloc[:, ci].sum()
                        grand_total = ct.values.sum()
                        expected = (row_total * col_total / grand_total) if grand_total > 0 else 0.0
                        cell_values.append(float(count - expected))
                    if "SRESID" in cells_req:
                        row_total = ct.iloc[ri].sum()
                        col_total = ct.iloc[:, ci].sum()
                        grand_total = ct.values.sum()
                        expected = (row_total * col_total / grand_total) if grand_total > 0 else 0.0
                        std_resid = (count - expected) / np.sqrt(expected) if expected > 0 else 0.0
                        cell_values.append(float(std_resid))

            # Build column labels based on cells requested
            col_labels: List[str] = []
            for cv_name in col_cats:
                col_labels.append(f"{cv_name} Count" if "COUNT" in cells_req else cv_name)
                if "ROW" in cells_req:
                    col_labels.append(f"{cv_name} % within {rv}")
                if "COLUMN" in cells_req:
                    col_labels.append(f"{cv_name} % within {cv}")
                if "TOTAL" in cells_req:
                    col_labels.append(f"{cv_name} % of Total")
                if "EXPECTED" in cells_req:
                    col_labels.append(f"{cv_name} Expected")
                if "RESID" in cells_req:
                    col_labels.append(f"{cv_name} Residual")
                if "SRESID" in cells_req:
                    col_labels.append(f"{cv_name} Std. Residual")

            # Simpler approach: just count + percentages as separate columns
            # Rebuild with simple format
            table = PivotTable(title=f"{rv} * {cv} Crosstabulation")
            simple_cols = ["Count"]
            if "ROW" in cells_req:
                simple_cols.append("% within " + rv)
            if "COLUMN" in cells_req:
                simple_cols.append("% within " + cv)
            if "EXPECTED" in cells_req:
                simple_cols.append("Expected")
            if "RESID" in cells_req:
                simple_cols.append("Residual")
            if "SRESID" in cells_req:
                simple_cols.append("Std. Residual")

            # Build cells: for each row x col combination
            simple_cells: List[Any] = []
            for ri, row_val in enumerate(ct.index):
                for ci, col_val in enumerate(ct.columns):
                    count = int(ct.iloc[ri, ci])
                    simple_cells.append(float(count))
                    if "ROW" in cells_req:
                        row_total = ct.iloc[ri].sum()
                        simple_cells.append(float(count / row_total * 100) if row_total > 0 else 0.0)
                    if "COLUMN" in cells_req:
                        col_total = ct.iloc[:, ci].sum()
                        simple_cells.append(float(count / col_total * 100) if col_total > 0 else 0.0)
                    if "EXPECTED" in cells_req:
                        row_total = ct.iloc[ri].sum()
                        col_total = ct.iloc[:, ci].sum()
                        grand_total = ct.values.sum()
                        expected = (row_total * col_total / grand_total) if grand_total > 0 else 0.0
                        simple_cells.append(float(expected))
                    if "RESID" in cells_req:
                        row_total = ct.iloc[ri].sum()
                        col_total = ct.iloc[:, ci].sum()
                        grand_total = ct.values.sum()
                        expected = (row_total * col_total / grand_total) if grand_total > 0 else 0.0
                        simple_cells.append(float(count - expected))
                    if "SRESID" in cells_req:
                        row_total = ct.iloc[ri].sum()
                        col_total = ct.iloc[:, ci].sum()
                        grand_total = ct.values.sum()
                        expected = (row_total * col_total / grand_total) if grand_total > 0 else 0.0
                        std_resid = (count - expected) / np.sqrt(expected) if expected > 0 else 0.0
                        simple_cells.append(float(std_resid))

            # Use a combined row label: "row_val * col_val"
            combined_rows = [f"{r} * {c}" for r in row_cats for c in col_cats]
            table.simple_pivot_table(
                rowdim=f"{rv} * {cv}",
                rowlabels=combined_rows,
                coldim="Statistic",
                collabels=simple_cols,
                cells=simple_cells,
            )
            tables.append(table)

            # Chi-square test
            if "CHISQ" in stats_req and ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2, p, dof, expected = scipy_stats.chi2_contingency(ct.values)
                n = ct.values.sum()

                chi_table = PivotTable(title="Chi-Square Tests")
                chi_table.simple_pivot_table(
                    rowdim="Test",
                    rowlabels=["Pearson Chi-Square", "Likelihood Ratio", "N of Valid Cases"],
                    coldim="Value",
                    collabels=["Chi-Square", "df", "Asymp. Sig. (2-sided)"],
                    cells=[chi2, float(dof), p,
                           _likelihood_ratio(ct.values), float(dof), _likelihood_ratio_sig(ct.values),
                           float(n), np.nan, np.nan],
                )
                tables.append(chi_table)

            # Symmetric measures (Phi, Cramér's V)
            if "PHI" in stats_req or "CC" in stats_req:
                n = ct.values.sum()
                chi2_val = scipy_stats.chi2_contingency(ct.values)[0] if ct.shape[0] >= 2 and ct.shape[1] >= 2 else 0.0
                min_dim = min(ct.shape[0], ct.shape[1])
                phi = np.sqrt(chi2_val / n) if n > 0 else 0.0
                cramers_v = phi / np.sqrt(min_dim - 1) if min_dim > 1 else phi
                cc = np.sqrt(chi2_val / (chi2_val + n)) if n > 0 else 0.0

                measures_table = PivotTable(title="Symmetric Measures")
                measures_table.simple_pivot_table(
                    rowdim="Measure",
                    rowlabels=["Phi", "Cramér's V", "Contingency Coefficient"],
                    coldim="Value",
                    collabels=["Value"],
                    cells=[phi, cramers_v, cc],
                )
                tables.append(measures_table)

    return tables


def _likelihood_ratio(observed: np.ndarray) -> float:
    """Compute likelihood ratio chi-square statistic."""
    total = observed.sum()
    if total == 0:
        return 0.0
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums * col_sums / total
    # Avoid log(0)
    mask = (observed > 0) & (expected > 0)
    ratio = 2.0 * np.sum(observed[mask] * np.log(observed[mask] / expected[mask]))
    return float(ratio)


def _likelihood_ratio_sig(observed: np.ndarray) -> float:
    """Compute significance of likelihood ratio chi-square."""
    from scipy.stats import chi2 as chi2_dist
    lr = _likelihood_ratio(observed)
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    if dof == 0:
        return np.nan
    return float(chi2_dist.sf(lr, dof))