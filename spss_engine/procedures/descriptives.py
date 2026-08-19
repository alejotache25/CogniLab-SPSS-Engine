"""
DESCRIPTIVES procedure for SPSS engine.

Computes descriptive statistics for numeric variables:
  Mean, Std Dev, Minimum, Maximum, Range, Sum, Variance,
  Skewness, Kurtosis, SEMean.

Subcommands:
  VARIABLES=varlist [(zvarname)]   (required)
  /STATISTICS=DEFAULT|MEAN|STDDEV|MINIMUM|MAXIMUM|RANGE|SUM|VARIANCE|SKEWNESS|KURTOSIS|SEMEAN|ALL|NONE
  /SAVE                            (save z-scores as new variables)
  /MISSING=VARIABLE|LISTWISE|INCLUDE
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
import numpy as np

from spss_engine.data.dataset import Dataset
from spss_engine.data.variable import Variable
from spss_engine.output.pivot_table import PivotTable, CellText, FormatSpec
from spss_engine.parser.ast_nodes import CommandNode
from spss_engine.utils.errors import SPSSRuntimeError


def execute_descriptives(cmd: CommandNode, ds: Dataset) -> List[PivotTable]:
    """Execute DESCRIPTIVES command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    # Parse VARIABLES subcommand
    var_names: List[str] = []
    zscore_names: Dict[str, Optional[str]] = {}
    for sc in cmd.subcommands:
        if sc.name.upper() == "VARIABLES" or sc.name == "_MAIN":
            # Variables may include (zvarname) in raw_tokens
            var_names = ds.get_varlist(sc.variables)
            # Check for z-score names in parens
            if sc.raw_tokens:
                # raw_tokens contains parenthesized items
                for i, vn in enumerate(var_names):
                    if i < len(sc.raw_tokens) and sc.raw_tokens[i]:
                        zscore_names[vn] = sc.raw_tokens[i]
            break

    if not var_names:
        raise SPSSRuntimeError("DESCRIPTIVES requires VARIABLES subcommand")

    # Parse STATISTICS
    stats: Set[str] = set()
    save_zscores = False
    missing_mode = "VARIABLE"  # default

    for sc in cmd.subcommands:
        uname = sc.name.upper()
        if uname == "STATISTICS":
            for kw in sc.keywords:
                stats.add(kw.upper())
        elif uname == "SAVE":
            save_zscores = True
        elif uname == "MISSING":
            for kw in sc.keywords:
                missing_mode = kw.upper()

    # Default statistics
    if not stats or stats == {"DEFAULT"}:
        stats = {"MEAN", "STDDEV", "MINIMUM", "MAXIMUM"}
    if "ALL" in stats:
        stats = {"MEAN", "STDDEV", "MINIMUM", "MAXIMUM", "RANGE",
                 "SUM", "VARIANCE", "SKEWNESS", "KURTOSIS", "SEMEAN"}
    if "NONE" in stats:
        stats = set()

    # Check N=0
    df = ds.get_filtered_df()
    if len(df) == 0:
        # N=0: produce empty table + warning (handled by caller)
        table = PivotTable(title="Descriptive Statistics")
        table.simple_pivot_table(
            rowdim="Variable", rowlabels=[],
            coldim="Statistic", collabels=["N"],
            cells=[],
        )
        tables.append(table)
        return tables

    # Build table
    stat_labels: List[str] = ["N"]
    if "MEAN" in stats:
        stat_labels.append("Mean")
    if "SEMEAN" in stats:
        stat_labels.append("Std. Error of Mean")
    if "STDDEV" in stats:
        stat_labels.append("Std. Deviation")
    if "VARIANCE" in stats:
        stat_labels.append("Variance")
    if "MINIMUM" in stats:
        stat_labels.append("Minimum")
    if "MAXIMUM" in stats:
        stat_labels.append("Maximum")
    if "RANGE" in stats:
        stat_labels.append("Range")
    if "SUM" in stats:
        stat_labels.append("Sum")
    if "SKEWNESS" in stats:
        stat_labels.extend(["Skewness", "Std. Error of Skewness"])
    if "KURTOSIS" in stats:
        stat_labels.extend(["Kurtosis", "Std. Error of Kurtosis"])

    row_labels: List[str] = []
    cell_values: List[Any] = []

    for vn in var_names:
        if not ds.has_variable(vn):
            raise SPSSRuntimeError(f"Variable not found: {vn}")
        var = ds.get_variable(vn)
        if not var.is_numeric:
            continue  # Skip string variables

        col = df[vn].dropna()
        n = len(col)
        row_labels.append(vn)
        cell_values.append(float(n))

        if n == 0:
            # Fill with NaN for all stats
            for _ in range(len(stat_labels) - 1):
                cell_values.append(np.nan)
            continue

        arr = col.values.astype(float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else np.nan
        var_val = float(np.var(arr, ddof=1)) if n > 1 else 0.0
        minimum = float(np.min(arr))
        maximum = float(np.max(arr))
        rng = maximum - minimum
        total = float(np.sum(arr))

        # Skewness and Kurtosis (SPSS formulas)
        if n > 2:
            m3 = np.sum((arr - mean) ** 3) / (n - 1)
            m4 = np.sum((arr - mean) ** 4) / (n - 1)
            s = std
            if s > 0:
                skewness = m3 / (s ** 3)
                kurtosis = m4 / (s ** 4) - 3.0
            else:
                skewness = 0.0
                kurtosis = 0.0
            se_skew = np.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))) if n > 2 else np.nan
            se_kurt = 2.0 * se_skew * np.sqrt((n ** 2 - 1) / ((n - 3) * (n + 5))) if n > 3 else np.nan
        else:
            skewness = np.nan
            kurtosis = np.nan
            se_skew = np.nan
            se_kurt = np.nan

        if "MEAN" in stats:
            cell_values.append(mean)
        if "SEMEAN" in stats:
            cell_values.append(sem)
        if "STDDEV" in stats:
            cell_values.append(std)
        if "VARIANCE" in stats:
            cell_values.append(var_val)
        if "MINIMUM" in stats:
            cell_values.append(minimum)
        if "MAXIMUM" in stats:
            cell_values.append(maximum)
        if "RANGE" in stats:
            cell_values.append(rng)
        if "SUM" in stats:
            cell_values.append(total)
        if "SKEWNESS" in stats:
            cell_values.extend([skewness, se_skew])
        if "KURTOSIS" in stats:
            cell_values.extend([kurtosis, se_kurt])

        # Save z-scores if requested
        if save_zscores:
            zname = zscore_names.get(vn) or f"Z" + vn
            z_scores = (col - mean) / std if std > 0 else np.zeros(len(col))
            ds.df.loc[col.index, zname] = z_scores
            if not ds.has_variable(zname):
                ds.add_variable(Variable(name=zname, var_type="numeric"))

    # Build pivot table
    table = PivotTable(title="Descriptive Statistics")
    table.simple_pivot_table(
        rowdim="Variable", rowlabels=row_labels,
        coldim="Statistic", collabels=stat_labels,
        cells=cell_values,
    )
    tables.append(table)
    return tables