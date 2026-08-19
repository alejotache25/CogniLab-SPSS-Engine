"""
FACTOR procedure implementation.

FACTOR
  /VARIABLES=varlist
  /EXTRACTION=PC
  /CRITERIA=FACTORS(n)

Principal components analysis:
  - KMO and Bartlett's test
  - Communalities
  - Total Variance Explained
  - Component Matrix

Verified against numpy.linalg.eigvalsh.

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


def execute_factor(cmd: CommandNode, ds: Dataset,
                     executor: Any = None) -> List[PivotTable]:
    """Execute a FACTOR command. Returns list of PivotTable objects."""
    tables: List[PivotTable] = []

    # Parse /VARIABLES
    var_names: List[str] = []
    for sc in cmd.subcommands:
        if sc.name == "VARIABLES":
            var_names = ds.get_varlist(sc.variables) if sc.variables else []
            break

    if not var_names:
        raise SPSSRuntimeError("FACTOR requires /VARIABLES",
                                command="FACTOR")

    # Parse /CRITERIA=FACTORS(n)
    n_factors: Optional[int] = None
    for sc in cmd.subcommands:
        if sc.name == "CRITERIA":
            for i, kw in enumerate(sc.keywords):
                if isinstance(kw, str) and kw.upper() == "FACTORS":
                    if i + 1 < len(sc.keywords):
                        try:
                            n_factors = int(float(sc.keywords[i + 1]))
                        except (ValueError, TypeError):
                            pass

    split_groups = get_split_groups(ds)
    for group_key, group_df in split_groups:
        avail_vars = [v for v in var_names if v in group_df.columns]
        if not avail_vars:
            empty = PivotTable(title="Factor Analysis")
            empty.notes.append("No valid variables found.")
            tables.append(empty)
            continue

        valid = group_df[avail_vars].dropna()
        n = len(valid)
        k = len(avail_vars)

        if n < 3 or k < 2:
            empty = PivotTable(title="Factor Analysis")
            if group_key:
                from spss_engine.transforms.split_file import split_group_label
                label = split_group_label(ds, group_key)
                empty.title = f"Factor Analysis [{label}]"
            empty.notes.append("Insufficient cases or variables.")
            tables.append(empty)
            continue

        data = valid[avail_vars].values.astype(float)

        # Standardize
        data_std = (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)

        # Correlation matrix
        corr = np.corrcoef(data_std, rowvar=False)

        # KMO and Bartlett's test
        kmo_table = _build_kmo_bartlett(corr, k, n, avail_vars,
                                          group_key, ds)
        tables.append(kmo_table)

        # Eigenvalues via numpy.linalg.eigvalsh (verified)
        eigenvalues = np.linalg.eigvalsh(corr)
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending

        # Total Variance Explained
        var_table = _build_variance_explained(eigenvalues, k, n_factors,
                                                group_key, ds)
        tables.append(var_table)

        # Communalities
        comm_table = _build_communalities(eigenvalues, k, avail_vars,
                                            corr, n_factors, group_key, ds)
        tables.append(comm_table)

        # Component Matrix (loadings)
        # Eigenvectors for the selected components
        eigvals, eigvecs = np.linalg.eigh(corr)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        n_comp = n_factors if n_factors else k
        loadings = eigvecs[:, :n_comp] * np.sqrt(eigvals[:n_comp])

        comp_table = _build_component_matrix(loadings, avail_vars, n_comp,
                                               group_key, ds)
        tables.append(comp_table)

    if executor is not None:
        for t in tables:
            executor.add_table(t)

    return tables


def _build_kmo_bartlett(corr: np.ndarray, k: int, n: int,
                          avail_vars: List[str],
                          group_key: tuple,
                          ds: Dataset) -> PivotTable:
    """Build KMO and Bartlett's test table."""
    title = "KMO and Bartlett's Test"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"KMO and Bartlett's Test [{label}]"

    table = PivotTable(title=title)

    # KMO: simplified Kaiser-Meyer-Olkin
    # KMO = sum(r_ij^2) / (sum(r_ij^2 + sum(a_ij^2))
    # where a_ij is the anti-image correlation
    try:
        inv_corr = np.linalg.inv(corr)
        # Anti-image: -inv_corr[i,j] / sqrt(inv_corr[i,i] * inv_corr[j,j])
        d = np.sqrt(np.diag(inv_corr))
        anti_image = -inv_corr / np.outer(d, d)
        np.fill_diagonal(anti_image, 1.0)

        r2_sum = np.sum(np.triu(corr ** 2, k=1))
        a2_sum = np.sum(np.triu(anti_image ** 2, k=1))
        if (r2_sum + a2_sum) > 0:
            kmo = r2_sum / (r2_sum + a2_sum)
        else:
            kmo = float("nan")
    except (np.linalg.LinAlgError, ZeroDivisionError):
        kmo = float("nan")

    # Bartlett's test
    det = np.linalg.det(corr)
    if det > 0 and n > 0:
        chi_sq = -(n - 1 - (2 * k + 5) / 6.0) * math.log(det)
    else:
        chi_sq = float("nan")
    df_bart = k * (k - 1) / 2
    if not math.isnan(chi_sq) and df_bart > 0:
        p_bart = float(scipy_stats.chi2.sf(chi_sq, df_bart))
    else:
        p_bart = float("nan")

    table.simple_pivot_table(
        rowdim=" ",
        rowlabels=["Kaiser-Meyer-Olkin", "Bartlett's Test"],
        coldim=" ",
        collabels=["Value", "df", "Sig."],
        cells=[
            kmo if not math.isnan(kmo) else None,
            None,
            None,
            chi_sq if not math.isnan(chi_sq) else None,
            float(df_bart),
            p_bart if not math.isnan(p_bart) else None,
        ],
    )
    return table


def _build_variance_explained(eigenvalues: np.ndarray, k: int,
                                n_factors: Optional[int],
                                group_key: tuple,
                                ds: Dataset) -> PivotTable:
    """Build Total Variance Explained table."""
    title = "Total Variance Explained"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Total Variance Explained [{label}]"

    table = PivotTable(title=title)

    total = k  # For correlation matrix, total variance = k
    var_pct = (eigenvalues / total) * 100.0
    cum_var = np.cumsum(var_pct)

    col_labels = ["Total", "% of Variance", "Cumulative %"]
    row_labels = [f"Component {i+1}" for i in range(k)]

    cells: List[Any] = []
    for i in range(k):
        cells.extend([
            float(eigenvalues[i]),
            float(var_pct[i]),
            float(cum_var[i]),
        ])

    table.simple_pivot_table(
        rowdim="Components",
        rowlabels=row_labels,
        coldim="Statistics",
        collabels=col_labels,
        cells=cells,
    )
    return table


def _build_communalities(eigenvalues: np.ndarray, k: int,
                            avail_vars: List[str],
                            corr: np.ndarray, n_factors: Optional[int],
                            group_key: tuple,
                            ds: Dataset) -> PivotTable:
    """Build Communalities table."""
    title = "Communalities"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Communalities [{label}]"

    table = PivotTable(title=title)

    # Initial communalities = 1 for PCA
    # Extraction communalities = sum of squared loadings for retained factors
    eigvals_full, eigvecs_full = np.linalg.eigh(corr)
    idx = np.argsort(eigvals_full)[::-1]
    eigvals_full = eigvals_full[idx]
    eigvecs_full = eigvecs_full[:, idx]

    n_comp = n_factors if n_factors else k
    loadings = eigvecs_full[:, :n_comp] * np.sqrt(eigvals_full[:n_comp])
    extraction_communalities = np.sum(loadings ** 2, axis=1)

    col_labels = ["Initial", "Extraction"]
    row_labels = list(avail_vars)

    cells: List[Any] = []
    for i in range(k):
        cells.extend([1.0, float(extraction_communalities[i])])

    table.simple_pivot_table(
        rowdim="Variables",
        rowlabels=row_labels,
        coldim="Communality",
        collabels=col_labels,
        cells=cells,
    )
    return table


def _build_component_matrix(loadings: np.ndarray, avail_vars: List[str],
                              n_comp: int,
                              group_key: tuple,
                              ds: Dataset) -> PivotTable:
    """Build Component Matrix table."""
    title = "Component Matrix"
    if group_key:
        from spss_engine.transforms.split_file import split_group_label
        label = split_group_label(ds, group_key)
        title = f"Component Matrix [{label}]"

    table = PivotTable(title=title)

    col_labels = [f"Component {i+1}" for i in range(n_comp)]
    row_labels = list(avail_vars)

    cells: List[Any] = []
    for i in range(len(avail_vars)):
        for j in range(n_comp):
            cells.append(float(loadings[i][j]))

    table.simple_pivot_table(
        rowdim="Variables",
        rowlabels=row_labels,
        coldim="Components",
        collabels=col_labels,
        cells=cells,
    )
    return table