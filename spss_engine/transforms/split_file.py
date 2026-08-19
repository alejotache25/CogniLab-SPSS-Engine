"""
SPLIT FILE command implementation.

SPLIT FILE BY vars         — separate output groups per combination of vars
SPLIT FILE LAYERED BY vars — output layered (not separate) per combination
SPLIT FILE OFF             — clear split

Procedures that respect split file iterate over each group defined by the
split variables and produce separate output (pivot tables) per group.

The split state is stored in Dataset.state.split_vars and
Dataset.state.split_layered. This module provides:
  - execute_split_file(): command handler
  - get_split_groups(): helper used by procedures to iterate groups

When split is active, procedures call get_split_groups(ds) which returns
a list of (group_key, filtered_df) tuples. When split is off, returns a
single group with all (filtered) data.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple

import pandas as pd

from spss_engine.data.dataset import Dataset
from spss_engine.parser.ast_nodes import CommandNode, SubcommandNode
from spss_engine.utils.errors import SPSSRuntimeError

logger = logging.getLogger(__name__)


def execute_split_file(cmd: CommandNode, ds: Dataset) -> None:
    """Execute a SPLIT FILE command.

    Handles:
      - SPLIT FILE BY vars
      - SPLIT FILE LAYERED BY vars
      - SPLIT FILE OFF
    """
    for sc in cmd.subcommands:
        if sc.name == "OFF":
            ds.clear_split()
            return
        if sc.name == "BY" and sc.variables:
            layered = bool(sc.raw_tokens)  # raw_tokens=["LAYERED"] if set
            # Validate variables exist
            for v in sc.variables:
                if not ds.has_variable(v):
                    raise SPSSRuntimeError(
                        f"Split variable not found: {v}",
                        command="SPLIT FILE")
            ds.set_split(sc.variables, layered)
            return
    # No subcommand: treat as OFF
    ds.clear_split()


def get_split_groups(ds: Dataset) -> List[Tuple[Tuple, pd.DataFrame]]:
    """Return a list of (group_key, df) tuples for the current split state.

    If split is not active, returns a single group with the entire filtered
    DataFrame and an empty key tuple.

    If split is active, groups the filtered DataFrame by the unique
    combinations of split variable values. Missing values in split vars
    form their own group (SPSS excludes cases with missing split values
    by default, but we include them as a distinct group for simplicity).

    Args:
        ds: The active Dataset.

    Returns:
        List of (group_key, DataFrame) tuples. group_key is a tuple of
        the split variable values for that group.
    """
    df = ds.get_filtered_df()
    split_vars = ds.state.split_vars

    if not split_vars or len(df) == 0:
        return [((), df)]

    # Group by unique combinations of split vars
    groups: List[Tuple[Tuple, pd.DataFrame]] = []
    # Sort by split vars for deterministic order
    df_sorted = df.sort_values(by=split_vars)
    for key, group_df in df_sorted.groupby(split_vars, sort=True,
                                            dropna=False):
        # key is a tuple (single var → scalar, so wrap)
        if not isinstance(key, tuple):
            key = (key,)
        groups.append((key, group_df.copy()))

    return groups


def split_group_label(ds: Dataset, group_key: Tuple) -> str:
    """Build a human-readable label for a split group.

    Example: "SEX=1, REGION=North"
    """
    split_vars = ds.state.split_vars
    if not split_vars or not group_key:
        return ""
    parts: List[str] = []
    for var_name, val in zip(split_vars, group_key):
        var = ds.get_variable(var_name) if ds.has_variable(var_name) else None
        # Try to resolve value label
        if var is not None and val in var.value_labels:
            parts.append(f"{var_name}={var.value_labels[val]}")
        else:
            parts.append(f"{var_name}={val}")
    return ", ".join(parts)