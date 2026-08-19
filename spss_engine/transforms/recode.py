"""
RECODE command implementation.

Syntax:
  RECODE varlist (val1=val2) (val2=val3) ... [INTO newvarlist].

Value spec keywords:
  LOWEST / LO  — lowest value (or -inf for matching)
  HIGHEST / HI — highest value (or +inf for matching)
  THRU         — range: LO THRU HI, or val1 THRU val2
  ELSE         — any value not matched by previous groups
  SYSMIS      — system-missing value
  MISSING      — any missing (system or user-defined)
  COPY         — copy the source value as-is

Each group is (source_spec = target_value). Source specs can be:
  - Single value: (1=1)
  - Range: (1 THRU 5=1)
  - Open range: (LO THRU 18=1) (36 THRU HI=3)
  - List: (1 2 3=1)
  - Keyword: (SYSMIS=-9) (ELSE=COPY) (MISSING=SYSMIS)

Groups are applied in order; the first matching group wins. ELSE matches
any value that hasn't been matched by a prior group.

INTO creates a new variable. Without INTO, the recode is applied in-place
to the source variables.
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from spss_engine.data.dataset import Dataset
from spss_engine.data.variable import Variable
from spss_engine.parser.ast_nodes import CommandNode, SubcommandNode
from spss_engine.utils.errors import SPSSRuntimeError

logger = logging.getLogger(__name__)

# Sentinel constants for special source specs
_LO = float("-inf")
_HI = float("inf")
_SYSMIS = float("nan")  # used as a marker; we check isnan
_MISSING = "MISSING"
_ELSE = "ELSE"
_COPY = "COPY"


@dataclass
class SourceSpec:
    """A source value specification within a recode group.

    A spec can be:
      - a single value (value is set, is_range=False, lo/hi=None)
      - a range (is_range=True, lo/hi set)
      - a keyword (keyword set: SYSMIS, MISSING, ELSE)
      - COPY (keyword="COPY")
    """
    value: Optional[float] = None
    is_range: bool = False
    lo: Optional[float] = None
    hi: Optional[float] = None
    keyword: Optional[str] = None  # SYSMIS, MISSING, ELSE, COPY


@dataclass
class RecodeGroup:
    """A single (source_specs = target) recode group."""
    sources: List[SourceSpec] = field(default_factory=list)
    target_value: Union[float, str, None] = None
    target_is_sysmis: bool = False
    target_is_copy: bool = False


def _parse_value_token(tok: str) -> Union[float, str]:
    """Parse a token string to a float or keep as string."""
    try:
        return float(tok)
    except (ValueError, TypeError):
        return tok


def _parse_group(group_tokens: List[str]) -> RecodeGroup:
    """Parse a parenthesized recode group token list into a RecodeGroup.

    group_tokens example: ["1", "=", "1"] or
                        ["LO", "THRU", "18", "=", "1"] or
                        ["ELSE", "=", "SYSMIS"]
    """
    # Split on "="
    eq_idx = -1
    for i, t in enumerate(group_tokens):
        if t == "=":
            eq_idx = i
            break
    if eq_idx < 0:
        raise SPSSRuntimeError(
            f"RECODE group missing '=' separator: {group_tokens}")

    source_tokens = group_tokens[:eq_idx]
    target_tokens = group_tokens[eq_idx + 1:]

    # Parse target
    target_val: Union[float, str, None] = None
    target_is_sysmis = False
    target_is_copy = False
    if len(target_tokens) == 1:
        t = target_tokens[0].upper() if isinstance(target_tokens[0], str) \
            else target_tokens[0]
        if isinstance(t, str):
            tu = t.upper()
            if tu == "SYSMIS":
                target_is_sysmis = True
            elif tu == "COPY":
                target_is_copy = True
            else:
                target_val = _parse_value_token(t)
        else:
            target_val = float(t)
    elif len(target_tokens) == 0:
        raise SPSSRuntimeError(
            f"RECODE group missing target value: {group_tokens}")
    else:
        # Multi-token target (e.g., a string); join and parse
        joined = "".join(str(t) for t in target_tokens)
        target_val = _parse_value_token(joined)

    # Parse sources
    sources: List[SourceSpec] = []
    i = 0
    while i < len(source_tokens):
        tok = source_tokens[i]
        tok_u = tok.upper() if isinstance(tok, str) else str(tok)

        if tok_u in ("LO", "LOWEST", "LOWEST"):
            # LO THRU value  or  LO THRU HI
            if i + 2 < len(source_tokens) and \
               source_tokens[i + 1].upper() == "THRU":
                hi_tok = source_tokens[i + 2]
                hi_u = hi_tok.upper() if isinstance(hi_tok, str) else str(hi_tok)
                if hi_u in ("HI", "HIGHEST", "HIG"):
                    sources.append(SourceSpec(is_range=True, lo=_LO, hi=_HI))
                else:
                    sources.append(SourceSpec(is_range=True, lo=_LO,
                                               hi=float(hi_tok)))
                i += 3
            else:
                # LO alone = lowest value (single)
                sources.append(SourceSpec(value=_LO))
                i += 1
        elif tok_u in ("HI", "HIGHEST", "HIG"):
            # HI alone = highest value, or value THRU HI
            sources.append(SourceSpec(value=_HI))
            i += 1
        elif tok_u == "THRU":
            # Should have been handled by LO THRU / value THRU
            i += 1
        elif tok_u == "SYSMIS":
            sources.append(SourceSpec(keyword="SYSMIS"))
            i += 1
        elif tok_u == "MISSING":
            sources.append(SourceSpec(keyword="MISSING"))
            i += 1
        elif tok_u == "ELSE":
            sources.append(SourceSpec(keyword="ELSE"))
            i += 1
        elif tok_u == "COPY":
            sources.append(SourceSpec(keyword="COPY"))
            i += 1
        else:
            # Numeric value — could be start of a range (val THRU val)
            val = _parse_value_token(tok)
            if (i + 2 < len(source_tokens) and
                    isinstance(source_tokens[i + 1], str) and
                    source_tokens[i + 1].upper() == "THRU"):
                hi_tok = source_tokens[i + 2]
                hi_u = hi_tok.upper() if isinstance(hi_tok, str) else str(hi_tok)
                if hi_u in ("HI", "HIGHEST", "HIG"):
                    sources.append(SourceSpec(is_range=True,
                                               lo=float(val), hi=_HI))
                else:
                    sources.append(SourceSpec(is_range=True,
                                               lo=float(val),
                                               hi=float(hi_tok)))
                i += 3
            else:
                # Single value (could be string)
                sources.append(SourceSpec(value=val))
                i += 1

    return RecodeGroup(sources=sources, target_value=target_val,
                        target_is_sysmis=target_is_sysmis,
                        target_is_copy=target_is_copy)


def _match_spec(spec: SourceSpec, value: Any,
                var: Variable,
                is_user_missing: bool) -> bool:
    """Check if a value matches a source spec."""
    # Keyword specs
    if spec.keyword == "ELSE":
        return True
    if spec.keyword == "COPY":
        return True
    if spec.keyword == "SYSMIS":
        if isinstance(value, float) and math.isnan(value):
            return True
        return var.is_system_missing(value)
    if spec.keyword == "MISSING":
        if var.is_missing(value):
            return True
        return False

    # Range specs
    if spec.is_range:
        if isinstance(value, float) and math.isnan(value):
            return False
        if not isinstance(value, (int, float)):
            return False
        lo = spec.lo if spec.lo is not None else float("-inf")
        hi = spec.hi if spec.hi is not None else float("inf")
        return lo <= float(value) <= hi

    # Single-value spec
    if spec.value is not None:
        sv = spec.value
        if sv == _LO:
            # Lowest — matches the minimum; we treat as <= any value
            # In practice, SPSS uses LO only in ranges. As a single spec,
            # treat as matching any non-missing value.
            if isinstance(value, float) and math.isnan(value):
                return False
            return isinstance(value, (int, float))
        if sv == _HI:
            if isinstance(value, float) and math.isnan(value):
                return False
            return isinstance(value, (int, float))
        # Exact match
        if isinstance(sv, str) and isinstance(value, str):
            return sv == value
        if isinstance(sv, (int, float)) and isinstance(value, (int, float)):
            if math.isnan(float(sv)) or math.isnan(float(value)):
                return False
            return abs(float(sv) - float(value)) < 1e-10
        return False

    return False


def _apply_groups(groups: List[RecodeGroup], value: Any,
                   var: Variable) -> Any:
    """Apply recode groups to a single value.

    Returns the recoded value, or the original value if no group matched
    (when no ELSE group present).
    """
    is_user_missing = var.is_user_missing(value)
    for grp in groups:
        for spec in grp.sources:
            if _match_spec(spec, value, var, is_user_missing):
                if grp.target_is_copy:
                    return value
                if grp.target_is_sysmis:
                    return np.nan
                return grp.target_value
    # No group matched — keep original value
    return value


def _resolve_var_list(cmd: CommandNode, ds: Dataset) -> List[str]:
    """Extract the list of source variable names from the command."""
    var_names: List[str] = []
    for sc in cmd.subcommands:
        if sc.name == "VARS":
            # Expand TO ranges
            expanded = ds.get_varlist(sc.variables)
            var_names.extend(expanded)
    return var_names


def _resolve_groups(cmd: CommandNode) -> List[RecodeGroup]:
    """Extract the recode groups from the command."""
    for sc in cmd.subcommands:
        if sc.name == "GROUPS":
            groups: List[RecodeGroup] = []
            for group_tokens in sc.raw_tokens:
                groups.append(_parse_group(group_tokens))
            return groups
    return []


def execute_recode(cmd: CommandNode, ds: Dataset) -> None:
    """Execute a RECODE command against the dataset.

    Supports:
      - In-place recode: RECODE varlist (groups).
      - INTO newvar: RECODE varlist (groups) INTO newvar.
        The new variable is created as numeric (F8.2) if it doesn't exist.
        For multiple source vars with INTO, the target count must match.
    """
    source_vars = _resolve_var_list(cmd, ds)
    if not source_vars:
        raise SPSSRuntimeError("RECODE requires at least one source variable",
                               command="RECODE")
    groups = _resolve_groups(cmd)
    if not groups:
        raise SPSSRuntimeError("RECODE requires at least one recode group",
                               command="RECODE")

    target_var = cmd.target_var  # INTO target (single var or None)

    if ds.df is None or ds.n_cases == 0:
        # Nothing to recode
        if target_var and not ds.has_variable(target_var):
            ds.add_variable(Variable(name=target_var, var_type="numeric",
                                      width=8, format="F8.2"))
            ds.df[target_var] = np.nan
        return

    df = ds.df

    if target_var:
        # INTO mode: recode each source var into the corresponding target
        # For single source + single target
        if len(source_vars) == 1:
            src = source_vars[0]
            src_var = ds.get_variable(src)
            if not ds.has_variable(target_var):
                ds.add_variable(Variable(name=target_var,
                                          var_type="numeric", width=8,
                                          format="F8.2"))
                df[target_var] = np.nan
            tgt_col = df[src].apply(
                lambda v: _apply_groups(groups, v, src_var))
            df[target_var] = _coerce_series(tgt_col, src_var)
        else:
            # Multiple sources → multiple targets (not yet supported for
            # multi-target INTO; apply same groups to all sources, creating
            # one target per source)
            for src in source_vars:
                src_var = ds.get_variable(src)
                tgt = target_var if len(source_vars) == 1 else \
                    f"{target_var}_{src}"
                if not ds.has_variable(tgt):
                    ds.add_variable(Variable(name=tgt, var_type="numeric",
                                              width=8, format="F8.2"))
                    df[tgt] = np.nan
                tgt_col = df[src].apply(
                    lambda v: _apply_groups(groups, v, src_var))
                df[tgt] = _coerce_series(tgt_col, src_var)
    else:
        # In-place recode
        for src in source_vars:
            src_var = ds.get_variable(src)
            new_col = df[src].apply(
                lambda v: _apply_groups(groups, v, src_var))
            df[src] = _coerce_series(new_col, src_var)


def _coerce_series(series: pd.Series, src_var: Variable) -> pd.Series:
    """Convert a recoded series to the proper type based on the source
    variable type. Numeric source → numeric series; string source →
    object series."""
    if src_var.is_numeric:
        return pd.to_numeric(series, errors="coerce")
    return series.astype("object")


# Late import for pandas (used in to_numeric)
import pandas as pd  # noqa: E402