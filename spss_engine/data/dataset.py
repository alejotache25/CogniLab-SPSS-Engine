"""
Dataset model and DATA LIST implementation for the SPSS engine.

Dataset wraps:
  - pandas.DataFrame: the data (cases × variables)
  - variables dict: Variable metadata (name → Variable)
  - var_order: list of variable names in definition order
  - state: filter, split, weight state

DATA LIST is implemented as a method on Dataset that creates the
DataFrame and variable metadata from inline data or external files.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from spss_engine.data.variable import Variable
from spss_engine.data.missing import MissingValues
from spss_engine.parser.ast_nodes import CommandNode, VarDefNode
from spss_engine.utils.errors import SPSSRuntimeError

logger = logging.getLogger(__name__)


@dataclass
class DatasetState:
    """Runtime state for filter, split, and weight."""
    filter_var: Optional[str] = None      # Variable name for FILTER
    filter_active: bool = False
    split_vars: List[str] = field(default_factory=list)  # SPLIT FILE variables
    split_layered: bool = False
    weight_var: Optional[str] = None      # WEIGHT BY variable


class Dataset:
    """In-memory SPSS dataset: DataFrame + variable metadata + state."""

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._variables: Dict[str, Variable] = {}
        self._var_order: List[str] = []
        self._state: DatasetState = DatasetState()
        self._n_cases: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            return pd.DataFrame()
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        self._df = value
        self._n_cases = len(value)

    @property
    def variables(self) -> Dict[str, Variable]:
        return self._variables

    @property
    def var_order(self) -> List[str]:
        return self._var_order

    @property
    def n_cases(self) -> int:
        return self._n_cases

    @property
    def state(self) -> DatasetState:
        return self._state

    @property
    def n_variables(self) -> int:
        return len(self._var_order)

    @property
    def is_empty(self) -> bool:
        return self._df is None or len(self._df) == 0

    # ------------------------------------------------------------------
    # Variable management
    # ------------------------------------------------------------------

    def add_variable(self, var: Variable) -> None:
        """Add a variable to the dataset dictionary."""
        if var.name in self._variables:
            raise SPSSRuntimeError(f"Variable already exists: {var.name}")
        self._variables[var.name] = var
        self._var_order.append(var.name)

    def get_variable(self, name: str) -> Variable:
        """Get a variable by name (case-sensitive)."""
        if name not in self._variables:
            raise SPSSRuntimeError(f"Variable not found: {name}")
        return self._variables[name]

    def has_variable(self, name: str) -> bool:
        """Check if a variable exists."""
        return name in self._variables

    def get_all_var_names(self) -> List[str]:
        """Return all variable names in definition order."""
        return list(self._var_order)

    # ------------------------------------------------------------------
    # Variable list resolution (ALL, TO)
    # ------------------------------------------------------------------

    def get_varlist(self, spec: Union[str, List[str]]) -> List[str]:
        """Resolve a variable list specification.

        Supports:
          - "ALL": all variables in order
          - "var1 TO var5": expand range
          - ["var1", "var2"]: explicit list
          - ["var1", "TO", "var5"]: expand TO within list
        """
        if isinstance(spec, str):
            spec_list: list[str] = [spec]
        else:
            spec_list = list(spec)

        # Single "ALL" keyword
        if len(spec_list) == 1 and spec_list[0].upper() == "ALL":
            return list(self._var_order)

        # Process TO expansions
        result: list[str] = []
        i = 0
        while i < len(spec_list):
            item = spec_list[i]
            if i + 2 < len(spec_list) and spec_list[i + 1].upper() == "TO":
                # Expand TO range
                start = item
                end = spec_list[i + 2]
                result.extend(self._expand_to(start, end))
                i += 3
            else:
                result.append(item)
                i += 1

        # Validate all variables exist
        for v in result:
            if v not in self._variables:
                raise SPSSRuntimeError(f"Variable not found: {v}")

        return result

    def _expand_to(self, start: str, end: str) -> List[str]:
        """Expand a TO range: var1 TO var5 → [var1, var2, var3, var4, var5]."""
        # Find indices in var_order
        try:
            start_idx = self._var_order.index(start)
            end_idx = self._var_order.index(end)
        except ValueError:
            raise SPSSRuntimeError(
                f"Variable not found in TO range: {start} or {end}")

        if start_idx > end_idx:
            raise SPSSRuntimeError(
                f"TO range start ({start}) comes after end ({end})")

        return self._var_order[start_idx:end_idx + 1]

    # ------------------------------------------------------------------
    # Missing value checking
    # ------------------------------------------------------------------

    def is_missing(self, var_name: str, value: Union[float, str, None]) -> bool:
        """Check if a value is missing for a given variable."""
        var = self.get_variable(var_name)
        return var.is_missing(value)

    def is_system_missing(self, var_name: str,
                          value: Union[float, str, None]) -> bool:
        """Check if a value is system-missing only."""
        var = self.get_variable(var_name)
        return var.is_system_missing(value)

    # ------------------------------------------------------------------
    # DATA LIST implementation
    # ------------------------------------------------------------------

    def load_from_data_list(self, cmd: CommandNode) -> None:
        """Load data from a parsed DATA LIST command.

        Handles FIXED, FREE, and LIST formats with inline (BEGIN DATA)
        or external file data.
        """
        # Build variable definitions
        var_defs: List[VarDefNode] = cmd.var_defs
        if not var_defs:
            raise SPSSRuntimeError("DATA LIST: no variables defined")

        # Create Variable objects
        for vd in var_defs:
            var = Variable(
                name=vd.name,
                var_type=vd.var_type,
                width=vd.width,
                format=vd.format_spec or ("A" + str(vd.width) if vd.var_type == "string" else "F8.2"),
            )
            self.add_variable(var)

        # Get data source
        if cmd.raw_data is not None:
            data_text = cmd.raw_data
        elif cmd.file_path is not None:
            data_text = self._read_external_file(cmd.file_path, cmd.data_format or "FIXED")
        else:
            # No data provided — create empty dataset with variables
            self._df = pd.DataFrame(columns=self._var_order)
            self._n_cases = 0
            return

        # Parse data according to format
        fmt = cmd.data_format or "FIXED"
        if fmt == "FIXED":
            self._load_fixed(var_defs, data_text)
        elif fmt == "LIST":
            self._load_free(var_defs, data_text, is_list=True)
        elif fmt == "FREE":
            self._load_free(var_defs, data_text, is_list=False)
        else:
            raise SPSSRuntimeError(f"Unknown DATA LIST format: {fmt}")

    def _read_external_file(self, path: str, fmt: str) -> str:
        """Read data from an external file."""
        import os
        if not os.path.exists(path):
            raise SPSSRuntimeError(f"Data file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_fixed(self, var_defs: List[VarDefNode], data_text: str) -> None:
        """Load fixed-format data using column positions."""
        lines = [line for line in data_text.split("\n") if line.strip()]
        data: Dict[str, list] = {vd.name: [] for vd in var_defs}

        for line in lines:
            for vd in var_defs:
                if vd.start_col is None or vd.end_col is None:
                    # No column spec — shouldn't happen for FIXED
                    data[vd.name].append(np.nan if vd.var_type == "numeric" else "")
                    continue
                # Columns are 1-indexed in SPSS
                start = vd.start_col - 1
                end = vd.end_col
                raw = line[start:end] if end <= len(line) else line[start:]
                raw = raw.strip()
                if vd.var_type == "string":
                    data[vd.name].append(raw)
                else:
                    # Numeric: blank → system-missing (NaN)
                    if raw == "" or not self._is_valid_numeric(raw):
                        data[vd.name].append(np.nan)
                    else:
                        try:
                            data[vd.name].append(float(raw))
                        except ValueError:
                            data[vd.name].append(np.nan)

        self._build_dataframe(data)

    def _load_free(self, var_defs: List[VarDefNode], data_text: str,
                   is_list: bool = False) -> None:
        """Load free-format data (FREE or LIST).

        Values are separated by blanks or commas. Multiple blanks/commas
        indicate missing values.

        FREE: values flow across lines (multiple cases per line allowed).
        LIST: one case per line.
        """
        lines = [line for line in data_text.split("\n") if line.strip()]
        var_names = [vd.name for vd in var_defs]
        data: Dict[str, list] = {name: [] for name in var_names}

        if is_list:
            # LIST: one case per line
            for line in lines:
                values = self._split_free_values(line)
                for i, vd in enumerate(var_defs):
                    if i < len(values):
                        val = values[i]
                        data[vd.name].append(self._parse_value(val, vd.var_type))
                    else:
                        data[vd.name].append(
                            np.nan if vd.var_type == "numeric" else "")
        else:
            # FREE: values flow across lines
            all_values: list[str] = []
            for line in lines:
                all_values.extend(self._split_free_values(line))

            n_vars = len(var_defs)
            n_cases = (len(all_values) + n_vars - 1) // n_vars  # ceil div
            for case_idx in range(n_cases):
                for i, vd in enumerate(var_defs):
                    val_idx = case_idx * n_vars + i
                    if val_idx < len(all_values):
                        val = all_values[val_idx]
                        data[vd.name].append(
                            self._parse_value(val, vd.var_type))
                    else:
                        data[vd.name].append(
                            np.nan if vd.var_type == "numeric" else "")

        self._build_dataframe(data)

    def _is_list_format(self, var_defs: List[VarDefNode]) -> bool:
        """Heuristic: if all var_defs have no column spec, treat as LIST/FREE."""
        return all(vd.start_col is None for vd in var_defs)

    def _split_free_values(self, line: str) -> List[str]:
        """Split a line into free-format values.

        Handles comma and space delimiters. Multiple delimiters = missing.
        """
        # Replace tabs with spaces
        line = line.replace("\t", " ")
        # Split by comma or space
        # First, normalize commas: replace comma with space
        line = line.replace(",", " ")
        # Split by whitespace
        parts = line.split()
        return parts

    def _parse_value(self, raw: str, var_type: str) -> Union[float, str, None]:
        """Parse a raw string value into the appropriate type."""
        raw = raw.strip()
        if var_type == "string":
            # Strip quotes if present
            if len(raw) >= 2 and raw[0] in ("'", '"') and raw[-1] == raw[0]:
                raw = raw[1:-1]
            return raw
        # Numeric
        if raw == "" or raw == ".":
            return np.nan
        try:
            return float(raw)
        except ValueError:
            return np.nan

    def _is_valid_numeric(self, s: str) -> bool:
        """Check if a string is a valid numeric value."""
        s = s.strip()
        if not s or s == ".":
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _build_dataframe(self, data: Dict[str, list]) -> None:
        """Build a pandas DataFrame from the parsed data dict."""
        # Ensure all columns have the same length
        max_len = max(len(v) for v in data.values()) if data else 0
        for name, vals in data.items():
            while len(vals) < max_len:
                vals.append(np.nan)

        self._df = pd.DataFrame(data, columns=self._var_order)
        self._n_cases = len(self._df)

        # Set proper dtypes
        for name in self._var_order:
            var = self._variables[name]
            if var.is_numeric:
                self._df[name] = pd.to_numeric(self._df[name], errors="coerce")
            else:
                self._df[name] = self._df[name].astype("object")

    # ------------------------------------------------------------------
    # State management (filter, split, weight)
    # ------------------------------------------------------------------

    def set_filter(self, var_name: Optional[str]) -> None:
        """Set or clear the filter variable."""
        if var_name is None:
            self._state.filter_var = None
            self._state.filter_active = False
        else:
            if not self.has_variable(var_name):
                raise SPSSRuntimeError(f"Filter variable not found: {var_name}")
            self._state.filter_var = var_name
            self._state.filter_active = True

    def clear_filter(self) -> None:
        """Clear the filter (FILTER OFF)."""
        self._state.filter_var = None
        self._state.filter_active = False

    def set_split(self, var_names: List[str], layered: bool = False) -> None:
        """Set SPLIT FILE variables."""
        for v in var_names:
            if not self.has_variable(v):
                raise SPSSRuntimeError(f"Split variable not found: {v}")
        self._state.split_vars = var_names
        self._state.split_layered = layered

    def clear_split(self) -> None:
        """Clear SPLIT FILE (SPLIT FILE OFF)."""
        self._state.split_vars = []
        self._state.split_layered = False

    def set_weight(self, var_name: Optional[str]) -> None:
        """Set WEIGHT BY variable."""
        if var_name is not None and not self.has_variable(var_name):
            raise SPSSRuntimeError(f"Weight variable not found: {var_name}")
        self._state.weight_var = var_name

    def clear_weight(self) -> None:
        """Clear WEIGHT (WEIGHT OFF)."""
        self._state.weight_var = None

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_filtered_df(self) -> pd.DataFrame:
        """Return the DataFrame with filter applied.

        If no filter is active, returns the full DataFrame.
        """
        if self._df is None:
            return pd.DataFrame()
        if self._state.filter_active and self._state.filter_var:
            filter_col = self._df[self._state.filter_var]
            # Filter: keep cases where filter_var is not missing and is
            # truthy (non-zero, non-missing)
            mask = filter_col.notna() & (filter_col != 0)
            return self._df[mask].copy()
        return self._df.copy()

    def get_column(self, var_name: str) -> pd.Series:
        """Get a column from the (filtered) DataFrame."""
        df = self.get_filtered_df()
        if var_name not in df.columns:
            raise SPSSRuntimeError(f"Variable not found: {var_name}")
        return df[var_name]

    def __repr__(self) -> str:
        return (f"Dataset(n_vars={self.n_variables}, n_cases={self.n_cases}, "
                f"vars={self._var_order})")