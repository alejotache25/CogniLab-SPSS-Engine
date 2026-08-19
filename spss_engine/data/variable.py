"""
Variable model for SPSS dataset.

Each Variable has:
  - name: variable name (case preserved)
  - var_type: "numeric" or "string"
  - width: display width
  - format: format spec (e.g., "F8.2", "A10")
  - label: descriptive label
  - value_labels: dict mapping values to labels
  - missing: MissingValues definition
  - measure: measurement level ("nominal", "ordinal", "scale")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import numpy as np

from spss_engine.data.missing import MissingValues


@dataclass
class Variable:
    """Represents a variable in the SPSS dataset dictionary."""
    name: str
    var_type: str = "numeric"           # "numeric" or "string"
    width: int = 8                      # Display width
    format: str = "F8.2"               # Display format spec
    label: str = ""                     # Variable label
    value_labels: Dict[Union[float, str], str] = field(default_factory=dict)
    missing: MissingValues = field(default_factory=MissingValues)
    measure: str = "scale"              # "nominal", "ordinal", "scale"

    @property
    def is_numeric(self) -> bool:
        return self.var_type == "numeric"

    @property
    def is_string(self) -> bool:
        return self.var_type == "string"

    def is_missing(self, value: Union[float, str, None]) -> bool:
        """Check if a value is missing (system or user-defined).

        System-missing: numpy.nan for numerics, None for strings.
        User-missing: matches the MissingValues definition.
        """
        if value is None:
            return True
        if self.is_numeric:
            if isinstance(value, float) and np.isnan(value):
                return True
            # Check user-missing (handles int and float)
            if self.missing.contains(value):
                return True
            return False
        else:
            # String variables: no system-missing, only user-missing
            if value is None:
                return True
            if self.missing.contains(value):
                return True
            return False

    def is_system_missing(self, value: Union[float, str, None]) -> bool:
        """Check if a value is system-missing only (not user-missing)."""
        if self.is_numeric:
            return isinstance(value, float) and np.isnan(value)
        return value is None

    def is_user_missing(self, value: Union[float, str, None]) -> bool:
        """Check if a value is user-missing (not system-missing)."""
        if self.is_system_missing(value):
            return False
        return self.missing.contains(value)

    def set_label(self, label: str) -> None:
        """Set the variable label."""
        self.label = label

    def set_value_labels(self, labels: Dict[Union[float, str], str]) -> None:
        """Set value labels (replaces existing)."""
        self.value_labels = labels

    def add_value_label(self, value: Union[float, str], label: str) -> None:
        """Add a single value-label mapping."""
        self.value_labels[value] = label

    def __repr__(self) -> str:
        return (f"Variable({self.name}, type={self.var_type}, "
                f"width={self.width}, label={self.label!r})")