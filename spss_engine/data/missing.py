"""
Missing value definitions for SPSS variables.

Two types of missing values:
  - System-missing: numpy.nan for numerics, None for strings.
  - User-missing: user-defined values (e.g., 99, -1, 'N/A') that are
    treated as missing by procedures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np


@dataclass
class MissingValues:
    """User-defined missing values for a variable.

    SPSS supports up to 3 discrete missing values for numeric variables,
    or a range, or a range plus discrete values.
    For string variables, up to 3 discrete string values.
    """
    discrete: List[Union[float, str]] = field(default_factory=list)
    range_low: Optional[float] = None
    range_high: Optional[float] = None

    def is_empty(self) -> bool:
        """Return True if no missing values are defined."""
        return (len(self.discrete) == 0 and
                self.range_low is None and
                self.range_high is None)

    def contains(self, value: Union[float, str, None]) -> bool:
        """Check if a value matches a user-missing definition.

        Does NOT check for system-missing — use is_missing() on Variable for that.
        """
        if value is None:
            return False
        # Numeric value (int or float, but not str)
        if isinstance(value, (int, float)) and not isinstance(value, str):
            fval = float(value)
            if isinstance(value, float) and np.isnan(fval):
                return False  # system-missing is not user-missing
            # Check discrete
            for d in self.discrete:
                if isinstance(d, (int, float)) and not isinstance(d, str):
                    if abs(float(d) - fval) < 1e-10:
                        return True
            # Check range
            if self.range_low is not None and self.range_high is not None:
                if self.range_low <= fval <= self.range_high:
                    return True
            return False
        # String value
        if isinstance(value, str):
            for d in self.discrete:
                if isinstance(d, str) and d == value:
                    return True
        return False

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.discrete:
            parts.append(f"discrete={self.discrete}")
        if self.range_low is not None:
            parts.append(f"range={self.range_low}..{self.range_high}")
        return f"MissingValues({', '.join(parts)})"