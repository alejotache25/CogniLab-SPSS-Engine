"""
Percentile calculation using the HAVERAGE method (SPSS default).

The HAVERAGE (average at weighted mean) method is SPSS's default percentile
calculation method. It interpolates between the two closest ranks.

Algorithm:
  1. Sort the values in ascending order.
  2. Compute rank = (p/100) * (N + 1), where N is the number of valid values.
  3. If rank is an integer, the percentile is the value at that rank.
  4. If rank is fractional, interpolate between the values at floor(rank)
     and ceil(rank).

This is equivalent to numpy's 'linear' interpolation method
(np.percentile(..., method='linear')), but implemented explicitly for clarity.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np


def percentile_haverage(values: List[float], p: float) -> float:
    """Compute a single percentile using the HAVERAGE method.

    Args:
        values: List of numeric values (NaNs are excluded).
        p: Percentile value (0-100). E.g., 25 for the 25th percentile.

    Returns:
        The percentile value. Returns NaN if no valid values.
    """
    # Filter out NaN values
    valid = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(valid) == 0:
        return float("nan")
    if len(valid) == 1:
        return float(valid[0])

    # Sort
    valid.sort()

    # Compute rank: r = (p/100) * (N + 1)
    n = len(valid)
    rank = (p / 100.0) * (n + 1)

    # Handle edge cases
    if rank <= 1:
        return float(valid[0])
    if rank >= n:
        return float(valid[-1])

    # Interpolate
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    frac = rank - lower

    if lower == upper:
        return float(valid[lower - 1])  # 1-indexed rank

    # Linear interpolation between valid[lower-1] and valid[upper-1]
    lower_val = float(valid[lower - 1])
    upper_val = float(valid[upper - 1])
    return lower_val + frac * (upper_val - lower_val)


def percentiles_haverage(values: List[float],
                          percents: List[float]) -> List[float]:
    """Compute multiple percentiles using the HAVERAGE method.

    Args:
        values: List of numeric values.
        percents: List of percentile values (0-100).

    Returns:
        List of percentile values corresponding to the input percents.
    """
    return [percentile_haverage(values, p) for p in percents]


def median_haverage(values: List[float]) -> float:
    """Compute the median (50th percentile) using HAVERAGE."""
    return percentile_haverage(values, 50.0)


def quartiles_haverage(values: List[float]) -> tuple:
    """Compute Q1, Q2 (median), Q3 using HAVERAGE.

    Returns:
        Tuple of (Q1, Q2, Q3).
    """
    q1 = percentile_haverage(values, 25.0)
    q2 = percentile_haverage(values, 50.0)
    q3 = percentile_haverage(values, 75.0)
    return (q1, q2, q3)