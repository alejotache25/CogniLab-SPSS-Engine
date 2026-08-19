"""
Base utilities for statistical procedures.

Provides shared statistical functions used by FREQUENCIES, DESCRIPTIVES,
CROSSTABS, etc. Algorithms follow the SPSS alg_*.txt docs.

Statistics implemented from scratch (verified against numpy/scipy in tests):
  - mean, variance (sample, ddof=1), std
  - min, max, range, sum
  - skewness (moment-based: m3 / s^3), kurtosis (m4 / s^4 - 3)
  - standard error of mean, skew, kurtosis
  - median (HAVERAGE percentile)
  - percentiles (HAVERAGE)
  - mode (most frequent value)
"""

from __future__ import annotations
import math
from typing import List, Optional, Sequence, Tuple
import numpy as np


def valid_values(values: Sequence[float]) -> np.ndarray:
    """Return non-NaN values as a float array."""
    arr = np.array(values, dtype=float)
    return arr[~np.isnan(arr)]


def mean(values: Sequence[float]) -> float:
    v = valid_values(values)
    if len(v) == 0:
        return float("nan")
    return float(np.sum(v) / len(v))


def variance(values: Sequence[float], ddof: int = 1) -> float:
    """Sample variance (ddof=1) or population (ddof=0)."""
    v = valid_values(values)
    n = len(v)
    if n <= ddof:
        return float("nan")
    m = np.sum(v) / n
    ss = np.sum((v - m) ** 2)
    return float(ss / (n - ddof))


def std(values: Sequence[float], ddof: int = 1) -> float:
    var = variance(values, ddof=ddof)
    if math.isnan(var):
        return float("nan")
    return float(math.sqrt(var))


def semean(values: Sequence[float]) -> float:
    """Standard error of the mean: std / sqrt(n)."""
    v = valid_values(values)
    n = len(v)
    if n < 2:
        return float("nan")
    s = std(v, ddof=1)
    return float(s / math.sqrt(n))


def minimum(values: Sequence[float]) -> float:
    v = valid_values(values)
    if len(v) == 0:
        return float("nan")
    return float(np.min(v))


def maximum(values: Sequence[float]) -> float:
    v = valid_values(values)
    if len(v) == 0:
        return float("nan")
    return float(np.max(v))


def range_stat(values: Sequence[float]) -> float:
    """Range = max - min."""
    mn = minimum(values)
    mx = maximum(values)
    if math.isnan(mn) or math.isnan(mx):
        return float("nan")
    return float(mx - mn)


def sum_stat(values: Sequence[float]) -> float:
    v = valid_values(values)
    if len(v) == 0:
        return 0.0
    return float(np.sum(v))


def skewness(values: Sequence[float]) -> float:
    """Moment-based skewness: m3 / s^3.

    Computed only if W >= 3 and variance > 0.
    """
    v = valid_values(values)
    n = len(v)
    if n < 3:
        return float("nan")
    m = np.sum(v) / n
    s2 = np.sum((v - m) ** 2) / (n - 1)
    if s2 <= 0:
        return float("nan")
    s = math.sqrt(s2)
    m3 = np.sum((v - m) ** 3) / (n - 1)
    return float(m3 / (s ** 3))


def kurtosis(values: Sequence[float]) -> float:
    """Moment-based kurtosis: m4 / s^4 - 3.

    Computed only if W >= 4 and variance > 0.
    """
    v = valid_values(values)
    n = len(v)
    if n < 4:
        return float("nan")
    m = np.sum(v) / n
    s2 = np.sum((v - m) ** 2) / (n - 1)
    if s2 <= 0:
        return float("nan")
    s = math.sqrt(s2)
    m4 = np.sum((v - m) ** 4) / (n - 1)
    return float(m4 / (s ** 4) - 3.0)


def seskew(values: Sequence[float]) -> float:
    """Standard error of skewness: sqrt(6*n*(n-1) / ((n-2)*(n+1)*(n+3)))."""
    v = valid_values(values)
    n = len(v)
    if n < 3:
        return float("nan")
    return float(math.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))))


def sekurt(values: Sequence[float]) -> float:
    """Standard error of kurtosis."""
    v = valid_values(values)
    n = len(v)
    if n < 4:
        return float("nan")
    num = 6 * n * (n - 1) * (n - 1) * (n - 1) * (n - 1)
    den = (n - 2) * (n - 3) * (n + 3) * (n + 5)
    return float(math.sqrt(num / den))


def mode_stat(values: Sequence[float]) -> float:
    """Mode: most frequent value. Ties → smallest value."""
    v = valid_values(values)
    if len(v) == 0:
        return float("nan")
    vals, counts = np.unique(v, return_counts=True)
    max_count = np.max(counts)
    candidates = vals[counts == max_count]
    return float(np.min(candidates))


def cv_stat(values: Sequence[float]) -> float:
    """Coefficient of variation: std / mean (as fraction)."""
    v = valid_values(values)
    n = len(v)
    if n < 2:
        return float("nan")
    m = mean(v)
    if abs(m) < 1e-15:
        return float("nan")
    s = std(v, ddof=1)
    return float(s / m)


def zscore(values: Sequence[float]) -> np.ndarray:
    """Compute z-scores: (x - mean) / std. Returns array same length as
    input with NaN for missing values."""
    arr = np.array(values, dtype=float)
    v = arr[~np.isnan(arr)]
    if len(v) < 2:
        return np.full_like(arr, np.nan)
    m = float(np.sum(v) / len(v))
    s = float(np.sqrt(np.sum((v - m) ** 2) / (len(v) - 1)))
    if s == 0:
        return np.full_like(arr, np.nan)
    result = (arr - m) / s
    return result