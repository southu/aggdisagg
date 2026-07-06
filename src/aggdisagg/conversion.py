"""Frequency conversion matrix construction (Polars + NumPy native)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
import polars as pl

class Conversion(str, Enum):
    """How low-frequency values relate to high-frequency observations."""

    SUM = "sum"
    MEAN = "mean"
    FIRST = "first"
    LAST = "last"

    @property
    def is_flow(self) -> bool:
        return self in (Conversion.SUM, Conversion.MEAN)


def make_aggregation_matrix(
    n_high: int,
    n_low: int,
    conversion: Conversion | str = Conversion.SUM,
) -> np.ndarray:
    """Return the C matrix such that y_low ≈ C @ y_high.

    Guarantees exact reconstruction when applied to uniformly distributed data.
    """
    if isinstance(conversion, str):
        conversion = Conversion(conversion)

    if n_high % n_low != 0:
        raise ValueError(
            f"n_high ({n_high}) must be divisible by n_low ({n_low}) "
            "for regular frequency conversion."
        )
    freq = n_high // n_low
    C = np.zeros((n_low, n_high), dtype=np.float64)

    for i in range(n_low):
        start = i * freq
        if conversion == Conversion.SUM:
            C[i, start : start + freq] = 1.0
        elif conversion == Conversion.MEAN:
            C[i, start : start + freq] = 1.0 / freq
        elif conversion == Conversion.FIRST:
            C[i, start] = 1.0
        elif conversion == Conversion.LAST:
            C[i, start + freq - 1] = 1.0
    return C


def infer_n_high_from_index(
    low_index: pl.Series, target_freq: str
) -> int:
    """Helper to compute expected high-frequency length from low freq index."""
    # Simplified: assumes regular spacing. Real impl would use Polars date range.
    n_low = len(low_index)
    # Placeholder – in real code we would expand using pl.date_range
    # For skeleton we just return a reasonable multiple
    if "y" in str(target_freq).lower():
        return n_low * 12
    if "q" in str(target_freq).lower():
        return n_low * 4
    return n_low * 3  # default monthly-ish
