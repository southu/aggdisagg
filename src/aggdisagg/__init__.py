"""aggdisagg: Temporal Aggregation & Disaggregation for Modern Python.

Polars-first library for converting time series between frequencies
while preserving exact aggregation consistency.

Example:
    >>> import polars as pl
    >>> from aggdisagg import disaggregate, aggregate
    >>> # ... see README for full examples
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import (
    aggregate,
    disaggregate,
    AggDisaggModel,
)
from .conversion import (
    Conversion,
    make_aggregation_matrix,
)
from .methods import (
    Method,
    Denton,
    Uniform,
    Linear,
)

__all__ = [
    "aggregate",
    "disaggregate",
    "AggDisaggModel",
    "Conversion",
    "make_aggregation_matrix",
    "Method",
    "Denton",
    "Uniform",
    "Linear",
]
