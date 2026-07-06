"""aggdisagg — Temporal Aggregation & Disaggregation for Modern Python.

Polars-first, production-grade library with perfect consistency guarantees.

Main entrypoint:
    from aggdisagg import TemporalAligner
"""

from __future__ import annotations

__version__ = "1.0.0"

# Backwards compatible convenience (optional)
from .api import AggDisaggModel, aggregate, disaggregate
from .conversion import Conversion, make_aggregation_matrix
from .core import TemporalAligner
from .methods import Method

__all__ = [
    "AggDisaggModel",
    "Conversion",
    "Method",
    "TemporalAligner",
    "aggregate",
    "disaggregate",
    "make_aggregation_matrix",
]

# Optional sktime export
import contextlib

with contextlib.suppress(ImportError):
    from .core import TemporalAligner  # re-export for .get_sktime_transformer() access
