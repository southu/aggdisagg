"""aggdisagg — Temporal Aggregation & Disaggregation for Modern Python.

Polars-first, production-grade library with perfect consistency guarantees.

Main entrypoint:
    from aggdisagg import TemporalAligner
"""

from __future__ import annotations

__version__ = "1.0.0"

from .core import TemporalAligner
from .conversion import Conversion, make_aggregation_matrix
from .methods import Method

# Backwards compatible convenience (optional)
from .api import disaggregate, aggregate, AggDisaggModel

__all__ = [
    "TemporalAligner",
    "Conversion",
    "make_aggregation_matrix",
    "Method",
    "disaggregate",
    "aggregate",
    "AggDisaggModel",
]

# Re-export main
TemporalAligner = TemporalAligner  # explicit


# Optional sktime export
try:
    from .core import TemporalAligner as _TA
    # users can do aligner.get_sktime_transformer()
except Exception:
    pass
