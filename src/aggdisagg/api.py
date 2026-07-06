"""High-level scikit-learn / Polars-style API for aggdisagg."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from .conversion import Conversion
from .methods import BaseMethod, Linear, Uniform


@dataclass
class AggDisaggResult:
    """Container for results with perfect round-trip guarantee."""

    y_high: pl.Series
    method: str
    conversion: str
    n_low: int
    n_high: int
    _low_values: np.ndarray  # for internal consistency check

    def aggregate(self) -> pl.Series:
        """Re-aggregate the high-frequency result. Must match original y_low."""
        # Simple implementation for skeleton
        freq = self.n_high // self.n_low
        if self.conversion == "sum":
            agg = self.y_high.to_numpy().reshape(self.n_low, freq).sum(axis=1)
        elif self.conversion == "mean":
            agg = self.y_high.to_numpy().reshape(self.n_low, freq).mean(axis=1)
        else:
            agg = self.y_high.to_numpy()[::freq]  # first or last simplified
        return pl.Series("y_low_reagg", agg)

    def check_consistency(self, tol: float = 1e-10) -> bool:
        """Verify that aggregation of result recovers the low-frequency input."""
        if self._low_values is None:
            return True
        try:
            reagg = self.aggregate().to_numpy()
            return np.allclose(reagg, self._low_values, atol=tol)
        except Exception:
            return False


class AggDisaggModel:
    """Scikit-learn style model for temporal agg/disagg.

    Polars-first. Also accepts pandas via .to_pandas() internally when needed.
    """

    def __init__(
        self,
        method: Literal["uniform", "linear"] = "uniform",
        conversion: Literal["sum", "mean", "first", "last"] = "sum",
        **method_kwargs: Any,
    ):
        self.method_name = method
        self.conversion = Conversion(conversion)
        self.method_kwargs = method_kwargs
        self._method: BaseMethod = self._get_method(method)
        self._fitted = False
        self._low_values: np.ndarray | None = None
        self._n_high: int | None = None

    def _get_method(self, name: str) -> BaseMethod:
        if name == "uniform":
            return Uniform()
        if name == "linear":
            return Linear()
        raise ValueError(f"Unknown method: {name}. Use 'uniform' or 'linear' for now.")

    def fit(self, df: pl.DataFrame | Any, y_col: str = "y", **kwargs: Any) -> AggDisaggModel:
        """Fit on a Polars (or pandas) DataFrame in long format.

        Expects columns that allow inferring low-frequency groups and target length.
        For the skeleton we accept a simple 'y' column of low-frequency values
        and compute n_high from context or explicit kwarg.
        """
        if isinstance(df, pl.DataFrame):
            y = df[y_col].to_numpy()
        else:
            # pandas fallback
            y = np.asarray(df[y_col])

        self._low_values = y
        self._n_high = kwargs.get("n_high")
        if self._n_high is None:
            # Heuristic for demo: assume monthly from yearly, etc.
            self._n_high = len(y) * 12
        self._fitted = True
        return self

    def predict(self, *, full: bool = True) -> pl.Series:
        """Generate the high-frequency series."""
        if not self._fitted or self._low_values is None or self._n_high is None:
            raise RuntimeError("Call .fit() before .predict()")

        y_high = self._method.disaggregate(
            self._low_values, self._n_high, self.conversion, **self.method_kwargs
        )
        return pl.Series("y_high", y_high)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """sklearn-style transform that adds the disaggregated column."""
        y_high = self.predict()
        return df.with_columns(y_high.alias("y_disagg"))

    def aggregate(self, y_high: pl.Series) -> pl.Series:
        """Convenience to aggregate a high-freq series back."""
        n_low = len(self._low_values) if self._low_values is not None else len(y_high) // 12
        return pl.Series(
            "y_low",
            self._method.aggregate(y_high.to_numpy(), n_low, self.conversion),
        )


def disaggregate(
    y_low: pl.Series | list | np.ndarray,
    *,
    n_high: int | None = None,
    method: Literal["uniform", "linear"] = "uniform",
    conversion: Literal["sum", "mean", "first", "last"] = "sum",
    **kwargs: Any,
) -> pl.Series:
    """One-shot disaggregation function (Polars native feel)."""
    if n_high is None:
        n_high = len(y_low) * 12  # reasonable default for demo
    model = AggDisaggModel(method=method, conversion=conversion, **kwargs)
    # Fake a minimal fit
    model._low_values = np.asarray(y_low)
    model._n_high = n_high
    model._fitted = True
    return model.predict()


def aggregate(
    y_high: pl.Series | list | np.ndarray,
    *,
    n_low: int,
    method: Literal["uniform", "linear"] = "uniform",
    conversion: Literal["sum", "mean", "first", "last"] = "sum",
) -> pl.Series:
    """One-shot aggregation (perfect inverse of disaggregate when using same method)."""
    model = AggDisaggModel(method=method, conversion=conversion)
    return pl.Series("y_low", model._method.aggregate(np.asarray(y_high), n_low, model.conversion))
