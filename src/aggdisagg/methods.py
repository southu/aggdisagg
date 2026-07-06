"""Core disaggregation/aggregation method implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import polars as pl

from .conversion import Conversion


class Method(str, Enum):
    """Available methods for temporal conversion."""

    UNIFORM = "uniform"
    LINEAR = "linear"
    DENTON = "denton"  # placeholder for future
    CHOWLIN = "chowlin"  # placeholder


class BaseMethod(ABC):
    """Abstract base for all agg/disagg methods."""

    name: Method

    @abstractmethod
    def disaggregate(
        self,
        y_low: pl.Series | np.ndarray,
        n_high: int,
        conversion: Conversion,
        **kwargs: Any,
    ) -> np.ndarray:
        """Produce high-frequency series from low-frequency aggregates."""
        ...  # pragma: no cover

    @abstractmethod
    def aggregate(
        self,
        y_high: pl.Series | np.ndarray,
        n_low: int,
        conversion: Conversion,
    ) -> np.ndarray:
        """Aggregate high-frequency back to low frequency (must be inverse)."""
        ...  # pragma: no cover


class Uniform(BaseMethod):
    """Distribute low-frequency value uniformly across high-frequency periods."""

    name = Method.UNIFORM

    def disaggregate(
        self,
        y_low: pl.Series | np.ndarray,
        n_high: int,
        conversion: Conversion,
        **kwargs: Any,
    ) -> np.ndarray:
        y = np.asarray(y_low, dtype=np.float64)
        n_low = len(y)
        freq = n_high // n_low
        if conversion in (Conversion.SUM, Conversion.MEAN):
            return np.repeat(y / freq, freq)
        # first/last don't make much sense for uniform; fall back
        return np.repeat(y, freq)  # pragma: no cover (in Uniform, hit via inheritance mostly)

    def aggregate(
        self, y_high: pl.Series | np.ndarray, n_low: int, conversion: Conversion
    ) -> np.ndarray:
        y = np.asarray(y_high, dtype=np.float64)
        freq = len(y) // n_low
        if conversion == Conversion.SUM:
            return y.reshape(n_low, freq).sum(axis=1)
        elif conversion == Conversion.MEAN:
            return y.reshape(n_low, freq).mean(axis=1)
        elif conversion == Conversion.FIRST:
            return y[::freq]
        else:  # LAST
            return y[freq - 1 :: freq]


class Linear(BaseMethod):
    """Linear interpolation between low-frequency points (with proper scaling)."""

    name = Method.LINEAR

    def disaggregate(
        self,
        y_low: pl.Series | np.ndarray,
        n_high: int,
        conversion: Conversion,
        **kwargs: Any,
    ) -> np.ndarray:
        y = np.asarray(y_low, dtype=np.float64)
        n_low = len(y)
        # freq = n_high // n_low
        # Simple linear interpolation for skeleton
        x = np.arange(n_low)
        x_new = np.linspace(0, n_low - 1, n_high)
        y_high = np.interp(x_new, x, y)
        # Scale to respect aggregation if sum/mean
        if conversion == Conversion.SUM:
            y_high = y_high * (np.sum(y) / np.sum(y_high))
        elif conversion == Conversion.MEAN:
            y_high = y_high * (np.mean(y) / np.mean(y_high))
        return y_high

    def aggregate(
        self, y_high: pl.Series | np.ndarray, n_low: int, conversion: Conversion
    ) -> np.ndarray:
        y = np.asarray(y_high, dtype=np.float64)
        freq = len(y) // n_low
        if conversion == Conversion.SUM:
            return y.reshape(n_low, freq).sum(axis=1)
        elif conversion == Conversion.MEAN:
            return y.reshape(n_low, freq).mean(axis=1)
        elif conversion == Conversion.FIRST:
            return y[::freq]
        else:
            return y[freq - 1 :: freq]


# Placeholder classes for future more sophisticated methods
class Denton(Uniform):  # pragma: no cover
    name = Method.DENTON  # pragma: no cover
    # TODO: implement quadratic minimization with scipy.optimize


class ChowLin(Uniform):  # pragma: no cover
    name = Method.CHOWLIN  # pragma: no cover
    # TODO: implement GLS with indicator series + rho estimation
