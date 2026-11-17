"""Devigging: turn one book's decimal prices into fair probabilities.

Each method takes the prices of every outcome of one market and returns
probabilities in the same order, summing to 1. Multiplicative scales them down
evenly, while power shrinks longshots the most.
"""

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import brentq

from oddsengine.types import FloatArray


def _implied(odds: Sequence[float]) -> FloatArray:
    # checks one market's prices and returns the raw implied probabilities
    arr = np.asarray(odds, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"need prices for at least 2 outcomes, got {arr.size}")
    if not np.all(arr > 1.0):
        raise ValueError("every decimal price must be > 1.0")
    return 1.0 / arr


def devig_multiplicative(odds: Sequence[float]) -> FloatArray:
    # scales the implied probabilities down so they sum to 1
    raw = _implied(odds)
    return raw / raw.sum()


def devig_power(odds: Sequence[float]) -> FloatArray:
    # raises the implied probabilities to the power that makes them sum to 1
    raw = _implied(odds)
    k = brentq(lambda k: float(np.sum(raw**k)) - 1.0, 1e-6, 100.0)
    return raw ** float(k)


DEVIG_METHODS: dict[str, Callable[[Sequence[float]], FloatArray]] = {
    "multiplicative": devig_multiplicative,
    "power": devig_power,
}
