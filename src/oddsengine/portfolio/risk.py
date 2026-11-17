"""CVaR of a P&L vector of simulated worlds.

Losses are positive and every number is a fraction of bankroll.
"""

import math

import numpy as np

from oddsengine.types import FloatArray


def cvar(losses: FloatArray, alpha: float) -> float:
    # mean loss in the worst tail of worlds, with the boundary sample part-weighted
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    ordered = np.sort(losses)
    n = ordered.size
    if n == 0:
        raise ValueError("need at least one sample")
    k = math.ceil(n * alpha)
    tail = ordered[k:].sum() / n
    boundary = ordered[max(k - 1, 0)] * (k / n - alpha)
    return float((tail + boundary) / (1.0 - alpha))
