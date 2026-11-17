"""Chooses every stake in the slate at once, as one convex program over the simulated worlds.

It maximises expected log growth (Kelly) under a stake budget, a drawdown
promise, optional tail and variance limits, and each book's cap. Legs that can
lose together get smaller stakes, because the worlds hold joint outcomes.
"""

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import cvxpy as cp
import numpy as np

from oddsengine.config import RiskConfig
from oddsengine.types import FloatArray

log = logging.getLogger(__name__)

# largest constraint breach we accept in an answer
MAX_VIOLATION = 1e-4

# below this the drawdown constraint is left out, as it holds for any stakes
_TINY_LAMBDA = 1e-3


def optimise_slate(
    returns: FloatArray,
    risk: RiskConfig,
    *,
    books: Sequence[str] = (),
    room: Mapping[str, float] | None = None,
) -> FloatArray:
    # returns the stake per leg; room is what each book may still take
    n_sims, n_legs = returns.shape

    # no leg has a positive mean return: stake nothing
    if np.all(returns.mean(axis=0) <= 0.0):
        return np.zeros(n_legs)

    w = cp.Variable(n_legs, nonneg=True)
    wealth = 1.0 + returns @ w
    constraints: list[Any] = [cp.sum(w) <= risk.max_total_stake]

    # drawdown promise (risk-constrained Kelly), using exact power cones
    lam = math.log(risk.drawdown_prob) / math.log(risk.drawdown_floor)
    if lam > _TINY_LAMBDA:
        constraints.append(cp.sum(cp.power(wealth, -lam, approx=False)) / n_sims <= 1.0)

    # bounds the CVaR with a linear program
    if risk.cvar_limit is not None:
        t = cp.Variable()
        excess = cp.sum(cp.pos(-(returns @ w) - t)) / (n_sims * (1.0 - risk.tail))
        constraints.append(t + excess <= risk.cvar_limit)

    # bounds the variance of the slate's returns
    if risk.variance_limit is not None:
        centred = returns - returns.mean(axis=0)
        constraints.append(cp.sum_squares(centred @ w) / n_sims <= risk.variance_limit)

    # each book's new stakes fit in what is left of its cap
    room = room or {}
    for book in sorted(set(books) & set(room)):
        legs = [j for j, b in enumerate(books) if b == book]
        constraints.append(cp.sum(w[legs]) <= max(room[book], 0.0))

    growth = cp.sum(cp.log(wealth)) / n_sims
    return solve(cp.Problem(cp.Maximize(growth), constraints), w)


def solve(problem: cp.Problem, w: cp.Variable) -> FloatArray:
    # tries Clarabel then SCS, and checks each answer against the constraints itself
    # an inaccurate answer is only kept as a fallback
    fallback: FloatArray | None = None
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            problem.solve(solver=solver)
        except cp.error.SolverError:
            continue
        if w.value is None:
            continue

        # the worst breach across every constraint
        violation = max((float(c.violation().max()) for c in problem.constraints), default=0.0)
        if violation > MAX_VIOLATION:
            log.warning("%s broke a constraint by %.2e", solver, violation)
            continue
        stakes = np.clip(np.asarray(w.value, dtype=np.float64), 0.0, None)
        if problem.status == cp.OPTIMAL:
            return stakes
        if problem.status == cp.OPTIMAL_INACCURATE and fallback is None:
            fallback = stakes
    if fallback is not None:
        return fallback
    raise RuntimeError(f"no solver found a feasible solution (status: {problem.status})")
